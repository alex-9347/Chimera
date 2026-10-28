"""
NEXUS-1 Distillation Worker.
Läuft auf einem GitHub-Actions-Runner. Sampelt einen disjunkten Bereich
der Lichess-PGN, bewertet mit Stockfish depth=10 (MultiPV=5), speichert
FEN + Top-5-Indizes + Top-5-Gewichte + SF-Eval als .npz.
"""
import os, sys, io, time, random, zipfile, urllib.request, gzip, json
import numpy as np
import chess, chess.engine, chess.pgn
from pathlib import Path

# ── Config (aus Umgebungsvariablen) ────────────────────────────────────────
WORKER_ID = int(os.environ.get("WORKER_ID", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "40"))
POSITIONS_PER_WORKER = int(os.environ.get("POSITIONS_PER_WORKER", "400000"))
SF_DEPTH = int(os.environ.get("SF_DEPTH", "10"))
SF_MULTIPV = int(os.environ.get("SF_MULTIPV", "5"))
SF_THREADS = int(os.environ.get("SF_THREADS", "2"))
SF_HASH_MB = int(os.environ.get("SF_HASH_MB", "256"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "distill_out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT_DIR / f"worker_{WORKER_ID:02d}.log"
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n"); f.flush()

log(f"Worker {WORKER_ID}/{NUM_WORKERS} | target={POSITIONS_PER_WORKER} pos | "
    f"depth={SF_DEPTH} multipv={SF_MULTIPV} threads={SF_THREADS}")

# ── Stockfish ──────────────────────────────────────────────────────────────
def ensure_stockfish():
    exe = Path("/tmp/stockfish")
    if exe.exists(): return exe
    url = ("https://github.com/official-stockfish/Stockfish/releases/"
           "download/sf_16.1/stockfish-ubuntu-x86-64-avx2.tar")
    tp = Path("/tmp/stockfish.tar")
    log(f"Downloading Stockfish from {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(tp, "wb") as f:
        while c := r.read(1 << 20): f.write(c)
    import tarfile
    with tarfile.open(tp, "r") as tf:
        for m in tf.getmembers():
            if m.isfile():
                ef = tf.extractfile(m)
                if ef and ef.read(4) == b"\x7fELF":
                    ef.seek(0)
                    with open(exe, "wb") as d:
                        import shutil; shutil.copyfileobj(ef, d)
                    break
    tp.unlink(missing_ok=True)
    os.chmod(exe, 0o755)
    log(f"Stockfish ready: {exe}")
    return exe

SF_PATH = ensure_stockfish()

# ── Lichess-PGN ────────────────────────────────────────────────────────────
LICHESS_URLS = [
    "https://database.nikonoel.fr/lichess_elite_2025-11.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-10.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-09.zip",
]

def download_lichess():
    local = Path("/tmp/lichess.zip")
    if local.exists() and local.stat().st_size > 10_000_000:
        return local
    for url in LICHESS_URLS:
        try:
            log(f"Trying {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(local, "wb") as f:
                while c := r.read(8 << 20): f.write(c)
            if local.stat().st_size > 10_000_000:
                return local
        except Exception as e:
            log(f"  failed: {e}"); local.unlink(missing_ok=True)
    raise RuntimeError("Could not download Lichess PGN")

PGN_ZIP = download_lichess()
log(f"PGN: {PGN_ZIP} ({PGN_ZIP.stat().st_size/1e6:.0f} MB)")

# ── Move-Encoding (20480 = 64×64×5) ────────────────────────────────────────
def encode_move(move, turn):
    fs = move.from_square ^ 56 if turn == chess.BLACK else move.from_square
    ts = move.to_square ^ 56 if turn == chess.BLACK else move.to_square
    promo = 0 if move.promotion is None else move.promotion - 1
    return fs * 320 + ts * 5 + promo

# ── Main loop ──────────────────────────────────────────────────────────────
engine = chess.engine.SimpleEngine.popen_uci(str(SF_PATH))
engine.configure({"Threads": SF_THREADS, "Hash": SF_HASH_MB})

fens, top_idx, top_w, vals = [], [], [], []
t0 = time.time()
games_seen = 0
positions_done = 0
skip_games = (WORKER_ID * 5000) // NUM_WORKERS  # disjunkter Startbereich
stride = max(1, NUM_WORKERS)  # jeder N-te Worker nimmt jede N-te Partie

def sf_eval_board(board):
    try:
        infos = engine.analyse(board, chess.engine.Limit(depth=SF_DEPTH),
                               multipv=SF_MULTIPV, info=chess.engine.INFO_ALL)
    except Exception:
        return None
    moves = []
    for info in infos:
        if "pv" not in info or not info["pv"]: continue
        mv = info["pv"][0]; sc = info.get("score")
        if sc is None: continue
        pov = sc.pov(board.turn)
        cp = (10000 if pov.is_mate() and (pov.mate() or 0) > 0
              else -10000 if pov.is_mate()
              else float(pov.score(mate_score=10000) or 0))
        moves.append((mv, cp))
    if not moves: return None
    cps = np.clip(np.array([c for _, c in moves], dtype=np.float32), -1000, 1000)
    w = np.exp(cps / 100.0); w = w / max(w.sum(), 1e-9)
    idx = [encode_move(m, board.turn) for m, _ in moves]
    # Pad to SF_MULTIPV
    while len(idx) < SF_MULTIPV: idx.append(0); w = np.append(w, 0.0)
    return idx[:SF_MULTIPV], w[:SF_MULTIPV].tolist(), cps[0]

with zipfile.ZipFile(PGN_ZIP, "r") as zf:
    pgn_name = next(n for n in zf.namelist() if n.endswith(".pgn"))
    with zf.open(pgn_name) as raw:
        stream = io.TextIOWrapper(raw, encoding="utf-8", errors="ignore")
        game_i = 0
        while positions_done < POSITIONS_PER_WORKER:
            game = chess.pgn.read_game(stream)
            if game is None: break
            game_i += 1
            # Disjunkte Auswahl: jeder Worker nimmt jede NUM_WORKERS-te Partie
            if (game_i - skip_games) % stride != 0: continue
            moves = list(game.mainline_moves())
            if len(moves) < 10: continue
            games_seen += 1
            pop = list(range(5, len(moves) - 2))
            if not pop: continue
            sample = set(random.sample(pop, min(10, len(pop))))
            b = game.board()
            for i, m in enumerate(moves):
                if i in sample and positions_done < POSITIONS_PER_WORKER:
                    res = sf_eval_board(b)
                    if res is not None:
                        fen = b.fen()
                        fi, fw, fv = res
                        fens.append(fen); top_idx.append(fi); top_w.append(fw); vals.append(fv)
                        positions_done += 1
                b.push(m)
            if game_i % 200 == 0:
                el = time.time() - t0
                rate = positions_done / max(el, 1)
                log(f"  games={games_seen} pos={positions_done}/{POSITIONS_PER_WORKER} "
                    f"({rate:.0f}/s, {el/60:.1f}min)")

engine.quit()
log(f"DONE: {positions_done} positions in {(time.time()-t0)/60:.1f} min")

# ── Speichern ──────────────────────────────────────────────────────────────
out = OUT_DIR / f"distill_{WORKER_ID:02d}.npz.gz"
np.savez_compressed(
    out,
    fens=np.array(fens, dtype=object),
    top_idx=np.array(top_idx, dtype=np.int32),
    top_w=np.array(top_w, dtype=np.float32),
    val_cp=np.array(vals, dtype=np.float32),
)
with open(out, "rb") as f_in, gzip.open(str(out) + ".gz2", "wb") as f_out:
    f_out.write(f_in.read())
# Doppelte Kompression vermeiden: npz ist schon komprimiert, gz2 löschen
Path(str(out) + ".gz2").unlink(missing_ok=True)
log(f"Saved: {out} ({out.stat().st_size/1e6:.1f} MB)")

# Manifest
with open(OUT_DIR / f"manifest_{WORKER_ID:02d}.json", "w") as f:
    json.dump({"worker": WORKER_ID, "positions": positions_done,
               "games": games_seen, "depth": SF_DEPTH, "multipv": SF_MULTIPV}, f)
