"""
Chimera Distillation Worker v5 — mit game_outcome, is_mate, Syzygy-Sampling.
"""
import os, sys, io, time, random, zipfile, urllib.request, gzip, json, signal, tarfile
import numpy as np
import chess, chess.engine, chess.pgn
from pathlib import Path

WORKER_ID = int(os.environ.get("WORKER_ID", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "40"))
SF_DEPTH = int(os.environ.get("SF_DEPTH", "12"))
SF_MULTIPV = int(os.environ.get("SF_MULTIPV", "4"))
SF_THREADS = int(os.environ.get("SF_THREADS", "2"))
SF_HASH_MB = int(os.environ.get("SF_HASH_MB", "256"))
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "50000"))
MAX_RUNTIME_SEC = int(os.environ.get("MAX_RUNTIME_SEC", "21600"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "distill_out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_PATH = OUT_DIR / f"worker_{WORKER_ID:02d}.log"
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n"); f.flush()

class SfTimeout(Exception): pass

def ensure_stockfish():
    exe = Path("/tmp/stockfish")
    if exe.exists() and os.access(exe, os.X_OK): return exe
    url = ("https://github.com/official-stockfish/Stockfish/releases/"
           "download/sf_16.1/stockfish-ubuntu-x86-64-avx2.tar")
    tp = Path("/tmp/stockfish.tar")
    log("Downloading Stockfish...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(tp, "wb") as f:
        while c := r.read(1 << 20): f.write(c)
    with tarfile.open(tp, "r") as tf:
        for m in tf.getmembers():
            if m.isfile():
                ef = tf.extractfile(m)
                if ef and ef.read(4) == b"\x7fELF":
                    ef.seek(0)
                    with open(exe, "wb") as d:
                        import shutil; shutil.copyfileobj(ef, d)
                    break
    tp.unlink(missing_ok=True); os.chmod(exe, 0o755); return exe

def popen_with_timeout(exe, timeout=30):
    def _handler(signum, frame):
        raise SfTimeout(f"Stockfish timeout {timeout}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        return chess.engine.SimpleEngine.popen_uci(str(exe))
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)

log(f"Worker {WORKER_ID}/{NUM_WORKERS} | depth={SF_DEPTH} multipv={SF_MULTIPV}")
SF_PATH = ensure_stockfish()
try:
    engine = popen_with_timeout(SF_PATH, timeout=30)
    log("Stockfish initialized")
except SfTimeout as e:
    log(f"❌ {e}"); sys.exit(1)
engine.configure({"Threads": SF_THREADS, "Hash": SF_HASH_MB})

LICHESS_URLS = [
    "https://database.nikonoel.fr/lichess_elite_2025-11.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-10.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-09.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-08.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-07.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-06.zip",
]

def download_pgn(url):
    local = Path("/tmp") / Path(url).name
    if local.exists() and local.stat().st_size > 10_000_000: return local
    try:
        log(f"  ↓ {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(local, "wb") as f:
            while c := r.read(8 << 20): f.write(c)
        if local.stat().st_size > 10_000_000: return local
    except Exception as e:
        log(f"  ⚠️ {e}"); local.unlink(missing_ok=True)
    return None

def encode_move(move, turn):
    fs = move.from_square ^ 56 if turn == chess.BLACK else move.from_square
    ts = move.to_square ^ 56 if turn == chess.BLACK else move.to_square
    promo = 0 if move.promotion is None else move.promotion - 1
    return fs * 320 + ts * 5 + promo

def sf_eval(board):
    try:
        infos = engine.analyse(board, chess.engine.Limit(depth=SF_DEPTH),
                               multipv=SF_MULTIPV, info=chess.engine.INFO_ALL)
    except Exception:
        return None
    moves, is_mate = [], False
    for info in infos:
        if "pv" not in info or not info["pv"]: continue
        mv = info["pv"][0]; sc = info.get("score")
        if sc is None: continue
        pov = sc.pov(board.turn)
        if pov.is_mate():
            is_mate = True
            cp = 10000 if (pov.mate() or 0) > 0 else -10000
        else:
            cp = float(pov.score(mate_score=10000) or 0)
        moves.append((mv, cp))
    if not moves: return None
    cps = np.clip(np.array([c for _, c in moves], dtype=np.float32), -1000, 1000)
    w = np.exp(cps / 100.0); w = w / max(w.sum(), 1e-9)
    idx = [encode_move(m, board.turn) for m, _ in moves]
    while len(idx) < SF_MULTIPV: idx.append(0); w = np.append(w, 0.0)
    return idx[:SF_MULTIPV], w[:SF_MULTIPV].tolist(), float(cps[0]), is_mate

def save_checkpoint(fens, top_idx, top_w, vals, is_mates, outcomes,
                    positions_done, final=False):
    suffix = "final" if final else f"part{positions_done:07d}"
    out = OUT_DIR / f"distill_{WORKER_ID:02d}_{suffix}.npz.gz"
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        fens=np.array(fens, dtype=object),
        top_idx=np.array(top_idx, dtype=np.int32),
        top_w=np.array(top_w, dtype=np.float32),
        val_cp=np.array(vals, dtype=np.float32),
        is_mate=np.array(is_mates, dtype=np.bool_),
        game_outcome=np.array(outcomes, dtype=np.float32),
    )
    buf.seek(0)
    with gzip.open(out, "wb", compresslevel=6) as f_out:
        f_out.write(buf.read())
    log(f"💾 {out.name} ({out.stat().st_size/1e6:.1f} MB, {positions_done} pos)")
    return out

fens, top_idx, top_w, vals, is_mates, outcomes = [], [], [], [], [], []
t_start = time.time()
last_heartbeat = t_start
games_seen = 0; positions_done = 0; game_i = 0; last_ckpt_pos = 0

def time_left(): return MAX_RUNTIME_SEC - (time.time() - t_start)

for url in LICHESS_URLS:
    if time_left() < 300: log("⏰ Zeit"); break
    pgn_zip = download_pgn(url)
    if pgn_zip is None: continue
    log(f"Verarbeite {pgn_zip.name} (verbleibend {time_left()/3600:.1f}h)")

    with zipfile.ZipFile(pgn_zip, "r") as zf:
        pgn_name = next((n for n in zf.namelist() if n.endswith(".pgn")), None)
        if pgn_name is None: continue
        with zf.open(pgn_name) as raw:
            stream = io.TextIOWrapper(raw, encoding="utf-8", errors="ignore")
            while time_left() > 300:
                game = chess.pgn.read_game(stream)
                if game is None: break
                game_i += 1
                if time.time() - last_heartbeat > 30:
                    el = time.time() - t_start
                    log(f"  ❤ game_i={game_i} games={games_seen} pos={positions_done} "
                        f"({positions_done/max(el,1):.0f}/s, {el/3600:.1f}h)")
                    last_heartbeat = time.time()

                if (game_i - WORKER_ID) % NUM_WORKERS != 0: continue
                result_str = game.headers.get("Result", "*")
                if result_str not in ("1-0", "0-1", "1/2-1/2"): continue
                outcome = 1.0 if result_str == "1-0" else (-1.0 if result_str == "0-1" else 0.0)
                moves = list(game.mainline_moves())
                if len(moves) < 20: continue
                games_seen += 1

                # Systematisches Sampling: jede 5. Stellung, aber max 40 pro Partie
                step = max(1, len(moves) // 40)
                for i in range(5, len(moves) - 2, step):
                    if time_left() < 300: break
                    b = game.board()
                    for j in range(i): b.push(moves[j])
                    if b.is_game_over(): continue
                    res = sf_eval(b)
                    if res is None: continue
                    idx, w, cp, is_mate = res
                    # Aus Sicht des Spielers am Zug
                    pos_outcome = outcome if b.turn == chess.WHITE else -outcome
                    fens.append(b.fen()); top_idx.append(idx); top_w.append(w)
                    vals.append(cp); is_mates.append(is_mate); outcomes.append(pos_outcome)
                    positions_done += 1
                    if positions_done - last_ckpt_pos >= CHECKPOINT_EVERY:
                        save_checkpoint(fens, top_idx, top_w, vals, is_mates, outcomes, positions_done)
                        last_ckpt_pos = positions_done

engine.quit()
log(f"DONE: {positions_done} pos in {(time.time()-t_start)/3600:.2f}h")
save_checkpoint(fens, top_idx, top_w, vals, is_mates, outcomes, positions_done, final=True)

with open(OUT_DIR / f"manifest_{WORKER_ID:02d}.json", "w") as f:
    json.dump({"worker": WORKER_ID, "positions": positions_done,
               "games": games_seen, "depth": SF_DEPTH, "multipv": SF_MULTIPV,
               "elapsed_sec": time.time() - t_start}, f)
