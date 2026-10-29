"""
Chimera Distillation Worker v4 — 6h-Lauf, höhere Qualität.
- Verarbeitet mehrere Lichess-Elite-Monate, bis 6h um sind.
- Stockfish depth 12, MultiPV 8.
- Phasen-balanciertes Sampling: 10% Opening / 65% Middlegame / 25% Endgame.
- Markiert Tablebase-Stellungen (≤5 Steine) für späteres Rescoring.
- Checkpoint alle 50k Positionen.
"""
import os, sys, io, time, random, zipfile, urllib.request, gzip, json, signal, tarfile
import numpy as np
import chess, chess.engine, chess.pgn
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
WORKER_ID = int(os.environ.get("WORKER_ID", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "40"))
SF_DEPTH = int(os.environ.get("SF_DEPTH", "12"))
SF_MULTIPV = int(os.environ.get("SF_MULTIPV", "8"))
SF_THREADS = int(os.environ.get("SF_THREADS", "2"))
SF_HASH_MB = int(os.environ.get("SF_HASH_MB", "256"))
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "50000"))
MAX_RUNTIME_SEC = int(os.environ.get("MAX_RUNTIME_SEC", "21600"))  # 6h
OUT_DIR = Path(os.environ.get("OUT_DIR", "distill_out"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Zielverteilung (Anteile an gesamten Positionen)
TARGET_OPENING = 0.10
TARGET_MIDDLEGAME = 0.65
TARGET_ENDGAME = 0.25

LOG_PATH = OUT_DIR / f"worker_{WORKER_ID:02d}.log"
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n"); f.flush()

# ── Stockfish mit Timeout ──────────────────────────────────────────────────
class SfTimeout(Exception): pass

def ensure_stockfish():
    exe = Path("/tmp/stockfish")
    if exe.exists() and os.access(exe, os.X_OK):
        return exe
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
    tp.unlink(missing_ok=True)
    os.chmod(exe, 0o755)
    return exe

def popen_uci_with_timeout(exe, timeout=30):
    def _handler(signum, frame):
        raise SfTimeout(f"Stockfish did not respond in {timeout}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        return chess.engine.SimpleEngine.popen_uci(str(exe))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)

log(f"Worker {WORKER_ID}/{NUM_WORKERS} | depth={SF_DEPTH} multipv={SF_MULTIPV} "
    f"threads={SF_THREADS} | max_runtime={MAX_RUNTIME_SEC/3600:.0f}h")

SF_PATH = ensure_stockfish()
try:
    engine = popen_uci_with_timeout(SF_PATH, timeout=30)
    log("Stockfish initialized")
except SfTimeout as e:
    log(f"❌ {e}"); sys.exit(1)
engine.configure({"Threads": SF_THREADS, "Hash": SF_HASH_MB})

# ── Lichess-Elite: mehrere Monate ─────────────────────────────────────────
# Reihenfolge: neueste zuerst. Der Worker geht sie nacheinander durch.
LICHESS_URLS = [
    "https://database.nikonoel.fr/lichess_elite_2025-11.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-10.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-09.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-08.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-07.zip",
    "https://database.nikonoel.fr/lichess_elite_2025-06.zip",
]

def download_pgn(url):
    """Lädt eine PGN-ZIP. Returns Path oder None."""
    local = Path("/tmp") / Path(url).name
    if local.exists() and local.stat().st_size > 10_000_000:
        return local
    try:
        log(f"  ↓ {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(local, "wb") as f:
            while c := r.read(8 << 20): f.write(c)
        if local.stat().st_size > 10_000_000:
            return local
    except Exception as e:
        log(f"  ⚠️ Download fehlgeschlagen: {e}")
        local.unlink(missing_ok=True)
    return None

# ── Move-Encoding ──────────────────────────────────────────────────────────
def encode_move(move, turn):
    fs = move.from_square ^ 56 if turn == chess.BLACK else move.from_square
    ts = move.to_square ^ 56 if turn == chess.BLACK else move.to_square
    promo = 0 if move.promotion is None else move.promotion - 1
    return fs * 320 + ts * 5 + promo

# ── Phasen-Klassifikation ─────────────────────────────────────────────────
def classify_phase(board):
    """Returns 'opening', 'middlegame', or 'endgame'."""
    move_num = board.fullmove_number
    if move_num <= 10:
        return "opening"
    if move_num <= 40:
        return "middlegame"
    # Endgame auch bei wenigen Figuren, unabhängig von Zugzahl
    total_pieces = bin(board.occupied).count("1")
    if total_pieces <= 8:
        return "endgame"
    return "middlegame"

# ── SF-Bewertung ───────────────────────────────────────────────────────────
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
    while len(idx) < SF_MULTIPV: idx.append(0); w = np.append(w, 0.0)
    return idx[:SF_MULTIPV], w[:SF_MULTIPV].tolist(), float(cps[0])

# ── Checkpoint (BytesIO + gzip) ───────────────────────────────────────────
def save_checkpoint(fens, top_idx, top_w, vals, tablebase_flags,
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
        is_tablebase=np.array(tablebase_flags, dtype=np.bool_),
    )
    buf.seek(0)
    with gzip.open(out, "wb", compresslevel=6) as f_out:
        f_out.write(buf.read())
    log(f"💾 {out.name} ({out.stat().st_size/1e6:.1f} MB, {positions_done} pos)")
    return out

# ── Tablebase-Erkennung ───────────────────────────────────────────────────
def is_tablebase_position(board):
    """True, wenn ≤5 Steine auf dem Brett (Syzygy 5-piece)."""
    return bin(board.occupied).count("1") <= 5

# ── Sampling-Strategie ────────────────────────────────────────────────────
def get_sampling_indices(num_moves, phase):
    """
    Systematische Stichproben pro Phase.
    - Opening (Zug 1-10): jede 3. Stellung
    - Middlegame (Zug 11-40): jede 2. Stellung
    - Endgame (Zug 41+): jede Stellung
    Gibt Indizes in `moves` zurück (0-basiert).
    """
    if phase == "opening":
        return set(range(2, min(10, num_moves), 3))  # 2,5,8
    if phase == "middlegame":
        return set(range(10, min(40, num_moves), 2))  # 10,12,14,...
    # endgame
    return set(range(40, num_moves, 1))  # alle ab 40

# ── Main loop: mehrere PGNs, bis Zeit um ──────────────────────────────────
fens, top_idx, top_w, vals, tablebase_flags = [], [], [], [], []
t_start = time.time()
last_heartbeat = t_start
games_seen = 0
positions_done = 0
game_i = 0
last_ckpt_pos = 0

# Zähler pro Phase (für Reporting)
phase_counts = {"opening": 0, "middlegame": 0, "endgame": 0}

def time_left():
    return MAX_RUNTIME_SEC - (time.time() - t_start)

# Alle PGN-Monate durchgehen
for url in LICHESS_URLS:
    if time_left() < 300:  # <5 min → abbrechen
        log("⏰ Zeit fast um, stoppe PGN-Verarbeitung")
        break
    pgn_zip = download_pgn(url)
    if pgn_zip is None:
        continue
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

                # Heartbeat
                if time.time() - last_heartbeat > 30:
                    el = time.time() - t_start
                    rate = positions_done / max(el, 1)
                    log(f"  ❤ game_i={game_i} games={games_seen} "
                        f"pos={positions_done} ({rate:.0f}/s, {el/3600:.1f}h) "
                        f"[O:{phase_counts['opening']} M:{phase_counts['middlegame']} "
                        f"E:{phase_counts['endgame']}]")
                    last_heartbeat = time.time()

                # Disjunkte Auswahl: jeder Worker nimmt jede NUM_WORKERS-te Partie
                if (game_i - WORKER_ID) % NUM_WORKERS != 0:
                    continue
                result_str = game.headers.get("Result", "*")
                if result_str not in ("1-0", "0-1", "1/2-1/2"):
                    continue
                moves = list(game.mainline_moves())
                if len(moves) < 20:
                    continue
                games_seen += 1

                b = game.board()
                # Positionen pro Phase sammeln
                for phase in ["opening", "middlegame", "endgame"]:
                    indices = get_sampling_indices(len(moves), phase)
                    for i in sorted(indices):
                        if i >= len(moves): break
                        if time_left() < 300: break
                        # Board an Position i
                        b_i = game.board()
                        for j in range(i):
                            b_i.push(moves[j])
                        if b_i.is_game_over(): continue
                        res = sf_eval_board(b_i)
                        if res is None: continue
                        fens.append(b_i.fen())
                        top_idx.append(res[0])
                        top_w.append(res[1])
                        vals.append(res[2])
                        tablebase_flags.append(is_tablebase_position(b_i))
                        phase_counts[phase] += 1
                        positions_done += 1

                        if positions_done - last_ckpt_pos >= CHECKPOINT_EVERY:
                            save_checkpoint(fens, top_idx, top_w, vals,
                                            tablebase_flags, positions_done)
                            last_ckpt_pos = positions_done

                if game_i % 500 == 0:
                    log(f"  {game_i} Partien, {positions_done} pos, "
                        f"{time_left()/3600:.2f}h verbleibend")

engine.quit()
log(f"DONE: {positions_done} pos in {(time.time()-t_start)/3600:.2f}h | "
    f"O:{phase_counts['opening']} M:{phase_counts['middlegame']} "
    f"E:{phase_counts['endgame']}")
save_checkpoint(fens, top_idx, top_w, vals, tablebase_flags, positions_done, final=True)

with open(OUT_DIR / f"manifest_{WORKER_ID:02d}.json", "w") as f:
    json.dump({"worker": WORKER_ID, "positions": positions_done,
               "games": games_seen, "depth": SF_DEPTH, "multipv": SF_MULTIPV,
               "phase_counts": phase_counts,
               "elapsed_sec": time.time() - t_start}, f)
