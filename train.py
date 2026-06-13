#!/usr/bin/env python3
"""
Sable NNUE trainer (pure NumPy - no GPU/torch needed).
Arch: 768 -> (256 x 2 perspectives, shared weights) -> 1

Usage:
  python train.py --data "data*.bin" --epochs 30 --out sable.nnue --threads 4

PC-friendly: set --threads to (your cores - 2) or less. Exports a usable
net after EVERY epoch, so you can stop any time and keep the latest.
Resumes automatically from checkpoint.npz if present.
"""
import os, sys, argparse, glob, time, math

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data*.bin")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=16384)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--lam", type=float, default=0.7, help="weight on search eval vs game result")
ap.add_argument("--out", default="sable.nnue")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--hidden", type=int, default=512, choices=[256, 512])
ap.add_argument("--checkpoint", default="checkpoint.npz")
ap.add_argument("--epoch-size", type=int, default=25_000_000,
                help="positions sampled per epoch (0 = all)")
args = ap.parse_args()

# limit BLAS threads BEFORE importing numpy
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = str(args.threads)
import numpy as np
HID = args.hidden

REC = np.dtype([("pc", "u1", 32), ("sq", "u1", 32),
                ("score", "<i2"), ("result", "u1"), ("stm", "u1")])

# ---------------- data (memory-mapped: RAM stays tiny) ----------------
files = sorted(glob.glob(args.data))
if not files:
    sys.exit(f"no data files match {args.data}")
maps = [np.memmap(f, dtype=REC, mode="r",
                  shape=(os.path.getsize(f) // REC.itemsize,)) for f in files]
counts = [len(m) for m in maps]
offsets = np.cumsum([0] + counts)
N = int(offsets[-1])
print(f"found {N:,} positions in {len(files)} file(s) "
      f"({sum(os.path.getsize(f) for f in files)/1e9:.2f} GB on disk, streamed)")

# chunked shuffling: sequential disk reads, shuffle in RAM pools
CHUNK = 1 << 18          # 262,144 records (~17 MB) per sequential read
GROUP = 8                # chunks pooled+shuffled together (~2M records)
chunks = []
for k, m in enumerate(maps):
    for s in range(0, len(m), CHUNK):
        chunks.append((k, s, min(CHUNK, len(m) - s)))

def make_batch(recs):
    PC = recs["pc"].astype(np.int32)
    SQ = recs["sq"].astype(np.int32)
    ok = (((PC < 12) | (PC == 255)).all(axis=1)
          & ((SQ < 64) | (PC == 255)).all(axis=1)
          & (np.abs(recs["score"].astype(np.int32)) <= 3000)
          & (recs["result"] <= 2) & (recs["stm"] <= 1) & (PC[:, 0] != 255))
    recs, PC, SQ = recs[ok], PC[ok], SQ[ok]
    if len(recs) == 0:
        return None, None, None
    STM = recs["stm"].astype(np.int32)
    mask = PC != 255
    pt = np.where(mask, PC % 6, 0)
    col = np.where(mask, PC // 6, 0)
    fW = np.where(mask, ((col != 0) * 6 + pt) * 64 + SQ, 768)
    fB = np.where(mask, ((col != 1) * 6 + pt) * 64 + (SQ ^ 56), 768)
    w = (STM == 0)[:, None]
    iS, iN = np.where(w, fW, fB), np.where(w, fB, fW)
    res_stm = np.where(STM == 0, recs["result"], 2.0 - recs["result"]) / 2.0
    sc = recs["score"].astype(np.float32)
    t = (args.lam * (1.0 / (1.0 + np.exp(-sc / 400.0)))
         + (1.0 - args.lam) * res_stm).astype(np.float32)
    return iS, iN, t

# ---------------- model ----------------
rng = np.random.default_rng(42)
W1 = (rng.standard_normal((768, HID)) * 0.05).astype(np.float32)
B1 = np.zeros(HID, np.float32)
W2 = (rng.standard_normal(2 * HID) * 0.05).astype(np.float32)
B2 = np.zeros(1, np.float32)
params = {"W1": W1, "B1": B1, "W2": W2, "B2": B2}
adam = {k: [np.zeros_like(v), np.zeros_like(v)] for k, v in params.items()}
step_t = 0
start_epoch = 0

if os.path.exists(args.checkpoint):
    ck = np.load(args.checkpoint)
    for k in params:
        params[k][...] = ck[k]
    for k in adam:
        adam[k][0][...] = ck[k + "_m"]
        adam[k][1][...] = ck[k + "_v"]
    step_t = int(ck["t"]); start_epoch = int(ck["epoch"])
    print(f"resumed from {args.checkpoint} (epoch {start_epoch})")
W1, B1, W2, B2 = params["W1"], params["B1"], params["W2"], params["B2"]

def adam_step(name, g, lr):
    global step_t
    m, v = adam[name]
    m *= 0.9;  m += 0.1 * g
    v *= 0.999; v += 0.001 * g * g
    mhat = m / (1 - 0.9 ** step_t)
    vhat = v / (1 - 0.999 ** step_t)
    params[name] -= lr * mhat / (np.sqrt(vhat) + 1e-8)

def export(path):
    with open(path, "wb") as f:
        f.write(b"SABLE2" if HID == 512 else b"SABLE1")
        np.clip(np.round(W1 * 255), -32767, 32767).astype("<i2").tofile(f)
        np.clip(np.round(B1 * 255), -32767, 32767).astype("<i2").tofile(f)
        np.clip(np.round(W2 * 64), -32767, 32767).astype("<i2").tofile(f)
        np.array([round(float(B2[0]) * 255 * 64)], dtype="<i4").tofile(f)

# ---------------- training ----------------
B = args.batch
E = min(args.epoch_size, N) if args.epoch_size else N
print(f"training: {N:,} positions ({E:,} sampled/epoch), batch {B}, "
      f"{args.threads} threads, lambda {args.lam}")

def batches(erng):
    done = 0
    order = erng.permutation(len(chunks))
    for g in range(0, len(order), GROUP):
        pool = np.concatenate([maps[k][s:s + n]
                               for k, s, n in (chunks[c] for c in order[g:g + GROUP])])
        pool = pool[erng.permutation(len(pool))]
        for i in range(0, len(pool) - B + 1, B):
            yield pool[i:i + B]
            done += B
            if done >= E:
                return

for epoch in range(start_epoch, args.epochs):
    erng = np.random.default_rng(1000 + epoch)
    tot_loss, nb = 0.0, 0
    t0 = time.time()
    for recs in batches(erng):
        iS, iN, t = make_batch(recs)
        if iS is None:
            continue
        b = len(t)
        rows = np.arange(b)[:, None]
        # dense multi-hot inputs (BLAS-friendly)
        XS = np.zeros((b, 769), np.float32); XS[rows, iS] = 1.0
        XN = np.zeros((b, 769), np.float32); XN[rows, iN] = 1.0
        XS = XS[:, :768]; XN = XN[:, :768]
        aS = XS @ W1 + B1
        aN = XN @ W1 + B1
        h = np.clip(np.concatenate([aS, aN], 1), 0.0, 1.0)
        y = h @ W2 + B2[0]
        p = 1.0 / (1.0 + np.exp(-y))
        diff = p - t
        loss = float(np.mean(diff * diff))
        tot_loss += loss; nb += 1
        # backward
        dy = (2.0 / b) * diff * p * (1.0 - p)
        gW2 = h.T @ dy
        gB2 = np.array([dy.sum()], np.float32)
        dh = np.outer(dy, W2)
        pre = np.concatenate([aS, aN], 1)
        dh *= (pre > 0) & (pre < 1)
        dS, dN = dh[:, :HID], dh[:, HID:]
        gB1 = dS.sum(0) + dN.sum(0)
        gW1 = XS.T @ dS + XN.T @ dN
        step_t += 1
        adam_step("W1", gW1, args.lr)
        adam_step("B1", gB1, args.lr)
        adam_step("W2", gW2, args.lr)
        adam_step("B2", gB2, args.lr)
        np.clip(W1, -1.27, 1.27, out=W1)
        np.clip(B1, -1.27, 1.27, out=B1)
        np.clip(W2, -1.98, 1.98, out=W2)
        if nb % 200 == 0:
            sp = nb * B / (time.time() - t0)
            print(f"  epoch {epoch+1} batch {nb}: loss {tot_loss/nb:.5f} "
                  f"({sp:,.0f} pos/s)", flush=True)
    dt = time.time() - t0
    print(f"epoch {epoch+1}/{args.epochs}: loss {tot_loss/max(nb,1):.5f} "
          f"({dt:.0f}s, {nb*B/dt:,.0f} pos/s)", flush=True)
    export(args.out)
    save = {k: v for k, v in params.items()}
    for k in adam:
        save[k + "_m"], save[k + "_v"] = adam[k]
    save["t"] = step_t; save["epoch"] = epoch + 1
    np.savez(args.checkpoint, **save)
    print(f"  exported {args.out} + checkpoint", flush=True)
print("done.")
