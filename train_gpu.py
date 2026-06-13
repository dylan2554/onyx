#!/usr/bin/env python3
"""
Sable NNUE trainer - PyTorch (GPU or CPU).
Same data files, same .nnue export, same arch as train.py - just fast.

Setup (Windows, NVIDIA):  pip install torch --index-url https://download.pytorch.org/whl/cu126
Usage:  python train_gpu.py --data "*.bin" --epochs 30 --out sable.nnue
"""
import os, sys, argparse, glob, time, math, threading, queue

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="*.bin")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=16384)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--lam", type=float, default=0.7)
ap.add_argument("--out", default="sable.nnue")
ap.add_argument("--hidden", type=int, default=512, choices=[256, 512])
ap.add_argument("--epoch-size", type=int, default=25_000_000)
ap.add_argument("--checkpoint", default="checkpoint_gpu.pt")
ap.add_argument("--device", default=None, help="cuda / cpu (auto-detect)")
args = ap.parse_args()

import numpy as np
import torch
import torch.nn as nn

dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
HID = args.hidden

REC = np.dtype([("pc", "u1", 32), ("sq", "u1", 32),
                ("score", "<i2"), ("result", "u1"), ("stm", "u1")])
files = sorted(glob.glob(args.data))
if not files:
    sys.exit(f"no data files match {args.data}")
maps = [np.memmap(f, dtype=REC, mode="r",
                  shape=(os.path.getsize(f) // REC.itemsize,)) for f in files]
N = sum(len(m) for m in maps)
print(f"found {N:,} positions in {len(files)} file(s) "
      f"({sum(os.path.getsize(f) for f in files)/1e9:.2f} GB, streamed)")

CHUNK = 1 << 18
GROUP = 8
chunks = []
for k, m in enumerate(maps):
    for s in range(0, len(m), CHUNK):
        chunks.append((k, s, min(CHUNK, len(m) - s)))

def make_batch(recs):
    PC = recs["pc"].astype(np.int64)
    SQ = recs["sq"].astype(np.int64)
    ok = (((PC < 12) | (PC == 255)).all(axis=1)
          & ((SQ < 64) | (PC == 255)).all(axis=1)
          & (np.abs(recs["score"].astype(np.int32)) <= 3000)
          & (recs["result"] <= 2) & (recs["stm"] <= 1) & (PC[:, 0] != 255))
    recs, PC, SQ = recs[ok], PC[ok], SQ[ok]
    if len(recs) == 0:
        return None
    STM = recs["stm"].astype(np.int64)
    mask = PC != 255
    pt = np.where(mask, PC % 6, 0)
    col = np.where(mask, PC // 6, 0)
    fW = np.where(mask, ((col != 0) * 6 + pt) * 64 + SQ, 768)
    fB = np.where(mask, ((col != 1) * 6 + pt) * 64 + (SQ ^ 56), 768)
    w = (STM == 0)[:, None]
    iS = np.where(w, fW, fB)
    iN = np.where(w, fB, fW)
    res_stm = np.where(STM == 0, recs["result"], 2.0 - recs["result"]) / 2.0
    sc = recs["score"].astype(np.float32)
    t = (args.lam * (1.0 / (1.0 + np.exp(-sc / 400.0)))
         + (1.0 - args.lam) * res_stm).astype(np.float32)
    return iS, iN, t

def producer(erng, q, E, B):
    done = 0
    order = erng.permutation(len(chunks))
    for g in range(0, len(order), GROUP):
        pool = np.concatenate([maps[k][s:s + n]
                               for k, s, n in (chunks[c] for c in order[g:g + GROUP])])
        pool = pool[erng.permutation(len(pool))]
        for i in range(0, len(pool) - B + 1, B):
            mb = make_batch(pool[i:i + B])
            if mb is not None:
                q.put(mb)
            done += B
            if done >= E:
                q.put(None)
                return
    q.put(None)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(769, HID, padding_idx=768)
        nn.init.normal_(self.emb.weight, std=0.05)
        with torch.no_grad():
            self.emb.weight[768].zero_()
        self.b1 = nn.Parameter(torch.zeros(HID))
        self.out = nn.Linear(2 * HID, 1)
        nn.init.normal_(self.out.weight, std=0.05)
        nn.init.zeros_(self.out.bias)
    def forward(self, iS, iN):
        aS = self.emb(iS).sum(1) + self.b1
        aN = self.emb(iN).sum(1) + self.b1
        h = torch.clamp(torch.cat([aS, aN], 1), 0.0, 1.0)
        return self.out(h).squeeze(1)

net = Net().to(dev)
opt = torch.optim.Adam(net.parameters(), lr=args.lr)
start_epoch = 0
if os.path.exists(args.checkpoint):
    ck = torch.load(args.checkpoint, map_location=dev)
    net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
    start_epoch = ck["epoch"]
    print(f"resumed from {args.checkpoint} (epoch {start_epoch})")

def export(path):
    with torch.no_grad():
        W1 = net.emb.weight[:768].cpu().numpy()
        B1 = net.b1.cpu().numpy()
        W2 = net.out.weight[0].cpu().numpy()
        B2 = float(net.out.bias[0])
    with open(path, "wb") as f:
        f.write(b"SABLE2" if HID == 512 else b"SABLE1")
        np.clip(np.round(W1 * 255), -32767, 32767).astype("<i2").tofile(f)
        np.clip(np.round(B1 * 255), -32767, 32767).astype("<i2").tofile(f)
        np.clip(np.round(W2 * 64), -32767, 32767).astype("<i2").tofile(f)
        np.array([round(B2 * 255 * 64)], dtype="<i4").tofile(f)

B = args.batch
E = min(args.epoch_size, N) if args.epoch_size else N
print(f"training: {N:,} positions ({E:,} sampled/epoch), batch {B}, lambda {args.lam}")
for epoch in range(start_epoch, args.epochs):
    erng = np.random.default_rng(1000 + epoch)
    q = queue.Queue(maxsize=8)
    th = threading.Thread(target=producer, args=(erng, q, E, B), daemon=True)
    th.start()
    tot, nb, t0 = 0.0, 0, time.time()
    while True:
        item = q.get()
        if item is None:
            break
        iS, iN, t = item
        iS = torch.from_numpy(iS).to(dev, non_blocking=True)
        iN = torch.from_numpy(iN).to(dev, non_blocking=True)
        t = torch.from_numpy(t).to(dev, non_blocking=True)
        y = net(iS, iN)
        loss = torch.mean((torch.sigmoid(y) - t) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            net.emb.weight.clamp_(-1.27, 1.27)
            net.emb.weight[768].zero_()
            net.b1.clamp_(-1.27, 1.27)
            net.out.weight.clamp_(-1.98, 1.98)
        tot += float(loss); nb += 1
        if nb % 200 == 0:
            print(f"  epoch {epoch+1} batch {nb}: loss {tot/nb:.5f} "
                  f"({nb*B/(time.time()-t0):,.0f} pos/s)", flush=True)
    dt = time.time() - t0
    print(f"epoch {epoch+1}/{args.epochs}: loss {tot/max(nb,1):.5f} "
          f"({dt:.0f}s, {nb*B/dt:,.0f} pos/s)", flush=True)
    export(args.out)
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                "epoch": epoch + 1}, args.checkpoint)
    print(f"  exported {args.out} + checkpoint", flush=True)
print("done.")
