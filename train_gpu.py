#!/usr/bin/env python3
"""Onyx single-head NNUE trainer for an external GPU runtime.

The training and validation corpora must be separate DGRecord files.  This
script exports SABLE1/2/3 networks; production Onyx runs should name SABLE3
explicitly so an accidental hidden-size mismatch fails before training.
"""
import os, sys, argparse, glob, time, threading, queue

ARCH_HIDDEN = {"SABLE1": 256, "SABLE2": 512, "SABLE3": 768, "SABLE5": 1024}

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data*.bin")
ap.add_argument("--val-data", required=True,
                help="held-out DGRecord glob; files must not overlap --data")
ap.add_argument("--val-size", type=int, default=1_000_000,
                help="held-out records evaluated per epoch (0 = all)")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--batch", type=int, default=16384)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--lr-schedule", default="",
                help="1-based stages, e.g. 1:0.001,11:0.0003,21:0.0001")
ap.add_argument("--lam", type=float, default=0.7)
ap.add_argument("--out", default="candidate_sable3.nnue")
ap.add_argument("--arch", type=lambda s: s.upper(), choices=tuple(ARCH_HIDDEN),
                default="SABLE3", help="explicit net format; SABLE3 is 768 hidden")
ap.add_argument("--hidden", type=int, default=None, choices=[256, 512, 768, 1024],
                help="legacy size selector; must agree with --arch when both are set")
ap.add_argument("--epoch-size", type=int, default=25_000_000)
ap.add_argument("--checkpoint", default="candidate_sable3.pt")
ap.add_argument("--snapshot-dir", default="snapshots",
                help="directory for immutable per-epoch .nnue exports (empty disables)")
ap.add_argument("--device", default=None, help="cuda / cpu (auto-detect)")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

ARCH = args.arch
HID = ARCH_HIDDEN[ARCH]
if args.hidden is not None and args.hidden != HID:
    ap.error(f"--arch {ARCH} requires --hidden {HID}, not {args.hidden}")
if args.lr <= 0 or args.epochs < 1 or args.batch < 1:
    ap.error("--lr, --epochs and --batch must be positive")
if not 0.0 <= args.lam <= 1.0:
    ap.error("--lam must be between 0 and 1")
if args.epoch_size < 0 or args.val_size < 0:
    ap.error("--epoch-size and --val-size cannot be negative")
if args.seed < 0:
    ap.error("--seed cannot be negative")

def parse_lr_stages(spec, base_lr):
    stages = {1: base_lr}
    seen = set()
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            epoch_s, lr_s = raw.split(":", 1)
            epoch, lr = int(epoch_s), float(lr_s)
        except ValueError:
            ap.error(f"invalid --lr-schedule stage: {raw!r}")
        if epoch < 1 or lr <= 0 or epoch in seen:
            ap.error(f"invalid or duplicate --lr-schedule stage: {raw!r}")
        seen.add(epoch)
        stages[epoch] = lr
    return sorted(stages.items())

LR_STAGES = parse_lr_stages(args.lr_schedule, args.lr)
LR_SCHEDULE_ID = ",".join(f"{epoch}:{lr:.12g}" for epoch, lr in LR_STAGES)

def lr_for_epoch(epoch_zero_based):
    display_epoch = epoch_zero_based + 1
    lr = LR_STAGES[0][1]
    for start, staged_lr in LR_STAGES:
        if start > display_epoch:
            break
        lr = staged_lr
    return lr

import numpy as np
import torch
import torch.nn as nn

dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
if dev.startswith("cuda") and not torch.cuda.is_available():
    sys.exit("CUDA was requested but is unavailable")
np.random.seed(args.seed & 0xFFFFFFFF)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))
print(f"architecture: {ARCH}, hidden {HID}, seed {args.seed}")
print(f"lr schedule: {LR_SCHEDULE_ID}")

REC = np.dtype([("pc", "u1", 32), ("sq", "u1", 32),
                ("score", "<i2"), ("result", "u1"), ("stm", "u1")])
def matching_files(pattern, label):
    paths = sorted(os.path.abspath(p) for p in glob.glob(pattern) if os.path.isfile(p))
    if not paths:
        sys.exit(f"no {label} files match {pattern}")
    for path in paths:
        size = os.path.getsize(path)
        if size == 0 or size % REC.itemsize:
            sys.exit(f"{label} file is empty or not a multiple of {REC.itemsize}: {path}")
    return paths

def map_files(paths):
    return [np.memmap(path, dtype=REC, mode="r") for path in paths]

files = matching_files(args.data, "training")
val_files = matching_files(args.val_data, "validation")
train_ids = {(os.stat(path).st_dev, os.stat(path).st_ino): path for path in files}
val_ids = {(os.stat(path).st_dev, os.stat(path).st_ino): path for path in val_files}
overlap = set(train_ids) & set(val_ids)
if overlap:
    pairs = [f"{train_ids[key]} == {val_ids[key]}" for key in overlap]
    sys.exit("training/validation file overlap: " + ", ".join(sorted(pairs)))
maps = map_files(files)
val_maps = map_files(val_files)
N = sum(len(m) for m in maps)
VN = sum(len(m) for m in val_maps)
TRAIN_LAYOUT = [(path, os.path.getsize(path)) for path in files]
VAL_LAYOUT = [(path, os.path.getsize(path)) for path in val_files]
print(f"training data: {N:,} positions in {len(files)} file(s) "
      f"({sum(os.path.getsize(f) for f in files)/1e9:.2f} GB, streamed)")
print(f"validation data: {VN:,} held-out positions in {len(val_files)} file(s)")

CHUNK = 1 << 18
GROUP = 8
def make_chunks(source_maps):
    result = []
    for k, source in enumerate(source_maps):
        for start in range(0, len(source), CHUNK):
            result.append((k, start, min(CHUNK, len(source) - start)))
    return result

chunks = make_chunks(maps)
val_chunks = make_chunks(val_maps)

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
    expected = {"format_version": 1, "trainer": "train_gpu.py",
                "arch": ARCH, "hidden": HID, "seed": args.seed,
                "lr_schedule": LR_SCHEDULE_ID, "lambda": args.lam,
                "batch": args.batch, "epoch_size_arg": args.epoch_size,
                "val_size": args.val_size, "train_layout": TRAIN_LAYOUT,
                "val_layout": VAL_LAYOUT}
    missing = [key for key in expected if key not in ck]
    if missing:
        sys.exit("checkpoint lacks reproducibility metadata (start a fresh run): "
                 + ", ".join(missing))
    for key, value in expected.items():
        if ck[key] != value:
            sys.exit(f"checkpoint {key} mismatch: {ck[key]!r} != {value!r}")
    net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"])
    start_epoch = int(ck["epoch"])
    if start_epoch < 0 or start_epoch > args.epochs:
        sys.exit(f"checkpoint epoch {start_epoch} is outside requested run")
    print(f"resumed from {args.checkpoint} (epoch {start_epoch})")

EXPECTED_EXPORT_SIZE = 6 + 768 * HID * 2 + HID * 2 + 2 * HID * 2 + 4

def validate_export(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        magic = f.read(6)
    if magic != ARCH.encode("ascii") or size != EXPECTED_EXPORT_SIZE:
        raise RuntimeError(
            f"invalid {ARCH} export {path}: magic={magic!r}, size={size}, "
            f"expected={EXPECTED_EXPORT_SIZE}")

def export(path):
    with torch.no_grad():
        W1 = net.emb.weight[:768].detach().cpu().numpy()
        B1 = net.b1.detach().cpu().numpy()
        W2 = net.out.weight[0].detach().cpu().numpy()
        B2 = net.out.bias[0].detach().item()
    if W1.shape != (768, HID) or B1.shape != (HID,) or W2.shape != (2 * HID,):
        raise RuntimeError(f"unexpected {ARCH} tensor shapes")
    if not (np.isfinite(W1).all() and np.isfinite(B1).all()
            and np.isfinite(W2).all() and np.isfinite(B2)):
        raise RuntimeError("refusing to export non-finite network parameters")
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(ARCH.encode("ascii"))
        np.clip(np.round(W1 * 255), -32767, 32767).astype("<i2").tofile(f)
        np.clip(np.round(B1 * 255), -32767, 32767).astype("<i2").tofile(f)
        np.clip(np.round(W2 * 64), -32767, 32767).astype("<i2").tofile(f)
        np.array([round(B2 * 255 * 64)], dtype="<i4").tofile(f)
    validate_export(tmp)
    os.replace(tmp, path)
    validate_export(path)

def save_checkpoint(payload, path):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)

def snapshot_path(display_epoch):
    if not args.snapshot_dir:
        return None
    base = os.path.basename(args.out)
    stem, ext = os.path.splitext(base)
    return os.path.join(args.snapshot_dir,
                        f"{stem}.epoch-{display_epoch:03d}{ext or '.nnue'}")

def validation_records(limit, batch_size):
    if not val_chunks:
        return
    order = np.random.default_rng(args.seed + 9_000_001).permutation(len(val_chunks))
    raw_done = 0
    for chunk_index in order:
        k, start, count = val_chunks[chunk_index]
        block = val_maps[k][start:start + count]
        for offset in range(0, len(block), batch_size):
            if raw_done >= limit:
                return
            take = min(batch_size, len(block) - offset, limit - raw_done)
            yield block[offset:offset + take]
            raw_done += take

def heldout_mse(limit, batch_size):
    if not val_maps:
        return None, 0
    net.eval()
    squared_error, valid = 0.0, 0
    with torch.no_grad():
        for recs in validation_records(limit, batch_size):
            item = make_batch(recs)
            if item is None:
                continue
            iS, iN, target = item
            iS = torch.from_numpy(iS).to(dev, non_blocking=True)
            iN = torch.from_numpy(iN).to(dev, non_blocking=True)
            target = torch.from_numpy(target).to(dev, non_blocking=True)
            diff = torch.sigmoid(net(iS, iN)) - target
            squared_error += torch.sum(diff * diff).item()
            valid += int(target.numel())
    net.train()
    if valid == 0:
        sys.exit("held-out validation contained no valid records")
    return squared_error / valid, valid

B = args.batch
E = min(args.epoch_size, N) if args.epoch_size else N
V = min(args.val_size, VN) if args.val_size else VN
if E < B:
    sys.exit(f"epoch sample ({E}) is smaller than one batch ({B})")
print(f"training: {N:,} positions ({E:,} sampled/epoch), batch {B}, lambda {args.lam}")
for epoch in range(start_epoch, args.epochs):
    current_lr = lr_for_epoch(epoch)
    for group in opt.param_groups:
        group["lr"] = current_lr
    net.train()
    erng = np.random.default_rng(args.seed + 1000 + epoch)
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
        tot += loss.item(); nb += 1
        if nb % 200 == 0:
            print(f"  epoch {epoch+1} batch {nb}: loss {tot/nb:.5f} "
                  f"lr {current_lr:.3g} ({nb*B/(time.time()-t0):,.0f} pos/s)",
                  flush=True)
    th.join()
    if nb == 0:
        sys.exit("training produced no valid batches")
    dt = time.time() - t0
    train_mse = tot / nb
    val_mse, val_records = heldout_mse(V, B)
    val_text = (f", val_mse {val_mse:.6f} ({val_records:,})"
                if val_mse is not None else "")
    print(f"epoch {epoch+1}/{args.epochs}: train_mse {train_mse:.6f}{val_text}, "
          f"lr {current_lr:.3g} "
          f"({dt:.0f}s, {nb*B/dt:,.0f} pos/s)", flush=True)
    export(args.out)
    snap = snapshot_path(epoch + 1)
    if snap:
        export(snap)
    save_checkpoint({"format_version": 1, "trainer": "train_gpu.py",
                     "arch": ARCH, "hidden": HID, "seed": args.seed,
                     "lr_schedule": LR_SCHEDULE_ID, "lr": current_lr,
                     "lambda": args.lam, "batch": args.batch,
                     "epoch_size_arg": args.epoch_size, "epoch_size": E,
                     "val_size": args.val_size, "train_layout": TRAIN_LAYOUT,
                     "val_layout": VAL_LAYOUT,
                     "train_mse": train_mse, "val_mse": val_mse,
                     "val_records": val_records,
                     "net": net.state_dict(), "opt": opt.state_dict(),
                     "epoch": epoch + 1}, args.checkpoint)
    print(f"  exported {args.out}" + (f" + {snap}" if snap else "")
          + " + checkpoint", flush=True)
print("done.")
