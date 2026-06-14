# Sable

A UCI chess engine written from scratch in C++, with a **self-trained NNUE evaluation**.
Estimated strength: **~3200 Elo** (see [Strength](#strength)).

Sable was built and trained on a single desktop PC (Intel i7-11700K + RTX 3070) — the
neural network's training data was generated entirely by the engine playing itself,
with no external datasets or pretrained weights.

---

## Features

**Board & move generation**
- Bitboard representation with magic-bitboard sliding-piece attacks
- Fully legal move generation (verified by perft against known node counts, including Kiwipete)
- Zobrist hashing

**Search**
- Iterative deepening with aspiration windows
- Principal Variation Search (PVS) inside alpha-beta
- Transposition table with depth-preferred replacement and aging
- Null-move pruning
- Late Move Reductions (history-adjusted)
- Singular extensions (with double extensions and multi-cut)
- Reverse futility pruning, futility pruning, razoring
- Late move pruning and history pruning
- SEE-based pruning of losing captures
- Killer moves, counter-moves, and butterfly + continuation history (with maluses)
- Quiescence search with SEE filtering, delta pruning, and quiet checks
- Mate-distance pruning
- **Lazy SMP** multithreading (1–16 threads)

**Evaluation**
- NNUE: `768 → 512×2 (perspective accumulators) → 1`, clipped-ReLU, quantized int16
- Incrementally updated accumulators, accelerated with AVX2 SIMD
- Falls back to a hand-crafted tapered evaluation (PeSTO + king safety, pawn structure,
  mobility) when no network file is present

**Training pipeline** (included)
- `sable datagen` — self-play data generation, writing labeled positions to disk
- `train_gpu.py` — PyTorch NNUE trainer (GPU or CPU)
- `train.py` — pure-NumPy trainer (no GPU required)
- The net was trained across several "generations": each generation regenerates data
  using the previous, stronger network to label positions, then retrains.

---

## Usage

Sable speaks the [UCI protocol](https://en.wikipedia.org/wiki/Universal_Chess_Interface),
so it works in any UCI GUI (Arena, Cute Chess, BanksiaGUI, etc.).

1. Download `sable.exe` and `sable.nnue` from the [Releases](../../releases) page.
2. Keep **both files in the same folder** — the engine loads `sable.nnue` automatically.
3. In your GUI: add a new UCI engine and point it at `sable.exe`.

UCI options:
- `Hash` (MB) — transposition table size
- `Threads` — number of search threads
- `EvalFile` — path to an NNUE file (defaults to `sable.nnue`)

## Building from source

Requires a C++17 compiler.

```sh
# Linux / macOS
g++ -O3 -mavx2 -pthread sable.cpp -o sable

# Windows (MinGW-w64)
g++ -O3 -mavx2 -static -static-libgcc -static-libstdc++ -pthread sable.cpp -o sable.exe
```

A build without `-mavx2` also works (slower NNUE inference) for older CPUs.

## Strength

Sable scored **45% in a 100-game match against Stockfish limited to UCI_Elo 3000**
(time control 3s + 0.05s, 8 threads), for an estimated **~2965 Elo** at that setting.

| Opponent (Stockfish UCI_Elo) | Result (Sable) | Games |
|------------------------------|----------------|-------|
| 1600 | 99% | 100 |
| 2200 | 94% | 100 |
| 2600 | 70% | 100 |
| 2800 | 52% | 100 |
| 3000 | 45% | 100 |

Strength is estimated from the match score against opponents of known rating. Note that
published engine-list ratings (e.g. CCRL) are measured single-threaded on normalized
hardware, so a directly-comparable single-thread figure would be somewhat lower; these
numbers reflect 8-thread play at fast time control.

## Training your own network

See `train_gpu.py` (or `train.py`). The pipeline is:

1. Generate self-play data: `sable datagen <games> <nodes_per_move> <out.bin> <seed> <hashMB>`
2. Train: `python train_gpu.py --data "*.bin" --epochs 30 --lam 0.85`
3. The trainer exports `sable.nnue` after every epoch; drop it next to the engine.

## About this project

Sable was built as a collaboration between me (Dylan) and Anthropic's Claude. Claude
wrote the engine and training code; I directed the design, ran all training and testing
on my own hardware (i7-11700K + RTX 3070), debugged the pipeline, and verified strength
at every step. The neural network was trained from zero on self-play games generated on
my own PC — no external data and no pretrained weights.

I'm releasing it to show what one person and a current AI model can build together in a
few days: a chess engine going from non-existent to ~2965 Elo in under a week.

## License

[MIT](LICENSE) © Dylan

## Acknowledgements

Engine and training code written with Anthropic's Claude. Evaluation piece-square tables
are based on the public-domain PeSTO values; search techniques follow ideas long
established in the open-source computer-chess community (the Chess Programming Wiki,
Stockfish, and others).
