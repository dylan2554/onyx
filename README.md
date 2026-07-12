# Onyx 2.0

A UCI chess engine, ~3390 blitz strength (CCRL-equivalent estimate), written entirely by AI
agents (Anthropic's Claude and OpenAI's Codex) working under human
direction. Every line of engine code is original to this project - nothing
was copied from Stockfish or any other engine. The NNUE network is trained
exclusively on Onyx's own self-play data, generated and trained on free
cloud compute.

Formerly named "Sable" (v1.0-1.7); renamed Onyx from v1.8.

## Files

- `onyx-2.0-win-avx2.exe`    - Windows x64, AVX2 (2013+ CPUs; recommended)
- `onyx-2.0-win-generic.exe` - Windows x64, any CPU (slower)
- `onyx.nnue`                - the gen8 neural network - KEEP NEXT TO THE EXE
- `book.bin`                 - small Polyglot opening book (optional;
                               engine plays without it, UCI OwnBook toggles)
- `src/onyx.cpp`             - complete source, one file
- `SHA256SUMS.txt`           - artifact hashes

## Install

Point any UCI GUI (Arena, Cute Chess, En Croissant, Banksia) at the exe.
Keep `onyx.nnue` in the same folder. Options: Hash (MB), Threads, OwnBook,
EvalFile, plus exposed search-tuning spins (leave at defaults).

## Build from source

    g++ -std=c++17 -O3 -march=native -static -pthread onyx.cpp -o onyx.exe

GCC 10+ on any x64 platform (Windows/Linux). AVX2 strongly recommended.

## Strength

- 1,000 games vs Stash 34.0 (CCRL Blitz 3328) at 10s+0.1s:
  **+62.9 +/- 17.1 Elo** (58.95%).
- Onyx 2.0 vs Onyx 1.8 at fixed 20k nodes/move: ~+280 Elo across the
  campaign's SPRT-gated changes (12 accepted search/eval changes plus the
  gen8 network; every change validated by sequential probability ratio
  tests on disjoint fresh opening suites).

## Architecture (v2.0)

- Bitboards with magic move generation; single-file C++17.
- NNUE 768->768x2->1 (CReLU, int16 SIMD), fused AVX2 accumulator updates,
  per-ply accumulator stack. Net trained with lambda-blended WDL/score
  targets on 97.5M self-play positions (~200M-position successor training
  in progress).
- PVS/alpha-beta: aspiration windows, TT with depth/bound-aware same-key
  replacement, quiescence TT, singular extensions + multicut, null-move
  pruning, razoring, reverse futility, futility + history + late-move
  pruning with quiet-check guards, SEE pruning of quiets and captures
  (threshold SEE fast path), killer/counter/continuation/capture history,
  and four static-eval correction histories (pawn-structure, previous-move,
  non-pawn placement per color, pawn-king).
- All search constants machine-tuned via SPSA match-play (no hand-copied
  values).

## License

MIT. (c) 2026 Dylan Hogarth and contributors (AI-generated code
directed and validated by the project owner).
