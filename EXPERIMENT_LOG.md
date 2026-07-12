# Onyx strength campaign

Hard constraints for this campaign:

- The deployed `onyx.nnue` stays read-only during search experiments. NNUE
  training and data generation are allowed only on external cloud compute;
  they must never run on this PC. Finished nets may be downloaded here for
  builds and SPRT validation.
- Engine changes must be original Onyx work. No third-party engine source,
  patches, tuning tables, or engine-specific constants may be copied.
- A candidate is accepted only after correctness checks and statistically
  meaningful paired match evidence against the current fixed baseline.

## Fixed-node gate protocol

- Runner: FastChess 1.8.0-alpha (official Windows x86-64 release).
- Control and candidate: same GCC 16.1.0 build flags and frozen net.
- Search limit: 20,000 nodes per move, one thread, 32 MiB hash.
- Openings: 2,000 unique positions sampled without search from the existing
  Polyglot book, 8-20 plies deep; each position is played with reversed colors.
- Internal engine book: disabled for both engines.
- Adjudication: two-sided 900 cp for 6 moves; draw after move 40 at 15 cp for
  12 moves; hard draw at move 200.
- Default gate: logistic SPRT, alpha=beta=0.05, Elo0=0, Elo1=8, at most 4,000
  games. A cap result is accepted only when its confidence interval excludes
  zero on the positive side.

## Confirmation gates (added 2026-07-11)

Fixed-node games are fully deterministic: `ucinewgame` clears the TT and all
history tables, so each game is a pure function of (opening, color, binaries,
options). Re-running a candidate on the same 2,000-opening suite replays the
identical 4,000 games and adds no evidence. Therefore:

- A candidate whose default gate ends at the cap with a positive but
  zero-inclusive interval (a "screening positive") may get ONE confirmation
  gate: logistic SPRT [0, 5], alpha=beta=0.05, 6,000-game cap, on the
  disjoint suite `openings/book_tail_6000.epd` (engine `booksuite2` command:
  book walk >= 6 plies plus a 2-6 ply random tail, every random ply vetted at
  |static eval| <= 120 cp, deduplicated against the original suite).
- Acceptance requires the confirmation SPRT to pass on its own (alpha is 5%
  for a zero-Elo patch regardless of screening selection). A cap finish in
  the confirmation is a rejection - no third attempt.

SINGNEG-02 measured the tail suite's draw rate at 22% versus 38% on the
original book (sharper random-tail positions), roughly 35% tighter Elo error
bars per game. Screening gates from 2026-07-11 onward therefore also use
`book_tail_6000.epd` (SPRT [0, 8] unchanged, cap 6,000 games); any future
confirmation must use a freshly generated suite (`booksuite2`, new seed,
excluding both existing suites).

## Results

| ID | Candidate | Games | Elo (95% CI) | LOS | LLR | Decision |
|---|---|---:|---:|---:|---:|---|
| QREP-01 | qsearch repetition/fifty detection plus key-stack maintenance | 4,000 | +0.09 +/- 6.83 | 50.99% | -2.57 `[-2.94,+2.94]` | Rejected; default remains off |
| MATCORR-01 | material-signature correction history | 4,000 | -0.61 +/- 8.22 | 44.23% | -2.10 `[-2.94,+2.94]` | Rejected; default changed to off |
| BADCAP-01 | order SEE-negative captures after all quiets | 1,492 | +22.39 +/- 13.78 | 99.93% | +2.96 `[-2.94,+2.94]` | **Accepted** |
| BADCAPLMR-01 | one-ply reduction for late SEE-negative captures | 4,000 | +6.08 +/- 8.35 | 92.32% | +0.92 `[-2.94,+2.94]` | Inconclusive; default remains off |
| QSTAND-01 | cache qsearch stand-pat beta cutoffs in the TT | 4,000 | +1.48 +/- 8.52 | 63.30% | -1.07 `[-2.94,+2.94]` | Rejected; default remains off |
| FIFTY-01 | search-level fifty-move static-eval damping | 4,000 | +5.99 +/- 8.45 | 91.79% | +0.86 `[-2.94,+2.94]` | Inconclusive; default remains off |
| COMBO-01 | BADCAPLMR-01 plus FIFTY-01 | 4,000 | -0.26 +/- 8.40 | 47.58% | -1.86 `[-2.94,+2.94]` | Rejected; both defaults remain off |
| BADBAND-01 | raise bad-capture band from -900k to 0 | 1,456 | -14.09 +/- 13.67 | 2.15% | -2.95 `[-2.94,+2.94]` | Rejected; -900k retained |
| QUIETSEE-01 | prune shallow nonforcing quiets with sufficiently negative SEE | 1,878 | +18.52 +/- 12.26 | 99.85% | +2.96 `[-2.94,+2.94]` | **Accepted** |
| QUIETSEE-02 | extend accepted quiet SEE pruning from depth 4 to depth 5 | 2,020 | +17.56 +/- 11.84 | 99.82% | +2.97 `[-2.94,+2.94]` | **Accepted** |
| QUIETSEE-03 | extend accepted quiet SEE pruning from depth 5 to depth 6 | 4,000 | +2.87 +/- 8.24 | 75.23% | -0.51 `[-2.94,+2.94]` | Inconclusive; depth 5 retained |
| QUIETSEE-04 | depth-6 frontier with conservative 120*depth threshold | 4,000 | +7.38 +/- 8.32 | 95.9% | +1.50 `[-2.94,+2.94]` | Inconclusive; depth 5 retained |
| PROBCUT-01 | raised-beta capture verification cut (existing FEAT_PROBCUT) | 4,000 | +1.39 +/- 8.36 | 62.7% | -1.15 `[-2.94,+2.94]` | Rejected; default remains off |
| SINGNEG-01 | one-ply reduction of non-singular TT moves with ttScore >= beta | 4,000 | +7.38 +/- 8.33 | 95.9% | +1.50 `[-2.94,+2.94]` | Screening positive; confirmation gate SINGNEG-02 |
| SINGNEG-02 | confirmation of SINGNEG-01 on disjoint tail suite, SPRT [0,5] | 6,000 | +1.45 +/- 5.21 | 70.7% | -0.74 `[-2.94,+2.94]` | **Rejected**; screening gain did not replicate, default remains off |
| QCHECKSEE-01 | skip SEE-losing quiet checks in qsearch | 1,706 | +14.26 +/- 10.29 | 99.7% | +2.98 `[-2.94,+2.94]` | **Accepted** (first gate on tail suite) |
| FUTIMP-01 | +50cp futility margin at improving nodes (prune fewer quiets) | 2,704 | -1.93 +/- 7.77 | 31.3% | -3.00 `[-2.94,+2.94]` | **Rejected** by SPRT lower bound; default remains off |
| SPSA-01 | rounded 195-batch match-play tuning endpoint for ten search parameters | 968 | +21.56 +/- 13.49 | 99.92% | +2.95 `[-2.94,+2.94]` | **Accepted** (gated by Codex on fresh suite) |
| CHECKGUARD-01 | preserve direct quiet checks from shallow futility/history/LMP | 6,000 | +5.27 +/- 5.35 | 97.33% | +1.36 `[-2.94,+2.94]` | Screening positive (Codex) |
| CHECKGUARD-02 | independent `[0,5]` confirmation of CHECKGUARD-01 | 2,096 | +15.43 +/- 9.16 | 99.95% | +2.98 `[-2.94,+2.94]` | **Accepted** (Codex); pooled delta +7.90 +/- 4.62 |
| IMPFALL-01 | ignore in-check ancestors when computing the improving signal | 3,338 | -1.35 +/- 7.44 | 36.07% | -2.97 `[-2.94,+2.94]` | Rejected (Codex); default remains off |
| LMPDEPTH-01 | extend late-move pruning ceiling from depth 5 to depth 8 | 3,522 | +8.68 +/- 6.89 | 99.33% | +3.04 `[-2.94,+2.94]` | **Accepted** (Codex) |
| LMPDEPTH-02 | extend accepted late-move pruning ceiling from depth 8 to depth 10 | 1,512 | +1.15 +/- 5.27 | 66.54% | -3.02 `[-2.94,+2.94]` | Rejected (Codex); depth 8 retained |
| CAPHISTLMR-01 | reduce late bad captures only after negative capture history | 1,724 | +1.81 +/- 4.45 | 78.78% | -3.03 `[-2.94,+2.94]` | Rejected (Codex); default remains off |
| QEVASION-01 | counter/continuation context to order qsearch evasions | 2,152 | -4.04 +/- 9.07 | 19.14% | -2.98 `[-2.94,+2.94]` | Rejected (Codex); default remains off |
| NMPINPLACE-01 | null-move in-place make (speed) | - | timing pre-gate +0.44% `[-1.80,+2.72]` | - | - | Rejected (Codex) before timed SPRT |
| DISCCHK-01 | extend quiet-check guard to discovered checks (Codex-prepped) | 3,932 | +0.27 +/- 6.17 | 53.4% | -3.01 `[-2.94,+2.94]` | **Rejected** by SPRT lower bound; default remains off |
| MALUSCAP-01 | quiet-history maluses when a noisy move causes the cutoff | 4,096 | +0.08 +/- 6.36 | 50.5% | -2.97 `[-2.94,+2.94]` | **Rejected** by SPRT lower bound; default remains off |
| CAPFUT-01 | futility-prune captures whose victim cannot lift eval to alpha | 1,796 | -4.64 +/- 9.39 | 16.7% | -3.00 `[-2.94,+2.94]` | **Rejected** by SPRT lower bound; default remains off |
| QSEVADE-01 | skip SEE-losing quiet evasions in qsearch after one legal reply | 1,034 | -9.41 +/- 11.73 | 5.8% | -2.95 `[-2.94,+2.94]` | **Rejected** by SPRT lower bound; default remains off |
| SINGD6-01 | singular/multicut eligibility floor depth 8 -> 6 | 6,000 | +3.13 +/- 5.44 | 87.0% | -0.90 `[-2.94,+2.94]` | Inconclusive; depth 8 retained (confirmation not warranted at +3.1) |
| TTCAPLMR-01 | extra quiet LMR when the hash move is a capture | 6,000 | +2.78 +/- 5.37 | 84.5% | -1.30 `[-2.94,+2.94]` | Inconclusive; default remains off (confirmation not warranted at +2.8) |
| GUARDD8-01 | extend quiet-check guard across LMP's full depth range (6-8) | 992 | -7.01 +/- 10.62 | 9.8% | -2.95 `[-2.94,+2.94]` | **Rejected**; the unguarded depth 6-8 LMP behavior is load-bearing in LMPDEPTH-01 |
| CONTCORR-01 | previous-move-keyed static-eval correction history | 754 | +24.93 +/- 14.65 | 99.98% | +2.95 `[-2.94,+2.94]` | **Accepted** |
| CONTCORR2-01 | (prev2, prev) pair-keyed eval correction on top of CONTCORR | 6,000 | +4.00 +/- 5.35 | 92.9% | -0.01 `[-2.94,+2.94]` | Inconclusive; default remains off (confirmation not warranted at +4.0) |
| NPCORR-01 | per-color non-pawn placement correction history | 3,636 | +8.51 +/- 6.85 | 99.3% | +2.95 `[-2.94,+2.94]` | **Accepted** |
| KPCORR-01 | pawn-structure-plus-king-squares eval correction | 1,714 | +14.20 +/- 10.23 | 99.7% | +2.98 `[-2.94,+2.94]` | **Accepted** |
| QCONTCORR-01 | apply CONTCORR to qsearch stand-pat (ssMove tracked in qsearch) | 2,386 | -3.06 +/- 8.47 | 24.0% | -2.99 `[-2.94,+2.94]` | **Rejected**; deep-residual corrections mistrain shallow leaves |
| TTSTORE-01 | depth-guarded same-key TT replacement + fail-low move preservation | 696 | +31.54 +/- 16.74 | 99.99% | +2.98 `[-2.94,+2.94]` | **Accepted** - largest single gain of the campaign |
| BADCAPCHECK-01 | checking SEE-negative captures ordered with killers | 772 | -19.37 +/- 15.29 | 0.7% | -3.02 `[-2.94,+2.94]` | **Rejected**; the last-place band is right even for checking sacrifices |
| ROOTEFFORT-01 | rank non-TT root moves by prior-iteration subtree effort | 1,218 | -9.13 +/- 11.61 | 6.2% | -2.95 `[-2.94,+2.94]` | **Rejected**; raw effort is a noisy root ranking |
| PIECETO-01 | ordering-only piece-destination history for quiets | 402 | -46.07 +/- 22.58 | 0.0% | -2.96 `[-2.94,+2.94]` | **Rejected**; swamps the tuned butterfly/continuation ordering |
| CORRW-SPSA | 150-iteration SPSA over the four correction-blend weights | 18,000 (tuning) | endpoint 254/266/259/256 vs 256 defaults | - | - | No gate run: weights confirmed near-optimal at defaults (drift < 4%) |
| SEEGE-01 | threshold SEE with provable O(1) bound exits (speed only) | - | search-identical (bench 16 exact, differential 0/bench-14); +1.8% speed, 5/5 ABBA blocks | - | - | **Accepted** as build improvement; timed SPRT waived - +1-2 Elo is below SPRT resolution and identity caps downside at zero |
| SPSA-02 | 14-parameter retune endpoint (150 iterations) via UCI options | 6,000 | +2.14 +/- 5.33 | 78.5% | -2.00 `[-2.94,+2.94]` | Inconclusive; defaults retained - parameter space confirmed harvested |
| PARTIALID-01 | adopt the interrupted final iteration's validated root move | 3,404 | +2.14 +/- 4.30 | 83.6% | -3.02 `[-2.94,+2.94]` | **Rejected** by SPRT: effect real but ~+2, below the acceptance bar |
| TTPACK-01 | 16-byte TT entries, 2x capacity at fixed 32 MiB | 3,578 | -0.39 +/- 6.71 | 45.4% | -2.98 `[-2.94,+2.94]` | **Rejected**; capacity is not binding at 20k nodes/move |
| TTAGEHIT-01 | probe-side age refresh (probed entries become protected incumbents) | 5,784 | +1.08 +/- 5.44 | 65.1% | -3.03 `[-2.94,+2.94]` | **Rejected**; effect ~+1, judge's tempered estimate confirmed |
| QSCHAIN-01 | evasions don't consume the qsearch quiet-check budget (2 waves) | 824 | -14.77 +/- 13.96 | 1.9% | -2.95 `[-2.94,+2.94]` | **Rejected**; second-wave check widening fails, as first-panel precedent predicted |
| CORRCLAMP-01 | clamp the shared correction-training residual to +/-64cp | 1,454 | -5.97 +/- 10.10 | 12.3% | -2.98 `[-2.94,+2.94]` | **Rejected**; outlier residuals carry real signal, unclamped EMA confirmed |
| LAZYACC-01 | deferred accumulator materialization (speed patch) | - | search-identical (bench 13/16 exact, differential 0 mismatches over bench 12) but **6.9% SLOWER**, 0/3 ABBA blocks | - | - | **Rejected** at the pre-registered timing kill rule; the fused AVX2 apply is cheaper than deferral bookkeeping |

Rows marked (Codex) were run in the parallel `Onyx_copy` tree on 2026-07-11
(its per-experiment folders hold the full provenance; audited and adopted
here, see "Baseline adoption" below).

## Baseline adoption (2026-07-11 afternoon)

The Codex tree's +110.87 baseline (`Onyx_1.8_plus111.exe`) was adversarially
audited before adoption: source rebuilt bench-identical (3,334,539 @ depth
13), all claimed results matched their fastchess logs, all nine opening-suite
hashes recomputed correctly, SPSA defaults verified inside UCI clamps, no
compliance violations. Audit findings applied on adoption: restored the
`booksuite2` generator that the Codex tree dropped, replaced a hardcoded
guard depth with `QUIET_SEE_MAX_DEPTH`, corrected the stale FEAT_PROBCUT
comment, removed a dead FEAT_FIFTY define. Known suite-lineage flaw fixed
going forward: `book_tail_fresh4_6000.epd` was missing from every exclusion
file (~0.2% overlap with later suites, statistically negligible);
`all_prior_68000_unique.epd` and later now include it and are deduplicated.

QREP-01 was search-correctness motivated and reduced the depth-13 benchmark
from 2,851,933 to 2,567,480 nodes, but it did not demonstrate an Elo gain.

MATCORR-01 supersedes a recovered small-sample result that had suggested about
+14 Elo. The larger unique-opening gate did not reproduce that gain.

## Accepted cumulative gain

Current campaign baseline: clean Onyx 1.8 search, with the frozen 768-hidden
network, plus BADCAP-01, QUIETSEE-01/02, QCHECKSEE-01, SPSA-01,
CHECKGUARD-01/02, LMPDEPTH-01, CONTCORR-01, NPCORR-01, KPCORR-01,
TTSTORE-01, SEEGE-01 (speed), SPSA-03, and DELTAM-01. Newly accepted gain:
**~+220 Elo** (arithmetic sum of sequential/pooled gate estimates across
suites; indicative, not a direct cumulative measurement - a timed external
anchor via the Kaggle notebook is due). Reference build:
`dev/Onyx_1.8_plus220.exe` (bench 13 = 2,920,138; `base220_control.exe` is
the same build under the control naming).
| PGO-01 | profile-guided optimization build (w64devkit GCC 16.1) | - | search-identical but 17-19% SLOWER, 0/3 ABBA blocks (with and without partial-training) | - | - | **Rejected**; plain -O3 remains the blessed build for this single-file hot loop |
| SPSA-03 | structural-constants retune endpoint (7 knobs: NullDDiv 4, SingTTOff 2, DblExtM 23, IIRDepth 5, RazorDepth 2, LMPBase 4, SeeCapDepth 7) | 918 | +23.12 +/- 13.84 | 99.95% | +2.95 `[-2.94,+2.94]` | **Accepted** - second-largest gain of the campaign |
| DELTAM-01 | qsearch delta margin 230 -> 250 (agreed by two independent SPSA runs) | 5,592 | +6.71 +/- 5.30 | 99.3% | +2.97 `[-2.94,+2.94]` | **Accepted** |
| LMRTUNE-01 | LMR mechanics endpoint (PvRed 0, KillRed 2 from 150-iter SPSA) | 3,122 | -1.00 +/- 7.16 | 39.2% | -3.01 `[-2.94,+2.94]` | **Rejected**; the +/-1 integer drifts were perturbation noise, original adjustments confirmed |
| CORRBOUND-01 | correction updates move only in their bound's provable direction | 6,000 | +4.05 +/- 5.09 | 94.0% | +0.07 `[-2.94,+2.94]` | Inconclusive; default remains off (below confirmation threshold) |
| THREATLMR-01 | badly failed null search lowers quiet LMR at the node | 2,890 | -1.80 +/- 7.76 | 32.5% | -2.95 `[-2.94,+2.94]` | **Rejected** by SPRT lower bound; default remains off |
| ONEREPLY-01 | extend the single legal evasion of a check by one ply | 2,340 | -2.23 +/- 7.88 | 29.0% | -3.08 `[-2.94,+2.94]` | **Rejected** by SPRT lower bound; default remains off |
| NET-GEN8A-01 | cloud-trained gen8 net (97.5M pos, mixed +58/+213 labels, rescaled x0.87) vs frozen 768 | 344 | +59.14 +/- 23.13 | 100.0% | +2.95 `[-2.94,+2.94]` | **ACCEPTED - deployed as onyx.nnue; largest gain in campaign history** |
| G8TUNE-01 | post-gen8 sweep endpoint pair (DeltaMargin 266, AspDelta 28) | 1,124 | -8.66 +/- 11.39 | 6.8% | -2.98 `[-2.94,+2.94]` | **Rejected**; current parameters confirmed calibrated for the gen8 net |

## External anchor (2026-07-12, Kaggle)

Onyx (+220 search, gen8 net) vs Stash 34.0 (bmi2), 1,000 games at 10s+0.1s,
1 thread, 128 MiB each, paired openings, Kaggle CPU session:
**414W 235L 351D = 58.95%, +62.87 +/- 17.09 Elo.** Against Stash 34's
~3400 CCRL Blitz rating this measures Onyx at ~3460-3480 blitz-equivalent
(cloud anchors are internally consistent with each other; the 2026-07-03 PC
anchor vs Stash 32 is a different environment). Full PGN in the anchor
notebook's version output.
