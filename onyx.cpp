// ============================================================
//  Onyx 2.0 — a UCI chess engine written by AI agents
//  (Anthropic Claude and OpenAI Codex, under human direction;
//  formerly "Sable", renamed because that name was taken)
//
//  Bitboards + magic move generation; PVS/alpha-beta with TT,
//  null move, singular extensions + multicut, razoring/RFP,
//  futility + history + late-move + SEE pruning with quiet-check
//  guards, killer/counter/continuation/capture history, and four
//  static-eval correction histories (pawn, previous-move,
//  non-pawn placement, pawn-king). NNUE 768->768x2->1 (gen8 net,
//  onyx.nnue) with fused AVX2 accumulator updates and a PeSTO
//  classical fallback. All constants machine-tuned via SPSA
//  match play; every change SPRT-gated (~45 experiments; see the
//  repo's EXPERIMENT_LOG.md for the complete accept/reject
//  history, including 2.0's twelve accepted changes:
//  BADCAP/QUIETSEEx2/QCHECKSEE/CHECKGUARD/LMPDEPTH/CONTCORR/
//  NPCORR/KPCORR/TTSTORE/SEEGE/SPSA-01+03/DELTAM + the gen8 net).
//
//  Measured: +62.9 +/- 17.1 vs Stash 34.0 (CCRL Blitz 3328), 1,000 games @ 10s+0.1s -> ~3390 CCRL-Blitz-equivalent.
//
//  Single file. Compile:  g++ -O3 -march=native -pthread onyx.cpp -o onyx
// ============================================================
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <algorithm>

typedef uint64_t U64;
using namespace std;

// ----------------------- basics ----------------------------
enum { WHITE, BLACK, BOTH };
enum { WP, WN, WB, WR, WQ, WK, BP, BN, BB_, BR, BQ, BK, NO_PIECE };
enum { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };

static inline int pieceType(int pc) { return pc % 6; }
static inline int pieceColor(int pc) { return pc / 6; }

#define SQ(f, r) ((r) * 8 + (f))
static inline int fileOf(int s) { return s & 7; }
static inline int rankOf(int s) { return s >> 3; }

static inline int lsb(U64 b) { return __builtin_ctzll(b); }
static inline int popcnt(U64 b) { return __builtin_popcountll(b); }
static inline int poplsb(U64 &b) { int s = lsb(b); b &= b - 1; return s; }

const U64 FILE_A = 0x0101010101010101ULL;
const U64 FILE_H = 0x8080808080808080ULL;
const U64 RANK_1 = 0xFFULL;
const U64 RANK_2 = 0xFF00ULL;
const U64 RANK_7 = 0xFF000000000000ULL;
const U64 RANK_8 = 0xFF00000000000000ULL;

// ----------------------- attack tables ---------------------
U64 pawnAtt[2][64];
U64 knightAtt[64];
U64 kingAtt[64];

// magic bitboards
struct Magic {
    U64 mask;
    U64 magic;
    int shift;
    U64 *attacks;
};
Magic rookMagics[64], bishopMagics[64];
vector<U64> rookTable[64], bishopTable[64];

static U64 rngState = 0x9E3779B97F4A7C15ULL;
static inline U64 rand64() {
    rngState ^= rngState << 13;
    rngState ^= rngState >> 7;
    rngState ^= rngState << 17;
    return rngState;
}
static inline U64 randSparse() { return rand64() & rand64() & rand64(); }

U64 slidingAttack(int sq, U64 block, bool rook) {
    U64 att = 0;
    int dr[2][4][2] = {{{1,1},{1,-1},{-1,1},{-1,-1}},   // bishop
                       {{1,0},{-1,0},{0,1},{0,-1}}};    // rook
    int idx = rook ? 1 : 0;
    for (int d = 0; d < 4; d++) {
        int f = fileOf(sq), r = rankOf(sq);
        while (true) {
            f += dr[idx][d][1]; r += dr[idx][d][0];
            if (f < 0 || f > 7 || r < 0 || r > 7) break;
            att |= 1ULL << SQ(f, r);
            if (block & (1ULL << SQ(f, r))) break;
        }
    }
    return att;
}

U64 sliderMask(int sq, bool rook) {
    U64 att = 0;
    int f0 = fileOf(sq), r0 = rankOf(sq);
    if (rook) {
        for (int r = r0 + 1; r <= 6; r++) att |= 1ULL << SQ(f0, r);
        for (int r = r0 - 1; r >= 1; r--) att |= 1ULL << SQ(f0, r);
        for (int f = f0 + 1; f <= 6; f++) att |= 1ULL << SQ(f, r0);
        for (int f = f0 - 1; f >= 1; f--) att |= 1ULL << SQ(f, r0);
    } else {
        for (int f = f0 + 1, r = r0 + 1; f <= 6 && r <= 6; f++, r++) att |= 1ULL << SQ(f, r);
        for (int f = f0 + 1, r = r0 - 1; f <= 6 && r >= 1; f++, r--) att |= 1ULL << SQ(f, r);
        for (int f = f0 - 1, r = r0 + 1; f >= 1 && r <= 6; f--, r++) att |= 1ULL << SQ(f, r);
        for (int f = f0 - 1, r = r0 - 1; f >= 1 && r >= 1; f--, r--) att |= 1ULL << SQ(f, r);
    }
    return att;
}

void initMagic(int sq, bool rook) {
    Magic &m = rook ? rookMagics[sq] : bishopMagics[sq];
    m.mask = sliderMask(sq, rook);
    int bits = popcnt(m.mask);
    m.shift = 64 - bits;
    int size = 1 << bits;

    vector<U64> occs(size), refs(size);
    U64 b = 0;
    for (int i = 0; i < size; i++) {
        occs[i] = b;
        refs[i] = slidingAttack(sq, b, rook);
        b = (b - m.mask) & m.mask; // carry-rippler
    }
    vector<U64> &table = rook ? rookTable[sq] : bishopTable[sq];
    table.assign(size, 0);
    vector<int> epoch(size, -1);
    int tries = 0;
    while (true) {
        m.magic = randSparse();
        if (popcnt((m.mask * m.magic) >> 56) < 6) continue;
        bool ok = true;
        tries++;
        for (int i = 0; i < size; i++) {
            unsigned idx = (unsigned)((occs[i] * m.magic) >> m.shift);
            if (epoch[idx] != tries) { epoch[idx] = tries; table[idx] = refs[i]; }
            else if (table[idx] != refs[i]) { ok = false; break; }
        }
        if (ok) break;
    }
    m.attacks = table.data();
}

static inline U64 rookAttacks(int sq, U64 occ) {
    Magic &m = rookMagics[sq];
    return m.attacks[((occ & m.mask) * m.magic) >> m.shift];
}
static inline U64 bishopAttacks(int sq, U64 occ) {
    Magic &m = bishopMagics[sq];
    return m.attacks[((occ & m.mask) * m.magic) >> m.shift];
}
static inline U64 queenAttacks(int sq, U64 occ) {
    return rookAttacks(sq, occ) | bishopAttacks(sq, occ);
}

void initAttacks() {
    for (int sq = 0; sq < 64; sq++) {
        int f = fileOf(sq), r = rankOf(sq);
        U64 b = 1ULL << sq;
        pawnAtt[WHITE][sq] = ((b << 7) & ~FILE_H) | ((b << 9) & ~FILE_A);
        pawnAtt[BLACK][sq] = ((b >> 7) & ~FILE_A) | ((b >> 9) & ~FILE_H);
        U64 n = 0;
        int nd[8][2] = {{1,2},{2,1},{2,-1},{1,-2},{-1,-2},{-2,-1},{-2,1},{-1,2}};
        for (auto &d : nd) {
            int nf = f + d[0], nr = r + d[1];
            if (nf >= 0 && nf <= 7 && nr >= 0 && nr <= 7) n |= 1ULL << SQ(nf, nr);
        }
        knightAtt[sq] = n;
        U64 k = 0;
        for (int df = -1; df <= 1; df++)
            for (int dr2 = -1; dr2 <= 1; dr2++) {
                if (!df && !dr2) continue;
                int nf = f + df, nr = r + dr2;
                if (nf >= 0 && nf <= 7 && nr >= 0 && nr <= 7) k |= 1ULL << SQ(nf, nr);
            }
        kingAtt[sq] = k;
        initMagic(sq, true);
        initMagic(sq, false);
    }
}

// ----------------------- zobrist ---------------------------
U64 zPiece[12][64], zCastle[16], zEp[8], zSide;
void initZobrist() {
    for (int p = 0; p < 12; p++)
        for (int s = 0; s < 64; s++) zPiece[p][s] = rand64();
    for (int i = 0; i < 16; i++) zCastle[i] = rand64();
    for (int i = 0; i < 8; i++) zEp[i] = rand64();
    zSide = rand64();
}

// ----------------------- NNUE ------------------------------
// arch: 768 -> (256 x 2 perspectives, shared weights) -> 1
// quantization: L1 weights/bias x255 (int16), L2 weights x64 (int16)
#ifndef NN_MAX_HIDDEN
#define NN_MAX_HIDDEN 768   // build with -DNN_MAX_HIDDEN=1024 for SABLE5 nets
#endif
const int NN_MAX_BUCKETS = 8;           // output buckets by piece count (SABLE4)
int nnHidden = 256;                     // set from net file magic
int nnBuckets = 1;
bool nnueLoaded = false;
alignas(64) int16_t nnW1[768][NN_MAX_HIDDEN];
int16_t nnB1[NN_MAX_HIDDEN];
int16_t nnW2[NN_MAX_BUCKETS][2 * NN_MAX_HIDDEN];
int32_t nnB2[NN_MAX_BUCKETS];

struct alignas(64) Accumulator { int16_t v[2][NN_MAX_HIDDEN]; };

// ----------------------- position --------------------------
#ifndef FEAT_NPCORR
#define FEAT_NPCORR 1  // +8.51 +/- 6.85 elo, 3,636-game SPRT (NPCORR-01)
#endif
#ifndef FEAT_LAZYACC
#define FEAT_LAZYACC 0  // defer accumulator updates until an eval needs them (2 = differential check)
#endif

struct Position {
    U64 bb[12];
    U64 occ[3];
    int side;
    int ep;        // ep square or -1
    int castle;    // 1 WK 2 WQ 4 BK 8 BQ
    int fifty;
    U64 key;
    U64 pawnKey;         // zobrist over pawns only (correction history)
#if FEAT_NPCORR
    U64 npKey[2];        // zobrist over each color's non-pawn pieces
#endif
    uint8_t board[64];   // mailbox: piece index or NO_PIECE

    void updateOcc() {
        occ[WHITE] = occ[BLACK] = 0;
        for (int p = WP; p <= WK; p++) occ[WHITE] |= bb[p];
        for (int p = BP; p <= BK; p++) occ[BLACK] |= bb[p];
        occ[BOTH] = occ[WHITE] | occ[BLACK];
    }
    void fillBoard() {
        memset(board, NO_PIECE, sizeof(board));
        for (int p = 0; p < 12; p++) {
            U64 b = bb[p];
            while (b) board[poplsb(b)] = (uint8_t)p;
        }
    }
    int pieceOn(int sq) const { return board[sq]; }
    U64 computeKey() const {
        U64 k = 0;
        for (int p = 0; p < 12; p++) {
            U64 b = bb[p];
            while (b) k ^= zPiece[p][poplsb(b)];
        }
        k ^= zCastle[castle];
        if (ep != -1) k ^= zEp[fileOf(ep)];
        if (side == BLACK) k ^= zSide;
        return k;
    }
};

// NNUE feature: for perspective c, index = (ownness*6 + pieceType)*64 + orientedSq
static inline int featIdx(int persp, int pc, int sq) {
    int pt = pc % 6, col = pc / 6;
    return ((col != persp) * 6 + pt) * 64 + (persp == WHITE ? sq : (sq ^ 56));
}
#ifdef __AVX2__
#include <immintrin.h>
#endif
static inline void nnAddFeature(Accumulator &acc, int pc, int sq) {
    const int16_t *w0 = nnW1[featIdx(WHITE, pc, sq)];
    const int16_t *w1 = nnW1[featIdx(BLACK, pc, sq)];
    for (int i = 0; i < nnHidden; i++) acc.v[WHITE][i] += w0[i];
    for (int i = 0; i < nnHidden; i++) acc.v[BLACK][i] += w1[i];
}
void nnRefresh(Accumulator &acc, const Position &p) {
    for (int c = 0; c < 2; c++)
        for (int i = 0; i < nnHidden; i++) acc.v[c][i] = nnB1[i];
    for (int pc = 0; pc < 12; pc++) {
        U64 b = p.bb[pc];
        while (b) nnAddFeature(acc, pc, poplsb(b));
    }
}
int nnEval(const Position &p, const Accumulator &acc) {
    int stm = p.side;
    int bk = nnBuckets > 1
        ? min((popcnt(p.occ[BOTH]) - 2) >> 2, nnBuckets - 1) : 0;
    int64_t sum = nnB2[bk];
    const int16_t *w2 = nnW2[bk];
    const int16_t *a = acc.v[stm], *b = acc.v[stm ^ 1];
#ifdef __AVX2__
    __m256i accv = _mm256_setzero_si256();
    const __m256i zero = _mm256_setzero_si256();
    const __m256i cap = _mm256_set1_epi16(255);
    for (int i = 0; i < nnHidden; i += 16) {
        __m256i va = _mm256_min_epi16(_mm256_max_epi16(
            _mm256_load_si256((const __m256i *)&a[i]), zero), cap);
        __m256i vb = _mm256_min_epi16(_mm256_max_epi16(
            _mm256_load_si256((const __m256i *)&b[i]), zero), cap);
        accv = _mm256_add_epi32(accv, _mm256_madd_epi16(va,
            _mm256_loadu_si256((const __m256i *)&w2[i])));
        accv = _mm256_add_epi32(accv, _mm256_madd_epi16(vb,
            _mm256_loadu_si256((const __m256i *)&w2[nnHidden + i])));
    }
    __m128i lo = _mm256_castsi256_si128(accv);
    __m128i hi = _mm256_extracti128_si256(accv, 1);
    __m128i s4 = _mm_add_epi32(lo, hi);
    s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
    s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
    sum += _mm_cvtsi128_si32(s4);
#else
    for (int i = 0; i < nnHidden; i++) {
        int v = a[i]; v = v < 0 ? 0 : (v > 255 ? 255 : v);
        sum += v * w2[i];
    }
    for (int i = 0; i < nnHidden; i++) {
        int v = b[i]; v = v < 0 ? 0 : (v > 255 ? 255 : v);
        sum += v * w2[nnHidden + i];
    }
#endif
    int cp = (int)(sum * 400 / (255 * 64));
    if (cp > 8000) cp = 8000;
    if (cp < -8000) cp = -8000;
    return cp;
}
bool nnLoad(const string &path) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[6];
    if (fread(magic, 1, 6, f) != 6) { fclose(f); return false; }
    int h, nb = 1;
    // net-file magic strings keep the original "SABLE" tag so every
    // previously trained .nnue stays loadable after the Onyx rename
    if (!memcmp(magic, "SABLE1", 6)) h = 256;
    else if (!memcmp(magic, "SABLE2", 6)) h = 512;
    else if (!memcmp(magic, "SABLE3", 6)) h = 768;
    else if (!memcmp(magic, "SABLE4", 6)) { h = 768; nb = 8; }  // 8 output buckets
    else if (!memcmp(magic, "SABLE5", 6)) h = 1024;
    else { fclose(f); return false; }
    if (h > NN_MAX_HIDDEN) {   // net larger than this build's static arrays
        fclose(f);
        printf("info string net needs -DNN_MAX_HIDDEN=%d build\n", h);
        return false;
    }
    bool ok = true;
    for (int ft = 0; ft < 768 && ok; ft++)
        ok = fread(nnW1[ft], sizeof(int16_t), h, f) == (size_t)h;
    ok = ok && fread(nnB1, sizeof(int16_t), h, f) == (size_t)h;
    for (int b = 0; b < nb && ok; b++)
        ok = fread(nnW2[b], sizeof(int16_t), 2 * h, f) == 2 * (size_t)h;
    ok = ok && fread(nnB2, sizeof(int32_t), nb, f) == (size_t)nb;
    fclose(f);
    if (ok) { nnHidden = h; nnBuckets = nb; nnueLoaded = true; }
    return ok;
}

bool sqAttacked(const Position &p, int sq, int bySide) {
    if (pawnAtt[bySide ^ 1][sq] & p.bb[bySide == WHITE ? WP : BP]) return true;
    if (knightAtt[sq] & p.bb[bySide == WHITE ? WN : BN]) return true;
    if (kingAtt[sq] & p.bb[bySide == WHITE ? WK : BK]) return true;
    U64 bq = p.bb[bySide == WHITE ? WB : BB_] | p.bb[bySide == WHITE ? WQ : BQ];
    if (bishopAttacks(sq, p.occ[BOTH]) & bq) return true;
    U64 rq = p.bb[bySide == WHITE ? WR : BR] | p.bb[bySide == WHITE ? WQ : BQ];
    if (rookAttacks(sq, p.occ[BOTH]) & rq) return true;
    return false;
}
static inline bool inCheck(const Position &p) {
    return sqAttacked(p, lsb(p.bb[p.side == WHITE ? WK : BK]), p.side ^ 1);
}

U64 attackersTo(const Position &p, int sq, U64 occ) {
    return (pawnAtt[BLACK][sq] & p.bb[WP])
         | (pawnAtt[WHITE][sq] & p.bb[BP])
         | (knightAtt[sq] & (p.bb[WN] | p.bb[BN]))
         | (kingAtt[sq] & (p.bb[WK] | p.bb[BK]))
         | (bishopAttacks(sq, occ) & (p.bb[WB] | p.bb[BB_] | p.bb[WQ] | p.bb[BQ]))
         | (rookAttacks(sq, occ) & (p.bb[WR] | p.bb[BR] | p.bb[WQ] | p.bb[BQ]));
}

// ----------------------- moves -----------------------------
// move encoding: from(0-5) to(6-11) piece(12-15) promo(16-19)
// flags: 20 capture, 21 doublepush, 22 ep, 23 castle
#define M_FROM(m)   ((m) & 63)
#define M_TO(m)     (((m) >> 6) & 63)
#define M_PIECE(m)  (((m) >> 12) & 15)
#define M_PROMO(m)  (((m) >> 16) & 15)
#define M_CAP(m)    ((m) & (1 << 20))
#define M_DP(m)     ((m) & (1 << 21))
#define M_EP(m)     ((m) & (1 << 22))
#define M_CASTLE(m) ((m) & (1 << 23))

static inline int makeMoveInt(int from, int to, int pc, int promo, int cap, int dp, int ep, int cas) {
    return from | (to << 6) | (pc << 12) | (promo << 16) | (cap << 20) | (dp << 21) | (ep << 22) | (cas << 23);
}

// fused accumulator update: dst = src + adds - subs in one pass.
// p is the position BEFORE the move (side = mover, board intact).
// compact record of one move's accumulator update (feature index deltas)
struct AccDelta {
    int addF[2][2], subF[2][2];   // [perspective][slot]
    int na, ns;
};

static inline void nnComputeDelta(const Position &p, int m, AccDelta &d) {
    int from = M_FROM(m), to = M_TO(m), pc = M_PIECE(m);
    int addPc = M_PROMO(m) ? M_PROMO(m) : pc;
    d.na = 1; d.ns = 1;
    d.addF[WHITE][0] = featIdx(WHITE, addPc, to);
    d.addF[BLACK][0] = featIdx(BLACK, addPc, to);
    d.subF[WHITE][0] = featIdx(WHITE, pc, from);
    d.subF[BLACK][0] = featIdx(BLACK, pc, from);
    if (M_CAP(m)) {
        int capSq = M_EP(m) ? to + (p.side == WHITE ? -8 : 8) : to;
        int victim = p.board[capSq];
        d.subF[WHITE][1] = featIdx(WHITE, victim, capSq);
        d.subF[BLACK][1] = featIdx(BLACK, victim, capSq);
        d.ns = 2;
    } else if (M_CASTLE(m)) {
        int rk = p.side == WHITE ? WR : BR;
        int rf, rt;
        if (to == SQ(6, 0)) { rf = SQ(7, 0); rt = SQ(5, 0); }
        else if (to == SQ(2, 0)) { rf = SQ(0, 0); rt = SQ(3, 0); }
        else if (to == SQ(6, 7)) { rf = SQ(7, 7); rt = SQ(5, 7); }
        else { rf = SQ(0, 7); rt = SQ(3, 7); }
        d.addF[WHITE][1] = featIdx(WHITE, rk, rt);
        d.addF[BLACK][1] = featIdx(BLACK, rk, rt);
        d.subF[WHITE][1] = featIdx(WHITE, rk, rf);
        d.subF[BLACK][1] = featIdx(BLACK, rk, rf);
        d.na = d.ns = 2;
    }
}

void nnApplyDelta(const AccDelta &dd, const Accumulator &src, Accumulator &dst) {
    const auto &addF = dd.addF; const auto &subF = dd.subF;
    const int na = dd.na, ns = dd.ns;
    for (int c = 0; c < 2; c++) {
        const int16_t *a0 = nnW1[addF[c][0]], *s0 = nnW1[subF[c][0]];
        const int16_t *sp = src.v[c];
        int16_t *dp = dst.v[c];
#ifdef __AVX2__
        if (na == 1 && ns == 1) {
            for (int i = 0; i < nnHidden; i += 16) {
                __m256i v = _mm256_load_si256((const __m256i *)&sp[i]);
                v = _mm256_add_epi16(v, _mm256_load_si256((const __m256i *)&a0[i]));
                v = _mm256_sub_epi16(v, _mm256_load_si256((const __m256i *)&s0[i]));
                _mm256_store_si256((__m256i *)&dp[i], v);
            }
        } else if (na == 1) {
            const int16_t *s1 = nnW1[subF[c][1]];
            for (int i = 0; i < nnHidden; i += 16) {
                __m256i v = _mm256_load_si256((const __m256i *)&sp[i]);
                v = _mm256_add_epi16(v, _mm256_load_si256((const __m256i *)&a0[i]));
                v = _mm256_sub_epi16(v, _mm256_load_si256((const __m256i *)&s0[i]));
                v = _mm256_sub_epi16(v, _mm256_load_si256((const __m256i *)&s1[i]));
                _mm256_store_si256((__m256i *)&dp[i], v);
            }
        } else {
            const int16_t *a1 = nnW1[addF[c][1]], *s1 = nnW1[subF[c][1]];
            for (int i = 0; i < nnHidden; i += 16) {
                __m256i v = _mm256_load_si256((const __m256i *)&sp[i]);
                v = _mm256_add_epi16(v, _mm256_load_si256((const __m256i *)&a0[i]));
                v = _mm256_add_epi16(v, _mm256_load_si256((const __m256i *)&a1[i]));
                v = _mm256_sub_epi16(v, _mm256_load_si256((const __m256i *)&s0[i]));
                v = _mm256_sub_epi16(v, _mm256_load_si256((const __m256i *)&s1[i]));
                _mm256_store_si256((__m256i *)&dp[i], v);
            }
        }
#else
        if (na == 1 && ns == 1) {
            for (int i = 0; i < nnHidden; i++) dp[i] = sp[i] + a0[i] - s0[i];
        } else if (na == 1) {
            const int16_t *s1 = nnW1[subF[c][1]];
            for (int i = 0; i < nnHidden; i++) dp[i] = sp[i] + a0[i] - s0[i] - s1[i];
        } else {
            const int16_t *a1 = nnW1[addF[c][1]], *s1 = nnW1[subF[c][1]];
            for (int i = 0; i < nnHidden; i++) dp[i] = sp[i] + a0[i] + a1[i] - s0[i] - s1[i];
        }
#endif
    }
}

void nnApply(const Position &p, int m, const Accumulator &src, Accumulator &dst) {
    AccDelta d;
    nnComputeDelta(p, m, d);
    nnApplyDelta(d, src, dst);
}

struct MoveList {
    int moves[256];
    int scores[256];
    int count = 0;
    void add(int m) { moves[count++] = m; }
};

// castle rights mask per square
int castleMask[64];
void initCastleMask() {
    for (int i = 0; i < 64; i++) castleMask[i] = 15;
    castleMask[SQ(4, 0)] &= ~3;   // e1
    castleMask[SQ(0, 0)] &= ~2;   // a1
    castleMask[SQ(7, 0)] &= ~1;   // h1
    castleMask[SQ(4, 7)] &= ~12;  // e8
    castleMask[SQ(0, 7)] &= ~8;   // a8
    castleMask[SQ(7, 7)] &= ~4;   // h8
}

void genMoves(const Position &p, MoveList &ml, bool capsOnly) {
    int us = p.side, them = us ^ 1;
    U64 own = p.occ[us], opp = p.occ[them], all = p.occ[BOTH];

    // pawns
    int pp = us == WHITE ? WP : BP;
    U64 pawns = p.bb[pp];
    int up = us == WHITE ? 8 : -8;
    U64 promoRank = us == WHITE ? RANK_8 : RANK_1;
    U64 startRank = us == WHITE ? RANK_2 : RANK_7;
    int q = us == WHITE ? WQ : BQ, r = us == WHITE ? WR : BR,
        b_ = us == WHITE ? WB : BB_, n = us == WHITE ? WN : BN;

    U64 pw = pawns;
    while (pw) {
        int from = poplsb(pw);
        // captures
        U64 caps = pawnAtt[us][from] & opp;
        while (caps) {
            int to = poplsb(caps);
            if ((1ULL << to) & promoRank) {
                ml.add(makeMoveInt(from, to, pp, q, 1, 0, 0, 0));
                ml.add(makeMoveInt(from, to, pp, r, 1, 0, 0, 0));
                ml.add(makeMoveInt(from, to, pp, b_, 1, 0, 0, 0));
                ml.add(makeMoveInt(from, to, pp, n, 1, 0, 0, 0));
            } else ml.add(makeMoveInt(from, to, pp, 0, 1, 0, 0, 0));
        }
        // en passant
        if (p.ep != -1 && (pawnAtt[us][from] & (1ULL << p.ep)))
            ml.add(makeMoveInt(from, p.ep, pp, 0, 1, 0, 1, 0));
        // pushes
        int to = from + up;
        if (!((1ULL << to) & all)) {
            if ((1ULL << to) & promoRank) {
                ml.add(makeMoveInt(from, to, pp, q, 0, 0, 0, 0));
                if (!capsOnly) {
                    ml.add(makeMoveInt(from, to, pp, r, 0, 0, 0, 0));
                    ml.add(makeMoveInt(from, to, pp, b_, 0, 0, 0, 0));
                    ml.add(makeMoveInt(from, to, pp, n, 0, 0, 0, 0));
                }
            } else if (!capsOnly) {
                ml.add(makeMoveInt(from, to, pp, 0, 0, 0, 0, 0));
                if ((1ULL << from) & startRank) {
                    int to2 = to + up;
                    if (!((1ULL << to2) & all))
                        ml.add(makeMoveInt(from, to2, pp, 0, 0, 1, 0, 0));
                }
            }
        }
    }

    // knights
    U64 ns = p.bb[n];
    while (ns) {
        int from = poplsb(ns);
        U64 att = knightAtt[from] & ~own;
        if (capsOnly) att &= opp;
        while (att) {
            int to = poplsb(att);
            ml.add(makeMoveInt(from, to, n, 0, (opp >> to) & 1, 0, 0, 0));
        }
    }
    // bishops
    U64 bs = p.bb[b_];
    while (bs) {
        int from = poplsb(bs);
        U64 att = bishopAttacks(from, all) & ~own;
        if (capsOnly) att &= opp;
        while (att) {
            int to = poplsb(att);
            ml.add(makeMoveInt(from, to, b_, 0, (opp >> to) & 1, 0, 0, 0));
        }
    }
    // rooks
    U64 rs = p.bb[r];
    while (rs) {
        int from = poplsb(rs);
        U64 att = rookAttacks(from, all) & ~own;
        if (capsOnly) att &= opp;
        while (att) {
            int to = poplsb(att);
            ml.add(makeMoveInt(from, to, r, 0, (opp >> to) & 1, 0, 0, 0));
        }
    }
    // queens
    U64 qs = p.bb[q];
    while (qs) {
        int from = poplsb(qs);
        U64 att = queenAttacks(from, all) & ~own;
        if (capsOnly) att &= opp;
        while (att) {
            int to = poplsb(att);
            ml.add(makeMoveInt(from, to, q, 0, (opp >> to) & 1, 0, 0, 0));
        }
    }
    // king
    int k = us == WHITE ? WK : BK;
    int from = lsb(p.bb[k]);
    U64 att = kingAtt[from] & ~own;
    if (capsOnly) att &= opp;
    while (att) {
        int to = poplsb(att);
        ml.add(makeMoveInt(from, to, k, 0, (opp >> to) & 1, 0, 0, 0));
    }
    // castling
    if (!capsOnly) {
        if (us == WHITE) {
            if ((p.castle & 1) && !(all & 0x60ULL)
                && !sqAttacked(p, SQ(4, 0), BLACK) && !sqAttacked(p, SQ(5, 0), BLACK))
                ml.add(makeMoveInt(SQ(4, 0), SQ(6, 0), WK, 0, 0, 0, 0, 1));
            if ((p.castle & 2) && !(all & 0xEULL)
                && !sqAttacked(p, SQ(4, 0), BLACK) && !sqAttacked(p, SQ(3, 0), BLACK))
                ml.add(makeMoveInt(SQ(4, 0), SQ(2, 0), WK, 0, 0, 0, 0, 1));
        } else {
            if ((p.castle & 4) && !(all & 0x6000000000000000ULL)
                && !sqAttacked(p, SQ(4, 7), WHITE) && !sqAttacked(p, SQ(5, 7), WHITE))
                ml.add(makeMoveInt(SQ(4, 7), SQ(6, 7), BK, 0, 0, 0, 0, 1));
            if ((p.castle & 8) && !(all & 0x0E00000000000000ULL)
                && !sqAttacked(p, SQ(4, 7), WHITE) && !sqAttacked(p, SQ(3, 7), WHITE))
                ml.add(makeMoveInt(SQ(4, 7), SQ(2, 7), BK, 0, 0, 0, 0, 1));
        }
    }
}

// copy-make; returns false if move leaves own king in check
bool makeMove(Position &p, int m) {
    int from = M_FROM(m), to = M_TO(m), pc = M_PIECE(m), promo = M_PROMO(m);
    int us = p.side, them = us ^ 1;

    if (p.ep != -1) { p.key ^= zEp[fileOf(p.ep)]; p.ep = -1; }

    p.fifty++;
    if (pieceType(pc) == PAWN) p.fifty = 0;

    if (M_CAP(m)) {
        p.fifty = 0;
        int capSq = to;
        if (M_EP(m)) capSq = to + (us == WHITE ? -8 : 8);
        int cp = p.board[capSq];
        p.bb[cp] ^= 1ULL << capSq;
        p.occ[them] ^= 1ULL << capSq;
        p.key ^= zPiece[cp][capSq];
        if (pieceType(cp) == PAWN) p.pawnKey ^= zPiece[cp][capSq];
#if FEAT_NPCORR
        else p.npKey[them] ^= zPiece[cp][capSq];
#endif
        p.board[capSq] = NO_PIECE;
    }

    U64 fromTo = (1ULL << from) | (1ULL << to);
    p.bb[pc] ^= fromTo;
    p.occ[us] ^= fromTo;
    p.key ^= zPiece[pc][from] ^ zPiece[pc][to];
    if (pieceType(pc) == PAWN) {
        p.pawnKey ^= zPiece[pc][from];
        if (!promo) p.pawnKey ^= zPiece[pc][to];
    }
#if FEAT_NPCORR
    else p.npKey[us] ^= zPiece[pc][from] ^ zPiece[pc][to];
#endif
    p.board[from] = NO_PIECE;
    p.board[to] = (uint8_t)pc;

    if (promo) {
        p.bb[pc] ^= 1ULL << to;
        p.bb[promo] |= 1ULL << to;
        p.key ^= zPiece[pc][to] ^ zPiece[promo][to];
#if FEAT_NPCORR
        p.npKey[us] ^= zPiece[promo][to];
#endif
        p.board[to] = (uint8_t)promo;
    }
    if (M_CASTLE(m)) {
        int rk = us == WHITE ? WR : BR;
        int rf, rt;
        if (to == SQ(6, 0)) { rf = SQ(7, 0); rt = SQ(5, 0); }
        else if (to == SQ(2, 0)) { rf = SQ(0, 0); rt = SQ(3, 0); }
        else if (to == SQ(6, 7)) { rf = SQ(7, 7); rt = SQ(5, 7); }
        else { rf = SQ(0, 7); rt = SQ(3, 7); }
        U64 rftBB = (1ULL << rf) | (1ULL << rt);
        p.bb[rk] ^= rftBB;
        p.occ[us] ^= rftBB;
        p.key ^= zPiece[rk][rf] ^ zPiece[rk][rt];
#if FEAT_NPCORR
        p.npKey[us] ^= zPiece[rk][rf] ^ zPiece[rk][rt];
#endif
        p.board[rf] = NO_PIECE;
        p.board[rt] = (uint8_t)rk;
    }
    p.key ^= zCastle[p.castle];
    p.castle &= castleMask[from] & castleMask[to];
    p.key ^= zCastle[p.castle];

    if (M_DP(m)) {
        p.ep = to + (us == WHITE ? -8 : 8);
        p.key ^= zEp[fileOf(p.ep)];
    }
    p.side ^= 1;
    p.key ^= zSide;
    p.occ[BOTH] = p.occ[WHITE] | p.occ[BLACK];

    return !sqAttacked(p, lsb(p.bb[us == WHITE ? WK : BK]), them);
}

void makeNull(Position &p) {
    if (p.ep != -1) { p.key ^= zEp[fileOf(p.ep)]; p.ep = -1; }
    p.side ^= 1;
    p.key ^= zSide;
    p.fifty++;
}

// static exchange evaluation (swap algorithm)
const int seeVal[6] = {100, 320, 330, 500, 950, 20000};
int see(const Position &p, int m) {
    int from = M_FROM(m), to = M_TO(m);
    int gain[32], d = 0;
    U64 occ = p.occ[BOTH];
    int attacker = pieceType(M_PIECE(m));
    if (M_EP(m)) {
        occ ^= 1ULL << (to + (p.side == WHITE ? -8 : 8));
        gain[0] = seeVal[PAWN];
    } else {
        int vp = p.pieceOn(to);
        gain[0] = vp == NO_PIECE ? 0 : seeVal[pieceType(vp)];
    }
    int stm = p.side;
    U64 fromSet = 1ULL << from;
    U64 attadef = attackersTo(p, to, occ);
    U64 diag = p.bb[WB] | p.bb[BB_] | p.bb[WQ] | p.bb[BQ];
    U64 orth = p.bb[WR] | p.bb[BR] | p.bb[WQ] | p.bb[BQ];
    do {
        d++;
        gain[d] = seeVal[attacker] - gain[d - 1];
        if (max(-gain[d - 1], gain[d]) < 0) break;
        attadef ^= fromSet;
        occ ^= fromSet;
        attadef |= ((bishopAttacks(to, occ) & diag) | (rookAttacks(to, occ) & orth)) & occ;
        stm ^= 1;
        fromSet = 0;
        int base = stm == WHITE ? WP : BP;
        for (int pt = 0; pt < 6; pt++) {
            U64 b = p.bb[base + pt] & attadef;
            if (b) { fromSet = b & -b; attacker = pt; break; }
        }
    } while (fromSet);
    while (--d) gain[d - 1] = -max(-gain[d - 1], gain[d]);
    return gain[0];
}

// does this move give a direct check? (discovered checks ignored - rare)
bool givesDirectCheck(const Position &p, int m) {
    int to = M_TO(m);
    int pc = M_PROMO(m) ? M_PROMO(m) : M_PIECE(m);
    int them = p.side ^ 1;
    int ksq = lsb(p.bb[them == WHITE ? WK : BK]);
    U64 kb = 1ULL << ksq;
    U64 occ = (p.occ[BOTH] ^ (1ULL << M_FROM(m))) | (1ULL << to);
    switch (pieceType(pc)) {
        case PAWN:   return (pawnAtt[p.side][to] & kb) != 0;
        case KNIGHT: return (knightAtt[to] & kb) != 0;
        case BISHOP: return (bishopAttacks(to, occ) & kb) != 0;
        case ROOK:   return (rookAttacks(to, occ) & kb) != 0;
        case QUEEN:  return (queenAttacks(to, occ) & kb) != 0;
        default:     return false;
    }
}

// ----------------------- FEN -------------------------------
void setFen(Position &p, const string &fen) {
    memset(p.bb, 0, sizeof(p.bb));
    p.castle = 0; p.ep = -1; p.fifty = 0; p.side = WHITE;
    istringstream ss(fen);
    string board, stm, cast, eps;
    ss >> board >> stm >> cast >> eps;
    int f = 0, r = 7;
    for (char c : board) {
        if (c == '/') { f = 0; r--; }
        else if (isdigit(c)) f += c - '0';
        else {
            int pc = -1;
            switch (c) {
                case 'P': pc = WP; break; case 'N': pc = WN; break;
                case 'B': pc = WB; break; case 'R': pc = WR; break;
                case 'Q': pc = WQ; break; case 'K': pc = WK; break;
                case 'p': pc = BP; break; case 'n': pc = BN; break;
                case 'b': pc = BB_; break; case 'r': pc = BR; break;
                case 'q': pc = BQ; break; case 'k': pc = BK; break;
            }
            if (pc >= 0) p.bb[pc] |= 1ULL << SQ(f, r);
            f++;
        }
    }
    p.side = (stm == "b") ? BLACK : WHITE;
    if (cast.find('K') != string::npos) p.castle |= 1;
    if (cast.find('Q') != string::npos) p.castle |= 2;
    if (cast.find('k') != string::npos) p.castle |= 4;
    if (cast.find('q') != string::npos) p.castle |= 8;
    if (eps.size() == 2) p.ep = SQ(eps[0] - 'a', eps[1] - '1');
    string fiftyStr;
    if (ss >> fiftyStr) p.fifty = atoi(fiftyStr.c_str());
    p.updateOcc();
    p.fillBoard();
    p.key = p.computeKey();
    p.pawnKey = 0;
    for (int pc = WP; pc <= BP; pc += 6) {
        U64 b = p.bb[pc];
        while (b) { int sq = poplsb(b); p.pawnKey ^= zPiece[pc][sq]; }
    }
#if FEAT_NPCORR
    p.npKey[WHITE] = p.npKey[BLACK] = 0;
    for (int pc = 0; pc < 12; pc++) {
        if (pieceType(pc) == PAWN) continue;
        U64 b = p.bb[pc];
        while (b) { int sq = poplsb(b); p.npKey[pc < 6 ? WHITE : BLACK] ^= zPiece[pc][sq]; }
    }
#endif
}

string moveToUci(int m) {
    string s;
    s += 'a' + fileOf(M_FROM(m));
    s += '1' + rankOf(M_FROM(m));
    s += 'a' + fileOf(M_TO(m));
    s += '1' + rankOf(M_TO(m));
    if (M_PROMO(m)) {
        const char *pr = "pnbrqk";
        s += pr[pieceType(M_PROMO(m))];
    }
    return s;
}

// ----------------------- evaluation (PeSTO) ----------------
// tables written as seen from white's side, a8 first (flip with ^56 for white)
const int mgValue[6] = {82, 337, 365, 477, 1025, 0};
const int egValue[6] = {94, 281, 297, 512, 936, 0};
const int phaseInc[6] = {0, 1, 1, 2, 4, 0};

const int mgPawn[64] = {
      0,   0,   0,   0,   0,   0,  0,   0,
     98, 134,  61,  95,  68, 126, 34, -11,
     -6,   7,  26,  31,  65,  56, 25, -20,
    -14,  13,   6,  21,  23,  12, 17, -23,
    -27,  -2,  -5,  12,  17,   6, 10, -25,
    -26,  -4,  -4, -10,   3,   3, 33, -12,
    -35,  -1, -20, -23, -15,  24, 38, -22,
      0,   0,   0,   0,   0,   0,  0,   0,
};
const int egPawn[64] = {
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
};
const int mgKnight[64] = {
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
};
const int egKnight[64] = {
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
};
const int mgBishop[64] = {
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
};
const int egBishop[64] = {
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
};
const int mgRook[64] = {
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26,
};
const int egRook[64] = {
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
};
const int mgQueen[64] = {
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
};
const int egQueen[64] = {
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
};
const int mgKing[64] = {
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
};
const int egKing[64] = {
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
};

const int *mgTable[6] = {mgPawn, mgKnight, mgBishop, mgRook, mgQueen, mgKing};
const int *egTable[6] = {egPawn, egKnight, egBishop, egRook, egQueen, egKing};

int mgPST[12][64], egPST[12][64];
U64 passedMask[2][64], isoMask[8], fileBB[8];

void initEval() {
    for (int pt = 0; pt < 6; pt++)
        for (int sq = 0; sq < 64; sq++) {
            mgPST[pt][sq]     = mgValue[pt] + mgTable[pt][sq ^ 56]; // white
            egPST[pt][sq]     = egValue[pt] + egTable[pt][sq ^ 56];
            mgPST[pt + 6][sq] = mgValue[pt] + mgTable[pt][sq];      // black
            egPST[pt + 6][sq] = egValue[pt] + egTable[pt][sq];
        }
    for (int f = 0; f < 8; f++) {
        fileBB[f] = FILE_A << f;
        isoMask[f] = 0;
        if (f > 0) isoMask[f] |= fileBB[f - 1];
        if (f < 7) isoMask[f] |= fileBB[f + 1];
    }
    for (int sq = 0; sq < 64; sq++) {
        int f = fileOf(sq), r = rankOf(sq);
        U64 fw = 0, bw = 0;
        for (int rr = r + 1; rr <= 7; rr++)
            for (int ff = max(0, f - 1); ff <= min(7, f + 1); ff++)
                fw |= 1ULL << SQ(ff, rr);
        for (int rr = r - 1; rr >= 0; rr--)
            for (int ff = max(0, f - 1); ff <= min(7, f + 1); ff++)
                bw |= 1ULL << SQ(ff, rr);
        passedMask[WHITE][sq] = fw;
        passedMask[BLACK][sq] = bw;
    }
}

const int passedBonusMg[8] = {0, 5, 10, 15, 25, 45, 90, 0};
const int passedBonusEg[8] = {0, 15, 20, 30, 50, 90, 140, 0};

// The historical FEAT_FIFTY blocks inside evaluate() appeared before their
// old default definition and therefore compiled out.  Keep raw evaluation
// independent of the un-hashed halfmove clock; search-level damping is gated
// separately below so TT eval fields remain reusable.
#ifndef FEAT_FIFTY
#define FEAT_FIFTY 0
#endif

int evaluate(const Position &p, const Accumulator *acc) {
    if (nnueLoaded) {
        int s = nnEval(p, *acc);
#if FEAT_FIFTY
        s = s * (200 - min(p.fifty, 100)) / 200;   // fade eval toward the 50-move draw
#endif
        return s;
    }
    int mg = 0, eg = 0, phase = 0;
    for (int pc = 0; pc < 12; pc++) {
        U64 b = p.bb[pc];
        int sign = pc < 6 ? 1 : -1;
        while (b) {
            int sq = poplsb(b);
            mg += sign * mgPST[pc][sq];
            eg += sign * egPST[pc][sq];
            phase += phaseInc[pieceType(pc)];
        }
    }
    // bishop pair
    if (popcnt(p.bb[WB]) >= 2) { mg += 25; eg += 45; }
    if (popcnt(p.bb[BB_]) >= 2) { mg -= 25; eg -= 45; }

    // pawn structure
    U64 wp = p.bb[WP], bp = p.bb[BP];
    U64 b = wp;
    while (b) {
        int sq = poplsb(b);
        int f = fileOf(sq);
        if (!(passedMask[WHITE][sq] & bp)) {
            mg += passedBonusMg[rankOf(sq)];
            eg += passedBonusEg[rankOf(sq)];
        }
        if (!(isoMask[f] & wp)) { mg -= 12; eg -= 15; }
        if (popcnt(fileBB[f] & wp) > 1) { mg -= 8; eg -= 12; }
    }
    b = bp;
    while (b) {
        int sq = poplsb(b);
        int f = fileOf(sq);
        if (!(passedMask[BLACK][sq] & wp)) {
            mg -= passedBonusMg[7 - rankOf(sq)];
            eg -= passedBonusEg[7 - rankOf(sq)];
        }
        if (!(isoMask[f] & bp)) { mg += 12; eg += 15; }
        if (popcnt(fileBB[f] & bp) > 1) { mg += 8; eg += 12; }
    }

    // rook on open / semi-open file
    b = p.bb[WR];
    while (b) {
        int f = fileOf(poplsb(b));
        if (!(fileBB[f] & wp)) { mg += (fileBB[f] & bp) ? 12 : 25; eg += 8; }
    }
    b = p.bb[BR];
    while (b) {
        int f = fileOf(poplsb(b));
        if (!(fileBB[f] & bp)) { mg -= (fileBB[f] & wp) ? 12 : 25; eg -= 8; }
    }

    // light mobility for knights/bishops
    U64 all = p.occ[BOTH];
    b = p.bb[WN];
    while (b) mg += 3 * popcnt(knightAtt[poplsb(b)] & ~p.occ[WHITE]) - 12;
    b = p.bb[BN];
    while (b) mg -= 3 * popcnt(knightAtt[poplsb(b)] & ~p.occ[BLACK]) - 12;
    b = p.bb[WB];
    while (b) mg += 3 * popcnt(bishopAttacks(poplsb(b), all) & ~p.occ[WHITE]) - 18;
    b = p.bb[BB_];
    while (b) mg -= 3 * popcnt(bishopAttacks(poplsb(b), all) & ~p.occ[BLACK]) - 18;

    // ---- king safety: attack units on the enemy king zone ----
    static const int safetyTable[100] = {
          0,   0,   1,   2,   3,   5,   7,   9,  12,  15,
         18,  22,  26,  30,  35,  39,  44,  50,  56,  62,
         68,  75,  82,  85,  89,  97, 105, 113, 122, 131,
        140, 150, 169, 180, 191, 202, 213, 225, 237, 248,
        260, 272, 283, 295, 307, 319, 330, 342, 354, 366,
        377, 389, 401, 412, 424, 436, 448, 459, 471, 483,
        494, 500, 500, 500, 500, 500, 500, 500, 500, 500,
        500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
        500, 500, 500, 500, 500, 500, 500, 500, 500, 500,
        500, 500, 500, 500, 500, 500, 500, 500, 500, 500};
    int wk = lsb(p.bb[WK]), bk = lsb(p.bb[BK]);
    U64 bZone = kingAtt[bk] | (1ULL << bk);
    U64 wZone = kingAtt[wk] | (1ULL << wk);
    {
        int units = 0, attackers = 0;
        U64 t = p.bb[WN];
        while (t) { U64 a = knightAtt[poplsb(t)] & bZone; if (a) { attackers++; units += 2 * popcnt(a); } }
        t = p.bb[WB];
        while (t) { U64 a = bishopAttacks(poplsb(t), all) & bZone; if (a) { attackers++; units += 2 * popcnt(a); } }
        t = p.bb[WR];
        while (t) { U64 a = rookAttacks(poplsb(t), all) & bZone; if (a) { attackers++; units += 3 * popcnt(a); } }
        t = p.bb[WQ];
        while (t) { U64 a = queenAttacks(poplsb(t), all) & bZone; if (a) { attackers++; units += 5 * popcnt(a); } }
        if (attackers >= 2) mg += safetyTable[min(units, 99)];
    }
    {
        int units = 0, attackers = 0;
        U64 t = p.bb[BN];
        while (t) { U64 a = knightAtt[poplsb(t)] & wZone; if (a) { attackers++; units += 2 * popcnt(a); } }
        t = p.bb[BB_];
        while (t) { U64 a = bishopAttacks(poplsb(t), all) & wZone; if (a) { attackers++; units += 2 * popcnt(a); } }
        t = p.bb[BR];
        while (t) { U64 a = rookAttacks(poplsb(t), all) & wZone; if (a) { attackers++; units += 3 * popcnt(a); } }
        t = p.bb[BQ];
        while (t) { U64 a = queenAttacks(poplsb(t), all) & wZone; if (a) { attackers++; units += 5 * popcnt(a); } }
        if (attackers >= 2) mg -= safetyTable[min(units, 99)];
    }
    // ---- pawn shield ----
    if (rankOf(wk) <= 1) {
        int kf = fileOf(wk);
        for (int f = max(0, kf - 1); f <= min(7, kf + 1); f++) {
            U64 fp = fileBB[f] & wp;
            if (fp & (RANK_2 << 0))      mg += 12;
            else if (fp & (RANK_2 << 8)) mg += 6;
            else if (!fp)                mg -= 14;
        }
    }
    if (rankOf(bk) >= 6) {
        int kf = fileOf(bk);
        for (int f = max(0, kf - 1); f <= min(7, kf + 1); f++) {
            U64 fp = fileBB[f] & bp;
            if (fp & RANK_7)             mg -= 12;
            else if (fp & (RANK_7 >> 8)) mg -= 6;
            else if (!fp)                mg += 14;
        }
    }

    if (phase > 24) phase = 24;
    int score = (mg * phase + eg * (24 - phase)) / 24;
    score += p.side == WHITE ? 14 : -14; // tempo
    int s = p.side == WHITE ? score : -score;
#if FEAT_FIFTY
    s = s * (200 - min(p.fifty, 100)) / 200;
#endif
    return s;
}

// ---- round 2 features (defined here: the TT code below needs them) ----
#ifndef FEAT_TTBUCKET
#define FEAT_TTBUCKET 0   // measured -13 elo: probe overhead without replacement payoff
#endif
#ifndef FEAT_NMPVERIFY
#define FEAT_NMPVERIFY 0  // prior test was neutral; must pass a fresh gate before enabling
#endif
#ifndef FEAT_NMPINPLACE
#define FEAT_NMPINPLACE 0 // speed-only: avoid copying Position for null moves
#endif
#ifndef FEAT_MATCORR
#define FEAT_MATCORR 0    // 4,000-game retest: -0.61 +/- 8.22 elo
#endif
#ifndef FEAT_PROBCUT
#define FEAT_PROBCUT 0    // rejected: PROBCUT-01 +1.39 +/- 8.36 at the 4,000-game cap
#endif

// ----------------------- transposition table ---------------
enum { TT_NONE, TT_EXACT, TT_ALPHA, TT_BETA };
#ifndef FEAT_TTPACK
#define FEAT_TTPACK 0  // 16-byte entries via 32-bit key verification: 2x capacity
#endif
struct TTEntry {
#if FEAT_TTPACK
    uint32_t key;   // top 32 zobrist bits; the index consumes the low bits,
                    // so verification and indexing stay disjoint (53-bit id)
#else
    U64 key;
#endif
    int move;
    int16_t score;
    int16_t eval;
    int8_t depth;
    uint8_t flag;
    uint8_t age;
};
#if FEAT_TTPACK
#define TT_KEYV(k)   ((uint32_t)((k) >> 32))
// a zeroed slot must not match a top-32-zero position: require a real flag
#define TT_HIT(e, k) ((e).key == TT_KEYV(k) && (e).flag != TT_NONE)
#else
#define TT_KEYV(k)   (k)
#define TT_HIT(e, k) ((e).key == (k))
#endif
uint8_t ttAge = 0;
vector<TTEntry> tt;
size_t ttMask = 0;

void ttResize(int mb) {
    size_t entries = ((size_t)mb * 1024 * 1024) / sizeof(TTEntry);
#if FEAT_TTBUCKET
    size_t pow2 = 1;
    while (pow2 * 2 <= entries / 2) pow2 *= 2;   // pow2 clusters of 2 entries
    tt.assign(pow2 * 2, TTEntry{0, 0, 0, 0, 0, TT_NONE, 0});
    ttMask = pow2 - 1;
#else
    size_t pow2 = 1;
    while (pow2 * 2 <= entries) pow2 *= 2;
    tt.assign(pow2, TTEntry{0, 0, 0, 0, 0, TT_NONE, 0});
    ttMask = pow2 - 1;
#endif
}
void ttClear() { fill(tt.begin(), tt.end(), TTEntry{0, 0, 0, 0, 0, TT_NONE, 0}); }

#if FEAT_TTBUCKET
static inline TTEntry *ttProbeEntry(U64 key, bool &hit) {
    TTEntry *c = &tt[(key & ttMask) * 2];
    if (c[0].key == key && c[0].flag != TT_NONE) { hit = true; return &c[0]; }
    if (c[1].key == key && c[1].flag != TT_NONE) { hit = true; return &c[1]; }
    hit = false;
    return &c[0];
}
static inline TTEntry *ttStoreSlot(U64 key) {
    TTEntry *c = &tt[(key & ttMask) * 2];
    if (c[0].key == key && c[0].flag != TT_NONE) return &c[0];
    if (c[1].key == key && c[1].flag != TT_NONE) return &c[1];
    // victim: empty first, then stale age, then shallower depth
    int s0 = (c[0].flag == TT_NONE ? -1000 : c[0].depth) - (c[0].age != ttAge ? 200 : 0);
    int s1 = (c[1].flag == TT_NONE ? -1000 : c[1].depth) - (c[1].age != ttAge ? 200 : 0);
    return s0 <= s1 ? &c[0] : &c[1];
}
#endif
static inline void ttPrefetch(U64 key) {
    __builtin_prefetch(&tt[(key & ttMask) * (1 + FEAT_TTBUCKET)]);
}

// ===== Polyglot opening book support =====
static const U64 POLY[781] = {
    0x9D39247E33776D41ULL, 0x2AF7398005AAA5C7ULL, 0x44DB015024623547ULL, 0x9C15F73E62A76AE2ULL,
    0x75834465489C0C89ULL, 0x3290AC3A203001BFULL, 0x0FBBAD1F61042279ULL, 0xE83A908FF2FB60CAULL,
    0x0D7E765D58755C10ULL, 0x1A083822CEAFE02DULL, 0x9605D5F0E25EC3B0ULL, 0xD021FF5CD13A2ED5ULL,
    0x40BDF15D4A672E32ULL, 0x011355146FD56395ULL, 0x5DB4832046F3D9E5ULL, 0x239F8B2D7FF719CCULL,
    0x05D1A1AE85B49AA1ULL, 0x679F848F6E8FC971ULL, 0x7449BBFF801FED0BULL, 0x7D11CDB1C3B7ADF0ULL,
    0x82C7709E781EB7CCULL, 0xF3218F1C9510786CULL, 0x331478F3AF51BBE6ULL, 0x4BB38DE5E7219443ULL,
    0xAA649C6EBCFD50FCULL, 0x8DBD98A352AFD40BULL, 0x87D2074B81D79217ULL, 0x19F3C751D3E92AE1ULL,
    0xB4AB30F062B19ABFULL, 0x7B0500AC42047AC4ULL, 0xC9452CA81A09D85DULL, 0x24AA6C514DA27500ULL,
    0x4C9F34427501B447ULL, 0x14A68FD73C910841ULL, 0xA71B9B83461CBD93ULL, 0x03488B95B0F1850FULL,
    0x637B2B34FF93C040ULL, 0x09D1BC9A3DD90A94ULL, 0x3575668334A1DD3BULL, 0x735E2B97A4C45A23ULL,
    0x18727070F1BD400BULL, 0x1FCBACD259BF02E7ULL, 0xD310A7C2CE9B6555ULL, 0xBF983FE0FE5D8244ULL,
    0x9F74D14F7454A824ULL, 0x51EBDC4AB9BA3035ULL, 0x5C82C505DB9AB0FAULL, 0xFCF7FE8A3430B241ULL,
    0x3253A729B9BA3DDEULL, 0x8C74C368081B3075ULL, 0xB9BC6C87167C33E7ULL, 0x7EF48F2B83024E20ULL,
    0x11D505D4C351BD7FULL, 0x6568FCA92C76A243ULL, 0x4DE0B0F40F32A7B8ULL, 0x96D693460CC37E5DULL,
    0x42E240CB63689F2FULL, 0x6D2BDCDAE2919661ULL, 0x42880B0236E4D951ULL, 0x5F0F4A5898171BB6ULL,
    0x39F890F579F92F88ULL, 0x93C5B5F47356388BULL, 0x63DC359D8D231B78ULL, 0xEC16CA8AEA98AD76ULL,
    0x5355F900C2A82DC7ULL, 0x07FB9F855A997142ULL, 0x5093417AA8A7ED5EULL, 0x7BCBC38DA25A7F3CULL,
    0x19FC8A768CF4B6D4ULL, 0x637A7780DECFC0D9ULL, 0x8249A47AEE0E41F7ULL, 0x79AD695501E7D1E8ULL,
    0x14ACBAF4777D5776ULL, 0xF145B6BECCDEA195ULL, 0xDABF2AC8201752FCULL, 0x24C3C94DF9C8D3F6ULL,
    0xBB6E2924F03912EAULL, 0x0CE26C0B95C980D9ULL, 0xA49CD132BFBF7CC4ULL, 0xE99D662AF4243939ULL,
    0x27E6AD7891165C3FULL, 0x8535F040B9744FF1ULL, 0x54B3F4FA5F40D873ULL, 0x72B12C32127FED2BULL,
    0xEE954D3C7B411F47ULL, 0x9A85AC909A24EAA1ULL, 0x70AC4CD9F04F21F5ULL, 0xF9B89D3E99A075C2ULL,
    0x87B3E2B2B5C907B1ULL, 0xA366E5B8C54F48B8ULL, 0xAE4A9346CC3F7CF2ULL, 0x1920C04D47267BBDULL,
    0x87BF02C6B49E2AE9ULL, 0x092237AC237F3859ULL, 0xFF07F64EF8ED14D0ULL, 0x8DE8DCA9F03CC54EULL,
    0x9C1633264DB49C89ULL, 0xB3F22C3D0B0B38EDULL, 0x390E5FB44D01144BULL, 0x5BFEA5B4712768E9ULL,
    0x1E1032911FA78984ULL, 0x9A74ACB964E78CB3ULL, 0x4F80F7A035DAFB04ULL, 0x6304D09A0B3738C4ULL,
    0x2171E64683023A08ULL, 0x5B9B63EB9CEFF80CULL, 0x506AACF489889342ULL, 0x1881AFC9A3A701D6ULL,
    0x6503080440750644ULL, 0xDFD395339CDBF4A7ULL, 0xEF927DBCF00C20F2ULL, 0x7B32F7D1E03680ECULL,
    0xB9FD7620E7316243ULL, 0x05A7E8A57DB91B77ULL, 0xB5889C6E15630A75ULL, 0x4A750A09CE9573F7ULL,
    0xCF464CEC899A2F8AULL, 0xF538639CE705B824ULL, 0x3C79A0FF5580EF7FULL, 0xEDE6C87F8477609DULL,
    0x799E81F05BC93F31ULL, 0x86536B8CF3428A8CULL, 0x97D7374C60087B73ULL, 0xA246637CFF328532ULL,
    0x043FCAE60CC0EBA0ULL, 0x920E449535DD359EULL, 0x70EB093B15B290CCULL, 0x73A1921916591CBDULL,
    0x56436C9FE1A1AA8DULL, 0xEFAC4B70633B8F81ULL, 0xBB215798D45DF7AFULL, 0x45F20042F24F1768ULL,
    0x930F80F4E8EB7462ULL, 0xFF6712FFCFD75EA1ULL, 0xAE623FD67468AA70ULL, 0xDD2C5BC84BC8D8FCULL,
    0x7EED120D54CF2DD9ULL, 0x22FE545401165F1CULL, 0xC91800E98FB99929ULL, 0x808BD68E6AC10365ULL,
    0xDEC468145B7605F6ULL, 0x1BEDE3A3AEF53302ULL, 0x43539603D6C55602ULL, 0xAA969B5C691CCB7AULL,
    0xA87832D392EFEE56ULL, 0x65942C7B3C7E11AEULL, 0xDED2D633CAD004F6ULL, 0x21F08570F420E565ULL,
    0xB415938D7DA94E3CULL, 0x91B859E59ECB6350ULL, 0x10CFF333E0ED804AULL, 0x28AED140BE0BB7DDULL,
    0xC5CC1D89724FA456ULL, 0x5648F680F11A2741ULL, 0x2D255069F0B7DAB3ULL, 0x9BC5A38EF729ABD4ULL,
    0xEF2F054308F6A2BCULL, 0xAF2042F5CC5C2858ULL, 0x480412BAB7F5BE2AULL, 0xAEF3AF4A563DFE43ULL,
    0x19AFE59AE451497FULL, 0x52593803DFF1E840ULL, 0xF4F076E65F2CE6F0ULL, 0x11379625747D5AF3ULL,
    0xBCE5D2248682C115ULL, 0x9DA4243DE836994FULL, 0x066F70B33FE09017ULL, 0x4DC4DE189B671A1CULL,
    0x51039AB7712457C3ULL, 0xC07A3F80C31FB4B4ULL, 0xB46EE9C5E64A6E7CULL, 0xB3819A42ABE61C87ULL,
    0x21A007933A522A20ULL, 0x2DF16F761598AA4FULL, 0x763C4A1371B368FDULL, 0xF793C46702E086A0ULL,
    0xD7288E012AEB8D31ULL, 0xDE336A2A4BC1C44BULL, 0x0BF692B38D079F23ULL, 0x2C604A7A177326B3ULL,
    0x4850E73E03EB6064ULL, 0xCFC447F1E53C8E1BULL, 0xB05CA3F564268D99ULL, 0x9AE182C8BC9474E8ULL,
    0xA4FC4BD4FC5558CAULL, 0xE755178D58FC4E76ULL, 0x69B97DB1A4C03DFEULL, 0xF9B5B7C4ACC67C96ULL,
    0xFC6A82D64B8655FBULL, 0x9C684CB6C4D24417ULL, 0x8EC97D2917456ED0ULL, 0x6703DF9D2924E97EULL,
    0xC547F57E42A7444EULL, 0x78E37644E7CAD29EULL, 0xFE9A44E9362F05FAULL, 0x08BD35CC38336615ULL,
    0x9315E5EB3A129ACEULL, 0x94061B871E04DF75ULL, 0xDF1D9F9D784BA010ULL, 0x3BBA57B68871B59DULL,
    0xD2B7ADEEDED1F73FULL, 0xF7A255D83BC373F8ULL, 0xD7F4F2448C0CEB81ULL, 0xD95BE88CD210FFA7ULL,
    0x336F52F8FF4728E7ULL, 0xA74049DAC312AC71ULL, 0xA2F61BB6E437FDB5ULL, 0x4F2A5CB07F6A35B3ULL,
    0x87D380BDA5BF7859ULL, 0x16B9F7E06C453A21ULL, 0x7BA2484C8A0FD54EULL, 0xF3A678CAD9A2E38CULL,
    0x39B0BF7DDE437BA2ULL, 0xFCAF55C1BF8A4424ULL, 0x18FCF680573FA594ULL, 0x4C0563B89F495AC3ULL,
    0x40E087931A00930DULL, 0x8CFFA9412EB642C1ULL, 0x68CA39053261169FULL, 0x7A1EE967D27579E2ULL,
    0x9D1D60E5076F5B6FULL, 0x3810E399B6F65BA2ULL, 0x32095B6D4AB5F9B1ULL, 0x35CAB62109DD038AULL,
    0xA90B24499FCFAFB1ULL, 0x77A225A07CC2C6BDULL, 0x513E5E634C70E331ULL, 0x4361C0CA3F692F12ULL,
    0xD941ACA44B20A45BULL, 0x528F7C8602C5807BULL, 0x52AB92BEB9613989ULL, 0x9D1DFA2EFC557F73ULL,
    0x722FF175F572C348ULL, 0x1D1260A51107FE97ULL, 0x7A249A57EC0C9BA2ULL, 0x04208FE9E8F7F2D6ULL,
    0x5A110C6058B920A0ULL, 0x0CD9A497658A5698ULL, 0x56FD23C8F9715A4CULL, 0x284C847B9D887AAEULL,
    0x04FEABFBBDB619CBULL, 0x742E1E651C60BA83ULL, 0x9A9632E65904AD3CULL, 0x881B82A13B51B9E2ULL,
    0x506E6744CD974924ULL, 0xB0183DB56FFC6A79ULL, 0x0ED9B915C66ED37EULL, 0x5E11E86D5873D484ULL,
    0xF678647E3519AC6EULL, 0x1B85D488D0F20CC5ULL, 0xDAB9FE6525D89021ULL, 0x0D151D86ADB73615ULL,
    0xA865A54EDCC0F019ULL, 0x93C42566AEF98FFBULL, 0x99E7AFEABE000731ULL, 0x48CBFF086DDF285AULL,
    0x7F9B6AF1EBF78BAFULL, 0x58627E1A149BBA21ULL, 0x2CD16E2ABD791E33ULL, 0xD363EFF5F0977996ULL,
    0x0CE2A38C344A6EEDULL, 0x1A804AADB9CFA741ULL, 0x907F30421D78C5DEULL, 0x501F65EDB3034D07ULL,
    0x37624AE5A48FA6E9ULL, 0x957BAF61700CFF4EULL, 0x3A6C27934E31188AULL, 0xD49503536ABCA345ULL,
    0x088E049589C432E0ULL, 0xF943AEE7FEBF21B8ULL, 0x6C3B8E3E336139D3ULL, 0x364F6FFA464EE52EULL,
    0xD60F6DCEDC314222ULL, 0x56963B0DCA418FC0ULL, 0x16F50EDF91E513AFULL, 0xEF1955914B609F93ULL,
    0x565601C0364E3228ULL, 0xECB53939887E8175ULL, 0xBAC7A9A18531294BULL, 0xB344C470397BBA52ULL,
    0x65D34954DAF3CEBDULL, 0xB4B81B3FA97511E2ULL, 0xB422061193D6F6A7ULL, 0x071582401C38434DULL,
    0x7A13F18BBEDC4FF5ULL, 0xBC4097B116C524D2ULL, 0x59B97885E2F2EA28ULL, 0x99170A5DC3115544ULL,
    0x6F423357E7C6A9F9ULL, 0x325928EE6E6F8794ULL, 0xD0E4366228B03343ULL, 0x565C31F7DE89EA27ULL,
    0x30F5611484119414ULL, 0xD873DB391292ED4FULL, 0x7BD94E1D8E17DEBCULL, 0xC7D9F16864A76E94ULL,
    0x947AE053EE56E63CULL, 0xC8C93882F9475F5FULL, 0x3A9BF55BA91F81CAULL, 0xD9A11FBB3D9808E4ULL,
    0x0FD22063EDC29FCAULL, 0xB3F256D8ACA0B0B9ULL, 0xB03031A8B4516E84ULL, 0x35DD37D5871448AFULL,
    0xE9F6082B05542E4EULL, 0xEBFAFA33D7254B59ULL, 0x9255ABB50D532280ULL, 0xB9AB4CE57F2D34F3ULL,
    0x693501D628297551ULL, 0xC62C58F97DD949BFULL, 0xCD454F8F19C5126AULL, 0xBBE83F4ECC2BDECBULL,
    0xDC842B7E2819E230ULL, 0xBA89142E007503B8ULL, 0xA3BC941D0A5061CBULL, 0xE9F6760E32CD8021ULL,
    0x09C7E552BC76492FULL, 0x852F54934DA55CC9ULL, 0x8107FCCF064FCF56ULL, 0x098954D51FFF6580ULL,
    0x23B70EDB1955C4BFULL, 0xC330DE426430F69DULL, 0x4715ED43E8A45C0AULL, 0xA8D7E4DAB780A08DULL,
    0x0572B974F03CE0BBULL, 0xB57D2E985E1419C7ULL, 0xE8D9ECBE2CF3D73FULL, 0x2FE4B17170E59750ULL,
    0x11317BA87905E790ULL, 0x7FBF21EC8A1F45ECULL, 0x1725CABFCB045B00ULL, 0x964E915CD5E2B207ULL,
    0x3E2B8BCBF016D66DULL, 0xBE7444E39328A0ACULL, 0xF85B2B4FBCDE44B7ULL, 0x49353FEA39BA63B1ULL,
    0x1DD01AAFCD53486AULL, 0x1FCA8A92FD719F85ULL, 0xFC7C95D827357AFAULL, 0x18A6A990C8B35EBDULL,
    0xCCCB7005C6B9C28DULL, 0x3BDBB92C43B17F26ULL, 0xAA70B5B4F89695A2ULL, 0xE94C39A54A98307FULL,
    0xB7A0B174CFF6F36EULL, 0xD4DBA84729AF48ADULL, 0x2E18BC1AD9704A68ULL, 0x2DE0966DAF2F8B1CULL,
    0xB9C11D5B1E43A07EULL, 0x64972D68DEE33360ULL, 0x94628D38D0C20584ULL, 0xDBC0D2B6AB90A559ULL,
    0xD2733C4335C6A72FULL, 0x7E75D99D94A70F4DULL, 0x6CED1983376FA72BULL, 0x97FCAACBF030BC24ULL,
    0x7B77497B32503B12ULL, 0x8547EDDFB81CCB94ULL, 0x79999CDFF70902CBULL, 0xCFFE1939438E9B24ULL,
    0x829626E3892D95D7ULL, 0x92FAE24291F2B3F1ULL, 0x63E22C147B9C3403ULL, 0xC678B6D860284A1CULL,
    0x5873888850659AE7ULL, 0x0981DCD296A8736DULL, 0x9F65789A6509A440ULL, 0x9FF38FED72E9052FULL,
    0xE479EE5B9930578CULL, 0xE7F28ECD2D49EECDULL, 0x56C074A581EA17FEULL, 0x5544F7D774B14AEFULL,
    0x7B3F0195FC6F290FULL, 0x12153635B2C0CF57ULL, 0x7F5126DBBA5E0CA7ULL, 0x7A76956C3EAFB413ULL,
    0x3D5774A11D31AB39ULL, 0x8A1B083821F40CB4ULL, 0x7B4A38E32537DF62ULL, 0x950113646D1D6E03ULL,
    0x4DA8979A0041E8A9ULL, 0x3BC36E078F7515D7ULL, 0x5D0A12F27AD310D1ULL, 0x7F9D1A2E1EBE1327ULL,
    0xDA3A361B1C5157B1ULL, 0xDCDD7D20903D0C25ULL, 0x36833336D068F707ULL, 0xCE68341F79893389ULL,
    0xAB9090168DD05F34ULL, 0x43954B3252DC25E5ULL, 0xB438C2B67F98E5E9ULL, 0x10DCD78E3851A492ULL,
    0xDBC27AB5447822BFULL, 0x9B3CDB65F82CA382ULL, 0xB67B7896167B4C84ULL, 0xBFCED1B0048EAC50ULL,
    0xA9119B60369FFEBDULL, 0x1FFF7AC80904BF45ULL, 0xAC12FB171817EEE7ULL, 0xAF08DA9177DDA93DULL,
    0x1B0CAB936E65C744ULL, 0xB559EB1D04E5E932ULL, 0xC37B45B3F8D6F2BAULL, 0xC3A9DC228CAAC9E9ULL,
    0xF3B8B6675A6507FFULL, 0x9FC477DE4ED681DAULL, 0x67378D8ECCEF96CBULL, 0x6DD856D94D259236ULL,
    0xA319CE15B0B4DB31ULL, 0x073973751F12DD5EULL, 0x8A8E849EB32781A5ULL, 0xE1925C71285279F5ULL,
    0x74C04BF1790C0EFEULL, 0x4DDA48153C94938AULL, 0x9D266D6A1CC0542CULL, 0x7440FB816508C4FEULL,
    0x13328503DF48229FULL, 0xD6BF7BAEE43CAC40ULL, 0x4838D65F6EF6748FULL, 0x1E152328F3318DEAULL,
    0x8F8419A348F296BFULL, 0x72C8834A5957B511ULL, 0xD7A023A73260B45CULL, 0x94EBC8ABCFB56DAEULL,
    0x9FC10D0F989993E0ULL, 0xDE68A2355B93CAE6ULL, 0xA44CFE79AE538BBEULL, 0x9D1D84FCCE371425ULL,
    0x51D2B1AB2DDFB636ULL, 0x2FD7E4B9E72CD38CULL, 0x65CA5B96B7552210ULL, 0xDD69A0D8AB3B546DULL,
    0x604D51B25FBF70E2ULL, 0x73AA8A564FB7AC9EULL, 0x1A8C1E992B941148ULL, 0xAAC40A2703D9BEA0ULL,
    0x764DBEAE7FA4F3A6ULL, 0x1E99B96E70A9BE8BULL, 0x2C5E9DEB57EF4743ULL, 0x3A938FEE32D29981ULL,
    0x26E6DB8FFDF5ADFEULL, 0x469356C504EC9F9DULL, 0xC8763C5B08D1908CULL, 0x3F6C6AF859D80055ULL,
    0x7F7CC39420A3A545ULL, 0x9BFB227EBDF4C5CEULL, 0x89039D79D6FC5C5CULL, 0x8FE88B57305E2AB6ULL,
    0xA09E8C8C35AB96DEULL, 0xFA7E393983325753ULL, 0xD6B6D0ECC617C699ULL, 0xDFEA21EA9E7557E3ULL,
    0xB67C1FA481680AF8ULL, 0xCA1E3785A9E724E5ULL, 0x1CFC8BED0D681639ULL, 0xD18D8549D140CAEAULL,
    0x4ED0FE7E9DC91335ULL, 0xE4DBF0634473F5D2ULL, 0x1761F93A44D5AEFEULL, 0x53898E4C3910DA55ULL,
    0x734DE8181F6EC39AULL, 0x2680B122BAA28D97ULL, 0x298AF231C85BAFABULL, 0x7983EED3740847D5ULL,
    0x66C1A2A1A60CD889ULL, 0x9E17E49642A3E4C1ULL, 0xEDB454E7BADC0805ULL, 0x50B704CAB602C329ULL,
    0x4CC317FB9CDDD023ULL, 0x66B4835D9EAFEA22ULL, 0x219B97E26FFC81BDULL, 0x261E4E4C0A333A9DULL,
    0x1FE2CCA76517DB90ULL, 0xD7504DFA8816EDBBULL, 0xB9571FA04DC089C8ULL, 0x1DDC0325259B27DEULL,
    0xCF3F4688801EB9AAULL, 0xF4F5D05C10CAB243ULL, 0x38B6525C21A42B0EULL, 0x36F60E2BA4FA6800ULL,
    0xEB3593803173E0CEULL, 0x9C4CD6257C5A3603ULL, 0xAF0C317D32ADAA8AULL, 0x258E5A80C7204C4BULL,
    0x8B889D624D44885DULL, 0xF4D14597E660F855ULL, 0xD4347F66EC8941C3ULL, 0xE699ED85B0DFB40DULL,
    0x2472F6207C2D0484ULL, 0xC2A1E7B5B459AEB5ULL, 0xAB4F6451CC1D45ECULL, 0x63767572AE3D6174ULL,
    0xA59E0BD101731A28ULL, 0x116D0016CB948F09ULL, 0x2CF9C8CA052F6E9FULL, 0x0B090A7560A968E3ULL,
    0xABEEDDB2DDE06FF1ULL, 0x58EFC10B06A2068DULL, 0xC6E57A78FBD986E0ULL, 0x2EAB8CA63CE802D7ULL,
    0x14A195640116F336ULL, 0x7C0828DD624EC390ULL, 0xD74BBE77E6116AC7ULL, 0x804456AF10F5FB53ULL,
    0xEBE9EA2ADF4321C7ULL, 0x03219A39EE587A30ULL, 0x49787FEF17AF9924ULL, 0xA1E9300CD8520548ULL,
    0x5B45E522E4B1B4EFULL, 0xB49C3B3995091A36ULL, 0xD4490AD526F14431ULL, 0x12A8F216AF9418C2ULL,
    0x001F837CC7350524ULL, 0x1877B51E57A764D5ULL, 0xA2853B80F17F58EEULL, 0x993E1DE72D36D310ULL,
    0xB3598080CE64A656ULL, 0x252F59CF0D9F04BBULL, 0xD23C8E176D113600ULL, 0x1BDA0492E7E4586EULL,
    0x21E0BD5026C619BFULL, 0x3B097ADAF088F94EULL, 0x8D14DEDB30BE846EULL, 0xF95CFFA23AF5F6F4ULL,
    0x3871700761B3F743ULL, 0xCA672B91E9E4FA16ULL, 0x64C8E531BFF53B55ULL, 0x241260ED4AD1E87DULL,
    0x106C09B972D2E822ULL, 0x7FBA195410E5CA30ULL, 0x7884D9BC6CB569D8ULL, 0x0647DFEDCD894A29ULL,
    0x63573FF03E224774ULL, 0x4FC8E9560F91B123ULL, 0x1DB956E450275779ULL, 0xB8D91274B9E9D4FBULL,
    0xA2EBEE47E2FBFCE1ULL, 0xD9F1F30CCD97FB09ULL, 0xEFED53D75FD64E6BULL, 0x2E6D02C36017F67FULL,
    0xA9AA4D20DB084E9BULL, 0xB64BE8D8B25396C1ULL, 0x70CB6AF7C2D5BCF0ULL, 0x98F076A4F7A2322EULL,
    0xBF84470805E69B5FULL, 0x94C3251F06F90CF3ULL, 0x3E003E616A6591E9ULL, 0xB925A6CD0421AFF3ULL,
    0x61BDD1307C66E300ULL, 0xBF8D5108E27E0D48ULL, 0x240AB57A8B888B20ULL, 0xFC87614BAF287E07ULL,
    0xEF02CDD06FFDB432ULL, 0xA1082C0466DF6C0AULL, 0x8215E577001332C8ULL, 0xD39BB9C3A48DB6CFULL,
    0x2738259634305C14ULL, 0x61CF4F94C97DF93DULL, 0x1B6BACA2AE4E125BULL, 0x758F450C88572E0BULL,
    0x959F587D507A8359ULL, 0xB063E962E045F54DULL, 0x60E8ED72C0DFF5D1ULL, 0x7B64978555326F9FULL,
    0xFD080D236DA814BAULL, 0x8C90FD9B083F4558ULL, 0x106F72FE81E2C590ULL, 0x7976033A39F7D952ULL,
    0xA4EC0132764CA04BULL, 0x733EA705FAE4FA77ULL, 0xB4D8F77BC3E56167ULL, 0x9E21F4F903B33FD9ULL,
    0x9D765E419FB69F6DULL, 0xD30C088BA61EA5EFULL, 0x5D94337FBFAF7F5BULL, 0x1A4E4822EB4D7A59ULL,
    0x6FFE73E81B637FB3ULL, 0xDDF957BC36D8B9CAULL, 0x64D0E29EEA8838B3ULL, 0x08DD9BDFD96B9F63ULL,
    0x087E79E5A57D1D13ULL, 0xE328E230E3E2B3FBULL, 0x1C2559E30F0946BEULL, 0x720BF5F26F4D2EAAULL,
    0xB0774D261CC609DBULL, 0x443F64EC5A371195ULL, 0x4112CF68649A260EULL, 0xD813F2FAB7F5C5CAULL,
    0x660D3257380841EEULL, 0x59AC2C7873F910A3ULL, 0xE846963877671A17ULL, 0x93B633ABFA3469F8ULL,
    0xC0C0F5A60EF4CDCFULL, 0xCAF21ECD4377B28CULL, 0x57277707199B8175ULL, 0x506C11B9D90E8B1DULL,
    0xD83CC2687A19255FULL, 0x4A29C6465A314CD1ULL, 0xED2DF21216235097ULL, 0xB5635C95FF7296E2ULL,
    0x22AF003AB672E811ULL, 0x52E762596BF68235ULL, 0x9AEBA33AC6ECC6B0ULL, 0x944F6DE09134DFB6ULL,
    0x6C47BEC883A7DE39ULL, 0x6AD047C430A12104ULL, 0xA5B1CFDBA0AB4067ULL, 0x7C45D833AFF07862ULL,
    0x5092EF950A16DA0BULL, 0x9338E69C052B8E7BULL, 0x455A4B4CFE30E3F5ULL, 0x6B02E63195AD0CF8ULL,
    0x6B17B224BAD6BF27ULL, 0xD1E0CCD25BB9C169ULL, 0xDE0C89A556B9AE70ULL, 0x50065E535A213CF6ULL,
    0x9C1169FA2777B874ULL, 0x78EDEFD694AF1EEDULL, 0x6DC93D9526A50E68ULL, 0xEE97F453F06791EDULL,
    0x32AB0EDB696703D3ULL, 0x3A6853C7E70757A7ULL, 0x31865CED6120F37DULL, 0x67FEF95D92607890ULL,
    0x1F2B1D1F15F6DC9CULL, 0xB69E38A8965C6B65ULL, 0xAA9119FF184CCCF4ULL, 0xF43C732873F24C13ULL,
    0xFB4A3D794A9A80D2ULL, 0x3550C2321FD6109CULL, 0x371F77E76BB8417EULL, 0x6BFA9AAE5EC05779ULL,
    0xCD04F3FF001A4778ULL, 0xE3273522064480CAULL, 0x9F91508BFFCFC14AULL, 0x049A7F41061A9E60ULL,
    0xFCB6BE43A9F2FE9BULL, 0x08DE8A1C7797DA9BULL, 0x8F9887E6078735A1ULL, 0xB5B4071DBFC73A66ULL,
    0x230E343DFBA08D33ULL, 0x43ED7F5A0FAE657DULL, 0x3A88A0FBBCB05C63ULL, 0x21874B8B4D2DBC4FULL,
    0x1BDEA12E35F6A8C9ULL, 0x53C065C6C8E63528ULL, 0xE34A1D250E7A8D6BULL, 0xD6B04D3B7651DD7EULL,
    0x5E90277E7CB39E2DULL, 0x2C046F22062DC67DULL, 0xB10BB459132D0A26ULL, 0x3FA9DDFB67E2F199ULL,
    0x0E09B88E1914F7AFULL, 0x10E8B35AF3EEAB37ULL, 0x9EEDECA8E272B933ULL, 0xD4C718BC4AE8AE5FULL,
    0x81536D601170FC20ULL, 0x91B534F885818A06ULL, 0xEC8177F83F900978ULL, 0x190E714FADA5156EULL,
    0xB592BF39B0364963ULL, 0x89C350C893AE7DC1ULL, 0xAC042E70F8B383F2ULL, 0xB49B52E587A1EE60ULL,
    0xFB152FE3FF26DA89ULL, 0x3E666E6F69AE2C15ULL, 0x3B544EBE544C19F9ULL, 0xE805A1E290CF2456ULL,
    0x24B33C9D7ED25117ULL, 0xE74733427B72F0C1ULL, 0x0A804D18B7097475ULL, 0x57E3306D881EDB4FULL,
    0x4AE7D6A36EB5DBCBULL, 0x2D8D5432157064C8ULL, 0xD1E649DE1E7F268BULL, 0x8A328A1CEDFE552CULL,
    0x07A3AEC79624C7DAULL, 0x84547DDC3E203C94ULL, 0x990A98FD5071D263ULL, 0x1A4FF12616EEFC89ULL,
    0xF6F7FD1431714200ULL, 0x30C05B1BA332F41CULL, 0x8D2636B81555A786ULL, 0x46C9FEB55D120902ULL,
    0xCCEC0A73B49C9921ULL, 0x4E9D2827355FC492ULL, 0x19EBB029435DCB0FULL, 0x4659D2B743848A2CULL,
    0x963EF2C96B33BE31ULL, 0x74F85198B05A2E7DULL, 0x5A0F544DD2B1FB18ULL, 0x03727073C2E134B1ULL,
    0xC7F6AA2DE59AEA61ULL, 0x352787BAA0D7C22FULL, 0x9853EAB63B5E0B35ULL, 0xABBDCDD7ED5C0860ULL,
    0xCF05DAF5AC8D77B0ULL, 0x49CAD48CEBF4A71EULL, 0x7A4C10EC2158C4A6ULL, 0xD9E92AA246BF719EULL,
    0x13AE978D09FE5557ULL, 0x730499AF921549FFULL, 0x4E4B705B92903BA4ULL, 0xFF577222C14F0A3AULL,
    0x55B6344CF97AAFAEULL, 0xB862225B055B6960ULL, 0xCAC09AFBDDD2CDB4ULL, 0xDAF8E9829FE96B5FULL,
    0xB5FDFC5D3132C498ULL, 0x310CB380DB6F7503ULL, 0xE87FBB46217A360EULL, 0x2102AE466EBB1148ULL,
    0xF8549E1A3AA5E00DULL, 0x07A69AFDCC42261AULL, 0xC4C118BFE78FEAAEULL, 0xF9F4892ED96BD438ULL,
    0x1AF3DBE25D8F45DAULL, 0xF5B4B0B0D2DEEEB4ULL, 0x962ACEEFA82E1C84ULL, 0x046E3ECAAF453CE9ULL,
    0xF05D129681949A4CULL, 0x964781CE734B3C84ULL, 0x9C2ED44081CE5FBDULL, 0x522E23F3925E319EULL,
    0x177E00F9FC32F791ULL, 0x2BC60A63A6F3B3F2ULL, 0x222BBFAE61725606ULL, 0x486289DDCC3D6780ULL,
    0x7DC7785B8EFDFC80ULL, 0x8AF38731C02BA980ULL, 0x1FAB64EA29A2DDF7ULL, 0xE4D9429322CD065AULL,
    0x9DA058C67844F20CULL, 0x24C0E332B70019B0ULL, 0x233003B5A6CFE6ADULL, 0xD586BD01C5C217F6ULL,
    0x5E5637885F29BC2BULL, 0x7EBA726D8C94094BULL, 0x0A56A5F0BFE39272ULL, 0xD79476A84EE20D06ULL,
    0x9E4C1269BAA4BF37ULL, 0x17EFEE45B0DEE640ULL, 0x1D95B0A5FCF90BC6ULL, 0x93CBE0B699C2585DULL,
    0x65FA4F227A2B6D79ULL, 0xD5F9E858292504D5ULL, 0xC2B5A03F71471A6FULL, 0x59300222B4561E00ULL,
    0xCE2F8642CA0712DCULL, 0x7CA9723FBB2E8988ULL, 0x2785338347F2BA08ULL, 0xC61BB3A141E50E8CULL,
    0x150F361DAB9DEC26ULL, 0x9F6A419D382595F4ULL, 0x64A53DC924FE7AC9ULL, 0x142DE49FFF7A7C3DULL,
    0x0C335248857FA9E7ULL, 0x0A9C32D5EAE45305ULL, 0xE6C42178C4BBB92EULL, 0x71F1CE2490D20B07ULL,
    0xF1BCC3D275AFE51AULL, 0xE728E8C83C334074ULL, 0x96FBF83A12884624ULL, 0x81A1549FD6573DA5ULL,
    0x5FA7867CAF35E149ULL, 0x56986E2EF3ED091BULL, 0x917F1DD5F8886C61ULL, 0xD20D8C88C8FFE65FULL,
    0x31D71DCE64B2C310ULL, 0xF165B587DF898190ULL, 0xA57E6339DD2CF3A0ULL, 0x1EF6E6DBB1961EC9ULL,
    0x70CC73D90BC26E24ULL, 0xE21A6B35DF0C3AD7ULL, 0x003A93D8B2806962ULL, 0x1C99DED33CB890A1ULL,
    0xCF3145DE0ADD4289ULL, 0xD0E4427A5514FB72ULL, 0x77C621CC9FB3A483ULL, 0x67A34DAC4356550BULL,
    0xF8D626AAAF278509ULL,
};
// polyglot offsets
// RandomPiece: 0..767 (64*kind), kind = 2*piecetype + color, order: bp,wp,bn,wn,bb,wb,br,wr,bq,wq,bk,wk
// RandomCastle: 768..771 (wK,wQ,bK,bQ)
// RandomEnPassant: 772..779
// RandomTurn: 780
static U64 polyKey(const Position &p) {
    U64 key = 0;
    // pieces. polyglot kind: pawn0 knight1 bishop2 rook3 queen4 king5; offset = 64*(2*kind + (whiteIsColor? ... ))
    // polyglot: kind_of_piece = 2*type + color where color: white=1? Actually: black pawn=0, white pawn=1, black knight=2 ...
    // mapping: pieceOffset = 64*polyPiece + 8*rank + file
    for (int pc = 0; pc < 12; pc++) {
        U64 b = p.bb[pc];
        int type = pc % 6;        // PAWN..KING 0..5
        int color = pc / 6;       // 0 white, 1 black
        // polyglot kind: black=0 base; kind = 2*type + (color==WHITE?1:0)
        int kind = 2 * type + (color == WHITE ? 1 : 0);
        while (b) {
            int sq = poplsb(b);
            int f = fileOf(sq), r = rankOf(sq);
            key ^= POLY[64 * kind + 8 * r + f];
        }
    }
    // castling: polyglot order wK(768) wQ(769) bK(770) bQ(771); our castle bits: 1 WK,2 WQ,4 BK,8 BQ
    if (p.castle & 1) key ^= POLY[768];
    if (p.castle & 2) key ^= POLY[769];
    if (p.castle & 4) key ^= POLY[770];
    if (p.castle & 8) key ^= POLY[771];
    // en passant: only if a pawn can actually capture (polyglot rule)
    if (p.ep != -1) {
        int epf = fileOf(p.ep);
        int us = p.side;
        U64 ourPawns = p.bb[us == WHITE ? WP : BP];
        // pawn that could capture sits adjacent on the rank behind ep
        bool canCap = (pawnAtt[us ^ 1][p.ep] & ourPawns) != 0;
        if (canCap) key ^= POLY[772 + epf];
    }
    if (p.side == WHITE) key ^= POLY[780];
    return key;
}

#pragma pack(push, 1)
struct PolyEntry { U64 key; uint16_t move; uint16_t weight; uint32_t learn; };
#pragma pack(pop)

static inline U64 bswap64(U64 x) { return __builtin_bswap64(x); }
static inline uint16_t bswap16(uint16_t x) { return __builtin_bswap16(x); }

vector<PolyEntry> bookEntries;
bool bookLoaded = false;

bool bookLoad(const string &path) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    long n = sz / 16;
    if (n <= 0) { fclose(f); return false; }
    bookEntries.resize(n);
    size_t got = fread(bookEntries.data(), 16, n, f);
    fclose(f);
    if (got != (size_t)n) { bookEntries.clear(); return false; }
    for (auto &e : bookEntries) { e.key = bswap64(e.key); e.move = bswap16(e.move); e.weight = bswap16(e.weight); }
    bookLoaded = true;
    return true;
}

// decode polyglot move to our move int (need legal move match)
int bookProbe(const Position &p, bool uniform = false) {
    if (!bookLoaded) return 0;
    U64 key = polyKey(p);
    // binary search (entries sorted by key)
    long lo = 0, hi = (long)bookEntries.size() - 1, first = -1;
    while (lo <= hi) {
        long mid = (lo + hi) / 2;
        if (bookEntries[mid].key >= key) { if (bookEntries[mid].key == key) first = mid; hi = mid - 1; }
        else lo = mid + 1;
    }
    if (first < 0) return 0;
    // gather all entries with this key, pick weighted-random
    int total = 0;
    long i = first;
    vector<pair<int,uint16_t>> cands; // polyMove, weight
    while (i < (long)bookEntries.size() && bookEntries[i].key == key) {
        cands.push_back({bookEntries[i].move, bookEntries[i].weight});
        total += bookEntries[i].weight;
        i++;
    }
    if (total <= 0) return 0;
    int chosen = -1;
    if (uniform) {
        chosen = cands[(size_t)(rand64() % cands.size())].first;
    } else {
        int pick = (int)(rand64() % (U64)total);
        int acc = 0;
        for (auto &c : cands) {
            acc += c.second;
            if (pick < acc) { chosen = c.first; break; }
        }
    }
    if (chosen < 0) chosen = cands[0].first;
    // decode polyglot move: bits: to_file 0-2, to_row 3-5, from_file 6-8, from_row 9-11, promo 12-14
    int tf = chosen & 7, tr = (chosen >> 3) & 7, ff = (chosen >> 6) & 7, fr = (chosen >> 9) & 7, promo = (chosen >> 12) & 7;
    int from = SQ(ff, fr), to = SQ(tf, tr);
    // polyglot encodes castling as king-captures-own-rook (e1h1 etc). Match against our legal moves.
    MoveList ml;
    genMoves(p, ml, false);
    int promoPiece[5] = {0, KNIGHT, BISHOP, ROOK, QUEEN}; // polyglot: 0 none,1 N,2 B,3 R,4 Q
    for (int j = 0; j < ml.count; j++) {
        int m = ml.moves[j];
        int mf = M_FROM(m), mt = M_TO(m);
        // handle castle: polyglot 'to' is the rook square; our castle 'to' is king dest
        if (M_CASTLE(m)) {
            int kf = mf; // king from
            // polyglot from == king from, to == rook square
            int rookSq;
            if (mt == SQ(6,0)) rookSq = SQ(7,0);
            else if (mt == SQ(2,0)) rookSq = SQ(0,0);
            else if (mt == SQ(6,7)) rookSq = SQ(7,7);
            else rookSq = SQ(0,7);
            if (from == kf && to == rookSq) { Position t = p; if (makeMove(t, m)) return m; }
        }
        if (mf == from && mt == to) {
            if (promo == 0 || (M_PROMO(m) && pieceType(M_PROMO(m)) == promoPiece[promo])) {
                Position t = p; if (makeMove(t, m)) return m;
            }
        }
    }
    return 0;
}

// Serialize a position for an EPD/FEN opening suite.  This is used only by
// the command-line test utility below; it does not participate in search.
string positionFen(const Position &p, int fullmove) {
    static const char pieceChar[12] = {
        'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k'
    };
    string out;
    for (int rank = 7; rank >= 0; rank--) {
        int empty = 0;
        for (int file = 0; file < 8; file++) {
            int pc = p.board[SQ(file, rank)];
            if (pc == NO_PIECE) {
                empty++;
            } else {
                if (empty) { out += char('0' + empty); empty = 0; }
                out += pieceChar[pc];
            }
        }
        if (empty) out += char('0' + empty);
        if (rank) out += '/';
    }
    out += p.side == WHITE ? " w " : " b ";
    if (!p.castle) out += '-';
    else {
        if (p.castle & 1) out += 'K';
        if (p.castle & 2) out += 'Q';
        if (p.castle & 4) out += 'k';
        if (p.castle & 8) out += 'q';
    }
    out += ' ';
    if (p.ep == -1) out += '-';
    else {
        out += char('a' + fileOf(p.ep));
        out += char('1' + rankOf(p.ep));
    }
    out += ' ' + to_string(p.fifty) + ' ' + to_string(fullmove);
    return out;
}

bool generateBookSuite(const char *path, int count, int minPlies,
                       int maxPlies, U64 seed) {
    if (!bookLoaded || count <= 0 || minPlies < 0 || maxPlies < minPlies)
        return false;
    FILE *f = fopen(path, "w");
    if (!f) return false;

    rngState = seed ? seed : 1;
    vector<U64> seen;
    int written = 0;
    int attempts = 0;
    const int attemptLimit = count * 500;
    while (written < count && attempts++ < attemptLimit) {
        Position p;
        setFen(p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        int target = minPlies;
        if (maxPlies > minPlies)
            target += (int)(rand64() % (U64)(maxPlies - minPlies + 1));

        int made = 0;
        while (made < target) {
            int m = bookProbe(p, true);
            if (!m) break;
            Position next = p;
            if (!makeMove(next, m)) break;
            p = next;
            made++;
        }
        if (made < minPlies || find(seen.begin(), seen.end(), p.key) != seen.end())
            continue;

        seen.push_back(p.key);
        string fen = positionFen(p, 1 + made / 2);
        fprintf(f, "%s\n", fen.c_str());
        written++;
    }
    fclose(f);
    printf("book suite: %d unique positions written to %s (%d attempts)\n",
           written, path, attempts);
    return written == count;
}

// Book walk plus a short random tail.  The tiny Polyglot book caps how many
// unique positions pure walks can yield; appending a few random legal moves,
// each vetted by the static evaluation, gives an effectively unlimited pool
// of fresh opening-like positions for confirmation gates.  Positions listed
// in excludePath (e.g. the original suite) are never re-emitted.
bool generateTailSuite(const char *path, int count, int minBook, int tailMin,
                       int tailMax, U64 seed, int evalMargin,
                       const char *excludePath) {
    if (!bookLoaded || !nnueLoaded || count <= 0 || minBook < 0
        || tailMin < 0 || tailMax < tailMin)
        return false;

    vector<U64> seen;
    if (excludePath) {
        FILE *ex = fopen(excludePath, "r");
        if (!ex) return false;
        char line[512];
        while (fgets(line, sizeof(line), ex)) {
            if (strlen(line) < 10) continue;
            Position q;
            setFen(q, line);
            seen.push_back(q.key);
        }
        fclose(ex);
        printf("excluding %zu prior positions from %s\n", seen.size(), excludePath);
    }

    FILE *f = fopen(path, "w");
    if (!f) return false;
    rngState = seed ? seed : 1;
    static Accumulator suiteAcc;
    int written = 0, attempts = 0;
    const long long attemptLimit = (long long)count * 500;
    while (written < count && attempts++ < attemptLimit) {
        Position p;
        setFen(p, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        int made = 0;
        while (true) {
            int m = bookProbe(p, true);
            if (!m) break;
            Position next = p;
            if (!makeMove(next, m)) break;
            p = next;
            made++;
        }
        if (made < minBook) continue;

        int tail = tailMin + (tailMax > tailMin
                   ? (int)(rand64() % (U64)(tailMax - tailMin + 1)) : 0);
        bool ok = true;
        for (int t = 0; t < tail && ok; t++) {
            MoveList ml;
            genMoves(p, ml, false);
            int legalM[256], n = 0;
            for (int j = 0; j < ml.count; j++) {
                Position q = p;
                if (makeMove(q, ml.moves[j])) legalM[n++] = ml.moves[j];
            }
            ok = false;
            // try random moves until one keeps the position roughly level
            for (int tries = 0; tries < 8 && n > 0; tries++) {
                int pick = (int)(rand64() % (U64)n);
                Position q = p;
                makeMove(q, legalM[pick]);
                nnRefresh(suiteAcc, q);
                if (!inCheck(q) && abs(evaluate(q, &suiteAcc)) <= evalMargin) {
                    p = q;
                    made++;
                    ok = true;
                    break;
                }
                legalM[pick] = legalM[--n];
            }
        }
        if (!ok || inCheck(p)) continue;
        nnRefresh(suiteAcc, p);
        if (abs(evaluate(p, &suiteAcc)) > evalMargin) continue;
        if (find(seen.begin(), seen.end(), p.key) != seen.end()) continue;

        seen.push_back(p.key);
        fprintf(f, "%s\n", positionFen(p, 1 + made / 2).c_str());
        written++;
    }
    fclose(f);
    printf("tail suite: %d unique positions written to %s (%d attempts)\n",
           written, path, attempts);
    return written == count;
}

#if FEAT_NPCORR
// Test-only: walk the move tree asserting the incrementally maintained
// per-color non-pawn keys always equal a from-scratch recomputation.
long long npVerifyFails = 0, npVerifyNodes = 0;
void npVerifyWalk(const Position &p, int depth) {
    U64 w = 0, b = 0;
    for (int pc = 0; pc < 12; pc++) {
        if (pieceType(pc) == PAWN) continue;
        U64 bb = p.bb[pc];
        while (bb) { int sq = poplsb(bb); (pc < 6 ? w : b) ^= zPiece[pc][sq]; }
    }
    npVerifyNodes++;
    if (w != p.npKey[WHITE] || b != p.npKey[BLACK]) npVerifyFails++;
    if (depth == 0) return;
    MoveList ml;
    genMoves(p, ml, false);
    for (int i = 0; i < ml.count; i++) {
        Position q = p;
        if (!makeMove(q, ml.moves[i])) continue;
        npVerifyWalk(q, depth - 1);
    }
}
#endif

// ----------------------- search ----------------------------
const int INF = 32000;
const int MATE = 31000;
const int MAX_PLY = 128;
const int EVAL_NONE = -32000;   // sentinel: TT entry has no usable static eval

// feature toggles for A/B testing (all default on)
#ifndef FEAT_QTT
#define FEAT_QTT 1
#endif
#ifndef FEAT_REFINE
#define FEAT_REFINE 0   // measured -151 elo here: TT bounds too noisy to refine eval
#endif
#ifndef FEAT_ASP
#define FEAT_ASP 1
#endif
#ifndef FEAT_CORR
#define FEAT_CORR 1
#endif
#ifndef FEAT_CAPHIST
#define FEAT_CAPHIST 1
#endif
#ifndef FEAT_CH2
#define FEAT_CH2 0   // measured -29 elo here: disturbs history scale used by pruning/LMR
#endif
#ifndef FEAT_CUTNODE
#define FEAT_CUTNODE 0   // measured -19 elo here: base LMR is already aggressive enough
#endif
#ifndef FEAT_KRESET
#define FEAT_KRESET 1
#endif
#ifndef FEAT_TM
#define FEAT_TM 0   // measured -8 elo (neutral) at 6+0.06s; stability heuristic already covers it
#endif
#ifndef FEAT_QREP
#define FEAT_QREP 0   // qsearch repetition/fifty detection and key-stack maintenance
#endif
#ifndef FEAT_BADCAPLAST
#define FEAT_BADCAPLAST 1  // +22.39 +/- 13.78 elo, 1,492-game SPRT
#endif
#ifndef BADCAP_SCORE_BAND
#define BADCAP_SCORE_BAND -900000
#endif
#ifndef FEAT_BADCAPLMR
#define FEAT_BADCAPLMR 0  // reduce late SEE-negative captures by one ply
#endif
#ifndef FEAT_CAPHIST_LMR
#define FEAT_CAPHIST_LMR 0  // reduce only bad captures with negative capture history
#endif
#ifndef FEAT_QEVASION_CONTEXT
#define FEAT_QEVASION_CONTEXT 0  // use counter/continuation context to order qsearch evasions
#endif
#ifndef FEAT_QSTANDPAT_TT
#define FEAT_QSTANDPAT_TT 0  // cache qsearch stand-pat beta cutoffs
#endif
#ifndef FEAT_FIFTY_DAMP
#define FEAT_FIFTY_DAMP 0  // damp corrected static eval without caching the damping
#endif
#ifndef FEAT_QUIET_SEE
#define FEAT_QUIET_SEE 1  // +18.52 +/- 12.26 elo, 1,878-game SPRT
#endif
#ifndef QUIET_SEE_MAX_DEPTH
#define QUIET_SEE_MAX_DEPTH 5  // +17.56 +/- 11.84 elo over depth 4
#endif
#ifndef QUIET_SEE_MARGIN
#define QUIET_SEE_MARGIN 80
#endif
#ifndef QUIET_SEE_FRONTIER_MARGIN
#define QUIET_SEE_FRONTIER_MARGIN QUIET_SEE_MARGIN
#endif
#ifndef FEAT_QCHECKSEE
#define FEAT_QCHECKSEE 1  // +14.26 +/- 10.29 elo, 1,706-game SPRT
#endif
#ifndef FEAT_QUIET_CHECK_GUARD
#define FEAT_QUIET_CHECK_GUARD 1  // pooled +7.90 +/- 4.62 elo over two disjoint gates
#endif
#ifndef FEAT_DISCOVERED_CHECK_GUARD
#define FEAT_DISCOVERED_CHECK_GUARD 0 // extend the accepted guard to quiet discovered checks
#endif
#ifndef FEAT_DISCOVERED_CHECK_VERIFY
#define FEAT_DISCOVERED_CHECK_VERIFY 0 // test-only exact copy/make oracle
#endif
#ifndef FEAT_MALUSCAP
#define FEAT_MALUSCAP 0  // apply quiet-history maluses when a noisy move causes the cutoff
#endif
#ifndef FEAT_CAPFUT
#define FEAT_CAPFUT 0  // futility-prune captures whose victim value cannot lift eval to alpha
#endif
#ifndef FEAT_QSEVADE
#define FEAT_QSEVADE 0  // in qsearch, skip SEE-losing quiet evasions once one legal reply exists
#endif
#ifndef SING_MIN_DEPTH
#define SING_MIN_DEPTH 8  // singular-extension/multicut eligibility floor
#endif
#ifndef FEAT_TTCAPLMR
#define FEAT_TTCAPLMR 0  // reduce quiets one extra ply when the TT move is a capture
#endif
#ifndef FEAT_GUARDDEPTH
#define FEAT_GUARDDEPTH 0  // extend the quiet-check guard to the full LMP depth range
#endif
#ifndef FEAT_CONTCORR
#define FEAT_CONTCORR 1  // +24.93 +/- 14.65 elo, 754-game SPRT (CONTCORR-01)
#endif
#ifndef FEAT_CONTCORR2
#define FEAT_CONTCORR2 0  // (prev2, prev) pair-keyed eval correction on top of CONTCORR
#endif
#ifndef FEAT_KPCORR
#define FEAT_KPCORR 1  // +14.20 +/- 10.23 elo, 1,714-game SPRT (KPCORR-01)
#endif
#ifndef FEAT_QCONTCORR
#define FEAT_QCONTCORR 0  // maintain ssMove in qsearch so CONTCORR applies there too
#endif
#ifndef FEAT_TTSTORE
#define FEAT_TTSTORE 1  // +31.54 +/- 16.74 elo, 696-game SPRT (TTSTORE-01)
#endif
#ifndef FEAT_BADCAPCHECK
#define FEAT_BADCAPCHECK 0  // rejected: BADCAPCHECK-01 -19.37 +/- 15.29 in 772 games
#endif
#ifndef FEAT_ROOTEFFORT
#define FEAT_ROOTEFFORT 0  // order root moves by previous-iteration subtree effort
#endif
#ifndef FEAT_PIECETO
#define FEAT_PIECETO 0  // piece-destination history, consumed in ordering only
#endif
#ifndef FEAT_SEEGE
#define FEAT_SEEGE 1  // search-identical; +1.8% speed, 5/5 timing blocks (SEEGE-01)
#endif
#ifndef FEAT_PARTIALID
#define FEAT_PARTIALID 0  // rejected: PARTIALID-01 +2.14 +/- 4.30, below the bar
#endif
#ifndef FEAT_TTAGEHIT
#define FEAT_TTAGEHIT 0  // rejected: TTAGEHIT-01 +1.08 +/- 5.44, below the bar
#endif
#ifndef FEAT_QSCHAIN
#define FEAT_QSCHAIN 0  // rejected: QSCHAIN-01 -14.77 +/- 13.96 in 824 games
#endif
#ifndef FEAT_CORRCLAMP
#define FEAT_CORRCLAMP 0  // rejected: CORRCLAMP-01 -5.97 +/- 10.10
#endif
#ifndef FEAT_CORRBOUND
#define FEAT_CORRBOUND 0  // correction updates move only in their bound's provable direction
#endif
#ifndef FEAT_ONEREPLY
#define FEAT_ONEREPLY 0  // extend the single legal evasion of a check by one ply
#endif
#ifndef FEAT_THREATLMR
#define FEAT_THREATLMR 0  // a badly failed null search lowers quiet LMR at the node
#endif
#if FEAT_DISCOVERED_CHECK_GUARD
// Exact for generated non-capturing, non-promotion moves.  The destination
// stays occupied because it can re-block the line exposed at the source.
static inline bool givesQuietOrDiscoveredCheck(const Position &p, int m) {
    const int from = M_FROM(m), to = M_TO(m);
    const int us = p.side, them = us ^ 1;
    const int ksq = lsb(p.bb[them == WHITE ? WK : BK]);
    const U64 fromBB = 1ULL << from, toBB = 1ULL << to;
    const U64 kingBB = 1ULL << ksq;
    U64 occ = (p.occ[BOTH] & ~fromBB) | toBB;
    U64 diag = p.bb[us == WHITE ? WB : BB_] | p.bb[us == WHITE ? WQ : BQ];
    U64 orth = p.bb[us == WHITE ? WR : BR] | p.bb[us == WHITE ? WQ : BQ];

    // Castling relocates a second piece, so mirror the rook move explicitly.
    if (M_CASTLE(m)) {
        const bool kingSide = fileOf(to) == 6;
        const int rookFrom = kingSide ? to + 1 : to - 2;
        const int rookTo = kingSide ? to - 1 : to + 1;
        const U64 rookFromBB = 1ULL << rookFrom;
        const U64 rookToBB = 1ULL << rookTo;
        occ = (occ & ~rookFromBB) | rookToBB;
        orth = (orth & ~rookFromBB) | rookToBB;
        return (bishopAttacks(ksq, occ) & diag)
            || (rookAttacks(ksq, occ) & orth);
    }

    bool direct = false;
    switch (pieceType(M_PIECE(m))) {
        case PAWN:   direct = (pawnAtt[us][to] & kingBB) != 0; break;
        case KNIGHT: direct = (knightAtt[to] & kingBB) != 0; break;
        case BISHOP: direct = (bishopAttacks(to, occ) & kingBB) != 0; break;
        case ROOK:   direct = (rookAttacks(to, occ) & kingBB) != 0; break;
        case QUEEN:  direct = (queenAttacks(to, occ) & kingBB) != 0; break;
        default: break; // legal kings cannot move adjacent to each other
    }
    if (direct) return true;

    // A non-mover can gain an attack only when the source square uncovered
    // an own slider.  Excluding fromBB avoids treating a moved slider as if it
    // still occupied its old square.
    const int df = fileOf(from) - fileOf(ksq);
    const int dr = rankOf(from) - rankOf(ksq);
    if (df == 0 || dr == 0)
        return (rookAttacks(ksq, occ) & (orth & ~fromBB)) != 0;
    if (abs(df) == abs(dr))
        return (bishopAttacks(ksq, occ) & (diag & ~fromBB)) != 0;
    return false;
}
#endif
#ifndef FEAT_IMPROVING_FALLBACK
#define FEAT_IMPROVING_FALLBACK 0  // skip in-check ancestors when measuring improvement
#endif
#ifndef LMP_MAX_DEPTH
#define LMP_MAX_DEPTH 8  // +8.68 +/- 6.89 elo, 3,522-game SPRT
#endif

static inline int dampForFiftyMoveRule(int score, const Position &p) {
#if FEAT_FIFTY_DAMP
    return score * (200 - min(p.fifty, 100)) / 200;
#else
    (void)p;
    return score;
#endif
}
const int CORR_SIZE = 16384;        // pawn-structure correction history
const int CORR_GRAIN = 256;
const int CORR_MAX = 64 * CORR_GRAIN;

static inline int matIndex(const Position &p) {   // material-signature hash
    U64 h = 0x9E3779B97F4A7C15ULL;
    for (int pc = 0; pc < 12; pc++) h = h * 31 + (U64)popcnt(p.bb[pc]);
    return (int)(h & (CORR_SIZE - 1));
}

struct SearchInfo {
    atomic<bool> stop{false};
    chrono::steady_clock::time_point start;
    long long softLimit = 0, hardLimit = 0; // ms; 0 = none
    int rootDepth = 1;                       // current ID iteration depth
    bool fixedTime = false;                  // movetime: use it fully
    int maxDepth = 64;
    long long nodes = 0;
    long long nodeLimit = 0;
    bool timed = false;
};
SearchInfo si;

// SPSA-tunable search constants, exposed as UCI spin options
// SPSA-01: +21.56 +/- 13.49 elo, 968-game SPRT on a disjoint suite.
int tRFPMargin = 75, tRazorMargin = 299, tFutBase = 86, tFutSlope = 73;
int tNullEvalDiv = 200, tLMRDiv = 235, tAspDelta = 26, tSeePruneM = 162;
int tDeltaMargin = 250, tHistPruneM = 1399;
// correction-table blend weights, /256 (NpW is /512 across the color pair);
// defaults reproduce the accepted unweighted sums bit-for-bit
int tCorrPawnW = 256, tCorrContW = 256, tCorrNpW = 256, tCorrKpW = 256;
// structural constants never covered by SPSA-01/02; defaults reproduce the
// accepted baked-in values exactly
int tNullBase = 3, tNullDDiv = 4;
int tSingDepth = SING_MIN_DEPTH, tSingBetaM = 2, tSingTTOff = 2, tDblExtM = 23;
int tIIRDepth = 5, tFutDepth = 6, tRFPDepth = 7, tRazorDepth = 2;
int tLMPBase = 4, tSeeCapDepth = 7;
int tLMRHistDiv = 8000, tLMRHistClamp = 2, tLMRPvRed = 1, tLMRImpRed = 1, tLMRKillRed = 1;
struct TunableRef { const char *name; int *p; int lo, hi; };
extern void initLMR();
TunableRef tunables[] = {
    {"RFPMargin", &tRFPMargin, 20, 300}, {"RazorMargin", &tRazorMargin, 60, 800},
    {"FutBase", &tFutBase, 10, 400},     {"FutSlope", &tFutSlope, 20, 300},
    {"NullEvalDiv", &tNullEvalDiv, 50, 800}, {"LMRDiv", &tLMRDiv, 100, 600},
    {"AspDelta", &tAspDelta, 5, 120},    {"SeePruneM", &tSeePruneM, 40, 600},
    {"DeltaMargin", &tDeltaMargin, 50, 800}, {"HistPruneM", &tHistPruneM, 300, 6000},
    {"CorrPawnW", &tCorrPawnW, 0, 512},  {"CorrContW", &tCorrContW, 0, 512},
    {"CorrNpW", &tCorrNpW, 0, 512},      {"CorrKpW", &tCorrKpW, 0, 512},
    {"NullBase", &tNullBase, 2, 5},      {"NullDDiv", &tNullDDiv, 3, 9},
    {"SingDepth", &tSingDepth, 6, 10},   {"SingBetaM", &tSingBetaM, 1, 4},
    {"SingTTOff", &tSingTTOff, 2, 5},    {"DblExtM", &tDblExtM, 10, 60},
    {"IIRDepth", &tIIRDepth, 3, 7},      {"FutDepth", &tFutDepth, 4, 9},
    {"RFPDepth", &tRFPDepth, 5, 10},     {"RazorDepth", &tRazorDepth, 2, 5},
    {"LMPBase", &tLMPBase, 1, 8},        {"SeeCapDepth", &tSeeCapDepth, 4, 9},
    {"LMRHistDiv", &tLMRHistDiv, 3000, 16000}, {"LMRHistClamp", &tLMRHistClamp, 1, 4},
    {"LMRPvRed", &tLMRPvRed, 0, 2},      {"LMRImpRed", &tLMRImpRed, 0, 2},
    {"LMRKillRed", &tLMRKillRed, 0, 2},
};

const int MAX_THREADS = 16;
struct ThreadData {
    int killers[MAX_PLY][2];
    int history[2][64][64];
    int counterMove[2][64][64];
    int16_t contHist[12][64][12][64];
    int ssMove[MAX_PLY];
    int ssEval[MAX_PLY];
    int pvTable[MAX_PLY][MAX_PLY];
    int pvLen[MAX_PLY];
    U64 keyHist[1024 + MAX_PLY + 8];
    int keyHistLen = 0;
    long long nodes = 0;
    int corrHist[2][CORR_SIZE];         // [stm][pawnKey] eval correction, grain 256
    int matCorr[2][CORR_SIZE];          // [stm][material signature] eval correction
    int contCorr[12][64];               // [piece(prev)][to(prev)] eval correction
    int16_t contCorrPair[12][64][12][64]; // [prev2 piece][to][prev piece][to] eval correction
#if FEAT_NPCORR
    int npCorr[2][2][CORR_SIZE];        // [stm][key color][npKey] eval correction
#endif
#if FEAT_KPCORR
    int kpCorr[2][CORR_SIZE];           // [stm][pawnKey ^ king squares] eval correction
#endif
#if FEAT_PIECETO
    int16_t pieceToHist[12][64];        // ordering-only piece-destination history
#endif
    int capHist[12][64][6];             // [attacker][to][victimType]
    int16_t contHist2[12][64][12][64];  // 2-ply continuation (separate from 1-ply)
    long long rootEffort[4096];         // nodes spent per root move (from|to<<6)
    Accumulator accStack[MAX_PLY + 4];
    Accumulator *accPtr[MAX_PLY + 4];   // null moves alias the parent accumulator
#if FEAT_LAZYACC
    AccDelta pendDelta[MAX_PLY + 4];    // recorded update from the source ply
    int16_t pendSrc[MAX_PLY + 4];       // ply whose accumulator feeds this one
    bool accReady[MAX_PLY + 4];         // accumulator materialized
    bool accAlias[MAX_PLY + 4];         // accPtr aliases the source (null move)
#endif
};
ThreadData tds[MAX_THREADS];
int optThreads = 1;
U64 gameHist[1024];      // zobrist keys of the actual game
int gameHistLen = 0;
int lmrTable[64][64];

void initLMR() {
    for (int d = 1; d < 64; d++)
        for (int m = 1; m < 64; m++)
            lmrTable[d][m] = (int)(0.5 + log(d) * log(m) / (tLMRDiv / 100.0));
}

static inline long long elapsedMs() {
    return chrono::duration_cast<chrono::milliseconds>(
        chrono::steady_clock::now() - si.start).count();
}

static inline void checkTime(ThreadData &td) {
    if (si.timed && si.hardLimit && elapsedMs() >= si.hardLimit) si.stop = true;
    if (si.nodeLimit && td.nodes >= si.nodeLimit) si.stop = true;
}

bool isRepetitionOrFifty(const Position &p, int plyFromRoot, ThreadData &td) {
    if (p.fifty >= 100) return true;
    int cnt = 0;
    int limit = td.keyHistLen - 1 - p.fifty;
    if (limit < 0) limit = 0;
    for (int i = td.keyHistLen - 3; i >= limit; i -= 2) {
        if (td.keyHist[i] == p.key) {
            cnt++;
            if (i >= td.keyHistLen - plyFromRoot) return true; // rep within search tree
            if (cnt >= 2) return true;                          // threefold w/ game history
        }
    }
    return false;
}

const int mvvVictim[6] = {100, 300, 310, 500, 900, 10000};

static inline int capVictim(const Position &p, int m) {   // call before the move is made
    if (M_EP(m)) return PAWN;
    int vp = p.board[M_TO(m)];
    return vp == NO_PIECE ? PAWN : pieceType(vp);
}
static inline void capHistUpdate(ThreadData &td, const Position &p, int m, int delta) {
    int &ch = td.capHist[M_PIECE(m)][M_TO(m)][capVictim(p, m)];
    ch += delta - ch * abs(delta) / 16384;
}

#if FEAT_LAZYACC
// Materialize a lazily recorded accumulator chain. Recursion terminates at
// the root, which think() marks ready after its full refresh; nodes cut off
// by the TT, repetitions, or cached stand-pats never pay for this.
static void ensureAcc(ThreadData &td, int ply) {
    if (td.accReady[ply]) return;
    ensureAcc(td, td.pendSrc[ply]);
    if (!td.accAlias[ply])
        nnApplyDelta(td.pendDelta[ply], *td.accPtr[td.pendSrc[ply]], td.accStack[ply]);
    td.accReady[ply] = true;
}
#endif

#if FEAT_LAZYACC == 2
// differential harness: after any lazy materialization feeding an eval,
// compare against a from-scratch refresh of the position
#define LAZYACC_VERIFY(p, td, ply) do { \
    static Accumulator lzChk; \
    nnRefresh(lzChk, (p)); \
    if (memcmp(lzChk.v[0], (td).accPtr[(ply)]->v[0], sizeof(int16_t) * nnHidden) \
     || memcmp(lzChk.v[1], (td).accPtr[(ply)]->v[1], sizeof(int16_t) * nnHidden)) { \
        static long long lzBad = 0; \
        if (lzBad++ < 20) printf("info string LAZYACC MISMATCH ply %d\n", (ply)); \
    } } while (0)
#else
#define LAZYACC_VERIFY(p, td, ply) ((void)0)
#endif

#if FEAT_SEEGE
// Threshold form of see(): the swap value always lies in
// [gain0 - seeVal[attacker], gain0], so most calls resolve without the
// attack-set computation and swap loop. Exactly equivalent to
// see(p, m) >= threshold - verified by bench node-count identity.
static inline bool seeGe(const Position &p, int m, int threshold) {
    int gain0;
    if (M_EP(m)) {
        gain0 = seeVal[PAWN];
    } else {
        int vp = p.pieceOn(M_TO(m));
        gain0 = vp == NO_PIECE ? 0 : seeVal[pieceType(vp)];
    }
    if (gain0 < threshold) return false;
    if (gain0 - seeVal[pieceType(M_PIECE(m))] >= threshold) return true;
    return see(p, m) >= threshold;
}
#if FEAT_SEEGE == 2
// differential test mode: run both paths, report disagreements, act on slow
static inline bool seeLtChk(const Position &p, int m, int t) {
    bool fast = !seeGe(p, m, t);
    bool slow = see(p, m) < t;
    if (fast != slow) {
        static long long bad = 0;
        if (bad++ < 20)
            printf("info string SEEGE MISMATCH t=%d see=%d ep=%d cap=%d promo=%d pc=%d victim=%d\n",
                   t, see(p, m), (int)!!M_EP(m), (int)!!M_CAP(m), (int)!!M_PROMO(m),
                   pieceType(M_PIECE(m)),
                   M_EP(m) ? PAWN : (p.pieceOn(M_TO(m)) == NO_PIECE ? -1
                                     : pieceType(p.pieceOn(M_TO(m)))));
    }
    return slow;
}
#define SEE_LT(p, m, t) (seeLtChk((p), (m), (t)))
#else
#define SEE_LT(p, m, t) (!seeGe((p), (m), (t)))
#endif
#else
#define SEE_LT(p, m, t) (see((p), (m)) < (t))
#endif

void scoreMoves(const Position &p, MoveList &ml, int ttMove, int ply, int cm, int prev, int prev2, ThreadData &td) {
    for (int i = 0; i < ml.count; i++) {
        int m = ml.moves[i];
        if (m == ttMove) { ml.scores[i] = 2000000; continue; }
        if (M_CAP(m)) {
            int victim = PAWN;
            if (!M_EP(m)) {
                int vp = p.pieceOn(M_TO(m));
                if (vp != NO_PIECE) victim = pieceType(vp);
            }
            int base = mvvVictim[victim] * 10 - pieceType(M_PIECE(m));
#if FEAT_CAPHIST
            base += td.capHist[M_PIECE(m)][M_TO(m)][victim] / 64;
#endif
            if (M_PROMO(m)) base += 50000;
            // Good captures precede quiets.  The candidate band is below the
            // full reachable quiet-history range, so losing captures come last.
            int band = 1000000;
            if (SEE_LT(p, m, 0)) {
#if FEAT_BADCAPLAST
                band = BADCAP_SCORE_BAND;
#else
                band = 50000;
#endif
#if FEAT_BADCAPCHECK
                // a losing capture that gives check is the classic sacrifice
                // shape; order it with the killers instead of dead last
                if (givesDirectCheck(p, m)) band = 860000;
#endif
            }
            ml.scores[i] = band + base;
        } else if (M_PROMO(m)) {
            ml.scores[i] = 950000 + pieceType(M_PROMO(m));
        } else if (m == td.killers[ply][0]) ml.scores[i] = 900000;
        else if (m == td.killers[ply][1]) ml.scores[i] = 870000;
        else if (m == cm) ml.scores[i] = 850000;
        else {
            int s = td.history[p.side][M_FROM(m)][M_TO(m)];
            if (prev) s += 2 * td.contHist[M_PIECE(prev)][M_TO(prev)][M_PIECE(m)][M_TO(m)];
#if FEAT_CH2
            if (prev2) s += td.contHist2[M_PIECE(prev2)][M_TO(prev2)][M_PIECE(m)][M_TO(m)];
#endif
#if FEAT_PIECETO
            // generalizes across from-squares and predecessors; ordering only,
            // never added to the hScore that pruning and LMR consume
            s += td.pieceToHist[M_PIECE(m)][M_TO(m)];
#endif
            ml.scores[i] = s;
        }
    }
}

static inline void pickMove(MoveList &ml, int idx) {
    int best = idx;
    for (int i = idx + 1; i < ml.count; i++)
        if (ml.scores[i] > ml.scores[best]) best = i;
    swap(ml.moves[idx], ml.moves[best]);
    swap(ml.scores[idx], ml.scores[best]);
}

int qsearch(Position &p, int alpha, int beta, int ply, ThreadData &td, int qsd) {
#if FEAT_QREP
    if (ply > 0 && isRepetitionOrFifty(p, ply, td)) return 0;
#endif
    td.nodes++;
    if ((td.nodes & 2047) == 0) checkTime(td);
    if (si.stop) return 0;
#if FEAT_LAZYACC
    if (ply >= MAX_PLY - 1) {
        if (nnueLoaded) ensureAcc(td, ply);
        return evaluate(p, td.accPtr[ply]);
    }
#else
    if (ply >= MAX_PLY - 1) return evaluate(p, td.accPtr[ply]);
#endif

    bool pvNode = beta - alpha > 1;
    int origAlpha = alpha;
    bool check = inCheck(p);
    // qsearch "depth": 0 when this node searches quiet checks or evasions, -1 otherwise
    int qsDepth = (check || qsd >= 0) ? 0 : -1;

#if FEAT_QTT
    // TT probe
#if FEAT_TTBUCKET
    bool ttHit;
    TTEntry &e = *ttProbeEntry(p.key, ttHit);
#else
    TTEntry &e = tt[p.key & ttMask];
    bool ttHit = TT_HIT(e, p.key);
#endif
    int ttMove = 0;
    if (ttHit) {
#if FEAT_TTAGEHIT
        e.age = ttAge;
#endif
        ttMove = e.move;
        int ttScore = e.score;
        if (ttScore > MATE - MAX_PLY) ttScore -= ply;
        else if (ttScore < -MATE + MAX_PLY) ttScore += ply;
        if (!pvNode && e.depth >= qsDepth) {
            if (e.flag == TT_EXACT) return ttScore;
            if (e.flag == TT_BETA && ttScore >= beta) return ttScore;
            if (e.flag == TT_ALPHA && ttScore <= alpha) return ttScore;
        }
    }
#else
    const int ttMove = 0;
#endif

    int best, rawEval = EVAL_NONE;
    if (!check) {
#if FEAT_QTT && FEAT_LAZYACC
        if (ttHit && e.eval != EVAL_NONE) {
            rawEval = e.eval;
        } else {
            if (nnueLoaded) { ensureAcc(td, ply); LAZYACC_VERIFY(p, td, ply); }
            rawEval = evaluate(p, td.accPtr[ply]);
        }
#elif FEAT_QTT
        rawEval = (ttHit && e.eval != EVAL_NONE) ? e.eval : evaluate(p, td.accPtr[ply]);
#else
        rawEval = evaluate(p, td.accPtr[ply]);
#endif
        best = rawEval;
#if FEAT_CORR
        {
            int corr = (td.corrHist[p.side][p.pawnKey & (CORR_SIZE - 1)] * tCorrPawnW) / 256;
#if FEAT_MATCORR
            corr += td.matCorr[p.side][matIndex(p)];
#endif
#if FEAT_NPCORR
            corr += ((td.npCorr[p.side][WHITE][p.npKey[WHITE] & (CORR_SIZE - 1)]
                    + td.npCorr[p.side][BLACK][p.npKey[BLACK] & (CORR_SIZE - 1)]) * tCorrNpW) / 512;
#endif
#if FEAT_KPCORR
            corr += (td.kpCorr[p.side][(p.pawnKey ^ zPiece[WK][lsb(p.bb[WK])]
                                      ^ zPiece[BK][lsb(p.bb[BK])]) & (CORR_SIZE - 1)]
                     * tCorrKpW) / 256;
#endif
#if FEAT_QCONTCORR
            if (ply > 0 && td.ssMove[ply - 1])
                corr += (td.contCorr[M_PIECE(td.ssMove[ply - 1])][M_TO(td.ssMove[ply - 1])]
                         * tCorrContW) / 256;
#endif
            best += corr / CORR_GRAIN;
        }
#endif
        best = dampForFiftyMoveRule(best, p);
#if FEAT_QTT
        // refine stand-pat with the TT score when its bound allows
        if (ttHit) {
            int ttScore = e.score;
            if (ttScore > MATE - MAX_PLY) ttScore -= ply;
            else if (ttScore < -MATE + MAX_PLY) ttScore += ply;
            if (e.flag == TT_EXACT
                || (e.flag == TT_BETA && ttScore > best)
                || (e.flag == TT_ALPHA && ttScore < best))
                best = ttScore;
        }
#endif
        if (best >= beta) {
#if FEAT_QTT && FEAT_QSTANDPAT_TT
            // Stand-pat cutoffs are common transpositions.  Cache the lower
            // bound without replacing a deeper entry in the same-age table.
#if FEAT_TTBUCKET
            TTEntry &w = *ttStoreSlot(p.key);
#else
            TTEntry &w = e;
#endif
            if (w.flag == TT_NONE || w.age != ttAge || w.depth <= qsDepth) {
                int storeScore = best;
                if (storeScore > MATE - MAX_PLY) storeScore += ply;
                else if (storeScore < -MATE + MAX_PLY) storeScore -= ply;
                int savedMove = TT_HIT(w, p.key) ? w.move : 0;
                w.key = TT_KEYV(p.key);
                w.move = savedMove;
                w.score = (int16_t)storeScore;
                w.eval = (int16_t)rawEval;
                w.depth = (int8_t)qsDepth;
                w.flag = TT_BETA;
                w.age = ttAge;
            }
#endif
            return best;
        }
        if (best > alpha) alpha = best;
    } else best = -INF;

    bool withQuietChecks = qsd >= 0 && !check;
    MoveList ml;
    genMoves(p, ml, check ? false : !withQuietChecks);
#if FEAT_QEVASION_CONTEXT
    int qPrev = 0, qCm = 0;
    if (check && ply > 0) {
        qPrev = td.ssMove[ply - 1];
        if (qPrev)
            qCm = td.counterMove[p.side][M_FROM(qPrev)][M_TO(qPrev)];
    }
    scoreMoves(p, ml, ttMove, ply, qCm, qPrev, 0, td);
#else
    scoreMoves(p, ml, ttMove, ply, 0, 0, 0, td);
#endif

    int legal = 0, bestMove = 0;
    for (int i = 0; i < ml.count; i++) {
        pickMove(ml, i);
        int m = ml.moves[i];
        bool quietM = !M_CAP(m) && !M_PROMO(m);
        if (!check && quietM && m != ttMove && !givesDirectCheck(p, m)) continue;
#if FEAT_QSEVADE
        // a quiet evasion that hangs the moving piece rarely holds; only
        // skip once a legal reply exists and we are not facing forced mate
        if (check && legal > 0 && best > -MATE + MAX_PLY && quietM
            && SEE_LT(p, m, 0))
            continue;
#endif
#if FEAT_QCHECKSEE
        // Keep the TT move, but skip a quiet check that immediately loses
        // material; the opponent's forced reply normally refutes it.
        if (!check && quietM && m != ttMove && SEE_LT(p, m, 0)) continue;
#endif
        if (!check && M_CAP(m)) {
            // delta pruning
            if (!M_PROMO(m)) {
                int victim = PAWN;
                if (!M_EP(m)) {
                    int vp = p.pieceOn(M_TO(m));
                    if (vp != NO_PIECE) victim = pieceType(vp);
                }
                if (best + mgValue[victim] + tDeltaMargin < alpha) continue;
            }
            // skip losing captures
            if (SEE_LT(p, m, 0)) continue;
        }
        Position next = p;
        if (!makeMove(next, m)) continue;
        legal++;
#if FEAT_QEVASION_CONTEXT || FEAT_QCONTCORR
        td.ssMove[ply] = m;
#endif
        ttPrefetch(next.key);
        if (nnueLoaded) {
#if FEAT_LAZYACC
            nnComputeDelta(p, m, td.pendDelta[ply + 1]);
            td.pendSrc[ply + 1] = (int16_t)ply;
            td.accReady[ply + 1] = false;
            td.accAlias[ply + 1] = false;
            td.accPtr[ply + 1] = &td.accStack[ply + 1];
#else
            nnApply(p, m, *td.accPtr[ply], td.accStack[ply + 1]);
            td.accPtr[ply + 1] = &td.accStack[ply + 1];
#endif
        }
#if FEAT_QREP
        td.keyHist[td.keyHistLen++] = next.key;
#endif
#if FEAT_QSCHAIN
        // an in-check node's forced evasions do not consume the quiet-check
        // budget, so check -> evasion -> second check stays visible
        int score = -qsearch(next, -beta, -alpha, ply + 1, td, check ? qsd : qsd - 1);
#else
        int score = -qsearch(next, -beta, -alpha, ply + 1, td, qsd - 1);
#endif
#if FEAT_QREP
        td.keyHistLen--;
#endif
        if (si.stop) return 0;
        if (score > best) {
            best = score;
            if (score > alpha) {
                alpha = score;
                bestMove = m;
                if (alpha >= beta) break;
            }
        }
    }
    if (check && legal == 0) return -MATE + ply;

#if FEAT_QTT
    // TT store: never clobber deeper entries, even for the same key
    if (!si.stop) {
#if FEAT_TTBUCKET
        TTEntry &w = *ttStoreSlot(p.key);
#else
        TTEntry &w = e;
#endif
        if (w.flag == TT_NONE || w.age != ttAge || w.depth <= qsDepth) {
            int storeScore = best;
            if (storeScore > MATE - MAX_PLY) storeScore += ply;
            else if (storeScore < -MATE + MAX_PLY) storeScore -= ply;
            w.key = TT_KEYV(p.key);
            w.move = bestMove;
            w.score = (int16_t)storeScore;
            w.eval = (int16_t)rawEval;   // EVAL_NONE when in check
            w.depth = (int8_t)qsDepth;
            w.flag = best <= origAlpha ? TT_ALPHA : (best >= beta ? TT_BETA : TT_EXACT);
            w.age = ttAge;
        }
    }
#endif
    return best;
}

int negamax(Position &p, int depth, int alpha, int beta, int ply, bool nullOk, int excluded, ThreadData &td, bool cutnode) {
    td.pvLen[ply] = ply;
    bool pvNode = beta - alpha > 1;
    bool rootNode = ply == 0;
#if FEAT_KRESET
    if (ply + 1 < MAX_PLY) td.killers[ply + 1][0] = td.killers[ply + 1][1] = 0;
#endif

    if (!rootNode) {
        if (isRepetitionOrFifty(p, ply, td)) return 0;
    #if FEAT_LAZYACC
    if (ply >= MAX_PLY - 1) {
        if (nnueLoaded) ensureAcc(td, ply);
        return evaluate(p, td.accPtr[ply]);
    }
#else
    if (ply >= MAX_PLY - 1) return evaluate(p, td.accPtr[ply]);
#endif
        // mate distance pruning
        alpha = max(alpha, -MATE + ply);
        beta = min(beta, MATE - ply - 1);
        if (alpha >= beta) return alpha;
    }

    bool check = inCheck(p);
    if (check) depth++;
#if FEAT_QSCHAIN
    // budget of quiet-check WAVES: 1 allows check -> forced evasion -> check
    if (depth <= 0) return qsearch(p, alpha, beta, ply, td, 1);
#else
    if (depth <= 0) return qsearch(p, alpha, beta, ply, td, 0);
#endif

    td.nodes++;
    if ((td.nodes & 2047) == 0) checkTime(td);
    if (si.stop) return 0;

    // TT probe
#if FEAT_TTBUCKET
    bool ttHit;
    TTEntry &e = *ttProbeEntry(p.key, ttHit);
#else
    TTEntry &e = tt[p.key & ttMask];
    bool ttHit = TT_HIT(e, p.key);
#endif
    int ttMove = 0, ttScore = -INF, ttDepth = -1, ttFlag = TT_NONE;
    if (ttHit) {
#if FEAT_TTAGEHIT
        // an entry probed this search is a protected incumbent: without the
        // refresh, last move's deep PV entries have zero depth protection
        e.age = ttAge;
#endif
        ttMove = e.move;
        ttDepth = e.depth;
        ttFlag = e.flag;
        ttScore = e.score;
        if (ttScore > MATE - MAX_PLY) ttScore -= ply;
        else if (ttScore < -MATE + MAX_PLY) ttScore += ply;
        if (!pvNode && !excluded && ttDepth >= depth) {
            if (ttFlag == TT_EXACT) return ttScore;
            if (ttFlag == TT_BETA && ttScore >= beta) return ttScore;
            if (ttFlag == TT_ALPHA && ttScore <= alpha) return ttScore;
        }
    }

    // internal iterative reduction: no TT move -> shallower search
    if (!ttMove && depth >= tIIRDepth) depth--;

#if FEAT_LAZYACC
    int rawEval;
    if (ttHit && e.eval != EVAL_NONE) {
        rawEval = e.eval;   // cached: the accumulator stays unmaterialized
    } else {
        if (nnueLoaded) { ensureAcc(td, ply); LAZYACC_VERIFY(p, td, ply); }
        rawEval = evaluate(p, td.accPtr[ply]);
    }
#else
    int rawEval = (ttHit && e.eval != EVAL_NONE) ? e.eval
                                                 : evaluate(p, td.accPtr[ply]);
#endif
    int staticEval = rawEval;
#if FEAT_CORR
    if (!check) {
        int corr = (td.corrHist[p.side][p.pawnKey & (CORR_SIZE - 1)] * tCorrPawnW) / 256;
#if FEAT_MATCORR
        corr += td.matCorr[p.side][matIndex(p)];
#endif
#if FEAT_CONTCORR
        // eval error correlated with the move that just landed
        if (ply > 0 && td.ssMove[ply - 1])
            corr += (td.contCorr[M_PIECE(td.ssMove[ply - 1])][M_TO(td.ssMove[ply - 1])]
                     * tCorrContW) / 256;
#endif
#if FEAT_CONTCORR2
        if (ply > 1 && td.ssMove[ply - 1] && td.ssMove[ply - 2])
            corr += td.contCorrPair[M_PIECE(td.ssMove[ply - 2])][M_TO(td.ssMove[ply - 2])]
                                   [M_PIECE(td.ssMove[ply - 1])][M_TO(td.ssMove[ply - 1])];
#endif
#if FEAT_NPCORR
        corr += ((td.npCorr[p.side][WHITE][p.npKey[WHITE] & (CORR_SIZE - 1)]
                + td.npCorr[p.side][BLACK][p.npKey[BLACK] & (CORR_SIZE - 1)]) * tCorrNpW) / 512;
#endif
#if FEAT_KPCORR
        corr += (td.kpCorr[p.side][(p.pawnKey ^ zPiece[WK][lsb(p.bb[WK])]
                                  ^ zPiece[BK][lsb(p.bb[BK])]) & (CORR_SIZE - 1)]
                 * tCorrKpW) / 256;
#endif
        staticEval += corr / CORR_GRAIN;
        if (staticEval > MATE - MAX_PLY - 1) staticEval = MATE - MAX_PLY - 1;
        if (staticEval < -MATE + MAX_PLY + 1) staticEval = -MATE + MAX_PLY + 1;
    }
#endif
    int correctionEval = staticEval;
    staticEval = dampForFiftyMoveRule(staticEval, p);
#if FEAT_THREATLMR
    bool nullThreat = false;
#endif
#if FEAT_IMPROVING_FALLBACK
    td.ssEval[ply] = check ? EVAL_NONE : staticEval;
    bool improving = false;
    if (!check) {
        if (ply >= 2 && td.ssEval[ply - 2] != EVAL_NONE)
            improving = staticEval > td.ssEval[ply - 2];
        else if (ply >= 4 && td.ssEval[ply - 4] != EVAL_NONE)
            improving = staticEval > td.ssEval[ply - 4];
    }
#else
    td.ssEval[ply] = staticEval;
    bool improving = !check && ply >= 2 && staticEval > td.ssEval[ply - 2];
#endif

    // refine eval with TT score when its bound allows (better pruning decisions)
    int eval = staticEval;
#if FEAT_REFINE
    if (ttHit && abs(ttScore) < MATE - MAX_PLY
        && (ttFlag == TT_EXACT
            || (ttFlag == TT_BETA && ttScore > eval)
            || (ttFlag == TT_ALPHA && ttScore < eval)))
        eval = ttScore;
#endif

    // razoring
    if (!pvNode && !check && !excluded && depth <= tRazorDepth
        && eval + tRazorMargin * depth < alpha) {
        int q = qsearch(p, alpha, beta, ply, td, -1);
        if (q < alpha) return q;
    }

    // reverse futility pruning
    if (!pvNode && !check && !excluded && depth <= tRFPDepth
        && eval - tRFPMargin * (depth - (improving ? 1 : 0)) >= beta
        && abs(beta) < MATE - MAX_PLY)
        return eval;

    // null move pruning
    U64 bigPieces = p.side == WHITE
        ? p.bb[WN] | p.bb[WB] | p.bb[WR] | p.bb[WQ]
        : p.bb[BN] | p.bb[BB_] | p.bb[BR] | p.bb[BQ];
    if (!pvNode && !check && nullOk && !excluded && depth >= 3
        && eval >= beta && bigPieces) {
#if FEAT_NMPINPLACE
        const U64 nullSavedKey = p.key;
        const int nullSavedSide = p.side;
        const int nullSavedEp = p.ep;
        const int nullSavedFifty = p.fifty;
        makeNull(p);
        Position &next = p;
#else
        Position next = p;
        makeNull(next);
#endif
        td.ssMove[ply] = 0;
        td.accPtr[ply + 1] = td.accPtr[ply];
#if FEAT_LAZYACC
        // the alias is ready exactly when its source is; materialization of
        // the child resolves the parent chain first, then adopts the alias
        td.pendSrc[ply + 1] = (int16_t)ply;
        td.accReady[ply + 1] = td.accReady[ply];
        td.accAlias[ply + 1] = true;
#endif
        td.keyHist[td.keyHistLen++] = next.key;
        int R = tNullBase + depth / tNullDDiv + min((eval - beta) / tNullEvalDiv, 2);
        int score = -negamax(next, depth - 1 - R, -beta, -beta + 1, ply + 1, false, 0, td, !cutnode);
#if FEAT_NMPINPLACE
        p.key = nullSavedKey;
        p.side = nullSavedSide;
        p.ep = nullSavedEp;
        p.fifty = nullSavedFifty;
#endif
        td.keyHistLen--;
        if (si.stop) return 0;
        if (score >= beta) {
            if (score > MATE - MAX_PLY) score = beta;
#if FEAT_NMPVERIFY
            // at high depth, confirm the fail-high with a reduced real search
            // (guards against zugzwang); nullOk=false prevents recursion
            if (depth >= 12) {
                int v = negamax(p, depth - 1 - R, beta - 1, beta, ply, false, 0, td, cutnode);
                if (si.stop) return 0;
                if (v >= beta) return score;
            } else
#endif
            return score;
        }
#if FEAT_THREATLMR
        // passing loses badly: the opponent has a concrete threat here
        nullThreat = (score <= alpha - 120) || (score < -MATE + MAX_PLY);
#endif
    }

#if FEAT_PROBCUT
    // ProbCut: a capture that a reduced search confirms well above beta
    // almost certainly makes this node fail high
    int pcBeta = beta + 180;
    if (!pvNode && !check && !excluded && depth >= 5
        && abs(beta) < MATE - MAX_PLY
        && !(ttHit && ttDepth >= depth - 3 && ttScore < pcBeta)) {
        MoveList pml;
        genMoves(p, pml, true);
        scoreMoves(p, pml, ttMove, ply, 0, 0, 0, td);
        for (int i = 0; i < pml.count; i++) {
            pickMove(pml, i);
            int m = pml.moves[i];
            if (see(p, m) < pcBeta - staticEval) continue;
            Position next = p;
            if (!makeMove(next, m)) continue;
            ttPrefetch(next.key);
            if (nnueLoaded) {
                nnApply(p, m, *td.accPtr[ply], td.accStack[ply + 1]);
                td.accPtr[ply + 1] = &td.accStack[ply + 1];
            }
            td.ssMove[ply] = m;
            td.keyHist[td.keyHistLen++] = next.key;
            int v = -qsearch(next, -pcBeta, -pcBeta + 1, ply + 1, td, -1);
            if (v >= pcBeta && depth >= 7)
                v = -negamax(next, depth - 4, -pcBeta, -pcBeta + 1, ply + 1, true, 0, td, !cutnode);
            td.keyHistLen--;
            if (si.stop) return 0;
            if (v >= pcBeta) return v;
        }
    }
#endif

    int prev = ply > 0 ? td.ssMove[ply - 1] : 0;
    int prev2 = ply > 1 ? td.ssMove[ply - 2] : 0;
    int cm = 0;
    if (prev) cm = td.counterMove[p.side][M_FROM(prev)][M_TO(prev)];

    MoveList ml;
    genMoves(p, ml, false);
    scoreMoves(p, ml, ttMove, ply, cm, prev, prev2, td);
#if FEAT_ONEREPLY
    // a check with exactly one legal answer is a forced sequence: extend it
    // (depth-capped to twice the iteration depth to bound forced chains)
    bool oneReply = false;
    if (check && ply < 2 * si.rootDepth) {
        int nl = 0;
        for (int i = 0; i < ml.count && nl < 2; i++) {
            Position q = p;
            if (makeMove(q, ml.moves[i])) nl++;
        }
        oneReply = nl == 1;
    }
#endif
#if FEAT_ROOTEFFORT
    // Subtree size from earlier iterations is the best available proxy for
    // "hardest refutation" among root moves; rank non-TT roots by it.
    if (rootNode && depth > 1) {
        for (int i = 0; i < ml.count; i++) {
            if (ml.moves[i] == ttMove) continue;
            long long eff = td.rootEffort[ml.moves[i] & 4095];
            if (eff > 0)
                ml.scores[i] = 1000000 + (int)min(eff >> 3, 900000LL);
        }
    }
#endif

    int bestScore = -INF, bestMove = 0, legal = 0;
    int quietsTried[64], nQuiets = 0;
    int capsTried[48], nCaps = 0;
    int origAlpha = alpha;
    bool futile = !pvNode && !check && depth <= tFutDepth
                  && eval + tFutBase + tFutSlope * depth <= alpha;
    int lmpLimit = (tLMPBase + depth * depth) / (improving ? 1 : 2);

    for (int i = 0; i < ml.count; i++) {
        pickMove(ml, i);
        int m = ml.moves[i];
        if (m == excluded) continue;
        long long moveNodes0 = td.nodes;
        bool quiet = !M_CAP(m) && !M_PROMO(m);
#if FEAT_BADCAPLMR || FEAT_CAPHIST_LMR
        bool badCapture = M_CAP(m) && see(p, m) < 0;
#endif

        int hScore = 0;
        if (quiet) {
            hScore = td.history[p.side][M_FROM(m)][M_TO(m)];
            if (prev) hScore += 2 * td.contHist[M_PIECE(prev)][M_TO(prev)][M_PIECE(m)][M_TO(m)];
#if FEAT_CH2
            if (prev2) hScore += td.contHist2[M_PIECE(prev2)][M_TO(prev2)][M_PIECE(m)][M_TO(m)];
#endif
        }
        if (legal > 0 && bestScore > -MATE + MAX_PLY) {
#if FEAT_QUIET_CHECK_GUARD
            // Compute this only where at least one guarded pruner can fire.
            // The depth bound must track quiet-SEE pruning's ceiling: the guard
            // substitutes for its old inline check exemption (QUIETSEE-01/02).
            // GUARDDEPTH widens it so LMP exempts checks over its whole range.
            const bool directQuietCheck = quiet && !check
#if FEAT_GUARDDEPTH
                && (futile || (!pvNode && depth <= LMP_MAX_DEPTH))
#else
                && (futile || (!pvNode && depth <= QUIET_SEE_MAX_DEPTH))
#endif
#if FEAT_DISCOVERED_CHECK_GUARD
                && givesQuietOrDiscoveredCheck(p, m);
#else
                && givesDirectCheck(p, m);
#endif
#endif
            // futility pruning of quiets
            if (futile && quiet
#if FEAT_QUIET_CHECK_GUARD
                && !directQuietCheck
#endif
            ) continue;
            // history pruning: quiets with awful history at shallow depth
            if (!pvNode && !check && quiet && depth <= 4 && hScore < -tHistPruneM * depth
#if FEAT_QUIET_CHECK_GUARD
                && !directQuietCheck
#endif
            )
                continue;
            // late move pruning
            if (!pvNode && !check && quiet && depth <= LMP_MAX_DEPTH && legal > lmpLimit
#if FEAT_QUIET_CHECK_GUARD
                && !directQuietCheck
#endif
            )
                continue;
#if FEAT_QUIET_SEE
            // A late quiet that immediately loses substantially more than the
            // remaining depth can repair is unlikely to raise alpha.  Preserve
            // forcing checks and established ordering heuristics.  The final
            // enabled depth can use a more conservative threshold without
            // changing the already-tested shallower behavior.
            const int quietSeeMargin = depth == QUIET_SEE_MAX_DEPTH
                ? QUIET_SEE_FRONTIER_MARGIN : QUIET_SEE_MARGIN;
            if (!pvNode && !check && quiet && depth <= QUIET_SEE_MAX_DEPTH
                && m != td.killers[ply][0] && m != td.killers[ply][1] && m != cm
#if FEAT_QUIET_CHECK_GUARD
                && !directQuietCheck
#else
                && !givesDirectCheck(p, m)
#endif
                && SEE_LT(p, m, -quietSeeMargin * depth))
                continue;
#endif
            // SEE pruning of bad captures at shallow depth
            if (!pvNode && !check && depth <= tSeeCapDepth && M_CAP(m)
                && SEE_LT(p, m, -tSeePruneM * depth))
                continue;
#if FEAT_CAPFUT
            // even winning the victim outright cannot lift eval to alpha
            if (!pvNode && !check && M_CAP(m) && !M_PROMO(m) && depth <= 6
                && eval + mvvVictim[capVictim(p, m)] + tDeltaMargin
                   + tFutSlope * depth <= alpha)
                continue;
#endif
        }

        // singular extension: is the TT move much better than everything else?
        int ext = 0;
        if (!rootNode && !excluded && depth >= tSingDepth && m == ttMove
            && ttDepth >= depth - tSingTTOff && ttFlag != TT_ALPHA
            && abs(ttScore) < MATE - MAX_PLY) {
            int sBeta = ttScore - tSingBetaM * depth;
            int sScore = negamax(p, (depth - 1) / 2, sBeta - 1, sBeta, ply, false, m, td, cutnode);
            if (si.stop) return 0;
            if (sScore < sBeta) {
                ext = 1;                                       // singular -> extend
                if (!pvNode && sScore < sBeta - tDblExtM) ext = 2;   // strongly singular
            }
            else if (sBeta >= beta) return sBeta; // multi-cut
        }
#if FEAT_ONEREPLY
        if (oneReply && ext < 2) ext++;
#endif

        Position next = p;
        if (!makeMove(next, m)) continue;
        legal++;
        ttPrefetch(next.key);
        if (nnueLoaded) {
#if FEAT_LAZYACC
            nnComputeDelta(p, m, td.pendDelta[ply + 1]);
            td.pendSrc[ply + 1] = (int16_t)ply;
            td.accReady[ply + 1] = false;
            td.accAlias[ply + 1] = false;
            td.accPtr[ply + 1] = &td.accStack[ply + 1];
#else
            nnApply(p, m, *td.accPtr[ply], td.accStack[ply + 1]);
            td.accPtr[ply + 1] = &td.accStack[ply + 1];
#endif
        }
        if (quiet && nQuiets < 64) quietsTried[nQuiets++] = m;
        else if (M_CAP(m) && nCaps < 48) capsTried[nCaps++] = m;
        td.ssMove[ply] = m;
        td.keyHist[td.keyHistLen++] = next.key;

        int nd = depth - 1 + ext;
        int score;
        if (legal == 1) {
            score = -negamax(next, nd, -beta, -alpha, ply + 1, true, 0, td, false);
        } else {
            int R = 0;
            bool reduceQuiet = depth >= 3 && quiet && !check && legal > 3;
#if FEAT_CAPHIST_LMR
            bool reduceBadCapture = !pvNode && !check && !excluded && badCapture
                                 && !M_PROMO(m) && depth >= 5 && legal > 4
                                 && !inCheck(next)
                                 && td.capHist[M_PIECE(m)][M_TO(m)][capVictim(p, m)] / 64 < 0;
#elif FEAT_BADCAPLMR
            bool reduceBadCapture = !pvNode && !check && badCapture
                                 && depth >= 5 && legal > 4;
#else
            bool reduceBadCapture = false;
#endif
            if (reduceQuiet || reduceBadCapture) {
                if (reduceBadCapture) {
                    R = 1;
                } else {
                R = lmrTable[min(depth, 63)][min(legal, 63)];
                if (pvNode) R -= tLMRPvRed;
                if (!improving) R += tLMRImpRed;
#if FEAT_TTCAPLMR
                // a capture hash move marks the node tactical: late quiets
                // are even less likely than usual to become best here
                if (ttMove && M_CAP(ttMove)) R++;
#endif
#if FEAT_CUTNODE
                if (cutnode) R++;
#endif
                if (m == td.killers[ply][0] || m == td.killers[ply][1] || m == cm) R -= tLMRKillRed;
#if FEAT_THREATLMR
                if (nullThreat) R--;
#endif
                int hAdj = hScore / tLMRHistDiv;
                if (hAdj > tLMRHistClamp) hAdj = tLMRHistClamp;
                if (hAdj < -tLMRHistClamp) hAdj = -tLMRHistClamp;
                R -= hAdj;
                }
                if (R < 0) R = 0;
                if (R > nd - 1) R = nd - 1;
            }
            score = -negamax(next, nd - R, -alpha - 1, -alpha, ply + 1, true, 0, td, true);
            if (score > alpha && R > 0)
                score = -negamax(next, nd, -alpha - 1, -alpha, ply + 1, true, 0, td, !cutnode);
            if (score > alpha && score < beta)
                score = -negamax(next, nd, -beta, -alpha, ply + 1, true, 0, td, false);
        }
        td.keyHistLen--;
        if (rootNode) td.rootEffort[m & 4095] += td.nodes - moveNodes0;
        if (si.stop) return 0;

        if (score > bestScore) {
            bestScore = score;
            bestMove = m;
            if (score > alpha) {
                alpha = score;
                // update PV
                td.pvTable[ply][ply] = m;
                for (int j = ply + 1; j < td.pvLen[ply + 1]; j++)
                    td.pvTable[ply][j] = td.pvTable[ply + 1][j];
                td.pvLen[ply] = td.pvLen[ply + 1];
                if (alpha >= beta) {
                    if (quiet) {
                        if (td.killers[ply][0] != m) {
                            td.killers[ply][1] = td.killers[ply][0];
                            td.killers[ply][0] = m;
                        }
                        if (prev) {
                            td.counterMove[p.side][M_FROM(prev)][M_TO(prev)] = m;
                            int16_t &c = td.contHist[M_PIECE(prev)][M_TO(prev)][M_PIECE(m)][M_TO(m)];
                            int nc = c + depth * depth;
                            c = nc > 16000 ? 16000 : (int16_t)nc;
                        }
#if FEAT_CH2
                        if (prev2) {
                            int16_t &c2 = td.contHist2[M_PIECE(prev2)][M_TO(prev2)][M_PIECE(m)][M_TO(m)];
                            int nc2 = c2 + depth * depth;
                            c2 = nc2 > 16000 ? 16000 : (int16_t)nc2;
                        }
#endif
                        int &h = td.history[p.side][M_FROM(m)][M_TO(m)];
                        h += depth * depth;
                        if (h > 800000) h /= 2;
#if FEAT_PIECETO
                        {
                            int16_t &pt = td.pieceToHist[M_PIECE(m)][M_TO(m)];
                            int npt = pt + depth * depth;
                            pt = npt > 16000 ? 16000 : (int16_t)npt;
                        }
#endif
                        // maluses: punish quiets that were tried and failed
                        for (int q = 0; q < nQuiets - 1; q++) {
                            int qm = quietsTried[q];
                            int &hq = td.history[p.side][M_FROM(qm)][M_TO(qm)];
                            hq -= depth * depth;
                            if (hq < -800000) hq /= 2;
#if FEAT_PIECETO
                            {
                                int16_t &ptq = td.pieceToHist[M_PIECE(qm)][M_TO(qm)];
                                int nptq = ptq - depth * depth;
                                ptq = nptq < -16000 ? -16000 : (int16_t)nptq;
                            }
#endif
                            if (prev) {
                                int16_t &cq = td.contHist[M_PIECE(prev)][M_TO(prev)][M_PIECE(qm)][M_TO(qm)];
                                int ncq = cq - depth * depth;
                                cq = ncq < -16000 ? -16000 : (int16_t)ncq;
                            }
#if FEAT_CH2
                            if (prev2) {
                                int16_t &cq2 = td.contHist2[M_PIECE(prev2)][M_TO(prev2)][M_PIECE(qm)][M_TO(qm)];
                                int ncq2 = cq2 - depth * depth;
                                cq2 = ncq2 < -16000 ? -16000 : (int16_t)ncq2;
                            }
#endif
                        }
                    }
#if FEAT_MALUSCAP
                    // a noisy cutoff still proves every tried quiet failed;
                    // without this they keep their inflated history (common
                    // since BADCAP-01 orders losing captures after quiets)
                    else {
                        for (int q = 0; q < nQuiets; q++) {
                            int qm = quietsTried[q];
                            int &hq = td.history[p.side][M_FROM(qm)][M_TO(qm)];
                            hq -= depth * depth;
                            if (hq < -800000) hq /= 2;
                            if (prev) {
                                int16_t &cq = td.contHist[M_PIECE(prev)][M_TO(prev)][M_PIECE(qm)][M_TO(qm)];
                                int ncq = cq - depth * depth;
                                cq = ncq < -16000 ? -16000 : (int16_t)ncq;
                            }
                        }
                    }
#endif
#if FEAT_CAPHIST
                    if (M_CAP(m)) {
                        int bonus = depth * depth;
                        capHistUpdate(td, p, m, bonus);
                        for (int q = 0; q < nCaps; q++)
                            if (capsTried[q] != m) capHistUpdate(td, p, capsTried[q], -bonus);
                    }
#endif
                    break;
                }
            }
        }
    }

    if (legal == 0) return excluded ? alpha : (check ? -MATE + ply : 0);

#if FEAT_CORR
    // teach the correction history the residual between search and static eval
    if (!excluded && !check && abs(bestScore) < MATE - MAX_PLY
        && (!bestMove || !M_CAP(bestMove))) {
        bool failHigh = bestScore >= beta, failLow = bestScore <= origAlpha;
        if (!(failHigh && bestScore <= staticEval) && !(failLow && bestScore >= staticEval)) {
            // Train correction histories against the undamped predictor so a
            // high halfmove clock cannot contaminate future low-clock nodes.
            int diff = (bestScore - correctionEval) * CORR_GRAIN;
#if FEAT_CORRCLAMP
            // a residual beyond the tables' representable range is tactical
            // noise, not correctable eval bias; cap its training influence
            if (diff > CORR_MAX) diff = CORR_MAX;
            if (diff < -CORR_MAX) diff = -CORR_MAX;
#endif
            int w = depth + 1 > 16 ? 16 : depth + 1;
#if FEAT_CORRBOUND
            // a fail-high proves only a lower bound on the residual (and a
            // fail-low only an upper one): let each bound pull an entry
            // exclusively in the direction it can prove
#define CORRB_OK(cur) (!(failHigh && diff <= (cur)) && !(failLow && diff >= (cur)))
#else
#define CORRB_OK(cur) 1
#endif
            int &ch = td.corrHist[p.side][p.pawnKey & (CORR_SIZE - 1)];
            if (CORRB_OK(ch)) {
                ch = (ch * (256 - w) + diff * w) / 256;
                if (ch > CORR_MAX) ch = CORR_MAX;
                if (ch < -CORR_MAX) ch = -CORR_MAX;
            }
#if FEAT_CONTCORR
            if (prev) {
                int &cc = td.contCorr[M_PIECE(prev)][M_TO(prev)];
                if (CORRB_OK(cc)) {
                    cc = (cc * (256 - w) + diff * w) / 256;
                    if (cc > CORR_MAX) cc = CORR_MAX;
                    if (cc < -CORR_MAX) cc = -CORR_MAX;
                }
            }
#endif
#if FEAT_CONTCORR2
            if (prev && prev2) {
                int16_t &cp = td.contCorrPair[M_PIECE(prev2)][M_TO(prev2)][M_PIECE(prev)][M_TO(prev)];
                int np = (cp * (256 - w) + diff * w) / 256;
                if (np > CORR_MAX) np = CORR_MAX;
                if (np < -CORR_MAX) np = -CORR_MAX;
                cp = (int16_t)np;
            }
#endif
#if FEAT_NPCORR
            for (int c = 0; c < 2; c++) {
                int &nh = td.npCorr[p.side][c][p.npKey[c] & (CORR_SIZE - 1)];
                if (!CORRB_OK(nh)) continue;
                nh = (nh * (256 - w) + diff * w) / 256;
                if (nh > CORR_MAX) nh = CORR_MAX;
                if (nh < -CORR_MAX) nh = -CORR_MAX;
            }
#endif
#if FEAT_KPCORR
            {
                int &kh = td.kpCorr[p.side][(p.pawnKey ^ zPiece[WK][lsb(p.bb[WK])]
                                           ^ zPiece[BK][lsb(p.bb[BK])]) & (CORR_SIZE - 1)];
                if (CORRB_OK(kh)) {
                    kh = (kh * (256 - w) + diff * w) / 256;
                    if (kh > CORR_MAX) kh = CORR_MAX;
                    if (kh < -CORR_MAX) kh = -CORR_MAX;
                }
            }
#endif
#undef CORRB_OK
#if FEAT_MATCORR
            int &mh = td.matCorr[p.side][matIndex(p)];
            mh = (mh * (256 - w) + diff * w) / 256;
            if (mh > CORR_MAX) mh = CORR_MAX;
            if (mh < -CORR_MAX) mh = -CORR_MAX;
#endif
        }
    }
#endif

    // TT store
    if (!excluded) {
#if FEAT_TTBUCKET
        TTEntry &w = *ttStoreSlot(p.key);
#else
        TTEntry &w = e;
#endif
#if FEAT_TTSTORE
        // same-key stores must be near the incumbent's depth or carry an
        // exact bound; shallow re-visits no longer clobber deep entries
        bool sameKey = TT_HIT(w, p.key);
        if (w.flag == TT_NONE || w.age != ttAge || depth + 3 >= w.depth
            || (sameKey && bestScore > origAlpha && bestScore < beta)) {
            // a fail-low's "best" move is just the least-bad refuted move;
            // keep the incumbent's proven move as the ordering hint instead
            if (sameKey && bestScore <= origAlpha && w.move) bestMove = w.move;
#else
        if (w.flag == TT_NONE || w.age != ttAge || TT_HIT(w, p.key) || depth + 3 >= w.depth) {
#endif
            int storeScore = bestScore;
            if (storeScore > MATE - MAX_PLY) storeScore += ply;
            else if (storeScore < -MATE + MAX_PLY) storeScore -= ply;
            w.key = TT_KEYV(p.key);
            w.move = bestMove;
            w.score = (int16_t)storeScore;
            w.eval = (int16_t)rawEval;
            w.depth = (int8_t)depth;
            w.flag = bestScore <= origAlpha ? TT_ALPHA : (bestScore >= beta ? TT_BETA : TT_EXACT);
            w.age = ttAge;
        }
    }

    return bestScore;
}

int g_bestMove = 0, g_bestScore = 0;
bool quietMode = false;

void prepThread(ThreadData &td, const Position &pos) {
    memset(td.killers, 0, sizeof(td.killers));
    for (int c = 0; c < 2; c++)
        for (int f = 0; f < 64; f++)
            for (int t = 0; t < 64; t++) td.history[c][f][t] /= 8;
    int n = gameHistLen > 1024 ? 1024 : gameHistLen;
    memcpy(td.keyHist, gameHist, n * sizeof(U64));
    td.keyHistLen = n;
    td.nodes = 0;
    if (nnueLoaded) nnRefresh(td.accStack[0], pos);
    td.accPtr[0] = &td.accStack[0];
#if FEAT_LAZYACC
    td.accReady[0] = true;
    td.accAlias[0] = false;
    td.pendSrc[0] = 0;
#endif
}

void helperLoop(Position pos, int id) {
    ThreadData &td = tds[id];
    prepThread(td, pos);
    for (int depth = 1; depth <= 64 && !si.stop; depth++)
        negamax(pos, depth, -INF, INF, 0, true, 0, td, false);
}

long long totalNodes() {
    long long n = 0;
    for (int i = 0; i < optThreads; i++) n += tds[i].nodes;
    return n;
}

void think(Position pos) {
    if (!quietMode && bookLoaded) {
        int bm = bookProbe(pos);
        if (bm) {
            g_bestMove = bm; g_bestScore = 0;
            printf("info string book move\n");
            return;
        }
    }
    ttAge++;
    ThreadData &td = tds[0];
    prepThread(td, pos);
    memset(td.rootEffort, 0, sizeof(td.rootEffort));

    int nHelpers = quietMode ? 0 : min(optThreads, MAX_THREADS) - 1;
    vector<thread> helpers;
    for (int i = 1; i <= nHelpers; i++)
        helpers.emplace_back(helperLoop, pos, i);

    int bestMove = 0;
    int prevScore = 0;
    int stable = 0;

    for (int depth = 1; depth <= si.maxDepth; depth++) {
        si.rootDepth = depth;
#if FEAT_ASP
        int delta = tAspDelta;
        int alpha = -INF, beta = INF;
        if (depth >= 5) { alpha = prevScore - delta; beta = prevScore + delta; }
        int score;
        while (true) {
            score = negamax(pos, depth, alpha, beta, 0, true, 0, td, false);
            if (si.stop) break;
            if (score <= alpha) { beta = (alpha + beta) / 2; alpha = score - delta; }
            else if (score >= beta) { beta = score + delta; }
            else break;
            delta *= 2;
            // safety: full window if still failing
            if (alpha < -3000) alpha = -INF;
            if (beta > 3000) beta = INF;
        }
#else
        int alpha = -INF, beta = INF;
        if (depth >= 5) { alpha = prevScore - 30; beta = prevScore + 30; }
        int score;
        while (true) {
            score = negamax(pos, depth, alpha, beta, 0, true, 0, td, false);
            if (si.stop) break;
            if (score <= alpha) { alpha = max(-INF, alpha - 120); beta = (alpha + beta) / 2 + 60; if (beta > INF) beta = INF; }
            else if (score >= beta) { beta = min(INF, beta + 120); }
            else break;
            if (alpha < -8000) alpha = -INF;
            if (beta > 8000) beta = INF;
        }
#endif
        if (si.stop && depth > 1) {
#if FEAT_PARTIALID
            // The root PV update sits behind negamax's stop guard, so a
            // nonempty PV here is a fully-searched root move that raised
            // alpha in the interrupted deepest iteration - paid-for
            // evidence. Adopt the move; leave prevScore/stable untouched.
            if (td.pvLen[0] > 0) bestMove = td.pvTable[0][0];
#endif
            break;
        }
        prevScore = score;
        if (td.pvLen[0] > 0) {
            if (td.pvTable[0][0] == bestMove) stable++; else stable = 0;
            bestMove = td.pvTable[0][0];
        }
        if (quietMode) {
            if (si.nodeLimit && td.nodes >= si.nodeLimit) break;
            if (si.timed && si.softLimit && elapsedMs() >= si.softLimit * 6 / 10) break;
            if (abs(score) > MATE - MAX_PLY && depth >= 10) break;
            continue;
        }

        long long ms = elapsedMs();
        long long nodes = totalNodes();
        long long nps = ms > 0 ? nodes * 1000 / ms : 0;
        printf("info depth %d score ", depth);
        if (abs(score) > MATE - MAX_PLY) {
            int mate = (MATE - abs(score) + 1) / 2;
            printf("mate %d", score > 0 ? mate : -mate);
        } else printf("cp %d", score);
        printf(" nodes %lld nps %lld time %lld pv", nodes, nps, ms);
        for (int i = 0; i < td.pvLen[0]; i++)
            printf(" %s", moveToUci(td.pvTable[0][i]).c_str());
        printf("\n");
        fflush(stdout);

        if (si.timed && si.softLimit) {
            long long f = si.fixedTime ? 95 : (stable >= 5 ? 45 : (stable >= 2 ? 60 : 85));
#if FEAT_TM
            // spend less when the best move dominates the search effort
            if (!si.fixedTime && bestMove && td.nodes > 0) {
                double frac = (double)td.rootEffort[bestMove & 4095] / (double)td.nodes;
                double s = 1.45 - frac;
                if (s < 0.55) s = 0.55;
                if (s > 1.25) s = 1.25;
                f = (long long)(f * s);
            }
#endif
            if (elapsedMs() >= si.softLimit * f / 100) break;
        }
        if (abs(score) > MATE - MAX_PLY && depth >= 10) break;
    }

    if (!bestMove) {
        // emergency: pick first legal move
        MoveList ml;
        genMoves(pos, ml, false);
        for (int i = 0; i < ml.count; i++) {
            Position next = pos;
            if (makeMove(next, ml.moves[i])) { bestMove = ml.moves[i]; break; }
        }
    }
    si.stop = true;
    for (auto &h : helpers) h.join();
    g_bestMove = bestMove;
    g_bestScore = prevScore;
}

void searchRoot(Position pos) {
    think(pos);
    printf("bestmove %s\n", g_bestMove ? moveToUci(g_bestMove).c_str() : "0000");
    fflush(stdout);
}

// ----------------------- perft -----------------------------
long long perft(Position &p, int depth) {
    if (depth == 0) return 1;
    MoveList ml;
    genMoves(p, ml, false);
    long long n = 0;
    for (int i = 0; i < ml.count; i++) {
        Position next = p;
        if (!makeMove(next, ml.moves[i])) continue;
        n += perft(next, depth - 1);
    }
    return n;
}

// ----------------------- data generation -------------------
#pragma pack(push, 1)
struct DGRecord {
    uint8_t pc[32];
    uint8_t sq[32];
    int16_t score;   // from side-to-move perspective
    uint8_t result;  // 0 black win, 1 draw, 2 white win (white POV)
    uint8_t stm;
};
#pragma pack(pop)

bool threefoldInGame(const Position &p) {
    int cnt = 0;
    for (int i = 0; i < gameHistLen; i++)
        if (gameHist[i] == p.key) cnt++;
    return cnt >= 3;
}

void datagen(long long games, long long nodesPerMove, const string &out, U64 seed) {
    rngState ^= seed * 0x2545F4914F6CDD1DULL + 0x9E3779B97F4A7C15ULL;
    FILE *f = fopen(out.c_str(), "ab");
    if (!f) { printf("cannot open %s\n", out.c_str()); return; }
    quietMode = true;
    long long written = 0;
    auto t0 = chrono::steady_clock::now();
    for (long long g = 0; g < games; g++) {
        Position pos;
        setFen(pos, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        gameHistLen = 0;
        gameHist[gameHistLen++] = pos.key;
        // random opening (8-9 plies)
        int plies = 8 + (int)(rand64() & 1);
        bool ok = true;
        for (int i = 0; i < plies; i++) {
            MoveList ml;
            genMoves(pos, ml, false);
            int legalM[256], n = 0;
            for (int j = 0; j < ml.count; j++) {
                Position t = pos;
                if (makeMove(t, ml.moves[j])) legalM[n++] = ml.moves[j];
            }
            if (n == 0) { ok = false; break; }
            int m = legalM[rand64() % n];
            Position next = pos;
            makeMove(next, m);
            pos = next;
            gameHist[gameHistLen++] = pos.key;
        }
        if (!ok) { g--; continue; }

        vector<DGRecord> buf;
        int result = 1, adjCount = 0;
        for (int ply = 0; ply < 400; ply++) {
            MoveList ml;
            genMoves(pos, ml, false);
            int anyLegal = 0;
            for (int j = 0; j < ml.count && !anyLegal; j++) {
                Position t = pos;
                if (makeMove(t, ml.moves[j])) anyLegal = 1;
            }
            if (!anyLegal) {
                result = inCheck(pos) ? (pos.side == WHITE ? 0 : 2) : 1;
                break;
            }
            if (pos.fifty >= 100 || threefoldInGame(pos)) { result = 1; break; }

            si.stop = false;
            si.timed = false;
            si.softLimit = si.hardLimit = 0;
            si.nodeLimit = nodesPerMove;
            si.maxDepth = 64;
            si.start = chrono::steady_clock::now();
            think(pos);
            int m = g_bestMove, score = g_bestScore;
            if (!m) { result = 1; break; }

            int whiteScore = pos.side == WHITE ? score : -score;
            if (abs(score) > 2500) {
                if (++adjCount >= 4) { result = whiteScore > 0 ? 2 : 0; break; }
            } else adjCount = 0;

            // store quiet positions only
            if (!inCheck(pos) && !M_CAP(m) && !M_PROMO(m) && abs(score) < 2000) {
                DGRecord r;
                memset(r.pc, 255, 32);
                memset(r.sq, 255, 32);
                int n = 0;
                for (int pc2 = 0; pc2 < 12; pc2++) {
                    U64 b = pos.bb[pc2];
                    while (b) { r.sq[n] = (uint8_t)poplsb(b); r.pc[n] = (uint8_t)pc2; n++; }
                }
                r.score = (int16_t)score;
                r.stm = (uint8_t)pos.side;
                r.result = 1;
                buf.push_back(r);
            }
            Position next = pos;
            makeMove(next, m);
            pos = next;
            if (gameHistLen < 1000) gameHist[gameHistLen++] = pos.key;
        }
        for (auto &r : buf) r.result = (uint8_t)result;
        fwrite(buf.data(), sizeof(DGRecord), buf.size(), f);
        written += (long long)buf.size();
        if ((g + 1) % 50 == 0) {
            fflush(f);
            long long ms = chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now() - t0).count();
            printf("datagen: %lld/%lld games, %lld positions, %lld pos/min\n",
                   g + 1, games, written, ms > 0 ? written * 60000 / ms : 0);
            fflush(stdout);
        }
    }
    fclose(f);
    quietMode = false;
    si.nodeLimit = 0;
    printf("datagen done: %lld positions -> %s\n", written, out.c_str());
}

// ----------------------- UCI -------------------------------
Position rootPos;
thread searchThread;

void stopSearch() {
    si.stop = true;
    if (searchThread.joinable()) searchThread.join();
}

int parseUciMove(const Position &p, const string &s) {
    MoveList ml;
    genMoves(p, ml, false);
    for (int i = 0; i < ml.count; i++)
        if (moveToUci(ml.moves[i]) == s) return ml.moves[i];
    return 0;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    initAttacks();
    initZobrist();
    initCastleMask();
    initEval();
    initLMR();
    ttResize(128);
    if (nnLoad("onyx.nnue"))
        printf("info string NNUE loaded: onyx.nnue\n");
    else if (nnLoad("sable.nnue"))   // pre-rename folders keep working
        printf("info string NNUE loaded: sable.nnue\n");
    if (bookLoad("book.bin"))
        printf("info string book loaded: book.bin\n");

    // Test-only opening-suite sampler.  It walks the fixed Polyglot book
    // without searching or generating self-play positions.
    // onyx booksuite <count> <minPlies> <maxPlies> <outfile> [seed]
    if (argc >= 6 && string(argv[1]) == "booksuite") {
        U64 seed = argc >= 7 ? (U64)strtoull(argv[6], nullptr, 10) : 1;
        bool ok = generateBookSuite(argv[5], atoi(argv[2]), atoi(argv[3]),
                                    atoi(argv[4]), seed);
        return ok ? 0 : 1;
    }

    // Book walk + eval-vetted random tail (fresh openings for confirmations).
    // onyx booksuite2 <count> <minBook> <tailMin> <tailMax> <outfile> [seed] [margin] [excludeEpd]
    if (argc >= 7 && string(argv[1]) == "booksuite2") {
        U64 seed = argc >= 8 ? (U64)strtoull(argv[7], nullptr, 10) : 1;
        int margin = argc >= 9 ? atoi(argv[8]) : 120;
        const char *excl = argc >= 10 ? argv[9] : nullptr;
        bool ok = generateTailSuite(argv[6], atoi(argv[2]), atoi(argv[3]),
                                    atoi(argv[4]), atoi(argv[5]), seed,
                                    margin, excl);
        return ok ? 0 : 1;
    }

#if FEAT_NPCORR
    // onyx npverify <depth>: assert incremental non-pawn keys over move trees
    if (argc >= 3 && string(argv[1]) == "npverify") {
        const char *fens[] = {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        };
        for (const char *f : fens) {
            Position vp;
            setFen(vp, f);
            npVerifyWalk(vp, atoi(argv[2]));
        }
        printf("npverify: %lld nodes, %lld key mismatches\n",
               npVerifyNodes, npVerifyFails);
        return npVerifyFails ? 1 : 0;
    }
#endif

    // command-line datagen: onyx datagen <games> <nodes> <outfile> <seed> [hashMB]
    if (argc >= 6 && string(argv[1]) == "datagen") {
        int mb = argc >= 7 ? atoi(argv[6]) : 64;
        ttResize(mb);
        setFen(rootPos, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        datagen(atoll(argv[2]), atoll(argv[3]), argv[4], (U64)atoll(argv[5]));
        return 0;
    }
    setFen(rootPos, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    gameHist[0] = rootPos.key;
    gameHistLen = 1;

    string line;
    while (getline(cin, line)) {
        istringstream ss(line);
        string cmd;
        ss >> cmd;

        if (cmd == "uci") {
            printf("id name Onyx 2.0\n");
            printf("id author Dylan (with Claude)\n");
            printf("option name Hash type spin default 128 min 1 max 4096\n");
            printf("option name Threads type spin default 1 min 1 max 16\n");
            printf("option name OwnBook type check default true\n");
            printf("option name BookFile type string default book.bin\n");
            printf("option name EvalFile type string default onyx.nnue\n");
            for (auto &t : tunables)
                printf("option name %s type spin default %d min %d max %d\n",
                       t.name, *t.p, t.lo, t.hi);
            printf("uciok\n");
        } else if (cmd == "isready") {
            printf("readyok\n");
        } else if (cmd == "setoption") {
            string tok, name, value;
            while (ss >> tok) {
                if (tok == "name") ss >> name;
                else if (tok == "value") {
                    string rest;
                    getline(ss, rest);
                    size_t a = rest.find_first_not_of(' ');
                    value = a == string::npos ? "" : rest.substr(a);
                    break;
                }
            }
            if (name == "Hash") {
                int mb = atoi(value.c_str());
                if (mb >= 1 && mb <= 4096) ttResize(mb);
            } else if (name == "Threads") {
                int t = atoi(value.c_str());
                if (t >= 1 && t <= MAX_THREADS) optThreads = t;
            } else if (name == "OwnBook") {
                if (value == "false") bookLoaded = false;
            } else if (name == "BookFile") {
                printf(bookLoad(value) ? "info string book loaded: %s\n"
                                       : "info string book load FAILED: %s\n", value.c_str());
            } else if (name == "EvalFile") {
                printf(nnLoad(value) ? "info string NNUE loaded: %s\n"
                                     : "info string NNUE load FAILED: %s\n", value.c_str());
            } else {
                for (auto &t : tunables)
                    if (name == t.name) {
                        int v = atoi(value.c_str());
                        if (v < t.lo) v = t.lo;
                        if (v > t.hi) v = t.hi;
                        *t.p = v;
                        if (t.p == &tLMRDiv) initLMR();
                        break;
                    }
            }
        } else if (cmd == "ucinewgame") {
            stopSearch();
            ttClear();
            for (int i = 0; i < MAX_THREADS; i++) {
                memset(tds[i].history, 0, sizeof(tds[i].history));
                memset(tds[i].counterMove, 0, sizeof(tds[i].counterMove));
                memset(tds[i].contHist, 0, sizeof(tds[i].contHist));
                memset(tds[i].corrHist, 0, sizeof(tds[i].corrHist));
                memset(tds[i].matCorr, 0, sizeof(tds[i].matCorr));
                memset(tds[i].capHist, 0, sizeof(tds[i].capHist));
                memset(tds[i].contHist2, 0, sizeof(tds[i].contHist2));
                memset(tds[i].contCorr, 0, sizeof(tds[i].contCorr));
                memset(tds[i].contCorrPair, 0, sizeof(tds[i].contCorrPair));
#if FEAT_NPCORR
                memset(tds[i].npCorr, 0, sizeof(tds[i].npCorr));
#endif
#if FEAT_KPCORR
                memset(tds[i].kpCorr, 0, sizeof(tds[i].kpCorr));
#endif
#if FEAT_PIECETO
                memset(tds[i].pieceToHist, 0, sizeof(tds[i].pieceToHist));
#endif
            }
        } else if (cmd == "position") {
            stopSearch();
            string tok;
            ss >> tok;
            if (tok == "startpos") {
                setFen(rootPos, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                ss >> tok; // maybe "moves"
            } else if (tok == "fen") {
                string fen, part;
                while (ss >> part) {
                    if (part == "moves") break;
                    fen += part + " ";
                }
                setFen(rootPos, fen);
                part == "moves" ? tok = "moves" : tok = "";
            }
            gameHistLen = 0;
            gameHist[gameHistLen++] = rootPos.key;
            string mv;
            while (ss >> mv) {
                int m = parseUciMove(rootPos, mv);
                if (!m) break;
                Position next = rootPos;
                if (!makeMove(next, m)) break;
                rootPos = next;
                if (gameHistLen < 1000) gameHist[gameHistLen++] = rootPos.key;
            }
        } else if (cmd == "go") {
            stopSearch();
            long long wtime = -1, btime = -1, winc = 0, binc = 0, movetime = -1, nodeLimit = 0;
            int movestogo = 0, depth = 0;
            string tok;
            while (ss >> tok) {
                if (tok == "wtime") ss >> wtime;
                else if (tok == "btime") ss >> btime;
                else if (tok == "winc") ss >> winc;
                else if (tok == "binc") ss >> binc;
                else if (tok == "movestogo") ss >> movestogo;
                else if (tok == "movetime") ss >> movetime;
                else if (tok == "depth") ss >> depth;
                else if (tok == "nodes") ss >> nodeLimit;
                else if (tok == "infinite") { /* nothing */ }
            }
            si.stop = false;
            si.nodeLimit = nodeLimit > 0 ? nodeLimit : 0;
            si.start = chrono::steady_clock::now();
            si.maxDepth = depth > 0 ? depth : 64;
            si.timed = false;
            si.softLimit = si.hardLimit = 0;

            long long myTime = rootPos.side == WHITE ? wtime : btime;
            long long myInc = rootPos.side == WHITE ? winc : binc;
            si.fixedTime = false;
            if (movetime > 0) {
                si.timed = true;
                si.fixedTime = true;
                si.softLimit = si.hardLimit = max(1LL, movetime - 30);
            } else if (myTime > 0) {
                si.timed = true;
                long long alloc;
                if (movestogo > 0) alloc = myTime / (movestogo + 2) + myInc / 2;
                else alloc = myTime / 22 + myInc / 2;
                si.softLimit = alloc;
                si.hardLimit = min(alloc * 4, myTime / 3);
                if (si.hardLimit < 1) si.hardLimit = 1;
                if (si.softLimit > si.hardLimit) si.softLimit = si.hardLimit;
            }
            searchThread = thread(searchRoot, rootPos);
        } else if (cmd == "stop") {
            stopSearch();
#if FEAT_DISCOVERED_CHECK_GUARD && FEAT_DISCOVERED_CHECK_VERIFY
        } else if (cmd == "quietchecktest") {
            struct CheckCase { const char *fen; const char *move; bool expected; };
            static const CheckCase cases[] = {
                {"4k3/8/8/8/8/8/4N3/K3R3 w - - 0 1", "e2c1", true},
                {"4k3/8/8/8/8/8/4B3/K3R3 w - - 0 1", "e2c4", true},
                {"4k3/8/8/8/8/8/4P3/K3R3 w - - 0 1", "e2e3", false},
                {"5k2/8/8/8/8/8/8/4K2R w K - 0 1", "e1g1", true},
                {"4k3/8/3P4/8/8/8/8/K7 w - - 0 1", "d6d7", true},
                {"4k3/8/8/8/8/8/8/R6K w - - 0 1", "a1a8", true},
            };
            int tested = 0, failures = 0;
            for (const CheckCase &tc : cases) {
                Position cp;
                setFen(cp, tc.fen);
                MoveList cml;
                genMoves(cp, cml, false);
                bool found = false;
                for (int i = 0; i < cml.count; i++) {
                    int m = cml.moves[i];
                    if (M_CAP(m) || M_PROMO(m)) continue;
                    Position next = cp;
                    if (!makeMove(next, m)) continue;
                    bool fast = givesQuietOrDiscoveredCheck(cp, m);
                    bool exact = inCheck(next);
                    tested++;
                    if (fast != exact) {
                        failures++;
                        printf("quietcheck mismatch %s fast=%d exact=%d\n",
                               moveToUci(m).c_str(), fast, exact);
                    }
                    if (moveToUci(m) == tc.move) {
                        found = true;
                        if (fast != tc.expected) {
                            failures++;
                            printf("quietcheck expected mismatch %s got=%d expected=%d\n",
                                   tc.move, fast, tc.expected);
                        }
                    }
                }
                if (!found) {
                    failures++;
                    printf("quietcheck move missing %s\n", tc.move);
                }
            }
            printf("quietchecktest: %d legal quiet moves, %d failures\n", tested, failures);
#endif
        } else if (cmd == "bench") {
            static const char *benchFens[] = {
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
                "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
                "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
                "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
                "3r1rk1/ppp2ppp/2n5/8/2P5/2N2N2/PP3PPP/3R1RK1 w - - 0 15",
                "8/8/1p1k4/p1p2p2/P1P2P2/1P1K4/8/8 w - - 0 40",
                "2rq1rk1/pb2bppp/1p2pn2/2p5/2BP4/1PN1PN2/PB3PPP/R2Q1RK1 w - - 0 12",
            };
            int bd = 13;
            ss >> bd;
            quietMode = true;
            long long totNodes = 0;
            auto tb0 = chrono::steady_clock::now();
            for (auto *f : benchFens) {
                Position bp;
                setFen(bp, f);
                gameHistLen = 0;
                gameHist[gameHistLen++] = bp.key;
                si.stop = false; si.timed = false; si.nodeLimit = 0;
                si.softLimit = si.hardLimit = 0;
                si.maxDepth = bd;
                si.start = chrono::steady_clock::now();
                think(bp);
                totNodes += tds[0].nodes;
            }
            auto tbms = chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now() - tb0).count();
            quietMode = false;
            printf("bench depth %d: %lld nodes %lld ms %lld nps\n",
                   bd, totNodes, tbms, tbms > 0 ? totNodes * 1000 / tbms : 0);
        } else if (cmd == "perft") {
            int d;
            ss >> d;
            auto t0 = chrono::steady_clock::now();
            long long n = perft(rootPos, d);
            auto ms = chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now() - t0).count();
            printf("perft %d: %lld nodes (%lld ms)\n", d, n, ms);
        } else if (cmd == "eval") {
            static Accumulator evalAcc;
            if (nnueLoaded) nnRefresh(evalAcc, rootPos);
            printf("eval: %d cp (side to move, %s)\n", evaluate(rootPos, &evalAcc),
                   nnueLoaded ? "nnue" : "classical");
        } else if (cmd == "datagen") {
            long long g2, n2; string of; U64 sd;
            if (ss >> g2 >> n2 >> of >> sd) datagen(g2, n2, of, sd);
        } else if (cmd == "quit") {
            stopSearch();
            break;
        }
    }
    stopSearch();
    return 0;
}
