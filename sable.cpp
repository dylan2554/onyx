// ============================================================
//  Sable 1.6 — a UCI chess engine written by Claude (Anthropic)
//  Bitboards + magic move generation
//  PVS / alpha-beta, transposition table, null-move, LMR,
//  killers/history, quiescence, aspiration windows
//  PeSTO tapered evaluation + pawn structure terms
//  Single file. Compile:  g++ -O3 -march=native -pthread sable.cpp -o sable
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
const int NN_MAX_HIDDEN = 512;
int nnHidden = 256;                     // set from net file magic
bool nnueLoaded = false;
alignas(64) int16_t nnW1[768][NN_MAX_HIDDEN];
int16_t nnB1[NN_MAX_HIDDEN];
int16_t nnW2[2 * NN_MAX_HIDDEN];
int32_t nnB2;

struct alignas(32) Accumulator { int16_t v[2][NN_MAX_HIDDEN]; };

// ----------------------- position --------------------------
struct Position {
    U64 bb[12];
    U64 occ[3];
    int side;
    int ep;        // ep square or -1
    int castle;    // 1 WK 2 WQ 4 BK 8 BQ
    int fifty;
    U64 key;
    Accumulator acc;

    void updateOcc() {
        occ[WHITE] = occ[BLACK] = 0;
        for (int p = WP; p <= WK; p++) occ[WHITE] |= bb[p];
        for (int p = BP; p <= BK; p++) occ[BLACK] |= bb[p];
        occ[BOTH] = occ[WHITE] | occ[BLACK];
    }
    int pieceOn(int sq) const {
        U64 b = 1ULL << sq;
        for (int p = 0; p < 12; p++) if (bb[p] & b) return p;
        return NO_PIECE;
    }
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
static inline void nnAdd(Position &p, int pc, int sq) {
    const int16_t *w0 = nnW1[featIdx(WHITE, pc, sq)];
    const int16_t *w1 = nnW1[featIdx(BLACK, pc, sq)];
    for (int i = 0; i < nnHidden; i += 16) {
        _mm256_store_si256((__m256i *)&p.acc.v[WHITE][i],
            _mm256_add_epi16(_mm256_load_si256((__m256i *)&p.acc.v[WHITE][i]),
                             _mm256_load_si256((const __m256i *)&w0[i])));
        _mm256_store_si256((__m256i *)&p.acc.v[BLACK][i],
            _mm256_add_epi16(_mm256_load_si256((__m256i *)&p.acc.v[BLACK][i]),
                             _mm256_load_si256((const __m256i *)&w1[i])));
    }
}
static inline void nnSub(Position &p, int pc, int sq) {
    const int16_t *w0 = nnW1[featIdx(WHITE, pc, sq)];
    const int16_t *w1 = nnW1[featIdx(BLACK, pc, sq)];
    for (int i = 0; i < nnHidden; i += 16) {
        _mm256_store_si256((__m256i *)&p.acc.v[WHITE][i],
            _mm256_sub_epi16(_mm256_load_si256((__m256i *)&p.acc.v[WHITE][i]),
                             _mm256_load_si256((const __m256i *)&w0[i])));
        _mm256_store_si256((__m256i *)&p.acc.v[BLACK][i],
            _mm256_sub_epi16(_mm256_load_si256((__m256i *)&p.acc.v[BLACK][i]),
                             _mm256_load_si256((const __m256i *)&w1[i])));
    }
}
#else
static inline void nnAdd(Position &p, int pc, int sq) {
    const int16_t *w0 = nnW1[featIdx(WHITE, pc, sq)];
    const int16_t *w1 = nnW1[featIdx(BLACK, pc, sq)];
    for (int i = 0; i < nnHidden; i++) p.acc.v[WHITE][i] += w0[i];
    for (int i = 0; i < nnHidden; i++) p.acc.v[BLACK][i] += w1[i];
}
static inline void nnSub(Position &p, int pc, int sq) {
    const int16_t *w0 = nnW1[featIdx(WHITE, pc, sq)];
    const int16_t *w1 = nnW1[featIdx(BLACK, pc, sq)];
    for (int i = 0; i < nnHidden; i++) p.acc.v[WHITE][i] -= w0[i];
    for (int i = 0; i < nnHidden; i++) p.acc.v[BLACK][i] -= w1[i];
}
#endif
void nnRefresh(Position &p) {
    for (int c = 0; c < 2; c++)
        for (int i = 0; i < nnHidden; i++) p.acc.v[c][i] = nnB1[i];
    for (int pc = 0; pc < 12; pc++) {
        U64 b = p.bb[pc];
        while (b) nnAdd(p, pc, poplsb(b));
    }
}
int nnEval(const Position &p) {
    int stm = p.side;
    int64_t sum = nnB2;
    const int16_t *a = p.acc.v[stm], *b = p.acc.v[stm ^ 1];
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
            _mm256_loadu_si256((const __m256i *)&nnW2[i])));
        accv = _mm256_add_epi32(accv, _mm256_madd_epi16(vb,
            _mm256_loadu_si256((const __m256i *)&nnW2[nnHidden + i])));
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
        sum += v * nnW2[i];
    }
    for (int i = 0; i < nnHidden; i++) {
        int v = b[i]; v = v < 0 ? 0 : (v > 255 ? 255 : v);
        sum += v * nnW2[nnHidden + i];
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
    int h;
    if (!memcmp(magic, "SABLE1", 6)) h = 256;
    else if (!memcmp(magic, "SABLE2", 6)) h = 512;
    else { fclose(f); return false; }
    bool ok = true;
    for (int ft = 0; ft < 768 && ok; ft++)
        ok = fread(nnW1[ft], sizeof(int16_t), h, f) == (size_t)h;
    ok = ok && fread(nnB1, sizeof(int16_t), h, f) == (size_t)h
            && fread(nnW2, sizeof(int16_t), 2 * h, f) == 2 * (size_t)h
            && fread(&nnB2, sizeof(int32_t), 1, f) == 1;
    fclose(f);
    if (ok) { nnHidden = h; nnueLoaded = true; }
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

    p.bb[pc] ^= (1ULL << from) | (1ULL << to);
    p.key ^= zPiece[pc][from] ^ zPiece[pc][to];
    if (nnueLoaded) { nnSub(p, pc, from); nnAdd(p, pc, to); }
    p.fifty++;
    if (pieceType(pc) == PAWN) p.fifty = 0;

    if (M_CAP(m)) {
        p.fifty = 0;
        int capSq = to;
        if (M_EP(m)) capSq = to + (us == WHITE ? -8 : 8);
        U64 cb = 1ULL << capSq;
        int lo = them == WHITE ? WP : BP;
        for (int cp = lo; cp <= lo + 5; cp++) {
            if (p.bb[cp] & cb) {
                p.bb[cp] ^= cb;
                p.key ^= zPiece[cp][capSq];
                if (nnueLoaded) nnSub(p, cp, capSq);
                break;
            }
        }
    }
    if (promo) {
        p.bb[pc] ^= 1ULL << to;
        p.bb[promo] |= 1ULL << to;
        p.key ^= zPiece[pc][to] ^ zPiece[promo][to];
        if (nnueLoaded) { nnSub(p, pc, to); nnAdd(p, promo, to); }
    }
    if (M_CASTLE(m)) {
        int rk = us == WHITE ? WR : BR;
        int rf, rt;
        if (to == SQ(6, 0)) { rf = SQ(7, 0); rt = SQ(5, 0); }
        else if (to == SQ(2, 0)) { rf = SQ(0, 0); rt = SQ(3, 0); }
        else if (to == SQ(6, 7)) { rf = SQ(7, 7); rt = SQ(5, 7); }
        else { rf = SQ(0, 7); rt = SQ(3, 7); }
        p.bb[rk] ^= (1ULL << rf) | (1ULL << rt);
        p.key ^= zPiece[rk][rf] ^ zPiece[rk][rt];
        if (nnueLoaded) { nnSub(p, rk, rf); nnAdd(p, rk, rt); }
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
    p.updateOcc();

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
    p.key = p.computeKey();
    if (nnueLoaded) nnRefresh(p);
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

int evaluate(const Position &p) {
    if (nnueLoaded) return nnEval(p);
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
    return p.side == WHITE ? score : -score;
}

// ----------------------- transposition table ---------------
enum { TT_NONE, TT_EXACT, TT_ALPHA, TT_BETA };
struct TTEntry {
    U64 key;
    int move;
    int16_t score;
    int16_t eval;
    int8_t depth;
    uint8_t flag;
    uint8_t age;
};
uint8_t ttAge = 0;
vector<TTEntry> tt;
size_t ttMask = 0;

void ttResize(int mb) {
    size_t entries = ((size_t)mb * 1024 * 1024) / sizeof(TTEntry);
    size_t pow2 = 1;
    while (pow2 * 2 <= entries) pow2 *= 2;
    tt.assign(pow2, TTEntry{0, 0, 0, 0, 0, TT_NONE, 0});
    ttMask = pow2 - 1;
}
void ttClear() { fill(tt.begin(), tt.end(), TTEntry{0, 0, 0, 0, 0, TT_NONE, 0}); }

// ----------------------- search ----------------------------
const int INF = 32000;
const int MATE = 31000;
const int MAX_PLY = 128;

struct SearchInfo {
    atomic<bool> stop{false};
    chrono::steady_clock::time_point start;
    long long softLimit = 0, hardLimit = 0; // ms; 0 = none
    bool fixedTime = false;                  // movetime: use it fully
    int maxDepth = 64;
    long long nodes = 0;
    long long nodeLimit = 0;
    bool timed = false;
};
SearchInfo si;

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
    U64 keyHist[1024];
    int keyHistLen = 0;
    long long nodes = 0;
};
ThreadData tds[MAX_THREADS];
int optThreads = 1;
U64 gameHist[1024];      // zobrist keys of the actual game
int gameHistLen = 0;
int lmrTable[64][64];

void initLMR() {
    for (int d = 1; d < 64; d++)
        for (int m = 1; m < 64; m++)
            lmrTable[d][m] = (int)(0.5 + log(d) * log(m) / 2.4);
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

void scoreMoves(const Position &p, MoveList &ml, int ttMove, int ply, int cm, int prev, ThreadData &td) {
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
            if (M_PROMO(m)) base += 50000;
            // good captures above killers, losing captures below quiet history
            ml.scores[i] = (see(p, m) >= 0 ? 1000000 : 50000) + base;
        } else if (M_PROMO(m)) {
            ml.scores[i] = 950000 + pieceType(M_PROMO(m));
        } else if (m == td.killers[ply][0]) ml.scores[i] = 900000;
        else if (m == td.killers[ply][1]) ml.scores[i] = 870000;
        else if (m == cm) ml.scores[i] = 850000;
        else {
            int s = td.history[p.side][M_FROM(m)][M_TO(m)];
            if (prev) s += 2 * td.contHist[M_PIECE(prev)][M_TO(prev)][M_PIECE(m)][M_TO(m)];
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
    td.nodes++;
    if ((td.nodes & 2047) == 0) checkTime(td);
    if (si.stop) return 0;
    if (ply >= MAX_PLY - 1) return evaluate(p);

    bool check = inCheck(p);
    int best;
    if (!check) {
        best = evaluate(p);
        if (best >= beta) return best;
        if (best > alpha) alpha = best;
    } else best = -INF;

    bool withQuietChecks = qsd >= 0 && !check;
    MoveList ml;
    genMoves(p, ml, check ? false : !withQuietChecks);
    scoreMoves(p, ml, 0, ply, 0, 0, td);

    int legal = 0;
    for (int i = 0; i < ml.count; i++) {
        pickMove(ml, i);
        int m = ml.moves[i];
        bool quietM = !M_CAP(m) && !M_PROMO(m);
        if (!check && quietM && !givesDirectCheck(p, m)) continue;
        if (!check && M_CAP(m)) {
            // delta pruning
            if (!M_PROMO(m)) {
                int victim = PAWN;
                if (!M_EP(m)) {
                    int vp = p.pieceOn(M_TO(m));
                    if (vp != NO_PIECE) victim = pieceType(vp);
                }
                if (best + mgValue[victim] + 200 < alpha) continue;
            }
            // skip losing captures
            if (see(p, m) < 0) continue;
        }
        Position next = p;
        if (!makeMove(next, m)) continue;
        legal++;
        int score = -qsearch(next, -beta, -alpha, ply + 1, td, qsd - 1);
        if (si.stop) return 0;
        if (score > best) {
            best = score;
            if (score > alpha) {
                alpha = score;
                if (alpha >= beta) break;
            }
        }
    }
    if (check && legal == 0) return -MATE + ply;
    return best;
}

int negamax(Position &p, int depth, int alpha, int beta, int ply, bool nullOk, int excluded, ThreadData &td) {
    td.pvLen[ply] = ply;
    bool pvNode = beta - alpha > 1;
    bool rootNode = ply == 0;

    if (!rootNode) {
        if (isRepetitionOrFifty(p, ply, td)) return 0;
        if (ply >= MAX_PLY - 1) return evaluate(p);
        // mate distance pruning
        alpha = max(alpha, -MATE + ply);
        beta = min(beta, MATE - ply - 1);
        if (alpha >= beta) return alpha;
    }

    bool check = inCheck(p);
    if (check) depth++;
    if (depth <= 0) return qsearch(p, alpha, beta, ply, td, 0);

    td.nodes++;
    if ((td.nodes & 2047) == 0) checkTime(td);
    if (si.stop) return 0;

    // TT probe
    TTEntry &e = tt[p.key & ttMask];
    int ttMove = 0, ttScore = -INF, ttDepth = -1, ttFlag = TT_NONE;
    bool ttHit = e.key == p.key;
    if (ttHit) {
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
    if (!ttMove && depth >= 4) depth--;

    int staticEval = ttHit ? e.eval : evaluate(p);
    td.ssEval[ply] = staticEval;
    bool improving = !check && ply >= 2 && staticEval > td.ssEval[ply - 2];

    // razoring
    if (!pvNode && !check && !excluded && depth <= 3
        && staticEval + 250 * depth < alpha) {
        int q = qsearch(p, alpha, beta, ply, td, -1);
        if (q < alpha) return q;
    }

    // reverse futility pruning
    if (!pvNode && !check && !excluded && depth <= 7
        && staticEval - 80 * (depth - (improving ? 1 : 0)) >= beta
        && abs(beta) < MATE - MAX_PLY)
        return staticEval;

    // null move pruning
    U64 bigPieces = p.side == WHITE
        ? p.bb[WN] | p.bb[WB] | p.bb[WR] | p.bb[WQ]
        : p.bb[BN] | p.bb[BB_] | p.bb[BR] | p.bb[BQ];
    if (!pvNode && !check && nullOk && !excluded && depth >= 3
        && staticEval >= beta && bigPieces) {
        Position next = p;
        makeNull(next);
        td.ssMove[ply] = 0;
        td.keyHist[td.keyHistLen++] = next.key;
        int R = 3 + depth / 5 + min((staticEval - beta) / 200, 2);
        int score = -negamax(next, depth - 1 - R, -beta, -beta + 1, ply + 1, false, 0, td);
        td.keyHistLen--;
        if (si.stop) return 0;
        if (score >= beta) {
            if (score > MATE - MAX_PLY) score = beta;
            return score;
        }
    }

    int prev = ply > 0 ? td.ssMove[ply - 1] : 0;
    int cm = 0;
    if (prev) cm = td.counterMove[p.side][M_FROM(prev)][M_TO(prev)];

    MoveList ml;
    genMoves(p, ml, false);
    scoreMoves(p, ml, ttMove, ply, cm, prev, td);

    int bestScore = -INF, bestMove = 0, legal = 0;
    int quietsTried[64], nQuiets = 0;
    int origAlpha = alpha;
    bool futile = !pvNode && !check && depth <= 6
                  && staticEval + 90 + 75 * depth <= alpha;
    int lmpLimit = (3 + depth * depth) / (improving ? 1 : 2);

    for (int i = 0; i < ml.count; i++) {
        pickMove(ml, i);
        int m = ml.moves[i];
        if (m == excluded) continue;
        bool quiet = !M_CAP(m) && !M_PROMO(m);

        int hScore = 0;
        if (quiet) {
            hScore = td.history[p.side][M_FROM(m)][M_TO(m)];
            if (prev) hScore += 2 * td.contHist[M_PIECE(prev)][M_TO(prev)][M_PIECE(m)][M_TO(m)];
        }
        if (legal > 0 && bestScore > -MATE + MAX_PLY) {
            // futility pruning of quiets
            if (futile && quiet) continue;
            // history pruning: quiets with awful history at shallow depth
            if (!pvNode && !check && quiet && depth <= 4 && hScore < -1500 * depth)
                continue;
            // late move pruning
            if (!pvNode && !check && quiet && depth <= 5 && legal > lmpLimit)
                continue;
            // SEE pruning of bad captures at shallow depth
            if (!pvNode && !check && depth <= 6 && M_CAP(m)
                && see(p, m) < -160 * depth)
                continue;
        }

        // singular extension: is the TT move much better than everything else?
        int ext = 0;
        if (!rootNode && !excluded && depth >= 8 && m == ttMove
            && ttDepth >= depth - 3 && ttFlag != TT_ALPHA
            && abs(ttScore) < MATE - MAX_PLY) {
            int sBeta = ttScore - 2 * depth;
            int sScore = negamax(p, (depth - 1) / 2, sBeta - 1, sBeta, ply, false, m, td);
            if (si.stop) return 0;
            if (sScore < sBeta) {
                ext = 1;                                       // singular -> extend
                if (!pvNode && sScore < sBeta - 25) ext = 2;   // strongly singular
            }
            else if (sBeta >= beta) return sBeta; // multi-cut
        }

        Position next = p;
        if (!makeMove(next, m)) continue;
        legal++;
        if (quiet && nQuiets < 64) quietsTried[nQuiets++] = m;
        td.ssMove[ply] = m;
        td.keyHist[td.keyHistLen++] = next.key;

        int nd = depth - 1 + ext;
        int score;
        if (legal == 1) {
            score = -negamax(next, nd, -beta, -alpha, ply + 1, true, 0, td);
        } else {
            int R = 0;
            if (depth >= 3 && quiet && !check && legal > 3) {
                R = lmrTable[min(depth, 63)][min(legal, 63)];
                if (pvNode) R--;
                if (!improving) R++;
                if (m == td.killers[ply][0] || m == td.killers[ply][1] || m == cm) R--;
                int hAdj = hScore / 8000;
                if (hAdj > 2) hAdj = 2;
                if (hAdj < -2) hAdj = -2;
                R -= hAdj;
                if (R < 0) R = 0;
                if (R > nd - 1) R = nd - 1;
            }
            score = -negamax(next, nd - R, -alpha - 1, -alpha, ply + 1, true, 0, td);
            if (score > alpha && R > 0)
                score = -negamax(next, nd, -alpha - 1, -alpha, ply + 1, true, 0, td);
            if (score > alpha && score < beta)
                score = -negamax(next, nd, -beta, -alpha, ply + 1, true, 0, td);
        }
        td.keyHistLen--;
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
                        int &h = td.history[p.side][M_FROM(m)][M_TO(m)];
                        h += depth * depth;
                        if (h > 800000) h /= 2;
                        // maluses: punish quiets that were tried and failed
                        for (int q = 0; q < nQuiets - 1; q++) {
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
                    break;
                }
            }
        }
    }

    if (legal == 0) return excluded ? alpha : (check ? -MATE + ply : 0);

    // TT store
    if (!excluded
        && (e.flag == TT_NONE || e.age != ttAge || e.key == p.key || depth + 3 >= e.depth)) {
        int storeScore = bestScore;
        if (storeScore > MATE - MAX_PLY) storeScore += ply;
        else if (storeScore < -MATE + MAX_PLY) storeScore -= ply;
        e.key = p.key;
        e.move = bestMove;
        e.score = (int16_t)storeScore;
        e.eval = (int16_t)staticEval;
        e.depth = (int8_t)depth;
        e.flag = bestScore <= origAlpha ? TT_ALPHA : (bestScore >= beta ? TT_BETA : TT_EXACT);
        e.age = ttAge;
    }

    return bestScore;
}

int g_bestMove = 0, g_bestScore = 0;
bool quietMode = false;

void prepThread(ThreadData &td) {
    memset(td.killers, 0, sizeof(td.killers));
    for (int c = 0; c < 2; c++)
        for (int f = 0; f < 64; f++)
            for (int t = 0; t < 64; t++) td.history[c][f][t] /= 8;
    memcpy(td.keyHist, gameHist, gameHistLen * sizeof(U64));
    td.keyHistLen = gameHistLen;
    td.nodes = 0;
}

void helperLoop(Position pos, int id) {
    ThreadData &td = tds[id];
    prepThread(td);
    for (int depth = 1; depth <= 64 && !si.stop; depth++)
        negamax(pos, depth, -INF, INF, 0, true, 0, td);
}

long long totalNodes() {
    long long n = 0;
    for (int i = 0; i < optThreads; i++) n += tds[i].nodes;
    return n;
}

void think(Position pos) {
    ttAge++;
    ThreadData &td = tds[0];
    prepThread(td);

    int nHelpers = quietMode ? 0 : min(optThreads, MAX_THREADS) - 1;
    vector<thread> helpers;
    for (int i = 1; i <= nHelpers; i++)
        helpers.emplace_back(helperLoop, pos, i);

    int bestMove = 0;
    int prevScore = 0;
    int stable = 0;

    for (int depth = 1; depth <= si.maxDepth; depth++) {
        int alpha = -INF, beta = INF;
        if (depth >= 5) { alpha = prevScore - 30; beta = prevScore + 30; }
        int score;
        while (true) {
            score = negamax(pos, depth, alpha, beta, 0, true, 0, td);
            if (si.stop) break;
            if (score <= alpha) { alpha = max(-INF, alpha - 120); beta = (alpha + beta) / 2 + 60; if (beta > INF) beta = INF; }
            else if (score >= beta) { beta = min(INF, beta + 120); }
            else break;
            // safety: full window if still failing
            if (alpha < -8000) alpha = -INF;
            if (beta > 8000) beta = INF;
        }
        if (si.stop && depth > 1) break;
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
    if (nnLoad("sable.nnue"))
        printf("info string NNUE loaded: sable.nnue\n");

    // command-line datagen: sable datagen <games> <nodes> <outfile> <seed> [hashMB]
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
            printf("id name Sable 1.6\n");
            printf("id author Dylan (with Claude)\n");
            printf("option name Hash type spin default 128 min 1 max 2048\n");
            printf("option name Threads type spin default 1 min 1 max 16\n");
            printf("option name EvalFile type string default sable.nnue\n");
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
                if (mb >= 1 && mb <= 2048) ttResize(mb);
            } else if (name == "Threads") {
                int t = atoi(value.c_str());
                if (t >= 1 && t <= MAX_THREADS) optThreads = t;
            } else if (name == "EvalFile") {
                printf(nnLoad(value) ? "info string NNUE loaded: %s\n"
                                     : "info string NNUE load FAILED: %s\n", value.c_str());
                if (nnueLoaded) nnRefresh(rootPos);
            }
        } else if (cmd == "ucinewgame") {
            stopSearch();
            ttClear();
            for (int i = 0; i < MAX_THREADS; i++) {
                memset(tds[i].history, 0, sizeof(tds[i].history));
                memset(tds[i].counterMove, 0, sizeof(tds[i].counterMove));
                memset(tds[i].contHist, 0, sizeof(tds[i].contHist));
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
        } else if (cmd == "perft") {
            int d;
            ss >> d;
            auto t0 = chrono::steady_clock::now();
            long long n = perft(rootPos, d);
            auto ms = chrono::duration_cast<chrono::milliseconds>(
                chrono::steady_clock::now() - t0).count();
            printf("perft %d: %lld nodes (%lld ms)\n", d, n, ms);
        } else if (cmd == "eval") {
            printf("eval: %d cp (side to move, %s)\n", evaluate(rootPos),
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
