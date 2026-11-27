#include "threats.h"

#include "tables.h"

#include <cstring>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace jq::thr {

int DIMS = 0;
uint32_t E[NCODE][NUM_SQ][NUM_SQ];
alignas(64) uint32_t E2[NCODE][NUM_SQ][NSLOT];
uint8_t SLOT[NUM_SQ][NUM_SQ];
int8_t KREL[2][32];
uint8_t OT[2][2][NUM_SQ];
int8_t KIDX[32];

namespace {

// (horse square, target) pairs whose leg is s; (elephant square, target) pairs whose eye is s.
uint8_t legH[NUM_SQ][8], legT[NUM_SQ][8], legN[NUM_SQ];
uint8_t eyeE[NUM_SQ][4], eyeT[NUM_SQ][4], eyeN[NUM_SQ];

inline PieceType mkind(Piece p, int s) {
    const PieceType t = type_of(p);
    return t == DARK ? PieceType(T.darkRole[color_of(p)][s]) : t;
}

// Unblocked attack geometry of colour c / movement kind k at s (dark: face-down restrictions).
int geometry(Color c, PieceType k, bool dark, int s, uint8_t* out) {
    int n = 0;
    switch (k) {
    case ROOK: case CANNON:
        for (int d = 0; d < 4; ++d)
            for (int j = (k == CANNON ? 1 : 0); j < T.rayLen[s][d]; ++j) out[n++] = T.ray[s][d][j];
        break;
    case HORSE: for (int j = 0; j < T.horseN[s]; ++j) out[n++] = T.horseTo[s][j]; break;
    case ELEPHANT:
        for (int j = 0; j < T.elephN[s]; ++j)
            if (!dark || own_half(c, T.elephTo[s][j])) out[n++] = T.elephTo[s][j];
        break;
    case ADVISOR:
        for (int j = 0; j < T.advN[s]; ++j)
            if (!dark || in_palace(c, T.advTo[s][j])) out[n++] = T.advTo[s][j];
        break;
    case SOLDIER: {
        const int t = s + (c == RED ? 9 : -9);
        if (t >= 0 && t < NUM_SQ) out[n++] = uint8_t(t);
        if (!own_half(c, s)) {
            if (file_of(s) > 0) out[n++] = uint8_t(s - 1);
            if (file_of(s) < 8) out[n++] = uint8_t(s + 1);
        }
    } break;
    default: break;
    }
    return n;
}

constexpr PieceType KIND_TYPE[7] = {ROOK, HORSE, ELEPHANT, ADVISOR, CANNON, SOLDIER, DARK};

bool code_on(int code, int s) {  // may a piece of relative code stand on s (in the relative frame: own = RED)
    const Color c = code < 7 ? RED : BLACK;
    return KIND_TYPE[code % 7] != DARK || T.darkRole[c][s] != 0;
}
PieceType code_move(int code, int s) {
    const Color c = code < 7 ? RED : BLACK;
    const PieceType t = KIND_TYPE[code % 7];
    return t == DARK ? PieceType(T.darkRole[c][s]) : t;
}
bool attacks_geom(int code, int from, int to) {
    uint8_t g[20];
    const Color c = code < 7 ? RED : BLACK;
    const int n = geometry(c, code_move(code, from), KIND_TYPE[code % 7] == DARK, from, g);
    for (int i = 0; i < n; ++i)
        if (g[i] == to) return true;
    return false;
}

struct Gen {
    const Piece* b;
    uint32_t* out;
    int n = 0;
    inline void emit(bool add, Piece a, int from, Piece v, int to) {
        if (type_of(v) == KING || type_of(a) == KING) return;
        out[n++] = pack(add, a, from, v, to);
    }
    // first and second occupied squares along each ray from s (s itself is never read)
    inline void scan(int s, int (&f1)[4], int (&f2)[4]) const {
        for (int d = 0; d < 4; ++d) {
            f1[d] = f2[d] = -1;
            const int L = T.rayLen[s][d];
            int j = 0;
            for (; j < L; ++j) if (b[T.ray[s][d][j]]) { f1[d] = T.ray[s][d][j]; break; }
            for (++j; j < L; ++j) if (b[T.ray[s][d][j]]) { f2[d] = T.ray[s][d][j]; break; }
        }
    }
    void outgoing(Piece pc, int s, bool add, const int (&f1)[4], const int (&f2)[4]) {
        const Color c = color_of(pc);
        const bool dark = type_of(pc) == DARK;
        switch (mkind(pc, s)) {
        case ROOK: for (int d = 0; d < 4; ++d) if (f1[d] >= 0) emit(add, pc, s, b[f1[d]], f1[d]); break;
        case CANNON: for (int d = 0; d < 4; ++d) if (f2[d] >= 0) emit(add, pc, s, b[f2[d]], f2[d]); break;
        case HORSE:
            for (int j = 0; j < T.horseN[s]; ++j) {
                const int t = T.horseTo[s][j];
                if (b[t] && !b[T.horseLeg[s][j]]) emit(add, pc, s, b[t], t);
            }
            break;
        case ELEPHANT:
            for (int j = 0; j < T.elephN[s]; ++j) {
                const int t = T.elephTo[s][j];
                if (b[t] && !b[T.elephEye[s][j]] && (!dark || own_half(c, t))) emit(add, pc, s, b[t], t);
            }
            break;
        case ADVISOR:
            for (int j = 0; j < T.advN[s]; ++j) {
                const int t = T.advTo[s][j];
                if (b[t] && (!dark || in_palace(c, t))) emit(add, pc, s, b[t], t);
            }
            break;
        case SOLDIER: {
            const int t = s + (c == RED ? 9 : -9);
            if (t >= 0 && t < NUM_SQ && b[t]) emit(add, pc, s, b[t], t);
            if (!own_half(c, s)) {
                if (file_of(s) > 0 && b[s - 1]) emit(add, pc, s, b[s - 1], s - 1);
                if (file_of(s) < 8 && b[s + 1]) emit(add, pc, s, b[s + 1], s + 1);
            }
        } break;
        default: break;
        }
    }
    // attackers of s (victim v): emits for v and, if v2 != 0, the opposite sign for v2 (swap)
    void incoming(int s, Piece v, bool add, Piece v2, const int (&f1)[4], const int (&f2)[4]) {
        auto em = [&](Piece a, int from) {
            emit(add, a, from, v, s);
            if (v2) emit(!add, a, from, v2, s);
        };
        for (int d = 0; d < 4; ++d) {
            if (f1[d] >= 0 && mkind(b[f1[d]], f1[d]) == ROOK) em(b[f1[d]], f1[d]);
            if (f2[d] >= 0 && mkind(b[f2[d]], f2[d]) == CANNON) em(b[f2[d]], f2[d]);
        }
        for (int j = 0; j < T.horseFromN[s]; ++j) {
            const int p = T.horseFrom[s][j];
            if (b[p] && !b[T.horseFromLeg[s][j]] && mkind(b[p], p) == HORSE) em(b[p], p);
        }
        for (int j = 0; j < T.elephN[s]; ++j) {
            const int p = T.elephTo[s][j];
            if (b[p] && !b[T.elephEye[s][j]] && mkind(b[p], p) == ELEPHANT &&
                (type_of(b[p]) != DARK || own_half(color_of(b[p]), s)))
                em(b[p], p);
        }
        for (int j = 0; j < T.advN[s]; ++j) {
            const int p = T.advTo[s][j];
            if (b[p] && mkind(b[p], p) == ADVISOR && (type_of(b[p]) != DARK || in_palace(color_of(b[p]), s))) em(b[p], p);
        }
        if (s >= 9 && b[s - 9] && color_of(b[s - 9]) == RED && mkind(b[s - 9], s - 9) == SOLDIER) em(b[s - 9], s - 9);
        if (s + 9 < NUM_SQ && b[s + 9] && color_of(b[s + 9]) == BLACK && mkind(b[s + 9], s + 9) == SOLDIER) em(b[s + 9], s + 9);
        if (file_of(s) > 0) {
            const int p = s - 1;
            if (b[p] && mkind(b[p], p) == SOLDIER && !own_half(color_of(b[p]), p)) em(b[p], p);
        }
        if (file_of(s) < 8) {
            const int p = s + 1;
            if (b[p] && mkind(b[p], p) == SOLDIER && !own_half(color_of(b[p]), p)) em(b[p], p);
        }
    }
    // attacks of other pieces that pass through s (s empty <-> occupied); add = s becomes occupied
    void discovered(int s, bool add, const int (&f1)[4], const int (&f2)[4]) {
        for (int d = 0; d < 4; ++d) {
            const int od = d ^ 1, x1 = f1[d], x2 = f2[d], y1 = f1[od], y2 = f2[od];
            if (x1 >= 0 && y1 >= 0) {
                const PieceType k = mkind(b[x1], x1);
                if (k == ROOK) emit(!add, b[x1], x1, b[y1], y1);
                else if (k == CANNON) {
                    emit(add, b[x1], x1, b[y1], y1);
                    if (y2 >= 0) emit(!add, b[x1], x1, b[y2], y2);
                }
            }
            if (x2 >= 0 && y1 >= 0 && mkind(b[x2], x2) == CANNON) emit(!add, b[x2], x2, b[y1], y1);
        }
        for (int j = 0; j < legN[s]; ++j) {
            const int h = legH[s][j], t = legT[s][j];
            if (b[h] && b[t] && mkind(b[h], h) == HORSE) emit(!add, b[h], h, b[t], t);
        }
        for (int j = 0; j < eyeN[s]; ++j) {
            const int e = eyeE[s][j], t = eyeT[s][j];
            if (b[e] && b[t] && mkind(b[e], e) == ELEPHANT && (type_of(b[e]) != DARK || own_half(color_of(b[e]), t)))
                emit(!add, b[e], e, b[t], t);
        }
    }
};

}  // namespace

namespace { void init_fast(); }
void init_bb3();
void init() {
    static bool done = false;
    if (done) return;
    done = true;
    init_tables();
    std::memset(KIDX, -1, sizeof KIDX);
    for (int c = 0; c < 2; ++c)
        for (int k = 0; k < 7; ++k) KIDX[make_piece(Color(c), KIND_TYPE[k])] = int8_t(k);
    std::memset(KREL, -1, sizeof KREL);
    for (int p = 0; p < 2; ++p)
        for (int c = 0; c < 2; ++c)
            for (int k = 0; k < 7; ++k) KREL[p][make_piece(Color(c), KIND_TYPE[k])] = int8_t(k + (c == p ? 0 : 7));
    for (int p = 0; p < 2; ++p)
        for (int m = 0; m < 2; ++m)
            for (int s = 0; s < NUM_SQ; ++s) {
                int o = p == RED ? s : (9 - rank_of(s)) * 9 + file_of(s);
                if (m) o = rank_of(o) * 9 + 8 - file_of(o);
                OT[p][m][s] = uint8_t(o);
            }
    std::memset(legN, 0, sizeof legN);
    std::memset(eyeN, 0, sizeof eyeN);
    for (int h = 0; h < NUM_SQ; ++h) {
        for (int j = 0; j < T.horseN[h]; ++j) {
            const int l = T.horseLeg[h][j];
            legH[l][legN[l]] = uint8_t(h); legT[l][legN[l]++] = T.horseTo[h][j];
        }
        for (int j = 0; j < T.elephN[h]; ++j) {
            const int e = T.elephEye[h][j];
            eyeE[e][eyeN[e]] = uint8_t(h); eyeT[e][eyeN[e]++] = T.elephTo[h][j];
        }
    }
    std::memset(E, 0, sizeof E);
    uint32_t cnt = 0;
    for (int a = 0; a < NCODE; ++a)
        for (int from = 0; from < NUM_SQ; ++from) {
            if (!code_on(a, from)) continue;
            const PieceType ak = code_move(a, from);
            for (int to = 0; to < NUM_SQ; ++to) {
                if (!attacks_geom(a, from, to)) continue;
                uint16_t mask = 0;
                for (int v = 0; v < NCODE; ++v) {
                    if (!code_on(v, to)) continue;
                    if (code_move(v, to) == ak && ak != HORSE && from < to && attacks_geom(v, to, from)) continue;
                    mask = uint16_t(mask | (1u << v));
                }
                E[a][from][to] = cnt | (uint32_t(mask) << 17);
                cnt += uint32_t(__builtin_popcount(mask));
            }
        }
    DIMS = int(cnt);
    if (cnt >= (1u << 17)) __builtin_trap();
    // h32fatt compact table: slots per from square (file/rank squares, horse targets, 1- and 2-step diagonals)
    std::memset(SLOT, NSLOT - 1, sizeof SLOT);
    for (int from = 0; from < NUM_SQ; ++from) {
        int k = 0;
        const int f = file_of(from), r = rank_of(from);
        for (int to = 0; to < NUM_SQ; ++to)
            if (to != from && (file_of(to) == f || rank_of(to) == r)) SLOT[from][to] = uint8_t(k++);
        for (int j = 0; j < T.horseN[from]; ++j) SLOT[from][T.horseTo[from][j]] = uint8_t(k++);
        for (int df = -1; df <= 1; df += 2) for (int dr = -1; dr <= 1; dr += 2)
            for (int st = 1; st <= 2; ++st) {
                const int ff = f + st * df, rr = r + st * dr;
                if (ff < 0 || ff > 8 || rr < 0 || rr > 9) continue;
                SLOT[from][rr * 9 + ff] = uint8_t(k++);
            }
        if (k > NSLOT - 1) __builtin_trap();
    }
    std::memset(E2, 0, sizeof E2);
    for (int a = 0; a < NCODE; ++a)
        for (int from = 0; from < NUM_SQ; ++from)
            for (int to = 0; to < NUM_SQ; ++to) {
                if (!E[a][from][to]) continue;
                if (SLOT[from][to] == NSLOT - 1) __builtin_trap();   // geometry outside the slots
                E2[a][from][SLOT[from][to]] = E[a][from][to];
            }
    init_fast();
    init_bb3();
}

int gen_move_ref(const Piece* b0, Move m, uint32_t* out, Piece* lb) {
    const int from = from_sq(m), to = to_sq(m);
    std::memcpy(lb, b0, NUM_SQ);
    const Piece pc = lb[from], cap = lb[to];
    const Piece np = type_of(pc) == DARK ? make_piece(color_of(pc), reveal_of(m)) : pc;
    Gen g{lb, out};
    int f1[4], f2[4];
    // 1. the mover leaves `from`
    g.scan(from, f1, f2);
    g.outgoing(pc, from, false, f1, f2);
    if (type_of(pc) != KING) g.incoming(from, pc, false, 0, f1, f2);
    g.discovered(from, false, f1, f2);
    lb[from] = NO_PIECE;
    // 2. it arrives on `to` (replacing a captured piece: no change of occupancy there)
    g.scan(to, f1, f2);
    if (cap) {
        g.outgoing(cap, to, false, f1, f2);
        g.incoming(to, cap, false, type_of(np) == KING ? Piece(0) : np, f1, f2);
        lb[to] = np;
        g.outgoing(np, to, true, f1, f2);
    } else {
        g.outgoing(np, to, true, f1, f2);
        if (type_of(np) != KING) g.incoming(to, np, true, 0, f1, f2);
        g.discovered(to, true, f1, f2);
        lb[to] = np;
    }
    return g.n;
}


// ---------------------------------------------------------------------------------------------------------------------
// Fast generator (v2): 128-bit occupancy + movement-kind sets (horse / elephant / advisor, face-down resolved by
// darkRole) built with AVX2 from the board bytes, line tables for the first / second blocker of every ray, and masks so
// that the leg / eye / incoming-leaper checks are a couple of ANDs that are almost always zero. Emits the same tuple
// multiset as gen_move_ref (checked by thr_bench / thr_test).
namespace {
using BB = Bitboard;
inline BB bit(int s) { return BB(1) << s; }
template <class F> inline void each(BB m, F&& f) {
    uint64_t lo = uint64_t(m), hi = uint64_t(m >> 64);
    while (lo) { f(__builtin_ctzll(lo)); lo &= lo - 1; }
    while (hi) { f(64 + __builtin_ctzll(hi)); hi &= hi - 1; }
}
BB HTO[NUM_SQ], ETO[NUM_SQ], ATO[NUM_SQ], LEGOWN[NUM_SQ], EYEOWN[NUM_SQ], HALF[2], PAL[2];
uint8_t LEG[NUM_SQ][NUM_SQ], EYE[NUM_SQ][NUM_SQ];
int8_t HITF[10][1024][4];  // file line (10 ranks): [up1, up2, dn1, dn2] as rank*9, or -100
int8_t HITR[9][512][4];    // rank line (9 files):  [up1, up2, dn1, dn2] as file, or -100
alignas(32) uint8_t DR[2][96];  // darkRole padded

void init_fast2();
void init_fast() {
    std::memset(HTO, 0, sizeof HTO); std::memset(ETO, 0, sizeof ETO); std::memset(ATO, 0, sizeof ATO);
    std::memset(LEGOWN, 0, sizeof LEGOWN); std::memset(EYEOWN, 0, sizeof EYEOWN);
    HALF[0] = HALF[1] = PAL[0] = PAL[1] = 0;
    for (int s = 0; s < NUM_SQ; ++s) {
        for (int j = 0; j < T.horseN[s]; ++j) {
            HTO[s] |= bit(T.horseTo[s][j]); LEG[s][T.horseTo[s][j]] = T.horseLeg[s][j]; LEGOWN[T.horseLeg[s][j]] |= bit(s);
        }
        for (int j = 0; j < T.elephN[s]; ++j) {
            ETO[s] |= bit(T.elephTo[s][j]); EYE[s][T.elephTo[s][j]] = T.elephEye[s][j]; EYEOWN[T.elephEye[s][j]] |= bit(s);
        }
        for (int j = 0; j < T.advN[s]; ++j) ATO[s] |= bit(T.advTo[s][j]);
        for (int c = 0; c < 2; ++c) {
            if (own_half(Color(c), s)) HALF[c] |= bit(s);
            if (in_palace(Color(c), s)) PAL[c] |= bit(s);
        }
    }
    std::memset(DR, 0, sizeof DR);
    for (int c = 0; c < 2; ++c) for (int s = 0; s < NUM_SQ; ++s) DR[c][s] = T.darkRole[c][s];
    auto line = [](int len, int pos, uint32_t occ, int mul, int8_t* o) {
        o[0] = o[1] = o[2] = o[3] = -100;
        int k = 0;
        for (int i = pos + 1; i < len && k < 2; ++i) if (occ >> i & 1) o[k++] = int8_t(i * mul);
        k = 0;
        for (int i = pos - 1; i >= 0 && k < 2; --i) if (occ >> i & 1) o[2 + k++] = int8_t(i * mul);
    };
    for (int p = 0; p < 10; ++p) for (uint32_t o = 0; o < 1024; ++o) line(10, p, o, 9, HITF[p][o]);
    for (int p = 0; p < 9; ++p) for (uint32_t o = 0; o < 512; ++o) line(9, p, o, 1, HITR[p][o]);
    init_fast2();
}

constexpr int SENT = 95;      // sentinel square: empty, kind 0
uint8_t VV[256];              // piece is a valid victim (non-empty, not a king)
uint8_t FWD[2][NUM_SQ], WST[NUM_SQ], EST[NUM_SQ], SOUTH[NUM_SQ], NORTH[NUM_SQ];  // neighbours or SENT
uint8_t CROSSED[2][NUM_SQ];  // a soldier of colour c on s has crossed the river

void init_fast2() {
    for (int i = 0; i < 256; ++i) VV[i] = uint8_t(i != 0 && type_of(Piece(i)) != KING && (i & 15) <= DARK);
    for (int s = 0; s < NUM_SQ; ++s) {
        const int f = file_of(s), r = rank_of(s);
        NORTH[s] = uint8_t(r < 9 ? s + 9 : SENT); SOUTH[s] = uint8_t(r > 0 ? s - 9 : SENT);
        WST[s] = uint8_t(f > 0 ? s - 1 : SENT); EST[s] = uint8_t(f < 8 ? s + 1 : SENT);
        FWD[RED][s] = NORTH[s]; FWD[BLACK][s] = SOUTH[s];
        CROSSED[RED][s] = uint8_t(!own_half(RED, s)); CROSSED[BLACK][s] = uint8_t(!own_half(BLACK, s));
    }
}

struct Gen2 {
    Piece* b;      // 96 bytes, b[90..95] = 0
    uint8_t* K;    // movement kind per square (face-down -> darkRole), 96 bytes, K[90..95] = 0
    uint32_t* out;
    int n = 0;
    BB occ = 0, hz = 0, el = 0, ad = 0;
    // branchless emit: the slot is written unconditionally, kept only if c
    inline void emit_if(bool c, bool add, Piece a, int from, Piece v, int to) {
        out[n] = pack(add, a, from, v, to);
        n += int(c);
    }
    // occupancy / kind sets / K from the source board src (not from the just-copied scratch board: a wide load right
    // after the byte copy would stall on store forwarding). Chunks at 0, 32 and 58 (58..89, no over-read).
    void build(const Piece* src) {
#if defined(__AVX2__)
        uint64_t mo[3], mh[3], me[3], ma[3];
        const __m256i z = _mm256_setzero_si256(), m15 = _mm256_set1_epi8(15), m16 = _mm256_set1_epi8(16);
        for (int k = 0; k < 3; ++k) {
            const int off = k < 2 ? 32 * k : 58;
            const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + off));
            const __m256i t = _mm256_and_si256(v, m15);
            const __m256i blk = _mm256_cmpeq_epi8(_mm256_and_si256(v, m16), m16);
            const __m256i dr = _mm256_blendv_epi8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(DR[0] + off)),
                                                  _mm256_loadu_si256(reinterpret_cast<const __m256i*>(DR[1] + off)), blk);
            const __m256i mk = _mm256_blendv_epi8(t, dr, _mm256_cmpeq_epi8(t, _mm256_set1_epi8(DARK)));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(K + off), mk);
            const int sh = k < 2 ? 0 : 6;
            mo[k] = uint32_t(~_mm256_movemask_epi8(_mm256_cmpeq_epi8(v, z))) >> sh;
            mh[k] = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(mk, _mm256_set1_epi8(HORSE)))) >> sh;
            me[k] = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(mk, _mm256_set1_epi8(ELEPHANT)))) >> sh;
            ma[k] = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(mk, _mm256_set1_epi8(ADVISOR)))) >> sh;
        }
        K[90] = K[91] = K[92] = K[93] = K[94] = K[95] = 0;
        auto cat = [](const uint64_t* m) { return BB(m[0] | (m[1] << 32)) | (BB(m[2]) << 64); };
        occ = cat(mo); hz = cat(mh); el = cat(me); ad = cat(ma);
#else
        for (int s = 0; s < 96; ++s) K[s] = 0;
        for (int s = 0; s < NUM_SQ; ++s) if (src[s]) set(s, src[s]);
#endif
    }
    inline void clear(int s) { const BB m = ~bit(s); occ &= m; hz &= m; el &= m; ad &= m; K[s] = 0; }
    inline void set(int s, Piece p) {
        occ |= bit(s);
        const PieceType k = mkind(p, s);
        K[s] = uint8_t(k);
        if (k == HORSE) hz |= bit(s);
        else if (k == ELEPHANT) el |= bit(s);
        else if (k == ADVISOR) ad |= bit(s);
    }
    inline void scan(int s, int (&f1)[4], int (&f2)[4]) const {
        const int f = file_of(s), r = rank_of(s);
        const uint32_t ro = uint32_t(occ >> (9 * r)) & 0x1FFu & ~(1u << f);
        uint32_t fo = 0;
        for (int i = 0; i < 10; ++i) fo |= uint32_t(b[f + 9 * i] != 0) << i;
        fo &= ~(1u << r);
        const int8_t* hf = HITF[r][fo];
        const int8_t* hr = HITR[f][ro];
        auto fx = [](int x) { return x < 0 ? SENT : x; };  // cmov
        f1[0] = fx(hf[0] + f); f2[0] = fx(hf[1] + f); f1[1] = fx(hf[2] + f); f2[1] = fx(hf[3] + f);
        f1[2] = fx(hr[0] + 9 * r); f2[2] = fx(hr[1] + 9 * r); f1[3] = fx(hr[2] + 9 * r); f2[3] = fx(hr[3] + 9 * r);
    }
    void outgoing(Piece pc, int s, bool add, const int (&f1)[4], const int (&f2)[4]) {
        const Color c = color_of(pc);
        const bool dark = type_of(pc) == DARK;
        switch (mkind(pc, s)) {
        case ROOK: for (int d = 0; d < 4; ++d) emit_if(VV[b[f1[d]]], add, pc, s, b[f1[d]], f1[d]); break;
        case CANNON: for (int d = 0; d < 4; ++d) emit_if(VV[b[f2[d]]], add, pc, s, b[f2[d]], f2[d]); break;
        case HORSE: each(HTO[s] & occ, [&](int t) { emit_if(!b[LEG[s][t]] & VV[b[t]], add, pc, s, b[t], t); }); break;
        case ELEPHANT: {
            BB m = ETO[s] & occ;
            if (dark) m &= HALF[c];
            each(m, [&](int t) { emit_if(!b[EYE[s][t]] & VV[b[t]], add, pc, s, b[t], t); });
        } break;
        case ADVISOR: {
            BB m = ATO[s] & occ;
            if (dark) m &= PAL[c];
            each(m, [&](int t) { emit_if(VV[b[t]], add, pc, s, b[t], t); });
        } break;
        case SOLDIER: {
            const int t = FWD[c][s];
            emit_if(VV[b[t]], add, pc, s, b[t], t);
            if (CROSSED[c][s]) {
                const int w = WST[s], e = EST[s];
                emit_if(VV[b[w]], add, pc, s, b[w], w);
                emit_if(VV[b[e]], add, pc, s, b[e], e);
            }
        } break;
        default: break;
        }
    }
    // v is a valid victim (caller guarantees: not a king); v2 may be 0 (no swap)
    void incoming(int s, Piece v, bool add, Piece v2, const int (&f1)[4], const int (&f2)[4]) {
        const bool h2 = v2 != 0;
        auto em = [&](bool c, int from) {
            emit_if(c, add, b[from], from, v, s);
            emit_if(c & h2, !add, b[from], from, v2, s);
        };
        for (int d = 0; d < 4; ++d) {
            em(K[f1[d]] == ROOK, f1[d]);
            em(K[f2[d]] == CANNON, f2[d]);
        }
        if (BB m = HTO[s] & hz) each(m, [&](int p) { em(!b[LEG[p][s]], p); });
        if (BB m = ETO[s] & el)
            each(m, [&](int p) { em(!b[EYE[p][s]] & (type_of(b[p]) != DARK || own_half(color_of(b[p]), s)), p); });
        if (BB m = ATO[s] & ad) each(m, [&](int p) { em(type_of(b[p]) != DARK || in_palace(color_of(b[p]), s), p); });
        const int so = SOUTH[s], no = NORTH[s], w = WST[s], e = EST[s];
        em(K[so] == SOLDIER && color_of(b[so]) == RED, so);
        em(K[no] == SOLDIER && color_of(b[no]) == BLACK, no);
        em(K[w] == SOLDIER && CROSSED[color_of(b[w])][w], w);
        em(K[e] == SOLDIER && CROSSED[color_of(b[e])][e], e);
    }
    void discovered(int s, bool add, const int (&f1)[4], const int (&f2)[4]) {
        for (int d = 0; d < 4; ++d) {
            const int od = d ^ 1, x1 = f1[d], x2 = f2[d], y1 = f1[od], y2 = f2[od];
            const int k1 = K[x1];
            const bool v1 = VV[b[y1]];
            emit_if((k1 == ROOK) & v1, !add, b[x1], x1, b[y1], y1);
            emit_if((k1 == CANNON) & v1, add, b[x1], x1, b[y1], y1);
            emit_if((k1 == CANNON) & VV[b[y2]], !add, b[x1], x1, b[y2], y2);
            emit_if((K[x2] == CANNON) & v1, !add, b[x2], x2, b[y1], y1);
        }
        if (LEGOWN[s] & hz)
            for (int j = 0; j < legN[s]; ++j) {
                const int h = legH[s][j], t = legT[s][j];
                if ((hz & bit(h)) && VV[b[t]]) emit_if(true, !add, b[h], h, b[t], t);
            }
        if (EYEOWN[s] & el)
            for (int j = 0; j < eyeN[s]; ++j) {
                const int e = eyeE[s][j], t = eyeT[s][j];
                if ((el & bit(e)) && VV[b[t]] && (type_of(b[e]) != DARK || own_half(color_of(b[e]), t)))
                    emit_if(true, !add, b[e], e, b[t], t);
            }
    }
};
}  // namespace

int gen_move(const Piece* b0, Move m, uint32_t* out, Piece* after) {
    alignas(32) Piece lb[96];
    std::memcpy(lb, b0, NUM_SQ);
    std::memset(lb + NUM_SQ, 0, 96 - NUM_SQ);
    const int from = from_sq(m), to = to_sq(m);
    const Piece pc = lb[from], cap = lb[to];
    const Piece np = type_of(pc) == DARK ? make_piece(color_of(pc), reveal_of(m)) : pc;
    alignas(32) uint8_t K[96];
    Gen2 g{lb, K, out};
    g.build(b0);
    int f1[4], f2[4];
    g.scan(from, f1, f2);
    g.outgoing(pc, from, false, f1, f2);
    if (type_of(pc) != KING) g.incoming(from, pc, false, 0, f1, f2);
    g.discovered(from, false, f1, f2);
    lb[from] = NO_PIECE;
    g.clear(from);
    g.scan(to, f1, f2);
    if (cap) {
        g.outgoing(cap, to, false, f1, f2);
        g.incoming(to, cap, false, type_of(np) == KING ? Piece(0) : np, f1, f2);
        lb[to] = np;
        g.clear(to);
        g.set(to, np);
        g.outgoing(np, to, true, f1, f2);
    } else {
        g.outgoing(np, to, true, f1, f2);
        if (type_of(np) != KING) g.incoming(to, np, true, 0, f1, f2);
        g.discovered(to, true, f1, f2);
        lb[to] = np;
        g.set(to, np);
    }
    std::memcpy(after, lb, NUM_SQ);
    return g.n;
}

int gen_all(const Piece* b, uint32_t* out) {
    Gen g{b, out};
    int f1[4], f2[4];
    for (int s = 0; s < NUM_SQ; ++s) {
        if (!b[s] || type_of(b[s]) == KING) continue;
        g.scan(s, f1, f2);
        g.outgoing(b[s], s, true, f1, f2);
    }
    return g.n;
}

int active(const Piece* b, int p, int m, uint32_t* idx) {
    uint32_t raw[MAX_ACTIVE];
    const int n = gen_all(b, raw);
    int k = 0;
    for (int i = 0; i < n; ++i) {
        const int x = index_of(raw[i], p, m);
        if (x >= 0) idx[k++] = uint32_t(x);
    }
    return k;
}

}  // namespace jq::thr

// =====================================================================================================================
// h32fatt Gen3 (grok executor 2026-10-29): bitboard-input generator. Same tuple multiset as gen_move / gen_move_ref.
//  - no AVX2 rebuild of occupancy / kind sets / K[] from the board: the caller passes Position's incremental bitboards
//  - first / second blockers of the 8 rays: rank occupancy by shift, file occupancy by PEXT (two 64-bit halves)
//  - incoming attackers (rook / cannon via the blocker squares, leapers via masks, soldiers via SoldierFrom-like masks)
//    are collected into one 128-bit set, then emitted (avg ~1 attacker instead of 12 branchless checks x 2 emits)
//  - role tests are bit tests on the role sets (no K[] array)
// =====================================================================================================================
namespace jq::thr {
namespace {
using BB = Bitboard;
BB SQB[128];                     // square masks (index 95 = sentinel -> bit 95, never set in any set)
uint64_t FLO[9], FHI[9]; int FNLO[9];   // file masks split into 64-bit halves, popcount of the low half
BB SFROM[2][NUM_SQ], STO3[2][NUM_SQ];             // squares from which a colour-c soldier attacks s
BB HTO3[NUM_SQ], ETO3[NUM_SQ], ATO3[NUM_SQ], LEGOWN3[NUM_SQ], EYEOWN3[NUM_SQ], HALF3[2], PAL3[2];
uint8_t LEG3[NUM_SQ][NUM_SQ], EYE3[NUM_SQ][NUM_SQ];
uint8_t legH3[NUM_SQ][8], legT3[NUM_SQ][8], legN3[NUM_SQ];
uint8_t eyeE3[NUM_SQ][4], eyeT3[NUM_SQ][4], eyeN3[NUM_SQ];
int8_t HF3[10][1024][4];         // as HITF / HITR
int8_t HR3[9][512][4];
uint8_t VV3[256];
uint8_t MK[32][96];              // movement kind of piece p on s (face-down -> darkRole), 0 for empty / off-board
uint8_t HALFOF[NUM_SQ];          // colour whose own half contains s
bool g3init = false;
constexpr int SENT3 = 95;
void init3() {
    if (g3init) return;
    for (int s = 0; s < 128; ++s) SQB[s] = BB(1) << s;
    for (int f = 0; f < 9; ++f) {
        FLO[f] = FHI[f] = 0;
        for (int r = 0; r < 10; ++r) { const int s = r * 9 + f; if (s < 64) FLO[f] |= 1ull << s; else FHI[f] |= 1ull << (s - 64); }
        FNLO[f] = __builtin_popcountll(FLO[f]);
    }
    std::memset(HTO3, 0, sizeof HTO3); std::memset(ETO3, 0, sizeof ETO3); std::memset(ATO3, 0, sizeof ATO3);
    std::memset(LEGOWN3, 0, sizeof LEGOWN3); std::memset(EYEOWN3, 0, sizeof EYEOWN3); std::memset(SFROM, 0, sizeof SFROM); std::memset(STO3, 0, sizeof STO3);
    std::memset(legN3, 0, sizeof legN3); std::memset(eyeN3, 0, sizeof eyeN3);
    HALF3[0] = HALF3[1] = PAL3[0] = PAL3[1] = 0;
    for (int s = 0; s < NUM_SQ; ++s) {
        for (int j = 0; j < T.horseN[s]; ++j) {
            const int t = T.horseTo[s][j], l = T.horseLeg[s][j];
            HTO3[s] |= SQB[t]; LEG3[s][t] = uint8_t(l); LEGOWN3[l] |= SQB[s];
            legH3[l][legN3[l]] = uint8_t(s); legT3[l][legN3[l]++] = uint8_t(t);
        }
        for (int j = 0; j < T.elephN[s]; ++j) {
            const int t = T.elephTo[s][j], e = T.elephEye[s][j];
            ETO3[s] |= SQB[t]; EYE3[s][t] = uint8_t(e); EYEOWN3[e] |= SQB[s];
            eyeE3[e][eyeN3[e]] = uint8_t(s); eyeT3[e][eyeN3[e]++] = uint8_t(t);
        }
        for (int j = 0; j < T.advN[s]; ++j) ATO3[s] |= SQB[T.advTo[s][j]];
        for (int c = 0; c < 2; ++c) {
            if (own_half(Color(c), s)) { HALF3[c] |= SQB[s]; HALFOF[s] = uint8_t(c); }
            if (in_palace(Color(c), s)) PAL3[c] |= SQB[s];
            const int t = s + (c == RED ? 9 : -9);
            if (t >= 0 && t < NUM_SQ) { SFROM[c][t] |= SQB[s]; STO3[c][s] |= SQB[t]; }
            if (!own_half(Color(c), s)) {
                if (file_of(s) > 0) { SFROM[c][s - 1] |= SQB[s]; STO3[c][s] |= SQB[s - 1]; }
                if (file_of(s) < 8) { SFROM[c][s + 1] |= SQB[s]; STO3[c][s] |= SQB[s + 1]; }
            }
        }
    }
    auto line = [](int len, int pos, uint32_t occ, int mul, int8_t* o) {
        o[0] = o[1] = o[2] = o[3] = -100;
        int k = 0;
        for (int i = pos + 1; i < len && k < 2; ++i) if (occ >> i & 1) o[k++] = int8_t(i * mul);
        k = 0;
        for (int i = pos - 1; i >= 0 && k < 2; --i) if (occ >> i & 1) o[2 + k++] = int8_t(i * mul);
    };
    for (int p = 0; p < 10; ++p) for (uint32_t o = 0; o < 1024; ++o) line(10, p, o, 9, HF3[p][o]);
    for (int p = 0; p < 9; ++p) for (uint32_t o = 0; o < 512; ++o) line(9, p, o, 1, HR3[p][o]);
    for (int i = 0; i < 256; ++i) VV3[i] = uint8_t(i != 0 && type_of(Piece(i)) != KING && (i & 15) <= DARK);
    std::memset(MK, 0, sizeof MK);
    for (int p = 1; p < 32; ++p) {
        if ((p & 15) == 0 || (p & 15) > DARK) continue;
        for (int s = 0; s < NUM_SQ; ++s) MK[p][s] = uint8_t(mkind(Piece(p), s));
    }
    g3init = true;
}
inline bool tb(BB m, int s) { return (m & SQB[s]) != 0; }
struct Gen3 {
    Piece* b;      // local board, 96 bytes, b[90..95] = 0
    uint32_t* out;
    int n = 0;
    BB occ, rk, cn, hz, el, ad, so, red, blk, dk, kg;
    inline void emit_if(bool c, bool add, Piece a, int from, Piece v, int to) {
        out[n] = pack(add, a, from, v, to);
        n += int(c);
    }
    inline void emit(bool add, Piece a, int from, Piece v, int to) { out[n++] = pack(add, a, from, v, to); }
    inline BB* rolebb(int k) {
        switch (k) { case ROOK: return &rk; case CANNON: return &cn; case HORSE: return &hz; case ELEPHANT: return &el;
                     case ADVISOR: return &ad; case SOLDIER: return &so; case KING: return &kg; default: return nullptr; }
    }
    inline void clear(int s, Piece p) {
        const BB m = ~SQB[s];
        occ &= m; red &= m; blk &= m; dk &= m;
        if (BB* r = rolebb(MK[p][s])) *r &= m;
    }
    inline void set(int s, Piece p) {
        const BB m = SQB[s];
        occ |= m;
        if (color_of(p) == RED) red |= m; else blk |= m;
        if (type_of(p) == DARK) dk |= m;
        if (BB* r = rolebb(MK[p][s])) *r |= m;
    }
    inline void scan(int s, int (&f1)[4], int (&f2)[4]) const {
        const int f = file_of(s), r = rank_of(s);
        const uint64_t lo = uint64_t(occ), hi = uint64_t(occ >> 64);
        const uint32_t ro = uint32_t(occ >> (9 * r)) & 0x1FFu & ~(1u << f);
        const uint32_t fo = (uint32_t(_pext_u64(lo, FLO[f])) | (uint32_t(_pext_u64(hi, FHI[f])) << FNLO[f])) & ~(1u << r);
#ifdef JQ_THR_SCAN_TABLE
        const int8_t* hf = HF3[r][fo];
        const int8_t* hr = HR3[f][ro];
        auto fx = [](int x) { return x < 0 ? SENT3 : x; };
        f1[0] = fx(hf[0] + f); f2[0] = fx(hf[1] + f); f1[1] = fx(hf[2] + f); f2[1] = fx(hf[3] + f);
        f1[2] = fx(hr[0] + 9 * r); f2[2] = fx(hr[1] + 9 * r); f1[3] = fx(hr[2] + 9 * r); f2[3] = fx(hr[3] + 9 * r);
#else   // table-free: tzcnt / lzcnt on the line occupancy (no 58 KB line tables in L1)
        {
            uint32_t u = fo >> (r + 1);
            const int t1 = int(_tzcnt_u32(u)); u = _blsr_u32(u); const int t2 = int(_tzcnt_u32(u));
            f1[0] = t1 < 32 ? (r + 1 + t1) * 9 + f : SENT3; f2[0] = t2 < 32 ? (r + 1 + t2) * 9 + f : SENT3;
            const uint32_t dn = fo & ((1u << r) - 1);
            const int d1 = 31 - int(_lzcnt_u32(dn)); const int d2 = 31 - int(_lzcnt_u32(_bzhi_u32(dn, uint32_t(d1) & 255u)));
            f1[1] = d1 >= 0 ? d1 * 9 + f : SENT3; f2[1] = d2 >= 0 ? d2 * 9 + f : SENT3;
        }
        {
            uint32_t u = ro >> (f + 1);
            const int t1 = int(_tzcnt_u32(u)); u = _blsr_u32(u); const int t2 = int(_tzcnt_u32(u));
            f1[2] = t1 < 32 ? 9 * r + f + 1 + t1 : SENT3; f2[2] = t2 < 32 ? 9 * r + f + 1 + t2 : SENT3;
            const uint32_t dn = ro & ((1u << f) - 1);
            const int d1 = 31 - int(_lzcnt_u32(dn)); const int d2 = 31 - int(_lzcnt_u32(_bzhi_u32(dn, uint32_t(d1) & 255u)));
            f1[3] = d1 >= 0 ? 9 * r + d1 : SENT3; f2[3] = d2 >= 0 ? 9 * r + d2 : SENT3;
        }
#endif
    }
    // targets (valid victims) of piece pc on s, as one set; branch-free over the movement kind
    inline BB targets(Piece pc, int s, const int (&f1)[4], const int (&f2)[4]) const {
        const int k = MK[pc][s];
        const Color c = color_of(pc);
        const bool dark = type_of(pc) == DARK;
        auto mk = [](bool x) { return BB(0) - BB(x); };
        BB t = mk(k == ROOK) & (SQB[f1[0]] | SQB[f1[1]] | SQB[f1[2]] | SQB[f1[3]]);
        t |= mk(k == CANNON) & (SQB[f2[0]] | SQB[f2[1]] | SQB[f2[2]] | SQB[f2[3]]);
        if (k == HORSE) { BB m = HTO3[s] & occ; for_each(m, [&](int x) { if (!b[LEG3[s][x]]) t |= SQB[x]; }); }
        else if (k == ELEPHANT) { BB m = ETO3[s] & occ & (dark ? HALF3[c] : ~BB(0)); for_each(m, [&](int x) { if (!b[EYE3[s][x]]) t |= SQB[x]; }); }
        t |= mk(k == ADVISOR) & ATO3[s] & (dark ? PAL3[c] : ~BB(0));
        t |= mk(k == SOLDIER) & STO3[c][s];
        return t & occ & ~kg;
    }
    void outgoing(Piece pc, int s, bool add, const int (&f1)[4], const int (&f2)[4]) {
        for_each(targets(pc, s, f1, f2), [&](int x) { emit(add, pc, s, b[x], x); });
    }
    template <class F> static inline void for_each(BB m, F&& f) {
        uint64_t lo = uint64_t(m), hi = uint64_t(m >> 64);
        while (lo) { f(__builtin_ctzll(lo)); lo &= lo - 1; }
        while (hi) { f(64 + __builtin_ctzll(hi)); hi &= hi - 1; }
    }
    // v is a valid victim (not a king); v2 may be 0 (no swap)
    void incoming(int s, Piece v, bool add, Piece v2, const int (&f1)[4], const int (&f2)[4]) {
        BB A = ((SQB[f1[0]] | SQB[f1[1]] | SQB[f1[2]] | SQB[f1[3]]) & rk) | ((SQB[f2[0]] | SQB[f2[1]] | SQB[f2[2]] | SQB[f2[3]]) & cn);
        A |= (SFROM[RED][s] & red | SFROM[BLACK][s] & blk) & so;
        if (BB m = HTO3[s] & hz) for_each(m, [&](int p) { if (!b[LEG3[p][s]]) A |= SQB[p]; });
        // face-down elephants only within their own half, face-down advisors only into their palace
        if (BB m = ETO3[s] & el & ~(dk & (HALFOF[s] == RED ? blk : red))) for_each(m, [&](int p) { if (!b[EYE3[p][s]]) A |= SQB[p]; });
        A |= ATO3[s] & ad & ~(dk & ((tb(PAL3[RED], s) ? BB(0) : red) | (tb(PAL3[BLACK], s) ? BB(0) : blk)));
        if (v2) for_each(A, [&](int p) { emit(add, b[p], p, v, s); emit(!add, b[p], p, v2, s); });
        else for_each(A, [&](int p) { emit(add, b[p], p, v, s); });
    }
    void discovered(int s, bool add, const int (&f1)[4], const int (&f2)[4]) {
        for (int d = 0; d < 4; ++d) {
            const int od = d ^ 1, x1 = f1[d], x2 = f2[d], y1 = f1[od], y2 = f2[od];
            const bool v1 = VV3[b[y1]], r1 = tb(rk, x1), c1 = tb(cn, x1);
            emit_if(r1 & v1, !add, b[x1], x1, b[y1], y1);
            emit_if(c1 & v1, add, b[x1], x1, b[y1], y1);
            emit_if(c1 & VV3[b[y2]], !add, b[x1], x1, b[y2], y2);
            emit_if(tb(cn, x2) & v1, !add, b[x2], x2, b[y1], y1);
        }
        if (LEGOWN3[s] & hz)
            for (int j = 0; j < legN3[s]; ++j) {
                const int h = legH3[s][j], t = legT3[s][j];
                if (tb(hz, h) && VV3[b[t]]) emit(!add, b[h], h, b[t], t);
            }
        if (EYEOWN3[s] & el)
            for (int j = 0; j < eyeN3[s]; ++j) {
                const int e = eyeE3[s][j], t = eyeT3[s][j];
                if (tb(el, e) && VV3[b[t]] && (type_of(b[e]) != DARK || own_half(color_of(b[e]), t))) emit(!add, b[e], e, b[t], t);
            }
    }
};
}  // namespace

int gen_move_bb(const Piece* b0, const BBIn& in, Move m, uint32_t* out, Piece* after) {
    alignas(32) Piece lb[96];
    std::memcpy(lb, b0, NUM_SQ);
    std::memset(lb + NUM_SQ, 0, 96 - NUM_SQ);
    const int from = from_sq(m), to = to_sq(m);
    const Piece pc = lb[from], cap = lb[to];
    const Piece np = type_of(pc) == DARK ? make_piece(color_of(pc), reveal_of(m)) : pc;
    Gen3 g{lb, out};
    g.occ = in.occ; g.rk = in.role[ROOK]; g.cn = in.role[CANNON]; g.hz = in.role[HORSE]; g.el = in.role[ELEPHANT];
    g.ad = in.role[ADVISOR]; g.so = in.role[SOLDIER]; g.kg = in.role[KING]; g.red = in.color[RED]; g.blk = in.color[BLACK]; g.dk = in.dark;
    int f1[4], f2[4];
    g.scan(from, f1, f2);
    g.outgoing(pc, from, false, f1, f2);
    if (type_of(pc) != KING) g.incoming(from, pc, false, 0, f1, f2);
    g.discovered(from, false, f1, f2);
    g.clear(from, pc);
    lb[from] = NO_PIECE;
    g.scan(to, f1, f2);
    if (cap) {
        g.outgoing(cap, to, false, f1, f2);
        g.incoming(to, cap, false, type_of(np) == KING ? Piece(0) : np, f1, f2);
        g.clear(to, cap);
        lb[to] = np;
        g.set(to, np);
        g.outgoing(np, to, true, f1, f2);
    } else {
        g.outgoing(np, to, true, f1, f2);
        if (type_of(np) != KING) g.incoming(to, np, true, 0, f1, f2);
        g.discovered(to, true, f1, f2);
        lb[to] = np;
        g.set(to, np);
    }
    if (after) std::memcpy(after, lb, NUM_SQ);
    return g.n;
}
void init_bb3() { init3(); }
}  // namespace jq::thr
