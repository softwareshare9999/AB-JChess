// JQT1: Pikafish-style "full threat" input features for Jieqi (prototype, engine side).
//
// A feature is an attack (attacker piece, from) -> (victim piece, to) that is active in the position, oriented to a
// perspective P (P's home rank = rank 0, mirrored left-right with the HNN king-bucket mirror flag) and coloured
// relative to P (own = 0, enemy = 7). Kings are neither attackers nor victims (they still block / act as screens).
//   attacker kinds (7 per colour): revealed R N B A C P, and D = face-down (moves as darkRole of its square)
//   victim kinds   (7 per colour): revealed R N B A C P, and D = face-down (identity unknown)
// Geometry follows jqe movegen: revealed advisors / elephants move anywhere on the board (elephants cross the river,
// advisors leave the palace); face-down advisors only into their palace, face-down elephants only on their own half;
// soldiers forward, plus sideways once across the river; cannons need exactly one screen; horse legs and elephant
// eyes block.
// Index (dense, no holes): enumerate a = 0..13, from = 0..89, to = 0..89 ascending, v = 0..13 and count every valid
// tuple: from is a legal square of a (face-down: a start point of that colour; revealed: any square), to is in the
// unblocked attack geometry of a at from, v may stand on to (face-down: start point of that colour), and the tuple is
// not the "reverse duplicate" of another tuple (same movement kind, not a horse, the victim at to attacks from with the
// same geometry; then only from > to is kept). DIMS = 91521.
#pragma once
#include "bitboard.h"

#include <cstdint>

namespace jq::thr {

constexpr int NCODE = 14;
constexpr int MAX_ACTIVE = 192;   // active features of one perspective
constexpr int MAX_DIRTY = 256;    // raw tuple changes of one move
extern int DIMS;

// packed entry: bits 0-16 base index (DIMS < 2^17), bits 17-30 victim mask (14 relative victim codes)
extern uint32_t E[NCODE][NUM_SQ][NUM_SQ];
extern int8_t KREL[2][32];       // [persp][piece] -> relative code (own kind 0-6, enemy 7-13), -1 empty / king
extern uint8_t OT[2][2][NUM_SQ];  // [persp][mirror][abs square] -> oriented square
extern int8_t KIDX[32];          // piece code -> kind index (R N B A C P D = 0..6), -1 for empty / king

void init();

// raw tuple: to 0-6, from 7-13, victim piece 14-18, attacker piece 19-23, add bit 31
inline uint32_t pack(bool add, uint32_t a, int from, uint32_t v, int to) {
    return (uint32_t(add) << 31) | (a << 19) | (v << 14) | (uint32_t(from) << 7) | uint32_t(to);
}
// Index of a raw tuple for perspective p / mirror m; returns -1 for an excluded (reverse duplicate) tuple.
// h32fatt: compact entry table E2[a][from][SLOT[from][to]] (33 geometry slots per square: 17 file/rank squares, 8 horse
// targets, 4 one-step and 4 two-step diagonals; slot 33 = no geometry, entry 0) = E[a][from][to]; 166 KB instead of 453 KB.
constexpr int NSLOT = 34;
extern uint32_t E2[NCODE][NUM_SQ][NSLOT];
extern uint8_t SLOT[NUM_SQ][NUM_SQ];
inline int index_of(uint32_t raw, int p, int m) {
    const int from = OT[p][m][(raw >> 7) & 127], to = OT[p][m][raw & 127];
    const int ar = KREL[p][(raw >> 19) & 31], vr = KREL[p][(raw >> 14) & 31];
    const uint32_t e = E2[ar][from][SLOT[from][to]];
    if (!((e >> (17 + vr)) & 1)) return -1;
    return int(e & 0x1FFFFu) + __builtin_popcount((e >> 17) & ((1u << vr) - 1));
}
// original full-table version (tests)
inline int index_of_ref(uint32_t raw, int p, int m) {
    const int from = OT[p][m][(raw >> 7) & 127], to = OT[p][m][raw & 127];
    const int ar = KREL[p][(raw >> 19) & 31], vr = KREL[p][(raw >> 14) & 31];
    const uint32_t e = E[ar][from][to];
    if (!((e >> (17 + vr)) & 1)) return -1;
    return int(e & 0x1FFFFu) + __builtin_popcount((e >> 17) & ((1u << vr) - 1));
}

// Raw tuple changes of move m on board b (position before the move); the board copy `after` receives the board after
// the move. A face-down mover must carry its reveal type in m.
int gen_move(const Piece* b, Move m, uint32_t* out, Piece* after);
int gen_move_ref(const Piece* b, Move m, uint32_t* out, Piece* after);  // simple reference generator (tests)
// h32fatt fast generator (bitboard input): same tuple multiset as gen_move, but occupancy / role / colour / face-down
// sets come from the caller (Position keeps them incrementally) instead of being rebuilt from the board bytes; incoming
// attackers are collected as one bitboard; `after` may be nullptr (board after the move not needed).
struct BBIn {
    Bitboard occ, dark;
    const Bitboard* role;   // role[t]: pieces moving as t (face-down: darkRole of the square), t = 0..8
    const Bitboard* color;  // color[c]
};
int gen_move_bb(const Piece* b, const BBIn& in, Move m, uint32_t* out, Piece* after);
// All active raw tuples of a board (add bit set).
int gen_all(const Piece* b, uint32_t* out);
// Active feature indices of perspective p / mirror m.
int active(const Piece* b, int p, int m, uint32_t* idx);

}  // namespace jq::thr
