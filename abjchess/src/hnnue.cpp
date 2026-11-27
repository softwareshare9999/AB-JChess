#include "hnnue.h"
#include "jqnz.h"
#include "jqmd5.h"
#include <chrono>

#include "mem.h"
#include "nnkern.h"
#include "headkern.h"
#include "headfast.h"
#ifndef JQ_THR_PF_LINES
#define JQ_THR_PF_LINES 2
#endif

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace jq::hnn {

int g_scale = 1000;

namespace {

constexpr int NSTACKS = 16, L1 = nnk::L1, L2 = nnk::L2;
constexpr int PIECE_FEATS = 16 * NUM_SQ;  // per king bucket
constexpr int URN_FEATS = 30;
constexpr int URN_OFF[NUM_SLOTS] = {0, 2, 4, 6, 8, 10};
constexpr int PIECE_VAL[NUM_SLOTS] = {943, 438, 168, 190, 491, 101};
constexpr char MAGIC[8] = {'J', 'Q', 'H', 'N', 'N', 'U', 'E', '1'};
constexpr int KB_OF_SCHEME[4] = {1, 3, 2, 6};

struct FreeDel { void operator()(void* p) const { jq_aligned_free(p); } };
int g_W = 0, g_scheme = 0, g_nfeat = 0, g_urnBase = 0;
bool g_loaded = false;
std::atomic<unsigned> g_gen{0};
std::unique_ptr<int16_t[], FreeDel> g_ftW;
std::unique_ptr<int32_t[], FreeDel> g_psqt;
alignas(64) int16_t g_ftB[MAX_W];
std::unique_ptr<uint8_t[], FreeDel> g_l1w;
// threats
int g_thrOn = 0, g_thrInit = 0, g_thrAmp = 3, g_thrWt = 0;  // g_thrWt: threat row width (0 = full width W)
uint64_t g_thrSeed = 1;
int8_t* g_thrW = nullptr;
std::size_t g_thrWSize = 0;
std::vector<int8_t> g_thrFile;   // JQT1THR1 trailer rows (DIMS x g_thrFileWt); non-empty = threats forced on
int g_thrFileWt = 0, g_thrForced = 0;
#ifdef JQ_THR_STATS
std::atomic<uint64_t> st_makes{0}, st_gens{0}, st_tuples{0}, st_rows{0}, st_refr{0};
#define THRST(x, n) (x.fetch_add(n, std::memory_order_relaxed))
#else
#define THRST(x, n) ((void)0)
#endif
void thr_alloc() {
    if (!g_thrFile.empty()) { g_thrOn = 1; g_thrWt = g_thrFileWt; }
    if (!g_thrOn || !g_W) return;
    thr::init();
    if (g_thrWt <= 0 || g_thrWt > g_W) g_thrWt = g_W;
    const std::size_t need = std::size_t(thr::DIMS) * std::size_t(g_thrWt);
    if (g_thrW && g_thrWSize != need) { jq_aligned_free(g_thrW); g_thrW = nullptr; }
    if (!g_thrW) {
        g_thrW = static_cast<int8_t*>(alloc_large(need));
        g_thrWSize = need;
    }
    if (!g_thrFile.empty()) { std::memcpy(g_thrW, g_thrFile.data(), need); return; }
    uint64_t x = g_thrSeed * 0x9E3779B97F4A7C15ull + 1;
    for (std::size_t i = 0; i < need; ++i) {  // every page written (no shared zero page in the speed test)
        int8_t v = 0;
        if (g_thrInit) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            v = int8_t(int(x % uint64_t(2 * g_thrAmp + 1)) - g_thrAmp);
        }
        g_thrW[i] = v;
    }
}
// Add all active threat rows of `b` for perspective p / mirror m into v (in place).
void add_threats(const Piece* b, int p, int m, int16_t* v, int32_t* ps) {
    uint32_t idx[thr::MAX_ACTIVE];
    const int n = thr::active(b, p, m, idx);
    THRST(st_rows, uint64_t(n));
    nnk::apply_thr(nullptr, nullptr, g_W, v, ps, v, ps, nullptr, 0, nullptr, 0, g_thrW, g_thrWt, idx, n, nullptr, 0);
}
alignas(64) int32_t g_l1b[NSTACKS][L1];
alignas(64) int32_t g_l2b[NSTACKS][L2];
alignas(64) int8_t g_l2w[NSTACKS][L2 * L2];
int32_t g_fb[NSTACKS];
alignas(64) int8_t g_fw[NSTACKS][L2];
// version-3 head (configurable L1/L2/sq2); g_head = 0 for version-1 nets (original arrays and kernel above)
int g_head = 0, g_hL1 = 16, g_hL2 = 32, g_hSq2 = 0;
alignas(64) int32_t g3_l1b[NSTACKS][32];
alignas(64) int32_t g3_l2b[NSTACKS][64];
alignas(64) int8_t g3_l2w[NSTACKS][64 * 64];
int32_t g3_fb[NSTACKS];
alignas(64) int8_t g3_fw[NSTACKS][128];
// h32fast: re-laid-out stacks for the AVX2 fast kernel (headfast.h); g_fast = L1 (16 or 32) when the net's head is
// W -> L1 -> 32 -> 1 without sq2 (all version-1 nets, version-3 L2 32 no sq2), else 0 (generic kernels above)
int g_fast = 0;
alignas(64) nnk::FastHead g_fh[NSTACKS];
// Per scheme and oriented palace point (rank 0..2, file 3..5 -> 3 r + f - 3): bucket and mirror flag.
int8_t g_kbTab[9], g_mirTab[9];

inline int orient(Color persp, int s) { return persp == RED ? s : (9 - rank_of(s)) * 9 + file_of(s); }

void set_scheme(int scheme) {
    for (int r = 0; r < 3; ++r)
        for (int f = 3; f <= 5; ++f) {
            const int i = r * 3 + f - 3;
            int b = 0, m = 0;
            if (scheme == 1) b = f - 3;
            else if (scheme == 2) { m = f > 4; b = f == 4; }
            else if (scheme == 3) { m = f > 4; b = 2 * r + (f == 4); }
            g_kbTab[i] = int8_t(b);
            g_mirTab[i] = int8_t(m);
        }
}

inline void king_state(Color persp, int ksq, int8_t& kb, int8_t& mir) {
    const int o = orient(persp, ksq);
    const int i = rank_of(o) * 3 + file_of(o) - 3;  // the king never leaves its palace
    kb = g_kbTab[i];
    mir = g_mirTab[i];
}

inline int piece_feat(Color persp, Piece p, int s, int kb, int mir) {
    int o = orient(persp, s);
    if (mir) o = rank_of(o) * 9 + 8 - file_of(o);
    return kb * PIECE_FEATS + ((color_of(p) == persp ? 0 : 8) + int(type_of(p)) - 1) * NUM_SQ + o;
}
inline int urn_feat(Color persp, Color owner, int slot, int j) {  // urn[owner][slot] >= j, j >= 1
    return g_urnBase + (owner == persp ? 0 : 15) + URN_OFF[slot] + j - 1;
}

int features(const Position& pos, Color persp, int kb, int mir, uint16_t* f) {
    int n = 0;
    for (int c = 0; c < 2; ++c)
        for (int i = 0; i < pos.pcount[c]; ++i) {
            const int s = pos.plist[c][i];
            f[n++] = uint16_t(piece_feat(persp, pos.board[s], s, kb, mir));
        }
    for (int c = 0; c < 2; ++c)
        for (int sl = 0; sl < NUM_SLOTS; ++sl)
            for (int j = 1; j <= pos.urn[c][sl]; ++j) f[n++] = uint16_t(urn_feat(persp, Color(c), sl, j));
    return n;
}

inline void apply(int16_t* dst, int32_t* pdst, const int16_t* src, const int32_t* psrc, const uint16_t* add, int nAdd,
                  const uint16_t* rem, int nRem) {
    nnk::apply(g_ftW.get(), g_psqt.get(), g_W, dst, pdst, src, psrc, add, nAdd, rem, nRem);
}

struct FinnyEntry {
    alignas(64) int16_t v[MAX_W];
    int32_t ps[NPSQT];
    uint16_t f[64];
    int n = -1;
    unsigned gen = ~0u;
};
struct FinnyCache {
    FinnyEntry tab[2 * 6 * 2];
    FinnyCache() = default;
};
// Refresh of perspective p from the per-thread cache entry of its (bucket, mirror): only the difference between
// the cached and the new sorted feature lists is applied.
void finny_refresh(int p, int kb, int mir, uint16_t* f, int n, int16_t* out, int32_t* pout) {
    // Search helpers are short-lived on Windows. Keeping this cache in TLS storage avoids a heap allocation and
    // destructor callback on every helper thread; the MinGW/UCRT TLS teardown path is not robust under rapid churn.
    static thread_local FinnyCache cache;
    FinnyEntry* e = &cache.tab[(p * 6 + kb) * 2 + mir];
    const unsigned gen = g_gen.load(std::memory_order_relaxed);
    if (e->gen != gen) { e->n = -1; e->gen = gen; }
    std::sort(f, f + n);
    alignas(64) static const int32_t zero[NPSQT] = {};
    if (e->n < 0) {
        apply(e->v, e->ps, g_ftB, zero, f, n, nullptr, 0);
    } else {
        uint16_t add[64], rem[64];
        int na = 0, nr = 0, i = 0, j = 0;
        while (i < n || j < e->n) {
            if (j >= e->n || (i < n && f[i] < e->f[j])) add[na++] = f[i++];
            else if (i >= n || e->f[j] < f[i]) rem[nr++] = e->f[j++];
            else { ++i; ++j; }
        }
        if (na || nr) apply(e->v, e->ps, e->v, e->ps, add, na, rem, nr);
    }
    std::memcpy(e->f, f, sizeof(uint16_t) * std::size_t(n));
    e->n = n;
    std::memcpy(out, e->v, sizeof(int16_t) * std::size_t(g_W));
    std::memcpy(pout, e->ps, sizeof(e->ps));
}

}  // namespace

namespace { LoadInfo g_li; }
const LoadInfo& last_load() { return g_li; }
void bind_acc_stack(const Acc*, std::size_t, int) {}

bool is_hnn_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char m[8] = {};
    const bool ok = std::fread(m, 1, 8, f) == 8 &&
                    (std::memcmp(m, MAGIC, 8) == 0 || jqnz::is_compressed(reinterpret_cast<const uint8_t*>(m), 8));
    std::fclose(f);
    return ok;
}

bool load(const std::string& path, std::string* error) {
    auto fail = [&](const char* msg) { if (error) *error = msg; return false; };
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return fail("cannot open file");
    std::vector<uint8_t> buf;
    {
        std::fseek(f, 0, SEEK_END);
        const long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        buf.resize(sz > 0 ? std::size_t(sz) : 0);
        const bool ok = !buf.empty() && std::fread(buf.data(), 1, buf.size(), f) == buf.size();
        std::fclose(f);
        if (!ok) return fail("HNN: read error");
    }
    return load_buffer(buf.data(), buf.size(), error);
}

bool load_buffer(const uint8_t* data, std::size_t size, std::string* error) {
    auto fail = [&](const char* msg) { if (error) *error = msg; return false; };
    std::vector<uint8_t> dec;
    LoadInfo li;
    li.fileBytes = size;
    if (jqnz::is_compressed(data, size)) {  // LEB128 / range-coded container: decode to the exact original bytes
        li.fmt = std::string(reinterpret_cast<const char*>(data), 8);
        const auto t0 = std::chrono::steady_clock::now();
        if (!jqnz::decode(data, size, dec, error)) return false;
        li.decodeMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        data = dec.data();
        size = dec.size();
    }
    li.rawBytes = size;
    li.rawMd5 = md5_hex(data, size);
    struct View {
        const uint8_t* p; std::size_t n;
        std::size_t size() const { return n; }
        const uint8_t* data() const { return p; }
    } buf{data, size};
    if (buf.size() < 40 || std::memcmp(buf.data(), MAGIC, 8) != 0) return fail("HNN: bad magic");
    uint32_t h[8];
    std::memcpy(h, buf.data() + 8, sizeof h);
    const int W = int(h[1]), scheme = int(h[3]);
    const bool v3 = h[0] == 3;
    uint32_t ext[4] = {0, 0, 0, 0};
    if (v3) { if (buf.size() < 56) return fail("HNN: short v3 header"); std::memcpy(ext, buf.data() + 40, sizeof ext); }
    const int hL1 = int(h[6]), hL2 = int(h[7]), hSq2 = int(ext[0] & 1);
    if (!(h[0] == 1 || v3) || W < 128 || W > MAX_W || W % 128 || h[3] > 3 || h[4] != NPSQT || h[5] != NSTACKS ||
        (!v3 && (hL1 != L1 || hL2 != L2)) || (v3 && ((hL1 != 16 && hL1 != 32) || (hL2 != 32 && hL2 != 64) || (ext[0] & ~1u) || ext[1] || ext[2] || ext[3])))
        return fail("HNN: unsupported header (version 1: 16/16/16/32; version 3: L1 16|32, L2 32|64, flags sq2; W multiple of 128 up to 1024, scheme 0..3)");
    const int hL2IN = (2 * (hL1 - 1) + 31) / 32 * 32, hNO = hL2 * (hSq2 ? 2 : 1);
    const int nfeat = KB_OF_SCHEME[scheme] * PIECE_FEATS + URN_FEATS;
    if (int(h[2]) != nfeat) return fail("HNN: feature count does not match the king bucket scheme");
    const std::size_t headSz = v3 ? 4 * hL1 + std::size_t(W) * hL1 + 4 * hL2 + std::size_t(hL2) * hL2IN + 4 + hNO
                                  : 64 + std::size_t(W) * L1 + 128 + 1024 + 4 + 32;
    const std::size_t need = (v3 ? 56 : 40) + std::size_t(W) * 2 + std::size_t(nfeat) * W * 2 + std::size_t(nfeat) * NPSQT * 4 + NSTACKS * headSz;
    std::vector<int8_t> trl;
    int trlWt = 0;
    if (buf.size() > need) {  // optional JQT1 threat trailer: "JQT1THR1", u32 dims, Wt, 0, 0, int8 rows dims x Wt
        const std::size_t t = need;
        uint32_t th[4];
        if (buf.size() < t + 24 || std::memcmp(buf.data() + t, "JQT1THR1", 8) != 0) return fail("HNN: unexpected file size");
        std::memcpy(th, buf.data() + t + 8, sizeof th);
        thr::init();
        if (th[0] != uint32_t(thr::DIMS) || th[1] == 0 || th[1] % 32 || int(th[1]) > W || th[2] || th[3])
            return fail("HNN: bad JQT1 threat trailer header");
        if (buf.size() != t + 24 + std::size_t(th[0]) * th[1]) return fail("HNN: JQT1 threat trailer size");
        trl.assign(reinterpret_cast<const int8_t*>(buf.data() + t + 24), reinterpret_cast<const int8_t*>(buf.data() + buf.size()));
        trlWt = int(th[1]);
    }
    else if (buf.size() != need) return fail("HNN: unexpected file size");
    g_ftW.reset(static_cast<int16_t*>(alloc_large(std::size_t(nfeat) * W * sizeof(int16_t))));
    g_psqt.reset(static_cast<int32_t*>(alloc_large(std::size_t(nfeat) * NPSQT * sizeof(int32_t))));
    g_l1w.reset(static_cast<uint8_t*>(jq_aligned_alloc(64, std::size_t(NSTACKS) * W * (v3 ? hL1 : L1))));
    if (!g_ftW || !g_psqt || !g_l1w) return fail("HNN: out of memory");
    const uint8_t* p = buf.data() + (v3 ? 56 : 40);
    std::memset(g_ftB, 0, sizeof g_ftB);
    std::memcpy(g_ftB, p, std::size_t(W) * 2); p += std::size_t(W) * 2;
    std::memcpy(g_ftW.get(), p, std::size_t(nfeat) * W * 2); p += std::size_t(nfeat) * W * 2;
    std::memcpy(g_psqt.get(), p, std::size_t(nfeat) * NPSQT * 4); p += std::size_t(nfeat) * NPSQT * 4;
    if (v3) {
        std::memset(g3_l2w, 0, sizeof g3_l2w);
        for (int s = 0; s < NSTACKS; ++s) {
            std::memcpy(g3_l1b[s], p, 4 * hL1); p += 4 * hL1;
            std::memcpy(g_l1w.get() + std::size_t(s) * W * hL1, p, std::size_t(W) * hL1); p += std::size_t(W) * hL1;
            std::memcpy(g3_l2b[s], p, 4 * hL2); p += 4 * hL2;
            std::memcpy(g3_l2w[s], p, std::size_t(hL2) * hL2IN); p += std::size_t(hL2) * hL2IN;
            std::memcpy(&g3_fb[s], p, 4); p += 4;
            std::memcpy(g3_fw[s], p, hNO); p += hNO;
        }
    }
    for (int s = 0; s < NSTACKS && !v3; ++s) {
        std::memcpy(g_l1b[s], p, 64); p += 64;
        std::memcpy(g_l1w.get() + std::size_t(s) * W * L1, p, std::size_t(W) * L1); p += std::size_t(W) * L1;
        std::memcpy(g_l2b[s], p, 128); p += 128;
        std::memcpy(g_l2w[s], p, 1024); p += 1024;
        std::memcpy(&g_fb[s], p, 4); p += 4;
        std::memcpy(g_fw[s], p, 32); p += 32;
    }
    g_head = v3 ? (hL1 == 32 ? 2 : 0) + (hL2 == 64 ? 4 : 0) + (hSq2 ? 1 : 0) + 8 : 0;
    g_hL1 = v3 ? hL1 : L1; g_hL2 = v3 ? hL2 : L2; g_hSq2 = hSq2;
    g_fast = 0;
#ifdef JQ_PAIRWISE_NNZ
    if (!std::getenv("JQE_NO_FASTHEAD") && (!v3 || (hL2 == 32 && !hSq2))) {
        for (int s = 0; s < NSTACKS; ++s) {
            if (!v3) nnk::fast_head_init(g_fh[s], L1, W, g_l1b[s], g_l1w.get() + std::size_t(s) * W * L1, g_l2b[s], g_l2w[s], g_fb[s], g_fw[s]);
            else nnk::fast_head_init(g_fh[s], hL1, W, g3_l1b[s], g_l1w.get() + std::size_t(s) * W * hL1, g3_l2b[s], g3_l2w[s], g3_fb[s], g3_fw[s]);
        }
        g_fast = v3 ? hL1 : L1;
    }
#endif
    g_W = W;
    g_scheme = scheme;
    g_nfeat = nfeat;
    g_urnBase = KB_OF_SCHEME[scheme] * PIECE_FEATS;
    set_scheme(scheme);
    g_gen.fetch_add(1);
    g_loaded = true;
    g_li = std::move(li);
    if (!trl.empty()) { g_thrFile.swap(trl); g_thrFileWt = trlWt; g_thrForced = 1; }
    else if (g_thrForced) { g_thrFile.clear(); g_thrFileWt = 0; g_thrForced = 0; g_thrOn = 0; }
    thr_alloc();
    return true;
}

void unload() { g_loaded = false; }
bool loaded() { return g_loaded; }
int width() { return g_W; }
int kb_scheme() { return g_scheme; }
std::string head_desc() { return g_head ? "v3 L1 " + std::to_string(g_hL1) + " L2 " + std::to_string(g_hL2) + (g_hSq2 ? " sq2" : "") : "v1 L1 16 L2 32"; }

void set_threats(int on, int init, int amp, uint64_t seed, int width) {
    g_thrWt = width;
    g_thrOn = on; g_thrInit = init; g_thrAmp = amp; g_thrSeed = seed;
    thr_alloc();
}
bool threats_on() { return g_thrOn && g_thrW; }
int threat_width() { return threats_on() ? g_thrWt : 0; }
bool tacc() { return false; }
ThrStats thr_stats() {
#ifdef JQ_THR_STATS
    return {st_makes.load(), st_gens.load(), st_tuples.load(), st_rows.load(), st_refr.load()};
#else
    return {0, 0, 0, 0, 0};
#endif
}

void refresh(const Position& pos, Acc& acc) {
    Meta& m = acc.m;
    int material = 0;
    for (int c = 0; c < 2; ++c) {
        for (int sl = 0; sl < NUM_SLOTS; ++sl) material += pos.urn[c][sl] * PIECE_VAL[sl];
        for (int i = 0; i < pos.pcount[c]; ++i) {
            const PieceType t = type_of(pos.board[pos.plist[c][i]]);
            if (t >= ROOK && t <= CANNON) material += PIECE_VAL[type_slot(t)];
        }
    }
    m.material = int16_t(material);
    m.noCapture = pos.noCapture;
    m.pieces = int8_t(pos.pcount[0] + pos.pcount[1]);
    m.dark = int8_t(pos.darkCount[0] + pos.darkCount[1]);
    for (int p = 0; p < 2; ++p) {
        king_state(Color(p), pos.kingSq[p], m.kb[p], m.mirror[p]);
        uint16_t f[64];
        const int n = features(pos, Color(p), m.kb[p], m.mirror[p], f);
        finny_refresh(p, m.kb[p], m.mirror[p], f, n, acc.v[p], acc.ps[p]);
        if (g_thrOn) add_threats(pos.board, p, m.mirror[p], acc.v[p], acc.ps[p]);
    }
}

void refresh_nocache(const Position& pos, Acc& acc) {
    Meta& m = acc.m;
    refresh(pos, acc);  // meta (and a cached result we overwrite below)
    alignas(64) static const int32_t zero[NPSQT] = {};
    for (int p = 0; p < 2; ++p) {
        uint16_t f[64];
        const int n = features(pos, Color(p), m.kb[p], m.mirror[p], f);
        apply(acc.v[p], acc.ps[p], g_ftB, zero, f, n, nullptr, 0);
        if (g_thrOn) add_threats(pos.board, p, m.mirror[p], acc.v[p], acc.ps[p]);
    }
}

void make_delta(const Position& pos, Move mv, Delta& d) {
    const int from = from_sq(mv), to = to_sq(mv);
    const Piece pc = pos.board[from], cap = pos.board[to];
    const Color us = color_of(pc);
    Meta& m = d.m;
    int matD = 0;
    Piece np = pc;
    int revSlot = -1;
    if (type_of(pc) == DARK) {
        np = make_piece(us, reveal_of(mv));
        revSlot = type_slot(reveal_of(mv));
        if (revSlot == NUM_SLOTS - 1) matD -= PIECE_VAL[revSlot];  // urn value leaves, a soldier is not counted
    }
    if (cap && type_of(cap) >= ROOK && type_of(cap) <= CANNON) matD -= PIECE_VAL[type_slot(type_of(cap))];
    m.material = int16_t(matD);  // a delta here; update() adds the parent's material
    m.noCapture = cap ? 0 : int16_t(pos.noCapture + 1);
    m.pieces = int8_t(pos.pcount[0] + pos.pcount[1] - (cap ? 1 : 0));
    m.dark = int8_t(pos.darkCount[0] + pos.darkCount[1] - (type_of(pc) == DARK) - (cap && type_of(cap) == DARK));
    d.valid = true;
    d.nThr = -1;
    for (int p = 0; p < 2; ++p) {
        const Color persp = Color(p);
        king_state(persp, type_of(pc) == KING && us == persp ? to : pos.kingSq[p], m.kb[p], m.mirror[p]);
        int8_t kb0, mir0;
        king_state(persp, pos.kingSq[p], kb0, mir0);
        d.refP[p] = kb0 != m.kb[p] || mir0 != m.mirror[p];
        if (d.refP[p]) {  // only a king move: no capture of a face-down piece, no reveal
            uint16_t* f = d.full[p];
            int n = 0;
            for (int c = 0; c < 2; ++c)
                for (int i = 0; i < pos.pcount[c]; ++i) {
                    const int s = pos.plist[c][i];
                    if (s == from || s == to) continue;
                    f[n++] = uint16_t(piece_feat(persp, pos.board[s], s, m.kb[p], m.mirror[p]));
                }
            f[n++] = uint16_t(piece_feat(persp, np, to, m.kb[p], m.mirror[p]));
            for (int c = 0; c < 2; ++c)
                for (int sl = 0; sl < NUM_SLOTS; ++sl)
                    for (int j = 1; j <= pos.urn[c][sl]; ++j) f[n++] = uint16_t(urn_feat(persp, Color(c), sl, j));
            d.nFull[p] = int8_t(n);
            continue;
        }
        int nr = 0;
        d.rem[p][nr++] = uint16_t(piece_feat(persp, pc, from, m.kb[p], m.mirror[p]));
        if (cap) d.rem[p][nr++] = uint16_t(piece_feat(persp, cap, to, m.kb[p], m.mirror[p]));
        if (revSlot >= 0) d.rem[p][nr++] = uint16_t(urn_feat(persp, us, revSlot, pos.urn[us][revSlot]));
        d.add[p][0] = uint16_t(piece_feat(persp, np, to, m.kb[p], m.mirror[p]));
        d.nRem = int8_t(nr);
        d.nAdd = 1;
    }
    if (g_thrOn) {  // h32fatt: generated after the king states (the board after the move is only kept for a refresh)
        THRST(st_makes, 1);
        d.mv = mv;
#ifdef JQ_THR_LAZY
        std::memcpy(d.snap, pos.board, NUM_SQ);
#elif defined(JQ_THR_GEN2)
        d.nThr = int16_t(thr::gen_move(pos.board, mv, d.thr, d.snap));  // eager: snap = board AFTER the move
#elif !defined(JQ_THR_PREIDX)
        const thr::BBIn in{pos.occBB, pos.darkBB, pos.byRoleBB, pos.byColorBB};
#ifdef ABL_NOGEN
        d.nThr = 0; (void)in; if (d.refP[0] | d.refP[1]) std::memcpy(d.snap, pos.board, NUM_SQ);
#else
        d.nThr = int16_t(thr::gen_move_bb(pos.board, in, mv, d.thr, (d.refP[0] | d.refP[1]) ? d.snap : nullptr));
#endif
#else   // experimental: indices + row prefetch in make_delta (slower on the box, kept for reference)
        const thr::BBIn in{pos.occBB, pos.darkBB, pos.byRoleBB, pos.byColorBB};
        uint32_t raw[thr::MAX_DIRTY];
        const int nr = thr::gen_move_bb(pos.board, in, mv, raw, (d.refP[0] | d.refP[1]) ? d.snap : nullptr);
        d.nThr = int16_t(nr);
        const int lines = (g_thrWt + 63) / 64;
        for (int p = 0; p < 2; ++p) {
            if (d.refP[p]) continue;
            const int mir = m.mirror[p];
            uint32_t* o = d.tix[p];
            int na = 0, nm = 0;
            for (int i = 0; i < nr; ++i) {
                const int x = thr::index_of(raw[i], p, mir);
                if (x < 0) continue;
                const char* row = reinterpret_cast<const char*>(g_thrW + std::size_t(x) * std::size_t(g_thrWt));
                for (int l = 0; l < lines && l < JQ_THR_PF_LINES; ++l) __builtin_prefetch(row + 64 * l);
                if (raw[i] >> 31) o[na++] = uint32_t(x); else o[thr::MAX_DIRTY - 1 - nm++] = uint32_t(x);
            }
            d.nTA[p] = int16_t(na); d.nTR[p] = int16_t(nm);
        }
#endif
        THRST(st_gens, 1);
        THRST(st_tuples, uint64_t(d.nThr < 0 ? 0 : d.nThr));
    }
}

#ifdef ABL_NOROWS
volatile uint32_t sinkx;
#endif
void update(const Acc& prev, Acc& next, const Delta& d) {
    if (!d.valid) {
        std::memcpy(next.v, prev.v, sizeof(int16_t) * std::size_t(g_W));
        std::memcpy(next.v[1], prev.v[1], sizeof(int16_t) * std::size_t(g_W));
        std::memcpy(next.ps, prev.ps, sizeof prev.ps);
        next.m = prev.m;
        ++next.m.noCapture;
        return;
    }
#if defined(JQ_THR_PREIDX)
    if (g_thrOn) {  // h32fatt experimental: indices precomputed in make_delta
        for (int p = 0; p < 2; ++p) {
            if (d.refP[p]) {
                uint16_t f[64];
                std::memcpy(f, d.full[p], sizeof(uint16_t) * std::size_t(d.nFull[p]));
                finny_refresh(p, d.m.kb[p], d.m.mirror[p], f, d.nFull[p], next.v[p], next.ps[p]);
                add_threats(d.snap, p, d.m.mirror[p], next.v[p], next.ps[p]);
                THRST(st_refr, 1);
                continue;
            }
            const int na = d.nTA[p], nm = d.nTR[p];
            THRST(st_rows, uint64_t(na + nm));
            nnk::apply_thr(g_ftW.get(), g_psqt.get(), g_W, next.v[p], next.ps[p], prev.v[p], prev.ps[p], d.add[p], d.nAdd,
                           d.rem[p], d.nRem, g_thrW, g_thrWt, d.tix[p], na, d.tix[p] + (thr::MAX_DIRTY - nm), nm);
        }
    } else
#else
    if (g_thrOn) {
        const uint32_t* raw = d.thr;
        int nr = d.nThr;
        uint32_t lraw[thr::MAX_DIRTY];
        Piece lafter[NUM_SQ];
        const Piece* after = d.snap;  // eager: snap already holds the board after the move
        if (nr < 0) {  // lazy: snap = board before the move; generate now
            nr = thr::gen_move(d.snap, d.mv, lraw, lafter);
            after = lafter;
            raw = lraw;
            THRST(st_gens, 1);
            THRST(st_tuples, uint64_t(nr));
        }
        for (int p = 0; p < 2; ++p) {
            if (d.refP[p]) {
                uint16_t f[64];
                std::memcpy(f, d.full[p], sizeof(uint16_t) * std::size_t(d.nFull[p]));
                finny_refresh(p, d.m.kb[p], d.m.mirror[p], f, d.nFull[p], next.v[p], next.ps[p]);
                add_threats(after, p, d.m.mirror[p], next.v[p], next.ps[p]);
                THRST(st_refr, 1);
                continue;
            }
            uint32_t ta[thr::MAX_DIRTY], tr[thr::MAX_DIRTY];
            int na = 0, nrm = 0;
            const int mir = d.m.mirror[p];
#ifdef ABL_NOIDX
            nr = 0;
#endif
            for (int i = 0; i < nr; ++i) {
                const int x = thr::index_of(raw[i], p, mir);
                if (x < 0) continue;
                if (raw[i] >> 31) ta[na++] = uint32_t(x); else tr[nrm++] = uint32_t(x);
            }
            THRST(st_rows, uint64_t(na + nrm));
#ifdef ABL_NOROWS
            sinkx += uint32_t(na + nrm) + (na ? ta[0] : 0); na = nrm = 0;
#endif
            nnk::apply_thr(g_ftW.get(), g_psqt.get(), g_W, next.v[p], next.ps[p], prev.v[p], prev.ps[p], d.add[p], d.nAdd,
                           d.rem[p], d.nRem, g_thrW, g_thrWt, ta, na, tr, nrm);
        }
    } else
#endif
    for (int p = 0; p < 2; ++p) {
        if (d.refP[p]) {
            uint16_t f[64];
            std::memcpy(f, d.full[p], sizeof(uint16_t) * std::size_t(d.nFull[p]));
            finny_refresh(p, d.m.kb[p], d.m.mirror[p], f, d.nFull[p], next.v[p], next.ps[p]);
        } else
            apply(next.v[p], next.ps[p], prev.v[p], prev.ps[p], d.add[p], d.nAdd, d.rem[p], d.nRem);
    }
    next.m = d.m;
    next.m.material = int16_t(prev.m.material + d.m.material);
}

int runtime_value(const Acc& acc, Color stm, int* psqtRaw, int* positionalRaw) {
    const Meta& m = acc.m;
    const int b = std::min((m.pieces - 1) / 4, 7) + (m.dark > 0 ? 8 : 0);
    alignas(64) uint8_t x[MAX_W];
#ifdef JQ_PAIRWISE_NNZ
    alignas(32) uint16_t nnz[MAX_W / 4 + 16];
    const int nnzN = nnk::pairwise_nnz(acc.v[stm], acc.v[~stm], g_W, x, nnz);
#else
    const uint16_t* nnz = nullptr;
    const int nnzN = 0;
    nnk::pairwise(acc.v[stm], acc.v[~stm], g_W, x);
#endif
    const int psqt = (acc.ps[stm][b] - acc.ps[~stm][b]) / 2;
    int positional;
#ifdef JQ_PAIRWISE_NNZ
    if (g_fast == 32) positional = nnk::propagate_fast<32>(g_fh[b], g_l1w.get() + std::size_t(b) * std::size_t(g_W) * 32, x, nnz, nnzN);
    else if (g_fast == 16) positional = nnk::propagate_fast<16>(g_fh[b], g_l1w.get() + std::size_t(b) * std::size_t(g_W) * 16, x, nnz, nnzN);
    else
#endif
    if (!g_head)
        positional = nnk::propagate(g_W, g_l1b[b], g_l1w.get() + std::size_t(b) * std::size_t(g_W) * L1, g_l2b[b],
                                    g_l2w[b], g_fb[b], g_fw[b], x, nnz, nnzN);
    else {
        const uint8_t* w1 = g_l1w.get() + std::size_t(b) * std::size_t(g_W) * std::size_t(g_hL1);
#define JQ_H3(A, B, Q) nnk::propagate_g<A, B, Q>(g_W, g3_l1b[b], w1, g3_l2b[b], g3_l2w[b], g3_fb[b], g3_fw[b], x, nnz, nnzN)
        switch (g_head & 7) {
            case 0: positional = JQ_H3(16, 32, false); break;
            case 1: positional = JQ_H3(16, 32, true); break;
            case 2: positional = JQ_H3(32, 32, false); break;
            case 3: positional = JQ_H3(32, 32, true); break;
            case 4: positional = JQ_H3(16, 64, false); break;
            case 5: positional = JQ_H3(16, 64, true); break;
            case 6: positional = JQ_H3(32, 64, false); break;
            default: positional = JQ_H3(32, 64, true); break;
        }
#undef JQ_H3
    }
    if (psqtRaw) *psqtRaw = psqt;
    if (positionalRaw) *positionalRaw = positional;
    const int nn = (psqt + positional) / 16;
    const int materialScale = 935 + m.material * 93 / 5116;
    int value = (nn * materialScale) / 1024;
    value = value * (218 - std::min<int>(m.noCapture, 120)) / 176;
    return std::clamp(value, -29507, 29507);
}

Value evaluate(const Acc& acc, Color stm) {
    const int64_t v = int64_t(runtime_value(acc, stm)) * g_scale;
    const int64_t s = v >= 0 ? (v + 500) / 1000 : -((-v + 500) / 1000);
    return Value(std::clamp<int64_t>(s, -(VALUE_KNOWN_WIN - 1), VALUE_KNOWN_WIN - 1));
}

}  // namespace jq::hnn
