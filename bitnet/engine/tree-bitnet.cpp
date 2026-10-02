// Tree-BitNet decode ops for bitnet.cpp: sparse exact MLP and tree output layer. See tree-bitnet.h.
#include "tree-bitnet.h"

#if defined(__clang__)
#pragma clang attribute push (__attribute__((target("avx2,fma,f16c"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC target("avx2,fma,f16c")
#endif

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <atomic>
#include <vector>
#include <ctime>

// ------------------------------------------------------------------ config
static std::atomic<long long> g_sel_n{0}, g_sel_calls{0};
static std::atomic<long long> g_att_sel{0}, g_att_tot{0};
static std::atomic<long long> g_tns[5], g_tall[5];                         // per attention op: thread-0 time, all-thread busy time (ns)
static inline long long nowns() { timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1000000000LL + ts.tv_nsec; }
struct OpTimer { int k, ith; long long t0; OpTimer(int k_, int ith_) : k(k_), ith(ith_), t0(nowns()) {} ~OpTimer() { const long long d = nowns() - t0; g_tall[k] += d; if (ith == 0) g_tns[k] += d; } };
static void print_stats() {
    if (g_tns[0] + g_tns[1] + g_tns[2] + g_tns[3]) fprintf(stderr, "tree-bitnet: attention op time ms (thread 0 / all threads): kenc %.1f/%.1f tab %.1f/%.1f scan %.1f/%.1f select+exact %.1f/%.1f merge %.1f/%.1f\n",
        g_tns[4] / 1e6, g_tall[4] / 1e6, g_tns[0] / 1e6, g_tall[0] / 1e6, g_tns[1] / 1e6, g_tall[1] / 1e6, g_tns[2] / 1e6, g_tall[2] / 1e6, g_tns[3] / 1e6, g_tall[3] / 1e6);
    if (g_att_tot) fprintf(stderr, "tree-bitnet: attention read %.1f%% of keys/values\n", 100.0 * (double)g_att_sel / (double)g_att_tot);
    if (g_sel_calls) fprintf(stderr, "tree-bitnet: mean selected neurons %.1f over %lld calls\n", (double)g_sel_n / g_sel_calls, (long long)g_sel_calls); }

struct TreeCfg {
    float mlp_frac = 0; int mlp_k = 0; bool all = false;
    int part_m = 0, part_C = 0;
    bool attn = false; float tau = 8.f; int recent = 64;
    int aL = 0, aH = 0, aS = 0, aD = 0, ahd = 0;
    std::vector<float> aw, ab, ac;       // [L][H][S][2^D-1][hd], [L][H][S][2^D-1], [L][H][S][2^D][hd]
    std::vector<ggml_fp16_t> ac16;       // leaf vectors in f16 for the per-query tables (half the memory traffic)          // TREE_MLP_PARTIAL=m:C  outlier partial-sum candidate selection
    int head_N = 8192; bool head = false;
    int V = 0, d = 0, S = 0, NB = 0;
    std::vector<float> leaves;          // [S][16][d]
    std::vector<uint8_t> codes;         // [NB][S/2][32]
};
static TreeCfg & cfg() {
    static TreeCfg c; static std::once_flag once;
    std::call_once(once, [] {
        if (const char * s = getenv("TREE_MLP_FRAC")) c.mlp_frac = (float)atof(s);
        if (const char * s = getenv("TREE_MLP_K"))    c.mlp_k = atoi(s);
        if (const char * s = getenv("TREE_MLP_PARTIAL")) { if (sscanf(s, "%d:%d", &c.part_m, &c.part_C) != 2) c.part_m = c.part_C = 0; }
        if (const char * s = getenv("TREE_ALL"))      c.all = atoi(s) != 0;
        if (const char * s = getenv("TREE_HEAD_N"))   c.head_N = atoi(s);
        if (const char * p = getenv("TREE_HEAD")) {
            FILE * f = fopen(p, "rb");
            char magic[4]; int32_t h[4];
            if (!f || fread(magic, 1, 4, f) != 4 || memcmp(magic, "TVOC", 4) || fread(h, 4, 4, f) != 4) {
                fprintf(stderr, "tree-bitnet: cannot read %s\n", p); exit(1);
            }
            c.V = h[1]; c.d = h[2]; c.S = h[3]; c.NB = (c.V + 31) / 32;
            c.leaves.resize((size_t)c.S * 16 * c.d); c.codes.resize((size_t)c.NB * (c.S / 2) * 32);
            if (fread(c.leaves.data(), 4, c.leaves.size(), f) != c.leaves.size() ||
                fread(c.codes.data(), 1, c.codes.size(), f) != c.codes.size()) { fprintf(stderr, "tree-bitnet: short read %s\n", p); exit(1); }
            fclose(f); c.head = true;
            fprintf(stderr, "tree-bitnet: tree head V=%d d=%d S=%d N=%d\n", c.V, c.d, c.S, c.head_N);
        }
        if (const char * s = getenv("TREE_TAU"))      c.tau = (float)atof(s);
        if (const char * s = getenv("TREE_RECENT"))   c.recent = atoi(s);
        if (const char * p = getenv("TREE_ATTN")) {
            FILE * f = fopen(p, "rb"); char magic[4]; int32_t h[6];
            if (!f || fread(magic, 1, 4, f) != 4 || memcmp(magic, "TKEY", 4) || fread(h, 4, 6, f) != 6) { fprintf(stderr, "tree-bitnet: cannot read %s\n", p); exit(1); }
            c.aL = h[1]; c.aH = h[2]; c.aS = h[3]; c.aD = h[4]; c.ahd = h[5];
            const int NI = (1 << c.aD) - 1, NL = 1 << c.aD;
            c.aw.resize((size_t)c.aL * c.aH * c.aS * NI * c.ahd); c.ab.resize((size_t)c.aL * c.aH * c.aS * NI); c.ac.resize((size_t)c.aL * c.aH * c.aS * NL * c.ahd);
            for (int l = 0; l < c.aL; l++) for (int hh = 0; hh < c.aH; hh++) {
                size_t o = (size_t)l * c.aH + hh;
                bool ok = fread(c.aw.data() + o * c.aS * NI * c.ahd, 4, (size_t)c.aS * NI * c.ahd, f) == (size_t)c.aS * NI * c.ahd;
                ok = ok && fread(c.ab.data() + o * c.aS * NI, 4, (size_t)c.aS * NI, f) == (size_t)c.aS * NI;
                ok = ok && fread(c.ac.data() + o * c.aS * NL * c.ahd, 4, (size_t)c.aS * NL * c.ahd, f) == (size_t)c.aS * NL * c.ahd;
                if (!ok) { fprintf(stderr, "tree-bitnet: short read %s\n", p); exit(1); }
            }
            fclose(f); c.attn = true;
            c.ac16.resize(c.ac.size()); for (size_t i = 0; i < c.ac.size(); i++) c.ac16[i] = ggml_fp32_to_fp16(c.ac[i]);
            fprintf(stderr, "tree-bitnet: tree attention L=%d H=%d S=%d D=%d tau=%.1f recent=%d\n", c.aL, c.aH, c.aS, c.aD, c.tau, c.recent);
        }
        if (getenv("TREE_STATS")) atexit(print_stats);
        if (c.mlp_frac > 0 || c.mlp_k > 0) fprintf(stderr, "tree-bitnet: sparse MLP frac=%.3f k=%d\n", c.mlp_frac, c.mlp_k);
    });
    return c;
}

static inline void part(int n, int ith, int nth, int & a, int & b) { a = (int)((int64_t)n * ith / nth); b = (int)((int64_t)n * (ith + 1) / nth); }

// ------------------------------------------------------------------ I2_S helpers (bitnet.cpp packing)
// per 128 weights: 32 bytes, byte j holds weights j, 32+j, 64+j, 96+j at bits 6,4,2,0; codes 0,1,2 = -1,0,+1
static inline int hsum_i32(__m256i a) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, s));
    return _mm_cvtsi128_si32(_mm_add_epi32(s, _mm_shuffle_epi32(s, 1)));
}
static inline int dot_codes(const uint8_t * w, const int8_t * x, int n) {       // sum code_j * x_j
    const __m256i m3 = _mm256_set1_epi8(3), one = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < n; i += 128, w += 32, x += 128) {
        __m256i b = _mm256_loadu_si256((const __m256i *)w);
        __m256i s = _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(b, 6), m3), _mm256_loadu_si256((const __m256i *)x));
        s = _mm256_add_epi16(s, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(b, 4), m3), _mm256_loadu_si256((const __m256i *)(x + 32))));
        s = _mm256_add_epi16(s, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(b, 2), m3), _mm256_loadu_si256((const __m256i *)(x + 64))));
        s = _mm256_add_epi16(s, _mm256_maddubs_epi16(_mm256_and_si256(b, m3), _mm256_loadu_si256((const __m256i *)(x + 96))));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(s, one));
    }
    return hsum_i32(acc);
}
static inline int code_at(const uint8_t * row, int i) {
    return (row[(i / 128) * 32 + (i % 32)] >> (6 - 2 * ((i % 128) / 32))) & 3;
}
// same int8 quantization as bitnet.cpp's quantize_row_i8_s
static float quant_i8(const float * x, int8_t * q, int n, int * sum) {
    float amax = 0; for (int i = 0; i < n; i++) amax = std::max(amax, fabsf(x[i]));
    float s = amax > 0 ? 127.f / amax : 0.f; int sm = 0;
    for (int i = 0; i < n; i++) { int v = (int)roundf(x[i] * s); v = v > 127 ? 127 : v < -128 ? -128 : v; q[i] = (int8_t)v; sm += v; }
    *sum = sm; return s;
}

// ------------------------------------------------------------------ sparse exact MLP
struct MlpLayer {
    ggml_tensor * gate = nullptr, * up = nullptr, * down = nullptr;
    std::vector<uint8_t> gateT;        // [d rows][F] in I2_S packing (transposed gate), built on first use
    float gate_scale = 0;
    std::once_flag gbuilt;
    std::vector<uint8_t> downT;        // [F rows][d] in I2_S packing (transposed down), built on first use
    float down_scale = 0;
    std::once_flag built;
};
static MlpLayer g_mlp[256];

static void build_downT(MlpLayer & L) {
    const int d = (int)L.down->ne[1], F = (int)L.down->ne[0];          // down: [F (cols), d (rows)]
    const uint8_t * W = (const uint8_t *)L.down->data;
    L.down_scale = *(const float *)(W + (size_t)d * F / 4);
    L.downT.assign((size_t)F * d / 4, 0);
    for (int j = 0; j < d; j++) {
        const uint8_t * row = W + (size_t)j * F / 4;
        for (int i = 0; i < F; i++) {
            int c = code_at(row, i);
            uint8_t * trow = L.downT.data() + (size_t)i * d / 4;
            trow[(j / 128) * 32 + (j % 32)] |= (uint8_t)(c << (6 - 2 * ((j % 128) / 32)));
        }
    }
}

// select neurons for one token from the gate pre-activations g[F] -> idx list (sorted)
static int select_neurons(const float * g, int F, int * idx) {
    const TreeCfg & c = cfg();
    float emax = 0; double tot = 0;
    for (int i = 0; i < F; i++) if (g[i] > 0) { float e = g[i] * g[i]; emax = std::max(emax, e); tot += e; }
    if (emax <= 0) return 0;
    if (c.mlp_frac >= 1.f) { int n = 0; for (int i = 0; i < F; i++) if (g[i] > 0) idx[n++] = i; return n; }   // exact: all active
    const int NBIN = 2048; int hist[NBIN] = {0}; double hsum[NBIN] = {0};
    const float bs = (NBIN - 1) / emax;
    for (int i = 0; i < F; i++) if (g[i] > 0) { float e = g[i] * g[i]; int b = (int)(e * bs); hist[b]++; hsum[b] += e; }
    int thr = NBIN - 1, cnt = 0; double acc = 0;
    if (c.mlp_frac > 0) { while (thr > 0 && acc < c.mlp_frac * tot) { acc += hsum[thr]; cnt += hist[thr]; thr--; } }
    else                { while (thr > 0 && cnt + hist[thr] <= c.mlp_k) { cnt += hist[thr]; thr--; } }
    int n = 0;
    for (int i = 0; i < F; i++) if (g[i] > 0 && (int)(g[i] * g[i] * bs) > thr) idx[n++] = i;
    return n;
}

static void build_gateT(MlpLayer & L) {
    const int d = (int)L.gate->ne[0], F = (int)L.gate->ne[1];          // gate: [d (cols), F (rows)]
    const uint8_t * W = (const uint8_t *)L.gate->data;
    L.gate_scale = *(const float *)(W + (size_t)d * F / 4);
    L.gateT.assign((size_t)F * d / 4, 0);
    for (int i = 0; i < F; i++) {
        const uint8_t * row = W + (size_t)i * d / 4;
        for (int j = 0; j < d; j++) {
            int c = code_at(row, j);
            uint8_t * trow = L.gateT.data() + (size_t)j * F / 4;
            trow[(i / 128) * 32 + (i % 32)] |= (uint8_t)(c << (6 - 2 * ((i % 128) / 32)));
        }
    }
}

// top-m |x| input dims of the int8-quantized token (same quantization as the stock matmul)
static int outlier_dims(const float * xt, int d, int m, int8_t * xq, int * dims, int * sx, float * as) {
    *as = quant_i8(xt, xq, d, sx);
    int hist[128] = {0}; for (int j = 0; j < d; j++) hist[std::abs((int)xq[j]) > 127 ? 127 : std::abs((int)xq[j])]++;
    int thr = 127, cnt = 0; while (thr > 0 && cnt + hist[thr] <= m) cnt += hist[thr--];
    int n = 0; for (int j = 0; j < d && n < m; j++) if (std::abs((int)xq[j]) > thr) dims[n++] = j;
    return n;
}

// approximate gate ghat[F, T] (int sums, stored as float) = sum over the outlier dims of x_j * gate[:, j].  args: x
static void op_gapprox(ggml_tensor * dst, int ith, int nth, void * ud) {
    MlpLayer & L = *(MlpLayer *)ud;
    std::call_once(L.gbuilt, [&] { build_gateT(L); });
    const ggml_tensor * x = dst->src[0];
    const int d = (int)x->ne[0], F = (int)dst->ne[0], T = (int)x->ne[1], m = cfg().part_m;
    std::vector<int8_t> xq(d); std::vector<int> dims(d); std::vector<int16_t> acc(F); std::vector<int32_t> acc32(F);
    int b0, b1; part(F / 128, ith, nth, b0, b1);
    const __m256i m3 = _mm256_set1_epi8(3), one8 = _mm256_set1_epi8(1);
    for (int t = 0; t < T; t++) {
        const float * xt = (const float *)((const char *)x->data + t * x->nb[1]);
        float * o = (float *)((char *)dst->data + t * dst->nb[1]);
        int sx; float as; int n = outlier_dims(xt, d, m, xq.data(), dims.data(), &sx, &as);
        std::fill(acc32.begin() + b0 * 128, acc32.begin() + b1 * 128, 0);
        for (int k0 = 0; k0 < n; k0 += 256) {
            int k1 = std::min(n, k0 + 256);
            std::fill(acc.begin() + b0 * 128, acc.begin() + b1 * 128, 0);
            for (int s = k0; s < k1; s++) {
                const uint8_t * w = L.gateT.data() + (size_t)dims[s] * F / 4;
                __m256i av = _mm256_set1_epi8(xq[dims[s]]);
                for (int blk = b0; blk < b1; blk++) {
                    __m256i bb = _mm256_loadu_si256((const __m256i *)(w + blk * 32));
                    __m256i c4[4] = {_mm256_and_si256(_mm256_srli_epi16(bb, 6), m3), _mm256_and_si256(_mm256_srli_epi16(bb, 4), m3),
                                     _mm256_and_si256(_mm256_srli_epi16(bb, 2), m3), _mm256_and_si256(bb, m3)};
                    for (int qd = 0; qd < 4; qd++) {
                        __m256i p = _mm256_sign_epi8(av, _mm256_sub_epi8(c4[qd], one8));
                        __m256i * oo = (__m256i *)(acc.data() + blk * 128 + 32 * qd);
                        _mm256_storeu_si256(oo, _mm256_add_epi16(_mm256_loadu_si256(oo), _mm256_cvtepi8_epi16(_mm256_castsi256_si128(p))));
                        _mm256_storeu_si256(oo + 1, _mm256_add_epi16(_mm256_loadu_si256(oo + 1), _mm256_cvtepi8_epi16(_mm256_extracti128_si256(p, 1))));
                    }
                }
            }
            for (int i = b0 * 128; i < b1 * 128; i++) acc32[i] += acc[i];
        }
        for (int i = b0 * 128; i < b1 * 128; i++) o[i] = (float)acc32[i];
    }
}

// g[F, T]: exact gate for the top-C neurons by ghat, 0 elsewhere (relu -> not selected).  args: ghat, x
static void op_gcand(ggml_tensor * dst, int ith, int nth, void * ud) {
    MlpLayer & L = *(MlpLayer *)ud;
    const ggml_tensor * gh = dst->src[0], * x = dst->src[1];
    const int F = (int)gh->ne[0], d = (int)x->ne[0], T = (int)x->ne[1], Cn = cfg().part_C;
    const uint8_t * Wg = (const uint8_t *)L.gate->data;
    std::vector<int8_t> xq(d);
    for (int t = 0; t < T; t++) {
        const float * gt = (const float *)((const char *)gh->data + t * gh->nb[1]);
        const float * xt = (const float *)((const char *)x->data + t * x->nb[1]);
        float * o = (float *)((char *)dst->data + t * dst->nb[1]);
        float mx = 1e-9f; for (int i = 0; i < F; i++) mx = std::max(mx, gt[i]);
        const int NBIN = 2048; int hist[NBIN] = {0}; const float bs = (NBIN - 1) / mx;
        for (int i = 0; i < F; i++) if (gt[i] > 0) hist[(int)(gt[i] * bs)]++;
        int thr = NBIN - 1, cnt = 0; while (thr > 0 && cnt + hist[thr] <= Cn) cnt += hist[thr--];
        int sx; float as = quant_i8(xt, xq.data(), d, &sx);
        int a, b; part(F, ith, nth, a, b);
        for (int i = a; i < b; i++) {
            if (gt[i] > 0 && (int)(gt[i] * bs) > thr) {
                if (i + 8 < b) _mm_prefetch((const char *)(Wg + (size_t)(i + 8) * d / 4), _MM_HINT_T0);
                o[i] = (float)(dot_codes(Wg + (size_t)i * d / 4, xq.data(), d) - sx) / as * L.gate_scale;
            } else o[i] = 0.f;
        }
    }
}

// h[F, T] = relu(g)^2 * up(x) on the selected neurons, 0 elsewhere.  args: g [F,T], x [d,T]
static void op_hsel(ggml_tensor * dst, int ith, int nth, void * ud) {
    MlpLayer & L = *(MlpLayer *)ud;
    const ggml_tensor * g = dst->src[0], * x = dst->src[1];
    const int F = (int)g->ne[0], d = (int)x->ne[0], T = (int)g->ne[1];
    const uint8_t * Wu = (const uint8_t *)L.up->data; const float wscale = *(const float *)(Wu + (size_t)F * d / 4);
    std::vector<int> idx(F); std::vector<int8_t> xq(d);
    for (int t = 0; t < T; t++) {
        const float * gt = (const float *)((const char *)g->data + t * g->nb[1]);
        const float * xt = (const float *)((const char *)x->data + t * x->nb[1]);
        float * ht = (float *)((char *)dst->data + t * dst->nb[1]);
        int a, b; part(F, ith, nth, a, b);                                     // this thread owns neurons [a, b)
        memset(ht + a, 0, (b - a) * sizeof(float));
        int n = select_neurons(gt, F, idx.data());
        if (ith == 0) { g_sel_n += n; g_sel_calls++; }
        int sx; float as = quant_i8(xt, xq.data(), d, &sx);
        int s0 = (int)(std::lower_bound(idx.begin(), idx.begin() + n, a) - idx.begin());
        int s1 = (int)(std::lower_bound(idx.begin(), idx.begin() + n, b) - idx.begin());
        for (int s = s0; s < s1; s++) {
            int i = idx[s];
            if (s + 4 < s1) { const char * pn = (const char *)(Wu + (size_t)idx[s + 4] * d / 4); for (int q = 0; q < d / 4; q += 64) _mm_prefetch(pn + q, _MM_HINT_T0); }
            float u = (float)(dot_codes(Wu + (size_t)i * d / 4, xq.data(), d) - sx) / as * wscale;
            ht[i] = gt[i] * gt[i] * u;
        }
    }
}

// partial sums: thread ith accumulates full transposed rows for its share of the nonzero neurons into slot ith of
// out [d, NSLOT*T] (int32 accumulation converted to float with the activation/weight scales); unused slots are zeroed.
#define NSLOT 16
static void op_down_part(ggml_tensor * dst, int ith, int nth, void * ud) {
    MlpLayer & L = *(MlpLayer *)ud;
    std::call_once(L.built, [&] { build_downT(L); });                         // weights are mapped by now
    const ggml_tensor * h = dst->src[0];
    const int F = (int)h->ne[0], d = (int)L.down->ne[1], T = (int)h->ne[1];
    std::vector<int> nz(F); std::vector<int8_t> a(F), q(F); std::vector<int32_t> acc32(d); std::vector<int16_t> acc(d);
    const __m256i m3 = _mm256_set1_epi8(3), one8 = _mm256_set1_epi8(1);
    for (int t = 0; t < T; t++) {
        const float * ht = (const float *)((const char *)h->data + t * h->nb[1]);
        int sm; float as = quant_i8(ht, q.data(), F, &sm);
        int n = 0; for (int i = 0; i < F; i++) if (q[i]) { nz[n] = i; a[n] = q[i]; n++; }
        for (int slot = ith; slot < NSLOT; slot += nth) {
            float * o = (float *)dst->data + ((size_t)t * NSLOT + slot) * d;
            if (slot >= nth) { memset(o, 0, d * sizeof(float)); continue; }
            int s0, s1; part(n, slot, nth, s0, s1);
            std::fill(acc32.begin(), acc32.end(), 0);
            for (int k0 = s0; k0 < s1; k0 += 256) {
                int k1 = std::min(s1, k0 + 256);
                std::fill(acc.begin(), acc.end(), 0);
                for (int s = k0; s < k1; s++) {
                    const uint8_t * w = L.downT.data() + (size_t)nz[s] * d / 4;
                    if (s + 4 < k1) { const char * pn = (const char *)(L.downT.data() + (size_t)nz[s + 4] * d / 4); for (int c = 0; c < d / 4; c += 64) _mm_prefetch(pn + c, _MM_HINT_T0); }
                    __m256i av = _mm256_set1_epi8(a[s]);
                    for (int blk = 0; blk < d / 128; blk++) {
                        __m256i bb = _mm256_loadu_si256((const __m256i *)(w + blk * 32));
                        __m256i c4[4] = {_mm256_and_si256(_mm256_srli_epi16(bb, 6), m3), _mm256_and_si256(_mm256_srli_epi16(bb, 4), m3),
                                         _mm256_and_si256(_mm256_srli_epi16(bb, 2), m3), _mm256_and_si256(bb, m3)};
                        for (int qd = 0; qd < 4; qd++) {
                            __m256i p = _mm256_sign_epi8(av, _mm256_sub_epi8(c4[qd], one8));
                            __m256i * oo = (__m256i *)(acc.data() + blk * 128 + 32 * qd);
                            _mm256_storeu_si256(oo, _mm256_add_epi16(_mm256_loadu_si256(oo), _mm256_cvtepi8_epi16(_mm256_castsi256_si128(p))));
                            _mm256_storeu_si256(oo + 1, _mm256_add_epi16(_mm256_loadu_si256(oo + 1), _mm256_cvtepi8_epi16(_mm256_extracti128_si256(p, 1))));
                        }
                    }
                }
                for (int jj = 0; jj < d; jj++) acc32[jj] += acc[jj];
            }
            const float sc = as > 0 ? L.down_scale / as : 0.f;
            for (int jj = 0; jj < d; jj++) o[jj] = (float)acc32[jj] * sc;
        }
    }
}

// y[d, T] = sum over the NSLOT partial slots
static void op_down_sum(ggml_tensor * dst, int ith, int nth, void *) {
    const ggml_tensor * p = dst->src[0];
    const int d = (int)dst->ne[0], T = (int)dst->ne[1];
    int a, b; part(d, ith, nth, a, b);
    for (int t = 0; t < T; t++) {
        float * y = (float *)((char *)dst->data + t * dst->nb[1]);
        const float * pp = (const float *)p->data + (size_t)t * NSLOT * d;
        for (int j = a; j < b; j++) { float s = 0; for (int k = 0; k < NSLOT; k++) s += pp[(size_t)k * d + j]; y[j] = s; }
    }
}

ggml_tensor * tree_bitnet_ffn(ggml_context * ctx, ggml_tensor * x, ggml_tensor * gate, ggml_tensor * up,
                              ggml_tensor * down, ggml_tensor * sub_norm, float eps, int il, int n_tokens) {
    const TreeCfg & c = cfg();
    if (!(c.mlp_frac > 0 || c.mlp_k > 0) || (n_tokens > 1 && !c.all)) return nullptr;
    if (gate->type != GGML_TYPE_I2_S || up->type != GGML_TYPE_I2_S || down->type != GGML_TYPE_I2_S) return nullptr;
    MlpLayer & L = g_mlp[il];
    L.up = up; L.down = down;
    L.gate = gate;
    ggml_tensor * g;
    if (c.part_m > 0 && c.part_C > 0) {                                              // outlier partial-sum candidates
        ggml_tensor * a0[1] = {x};
        ggml_tensor * gh = ggml_custom_4d(ctx, GGML_TYPE_F32, gate->ne[1], x->ne[1], 1, 1, a0, 1, op_gapprox, GGML_N_TASKS_MAX, &L);
        ggml_tensor * a1[2] = {gh, x};
        g = ggml_custom_4d(ctx, GGML_TYPE_F32, gate->ne[1], x->ne[1], 1, 1, a1, 2, op_gcand, GGML_N_TASKS_MAX, &L);
    } else {
        g = ggml_mul_mat(ctx, gate, x);                                              // exact gate, stock kernel
    }
    ggml_tensor * args[2] = {g, x};
    ggml_tensor * h = ggml_custom_4d(ctx, GGML_TYPE_F32, g->ne[0], g->ne[1], 1, 1, args, 2, op_hsel, GGML_N_TASKS_MAX, &L);
    h = ggml_rms_norm(ctx, h, eps);
    h = ggml_mul(ctx, h, sub_norm);
    ggml_tensor * a1[1] = {h};
    ggml_tensor * part_sums = ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], NSLOT * x->ne[1], 1, 1, a1, 1, op_down_part, GGML_N_TASKS_MAX, &L);
    ggml_tensor * a2[1] = {part_sums};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, x->ne[0], x->ne[1], 1, 1, a2, 1, op_down_sum, GGML_N_TASKS_MAX, nullptr);
}

// ------------------------------------------------------------------ tree output layer
// tables [S*16, T] = h . leaves
static void op_tables(ggml_tensor * dst, int ith, int nth, void *) {
    const TreeCfg & c = cfg(); const ggml_tensor * h = dst->src[0]; const int T = (int)h->ne[1];
    for (int t = 0; t < T; t++) {
        const float * ht = (const float *)((const char *)h->data + t * h->nb[1]);
        float * o = (float *)((char *)dst->data + t * dst->nb[1]);
        int a, b; part(c.S * 16, ith, nth, a, b);
        for (int e = a; e < b; e++) {
            const float * lv = c.leaves.data() + (size_t)e * c.d;
            __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
            for (int i = 0; i < c.d; i += 16) {
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(lv + i), _mm256_loadu_ps(ht + i), s0);
                s1 = _mm256_fmadd_ps(_mm256_loadu_ps(lv + i + 8), _mm256_loadu_ps(ht + i + 8), s1);
            }
            s0 = _mm256_add_ps(s0, s1);
            __m128 q = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1)); q = _mm_hadd_ps(q, q); q = _mm_hadd_ps(q, q);
            o[e] = _mm_cvtss_f32(q);
        }
    }
}

// approximate logits [NB*32, T] by fast-scan over the vocab codes
static void op_scan(ggml_tensor * dst, int ith, int nth, void *) {
    const TreeCfg & c = cfg(); const ggml_tensor * tb = dst->src[0]; const int T = (int)tb->ne[1], S = c.S, P = S / 2;
    std::vector<uint8_t> tlo(S * 16), thi(S * 16);
    // token order inside a block after the even/odd split: 0,2,..,14,16,..,30, 1,3,..,15,17,..,31
    static int remap[32];
    for (int k = 0; k < 32; k++) remap[k] = (k < 16) ? ((k < 8) ? 2 * k : 2 * (k - 8) + 16) : ((k < 24) ? 2 * (k - 16) + 1 : 2 * (k - 24) + 17);
    for (int t = 0; t < T; t++) {
        const float * tab = (const float *)((const char *)tb->data + t * tb->nb[1]);
        float * o = (float *)((char *)dst->data + t * dst->nb[1]);
        double off = 0; float rmax = 1e-12f; std::vector<float> mn(S);
        for (int s = 0; s < S; s++) {
            float a = tab[s * 16], b = tab[s * 16];
            for (int l = 1; l < 16; l++) { a = std::min(a, tab[s * 16 + l]); b = std::max(b, tab[s * 16 + l]); }
            mn[s] = a; off += a; rmax = std::max(rmax, b - a);
        }
        const float step = rmax / 65535.f;                                      // shared step: score = off + step * sum q
        for (int s = 0; s < S; s++) for (int l = 0; l < 16; l++) {
            int q = (int)lrintf((tab[s * 16 + l] - mn[s]) / step); q = q > 65535 ? 65535 : q;
            tlo[s * 16 + l] = (uint8_t)(q & 255); thi[s * 16 + l] = (uint8_t)(q >> 8);
        }
        int b0, b1; part(c.NB, ith, nth, b0, b1);
        const __m256i m4 = _mm256_set1_epi8(15), m8 = _mm256_set1_epi16(255);
        for (int b = b0; b < b1; b++) {
            __m256i l0 = _mm256_setzero_si256(), l1 = l0, h0 = l0, h1 = l0;
            const uint8_t * cb = c.codes.data() + (size_t)b * P * 32;
            for (int p = 0; p < P; p++) {
                __m256i cc = _mm256_loadu_si256((const __m256i *)(cb + p * 32));
                __m256i i0 = _mm256_and_si256(cc, m4), i1 = _mm256_and_si256(_mm256_srli_epi16(cc, 4), m4);
                __m256i L0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(tlo.data() + 32 * p)));
                __m256i L1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(tlo.data() + 32 * p + 16)));
                __m256i H0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(thi.data() + 32 * p)));
                __m256i H1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(thi.data() + 32 * p + 16)));
                __m256i lo = _mm256_shuffle_epi8(L0, i0), lo2 = _mm256_shuffle_epi8(L1, i1);
                __m256i hi = _mm256_shuffle_epi8(H0, i0), hi2 = _mm256_shuffle_epi8(H1, i1);
                l0 = _mm256_add_epi16(l0, _mm256_add_epi16(_mm256_and_si256(lo, m8), _mm256_and_si256(lo2, m8)));
                l1 = _mm256_add_epi16(l1, _mm256_add_epi16(_mm256_srli_epi16(lo, 8), _mm256_srli_epi16(lo2, 8)));
                h0 = _mm256_add_epi16(h0, _mm256_add_epi16(_mm256_and_si256(hi, m8), _mm256_and_si256(hi2, m8)));
                h1 = _mm256_add_epi16(h1, _mm256_add_epi16(_mm256_srli_epi16(hi, 8), _mm256_srli_epi16(hi2, 8)));
            }
            alignas(32) uint16_t L[32], H[32];
            _mm256_store_si256((__m256i *)L, l0); _mm256_store_si256((__m256i *)(L + 16), l1);
            _mm256_store_si256((__m256i *)H, h0); _mm256_store_si256((__m256i *)(H + 16), h1);
            for (int k = 0; k < 32; k++) o[b * 32 + remap[k]] = (float)(off + (double)step * (256.0 * H[k] + L[k]));
        }
    }
}

// logits [V, T]: exact for the top-N tokens by approximate score, approximate elsewhere. args: scores, h, tok_embd
static void op_head(ggml_tensor * dst, int ith, int nth, void *) {
    const TreeCfg & c = cfg();
    const ggml_tensor * sc = dst->src[0], * h = dst->src[1], * E = dst->src[2];
    const int T = (int)h->ne[1], V = c.V, d = c.d;
    std::vector<uint8_t> cand(V);
    for (int t = 0; t < T; t++) {
        const float * st = (const float *)((const char *)sc->data + t * sc->nb[1]);
        const float * ht = (const float *)((const char *)h->data + t * h->nb[1]);
        float * o = (float *)((char *)dst->data + t * dst->nb[1]);
        float lo = st[0], hi = st[0]; for (int v = 1; v < V; v++) { lo = std::min(lo, st[v]); hi = std::max(hi, st[v]); }
        const int NBIN = 4096; std::vector<int> hist(NBIN, 0); const float bs = (NBIN - 1) / std::max(hi - lo, 1e-9f);
        for (int v = 0; v < V; v++) hist[(int)((st[v] - lo) * bs)]++;
        int thr = NBIN - 1, cnt = 0; while (thr > 0 && cnt + hist[thr] <= c.head_N) cnt += hist[thr--];
        int a, b; part(V, ith, nth, a, b);
        for (int v = a; v < b; v++) {
            if ((int)((st[v] - lo) * bs) > thr) {
                const ggml_fp16_t * w = (const ggml_fp16_t *)((const char *)E->data + (size_t)v * E->nb[1]);
                __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
                for (int i = 0; i < d; i += 16) {
                    s0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i))), _mm256_loadu_ps(ht + i), s0);
                    s1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 8))), _mm256_loadu_ps(ht + i + 8), s1);
                }
                s0 = _mm256_add_ps(s0, s1);
                __m128 q = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1)); q = _mm_hadd_ps(q, q); q = _mm_hadd_ps(q, q);
                o[v] = _mm_cvtss_f32(q);
            } else {
                o[v] = st[v];
            }
        }
    }
}

ggml_tensor * tree_bitnet_head(ggml_context * ctx, ggml_tensor * h, ggml_tensor * tok_embd, int n_tokens) {
    const TreeCfg & c = cfg();
    if (!c.head || (n_tokens > 1 && !c.all)) return nullptr;
    if (tok_embd->type != GGML_TYPE_F16 || tok_embd->ne[0] != c.d || tok_embd->ne[1] != c.V) {
        fprintf(stderr, "tree-bitnet: tree head needs an f16 token embedding of shape [%d, %d]\n", c.d, c.V); return nullptr;
    }
    ggml_tensor * a0[1] = {h};
    ggml_tensor * tab = ggml_custom_4d(ctx, GGML_TYPE_F32, c.S * 16, h->ne[1], 1, 1, a0, 1, op_tables, GGML_N_TASKS_MAX, nullptr);
    ggml_tensor * a1[1] = {tab};
    ggml_tensor * sc = ggml_custom_4d(ctx, GGML_TYPE_F32, (int64_t)c.NB * 32, h->ne[1], 1, 1, a1, 1, op_scan, GGML_N_TASKS_MAX, nullptr);
    ggml_tensor * a2[3] = {sc, h, tok_embd};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, c.V, h->ne[1], 1, 1, a2, 3, op_head, GGML_N_TASKS_MAX, nullptr);
}


// ------------------------------------------------------------------ select+rescore attention
struct AttnLayer {
    std::vector<uint8_t> codes[16];                  // per kv head: [block][S/2][32] nibble codes
    std::vector<int8_t> k8[16], v8[16];              // TREE_KV8: per kv head int8 copies [cell][hd] (per-row absmax)
    std::vector<float> ks[16], vs[16];               // ... and their per-row scales
    int64_t cap = 0;
};
static bool kv8_on() { static const bool on = getenv("TREE_KV8") && atoi(getenv("TREE_KV8")) != 0; return on; }
static inline void quant_row8(const float * x, int n, int8_t * q, float * scale) {
    float amax = 0; for (int i = 0; i < n; i++) amax = std::max(amax, fabsf(x[i]));
    const float s = amax > 0 ? 127.f / amax : 0.f;
    for (int i = 0; i < n; i++) q[i] = (int8_t)lrintf(x[i] * s);
    *scale = amax > 0 ? amax / 127.f : 0.f;
}
static AttnLayer g_attn[256];

bool tree_bitnet_attn_enabled(int n_tokens) { const TreeCfg & c = cfg(); return c.attn && (n_tokens == 1 || c.all); }

static inline float dotf(const float * a, const float * b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 16) { s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0); s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1); }
    s0 = _mm256_add_ps(s0, s1);
    __m128 q = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1)); q = _mm_hadd_ps(q, q); q = _mm_hadd_ps(q, q);
    return _mm_cvtss_f32(q);
}
static inline float doth(const float * a, const ggml_fp16_t * b, int n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b + i))), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b + i + 8))), s1);
    }
    s0 = _mm256_add_ps(s0, s1);
    __m128 q = _mm_add_ps(_mm256_castps256_ps128(s0), _mm256_extractf128_ps(s0, 1)); q = _mm_hadd_ps(q, q); q = _mm_hadd_ps(q, q);
    return _mm_cvtss_f32(q);
}

// args: k_cur [hd, H, T] f32, k_idxs [T] i64
static void op_kenc(ggml_tensor * dst, int ith, int nth, void * ud) {
    OpTimer timer_(4, ith);
    const TreeCfg & c = cfg(); const int il = (int)(intptr_t)ud; AttnLayer & A = g_attn[il];
    const ggml_tensor * k = dst->src[0], * idx = dst->src[1];
    const int hd = (int)k->ne[0], H = (int)k->ne[1], T = (int)k->ne[2], S = c.aS, P = S / 2, D = c.aD, NI = (1 << D) - 1, NL = 1 << D;
    std::vector<float> r(hd);
    int a, b; part(T * H, ith, nth, a, b);
    for (int it = a; it < b; it++) {
        const int t = it / H, h = it % H;
        const float * kv = (const float *)((const char *)k->data + t * k->nb[2] + h * k->nb[1]);
        const int64_t cell = ((const int64_t *)idx->data)[t];
        if (cell >= A.cap) continue;
        if (kv8_on()) {
            const ggml_tensor * v = dst->src[2];
            quant_row8(kv, hd, A.k8[h].data() + (size_t)cell * hd, &A.ks[h][cell]);
            quant_row8((const float *)((const char *)v->data + t * v->nb[2] + h * v->nb[1]), hd, A.v8[h].data() + (size_t)cell * hd, &A.vs[h][cell]);
        }
        memcpy(r.data(), kv, hd * sizeof(float));
        const float tiny = 1e-5f * sqrtf(dotf(kv, kv, hd));
        const size_t o = (size_t)il * c.aH + h;
        uint8_t * cb = A.codes[h].data() + (size_t)(cell / 32) * P * 32 + (cell % 32);
        for (int s = 0; s < S; s++) {
            const float * w = c.aw.data() + ((o * S + s) * NI) * hd; const float * bb = c.ab.data() + (o * S + s) * NI;
            int node = 0;
            for (int dd = 0; dd < D; dd++) node = 2 * node + 1 + (dotf(w + (size_t)node * hd, r.data(), hd) - bb[node] > 0 ? 1 : 0);
            const int leaf = node - NI;
            const float * cv = c.ac.data() + ((o * S + s) * NL + leaf) * hd;
            for (int i = 0; i < hd; i++) r[i] -= cv[i];
            if (sqrtf(dotf(r.data(), r.data(), hd)) < tiny) std::fill(r.begin(), r.end(), 0.f);
            uint8_t & byte = cb[(s / 2) * 32];
            byte = (s & 1) ? (uint8_t)((byte & 0x0F) | (leaf << 4)) : (uint8_t)((byte & 0xF0) | leaf);
        }
    }
}

ggml_tensor * tree_bitnet_kenc(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * v_cur, ggml_tensor * k_idxs, int64_t kv_size, int il) {
    const TreeCfg & c = cfg(); AttnLayer & A = g_attn[il];
    if (A.cap < kv_size) {                                               // graph build is single-threaded
        const int P = c.aS / 2; const int64_t nb = (kv_size + 31) / 32;
        for (int h = 0; h < c.aH; h++) A.codes[h].assign((size_t)nb * P * 32, 0);
        if (kv8_on()) for (int h = 0; h < c.aH; h++) {
            A.k8[h].assign((size_t)nb * 32 * c.ahd, 0); A.v8[h].assign((size_t)nb * 32 * c.ahd, 0);
            A.ks[h].assign((size_t)nb * 32, 0.f); A.vs[h].assign((size_t)nb * 32, 0.f);
        }
        A.cap = nb * 32;
    }
    ggml_tensor * args[3] = {k_cur, k_idxs, v_cur};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1, args, 3, op_kenc, GGML_N_TASKS_MAX, (void *)(intptr_t)il);
}

// Decode attention runs as four ops so that a single token still spreads over all threads (5 kv heads alone
// would not) and so that each selected key/value row is read once per kv head, not once per query head:
//   A tab   (t, q head)          leaf tables: scaled query . leaf vectors, 16-bit two-plane quantization
//   B scan  (t, q head, chunk)   fast-scan approximate scores, stock mask added (-inf), per-chunk max
//   C sel   (t, kv head, chunk)  union over the G query heads of {within tau of that head's best, first cell,
//                                newest `recent`}; exact f16 scores for all G heads, partial softmax over the chunk
//   D merge (t, q head)          combine the chunk partials
static const int TCH = getenv("TREE_CHUNK") ? atoi(getenv("TREE_CHUNK")) : 512;   // keys per chunk (multiple of 32)
static const int PFD = getenv("TREE_PF") ? atoi(getenv("TREE_PF")) : 8;      // prefetch distance (selected rows)
struct AttnParams { float scale; int il; int Hq; };
static AttnParams g_attn_par[256];
static inline int tab_floats(int S, int NL) { return 2 + (2 * S * NL + 3) / 4; }

// args: q [hd, Hq, T] f32, deps...   dst: [tab_floats, Hq*T]  = {off, step, lo plane bytes, hi plane bytes}
static void op_ttab(ggml_tensor * dst, int ith, int nth, void * ud) {
    OpTimer timer_(0, ith);
    const TreeCfg & c = cfg(); const AttnParams & ap = *(const AttnParams *)ud; const int il = ap.il;
    const ggml_tensor * q = dst->src[0];
    const int hd = (int)q->ne[0], Hq = (int)q->ne[1], T = (int)q->ne[2], G = Hq / c.aH, S = c.aS, NL = 1 << c.aD;
    std::vector<float> qs(hd), tab(S * NL);
    int a, b; part(T * Hq, ith, nth, a, b);
    for (int it = a; it < b; it++) {
        const int t = it / Hq, hq = it % Hq, h = hq / G;
        const float * qv = (const float *)((const char *)q->data + t * q->nb[2] + hq * q->nb[1]);
        for (int i = 0; i < hd; i++) qs[i] = qv[i] * ap.scale;
        const size_t o = (size_t)il * c.aH + h;
        double off = 0; float rmax = 1e-12f;
        for (int s = 0; s < S; s++) {
            float lo = 1e30f, hi = -1e30f;
            for (int l = 0; l < NL; l++) { float v = doth(qs.data(), c.ac16.data() + ((o * S + s) * NL + l) * hd, hd); tab[s * NL + l] = v; lo = std::min(lo, v); hi = std::max(hi, v); }
            off += lo; rmax = std::max(rmax, hi - lo);
            for (int l = 0; l < NL; l++) tab[s * NL + l] -= lo;
        }
        float * out = (float *)((char *)dst->data + it * dst->nb[1]);
        const float step = rmax / 65535.f; out[0] = (float)off; out[1] = step;
        uint8_t * tlo = (uint8_t *)(out + 2), * thi = tlo + S * NL;
        for (int e = 0; e < S * NL; e++) { int qq = (int)lrintf(tab[e] / step); qq = qq > 65535 ? 65535 : qq; tlo[e] = (uint8_t)(qq & 255); thi[e] = (uint8_t)(qq >> 8); }
    }
}

// args: tab, kq_mask [n_kv, >=T]   dst: [NP + C, Hq*T]: approximate scores (natural cell order, -inf where masked), chunk maxima
static void op_tscan(ggml_tensor * dst, int ith, int nth, void * ud) {
    OpTimer timer_(1, ith);
    const TreeCfg & c = cfg(); const AttnParams & ap = *(const AttnParams *)ud; AttnLayer & A = g_attn[ap.il];
    const ggml_tensor * TB = dst->src[0], * M = dst->src[1];
    const int n_kv = (int)M->ne[0], NP = (n_kv + 31) / 32 * 32, C = (int)dst->ne[0] - NP, HqT = (int)dst->ne[1];
    const int Hq = ap.Hq, G = Hq / c.aH, S = c.aS, P = S / 2, NL = 1 << c.aD;
    const __m256i m4 = _mm256_set1_epi8(15), m8 = _mm256_set1_epi16(255);
    for (int it = ith; it < HqT * C; it += nth) {                           // interleaved: chunk costs differ
        const int row = it / C, ch = it % C, t = row / Hq, h = (row % Hq) / G;      // rows are t-major, q-head-minor
        const float * tb = (const float *)((const char *)TB->data + row * TB->nb[1]);
        const float off = tb[0], step = tb[1];
        const uint8_t * tlo = (const uint8_t *)(tb + 2), * thi = tlo + S * NL;
        float * out = (float *)((char *)dst->data + row * dst->nb[1]);
        const char * mrow = (const char *)M->data + (size_t)t * M->nb[1];
        const __m256 vstep = _mm256_set1_ps(step), voff = _mm256_set1_ps(off);
        __m256 vmax = _mm256_set1_ps(-INFINITY);
        const int b0 = ch * (TCH / 32), b1 = std::min(NP / 32, b0 + TCH / 32);
        for (int bk = b0; bk < b1; bk++) {
            __m256i l0 = _mm256_setzero_si256(), l1 = l0, h0 = l0, h1 = l0;
            const uint8_t * cb = A.codes[h].data() + (size_t)bk * P * 32;
            for (int p = 0; p < P; p++) {
                __m256i cc = _mm256_loadu_si256((const __m256i *)(cb + p * 32));
                __m256i i0 = _mm256_and_si256(cc, m4), i1 = _mm256_and_si256(_mm256_srli_epi16(cc, 4), m4);
                __m256i L0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(tlo + 32 * p)));
                __m256i L1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(tlo + 32 * p + 16)));
                __m256i H0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(thi + 32 * p)));
                __m256i H1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(thi + 32 * p + 16)));
                __m256i lo = _mm256_shuffle_epi8(L0, i0), lo2 = _mm256_shuffle_epi8(L1, i1), hi = _mm256_shuffle_epi8(H0, i0), hi2 = _mm256_shuffle_epi8(H1, i1);
                l0 = _mm256_add_epi16(l0, _mm256_add_epi16(_mm256_and_si256(lo, m8), _mm256_and_si256(lo2, m8)));
                l1 = _mm256_add_epi16(l1, _mm256_add_epi16(_mm256_srli_epi16(lo, 8), _mm256_srli_epi16(lo2, 8)));
                h0 = _mm256_add_epi16(h0, _mm256_add_epi16(_mm256_and_si256(hi, m8), _mm256_and_si256(hi2, m8)));
                h1 = _mm256_add_epi16(h1, _mm256_add_epi16(_mm256_srli_epi16(hi, 8), _mm256_srli_epi16(hi2, 8)));
            }
            // l0/h0 hold even cells, l1/h1 odd cells; interleave to natural order: A = cells 0-7 | 16-23, B = 8-15 | 24-31
            const __m256i LA = _mm256_unpacklo_epi16(l0, l1), LB = _mm256_unpackhi_epi16(l0, l1);
            const __m256i HA = _mm256_unpacklo_epi16(h0, h1), HB = _mm256_unpackhi_epi16(h0, h1);
            const __m128i Ls[4] = {_mm256_castsi256_si128(LA), _mm256_castsi256_si128(LB), _mm256_extracti128_si256(LA, 1), _mm256_extracti128_si256(LB, 1)};
            const __m128i Hs[4] = {_mm256_castsi256_si128(HA), _mm256_castsi256_si128(HB), _mm256_extracti128_si256(HA, 1), _mm256_extracti128_si256(HB, 1)};
            float * o = out + bk * 32;
            for (int g8 = 0; g8 < 4; g8++) {
                const __m256i v = _mm256_add_epi32(_mm256_slli_epi32(_mm256_cvtepu16_epi32(Hs[g8]), 8), _mm256_cvtepu16_epi32(Ls[g8]));
                __m256 f = _mm256_fmadd_ps(vstep, _mm256_cvtepi32_ps(v), voff);
                const int j = bk * 32 + g8 * 8;
                if (j + 8 <= n_kv) {
                    const __m256 mk = M->type == GGML_TYPE_F16 ? _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(mrow + (size_t)j * 2)))
                                                               : _mm256_loadu_ps((const float *)mrow + j);
                    f = _mm256_add_ps(f, mk);
                    _mm256_storeu_ps(o + g8 * 8, f);
                } else {
                    _mm256_storeu_ps(o + g8 * 8, f);
                    for (int k = 0; k < 8; k++) {
                        const int jj = j + k;
                        const float mk = jj >= n_kv ? -INFINITY : M->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *)mrow)[jj]) : ((const float *)mrow)[jj];
                        o[g8 * 8 + k] += mk;
                    }
                    f = _mm256_loadu_ps(o + g8 * 8);
                }
                vmax = _mm256_max_ps(vmax, f);
            }
        }
        __m128 m = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
        m = _mm_max_ps(m, _mm_movehl_ps(m, m)); m = _mm_max_ss(m, _mm_shuffle_ps(m, m, 1));
        out[NP + ch] = _mm_cvtss_f32(m);
    }
}

// args: q [hd, Hq, T], K view [hd, H, n_kv] f16, V view, scan   dst: [G*(hd+2), C, H*T] = per head {max, sum, acc[hd]}
// exp for x <= 0 (softmax weights): Cephes-style range reduction + degree-5 polynomial, ~1e-7 relative error
static inline __m256 exp256(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.f));
    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(0.693359375f), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(-2.12194440e-4f), r);
    __m256 p = _mm256_set1_ps(1.9875691500E-4f);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.3981999507E-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(8.3334519073E-3f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(4.1665795894E-2f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.6666665459E-1f));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(5.0000001201E-1f));
    p = _mm256_fmadd_ps(p, _mm256_mul_ps(r, r), _mm256_add_ps(r, _mm256_set1_ps(1.f)));
    const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

template <int G>
static void tsel_item(const AttnParams & ap, const TreeCfg & c, const ggml_tensor * q, const ggml_tensor * K, const ggml_tensor * Vt,
                      const ggml_tensor * SC, ggml_tensor * dst, int t, int h, int ch, int C, int NP, int n_kv,
                      std::vector<int> & sel, std::vector<float> & sc, long long & nsel, long long & ntot) {
    const int hd = (int)q->ne[0], Hq = ap.Hq;
    const float * ap_[G]; float amax[G], qs[G][128];
    for (int g = 0; g < G; g++) {
        const int hq = h * G + g;
        ap_[g] = (const float *)((const char *)SC->data + (size_t)(t * Hq + hq) * SC->nb[1]);
        amax[g] = -INFINITY; for (int cc = 0; cc < C; cc++) amax[g] = std::max(amax[g], ap_[g][NP + cc]);
        const float * qv = (const float *)((const char *)q->data + t * q->nb[2] + hq * q->nb[1]);
        for (int i = 0; i < hd; i++) qs[g][i] = qv[i] * ap.scale;
    }
    int last = n_kv - 1; while (last >= 0 && ap_[0][last] == -INFINITY) last--;
    const int j0 = ch * TCH, j1 = std::min(std::min(n_kv, last + 1), j0 + TCH);
    int ns = 0, nv = 0;
    for (int j = j0; j < j1; j++) {
        if (ap_[0][j] == -INFINITY) continue;
        nv++;
        bool keep = j == 0 || j > last - c.recent;
        for (int g = 0; g < G && !keep; g++) keep = ap_[g][j] > amax[g] - c.tau;
        if (keep) sel[ns++] = j;
    }
    // exact scores: one K row read serves all G query heads; V rows are prefetched here for the second pass
    const bool vtrans = Vt->ne[0] != hd;
    const bool kv8 = kv8_on(); const AttnLayer & A = g_attn[ap.il];
    float mx[G]; for (int g = 0; g < G; g++) mx[g] = -INFINITY;
    for (int s2 = 0; s2 < ns; s2++) {
        if (kv8 && s2 + PFD < ns) {
            const char * pk = (const char *)(A.k8[h].data() + (size_t)sel[s2 + PFD] * hd), * pv = (const char *)(A.v8[h].data() + (size_t)sel[s2 + PFD] * hd);
            _mm_prefetch(pk, _MM_HINT_T0); _mm_prefetch(pk + 64, _MM_HINT_T0); _mm_prefetch(pv, _MM_HINT_T0); _mm_prefetch(pv + 64, _MM_HINT_T0);
        } else if (s2 + PFD < ns) {
            const char * pk = (const char *)K->data + (size_t)sel[s2 + PFD] * K->nb[2] + h * K->nb[1];
            for (int o = 0; o < hd * 2; o += 64) _mm_prefetch(pk + o, _MM_HINT_T0);
            if (!vtrans) { const char * pv = (const char *)Vt->data + (size_t)sel[s2 + PFD] * Vt->nb[2] + h * Vt->nb[1]; for (int o = 0; o < hd * 2; o += 64) _mm_prefetch(pv + o, _MM_HINT_T0); }
        }
        __m256 acc[G]; for (int g = 0; g < G; g++) acc[g] = _mm256_setzero_ps();
        float kscale = 1.f;
        if (kv8) {
            const int8_t * kr = A.k8[h].data() + (size_t)sel[s2] * hd; kscale = A.ks[h][sel[s2]];
            for (int i = 0; i < hd; i += 8) {
                const __m256 kk = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(kr + i))));
                for (int g = 0; g < G; g++) acc[g] = _mm256_fmadd_ps(_mm256_loadu_ps(qs[g] + i), kk, acc[g]);
            }
        } else {
            const ggml_fp16_t * kr = (const ggml_fp16_t *)((const char *)K->data + (size_t)sel[s2] * K->nb[2] + h * K->nb[1]);
            for (int i = 0; i < hd; i += 8) {
                const __m256 kk = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(kr + i)));
                for (int g = 0; g < G; g++) acc[g] = _mm256_fmadd_ps(_mm256_loadu_ps(qs[g] + i), kk, acc[g]);
            }
        }
        for (int g = 0; g < G; g++) {
            __m128 r = _mm_add_ps(_mm256_castps256_ps128(acc[g]), _mm256_extractf128_ps(acc[g], 1)); r = _mm_hadd_ps(r, r); r = _mm_hadd_ps(r, r);
            const float v = _mm_cvtss_f32(r) * kscale; sc[(size_t)g * TCH + s2] = v; mx[g] = std::max(mx[g], v);
        }
    }
    // softmax weights (in place), then values: 16 dims at a time with all G heads' accumulators in registers
    float * out = (float *)((char *)dst->data + (size_t)(t * c.aH + h) * dst->nb[2] + (size_t)ch * dst->nb[1]);
    float sum[G];
    for (int g = 0; g < G; g++) {
        float * w = sc.data() + (size_t)g * TCH; const __m256 vm = _mm256_set1_ps(mx[g]); __m256 vs = _mm256_setzero_ps();
        int s2 = 0;
        for (; s2 + 8 <= ns; s2 += 8) { const __m256 e = exp256(_mm256_sub_ps(_mm256_loadu_ps(w + s2), vm)); _mm256_storeu_ps(w + s2, e); vs = _mm256_add_ps(vs, e); }
        float st = 0; for (; s2 < ns; s2++) { w[s2] = expf(w[s2] - mx[g]); st += w[s2]; }
        __m128 r = _mm_add_ps(_mm256_castps256_ps128(vs), _mm256_extractf128_ps(vs, 1)); r = _mm_hadd_ps(r, r); r = _mm_hadd_ps(r, r);
        sum[g] = st + _mm_cvtss_f32(r);
    }
    if (kv8) {
        for (int i = 0; i < hd; i += 16) {
            __m256 a0[G], a1[G]; for (int g = 0; g < G; g++) { a0[g] = _mm256_setzero_ps(); a1[g] = _mm256_setzero_ps(); }
            for (int s2 = 0; s2 < ns; s2++) {
                const int8_t * vv = A.v8[h].data() + (size_t)sel[s2] * hd + i; const float vsc = A.vs[h][sel[s2]];
                const __m128i b = _mm_loadu_si128((const __m128i *)vv);
                const __m256 v0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b)), v1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(b, 8)));
                for (int g = 0; g < G; g++) { const __m256 w = _mm256_set1_ps(sc[(size_t)g * TCH + s2] * vsc); a0[g] = _mm256_fmadd_ps(w, v0, a0[g]); a1[g] = _mm256_fmadd_ps(w, v1, a1[g]); }
            }
            for (int g = 0; g < G; g++) { float * o = out + g * (hd + 2) + 2 + i; _mm256_storeu_ps(o, a0[g]); _mm256_storeu_ps(o + 8, a1[g]); }
        }
    } else if (!vtrans) {
        for (int i = 0; i < hd; i += 16) {
            __m256 a0[G], a1[G]; for (int g = 0; g < G; g++) { a0[g] = _mm256_setzero_ps(); a1[g] = _mm256_setzero_ps(); }
            for (int s2 = 0; s2 < ns; s2++) {
                const ggml_fp16_t * vv = (const ggml_fp16_t *)((const char *)Vt->data + (size_t)sel[s2] * Vt->nb[2] + h * Vt->nb[1]) + i;
                const __m256 v0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)vv)), v1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vv + 8)));
                for (int g = 0; g < G; g++) { const __m256 w = _mm256_broadcast_ss(sc.data() + (size_t)g * TCH + s2); a0[g] = _mm256_fmadd_ps(w, v0, a0[g]); a1[g] = _mm256_fmadd_ps(w, v1, a1[g]); }
            }
            for (int g = 0; g < G; g++) { float * o = out + g * (hd + 2) + 2 + i; _mm256_storeu_ps(o, a0[g]); _mm256_storeu_ps(o + 8, a1[g]); }
        }
    } else {
        for (int g = 0; g < G; g++) std::fill(out + g * (hd + 2) + 2, out + g * (hd + 2) + 2 + hd, 0.f);
        for (int s2 = 0; s2 < ns; s2++) for (int i = 0; i < hd; i++) {
            const float v = ggml_fp16_to_fp32(*(const ggml_fp16_t *)((const char *)Vt->data + (size_t)sel[s2] * Vt->nb[0] + h * Vt->nb[1] + (size_t)i * Vt->nb[2]));
            for (int g = 0; g < G; g++) out[g * (hd + 2) + 2 + i] += sc[(size_t)g * TCH + s2] * v;
        }
    }
    for (int g = 0; g < G; g++) { out[g * (hd + 2)] = mx[g]; out[g * (hd + 2) + 1] = sum[g]; }
    nsel += ns; ntot += nv;
}

static void op_tsel(ggml_tensor * dst, int ith, int nth, void * ud) {
    OpTimer timer_(2, ith);
    const TreeCfg & c = cfg(); const AttnParams & ap = *(const AttnParams *)ud;
    const ggml_tensor * q = dst->src[0], * K = dst->src[1], * Vt = dst->src[2], * SC = dst->src[3];
    const int T = (int)q->ne[2], H = c.aH, G = ap.Hq / H, n_kv = (int)K->ne[2], NP = (n_kv + 31) / 32 * 32, C = (int)dst->ne[1];
    std::vector<int> sel(TCH); std::vector<float> sc((size_t)TCH * G);
    long long nsel = 0, ntot = 0;
    for (int it = ith; it < T * H * C; it += nth) {                         // interleaved: the newest chunk is the heaviest
        const int ch = it % C, th = it / C, t = th / H, h = th % H;
        if (G == 4) tsel_item<4>(ap, c, q, K, Vt, SC, dst, t, h, ch, C, NP, n_kv, sel, sc, nsel, ntot);
        else if (G == 1) tsel_item<1>(ap, c, q, K, Vt, SC, dst, t, h, ch, C, NP, n_kv, sel, sc, nsel, ntot);
        else if (G == 2) tsel_item<2>(ap, c, q, K, Vt, SC, dst, t, h, ch, C, NP, n_kv, sel, sc, nsel, ntot);
        else if (G == 8) tsel_item<8>(ap, c, q, K, Vt, SC, dst, t, h, ch, C, NP, n_kv, sel, sc, nsel, ntot);
        else { fprintf(stderr, "tree-bitnet: unsupported GQA group %d\n", G); exit(1); }
    }
    g_att_sel += nsel; g_att_tot += ntot;
}

// args: partials   dst: [hd*Hq, T]
static void op_tmerge(ggml_tensor * dst, int ith, int nth, void * ud) {
    OpTimer timer_(3, ith);
    const TreeCfg & c = cfg(); const AttnParams & ap = *(const AttnParams *)ud;
    const ggml_tensor * PT = dst->src[0];
    const int Hq = ap.Hq, G = Hq / c.aH, hd = (int)dst->ne[0] / Hq, T = (int)dst->ne[1], C = (int)PT->ne[1];
    int a, b; part(T * Hq, ith, nth, a, b);
    for (int it = a; it < b; it++) {
        const int t = it / Hq, hq = it % Hq, h = hq / G, g = hq % G;
        float * out = (float *)((char *)dst->data + t * dst->nb[1]) + hq * hd;
        float mx = -INFINITY;
        for (int cc = 0; cc < C; cc++) { const float * p = (const float *)((const char *)PT->data + (size_t)(t * c.aH + h) * PT->nb[2] + cc * PT->nb[1]) + g * (hd + 2); if (p[1] > 0) mx = std::max(mx, p[0]); }
        std::fill(out, out + hd, 0.f); float sum = 0;
        for (int cc = 0; cc < C; cc++) {
            const float * p = (const float *)((const char *)PT->data + (size_t)(t * c.aH + h) * PT->nb[2] + cc * PT->nb[1]) + g * (hd + 2);
            if (!(p[1] > 0)) continue;
            const float w = expf(p[0] - mx); sum += w * p[1];
            for (int i = 0; i < hd; i++) out[i] += w * p[2 + i];
        }
        const float inv = sum > 0 ? 1.f / sum : 0.f;
        for (int i = 0; i < hd; i++) out[i] *= inv;
    }
}

ggml_tensor * tree_bitnet_attn(ggml_context * ctx, ggml_tensor * q_cur, ggml_tensor * k, ggml_tensor * v,
                               ggml_tensor * kq_mask, ggml_tensor ** deps, int n_deps, float kq_scale, int il) {
    const TreeCfg & c = cfg();
    if (k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) { fprintf(stderr, "tree-bitnet: tree attention needs an f16 KV cache\n"); exit(1); }
    if (q_cur->ne[0] > 128) { fprintf(stderr, "tree-bitnet: head dim > 128 unsupported\n"); exit(1); }
    const int hd = (int)q_cur->ne[0], Hq = (int)q_cur->ne[1], T = (int)q_cur->ne[2], n_kv = (int)k->ne[2];
    const int NP = (n_kv + 31) / 32 * 32, C = (n_kv + TCH - 1) / TCH, G = Hq / c.aH, NL = 1 << c.aD;
    g_attn_par[il] = {kq_scale, il, Hq};
    ggml_tensor * a0[9] = {q_cur};
    for (int i = 0; i < n_deps && i < 8; i++) a0[1 + i] = deps[i];
    ggml_tensor * tab = ggml_custom_4d(ctx, GGML_TYPE_F32, tab_floats(c.aS, NL), (int64_t)Hq * T, 1, 1, a0, 1 + std::min(n_deps, 8), op_ttab, GGML_N_TASKS_MAX, &g_attn_par[il]);
    ggml_tensor * a1[2] = {tab, kq_mask};
    ggml_tensor * scn = ggml_custom_4d(ctx, GGML_TYPE_F32, NP + C, (int64_t)Hq * T, 1, 1, a1, 2, op_tscan, GGML_N_TASKS_MAX, &g_attn_par[il]);
    ggml_tensor * a2[4] = {q_cur, k, v, scn};
    ggml_tensor * prt = ggml_custom_4d(ctx, GGML_TYPE_F32, (int64_t)G * (hd + 2), C, (int64_t)c.aH * T, 1, a2, 4, op_tsel, GGML_N_TASKS_MAX, &g_attn_par[il]);
    ggml_tensor * a3[1] = {prt};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, (int64_t)hd * Hq, T, 1, 1, a3, 1, op_tmerge, GGML_N_TASKS_MAX, &g_attn_par[il]);
}


// ------------------------------------------------------------------ TEAL-style activation sparsity (quality experiment)
static float g_teal[4] = {0, 0, 0, 0};
static bool teal_cfg() {
    static bool on = [] { const char * s = getenv("TREE_TEAL"); if (!s) return false;
        sscanf(s, "%f:%f:%f:%f", &g_teal[0], &g_teal[1], &g_teal[2], &g_teal[3]);
        fprintf(stderr, "tree-bitnet: TEAL sparsity qkv %.2f o %.2f gate/up %.2f down %.2f\n", g_teal[0], g_teal[1], g_teal[2], g_teal[3]);
        return true; }();
    return on;
}
static void op_teal(ggml_tensor * dst, int ith, int nth, void * ud) {
    const ggml_tensor * x = dst->src[0]; const float frac = g_teal[(int)(intptr_t)ud];
    const int n = (int)x->ne[0], T = (int)x->ne[1], k = (int)lrintf(frac * n);
    std::vector<float> a(n);
    int lo, hi; part(T, ith, nth, lo, hi);
    for (int t = lo; t < hi; t++) {
        const float * xr = (const float *)((const char *)x->data + t * x->nb[1]); float * o = (float *)((char *)dst->data + t * dst->nb[1]);
        for (int i = 0; i < n; i++) a[i] = fabsf(xr[i]);
        std::nth_element(a.begin(), a.begin() + k, a.end());
        const float thr = a[k];                                           // keep |x| >= k-th smallest magnitude
        for (int i = 0; i < n; i++) o[i] = fabsf(xr[i]) >= thr ? xr[i] : 0.f;
    }
}
ggml_tensor * tree_bitnet_teal(ggml_context * ctx, ggml_tensor * x, int site, int) {
    if (!teal_cfg() || g_teal[site] <= 0 || x->type != GGML_TYPE_F32) return x;
    ggml_tensor * xc = ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
    ggml_tensor * args[1] = {xc};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, xc->ne[0], xc->ne[1], xc->ne[2], xc->ne[3], args, 1, op_teal, GGML_N_TASKS_MAX, (void *)(intptr_t)site);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
