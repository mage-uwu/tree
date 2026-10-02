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

// ------------------------------------------------------------------ config
static std::atomic<long long> g_sel_n{0}, g_sel_calls{0};
static std::atomic<long long> g_att_sel{0}, g_att_tot{0};
static void print_stats() {
    if (g_att_tot) fprintf(stderr, "tree-bitnet: attention read %.1f%% of keys/values\n", 100.0 * (double)g_att_sel / (double)g_att_tot);
    if (g_sel_calls) fprintf(stderr, "tree-bitnet: mean selected neurons %.1f over %lld calls\n", (double)g_sel_n / g_sel_calls, (long long)g_sel_calls); }

struct TreeCfg {
    float mlp_frac = 0; int mlp_k = 0; bool all = false;
    int part_m = 0, part_C = 0;
    bool attn = false; float tau = 8.f; int recent = 64;
    int aL = 0, aH = 0, aS = 0, aD = 0, ahd = 0;
    std::vector<float> aw, ab, ac;       // [L][H][S][2^D-1][hd], [L][H][S][2^D-1], [L][H][S][2^D][hd]          // TREE_MLP_PARTIAL=m:C  outlier partial-sum candidate selection
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
struct AttnLayer { std::vector<uint8_t> codes[16]; int64_t cap = 0; };   // per kv head: [block][S/2][32] nibble codes
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

ggml_tensor * tree_bitnet_kenc(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int64_t kv_size, int il) {
    const TreeCfg & c = cfg(); AttnLayer & A = g_attn[il];
    if (A.cap < kv_size) {                                               // graph build is single-threaded
        const int P = c.aS / 2; const int64_t nb = (kv_size + 31) / 32;
        for (int h = 0; h < c.aH; h++) A.codes[h].assign((size_t)nb * P * 32, 0);
        A.cap = nb * 32;
    }
    ggml_tensor * args[2] = {k_cur, k_idxs};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, 1, 1, 1, 1, args, 2, op_kenc, GGML_N_TASKS_MAX, (void *)(intptr_t)il);
}

struct AttnParams { float scale; int il; };
static AttnParams g_attn_par[256];

// args: q [hd, Hq, T] f32, k view [hd, H, n_kv] f16, v view, kq_mask [n_kv, >=T], deps...
static void op_tattn(ggml_tensor * dst, int ith, int nth, void * ud) {
    const TreeCfg & c = cfg(); const AttnParams & ap = *(const AttnParams *)ud; const int il = ap.il; AttnLayer & A = g_attn[il];
    const ggml_tensor * q = dst->src[0], * K = dst->src[1], * Vt = dst->src[2], * M = dst->src[3];
    const int hd = (int)q->ne[0], Hq = (int)q->ne[1], T = (int)q->ne[2], H = (int)K->ne[1], G = Hq / H;
    const int n_kv = (int)K->ne[2], S = c.aS, P = S / 2, NL = 1 << c.aD;
    const bool vtrans = Vt->ne[0] != hd;                                    // transposed V cache: [n_kv, H, hd]
    std::vector<float> qs(hd), tab(S * NL), approx(((n_kv + 31) / 32) * 32), sc(n_kv), acc(hd);
    std::vector<int> sel(n_kv); std::vector<uint8_t> tlo(S * NL), thi(S * NL);
    long long nsel = 0, ntot = 0;
    int a, b; part(T * Hq, ith, nth, a, b);
    for (int it = a; it < b; it++) {
        const int t = it / Hq, hq = it % Hq, h = hq / G;
        const float * qv = (const float *)((const char *)q->data + t * q->nb[2] + hq * q->nb[1]);
        for (int i = 0; i < hd; i++) qs[i] = qv[i] * ap.scale;
        // valid cells from the stock mask (0 = attend, -inf = masked)
        auto valid = [&](int j) -> bool {
            const char * row = (const char *)M->data + (size_t)t * M->nb[1];
            float m = M->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *)row)[j]) : ((const float *)row)[j];
            return m > -1e30f;
        };
        int last = -1; for (int j = n_kv - 1; j >= 0; j--) if (valid(j)) { last = j; break; }
        if (last < 0) { std::fill((float *)((char *)dst->data + t * dst->nb[1]) + hq * hd, (float *)((char *)dst->data + t * dst->nb[1]) + (hq + 1) * hd, 0.f); continue; }
        const int n = last + 1;
        // leaf tables (scaled query . leaf vectors) and 16-bit two-plane quantization with a shared step
        const size_t o = (size_t)il * c.aH + h;
        double off = 0; float rmax = 1e-12f;
        for (int s = 0; s < S; s++) {
            float lo = 1e30f, hi = -1e30f;
            for (int l = 0; l < NL; l++) { float v = dotf(qs.data(), c.ac.data() + ((o * S + s) * NL + l) * hd, hd); tab[s * NL + l] = v; lo = std::min(lo, v); hi = std::max(hi, v); }
            off += lo; rmax = std::max(rmax, hi - lo);
            for (int l = 0; l < NL; l++) tab[s * NL + l] -= lo;
        }
        const float step = rmax / 65535.f;
        for (int e = 0; e < S * NL; e++) { int qq = (int)lrintf(tab[e] / step); qq = qq > 65535 ? 65535 : qq; tlo[e] = (uint8_t)(qq & 255); thi[e] = (uint8_t)(qq >> 8); }
        // fast-scan approximate scores for cells [0, n)
        const __m256i m4 = _mm256_set1_epi8(15), m8 = _mm256_set1_epi16(255);
        const int nblk = (n + 31) / 32;
        float amax = -1e30f;
        for (int bk = 0; bk < nblk; bk++) {
            __m256i l0 = _mm256_setzero_si256(), l1 = l0, h0 = l0, h1 = l0;
            const uint8_t * cb = A.codes[h].data() + (size_t)bk * P * 32;
            for (int p = 0; p < P; p++) {
                __m256i cc = _mm256_loadu_si256((const __m256i *)(cb + p * 32));
                __m256i i0 = _mm256_and_si256(cc, m4), i1 = _mm256_and_si256(_mm256_srli_epi16(cc, 4), m4);
                __m256i L0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(tlo.data() + 32 * p)));
                __m256i L1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(tlo.data() + 32 * p + 16)));
                __m256i H0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(thi.data() + 32 * p)));
                __m256i H1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(thi.data() + 32 * p + 16)));
                __m256i lo = _mm256_shuffle_epi8(L0, i0), lo2 = _mm256_shuffle_epi8(L1, i1), hi = _mm256_shuffle_epi8(H0, i0), hi2 = _mm256_shuffle_epi8(H1, i1);
                l0 = _mm256_add_epi16(l0, _mm256_add_epi16(_mm256_and_si256(lo, m8), _mm256_and_si256(lo2, m8)));
                l1 = _mm256_add_epi16(l1, _mm256_add_epi16(_mm256_srli_epi16(lo, 8), _mm256_srli_epi16(lo2, 8)));
                h0 = _mm256_add_epi16(h0, _mm256_add_epi16(_mm256_and_si256(hi, m8), _mm256_and_si256(hi2, m8)));
                h1 = _mm256_add_epi16(h1, _mm256_add_epi16(_mm256_srli_epi16(hi, 8), _mm256_srli_epi16(hi2, 8)));
            }
            alignas(32) uint16_t Lw[32], Hw[32];
            _mm256_store_si256((__m256i *)Lw, l0); _mm256_store_si256((__m256i *)(Lw + 16), l1);
            _mm256_store_si256((__m256i *)Hw, h0); _mm256_store_si256((__m256i *)(Hw + 16), h1);
            for (int kk = 0; kk < 32; kk++) {
                const int j = bk * 32 + (kk < 16 ? 2 * kk : 2 * (kk - 16) + 1);
                float v = (float)(off + (double)step * (256.0 * Hw[kk] + Lw[kk]));
                approx[j] = v;
                if (j < n && v > amax && valid(j)) amax = v;
            }
        }
        // select: within tau of the best tree score, plus the first cell and the newest `recent` cells
        int ns = 0;
        for (int j = 0; j < n; j++) {
            if (!valid(j)) continue;
            if (j == 0 || j >= n - c.recent || approx[j] > amax - c.tau) sel[ns++] = j;
        }
        // exact scores, softmax, values
        float mx = -1e30f;
        for (int s2 = 0; s2 < ns; s2++) {
            const int j = sel[s2];
            if (s2 + 4 < ns) _mm_prefetch((const char *)K->data + (size_t)sel[s2 + 4] * K->nb[2] + h * K->nb[1], _MM_HINT_T0);
            sc[s2] = doth(qs.data(), (const ggml_fp16_t *)((const char *)K->data + (size_t)j * K->nb[2] + h * K->nb[1]), hd);
            mx = std::max(mx, sc[s2]);
        }
        float sum = 0; std::fill(acc.begin(), acc.end(), 0.f);
        for (int s2 = 0; s2 < ns; s2++) {
            const float w = expf(sc[s2] - mx); sum += w; const int j = sel[s2];
            if (!vtrans) {
                const ggml_fp16_t * vv = (const ggml_fp16_t *)((const char *)Vt->data + (size_t)j * Vt->nb[2] + h * Vt->nb[1]);
                const __m256 wv = _mm256_set1_ps(w);
                for (int i = 0; i < hd; i += 8) _mm256_storeu_ps(acc.data() + i, _mm256_fmadd_ps(wv, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(vv + i))), _mm256_loadu_ps(acc.data() + i)));
            } else {
                for (int i = 0; i < hd; i++) acc[i] += w * ggml_fp16_to_fp32(*(const ggml_fp16_t *)((const char *)Vt->data + (size_t)j * Vt->nb[0] + h * Vt->nb[1] + (size_t)i * Vt->nb[2]));
            }
        }
        float * out = (float *)((char *)dst->data + t * dst->nb[1]) + hq * hd;
        const float inv = sum > 0 ? 1.f / sum : 0.f;
        for (int i = 0; i < hd; i++) out[i] = acc[i] * inv;
        nsel += ns; ntot += n;
    }
    g_att_sel += nsel; g_att_tot += ntot;
}

ggml_tensor * tree_bitnet_attn(ggml_context * ctx, ggml_tensor * q_cur, ggml_tensor * k, ggml_tensor * v,
                               ggml_tensor * kq_mask, ggml_tensor ** deps, int n_deps, float kq_scale, int il) {
    if (k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16) { fprintf(stderr, "tree-bitnet: tree attention needs an f16 KV cache\n"); exit(1); }
    g_attn_par[il] = {kq_scale, il};
    ggml_tensor * args[12] = {q_cur, k, v, kq_mask};
    for (int i = 0; i < n_deps && i < 8; i++) args[4 + i] = deps[i];
    return ggml_custom_4d(ctx, GGML_TYPE_F32, q_cur->ne[0] * q_cur->ne[1], q_cur->ne[2], 1, 1, args, 4 + n_deps, op_tattn, GGML_N_TASKS_MAX, &g_attn_par[il]);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
