// Sparse exact BitNet MLP vs dense, one token, d=2560, F=6912, ternary weights in bitnet.cpp's I2_S packing
// (2 bits/weight, row-major; per 128 weights 32 bytes, byte j holds weights j, 32+j, 64+j, 96+j at bits 6,4,2,0;
// codes 0,1,2 = -1,0,+1). Same AVX2 maddubs dot kernel style as bitnet.cpp for every path.
//   dense : gate (F rows) + up (F rows) + down (d rows of F)
//   sparse: gate (F rows) -> select k neurons -> up (k rows) + down^T (k rows of d, accumulated)
// Usage: sparsemlp [k ...]        (single thread; prints us per MLP call)
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define D 2560
#define F 6912
#ifndef PF
#define PF 4
#endif
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs = 88172645463325252ull;
static inline uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 32); }

static uint8_t *pack_random(int rows, int cols) {                  // random ternary, ~1/3 zeros
    uint8_t *w = aligned_alloc(64, (size_t)rows * cols / 4);
    for (size_t i = 0; i < (size_t)rows * cols / 4; i++) {
        uint8_t b = 0;
        for (int k = 0; k < 4; k++) b |= (uint8_t)(rnd() % 3) << (6 - 2 * k);
        w[i] = b;
    }
    return w;
}

static inline int hsum(__m256i a) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, s));
    return _mm_cvtsi128_si32(_mm_add_epi32(s, _mm_shuffle_epi32(s, 1)));
}

// sum_j (code_j) * x_j over one row of n weights; caller subtracts sum(x) to get the ternary dot
static inline int dot_row(const uint8_t *w, const int8_t *x, int n) {
    const __m256i m3 = _mm256_set1_epi8(3), one = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i < n; i += 128, w += 32, x += 128) {
        __m256i b = _mm256_loadu_si256((const __m256i *)w);
        __m256i c0 = _mm256_and_si256(_mm256_srli_epi16(b, 6), m3), c1 = _mm256_and_si256(_mm256_srli_epi16(b, 4), m3);
        __m256i c2 = _mm256_and_si256(_mm256_srli_epi16(b, 2), m3), c3 = _mm256_and_si256(b, m3);
        __m256i s = _mm256_maddubs_epi16(c0, _mm256_loadu_si256((const __m256i *)x));
        s = _mm256_add_epi16(s, _mm256_maddubs_epi16(c1, _mm256_loadu_si256((const __m256i *)(x + 32))));
        s = _mm256_add_epi16(s, _mm256_maddubs_epi16(c2, _mm256_loadu_si256((const __m256i *)(x + 64))));
        s = _mm256_add_epi16(s, _mm256_maddubs_epi16(c3, _mm256_loadu_si256((const __m256i *)(x + 96))));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(s, one));
    }
    return hsum(acc);
}

static int sum_i8(const int8_t *x, int n) { int s = 0; for (int i = 0; i < n; i++) s += x[i]; return s; }

static void matvec(int32_t *y, const uint8_t *W, const int8_t *x, int rows, int cols) {
    int sx = sum_i8(x, cols);
    for (int r = 0; r < rows; r++) y[r] = dot_row(W + (size_t)r * cols / 4, x, cols) - sx;
}

// y[0..D) += a * row(code-1) for each selected neuron, int16 accumulators flushed every 256 neurons
static void axpy_rows(int32_t *y, const uint8_t *WT, const int *idx, const int8_t *a, int k) {
    const __m256i m3 = _mm256_set1_epi8(3), one8 = _mm256_set1_epi8(1);
    static int16_t acc[D] __attribute__((aligned(64)));
    memset(y, 0, D * sizeof(int32_t));
    for (int k0 = 0; k0 < k; k0 += 256) {
        memset(acc, 0, sizeof acc);
        int k1 = k0 + 256 < k ? k0 + 256 : k;
        for (int t = k0; t < k1; t++) {
            const uint8_t *w = WT + (size_t)idx[t] * D / 4;
            if (t + PF < k) { const char *pn = (const char *)(WT + (size_t)idx[t + PF] * D / 4); for (int c = 0; c < D / 4; c += 64) _mm_prefetch(pn + c, _MM_HINT_T0); }
            __m256i av = _mm256_set1_epi8(a[t]);
            for (int i = 0; i < D; i += 128, w += 32) {
                __m256i b = _mm256_loadu_si256((const __m256i *)w);
                __m256i c[4] = {_mm256_and_si256(_mm256_srli_epi16(b, 6), m3), _mm256_and_si256(_mm256_srli_epi16(b, 4), m3),
                                _mm256_and_si256(_mm256_srli_epi16(b, 2), m3), _mm256_and_si256(b, m3)};
                for (int q = 0; q < 4; q++) {
                    __m256i p = _mm256_sign_epi8(av, _mm256_sub_epi8(c[q], one8));      // a * w, w in {-1,0,1}
                    __m256i *o = (__m256i *)(acc + i + 32 * q);
                    o[0] = _mm256_add_epi16(o[0], _mm256_cvtepi8_epi16(_mm256_castsi256_si128(p)));
                    o[1] = _mm256_add_epi16(o[1], _mm256_cvtepi8_epi16(_mm256_extracti128_si256(p, 1)));
                }
            }
        }
        for (int i = 0; i < D; i++) y[i] += acc[i];
    }
}

static void quant8(int8_t *q, const float *x, int n) {
    float m = 1e-6f; for (int i = 0; i < n; i++) m = fmaxf(m, fabsf(x[i]));
    const float s = 127.f / m;
    for (int i = 0; i < n; i++) { int v = (int)lrintf(x[i] * s); q[i] = (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v); }
}

int main(int argc, char **argv) {
    uint8_t *Wg = pack_random(F, D), *Wu = pack_random(F, D), *Wd = pack_random(D, F), *WdT = pack_random(F, D);
    int8_t x[D] __attribute__((aligned(64))), hq[F] __attribute__((aligned(64))), hk[F];
    int32_t g[F], u[F], y[D]; float h[F]; int idx[F], bk[F];
    for (int i = 0; i < D; i++) x[i] = (int8_t)(rnd() % 255 - 127);
    int reps = 200;
    // dense
    double t0 = now();
    for (int r = 0; r < reps; r++) {
        matvec(g, Wg, x, F, D); matvec(u, Wu, x, F, D);
        for (int i = 0; i < F; i++) { float a = g[i] > 0 ? (float)g[i] : 0.f; h[i] = a * a * (float)u[i] * 1e-9f; }
        quant8(hq, h, F); matvec(y, Wd, hq, D, F);
    }
    double dense = (now() - t0) / reps * 1e6;
    t0 = now();
    for (int r = 0; r < reps; r++) matvec(y, Wd, hq, D, F);
    printf("dense down alone %.0f us\n", (now() - t0) / reps * 1e6);
    t0 = now();
    for (int r = 0; r < reps; r++) matvec(g, Wg, x, F, D);
    double gate = (now() - t0) / reps * 1e6;
    printf("dense MLP %.0f us  (gate alone %.0f us)\n", dense, gate);
    int ks[] = {512, 1024, 1536, 2048, 4096}; int nk = 5;
    if (argc > 1) { nk = argc - 1; for (int i = 0; i < nk; i++) ks[i] = atoi(argv[i + 1]); }
    for (int ki = 0; ki < nk; ki++) {
        int k = ks[ki];
        double tsel = 0, tup = 0, tdown = 0, t1;
        t0 = now();
        for (int r = 0; r < reps; r++) {
            matvec(g, Wg, x, F, D); t1 = now();
            // select top-k positive gates: threshold by partial selection on a histogram of g (O(F))
            int hist[256] = {0}, gmax = 1;
            for (int i = 0; i < F; i++) gmax = g[i] > gmax ? g[i] : gmax;
            const float bs = 255.f / (float)gmax;
            for (int i = 0; i < F; i++) { int b = g[i] > 0 ? (int)((float)g[i] * bs) : -1; bk[i] = b; if (b >= 0) hist[b]++; }
            int c = 0, thr = 255; while (thr > 0 && c + hist[thr] <= k) c += hist[thr--];
            int n = 0; int sx = sum_i8(x, D);
            for (int i = 0; i < F && n < k; i++) if (bk[i] > thr) idx[n++] = i;
            tsel += now() - t1; t1 = now();
            for (int t = 0; t < n; t++) { int i = idx[t];
                if (t + PF < n) { const char *pn = (const char *)(Wu + (size_t)idx[t + PF] * D / 4); for (int c = 0; c < D / 4; c += 64) _mm_prefetch(pn + c, _MM_HINT_T0); } int ui = dot_row(Wu + (size_t)i * D / 4, x, D) - sx; float a = (float)g[i]; h[t] = a * a * (float)ui * 1e-9f; }
            quant8(hk, h, n);
            tup += now() - t1; t1 = now();
            axpy_rows(y, WdT, idx, hk, n);
            tdown += now() - t1;
        }
        double sp = (now() - t0) / reps * 1e6;
        printf("sparse k=%4d: %.0f us  -> %.2fx vs dense   [select %.0f, up+act %.0f, down %.0f us]\n", k, sp, dense / sp,
               tsel / reps * 1e6, tup / reps * 1e6, tdown / reps * 1e6);
    }
    return 0;
}
