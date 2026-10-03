// Ternary (I2_S) matvec: AVX2 (as in bitnet.cpp / tree-bitnet dot_codes) vs AVX-512 VNNI, against a pure read of the
// same bytes (the memory roofline). Rows of n int8 activations x 2-bit codes; working set larger than the caches, swept
// like one decode step. Build: gcc -O3 -march=native -fopenmp avx512bench.c -o avx512bench
// Run: ./avx512bench <MB> <n> ; OMP_NUM_THREADS sets threads.
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline int hsum256(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_unpackhi_epi64(s, s));
    return _mm_cvtsi128_si32(_mm_add_epi32(s, _mm_shuffle_epi32(s, 1)));
}
// AVX2: per 128 weights, 32 code bytes; byte j holds weights j, 32+j, 64+j, 96+j at bits 6,4,2,0
static inline int dot_avx2(const uint8_t *w, const int8_t *x, int n) {
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
    return hsum256(acc);
}
// AVX-512 VNNI: 64 code bytes = two 128-weight groups; activations pre-permuted per token so that sub-block p of both
// groups sits in one 64-byte vector: xp[(g/2)*256 + p*64 + {0..31 from group g, 32..63 from group g+1}]
static void permute_x(const int8_t *x, int8_t *xp, int n) {
    for (int g = 0; g < n / 128; g += 2)
        for (int p = 0; p < 4; p++) {
            memcpy(xp + (g / 2) * 256 + p * 64, x + g * 128 + p * 32, 32);
            memcpy(xp + (g / 2) * 256 + p * 64 + 32, x + (g + 1) * 128 + p * 32, 32);
        }
}
// 4 rows at once (activation loads shared); n multiple of 256
static inline void dot4_vnni(const uint8_t *w, size_t stride, const int8_t *xp, int n, int *out) {
    const __m512i m3 = _mm512_set1_epi8(3);
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    for (int i = 0; i < n; i += 256, w += 64, xp += 256) {
        __m512i x0 = _mm512_loadu_si512(xp), x1 = _mm512_loadu_si512(xp + 64), x2 = _mm512_loadu_si512(xp + 128), x3 = _mm512_loadu_si512(xp + 192);
#define ROW(A, off) { __m512i b = _mm512_loadu_si512(w + (off)); \
        A = _mm512_dpbusd_epi32(A, _mm512_and_si512(_mm512_srli_epi16(b, 6), m3), x0); \
        A = _mm512_dpbusd_epi32(A, _mm512_and_si512(_mm512_srli_epi16(b, 4), m3), x1); \
        A = _mm512_dpbusd_epi32(A, _mm512_and_si512(_mm512_srli_epi16(b, 2), m3), x2); \
        A = _mm512_dpbusd_epi32(A, _mm512_and_si512(b, m3), x3); }
        ROW(a0, 0) ROW(a1, stride) ROW(a2, 2 * stride) ROW(a3, 3 * stride)
    }
    out[0] = _mm512_reduce_add_epi32(a0); out[1] = _mm512_reduce_add_epi32(a1);
    out[2] = _mm512_reduce_add_epi32(a2); out[3] = _mm512_reduce_add_epi32(a3);
}
static int PF = 0;
static inline int dot_avx2_pf(const uint8_t *w, const int8_t *x, int n) {
    for (int o = 0; o < n / 4; o += 64) _mm_prefetch((const char *)w + PF + o, _MM_HINT_T0);
    return dot_avx2(w, x, n);
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char **argv) {
    size_t MB = argc > 1 ? atoi(argv[1]) : 512; int n = argc > 2 ? atoi(argv[2]) : 2560;
    size_t rb = n / 4, rows = (MB << 20) / rb; rows -= rows % 4;
    uint8_t *W = aligned_alloc(64, rows * rb); int8_t *x = aligned_alloc(64, n), *xp = aligned_alloc(64, n);
    int *y1 = malloc(rows * 4), *y2 = malloc(rows * 4);
    srand(1);
    for (size_t i = 0; i < rows * rb; i++) { uint8_t b = 0; for (int k = 0; k < 4; k++) b = (b << 2) | (rand() % 3); W[i] = b; }
    for (int i = 0; i < n; i++) x[i] = (int8_t)(rand() % 255 - 127);
    permute_x(x, xp, n);
    PF = getenv("PF") ? atoi(getenv("PF")) : 4096;
    double best[4] = {1e9, 1e9, 1e9, 1e9}; volatile long sink = 0;
    for (int rep = 0; rep < 7; rep++) {
        double t = now();
        #pragma omp parallel for schedule(static) reduction(+:sink)
        for (size_t r = 0; r < rows * rb; r += 256) {                      // pure read roofline
            __m512i v = _mm512_load_si512(W + r), u = _mm512_load_si512(W + r + 64);
            v = _mm512_xor_si512(v, _mm512_load_si512(W + r + 128)); u = _mm512_xor_si512(u, _mm512_load_si512(W + r + 192));
            sink += _mm512_reduce_add_epi32(_mm512_xor_si512(u, v)) & 1;
        }
        t = now() - t; if (t < best[0]) best[0] = t;
        t = now();
        #pragma omp parallel for schedule(static)
        for (size_t r = 0; r < rows; r++) y1[r] = dot_avx2(W + r * rb, x, n);
        t = now() - t; if (t < best[1]) best[1] = t;
        t = now();
        #pragma omp parallel for schedule(static)
        for (size_t r = 0; r < rows; r += 4) dot4_vnni(W + r * rb, rb, xp, n, y2 + r);
        t = now() - t; if (t < best[2]) best[2] = t;
        t = now();
        #pragma omp parallel for schedule(static)
        for (size_t r = 0; r < rows; r++) y1[r] = dot_avx2_pf(W + r * rb, x, n);
        t = now() - t; if (t < best[3]) best[3] = t;
    }
    for (size_t r = 0; r < rows; r++) if (y1[r] != y2[r]) { printf("MISMATCH row %zu: %d vs %d\n", r, y1[r], y2[r]); return 1; }
    double gb = rows * rb / 1e9;
    printf("threads %d  %.0f MB  n=%d | read %.1f GB/s | AVX2 %.1f GB/s (%.2f ms) | AVX-512 VNNI %.1f GB/s (%.2f ms) | VNNI/AVX2 %.2fx | AVX2+prefetch(%d) %.1f GB/s\n",
           omp_get_max_threads(), gb * 1e3, n, gb / best[0], gb / best[1], best[1] * 1e3, gb / best[2], best[2] * 1e3, best[1] / best[2], PF, gb / best[3]);
    return 0;
}
