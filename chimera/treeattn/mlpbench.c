// mlpbench.c — isolated MLP timing: dense ternary BitNet MLP (LUT kernel) vs tree-MLP (int8 leaf low-rank maps).
// Random weights; measures runtime only. Usage: ./mlpbench d hidden D r [calls]
//   dense: norm -> quant -> up (LUT) -> GELU -> quant -> down (LUT)
//   tree : norm -> route (D float dots) -> quant -> r x (int8 dot + int8 axpy) + leaf constant
// Leaves are visited in random order (cold) or a single leaf repeatedly (warm) to separate compute from memory.
#include <math.h>
#include <stdio.h>
#include <time.h>
#include "../attn/lut.h"
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rng = 88172645463325252ull;
static inline uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 32); }

static inline float dot_ps(const float *a, const float *b, int n) {
    __m256 s = _mm256_setzero_ps(); int i = 0;
    for (; i + 8 <= n; i += 8) s = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s);
    __m128 t = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    t = _mm_hadd_ps(t, t); t = _mm_hadd_ps(t, t); float r = _mm_cvtss_f32(t);
    for (; i < n; i++) r += a[i] * b[i];
    return r;
}
static inline int32_t dot_i8(const int8_t *a, const int8_t *b, int n) {
    const __m256i one = _mm256_set1_epi16(1); __m256i acc = _mm256_setzero_si256(); int i = 0;
    for (; i + 32 <= n; i += 32) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i)), vb = _mm256_loadu_si256((const __m256i *)(b + i));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(va), _mm256_sign_epi8(vb, va)), one));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s); int32_t r = _mm_cvtsi128_si32(s);
    for (; i < n; i++) r += a[i] * b[i];
    return r;
}
static inline void axpy_i8(float *y, const int8_t *x, float a, int n) {
    __m256 va = _mm256_set1_ps(a); int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 vx = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(x + i))));
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, vx, _mm256_loadu_ps(y + i)));
    }
    for (; i < n; i++) y[i] += a * x[i];
}
// vectorised norm + int8 absmax quant (shared by both paths)
static void rmsnorm(float *o, const float *x, int n) {
    float ss = dot_ps(x, x, n), r = 1.0f / sqrtf(ss / n + 1e-6f);
    __m256 vr = _mm256_set1_ps(r); for (int i = 0; i < n; i += 8) _mm256_storeu_ps(o + i, _mm256_mul_ps(vr, _mm256_loadu_ps(x + i)));
}
static float quant8(int8_t *q, const float *x, int n) {
    __m256 mx = _mm256_setzero_ps(), sgn = _mm256_set1_ps(-0.0f);
    for (int i = 0; i < n; i += 8) mx = _mm256_max_ps(mx, _mm256_andnot_ps(sgn, _mm256_loadu_ps(x + i)));
    float t[8]; _mm256_storeu_ps(t, mx); float m = 1e-5f; for (int i = 0; i < 8; i++) if (t[i] > m) m = t[i];
    float s = 127.0f / m; __m256 vs = _mm256_set1_ps(s);
    for (int i = 0; i < n; i += 8) {
        __m256i v = _mm256_cvtps_epi32(_mm256_mul_ps(vs, _mm256_loadu_ps(x + i)));
        __m128i w = _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        _mm_storel_epi64((__m128i *)(q + i), _mm_packs_epi16(w, w));
    }
    return 1.0f / s;
}
static float GT[65536];                                    // GELU lookup on a 16-bit grid (ggml-style)
static inline float gelu_tab(float a) { int i = (int)(a * 2048.0f) + 32768; i = i < 0 ? 0 : i > 65535 ? 65535 : i; return GT[i]; }

int main(int argc, char **argv) {
    int d = argc > 1 ? atoi(argv[1]) : 128, Hd = argc > 2 ? atoi(argv[2]) : 512, D = argc > 3 ? atoi(argv[3]) : 10, r = argc > 4 ? atoi(argv[4]) : 16;
    int calls = argc > 5 ? atoi(argv[5]) : 20000, L = 1 << D;
    lut_init();
    for (int i = 0; i < 65536; i++) { float a = (i - 32768) / 2048.0f; GT[i] = 0.5f * a * (1.0f + erff(a * 0.70710678f)); }
    int8_t *up = malloc((size_t)Hd * d), *down = malloc((size_t)d * Hd);
    for (size_t i = 0; i < (size_t)Hd * d; i++) { up[i] = (int8_t)(rnd() % 3) - 1; down[i] = (int8_t)(rnd() % 3) - 1; }
    LutW Lu = lut_pack(up, Hd, d), Ld = lut_pack(down, d, Hd);
    int mx = Hd > d ? Hd : d;
    uint8_t *tab = aligned_alloc(64, ((size_t)(mx + 2) / 3 * 64 + 63) / 64 * 64);
    int32_t *acc = malloc(((mx + 127) / 128 * 128) * 4), *scr = aligned_alloc(64, ((mx + 127) / 128 * 128) * 4);
    float *x = aligned_alloc(64, d * 4), *u = aligned_alloc(64, d * 4), *o = aligned_alloc(64, d * 4), *h = aligned_alloc(64, (Hd + 63) / 64 * 64 * 4);
    int8_t *xq = aligned_alloc(64, (mx + 63) / 64 * 64);
    // tree storage
    float *tw = malloc((size_t)(L - 1) * d * 4), *tb = calloc(L - 1, 4), *tc = malloc((size_t)L * d * 4), *sP = malloc((size_t)L * r * 4);
    int8_t *P8 = malloc((size_t)L * r * d), *Q8 = malloc((size_t)L * r * d);
    for (size_t i = 0; i < (size_t)(L - 1) * d; i++) tw[i] = (float)(int)(rnd() % 2001 - 1000) / 1000.0f;
    for (size_t i = 0; i < (size_t)L * d; i++) tc[i] = 0.01f;
    for (size_t i = 0; i < (size_t)L * r * d; i++) { P8[i] = (int8_t)(rnd() % 255) - 127; Q8[i] = (int8_t)(rnd() % 255) - 127; }
    for (size_t i = 0; i < (size_t)L * r; i++) sP[i] = 1e-4f;
    volatile float sink = 0; double t0;

#define NEWX for (int i = 0; i < d; i++) x[i] = (float)(int)(rnd() % 2001 - 1000) / 500.0f;
    // ---- dense, exact erf GELU
    double td[2];
    for (int g = 0; g < 2; g++) {
        t0 = now();
        for (int c = 0; c < calls; c++) {
            NEWX rmsnorm(u, x, d);
            float xi = quant8(xq, u, d); lut_tables(tab, xq, d); lut_mv(acc, &Lu, tab, scr);
            float f = 0.01f * xi;
            if (g == 0) for (int i = 0; i < Hd; i++) { float a = acc[i] * f; h[i] = 0.5f * a * (1.0f + erff(a * 0.70710678f)); }
            else for (int i = 0; i < Hd; i++) h[i] = gelu_tab(acc[i] * f);
            float hi = quant8(xq, h, Hd); lut_tables(tab, xq, Hd); lut_mv(acc, &Ld, tab, scr);
            for (int i = 0; i < d; i++) o[i] = acc[i] * hi;
            sink += o[c % d];
        }
        td[g] = (now() - t0) / calls;
    }
    // ---- tree, cold (data-dependent leaf) and warm (same leaf)
    double tt[2];
    for (int warm = 0; warm < 2; warm++) {
        t0 = now();
        for (int c = 0; c < calls * 4; c++) {
            NEWX rmsnorm(u, x, d);
            int node = 0;
            for (int lv = 0; lv < D; lv++) node = 2 * node + 1 + (dot_ps(tw + (size_t)(warm ? lv : node) * d, u, d) - tb[warm ? lv : node] > 0);
            int leaf = warm ? 0 : node - (L - 1);
            float ui = quant8(xq, u, d);
            memcpy(o, tc + (size_t)leaf * d, d * 4);
            for (int j = 0; j < r; j++) {
                size_t lj = (size_t)leaf * r + j, off = lj * d;
                axpy_i8(o, Q8 + off, (float)dot_i8(P8 + off, xq, d) * sP[lj] * ui, d);
            }
            sink += o[c % d];
        }
        tt[warm] = (now() - t0) / (calls * 4);
    }
    // input generation overhead
    t0 = now(); for (int c = 0; c < calls * 4; c++) { NEWX sink += x[c % d]; } double tx = (now() - t0) / (calls * 4);
    double leafMB = (double)L * r * d * 2 / 1e6;
    printf("d=%d hidden=%d | tree D=%d (%d leaves) r=%d, int8 leaf maps %.1f MB/layer (dense ternary MLP: %.2f MB @2bit)\n", d, Hd, D, L, r, leafMB, 2.0 * Hd * d * 2 / 8 / 1e6);
    printf("  dense (exact erf GELU)  %9.3f us\n  dense (table GELU)      %9.3f us\n  tree  (random leaves)   %9.3f us   -> %.1fx vs table-GELU dense, %.1fx vs exact\n  tree  (same leaf, warm) %9.3f us   -> %.1fx vs table-GELU dense\n",
           (td[0] - tx) * 1e6, (td[1] - tx) * 1e6, (tt[0] - tx) * 1e6, (td[1] - tx) / (tt[0] - tx), (td[0] - tx) / (tt[0] - tx), (tt[1] - tx) * 1e6, (td[1] - tx) / (tt[1] - tx));
    return 0;
}
