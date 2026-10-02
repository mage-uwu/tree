// MLP decode-step timing, one thread, real sizes (d=2560, F=6912), random ternary weights in I2_S packing:
//   dense            : gate + up + down over all 6912 neurons
//   exact-gate sparse: dense gate, then up/down for k neurons (k = 1916, mean of energy rule 0.99, phase 5)
//   partial selector : exact sum over the m=256 largest-|x| input dims (transposed gate rows), top C=3072
//                      candidates get an exact gate, then up/down for k = 1556 (mean measured in the all-layer run)
// Reuses the kernels of sparsemlp.c.
#define main sparsemlp_main
#include "sparsemlp.c"
#undef main

// y[0..n) = sum_t a[t] * row(idx[t]) of a transposed matrix with rows of length n (codes-1)
static void axpy_n(int32_t *y, const uint8_t *WT, int n, const int *idx, const int8_t *a, int k) {
    const __m256i m3 = _mm256_set1_epi8(3), one8 = _mm256_set1_epi8(1);
    static int16_t acc[F] __attribute__((aligned(64)));
    memset(y, 0, n * sizeof(int32_t));
    for (int k0 = 0; k0 < k; k0 += 256) {
        memset(acc, 0, n * sizeof(int16_t));
        int k1 = k0 + 256 < k ? k0 + 256 : k;
        for (int t = k0; t < k1; t++) {
            const uint8_t *w = WT + (size_t)idx[t] * n / 4;
            if (t + 4 < k1) { const char *pn = (const char *)(WT + (size_t)idx[t + 4] * n / 4); for (int c = 0; c < n / 4; c += 64) _mm_prefetch(pn + c, _MM_HINT_T0); }
            __m256i av = _mm256_set1_epi8(a[t]);
            for (int i = 0; i < n; i += 128, w += 32) {
                __m256i b = _mm256_loadu_si256((const __m256i *)w);
                __m256i c[4] = {_mm256_and_si256(_mm256_srli_epi16(b, 6), m3), _mm256_and_si256(_mm256_srli_epi16(b, 4), m3),
                                _mm256_and_si256(_mm256_srli_epi16(b, 2), m3), _mm256_and_si256(b, m3)};
                for (int q = 0; q < 4; q++) {
                    __m256i p = _mm256_sign_epi8(av, _mm256_sub_epi8(c[q], one8));
                    __m256i *o = (__m256i *)(acc + i + 32 * q);
                    o[0] = _mm256_add_epi16(o[0], _mm256_cvtepi8_epi16(_mm256_castsi256_si128(p)));
                    o[1] = _mm256_add_epi16(o[1], _mm256_cvtepi8_epi16(_mm256_extracti128_si256(p, 1)));
                }
            }
        }
        for (int i = 0; i < n; i++) y[i] += acc[i];
    }
}

// indices of the k largest values of v[0..n) (histogram threshold, O(n)); returns count
static int topk_idx(const int32_t *v, int n, int k, int *idx) {
    int32_t mx = 1; for (int i = 0; i < n; i++) mx = v[i] > mx ? v[i] : mx;
    int hist[1024] = {0}; static int bk[F]; float bs = 1023.f / (float)mx;
    for (int i = 0; i < n; i++) { int b = v[i] > 0 ? (int)((float)v[i] * bs) : -1; bk[i] = b; if (b >= 0) hist[b]++; }
    int c = 0, thr = 1023; while (thr > 0 && c + hist[thr] <= k) c += hist[thr--];
    int m = 0; for (int i = 0; i < n && m < k; i++) if (bk[i] > thr) idx[m++] = i;
    return m;
}

static void up_down(int32_t *y, const int32_t *g, const uint8_t *Wu, const uint8_t *WdT, const int8_t *x, const int *sel, int n) {
    static float h[F]; static int8_t hk[F]; int sx = sum_i8(x, D);
    for (int t = 0; t < n; t++) {
        int i = sel[t];
        if (t + 4 < n) { const char *pn = (const char *)(Wu + (size_t)sel[t + 4] * D / 4); for (int c = 0; c < D / 4; c += 64) _mm_prefetch(pn + c, _MM_HINT_T0); }
        int ui = dot_row(Wu + (size_t)i * D / 4, x, D) - sx; float a = (float)g[i]; h[t] = a * a * (float)ui * 1e-9f;
    }
    quant8(hk, h, n);
    axpy_rows(y, WdT, sel, hk, n);
}

int main(int argc, char **argv) {
    const int m = argc > 1 ? atoi(argv[1]) : 256, C = argc > 2 ? atoi(argv[2]) : 3072;
    const int k_part = argc > 3 ? atoi(argv[3]) : 1556, k_gate = argc > 4 ? atoi(argv[4]) : 1916;
    uint8_t *Wg = pack_random(F, D), *Wu = pack_random(F, D), *Wd = pack_random(D, F), *WdT = pack_random(F, D);
    uint8_t *WgT = pack_random(D, F);                                   // transposed gate: D rows of F
    static int8_t x[D] __attribute__((aligned(64))), hq[F], xa[D];
    static int32_t g[F], u[F], y[D], gh[F]; static float h[F]; static int sel[F], cand[F], dims[D];
    for (int i = 0; i < D; i++) { int v = (int)(rnd() % 61) - 30; if (rnd() % 40 == 0) v = (int)(rnd() % 255) - 127; x[i] = (int8_t)v; }  // heavy-tailed input
    const int reps = 300, rounds = 5;
    double best[3] = {1e9, 1e9, 1e9};
    for (int rd = 0; rd < rounds; rd++) {
        double t0 = now();
        for (int r = 0; r < reps; r++) {                                  // dense
            matvec(g, Wg, x, F, D); matvec(u, Wu, x, F, D);
            for (int i = 0; i < F; i++) { float a = g[i] > 0 ? (float)g[i] : 0.f; h[i] = a * a * (float)u[i] * 1e-9f; }
            quant8(hq, h, F); matvec(y, Wd, hq, D, F);
        }
        double t = (now() - t0) / reps * 1e6; best[0] = t < best[0] ? t : best[0];
        t0 = now();
        for (int r = 0; r < reps; r++) {                                  // exact-gate sparse
            matvec(g, Wg, x, F, D);
            int n = topk_idx(g, F, k_gate, sel);
            up_down(y, g, Wu, WdT, x, sel, n);
        }
        t = (now() - t0) / reps * 1e6; best[1] = t < best[1] ? t : best[1];
        t0 = now();
        for (int r = 0; r < reps; r++) {                                  // partial-sum selector
            int32_t ax[D]; for (int i = 0; i < D; i++) ax[i] = x[i] < 0 ? -x[i] : x[i];
            int nm = topk_idx(ax, D, m, dims);
            for (int j = 0; j < nm; j++) xa[j] = x[dims[j]];
            axpy_n(gh, WgT, F, dims, xa, nm);                             // approximate gate from m input dims
            int nc = topk_idx(gh, F, C, cand);
            int sx = sum_i8(x, D);
            for (int t2 = 0; t2 < nc; t2++) {                             // exact gate on candidates
                int i = cand[t2];
                if (t2 + 4 < nc) { const char *pn = (const char *)(Wg + (size_t)cand[t2 + 4] * D / 4); for (int c = 0; c < D / 4; c += 64) _mm_prefetch(pn + c, _MM_HINT_T0); }
                g[i] = dot_row(Wg + (size_t)i * D / 4, x, D) - sx;
            }
            for (int t2 = 0; t2 < nc; t2++) u[t2] = g[cand[t2]];          // select k among candidates
            int n = topk_idx(u, nc, k_part, sel);
            for (int t2 = 0; t2 < n; t2++) sel[t2] = cand[sel[t2]];
            up_down(y, g, Wu, WdT, x, sel, n);
        }
        t = (now() - t0) / reps * 1e6; best[2] = t < best[2] ? t : best[2];
    }
    printf("MLP per token, 1 thread (best of %d): dense %.0f us | exact-gate sparse k=%d %.0f us (%.2fx) | partial m=%d C=%d k=%d %.0f us (%.2fx)\n",
           rounds, best[0], k_gate, best[1], best[0] / best[1], m, C, k_part, best[2], best[0] / best[2]);
    return 0;
}
