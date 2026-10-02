// microbench.c — one FFN layer at BERT-base width, random ternary weights.
// dense ternary FFN (d -> N -> d) vs ternary FFF tree (N nodes, depth log2(N+1)).
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static inline int32_t dot8(const int8_t *restrict a, const int8_t *restrict b, int n) { int32_t s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }
static inline float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678f)); }
static void rnd_tern(int8_t *w, size_t n) { for (size_t i = 0; i < n; i++) w[i] = (int8_t)(rand() % 3) - 1; }
static float quant8(int8_t *q, const float *x, int n) {
    float mx = 1e-5f; for (int i = 0; i < n; i++) if (fabsf(x[i]) > mx) mx = fabsf(x[i]);
    float s = 127.f / mx; for (int i = 0; i < n; i++) q[i] = (int8_t)rintf(x[i] * s); return 1 / s;
}

int main(int argc, char **argv) {
    int d = argc > 1 ? atoi(argv[1]) : 768, depth = argc > 2 ? atoi(argv[2]) : 12, iters = 2000;
    int N = (1 << depth) - 1;
    int8_t *up = malloc((size_t)N * d), *down = malloc((size_t)d * N), *win = malloc((size_t)N * d), *wout = malloc((size_t)N * d);
    rnd_tern(up, (size_t)N * d); rnd_tern(down, (size_t)N * d); rnd_tern(win, (size_t)N * d); rnd_tern(wout, (size_t)N * d);
    float *x = malloc(d * 4), *y = malloc(d * 4), *h = malloc(N * 4);
    int8_t *xq = malloc(d), *hq = malloc(N);
    volatile float sink = 0;

    double t0 = now();
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < d; i++) x[i] = sinf(i * 0.37f + it);
        float xi = quant8(xq, x, d);
        for (int r = 0; r < N; r++) h[r] = gelu(dot8(up + (size_t)r * d, xq, d) * xi);
        float hi = quant8(hq, h, N);
        for (int r = 0; r < d; r++) y[r] = dot8(down + (size_t)r * N, hq, N) * hi;
        sink += y[it % d];
    }
    double td = (now() - t0) / iters;

    t0 = now();
    for (int it = 0; it < iters; it++) {
        for (int i = 0; i < d; i++) { x[i] = sinf(i * 0.37f + it); y[i] = 0; }
        float xi = quant8(xq, x, d);
        int node = 0;
        for (int lv = 0; lv < depth; lv++) {
            int32_t acc = dot8(win + (size_t)node * d, xq, d);
            float hs = gelu(acc * xi * 0.01f);
            const int8_t *wo = wout + (size_t)node * d;
            for (int i = 0; i < d; i++) y[i] += hs * wo[i];
            node = 2 * node + 1 + (acc > 0);
        }
        sink += y[it % d];
    }
    double tt = (now() - t0) / iters;
    printf("d=%d nodes=%d depth=%d | dense ternary FFN: %.1f us | tree path: %.2f us | speedup %.0fx | weight bytes touched (2-bit): %.0f KB vs %.2f KB\n",
           d, N, depth, td * 1e6, tt * 1e6, td / tt, 2.0 * N * d / 4 / 1024, 2.0 * depth * d / 4 / 1024);
    return 0;
}
