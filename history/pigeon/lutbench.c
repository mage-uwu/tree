// lutbench.c — exactness check + speed of LUT matvec vs int8 dot, random ternary weights.
#include <stdio.h>
#include <time.h>
#include "lut.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    lut_init();
    int sizes[][2] = {{128, 128}, {130, 127}, {768, 768}, {2048, 2048}, {4096, 4096}, {11008, 4096}};
    srand(1);
    for (int si = 0; si < 6; si++) {
        int R = sizes[si][0], C = sizes[si][1];
        int8_t *W = malloc((size_t)R * C), *x = malloc(C);
        for (size_t i = 0; i < (size_t)R * C; i++) { int r = rand() % 10; W[i] = r < 3 ? -1 : r < 6 ? 1 : 0; }
        LutW L = lut_pack(W, R, C);
        uint8_t *tab = aligned_alloc(64, ((size_t)L.G * 64 + 63) / 64 * 64);
        int32_t *y1 = malloc(R * 4), *y2 = malloc(R * 4), *y3 = malloc(R * 4), *scr = aligned_alloc(64, (size_t)L.NRB * 32 * 4);
        // exactness, including extreme activations
        long bad = 0;
        for (int trial = 0; trial < 20; trial++) {
            for (int i = 0; i < C; i++) x[i] = trial == 0 ? 127 : trial == 1 ? -127 : (int8_t)(rand() % 255 - 127);
            for (int r = 0; r < R; r++) y1[r] = dot8(W + (size_t)r * C, x, C);
            lut_tables(tab, x, C); lut_mv(y2, &L, tab, scr);
            for (int r = 0; r < R; r++) y3[r] = dot_tern(W + (size_t)r * C, x, C);
            for (int r = 0; r < R; r++) bad += (y1[r] != y2[r]) + (y1[r] != y3[r]);
        }
        int it = R * C > 4000000 ? 50 : R * C > 500000 ? 300 : 20000;
        volatile int32_t sink = 0;
        double t0 = now();
        for (int i = 0; i < it; i++) { x[i % C] ^= 1; for (int r = 0; r < R; r++) y1[r] = dot8(W + (size_t)r * C, x, C); sink += y1[0]; }
        double td = (now() - t0) / it;
        t0 = now();
        for (int i = 0; i < it; i++) { x[i % C] ^= 1; lut_tables(tab, x, C); lut_mv(y2, &L, tab, scr); sink += y2[0]; }
        double tl = (now() - t0) / it;
        t0 = now();
        for (int i = 0; i < it; i++) { x[i % C] ^= 1; for (int r = 0; r < R; r++) y3[r] = dot_tern(W + (size_t)r * C, x, C); sink += y3[0]; }
        double ts = (now() - t0) / it;
        printf("%5d x %-5d mismatches %ld | naive-dot %8.1f us | AVX2 sign-dot %8.1f us | LUT %8.1f us | LUT vs AVX2 %.2fx\n",
               R, C, bad, td * 1e6, ts * 1e6, tl * 1e6, ts / tl);
    }
    return 0;
}
