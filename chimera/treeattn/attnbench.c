// attnbench.c — one head, one query against N cached keys/values.
// standard: fp32 K,V cache, score = q.k (hd MACs), out += p*v (hd MACs)
// tree:     byte codes, score = sum_s T[s][code] (S lookups), P[s][code] += p (S adds), then S*L*hd once
// compiled with -ffast-math so the standard path gets full SIMD.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    int hd = argc > 1 ? atoi(argv[1]) : 128, S = argc > 2 ? atoi(argv[2]) : 16, D = argc > 3 ? atoi(argv[3]) : 6, L = 1 << D;
    int Ns[] = {512, 2048, 8192, 32768};
    float *q = malloc(hd * 4), *o = malloc(hd * 4), *att = malloc(32768 * 4);
    float *Ck = malloc((size_t)S * L * hd * 4), *Cv = malloc((size_t)S * L * hd * 4), *T = malloc(S * L * 4), *P = malloc(S * L * 4);
    for (int i = 0; i < hd; i++) q[i] = sinf(i);
    for (size_t i = 0; i < (size_t)S * L * hd; i++) { Ck[i] = cosf(i * 0.37f) * 0.1f; Cv[i] = sinf(i * 0.11f) * 0.1f; }
    printf("hd=%d S=%d L=%d | cache per key: standard fp16 %d B, tree %d B (%.0fx smaller)\n", hd, S, L, 4 * hd, 2 * S, 4.0 * hd / (2 * S));
    for (int ni = 0; ni < 4; ni++) {
        int N = Ns[ni];
        float *K = malloc((size_t)N * hd * 4), *V = malloc((size_t)N * hd * 4);
        uint8_t *KC = malloc((size_t)N * S), *VC = malloc((size_t)N * S);
        for (size_t i = 0; i < (size_t)N * hd; i++) { K[i] = cosf(i * 0.01f); V[i] = sinf(i * 0.013f); }
        for (size_t i = 0; i < (size_t)N * S; i++) { KC[i] = (uint8_t)((i * 2654435761u) >> 7) % L; VC[i] = (uint8_t)((i * 40503u) >> 3) % L; }
        int it = 4000000 / N + 3; volatile float sink = 0;
        double t0 = now();
        for (int r = 0; r < it; r++) {
            float mx = -1e30f;
            for (int j = 0; j < N; j++) { const float *k = K + (size_t)j * hd; float s = 0; for (int i = 0; i < hd; i++) s += q[i] * k[i]; att[j] = s; mx = s > mx ? s : mx; }
            float z = 0; for (int j = 0; j < N; j++) { att[j] = expf(att[j] - mx); z += att[j]; }
            memset(o, 0, hd * 4);
            for (int j = 0; j < N; j++) { const float *v = V + (size_t)j * hd; float p = att[j] / z; for (int i = 0; i < hd; i++) o[i] += p * v[i]; }
            sink += o[r % hd]; q[0] += 1e-6f;
        }
        double ts = (now() - t0) / it;
        t0 = now();
        for (int r = 0; r < it; r++) {
            for (int i = 0; i < S * L; i++) { const float *c = Ck + (size_t)i * hd; float s = 0; for (int e = 0; e < hd; e++) s += q[e] * c[e]; T[i] = s; }
            float mx = -1e30f;
            for (int j = 0; j < N; j++) { const uint8_t *cd = KC + (size_t)j * S; float s = 0; for (int t = 0; t < S; t++) s += T[t * L + cd[t]]; att[j] = s; mx = s > mx ? s : mx; }
            float z = 0; for (int j = 0; j < N; j++) { att[j] = expf(att[j] - mx); z += att[j]; }
            memset(P, 0, S * L * 4);
            for (int j = 0; j < N; j++) { const uint8_t *cd = VC + (size_t)j * S; float p = att[j]; for (int t = 0; t < S; t++) P[t * L + cd[t]] += p; }
            memset(o, 0, hd * 4);
            for (int i = 0; i < S * L; i++) { float p = P[i] / z; const float *c = Cv + (size_t)i * hd; for (int e = 0; e < hd; e++) o[e] += p * c[e]; }
            sink += o[r % hd]; q[0] += 1e-6f;
        }
        double tt = (now() - t0) / it;
        printf("  N=%6d keys: standard %8.1f us | tree %8.1f us | speedup %.2fx | cache standard %6.1f MB vs tree %5.2f MB\n",
               N, ts * 1e6, tt * 1e6, ts / tt, N * 4.0 * hd / 1e6, N * 2.0 * S / 1e6);
        free(K); free(V); free(KC); free(VC);
    }
    return 0;
}
