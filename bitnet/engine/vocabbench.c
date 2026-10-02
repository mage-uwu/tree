// Tree output layer vs dense, one token, V=128256, d=2560 (random data; speed only).
//   dense : logits = E h, E in f16 (what bitnet.cpp does for the tied head)
//   tree  : (1) leaf tables: S trees x 16 leaves, each entry = h . leaf_vector (S*16*d MACs), quantized to 16 bit
//           (2) fast-scan: every token has S 4-bit codes (S/2 bytes); pshufb looks up 32 tokens per instruction,
//               16-bit entries as two byte planes accumulated in uint16 (lo plane, hi plane)
//           (3) top-N by histogram threshold on the approximate scores
//           (4) exact f16 dot for the N candidates
// Usage: vocabbench [S N]
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define V 128256
#define D 2560
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rs = 88172645463325252ull;
static inline uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 32); }

static inline float dot_f16(const uint16_t *w, const float *x) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    for (int i = 0; i < D; i += 16) {
        a0 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i))), _mm256_loadu_ps(x + i), a0);
        a1 = _mm256_fmadd_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(w + i + 8))), _mm256_loadu_ps(x + i + 8), a1);
    }
    a0 = _mm256_add_ps(a0, a1);
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

int main(int argc, char **argv) {
    int S = argc > 1 ? atoi(argv[1]) : 128, N = argc > 2 ? atoi(argv[2]) : 8192;
    int P = S / 2, NB = (V + 31) / 32;
    uint16_t *E = aligned_alloc(64, (size_t)V * D * 2);
    for (size_t i = 0; i < (size_t)V * D; i++) E[i] = _cvtss_sh((float)((int)(rnd() % 2001) - 1000) / 30000.f, 0);
    float *leaf = aligned_alloc(64, (size_t)S * 16 * D * 4);
    for (size_t i = 0; i < (size_t)S * 16 * D; i++) leaf[i] = (float)((int)(rnd() % 2001) - 1000) / 30000.f;
    uint8_t *codes = aligned_alloc(64, (size_t)NB * P * 32);      // [block][pair][32 tokens]
    for (size_t i = 0; i < (size_t)NB * P * 32; i++) codes[i] = (uint8_t)rnd();
    float h[D] __attribute__((aligned(64)));
    for (int i = 0; i < D; i++) h[i] = (float)((int)(rnd() % 2001) - 1000) / 1000.f;
    float *logit = aligned_alloc(64, (size_t)V * 4);
    uint8_t *tlo = aligned_alloc(64, (size_t)S * 16), *thi = aligned_alloc(64, (size_t)S * 16);
    int32_t *sc = aligned_alloc(64, (size_t)NB * 32 * 4);
    int *cand = malloc(sizeof(int) * V);
    float tab[16];

    int reps = 10;
    double t0 = now();
    for (int r = 0; r < reps; r++) for (int v = 0; v < V; v++) logit[v] = dot_f16(E + (size_t)v * D, h);
    double dense = (now() - t0) / reps * 1e3;

    double tt = 0, ts = 0, tk = 0, te = 0;
    for (int r = 0; r < reps; r++) {
        double t1 = now();
        // (1) tables: 16-bit unsigned per tree (offset by the tree's min), split into byte planes
        for (int s = 0; s < S; s++) {
            float mn = 1e30f, mx = -1e30f;
            for (int l = 0; l < 16; l++) {
                const float *c = leaf + ((size_t)s * 16 + l) * D;
                __m256 a = _mm256_setzero_ps();
                for (int i = 0; i < D; i += 8) a = _mm256_fmadd_ps(_mm256_loadu_ps(c + i), _mm256_loadu_ps(h + i), a);
                __m128 q = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1)); q = _mm_hadd_ps(q, q); q = _mm_hadd_ps(q, q);
                tab[l] = _mm_cvtss_f32(q); mn = fminf(mn, tab[l]); mx = fmaxf(mx, tab[l]);
            }
            float st = 255.f / fmaxf(mx - mn, 1e-9f);       // shared step would be per-tree in a real engine
            for (int l = 0; l < 16; l++) { int q = (int)lrintf((tab[l] - mn) * st * 256.f); q = q > 65535 ? 65535 : q; tlo[s * 16 + l] = q & 255; thi[s * 16 + l] = q >> 8; }
        }
        double t2 = now(); tt += t2 - t1;
        // (2) fast-scan
        const __m256i m4 = _mm256_set1_epi8(15);
        for (int b = 0; b < NB; b++) {
            __m256i slo0 = _mm256_setzero_si256(), slo1 = slo0, shi0 = slo0, shi1 = slo0;
            const uint8_t *cb = codes + (size_t)b * P * 32;
            for (int p = 0; p < P; p++) {
                __m256i c = _mm256_loadu_si256((const __m256i *)(cb + p * 32));
                __m256i i0 = _mm256_and_si256(c, m4), i1 = _mm256_and_si256(_mm256_srli_epi16(c, 4), m4);
                __m256i L0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(tlo + 32 * p)));
                __m256i L1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(tlo + 32 * p + 16)));
                __m256i H0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(thi + 32 * p)));
                __m256i H1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(thi + 32 * p + 16)));
                __m256i lo = _mm256_add_epi8(_mm256_setzero_si256(), _mm256_shuffle_epi8(L0, i0));
                __m256i lo2 = _mm256_shuffle_epi8(L1, i1);
                __m256i hi = _mm256_shuffle_epi8(H0, i0), hi2 = _mm256_shuffle_epi8(H1, i1);
                // widen u8 -> u16 by interleaving with zero (even / odd tokens)
                slo0 = _mm256_add_epi16(slo0, _mm256_add_epi16(_mm256_and_si256(lo, _mm256_set1_epi16(255)), _mm256_and_si256(lo2, _mm256_set1_epi16(255))));
                slo1 = _mm256_add_epi16(slo1, _mm256_add_epi16(_mm256_srli_epi16(lo, 8), _mm256_srli_epi16(lo2, 8)));
                shi0 = _mm256_add_epi16(shi0, _mm256_add_epi16(_mm256_and_si256(hi, _mm256_set1_epi16(255)), _mm256_and_si256(hi2, _mm256_set1_epi16(255))));
                shi1 = _mm256_add_epi16(shi1, _mm256_add_epi16(_mm256_srli_epi16(hi, 8), _mm256_srli_epi16(hi2, 8)));
            }
            // score = 256*hi + lo, written as int32 (even tokens then odd tokens; order is irrelevant for top-N here)
            int32_t *o = sc + (size_t)b * 32;
            _mm256_storeu_si256((__m256i *)o, _mm256_add_epi32(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(shi0)), 8), _mm256_cvtepu16_epi32(_mm256_castsi256_si128(slo0))));
            _mm256_storeu_si256((__m256i *)(o + 8), _mm256_add_epi32(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(shi0, 1)), 8), _mm256_cvtepu16_epi32(_mm256_extracti128_si256(slo0, 1))));
            _mm256_storeu_si256((__m256i *)(o + 16), _mm256_add_epi32(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(shi1)), 8), _mm256_cvtepu16_epi32(_mm256_castsi256_si128(slo1))));
            _mm256_storeu_si256((__m256i *)(o + 24), _mm256_add_epi32(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(shi1, 1)), 8), _mm256_cvtepu16_epi32(_mm256_extracti128_si256(slo1, 1))));
        }
        double t3 = now(); ts += t3 - t2;
        // (3) top-N by histogram threshold
        int32_t mx = 0; for (int v = 0; v < V; v++) mx = sc[v] > mx ? sc[v] : mx;
        int hist[4096] = {0}; float bs = 4095.f / (float)(mx + 1);
        for (int v = 0; v < V; v++) hist[(int)((float)sc[v] * bs)]++;
        int c = 0, thr = 4095; while (thr > 0 && c + hist[thr] <= N) c += hist[thr--];
        int n = 0; for (int v = 0; v < V && n < N; v++) if ((int)((float)sc[v] * bs) > thr) cand[n++] = v;
        double t4 = now(); tk += t4 - t3;
        // (4) exact rescoring of candidates (prefetch next rows)
        for (int i = 0; i < n; i++) {
            if (i + 4 < n) { const char *pn = (const char *)(E + (size_t)cand[i + 4] * D); for (int q = 0; q < D * 2; q += 64) _mm_prefetch(pn + q, _MM_HINT_T0); }
            logit[cand[i]] = dot_f16(E + (size_t)cand[i] * D, h);
        }
        te += now() - t4;
    }
    double tree = (tt + ts + tk + te) / reps * 1e3;
    printf("V=%d d=%d: dense f16 head %.2f ms | tree S=%d (%d B/token) N=%d: %.2f ms [tables %.2f, scan %.2f, top-N %.2f, exact %.2f] -> %.1fx\n",
           V, D, dense, S, S / 2, N, tree, tt / reps * 1e3, ts / reps * 1e3, tk / reps * 1e3, te / reps * 1e3, dense / tree);
    return 0;
}
