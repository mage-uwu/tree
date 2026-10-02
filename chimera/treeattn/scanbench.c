// scanbench.c — scoring stage only: one query against N cached keys, one head.
//   standard : AVX2+FMA float dot per key (K cache fp32, hd floats per key)
//   fast-scan: build S x 16 uint8 leaf table (S*16 dots), then pshufb over nibble codes (S/2 bytes per key)
// Usage: ./scanbench hd S
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static inline float dot_ps(const float *a, const float *b, int n) {
    __m256 s = _mm256_setzero_ps(); int i = 0;
    for (; i + 8 <= n; i += 8) s = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s);
    __m128 t = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    t = _mm_hadd_ps(t, t); t = _mm_hadd_ps(t, t); return _mm_cvtss_f32(t);
}
// leaf table for one query: T[S*16] = Ct^T q (column-wise axpy over the transposed codebook), then
// 16-bit fixed point: entry = (T - min_tree) / delta with delta chosen so a key's total fits uint16.
// Stored as two byte planes (lo, hi), 16 bytes per tree each, for pshufb. Returns delta.
static inline float build_table(uint8_t *tlo, uint8_t *thi, float *T, const float *Ct, const float *q, int hd, int S, float scale) {
    int n = S * 16;
    for (int i = 0; i < n; i += 8) _mm256_storeu_ps(T + i, _mm256_setzero_ps());
    for (int e = 0; e < hd; e++) { __m256 qe = _mm256_set1_ps(q[e] * scale); const float *col = Ct + (size_t)e * n;
        for (int i = 0; i < n; i += 8) _mm256_storeu_ps(T + i, _mm256_fmadd_ps(qe, _mm256_loadu_ps(col + i), _mm256_loadu_ps(T + i))); }
    float mins[64], rsum = 0;
    for (int s = 0; s < S; s++) {
        __m256 a = _mm256_loadu_ps(T + s * 16), b = _mm256_loadu_ps(T + s * 16 + 8);
        __m256 mn = _mm256_min_ps(a, b), mx = _mm256_max_ps(a, b);
        __m128 mn4 = _mm_min_ps(_mm256_castps256_ps128(mn), _mm256_extractf128_ps(mn, 1)), mx4 = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
        mn4 = _mm_min_ps(mn4, _mm_movehl_ps(mn4, mn4)); mn4 = _mm_min_ss(mn4, _mm_shuffle_ps(mn4, mn4, 1));
        mx4 = _mm_max_ps(mx4, _mm_movehl_ps(mx4, mx4)); mx4 = _mm_max_ss(mx4, _mm_shuffle_ps(mx4, mx4, 1));
        mins[s] = _mm_cvtss_f32(mn4); rsum += _mm_cvtss_f32(mx4) - mins[s];
    }
    float delta = rsum > 0 ? rsum / 65000.0f : 1.0f; __m256 idl = _mm256_set1_ps(1.0f / delta);
    const __m128i lom = _mm_set1_epi16(0x00FF);
    for (int s = 0; s < S; s++) { __m256 mn = _mm256_set1_ps(mins[s]); const float *t = T + s * 16;
        __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(t), mn), idl)), i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(t + 8), mn), idl));
        __m128i u0 = _mm_packus_epi32(_mm256_castsi256_si128(i0), _mm256_extracti128_si256(i0, 1));     // 8 x uint16
        __m128i u1 = _mm_packus_epi32(_mm256_castsi256_si128(i1), _mm256_extracti128_si256(i1, 1));
        _mm_storeu_si128((__m128i *)(tlo + s * 16), _mm_packus_epi16(_mm_and_si128(u0, lom), _mm_and_si128(u1, lom)));
        _mm_storeu_si128((__m128i *)(thi + s * 16), _mm_packus_epi16(_mm_srli_epi16(u0, 8), _mm_srli_epi16(u1, 8))); }
    return delta;
}
int main(int argc, char **argv) {
    int hd = argc > 1 ? atoi(argv[1]) : 32, S = argc > 2 ? atoi(argv[2]) : 24;
    int Ns[] = {128, 512, 2048, 8192, 32768};
    float *q = aligned_alloc(64, hd * 4), *C = aligned_alloc(64, (size_t)S * 16 * hd * 4), *T = malloc(S * 16 * 4);
    for (int i = 0; i < hd; i++) q[i] = sinf(i * 1.3f);
    for (size_t i = 0; i < (size_t)S * 16 * hd; i++) C[i] = cosf(i * 0.37f) * 0.2f;
    printf("hd=%d, %d trees x 16 leaves (%d bits/key) | key cache: fp16 %d B/key vs codes %d B/key (%.0fx smaller)\n", hd, S, 4 * S, 2 * hd, S / 2, 2.0 * hd / (S / 2));
    for (int ni = 0; ni < 5; ni++) {
        int N = Ns[ni], nb = (N + 31) / 32;
        float *K = aligned_alloc(64, (size_t)N * hd * 4), *att = aligned_alloc(64, (size_t)(N + 32) * 4);
        uint8_t *KF = aligned_alloc(64, (size_t)nb * (S / 2) * 32); uint16_t *sc = aligned_alloc(64, (size_t)nb * 32 * 2);
        for (size_t i = 0; i < (size_t)N * hd; i++) K[i] = cosf(i * 0.011f);
        for (size_t i = 0; i < (size_t)nb * (S / 2) * 32; i++) KF[i] = (uint8_t)((i * 2654435761u) >> 11);
        int it = 3000000 / N + 20; volatile float sink = 0; double t0 = now();
        for (int r = 0; r < it; r++) {
            float mx = -1e30f;
            for (int j = 0; j < N; j++) { float s = dot_ps(q, K + (size_t)j * hd, hd); att[j] = s; if (s > mx) mx = s; }
            sink += mx; q[0] += 1e-6f;
        }
        double ts = (now() - t0) / it;
        t0 = now();
        for (int r = 0; r < it; r++) {
            uint8_t tlo[64 * 16] __attribute__((aligned(32))), thi[64 * 16] __attribute__((aligned(32)));
            build_table(tlo, thi, T, C, q, hd, S, 1.0f);
            const __m256i nib = _mm256_set1_epi8(0x0F), zero = _mm256_setzero_si256(); __m256i vmax = zero;
            for (int b = 0; b < nb; b++) {
                const uint8_t *blk = KF + (size_t)b * (S / 2) * 32; __m256i alo = zero, ahi = zero;
                for (int p = 0; p < S / 2; p++) {
                    __m256i cc = _mm256_load_si256((const __m256i *)(blk + p * 32));
                    __m256i ia = _mm256_and_si256(cc, nib), ib = _mm256_and_si256(_mm256_srli_epi16(cc, 4), nib);
                    __m256i la = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)(tlo + 32 * p))), ia);
                    __m256i ha = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)(thi + 32 * p))), ia);
                    __m256i lb = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)(tlo + 32 * p + 16))), ib);
                    __m256i hb = _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_load_si128((const __m128i *)(thi + 32 * p + 16))), ib);
                    alo = _mm256_add_epi16(alo, _mm256_add_epi16(_mm256_unpacklo_epi8(la, ha), _mm256_unpacklo_epi8(lb, hb)));
                    ahi = _mm256_add_epi16(ahi, _mm256_add_epi16(_mm256_unpackhi_epi8(la, ha), _mm256_unpackhi_epi8(lb, hb)));
                }
                __m256i s0 = _mm256_permute2x128_si256(alo, ahi, 0x20), s1 = _mm256_permute2x128_si256(alo, ahi, 0x31);
                _mm256_store_si256((__m256i *)(sc + b * 32), s0); _mm256_store_si256((__m256i *)(sc + b * 32 + 16), s1);
                vmax = _mm256_max_epu16(vmax, _mm256_max_epu16(s0, s1));
            }
            sink += (float)_mm256_extract_epi16(vmax, 0); q[0] += 1e-6f;
        }
        double tf = (now() - t0) / it;
        printf("  N=%6d keys: standard %8.2f us (%.2f ns/key) | fast-scan %7.2f us (%.2f ns/key incl. table) | %.1fx\n", N, ts * 1e6, ts / N * 1e9, tf * 1e6, tf / N * 1e9, ts / tf);
        free(K); free(att); free(KF); free(sc);
    }
    return 0;
}
