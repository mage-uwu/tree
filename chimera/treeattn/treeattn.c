// treeattn.c — BitNet b1.58 attention, inferenced either as standard softmax attention over a
// float KV cache (S=0) or as boosted-tree attention over a byte-code KV cache (S>0):
//
//   write key  : r = rope(k);  for s<S: leaf_s = tree_s(r) (D oblique splits); r -= c_s[leaf_s]
//                cache stores S bytes per key (and S bytes per value if values are converted)
//   per query  : T[s][l] = q . c^k_s[l] / sqrt(hd)          (S*L small dots, once per query)
//   per key    : score = sum_s T[s][code_s]                  (S byte lookups + adds, no dot product)
//   values     : P[s][l] += p_j for code^v_s(j);  o = sum_s,l P[s][l] c^v_s[l]
//
//   ./treeattn model.tre dump  <toks.bin> <logits.bin>
//   ./treeattn model.tre bench <n_tokens>
//   ./treeattn model.tre gen   <n> [temp] [seed] [prompt]
#include <math.h>
#include <x86intrin.h>
#include <stdio.h>
#include <time.h>
#include "../attn/lut.h"

#define MAXT 8192
#define NBLK (MAXT / 32)
typedef struct { float s; int rows, cols; int8_t *w; LutW L; } Tern;
typedef struct { float *w, *b, *c; } Trees;
// axis mode: keys are rotated once per head (k' = G k), then every node compares ONE coordinate with a threshold                 // [H][S][L-1][hd], [H][S][L-1], [H][S][L][hd]
typedef struct { float *nw, *sub, *n2, *tw, *tb, *tc, *tP, *tQt, *sP, *sQ, *sW; int8_t *P8, *Q8, *W8; Tern q, k, v, o, up, down; Trees kt, vt;
                 float *K, *V, *aW, *aC, *G, *GiT, *thr; int16_t *feat; int8_t *Wt8, *C8, *Ct8; uint8_t *KC, *VC, *KF; } Layer;
typedef struct { int vocab, d, H, hd, layers, S, D, L, values, mlp, tmD, tmL, tmr, q8, axis; float *emb, *norm, *head; Layer *ly; char chars[256]; } Model;

static FILE *fh;
static void rd(void *p, size_t n) { if (fread(p, 1, n, fh) != n) { fprintf(stderr, "short read\n"); exit(1); } }
static float *rdf(size_t n) { float *p = malloc(n * 4); rd(p, n * 4); return p; }
static Tern rdt(int rows, int cols) {
    Tern t; t.rows = rows; t.cols = cols; size_t n = (size_t)rows * cols, nb = (n + 3) / 4;
    rd(&t.s, 4); uint8_t *pk = malloc(nb); rd(pk, nb);
    t.w = malloc(n); for (size_t i = 0; i < n; i++) t.w[i] = (int8_t)((pk[i >> 2] >> (2 * (i & 3))) & 3) - 1;
    free(pk); t.L = lut_pack(t.w, rows, cols); return t;
}
static Trees rdtrees(Model *m) {
    Trees t; size_t hs = (size_t)m->H * m->S;
    t.w = rdf(hs * (m->L - 1) * m->hd); t.b = rdf(hs * (m->L - 1)); t.c = rdf(hs * m->L * m->hd); return t;
}
static void load(Model *m, const char *path) {
    fh = fopen(path, "rb"); if (!fh) { perror(path); exit(1); }
    char mg[4]; int h[13]; rd(mg, 4); rd(h, 52);
    if (memcmp(mg, "TRE1", 4) || h[0] != 5) { fprintf(stderr, "bad file\n"); exit(1); }
    m->q8 = h[11]; m->axis = h[12];
    m->mlp = h[8]; m->tmD = h[9]; m->tmr = h[10]; m->tmL = h[9] >= 0 ? 1 << h[9] : 0;      // tmD = -1: dense MLP
    m->vocab = h[1]; m->d = h[2]; m->H = h[3]; m->layers = h[4]; m->S = h[5]; m->D = h[6]; m->values = h[7];
    m->hd = m->d / m->H; m->L = 1 << m->D;
    int d = m->d; m->emb = rdf((size_t)m->vocab * d);
    m->ly = calloc(m->layers, sizeof(Layer));
    for (int l = 0; l < m->layers; l++) {
        Layer *y = &m->ly[l];
        y->nw = rdf(d); y->q = rdt(d, d); y->k = rdt(d, d); y->v = rdt(d, d); y->o = rdt(d, d); y->sub = rdf(d);
        if (m->S) { y->kt = rdtrees(m);
            if (m->axis) { int hd2 = m->hd; y->G = rdf((size_t)m->H * hd2 * hd2); float *Gi = rdf((size_t)m->H * hd2 * hd2);
                y->GiT = malloc((size_t)m->H * hd2 * hd2 * 4);
                for (int hh = 0; hh < m->H; hh++) for (int e = 0; e < hd2; e++) for (int dd = 0; dd < hd2; dd++)
                    y->GiT[((size_t)hh * hd2 + e) * hd2 + dd] = Gi[((size_t)hh * hd2 + dd) * hd2 + e];
                free(Gi);
                size_t nn = (size_t)m->H * m->S * (m->L - 1); y->feat = malloc(nn * 2); y->thr = malloc(nn * 4);
                for (size_t i = 0; i < nn; i++) { const float *w = y->kt.w + i * hd2; int f = -1;
                    for (int e = 0; e < hd2; e++) if (w[e] != 0) f = e;
                    y->feat[i] = (int16_t)(f < 0 ? 0 : f); y->thr[i] = f < 0 ? INFINITY : y->kt.b[i]; } }
            if (m->values) y->vt = rdtrees(m); }
        if (m->mlp) { y->n2 = rdf(d); y->up = rdt(m->mlp * d, d); y->down = rdt(d, m->mlp * d); }
        if (m->mlp && m->tmL) {                          // tree-MLP: router, leaf constants, leaf low-rank maps
            int L = m->tmL, r = m->tmr;
            size_t nn = (size_t)(L > 1 ? L - 1 : 1);
            if (m->q8) { y->W8 = aligned_alloc(64, (nn * d + 63) / 64 * 64); rd(y->W8, nn * d); y->sW = rdf(nn); }
            else y->tw = rdf(nn * d);
            y->tb = rdf(nn); y->tc = rdf((size_t)L * d);
            if (r && m->q8) {                            // int8 leaf maps + per-row scales
                size_t n = (size_t)L * r * d;
                y->P8 = aligned_alloc(64, (n + 63) / 64 * 64); rd(y->P8, n); y->sP = rdf((size_t)L * r);
                y->Q8 = aligned_alloc(64, (n + 63) / 64 * 64); rd(y->Q8, n); y->sQ = rdf((size_t)L * r);
            } else if (r) { y->tP = rdf((size_t)L * r * d); float *Q = rdf((size_t)L * d * r);
                y->tQt = malloc((size_t)L * r * d * 4);
                for (int l2 = 0; l2 < L; l2++) for (int j = 0; j < r; j++) for (int i = 0; i < d; i++)
                    y->tQt[((size_t)l2 * r + j) * d + i] = Q[((size_t)l2 * d + i) * r + j];
                free(Q); }
        }
        size_t ht = (size_t)m->H * MAXT;
        if (m->S) y->KC = malloc(ht * m->S); else y->K = malloc(ht * m->hd * 4);
        if (m->S && m->D == 4 && !(m->S & 1)) y->KF = aligned_alloc(64, (size_t)m->H * NBLK * (m->S / 2) * 32);
        if (y->KF) memset(y->KF, 0, (size_t)m->H * NBLK * (m->S / 2) * 32);
        if (y->KF) {       // int8 tree tables (each split vector / leaf value has its own absmax scale), laid out for SIMD
            int hd2 = m->hd, n16 = m->S * 16; size_t HS = (size_t)m->H * m->S;
            y->Wt8 = aligned_alloc(64, HS * hd2 * 16); memset(y->Wt8, 0, HS * hd2 * 16); y->aW = calloc(HS * 16, 4);
            y->C8 = aligned_alloc(64, (HS * 16 * hd2 + 63) / 64 * 64); y->aC = calloc(HS * 16, 4);
            y->Ct8 = aligned_alloc(64, HS * hd2 * 16); (void)n16;      // leaf values transposed per tree: [head*tree][dim][16 leaves]
            for (size_t hs = 0; hs < HS; hs++) {
                for (int nd = 0; nd < 15; nd++) { const float *w = y->kt.w + (hs * 15 + nd) * hd2; float mx = 1e-12f;
                    for (int e = 0; e < hd2; e++) if (fabsf(w[e]) > mx) mx = fabsf(w[e]);
                    float sc = mx / 127.0f; y->aW[hs * 16 + nd] = sc;
                    for (int e = 0; e < hd2; e++) y->Wt8[(hs * hd2 + e) * 16 + nd] = (int8_t)lrintf(w[e] / sc); }
                for (int lf = 0; lf < 16; lf++) { const float *cc = y->kt.c + (hs * 16 + lf) * hd2; float mx = 1e-12f;
                    for (int e = 0; e < hd2; e++) if (fabsf(cc[e]) > mx) mx = fabsf(cc[e]);
                    float sc = mx / 127.0f; y->aC[hs * 16 + lf] = sc; int hh = (int)(hs / m->S), s = (int)(hs % m->S);
                    for (int e = 0; e < hd2; e++) { int8_t v8 = (int8_t)lrintf(cc[e] / sc);
                        y->C8[(hs * 16 + lf) * hd2 + e] = v8; y->Ct8[(hs * hd2 + e) * 16 + lf] = v8; (void)hh; (void)s; } }
            }
        }
        if (m->S && m->values) y->VC = malloc(ht * m->S); else y->V = malloc(ht * m->hd * 4);
    }
    m->norm = rdf(d); m->head = rdf((size_t)m->vocab * d); rd(m->chars, m->vocab); fclose(fh);
}

static void rmsnorm(float *o, const float *x, const float *w, int n) {      // AVX2; same per-element op order
    __m256 s = _mm256_setzero_ps(); int i = 0;
    for (; i + 8 <= n; i += 8) { __m256 v = _mm256_loadu_ps(x + i); s = _mm256_fmadd_ps(v, v, s); }
    float t[8]; _mm256_storeu_ps(t, s); float ss = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
    for (; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / n + 1e-6f); __m256 vr = _mm256_set1_ps(r);
    for (i = 0; i + 8 <= n; i += 8) _mm256_storeu_ps(o + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + i), vr), _mm256_loadu_ps(w + i)));
    for (; i < n; i++) o[i] = x[i] * r * w[i];
}
static float quant8(int8_t *q, const float *x, int n) {                      // absmax int8, round-half-even (== rintf)
    __m256 mxv = _mm256_setzero_ps(), sgn = _mm256_set1_ps(-0.0f); int i = 0;
    for (; i + 8 <= n; i += 8) mxv = _mm256_max_ps(mxv, _mm256_andnot_ps(sgn, _mm256_loadu_ps(x + i)));
    float t[8]; _mm256_storeu_ps(t, mxv); float mx = 1e-5f; for (int j = 0; j < 8; j++) if (t[j] > mx) mx = t[j];
    for (; i < n; i++) { float a = fabsf(x[i]); if (a > mx) mx = a; }
    float s = 127.0f / mx; __m256 vs = _mm256_set1_ps(s);
    for (i = 0; i + 8 <= n; i += 8) {
        __m256i v = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + i), vs));
        __m128i w16 = _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        _mm_storel_epi64((__m128i *)(q + i), _mm_packs_epi16(w16, w16));
    }
    for (; i < n; i++) { float v = rintf(x[i] * s); q[i] = (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v); }
    return 1.0f / s;
}
static float ROPE_INV[256], ROPE_C[256], ROPE_S[256]; static int ROPE_T = -1, ROPE_HD = 0;
static void rope(float *x, int hd, int t) {                 // cos/sin computed once per token, shared by all heads
    int half = hd / 2;
    if (ROPE_HD != hd) { for (int i = 0; i < half; i++) ROPE_INV[i] = powf(10000.0f, -(float)i / half); ROPE_HD = hd; ROPE_T = -1; }
    if (ROPE_T != t) { for (int i = 0; i < half; i++) { float ang = (float)t * ROPE_INV[i]; ROPE_C[i] = cosf(ang); ROPE_S[i] = sinf(ang); } ROPE_T = t; }
    for (int i = 0; i < half; i++) {
        float a = x[i], b = x[i + half]; x[i] = a * ROPE_C[i] - b * ROPE_S[i]; x[i + half] = a * ROPE_S[i] + b * ROPE_C[i];
    }
}
static double MLP_T = 0; static long MLP_N = 0;
static unsigned long long TS[6]; static inline unsigned long long tsc(void) { return __rdtsc(); }
#define LAP(i) { unsigned long long _n = tsc(); TS[i] += _n - _t; _t = _n; }
static double ATT_T = 0; static long ATT_N = 0, ATT_KEYS = 0, ATT_KEPT = 0;
static int HALF = 1;            // key/value caches in fp16 (EXACT=1 -> fp32)
static int FAST = 1;            // fast-scan tree attention (depth-4 trees). EXACT=1 env -> reference float path
static float TAU = 16.0f;       // keys scoring more than TAU below the max get weight < e^-16: skipped

static double now(void);
static inline float dot_ps(const float *a, const float *b, int n) {       // AVX2+FMA float dot
    __m256 s = _mm256_setzero_ps(); int i = 0;
    for (; i + 8 <= n; i += 8) s = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s);
    __m128 t = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    t = _mm_hadd_ps(t, t); t = _mm_hadd_ps(t, t); float r = _mm_cvtss_f32(t);
    for (; i < n; i++) r += a[i] * b[i]; return r;
}
static inline void axpy_ps(float *y, const float *x, float a, int n) {
    __m256 va = _mm256_set1_ps(a); int i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_fmadd_ps(va, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
    for (; i < n; i++) y[i] += a * x[i];
}
static inline int32_t dot_i8(const int8_t *a, const int8_t *b, int n) {    // exact int8.int8 (|v|<=127)
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
static inline float dotf(const float *a, const float *b, int n) { float s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }

// AVX2 exp for x <= 0 (softmax weights): range reduction + degree-5 polynomial, ~1e-7 relative error
static inline __m256 exp256_ps(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f));
    __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x); x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f)); y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f)); y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_add_ps(_mm256_fmadd_ps(y, _mm256_mul_ps(x, x), x), _mm256_set1_ps(1.0f));
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}
static inline void store_half(void *dst, const float *x, int n) { for (int i = 0; i < n; i += 8) _mm_storeu_si128((__m128i *)((uint16_t *)dst + i), _mm256_cvtps_ph(_mm256_loadu_ps(x + i), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); }
static inline float dot_ph(const float *a, const uint16_t *b, int n) {
    __m256 s = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) s = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(b + i))), s);
    __m128 t = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    t = _mm_hadd_ps(t, t); t = _mm_hadd_ps(t, t); return _mm_cvtss_f32(t);
}
// o[hd] = sum_j w[j] * V[j]  (keys with w == 0 are skipped). Accumulates in registers, 64 dims per pass,
// so there is no load/store dependency on o between keys.
static void wsum(float *o, const float *V, const float *w, int n, int hd, long *kept) {
    long k = 0;
    if (HALF) { const uint16_t *Vh = (const uint16_t *)V;
        for (int c0 = 0; c0 < hd; c0 += 64) {
            int nr = (hd - c0 >= 64 ? 64 : hd - c0) / 8; __m256 a[8];
            for (int r = 0; r < nr; r++) a[r] = _mm256_setzero_ps();
            for (int j = 0; j < n; j++) {
                if (w[j] == 0.0f) continue;
                __m256 wj = _mm256_set1_ps(w[j]); const uint16_t *v = Vh + (size_t)j * hd + c0;
                for (int r = 0; r < nr; r++) a[r] = _mm256_fmadd_ps(wj, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(v + 8 * r))), a[r]);
                if (c0 == 0) k++;
            }
            for (int r = 0; r < nr; r++) _mm256_storeu_ps(o + c0 + 8 * r, a[r]);
        }
        *kept += k; return; }
    for (int c0 = 0; c0 < hd; c0 += 64) {
        int nr = (hd - c0 >= 64 ? 64 : hd - c0) / 8; __m256 a[8];
        for (int r = 0; r < nr; r++) a[r] = _mm256_setzero_ps();
        for (int j = 0; j < n; j++) {
            if (w[j] == 0.0f) continue;
            __m256 wj = _mm256_set1_ps(w[j]); const float *v = V + (size_t)j * hd + c0;
            for (int r = 0; r < nr; r++) a[r] = _mm256_fmadd_ps(wj, _mm256_loadu_ps(v + 8 * r), a[r]);
            if (c0 == 0) k++;
        }
        for (int r = 0; r < nr; r++) _mm256_storeu_ps(o + c0 + 8 * r, a[r]);
    }
    *kept += k;
}
// leaf table for one query: T[S*16] = Ct^T q (column-wise axpy over the transposed codebook), then
// 16-bit fixed point: entry = (T - min_tree) / delta with delta chosen so a key's total fits uint16.
// Stored as two byte planes (lo, hi), 16 bytes per tree each, for pshufb. Returns delta.
static inline float build_table(uint8_t *tlo, uint8_t *thi, float *T, const int8_t *Ct8, const float *sC, const float *q, int hd, int S, float scale) {
    int n = S * 16;
#define CV8(p) _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(p))
    for (int s = 0; s < S; s += 4) {                         // 4 independent trees per pass: 8 accumulators, no memory round trip
        int ns = S - s < 4 ? S - s : 4; __m256 a[8];
        for (int i = 0; i < 8; i++) a[i] = _mm256_setzero_ps();
        const int8_t *c0 = Ct8 + (size_t)s * hd * 16;
        for (int e = 0; e < hd; e++) { __m256 qe = _mm256_set1_ps(q[e]);
            for (int k = 0; k < ns; k++) { __m128i w = _mm_load_si128((const __m128i *)(c0 + ((size_t)k * hd + e) * 16));
                a[2 * k] = _mm256_fmadd_ps(qe, CV8(w), a[2 * k]); a[2 * k + 1] = _mm256_fmadd_ps(qe, CV8(_mm_srli_si128(w, 8)), a[2 * k + 1]); } }
        for (int k = 0; k < ns; k++) { _mm256_storeu_ps(T + (s + k) * 16, a[2 * k]); _mm256_storeu_ps(T + (s + k) * 16 + 8, a[2 * k + 1]); }
    }
    __m256 vsc = _mm256_set1_ps(scale);
    for (int i = 0; i < n; i += 8) _mm256_storeu_ps(T + i, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(T + i), _mm256_loadu_ps(sC + i)), vsc));
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
// encode one head-vector with the boosted trees of head h -> S leaf bytes
static void encode(const Model *m, const Trees *t, int h, const float *x, uint8_t *code) {
    int hd = m->hd, L = m->L; float r[256];
    memcpy(r, x, hd * 4);
    float tiny = 1e-5f * sqrtf(dotf(x, x, hd));
    for (int s = 0; s < m->S; s++) {
        size_t hs = (size_t)h * m->S + s;
        const float *W = t->w + hs * (L - 1) * hd, *B = t->b + hs * (L - 1);
        int node = 0;
        for (int l = 0; l < m->D; l++) node = 2 * node + 1 + (dot_ps(W + (size_t)node * hd, r, hd) - B[node] > 0);
        int leaf = node - (L - 1); code[s] = (uint8_t)leaf;
        const float *c = t->c + (hs * L + leaf) * hd;
        axpy_ps(r, c, -1.0f, hd);
        if (sqrtf(dot_ps(r, r, hd)) < tiny) memset(r, 0, hd * 4);         // snap float noise (same rule as torch)
    }
}

// depth-4 encoder: all 15 node projections of a tree in one small matvec over int8 split vectors
// (no branch-dependent loads), then walk; residual update from the int8 leaf value.
static void encode4(const Model *m, const Layer *y, int h, const float *x, uint8_t *code) {
    int hd = m->hd; float r[256] __attribute__((aligned(32))), a[16] __attribute__((aligned(32)));
    memcpy(r, x, hd * 4);
    float tiny = 1e-5f * sqrtf(dot_ps(x, x, hd));
    for (int s = 0; s < m->S; s++) {
        size_t hs = (size_t)h * m->S + s; const int8_t *W = y->Wt8 + hs * hd * 16; const float *B = y->kt.b + hs * 15, *sw = y->aW + hs * 16;
        __m256 p[8]; for (int i = 0; i < 8; i++) p[i] = _mm256_setzero_ps();
        for (int e = 0; e < hd; e += 4) for (int u = 0; u < 4; u++) {            // 4 independent chains
            __m256 re = _mm256_set1_ps(r[e + u]); __m128i wv = _mm_load_si128((const __m128i *)(W + (e + u) * 16));
            p[2 * u] = _mm256_fmadd_ps(re, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(wv)), p[2 * u]);
            p[2 * u + 1] = _mm256_fmadd_ps(re, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(wv, 8))), p[2 * u + 1]); }
        __m256 a0 = _mm256_add_ps(_mm256_add_ps(p[0], p[2]), _mm256_add_ps(p[4], p[6])), a1 = _mm256_add_ps(_mm256_add_ps(p[1], p[3]), _mm256_add_ps(p[5], p[7]));
        _mm256_store_ps(a, _mm256_mul_ps(a0, _mm256_loadu_ps(sw))); _mm256_store_ps(a + 8, _mm256_mul_ps(a1, _mm256_loadu_ps(sw + 8)));
        int node = 0; for (int l = 0; l < 4; l++) node = 2 * node + 1 + (a[node] - B[node] > 0);
        int leaf = node - 15; code[s] = (uint8_t)leaf;
        axpy_i8(r, y->C8 + (hs * 16 + leaf) * hd, -y->aC[hs * 16 + leaf], hd);
        if (sqrtf(dot_ps(r, r, hd)) < tiny) memset(r, 0, hd * 4);
    }
}

// axis encoder: no dot products. per tree: 4 compares of one coordinate each, then subtract the int8 leaf value
static void encode_axis(const Model *m, const Layer *y, int h, const float *kr, uint8_t *code) {
    int hd = m->hd; float r[256] __attribute__((aligned(32)));
    memcpy(r, kr, hd * 4);
    float tiny = 1e-5f * sqrtf(dot_ps(kr, kr, hd));
    for (int s = 0; s < m->S; s++) {
        size_t hs = (size_t)h * m->S + s; const int16_t *F = y->feat + hs * 15; const float *T = y->thr + hs * 15;
        int node = 0; for (int l = 0; l < 4; l++) node = 2 * node + 1 + (r[F[node]] > T[node]);
        int leaf = node - 15; code[s] = (uint8_t)leaf;
        axpy_i8(r, y->C8 + (hs * 16 + leaf) * hd, -y->aC[hs * 16 + leaf], hd);
        if (sqrtf(dot_ps(r, r, hd)) < tiny) memset(r, 0, hd * 4);
    }
}

typedef struct { float *x, *u, *q, *k, *v, *o, *t1, *hid, *att, *T, *P, *logits; int8_t *xq; uint8_t *tab; int32_t *acc, *scr; uint16_t *sc16; } Buf;
static void bitlin(float *y, const Tern *t, Buf *B, float xinv) {
    lut_mv(B->acc, &t->L, B->tab, B->scr); float f = t->s * xinv;
    for (int r = 0; r < t->rows; r++) y[r] = (float)B->acc[r] * f;
}

static void forward(Model *m, Buf *B, int tok, int t) {
    int d = m->d, H = m->H, hd = m->hd, S = m->S, L = m->L;
    float isq = 1.0f / sqrtf((float)hd);
    memcpy(B->x, m->emb + (size_t)tok * d, d * 4);
    for (int l = 0; l < m->layers; l++) {
        Layer *y = &m->ly[l];
        rmsnorm(B->u, B->x, y->nw, d);
        float xi = quant8(B->xq, B->u, d); lut_tables(B->tab, B->xq, d);
        bitlin(B->q, &y->q, B, xi); bitlin(B->k, &y->k, B, xi); bitlin(B->v, &y->v, B, xi);
        for (int h = 0; h < H; h++) {
            float *q = B->q + h * hd, *k = B->k + h * hd, *v = B->v + h * hd, *o = B->o + h * hd;
            rope(q, hd, t); rope(k, hd, t);
            size_t base = (size_t)h * MAXT;
            // ---- write
            double ta0 = now(); unsigned long long _t = tsc();
            float kr[256] __attribute__((aligned(32))), qr[256] __attribute__((aligned(32)));
            if (S && m->axis) {                              // one rotation per key and per query
                const float *G = y->G + (size_t)h * hd * hd, *GT = y->GiT + (size_t)h * hd * hd;
                for (int e = 0; e < hd; e++) { kr[e] = dot_ps(G + (size_t)e * hd, k, hd); qr[e] = dot_ps(GT + (size_t)e * hd, q, hd); }
                k = kr; q = qr;
            }
            int n = t + 1, fast = FAST && S && y->KF && !m->values;
            if (S) {
                uint8_t *cd = y->KC + (base + t) * S;
                if (m->axis && y->Wt8 && FAST) encode_axis(m, y, h, k, cd);
                else if (y->Wt8 && FAST) encode4(m, y, h, k, cd); else encode(m, &y->kt, h, k, cd);
                if (y->KF) {                                 // nibble-packed, 32 keys per block, one 32-byte row per tree pair
                    uint8_t *blk = y->KF + ((size_t)h * NBLK + t / 32) * (S / 2) * 32;
                    for (int p = 0; p < S / 2; p++) blk[p * 32 + (t & 31)] = (uint8_t)(cd[2 * p] | (cd[2 * p + 1] << 4));
                }
            } else if (HALF) store_half((uint16_t *)y->K + (base + t) * hd, k, hd); else memcpy(y->K + (base + t) * hd, k, hd * 4);
            if (S && m->values) encode(m, &y->vt, h, v, y->VC + (base + t) * S);
            else if (HALF) store_half((uint16_t *)y->V + (base + t) * hd, v, hd); else memcpy(y->V + (base + t) * hd, v, hd * 4);
            memset(o, 0, hd * 4); LAP(0)
            if (fast) {
                // ---- per-query leaf table (S x 16), quantised to uint8 with one shared step
                uint8_t tlo[64 * 16] __attribute__((aligned(32))), thi[64 * 16] __attribute__((aligned(32)));
                float delta = build_table(tlo, thi, B->T, y->Ct8 + (size_t)h * hd * S * 16, y->aC + (size_t)h * S * 16, q, hd, S, isq);
                LAP(1)
                // ---- scan: 32 keys per pshufb, two trees per code byte
                const __m256i nib = _mm256_set1_epi8(0x0F), zero = _mm256_setzero_si256();
                int nb = (n + 31) / 32; __m256i vmax = zero;
                const uint8_t *kf = y->KF + (size_t)h * NBLK * (S / 2) * 32;
                for (int b = 0; b < nb; b++) {
                    const uint8_t *blk = kf + (size_t)b * (S / 2) * 32; __m256i alo = zero, ahi = zero;
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
                    __m256i s0 = _mm256_permute2x128_si256(alo, ahi, 0x20), s1v = _mm256_permute2x128_si256(alo, ahi, 0x31);
                    _mm256_storeu_si256((__m256i *)(B->sc16 + b * 32), s0); _mm256_storeu_si256((__m256i *)(B->sc16 + b * 32 + 16), s1v);
                    if (b == nb - 1) { for (int j = n; j < nb * 32; j++) B->sc16[j] = 0;
                        s0 = _mm256_loadu_si256((__m256i *)(B->sc16 + b * 32)); s1v = _mm256_loadu_si256((__m256i *)(B->sc16 + b * 32 + 16)); }
                    vmax = _mm256_max_epu16(vmax, _mm256_max_epu16(s0, s1v));
                }
                uint16_t tm16[16]; _mm256_storeu_si256((__m256i *)tm16, vmax); int max16 = 0; for (int i = 0; i < 16; i++) if (tm16[i] > max16) max16 = tm16[i];
                LAP(2)
                // ---- weights: 8 keys per exp; a key's value is read only if it is within TAU of the best
                const float *V = HALF ? (const float *)((const uint16_t *)y->V + base * hd) : y->V + base * hd; __m256 vz = _mm256_setzero_ps();
                const __m256 vdl = _mm256_set1_ps(delta), vcut = _mm256_set1_ps(-TAU); const __m256i vm16 = _mm256_set1_epi32(max16);
                for (int g = 0; g < n; g += 8) {
                    __m256 x = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(B->sc16 + g))), vm16)), vdl);
                    __m256 keep = _mm256_cmp_ps(x, vcut, _CMP_GE_OQ);
                    if (g + 8 > n) keep = _mm256_and_ps(keep, _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(n - g), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7))));
                    __m256 w = _mm256_and_ps(exp256_ps(x), keep); vz = _mm256_add_ps(vz, w); _mm256_storeu_ps(B->att + g, w);
                }
                float zz[8]; _mm256_storeu_ps(zz, vz); float z = zz[0] + zz[1] + zz[2] + zz[3] + zz[4] + zz[5] + zz[6] + zz[7];
                LAP(3) wsum(o, V, B->att, n, hd, &ATT_KEPT); LAP(4)
                float iz = 1.0f / z; for (int i = 0; i < hd; i++) o[i] *= iz;
            } else {
                // ---- reference paths: float scores for every key
                float mx = -1e30f;
                if (S) {
                    const float *C = y->kt.c + (size_t)h * S * L * hd;
                    for (int i = 0; i < S * L; i++) B->T[i] = dot_ps(q, C + (size_t)i * hd, hd) * isq;   // query leaf table
                    for (int j = 0; j < n; j++) {
                        const uint8_t *cd = y->KC + (base + j) * S; float sc = 0;
                        for (int s = 0; s < S; s++) sc += B->T[s * L + cd[s]];                   // boosted-tree eval
                        B->att[j] = sc; if (sc > mx) mx = sc;
                    }
                } else {
                    const float *K = y->K + base * hd; const uint16_t *Kh = (const uint16_t *)y->K + base * hd;
                    if (HALF) for (int j = 0; j < n; j++) { float sc = dot_ph(q, Kh + (size_t)j * hd, hd) * isq; B->att[j] = sc; if (sc > mx) mx = sc; }
                    else for (int j = 0; j < n; j++) { float sc = dot_ps(q, K + (size_t)j * hd, hd) * isq; B->att[j] = sc; if (sc > mx) mx = sc; }
                } LAP(2)
                if (S && m->values) {
                    float z = 0; for (int j = 0; j < n; j++) { B->att[j] = expf(B->att[j] - mx); z += B->att[j]; }
                    float iz = 1.0f / z;
                    memset(B->P, 0, (size_t)S * L * 4);
                    for (int j = 0; j < n; j++) { const uint8_t *cd = y->VC + (base + j) * S; float p = B->att[j];
                        for (int s = 0; s < S; s++) B->P[s * L + cd[s]] += p; }
                    const float *C = y->vt.c + (size_t)h * S * L * hd;
                    for (int i = 0; i < S * L; i++) { float p = B->P[i]; if (p != 0) axpy_ps(o, C + (size_t)i * hd, p * iz, hd); }
                } else {                                  // same TAU pruning as the fast path, so the comparison is fair
                    const float *V = HALF ? (const float *)((const uint16_t *)y->V + base * hd) : y->V + base * hd; __m256 vz = _mm256_setzero_ps();
                    const __m256 vmx = _mm256_set1_ps(mx), vcut = _mm256_set1_ps(-TAU);
                    for (int g = 0; g < n; g += 8) {
                        __m256 x = _mm256_sub_ps(_mm256_loadu_ps(B->att + g), vmx);
                        __m256 keep = _mm256_cmp_ps(x, vcut, _CMP_GE_OQ);
                        if (g + 8 > n) keep = _mm256_and_ps(keep, _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(n - g), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7))));
                        __m256 w = _mm256_and_ps(exp256_ps(x), keep); vz = _mm256_add_ps(vz, w); _mm256_storeu_ps(B->att + g, w);
                    }
                    float zz[8]; _mm256_storeu_ps(zz, vz); float z = zz[0] + zz[1] + zz[2] + zz[3] + zz[4] + zz[5] + zz[6] + zz[7];
                    LAP(3) wsum(o, V, B->att, n, hd, &ATT_KEPT); LAP(4)
                    float iz = 1.0f / z; for (int i = 0; i < hd; i++) o[i] *= iz;
                }
            }
            ATT_T += now() - ta0; ATT_N++; ATT_KEYS += n;
        }
        rmsnorm(B->t1, B->o, y->sub, d);
        xi = quant8(B->xq, B->t1, d); lut_tables(B->tab, B->xq, d); bitlin(B->o, &y->o, B, xi);
        for (int i = 0; i < d; i++) B->x[i] += B->o[i];
        double tm0 = now();
        if (m->mlp && m->tmL) {                         // tree-MLP: route, then leaf constant + low-rank map
            rmsnorm(B->u, B->x, y->n2, d);
            int node = 0; float ui = 0;
            if (m->q8) {                                 // int8 router on int8 activations
                ui = quant8(B->xq, B->u, d);
                for (int lv = 0; lv < m->tmD; lv++)
                    node = 2 * node + 1 + ((float)dot_i8(y->W8 + (size_t)node * d, B->xq, d) * y->sW[node] * ui - y->tb[node] > 0);
            } else for (int lv = 0; lv < m->tmD; lv++) node = 2 * node + 1 + (dot_ps(y->tw + (size_t)node * d, B->u, d) - y->tb[node] > 0);
            int leaf = node - (m->tmL - 1);
            memcpy(B->o, y->tc + (size_t)leaf * d, d * 4);
            if (m->q8 && m->tmr) {                       // int8 activations x int8 leaf maps
                for (int j = 0; j < m->tmr; j++) {
                    size_t lj = (size_t)leaf * m->tmr + j, off = lj * d;
                    float z = (float)dot_i8(y->P8 + off, B->xq, d) * y->sP[lj] * ui;
                    axpy_i8(B->o, y->Q8 + off, z * y->sQ[lj], d);
                }
            } else for (int j = 0; j < m->tmr; j++) {
                size_t off = ((size_t)leaf * m->tmr + j) * d;
                axpy_ps(B->o, y->tQt + off, dot_ps(y->tP + off, B->u, d), d);
            }
            for (int i = 0; i < d; i++) B->x[i] += B->o[i];
        } else if (m->mlp) {                            // ternary MLP: down(gelu(up(norm(x))))
            int hn = m->mlp * d;
            rmsnorm(B->u, B->x, y->n2, d);
            xi = quant8(B->xq, B->u, d); lut_tables(B->tab, B->xq, d); bitlin(B->hid, &y->up, B, xi);
            for (int i = 0; i < hn; i++) { float a = B->hid[i]; B->hid[i] = 0.5f * a * (1.0f + erff(a * 0.70710678f)); }
            xi = quant8(B->xq, B->hid, hn); lut_tables(B->tab, B->xq, hn); bitlin(B->o, &y->down, B, xi);
            for (int i = 0; i < d; i++) B->x[i] += B->o[i];
        }
        MLP_T += now() - tm0; MLP_N++;
    }
    rmsnorm(B->u, B->x, m->norm, d);
    for (int v = 0; v < m->vocab; v++) B->logits[v] = dotf(m->head + (size_t)v * d, B->u, d);
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rng = 88172645463325252ull;
static float urand(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (rng >> 40) / 16777216.0f; }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: see header\n"); return 1; }
    lut_init(); Model m; load(&m, argv[1]);
    int d = m.d; Buf B;
    float **f[] = {&B.x, &B.u, &B.q, &B.k, &B.v, &B.o, &B.t1}; for (int i = 0; i < 7; i++) *f[i] = calloc(d, 4);
    if (getenv("EXACT")) { FAST = 0; TAU = 1e30f; HALF = 0; }
    if (getenv("TAU")) TAU = atof(getenv("TAU"));
    B.sc16 = aligned_alloc(64, (MAXT + 64) * 2);
    B.att = malloc((MAXT + 16) * 4); B.T = malloc((size_t)(m.S ? m.S : 1) * m.L * 4); B.P = malloc((size_t)(m.S ? m.S : 1) * m.L * 4);
    int wd = d * (m.mlp ? m.mlp : 1);                 // widest BitLinear input/output
    B.logits = malloc(m.vocab * 4); B.xq = malloc(wd); B.hid = malloc(wd * 4);
    B.tab = aligned_alloc(64, ((size_t)(wd + 2) / 3 * 64 + 63) / 64 * 64);
    B.acc = malloc(((wd + 127) / 128 * 128) * 4); B.scr = aligned_alloc(64, ((wd + 127) / 128 * 128) * 4);
    const char *mode = argv[2];
    if (!strcmp(mode, "dump")) {
        FILE *ft = fopen(argv[3], "rb"), *fl = fopen(argv[4], "wb"); int32_t tok; int t = 0;
        while (fread(&tok, 4, 1, ft) == 1) { forward(&m, &B, tok, t++); fwrite(B.logits, 4, m.vocab, fl); }
        fclose(ft); fclose(fl);
        if (getenv("ATTSTAT")) { double tot = 0; for (int i = 0; i < 5; i++) tot += TS[i];
            fprintf(stderr, "  breakdown: write/encode %.0f%% | table %.0f%% | score %.0f%% | weights %.0f%% | values %.0f%%\n", 100 * TS[0] / tot, 100 * TS[1] / tot, 100 * TS[2] / tot, 100 * TS[3] / tot, 100 * TS[4] / tot); }
        if (getenv("ATTSTAT")) fprintf(stderr, "attn mix %.2f us/head-call | %.1f%% of keys read (TAU=%g) | %ld calls\n", ATT_T / ATT_N * 1e6, 100.0 * ATT_KEPT / ATT_KEYS, TAU, ATT_N);
        if (getenv("MLPTIME")) fprintf(stderr, "MLP %s%s: %.3f us/call over %ld calls (real text)\n", m.tmL ? "tree" : "dense", m.q8 ? " int8" : "", MLP_T / MLP_N * 1e6, MLP_N);
        return 0;
    }
    if (!strcmp(mode, "bench")) {
        int n = argc > 3 ? atoi(argv[3]) : 4096; if (n > MAXT) n = MAXT;
        int marks[] = {128, 512, 1024, 2048, 4096, 8192}, mi = 0, tok = 1, w0 = 0; double t0 = now(), tw = t0;
        int fs = FAST && m.S && m.D == 4 && !(m.S & 1) && !m.values;
        printf("%s S=%d D=%d keys %d B/key/head |", !m.S ? "standard" : fs ? (m.axis ? "axis-tree fast-scan" : "tree fast-scan") : "tree (reference)", m.S, m.D,
               !m.S ? (HALF ? 2 : 4) * m.hd : fs ? m.S / 2 : m.S);
        for (int t = 0; t < n; t++) {
            forward(&m, &B, tok, t); tok = (tok * 7 + 3) % m.vocab;
            if (mi < 6 && t + 1 == marks[mi]) { double tn = now(); printf(" ctx<=%d: %.1fus", marks[mi], (tn - tw) / (t + 1 - w0) * 1e6); tw = tn; w0 = t + 1; mi++; }
        }
        printf(" | avg %.1f us/tok | MLP %s %.2f us | attn mix %.2f us/head, %.1f%% of keys read\n", (now() - t0) / n * 1e6, m.tmL ? "tree" : "dense",
               MLP_T / MLP_N * 1e6, ATT_T / ATT_N * 1e6, 100.0 * ATT_KEPT / ATT_KEYS); return 0;
    }
    int n = argc > 3 ? atoi(argv[3]) : 300; float temp = argc > 4 ? atof(argv[4]) : 0.8f;
    if (argc > 5) rng ^= (uint64_t)atoll(argv[5]) * 0x9E3779B97F4A7C15ull;
    const char *prompt = argc > 6 ? argv[6] : "\n"; int tok = 0, t = 0;
    for (const char *p = prompt; *p; p++) { for (int i = 0; i < m.vocab; i++) if (m.chars[i] == *p) { tok = i; break; } forward(&m, &B, tok, t++); putchar(*p); }
    for (int i = 0; i < n && t < MAXT; i++) {
        float mx = -1e30f; for (int v = 0; v < m.vocab; v++) if (B.logits[v] > mx) mx = B.logits[v];
        float z = 0; for (int v = 0; v < m.vocab; v++) { B.logits[v] = expf((B.logits[v] - mx) / temp); z += B.logits[v]; }
        float r = urand() * z; tok = m.vocab - 1; for (int v = 0; v < m.vocab; v++) { r -= B.logits[v]; if (r <= 0) { tok = v; break; } }
        putchar(m.chars[tok]); forward(&m, &B, tok, t++);
    }
    putchar('\n'); return 0;
}
