// attn.c — CPU inference for the single-attention-layer models (no MLP).
//   heads: 'S' softmax (RoPE, KV cache)  |  'R' routed retention (per-leaf state, lazy decay)
//   projections: BitNet b1.58 ternary, three exact backends: naive int8 dot / AVX2 sign-dot / LUT
//
//   ./attn model.cha2 dump  <toks.bin> <logits.bin> [backend]
//   ./attn model.cha2 bench <n_tokens> [backend]          (reports per-token cost vs position)
//   ./attn model.cha2 gen   <n> [temp] [seed] [prompt] [backend]
//   ./attn model.cha2 check                                (all backends give identical int32 projections)
//   backend: 0 naive, 1 avx2-sign, 2 lut (default)
#include <math.h>
#include <stdio.h>
#include <time.h>
#include "lut.h"

#define MAXT 8192
typedef struct { float s; int rows, cols; int8_t *w; LutW L; } Tern;
typedef struct { float *nw, *mk, *mv, *hn, *rw; Tern q, k, v, g, o;
                 float *uprev, *S, *Sl, *K, *V; int *tl; } Layer;
typedef struct { int vocab, d, H, hd, layers, rdepth, nleaf, mask, local; float *gamma, *gleaf, *emb, *norm, *head; Layer *ly; char chars[256]; } Model;

static int BACKEND = 2;
static FILE *fh;
static void rd(void *p, size_t n) { if (fread(p, 1, n, fh) != n) { fprintf(stderr, "short read\n"); exit(1); } }
static float *rdf(size_t n) { float *p = malloc(n * 4); rd(p, n * 4); return p; }
static Tern rdt(int rows, int cols) {
    Tern t; t.rows = rows; t.cols = cols; size_t n = (size_t)rows * cols, nb = (n + 3) / 4;
    rd(&t.s, 4); uint8_t *pk = malloc(nb); rd(pk, nb);
    t.w = aligned_alloc(64, (n + 63) / 64 * 64);
    for (size_t i = 0; i < n; i++) t.w[i] = (int8_t)((pk[i >> 2] >> (2 * (i & 3))) & 3) - 1;
    free(pk); t.L = lut_pack(t.w, rows, cols);
    return t;
}
static void load(Model *m, const char *path) {
    fh = fopen(path, "rb"); if (!fh) { perror(path); exit(1); }
    char mg[4]; int h[8]; rd(mg, 4); rd(h, 32);
    if (memcmp(mg, "CHA2", 4) || h[0] != 2) { fprintf(stderr, "bad file\n"); exit(1); }
    m->vocab = h[1]; m->d = h[2]; m->H = h[3]; m->layers = h[4]; m->rdepth = h[5]; m->mask = h[6]; m->local = h[7];
    m->hd = m->d / m->H; m->nleaf = 1 << m->rdepth;
    int d = m->d;
    m->gamma = rdf(m->H); m->gleaf = rdf(m->H); m->emb = rdf((size_t)m->vocab * d);
    m->ly = calloc(m->layers, sizeof(Layer));
    for (int l = 0; l < m->layers; l++) {
        Layer *y = &m->ly[l];
        y->nw = rdf(d); y->mk = rdf(d); y->mv = rdf(d);
        y->q = rdt(d, d); y->k = rdt(d, d); y->v = rdt(d, d); y->g = rdt(d, d); y->o = rdt(d, d);
        y->hn = rdf(d);
        y->rw = m->rdepth ? rdf((size_t)m->H * (m->nleaf - 1) * m->hd) : NULL;
        y->uprev = calloc(d, 4);
        y->S = calloc((size_t)m->H * m->nleaf * m->hd * m->hd, 4);
        y->Sl = calloc((size_t)m->H * m->hd * m->hd, 4);
        y->tl = calloc((size_t)m->H * m->nleaf, sizeof(int));
        y->K = malloc((size_t)m->H * MAXT * m->hd * 4); y->V = malloc((size_t)m->H * MAXT * m->hd * 4);
    }
    m->norm = rdf(d); m->head = rdf((size_t)m->vocab * d); rd(m->chars, m->vocab);
    fclose(fh);
}

// ---------------------------------------------------------------- helpers
static void rmsnorm(float *o, const float *x, const float *w, int n) {
    float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / n + 1e-6f); for (int i = 0; i < n; i++) o[i] = x[i] * r * w[i];
}
static float quant8(int8_t *q, const float *x, int n) {
    float mx = 1e-5f; for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > mx) mx = a; }
    float s = 127.0f / mx;
    for (int i = 0; i < n; i++) { float v = rintf(x[i] * s); q[i] = (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v); }
    return 1.0f / s;
}
typedef struct { int8_t *xq; float xinv; uint8_t *tab; int32_t *acc, *scr; } QIn;
static void qin_set(QIn *in, const float *x, int n) {
    in->xinv = quant8(in->xq, x, n);
    if (BACKEND == 2) lut_tables(in->tab, in->xq, n);
}
static void bitlin(float *y, const Tern *t, QIn *in) {
    int32_t *a = in->acc;
    if (BACKEND == 2) lut_mv(a, &t->L, in->tab, in->scr);
    else if (BACKEND == 1) for (int r = 0; r < t->rows; r++) a[r] = dot_tern(t->w + (size_t)r * t->cols, in->xq, t->cols);
    else for (int r = 0; r < t->rows; r++) a[r] = dot8(t->w + (size_t)r * t->cols, in->xq, t->cols);
    float f = t->s * in->xinv;
    for (int r = 0; r < t->rows; r++) y[r] = (float)a[r] * f;
}
static void rope(float *x, int hd, int t) {
    int half = hd / 2;
    for (int i = 0; i < half; i++) {
        float inv = powf(10000.0f, -(float)i / half), ang = (float)t * inv, c = cosf(ang), s = sinf(ang);
        float a = x[i], b = x[i + half];
        x[i] = a * c - b * s; x[i + half] = a * s + b * c;
    }
}
static int route(const float *W, const float *x, int hd, int depth) {
    int node = 0;
    for (int l = 0; l < depth; l++) {
        const float *w = W + (size_t)node * hd; float a = 0;
        for (int i = 0; i < hd; i++) a += w[i] * x[i];
        node = 2 * node + 1 + (a > 0);
    }
    return node - ((1 << depth) - 1);
}
static inline float silu(float x) { return x / (1.0f + expf(-x)); }

// ---------------------------------------------------------------- one token
typedef struct { float *x, *u, *t1, *q, *k, *v, *g, *o, *att, *logits; QIn a, b; } Buf;

static void forward(Model *m, Buf *B, int tok, int t) {
    int d = m->d, H = m->H, hd = m->hd, nl = m->nleaf;
    memcpy(B->x, m->emb + (size_t)tok * d, d * 4);
    for (int l = 0; l < m->layers; l++) {
        Layer *y = &m->ly[l];
        rmsnorm(B->u, B->x, y->nw, d);
        qin_set(&B->a, B->u, d); bitlin(B->q, &y->q, &B->a); bitlin(B->g, &y->g, &B->a);
        for (int i = 0; i < d; i++) B->t1[i] = B->u[i] * y->mk[i] + y->uprev[i] * (1 - y->mk[i]);
        qin_set(&B->b, B->t1, d); bitlin(B->k, &y->k, &B->b);
        for (int i = 0; i < d; i++) B->t1[i] = B->u[i] * y->mv[i] + y->uprev[i] * (1 - y->mv[i]);
        qin_set(&B->b, B->t1, d); bitlin(B->v, &y->v, &B->b);
        memcpy(y->uprev, B->u, d * 4);
        float ks = 1.0f / sqrtf((float)hd);
        for (int h = 0; h < H; h++) {
            float *q = B->q + h * hd, *k = B->k + h * hd, *v = B->v + h * hd, *o = B->o + h * hd;
            if (m->mask >> h & 1) {                       // ---- softmax head, KV cache
                rope(q, hd, t); rope(k, hd, t);
                float *K = y->K + (size_t)h * MAXT * hd, *V = y->V + (size_t)h * MAXT * hd;
                memcpy(K + (size_t)t * hd, k, hd * 4); memcpy(V + (size_t)t * hd, v, hd * 4);
                float mx = -1e30f;
                for (int j = 0; j <= t; j++) {
                    float s = 0; const float *kj = K + (size_t)j * hd;
                    for (int i = 0; i < hd; i++) s += q[i] * kj[i];
                    B->att[j] = s * ks; if (B->att[j] > mx) mx = B->att[j];
                }
                float z = 0; for (int j = 0; j <= t; j++) { B->att[j] = expf(B->att[j] - mx); z += B->att[j]; }
                for (int i = 0; i < hd; i++) o[i] = 0;
                for (int j = 0; j <= t; j++) { float p = B->att[j] / z; const float *vj = V + (size_t)j * hd; for (int i = 0; i < hd; i++) o[i] += p * vj[i]; }
            } else {                                      // ---- routed retention, per-leaf state
                int bk = 0, bq = 0;
                if (m->rdepth) {
                    const float *W = y->rw + (size_t)h * (nl - 1) * hd;
                    bk = route(W, k, hd, m->rdepth); bq = route(W, q, hd, m->rdepth);   // un-rotated
                }
                rope(q, hd, t); rope(k, hd, t);
                float gm = m->local ? m->gleaf[h] : m->gamma[h];
                int *tl = y->tl + h * nl;
                float *Sk = y->S + ((size_t)h * nl + bk) * hd * hd;
                float dk = powf(gm, (float)(t - tl[bk]));
                for (int i = 0; i < hd; i++) { float ki = k[i] * ks; float *row = Sk + i * hd;
                    for (int j = 0; j < hd; j++) row[j] = dk * row[j] + ki * v[j]; }
                tl[bk] = t;
                float *Sq = y->S + ((size_t)h * nl + bq) * hd * hd;
                float dq = powf(gm, (float)(t - tl[bq]));
                for (int j = 0; j < hd; j++) o[j] = 0;
                for (int i = 0; i < hd; i++) { float qi = q[i] * dq; const float *row = Sq + i * hd;
                    for (int j = 0; j < hd; j++) o[j] += qi * row[j]; }
                if (m->local) {                           // fast local state, written every token
                    float *Sl = y->Sl + (size_t)h * hd * hd, gl = m->gamma[h];
                    for (int i = 0; i < hd; i++) { float ki = k[i] * ks; float *row = Sl + i * hd;
                        for (int j = 0; j < hd; j++) row[j] = gl * row[j] + ki * v[j]; }
                    for (int i = 0; i < hd; i++) { float qi = q[i]; const float *row = Sl + i * hd;
                        for (int j = 0; j < hd; j++) o[j] += qi * row[j]; }
                }
            }
            rmsnorm(o, o, y->hn + h * hd, hd);
        }
        for (int i = 0; i < d; i++) B->o[i] *= silu(B->g[i]);
        qin_set(&B->a, B->o, d); bitlin(B->t1, &y->o, &B->a);
        for (int i = 0; i < d; i++) B->x[i] += B->t1[i];
    }
    rmsnorm(B->u, B->x, m->norm, d);
    for (int v = 0; v < m->vocab; v++) { float s = 0; const float *w = m->head + (size_t)v * d;
        for (int i = 0; i < d; i++) s += w[i] * B->u[i]; B->logits[v] = s; }
}

static void reset(Model *m) {
    for (int l = 0; l < m->layers; l++) { Layer *y = &m->ly[l];
        memset(y->uprev, 0, m->d * 4); memset(y->S, 0, (size_t)m->H * m->nleaf * m->hd * m->hd * 4);
        memset(y->tl, 0, (size_t)m->H * m->nleaf * sizeof(int)); }
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static uint64_t rng = 88172645463325252ull;
static float urand(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (rng >> 40) / 16777216.0f; }

static void qin_alloc(QIn *q, int d) {
    q->xq = malloc(d); q->tab = aligned_alloc(64, ((size_t)(d + 2) / 3 * 64 + 63) / 64 * 64);
    q->acc = malloc(((d + 127) / 128 * 128) * 4); q->scr = aligned_alloc(64, ((d + 127) / 128 * 128) * 4);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: see header\n"); return 1; }
    lut_init();
    Model m; load(&m, argv[1]);
    int d = m.d;
    Buf B; float **bufs[] = {&B.x, &B.u, &B.t1, &B.q, &B.k, &B.v, &B.g, &B.o};
    for (int i = 0; i < 8; i++) *bufs[i] = calloc(d, 4);
    B.att = malloc(MAXT * 4); B.logits = malloc(m.vocab * 4);
    qin_alloc(&B.a, d); qin_alloc(&B.b, d);
    const char *mode = argv[2];

    if (!strcmp(mode, "check")) {
        long bad = 0; int8_t *x = malloc(d); int32_t *r0 = malloc(d * 4), *r1 = malloc(d * 4), *r2 = malloc(d * 4);
        uint8_t *tab = B.a.tab;
        for (int l = 0; l < m.layers; l++) { Tern *ts[] = {&m.ly[l].q, &m.ly[l].k, &m.ly[l].v, &m.ly[l].g, &m.ly[l].o};
            for (int tt = 0; tt < 5; tt++) for (int trial = 0; trial < 100; trial++) {
                for (int i = 0; i < d; i++) x[i] = (int8_t)(urand() * 255) - 127;
                lut_tables(tab, x, d); lut_mv(r2, &ts[tt]->L, tab, B.a.scr);
                for (int r = 0; r < d; r++) { r0[r] = dot8(ts[tt]->w + (size_t)r * d, x, d); r1[r] = dot_tern(ts[tt]->w + (size_t)r * d, x, d);
                    bad += (r0[r] != r1[r]) + (r0[r] != r2[r]); } } }
        printf("backend check: %ld mismatches over %d int32 outputs\n", bad, m.layers * 5 * 100 * d);
        return bad != 0;
    }
    if (!strcmp(mode, "dump")) {
        if (argc > 5) BACKEND = atoi(argv[5]);
        FILE *ft = fopen(argv[3], "rb"), *fl = fopen(argv[4], "wb"); int32_t tok; int t = 0;
        while (fread(&tok, 4, 1, ft) == 1) { forward(&m, &B, tok, t++); fwrite(B.logits, 4, m.vocab, fl); }
        fclose(ft); fclose(fl); return 0;
    }
    if (!strcmp(mode, "bench")) {
        int n = argc > 3 ? atoi(argv[3]) : 4096; if (n > MAXT) n = MAXT;
        if (argc > 4) BACKEND = atoi(argv[4]);
        int marks[] = {128, 512, 1024, 2048, 4096, 8192}; int mi = 0;
        double t0 = now(), tw = t0; int tok = 1, w0 = 0;
        printf("backend=%d heads=", BACKEND); for (int h = 0; h < m.H; h++) putchar(m.mask >> h & 1 ? 'S' : 'R');
        printf(" leaves=%d local=%d |", m.nleaf, m.local);
        for (int t = 0; t < n; t++) {
            forward(&m, &B, tok, t); tok = (tok * 7 + 3) % m.vocab;
            if (mi < 6 && t + 1 == marks[mi]) { double tn = now(); printf(" ctx<=%d: %.1fus", marks[mi], (tn - tw) / (t + 1 - w0) * 1e6); tw = tn; w0 = t + 1; mi++; }
        }
        printf(" | avg %.1f us/tok\n", (now() - t0) / n * 1e6);
        return 0;
    }
    // gen
    int n = argc > 3 ? atoi(argv[3]) : 300; float temp = argc > 4 ? atof(argv[4]) : 0.8f;
    if (argc > 5) rng ^= (uint64_t)atoll(argv[5]) * 0x9E3779B97F4A7C15ull;
    const char *prompt = argc > 6 ? argv[6] : "\n"; if (argc > 7) BACKEND = atoi(argv[7]);
    int tok = 0, t = 0;
    for (const char *p = prompt; *p; p++) { for (int i = 0; i < m.vocab; i++) if (m.chars[i] == *p) { tok = i; break; }
        forward(&m, &B, tok, t++); putchar(*p); }
    for (int i = 0; i < n && t < MAXT; i++) {
        float mx = -1e30f; for (int v = 0; v < m.vocab; v++) if (B.logits[v] > mx) mx = B.logits[v];
        float z = 0; for (int v = 0; v < m.vocab; v++) { B.logits[v] = expf((B.logits[v] - mx) / temp); z += B.logits[v]; }
        float r = urand() * z; tok = m.vocab - 1; for (int v = 0; v < m.vocab; v++) { r -= B.logits[v]; if (r <= 0) { tok = v; break; } }
        putchar(m.chars[tok]); forward(&m, &B, tok, t++);
    }
    putchar('\n'); (void)reset; return 0;
}
