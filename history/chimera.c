// chimera.c — CPU inference for the Chimera model.
//   retention in recurrent form (O(1) state per token, no KV cache)
//   FFF in hard-tree form (one root->leaf path per tree, early-exit depth knob)
//   BitNet b1.58 arithmetic: int8 activations x ternary weights -> int32
//
// usage:
//   ./chimera model.chim gen   [n_tokens] [max_depth] [temp] [seed] [prompt]
//   ./chimera model.chim bench [n_tokens] [max_depth]
//   ./chimera model.chim dump  <tokens.bin> <logits.bin> [max_depth]   (parity test)
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct { float s; int8_t *w; int rows, cols; } Tern;
typedef struct {
    float *n1, *hn, *n2;
    Tern q, k, v, g, o, a, b;   // a,b = fff w_in/w_out  or dense up/down
} Layer;
typedef struct {
    int vocab, d, L, H, depth, trees, dense, hd, nn;
    float *gamma, *emb, *norm, *head;
    Layer *ly;
    char chars[256];
} Model;

static FILE *fh;
static void rd(void *p, size_t n) { if (fread(p, 1, n, fh) != n) { fprintf(stderr, "short read\n"); exit(1); } }
static float *rdf(size_t n) { float *p = malloc(n * 4); rd(p, n * 4); return p; }
static Tern rdt(int rows, int cols) {
    Tern t = {0, 0, rows, cols};
    size_t n = (size_t)rows * cols, nb = (n + 3) / 4;
    rd(&t.s, 4);
    uint8_t *pk = malloc(nb); rd(pk, nb);
    t.w = malloc(n);                         // unpack once to int8 {-1,0,1}
    for (size_t i = 0; i < n; i++) t.w[i] = (int8_t)((pk[i >> 2] >> (2 * (i & 3))) & 3) - 1;
    free(pk);
    return t;
}

static void load(Model *m, const char *path) {
    fh = fopen(path, "rb"); if (!fh) { perror(path); exit(1); }
    char mg[4]; int h[8]; rd(mg, 4); rd(h, 32);
    if (memcmp(mg, "CHIM", 4)) { fprintf(stderr, "bad magic\n"); exit(1); }
    m->vocab = h[1]; m->d = h[2]; m->L = h[3]; m->H = h[4]; m->depth = h[5]; m->trees = h[6]; m->dense = h[7];
    m->hd = m->d / m->H; m->nn = (1 << m->depth) - 1;
    int d = m->d, N = m->trees * m->nn;
    m->gamma = rdf(m->H); m->emb = rdf((size_t)m->vocab * d);
    m->ly = calloc(m->L, sizeof(Layer));
    for (int l = 0; l < m->L; l++) {
        Layer *y = &m->ly[l];
        y->n1 = rdf(d);
        y->q = rdt(d, d); y->k = rdt(d, d); y->v = rdt(d, d); y->g = rdt(d, d); y->o = rdt(d, d);
        y->hn = rdf(d); y->n2 = rdf(d);
        if (!m->dense) { y->a = rdt(N, d); y->b = rdt(N, d); }
        else           { y->a = rdt(N, d); y->b = rdt(d, N); }
    }
    m->norm = rdf(d); m->head = rdf((size_t)m->vocab * d);
    rd(m->chars, m->vocab);
    fclose(fh);
}

// ---------------------------------------------------------------- kernels
static void rmsnorm(float *o, const float *x, const float *w, int n) {
    float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++) o[i] = x[i] * r * w[i];
}
// absmax int8 per token; returns dequant factor 1/s
static float quant8(int8_t *q, const float *x, int n) {
    float mx = 1e-5f; for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > mx) mx = a; }
    float s = 127.0f / mx;
    for (int i = 0; i < n; i++) { float v = rintf(x[i] * s); q[i] = (int8_t)(v > 127 ? 127 : v < -128 ? -128 : v); }
    return 1.0f / s;
}
static inline int32_t dot8(const int8_t *restrict a, const int8_t *restrict b, int n) {
    int32_t s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s;
}
static void bitlin(float *y, const Tern *t, const int8_t *xq, float xinv) {
    float f = t->s * xinv;
    for (int r = 0; r < t->rows; r++) y[r] = (float)dot8(t->w + (size_t)r * t->cols, xq, t->cols) * f;
}
static inline float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678f)); }
static inline float silu(float x) { return x / (1.0f + expf(-x)); }

// ---------------------------------------------------------------- forward one token
typedef struct { float *x, *xn, *q, *k, *v, *g, *o, *hid, *logits; int8_t *xq, *hq; float *S; long nodes; } Buf;

static void forward(Model *m, Buf *b, int tok, int max_depth) {
    int d = m->d, H = m->H, hd = m->hd;
    memcpy(b->x, m->emb + (size_t)tok * d, d * 4);
    for (int l = 0; l < m->L; l++) {
        Layer *y = &m->ly[l];
        float *S = b->S + (size_t)l * H * hd * hd;
        // ---- retention, recurrent form
        rmsnorm(b->xn, b->x, y->n1, d);
        float xi = quant8(b->xq, b->xn, d);
        bitlin(b->q, &y->q, b->xq, xi); bitlin(b->k, &y->k, b->xq, xi);
        bitlin(b->v, &y->v, b->xq, xi); bitlin(b->g, &y->g, b->xq, xi);
        float ks = 1.0f / sqrtf((float)hd);
        for (int h = 0; h < H; h++) {
            float gm = m->gamma[h], *Sh = S + (size_t)h * hd * hd;
            float *q = b->q + h * hd, *k = b->k + h * hd, *v = b->v + h * hd, *o = b->o + h * hd;
            for (int i = 0; i < hd; i++) {
                float ki = k[i] * ks;
                for (int j = 0; j < hd; j++) Sh[i * hd + j] = gm * Sh[i * hd + j] + ki * v[j];
            }
            for (int j = 0; j < hd; j++) o[j] = 0;
            for (int i = 0; i < hd; i++) { float qi = q[i]; for (int j = 0; j < hd; j++) o[j] += qi * Sh[i * hd + j]; }
            rmsnorm(o, o, y->hn + h * hd, hd);
        }
        for (int i = 0; i < d; i++) b->o[i] *= silu(b->g[i]);
        xi = quant8(b->xq, b->o, d);
        bitlin(b->xn, &y->o, b->xq, xi);
        for (int i = 0; i < d; i++) b->x[i] += b->xn[i];
        // ---- feedforward
        rmsnorm(b->xn, b->x, y->n2, d);
        xi = quant8(b->xq, b->xn, d);
        if (!m->dense) {
            float fi = y->a.s * xi, fo = y->b.s;
            int depth = max_depth < m->depth ? max_depth : m->depth;
            for (int t = 0; t < m->trees; t++) {
                int node = 0;
                for (int lv = 0; lv < depth; lv++) {
                    size_t row = (size_t)t * m->nn + node;
                    int32_t acc = dot8(y->a.w + row * d, b->xq, d);   // the routing "event"
                    float a = (float)acc * fi, hs = gelu(a) * fo;
                    const int8_t *wo = y->b.w + row * d;
                    for (int i = 0; i < d; i++) b->x[i] += hs * wo[i];  // ternary: +hs, 0, -hs
                    node = 2 * node + 1 + (acc > 0);
                    b->nodes++;
                }
            }
        } else {
            int N = y->a.rows;
            bitlin(b->hid, &y->a, b->xq, xi);
            for (int i = 0; i < N; i++) b->hid[i] = gelu(b->hid[i]);
            float hi = quant8(b->hq, b->hid, N);
            bitlin(b->xn, &y->b, b->hq, hi);
            for (int i = 0; i < d; i++) b->x[i] += b->xn[i];
            b->nodes += N;
        }
    }
    rmsnorm(b->xn, b->x, m->norm, d);
    for (int v = 0; v < m->vocab; v++) {
        float s = 0; const float *w = m->head + (size_t)v * d;
        for (int i = 0; i < d; i++) s += w[i] * b->xn[i];
        b->logits[v] = s;
    }
}

static uint64_t rng = 88172645463325252ull;
static float urand(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (rng >> 40) / 16777216.0f; }
static int sample(float *lg, int n, float temp) {
    float mx = lg[0]; for (int i = 1; i < n; i++) if (lg[i] > mx) mx = lg[i];
    float z = 0; for (int i = 0; i < n; i++) { lg[i] = expf((lg[i] - mx) / temp); z += lg[i]; }
    float r = urand() * z;
    for (int i = 0; i < n; i++) { r -= lg[i]; if (r <= 0) return i; }
    return n - 1;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.chim gen|bench|dump ...\n", argv[0]); return 1; }
    Model m; load(&m, argv[1]);
    int d = m.d, N = m.trees * m.nn;
    Buf b = {0};
    b.x = malloc(d * 4); b.xn = malloc(d * 4); b.q = malloc(d * 4); b.k = malloc(d * 4);
    b.v = malloc(d * 4); b.g = malloc(d * 4); b.o = malloc(d * 4);
    b.hid = malloc(N * 4); b.hq = malloc(N); b.xq = malloc(d);
    b.logits = malloc(m.vocab * 4);
    b.S = calloc((size_t)m.L * m.H * m.hd * m.hd, 4);
    const char *mode = argv[2];

    if (!strcmp(mode, "dump")) {
        int md = argc > 5 ? atoi(argv[5]) : m.depth;
        FILE *ft = fopen(argv[3], "rb"), *fl = fopen(argv[4], "wb");
        int32_t tok;
        while (fread(&tok, 4, 1, ft) == 1) { forward(&m, &b, tok, md); fwrite(b.logits, 4, m.vocab, fl); }
        fclose(ft); fclose(fl);
        return 0;
    }
    int n = argc > 3 ? atoi(argv[3]) : 500;
    int md = argc > 4 ? atoi(argv[4]) : m.depth;
    if (!strcmp(mode, "bench")) {
        int tok = 0;
        for (int i = 0; i < 50; i++) { forward(&m, &b, tok, md); tok = (tok + 7) % m.vocab; }  // warmup
        b.nodes = 0;
        double t0 = now();
        for (int i = 0; i < n; i++) { forward(&m, &b, tok, md); tok = (tok + 7) % m.vocab; }
        double dt = now() - t0;
        printf("%s depth=%d  %.2f us/token  %.0f tok/s  ffn rows touched/token/layer=%.1f\n",
               m.dense ? "dense" : "tree", m.dense ? 0 : md, dt / n * 1e6, n / dt, (double)b.nodes / n / m.L);
        return 0;
    }
    // gen
    float temp = argc > 5 ? atof(argv[5]) : 0.8f;
    if (argc > 6) rng ^= (uint64_t)atoll(argv[6]) * 0x9E3779B97F4A7C15ull;
    const char *prompt = argc > 7 ? argv[7] : "\n";
    int tok = 0;
    for (const char *p = prompt; *p; p++) {
        for (int i = 0; i < m.vocab; i++) if (m.chars[i] == *p) { tok = i; break; }
        forward(&m, &b, tok, md); putchar(*p);
    }
    for (int i = 0; i < n; i++) { tok = sample(b.logits, m.vocab, temp); putchar(m.chars[tok]); forward(&m, &b, tok, md); }
    putchar('\n');
    return 0;
}
