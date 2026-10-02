// Time the tied output layer (vocab 128256 x 2560) matvec in ggml at several weight types and thread counts.
// Build against bitnet.cpp's ggml: see engine/README.md.
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
int main(int argc, char **argv) {
    int V = 128256, d = 2560, reps = 20;
    enum ggml_type types[] = {GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0};
    const char *names[] = {"f16", "q8_0", "q4_0"};
    float *src = malloc((size_t)V * d * sizeof(float));
    for (size_t i = 0; i < (size_t)V * d; i++) src[i] = (float)((rand() % 2001) - 1000) / 30000.0f;
    for (int ti = 0; ti < 3; ti++) {
        struct ggml_init_params ip = {(size_t)V * d * 2 + 512 * 1024 * 1024, NULL, false};
        struct ggml_context *ctx = ggml_init(ip);
        struct ggml_tensor *W = ggml_new_tensor_2d(ctx, types[ti], d, V);
        struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, 1);
        ggml_quantize_chunk(types[ti], src, W->data, 0, V, d, NULL);
        for (int i = 0; i < d; i++) ((float *)x->data)[i] = (float)((rand() % 2001) - 1000) / 1000.0f;
        struct ggml_tensor *y = ggml_mul_mat(ctx, W, x);
        struct ggml_cgraph *gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, y);
        for (int nt = 1; nt <= 4; nt *= 4) {
            ggml_graph_compute_with_ctx(ctx, gf, nt);
            double t0 = now();
            for (int r = 0; r < reps; r++) ggml_graph_compute_with_ctx(ctx, gf, nt);
            printf("output layer %s  %5.1f MB  threads %d: %.2f ms/token\n", names[ti], ggml_nbytes(W) / 1e6, nt, (now() - t0) / reps * 1e3);
        }
        ggml_free(ctx);
    }
    return 0;
}
