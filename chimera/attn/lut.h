// lut.h — T-MAC-style table-lookup matvec for ternary (b1.58) weights, AVX2, bit-exact.
//
// Weights are grouped 3 at a time along the input dim: 3^3 = 27 patterns. Patterns come in
// +/- pairs, so 13 nonzero canonical patterns + zero = 14 table entries, plus a sign bit.
// Each weight group is stored as ONE byte:  bits 0-3 = canonical index, bit 7 = negate.
// (2.67 bits/weight; no multiplies anywhere.)
//
// Per input vector we build, for every group of 3 int8 activations, the 14 int16 dot
// products of that chunk with each canonical pattern, split into lo/hi byte planes, for
// both +T and -T. Then one pshufb looks up 32 output rows at once:
//   pshufb(T,  idx)       -> T[c]  if sign bit clear, else 0   (pshufb zeroes on bit 7)
//   pshufb(-T, idx^0x80)  -> -T[c] if sign bit set,   else 0
// OR them, interleave lo/hi into int16, accumulate. int16 accumulators are flushed to
// int32 every LUT_CG groups (64 * 384 < 32767, so no overflow). Result == int8 x ternary dot.
#pragma once
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define LUT_CG 64      // groups per chunk (int16 overflow bound)
#define LUT_RT 4       // row blocks (of 32) per tile

static int8_t LUT_PAT[16][3];
static uint8_t LUT_ENC[27];

static void lut_init(void) {
    int n = 1;
    memset(LUT_PAT, 0, sizeof LUT_PAT);
    for (int p = 0; p < 27; p++) {
        int w[3] = {p % 3 - 1, (p / 3) % 3 - 1, p / 9 - 1}, f = 0;
        for (int i = 0; i < 3; i++) if (w[i]) { f = w[i]; break; }
        if (f > 0) { for (int i = 0; i < 3; i++) LUT_PAT[n][i] = (int8_t)w[i]; LUT_ENC[p] = (uint8_t)n++; }
        else if (f == 0) LUT_ENC[p] = 0;
    }
    for (int p = 0; p < 27; p++) {
        int w[3] = {p % 3 - 1, (p / 3) % 3 - 1, p / 9 - 1}, f = 0;
        for (int i = 0; i < 3; i++) if (w[i]) { f = w[i]; break; }
        if (f < 0) LUT_ENC[p] = LUT_ENC[(-w[0] + 1) + 3 * (-w[1] + 1) + 9 * (-w[2] + 1)] | 0x80;
    }
}

typedef struct { int rows, cols, G, NRB; uint8_t *w; } LutW;   // NRB padded to multiple of LUT_RT

// build packed LUT weights from int8 {-1,0,1} row-major [rows, cols]
static LutW lut_pack(const int8_t *W, int rows, int cols) {
    LutW t; t.rows = rows; t.cols = cols; t.G = (cols + 2) / 3;
    t.NRB = ((rows + 32 * LUT_RT - 1) / (32 * LUT_RT)) * LUT_RT;
    t.w = aligned_alloc(64, ((size_t)t.NRB * 32 * t.G + 63) / 64 * 64);
    size_t off = 0;
    for (int c0 = 0; c0 < t.G; c0 += LUT_CG) {
        int cg = t.G - c0 < LUT_CG ? t.G - c0 : LUT_CG;
        for (int rb = 0; rb < t.NRB; rb++)
            for (int gi = 0; gi < cg; gi++)
                for (int j = 0; j < 32; j++) {
                    int r = rb * 32 + j, g = c0 + gi, w[3] = {0, 0, 0};
                    if (r < rows) for (int i = 0; i < 3; i++) if (3 * g + i < cols) w[i] = W[(size_t)r * cols + 3 * g + i];
                    t.w[off + ((size_t)rb * cg + gi) * 32 + j] = LUT_ENC[(w[0] + 1) + 3 * (w[1] + 1) + 9 * (w[2] + 1)];
                }
        off += (size_t)t.NRB * cg * 32;
    }
    return t;
}

// tables for one int8 input vector: G * 64 bytes  [Tlo | Thi | NTlo | NThi], 16 each
static void lut_tables(uint8_t *tab, const int8_t *x, int cols) {
    int G = (cols + 2) / 3;
    for (int g = 0; g < G; g++) {
        int x0 = 3 * g < cols ? x[3 * g] : 0, x1 = 3 * g + 1 < cols ? x[3 * g + 1] : 0, x2 = 3 * g + 2 < cols ? x[3 * g + 2] : 0;
        uint8_t *tb = tab + (size_t)g * 64;
        for (int c = 0; c < 16; c++) {
            int16_t v = c < 14 ? (int16_t)(LUT_PAT[c][0] * x0 + LUT_PAT[c][1] * x1 + LUT_PAT[c][2] * x2) : 0;
            int16_t nv = (int16_t)-v;
            tb[c] = (uint8_t)(v & 0xFF); tb[16 + c] = (uint8_t)((v >> 8) & 0xFF);
            tb[32 + c] = (uint8_t)(nv & 0xFF); tb[48 + c] = (uint8_t)((nv >> 8) & 0xFF);
        }
    }
}

// y[rows] (int32, exact) = W . x
static void lut_mv(int32_t *y, const LutW *t, const uint8_t *tab, int32_t *scratch /* NRB*32 */) {
    memset(scratch, 0, (size_t)t->NRB * 32 * 4);
    const __m256i flip = _mm256_set1_epi8((char)0x80);
    size_t off = 0;
    for (int c0 = 0; c0 < t->G; c0 += LUT_CG) {
        int cg = t->G - c0 < LUT_CG ? t->G - c0 : LUT_CG;
        for (int rt = 0; rt < t->NRB; rt += LUT_RT) {
            __m256i a0[LUT_RT], a1[LUT_RT];
            for (int j = 0; j < LUT_RT; j++) a0[j] = a1[j] = _mm256_setzero_si256();
            const uint8_t *wb = t->w + off + (size_t)rt * cg * 32;
            for (int gi = 0; gi < cg; gi++) {
                const uint8_t *tb = tab + (size_t)(c0 + gi) * 64;
                __m256i tl = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)tb));
                __m256i th = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(tb + 16)));
                __m256i nl = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(tb + 32)));
                __m256i nh = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(tb + 48)));
                for (int j = 0; j < LUT_RT; j++) {
                    __m256i idx = _mm256_loadu_si256((const __m256i *)(wb + ((size_t)j * cg + gi) * 32));
                    __m256i nidx = _mm256_xor_si256(idx, flip);
                    __m256i lo = _mm256_or_si256(_mm256_shuffle_epi8(tl, idx), _mm256_shuffle_epi8(nl, nidx));
                    __m256i hi = _mm256_or_si256(_mm256_shuffle_epi8(th, idx), _mm256_shuffle_epi8(nh, nidx));
                    a0[j] = _mm256_add_epi16(a0[j], _mm256_unpacklo_epi8(lo, hi));   // rows 0-7, 16-23
                    a1[j] = _mm256_add_epi16(a1[j], _mm256_unpackhi_epi8(lo, hi));   // rows 8-15, 24-31
                }
            }
            for (int j = 0; j < LUT_RT; j++) {
                int32_t *yb = scratch + (size_t)(rt + j) * 32;
                __m256i r0 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a0[j]));
                __m256i r2 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a0[j], 1));
                __m256i r1 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(a1[j]));
                __m256i r3 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(a1[j], 1));
                _mm256_storeu_si256((__m256i *)(yb + 0),  _mm256_add_epi32(_mm256_loadu_si256((__m256i *)(yb + 0)), r0));
                _mm256_storeu_si256((__m256i *)(yb + 8),  _mm256_add_epi32(_mm256_loadu_si256((__m256i *)(yb + 8)), r1));
                _mm256_storeu_si256((__m256i *)(yb + 16), _mm256_add_epi32(_mm256_loadu_si256((__m256i *)(yb + 16)), r2));
                _mm256_storeu_si256((__m256i *)(yb + 24), _mm256_add_epi32(_mm256_loadu_si256((__m256i *)(yb + 24)), r3));
            }
        }
        off += (size_t)t->NRB * cg * 32;
    }
    memcpy(y, scratch, (size_t)t->rows * 4);
}

// reference int8 x ternary dot (unpacked weights), auto-vectorised
static inline int32_t dot8(const int8_t *restrict a, const int8_t *restrict b, int n) {
    int32_t s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s;
}

// strong AVX2 baseline: ternary multiply == sign_epi8(x, w); sum bytes via maddubs(1, .).
// Exact for x in [-127,127] (b1.58 absmax quant never produces -128). w: int8 {-1,0,1}.
static inline int32_t dot_tern(const int8_t *restrict w, const int8_t *restrict x, int n) {
    const __m256i one8 = _mm256_set1_epi8(1), one16 = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    int i = 0;
    while (i + 32 <= n) {
        __m256i a16 = _mm256_setzero_si256();
        int lim = i + 32 * 64 < n ? i + 32 * 64 : n;          // <= 64 steps of <=254 per int16 lane
        for (; i + 32 <= lim; i += 32) {
            __m256i p = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(x + i)), _mm256_loadu_si256((const __m256i *)(w + i)));
            a16 = _mm256_add_epi16(a16, _mm256_maddubs_epi16(one8, p));
        }
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a16, one16));
    }
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
    int32_t r = _mm_cvtsi128_si32(s);
    for (; i < n; i++) r += w[i] * x[i];
    return r;
}
