#include <cstddef>
#include <cstdint>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

// y[O] = block-scaled int8 W[O,I] dot block-scaled int8 x[I].
// Weight scales are [O, ceil(I/block)], activation scales are [ceil(I/block)].
// Quantization is performed by Rust so this function changes only the hot dot loop.
extern "C" int coli_q8_block_matvec(const int8_t *w,
                                      const float *w_scales,
                                      const int8_t *x,
                                      const float *x_scales,
                                      float *y,
                                      int O,
                                      int I,
                                      int block) {
    if (!w || !w_scales || !x || !x_scales || !y || O <= 0 || I <= 0 || block <= 0) {
        return 0;
    }
    const int nb = (I + block - 1) / block;

    for (int row = 0; row < O; ++row) {
        const int8_t *wr = w + static_cast<size_t>(row) * I;
        const float *sr = w_scales + static_cast<size_t>(row) * nb;
        float acc = 0.0f;
        for (int bi = 0; bi < nb; ++bi) {
            const int c0 = bi * block;
            const int c1 = (c0 + block < I) ? c0 + block : I;
            int32_t isum = 0;
            int col = c0;
#if defined(__aarch64__)
            int32x4_t vacc = vdupq_n_s32(0);
            for (; col + 16 <= c1; col += 16) {
                const int8x16_t wv = vld1q_s8(wr + col);
                const int8x16_t xv = vld1q_s8(x + col);
                vacc = vdotq_s32(vacc, wv, xv);
            }
            isum += vaddvq_s32(vacc);
#endif
            for (; col < c1; ++col) {
                isum += static_cast<int32_t>(wr[col]) * static_cast<int32_t>(x[col]);
            }
            acc += static_cast<float>(isum) * sr[bi] * x_scales[bi];
        }
        y[row] = acc;
    }
    return 1;
}
