#ifndef QWEN35_A113X_KERNELS_H
#define QWEN35_A113X_KERNELS_H

#include <arm_neon.h>

#define QWEN35_HAVE_TARGET_GROUP_DOT 1
#ifndef QWEN35_DISABLE_TARGET_DELTANET
#define QWEN35_HAVE_TARGET_DELTANET_HEAD 1
#endif

/* Cortex-A53 has ARMv8.0 NEON but no dot-product extension. Widen signed
 * bytes four lanes at a time, convert to float, and accumulate with NEON. */
static inline float qwen35_dot_s8_f32_16(int8x16_t weights, const float *values)
{
    const int16x8_t weights_low = vmovl_s8(vget_low_s8(weights));
    const int16x8_t weights_high = vmovl_s8(vget_high_s8(weights));
    float32x4_t sum = vmulq_f32(
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(weights_low))), vld1q_f32(values));
    sum = vfmaq_f32(sum,
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(weights_low))),
                    vld1q_f32(values + 4));
    sum = vfmaq_f32(sum,
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(weights_high))),
                    vld1q_f32(values + 8));
    sum = vfmaq_f32(sum,
                    vcvtq_f32_s32(vmovl_s16(vget_high_s16(weights_high))),
                    vld1q_f32(values + 12));
    return vaddvq_f32(sum);
}

static inline float qwen35_q4_group_dot(const uint8_t *packed, const float *values)
{
    float sum = 0.0f;
    for (uint32_t offset = 0; offset < 64; offset += 16) {
        const int8x16_t bytes = vreinterpretq_s8_u8(vld1q_u8(packed + offset));
        const int8x16_t low = vshrq_n_s8(vshlq_n_s8(bytes, 4), 4);
        const int8x16_t high = vshrq_n_s8(bytes, 4);
        const int8x16x2_t interleaved = vzipq_s8(low, high);
        sum += qwen35_dot_s8_f32_16(interleaved.val[0], values + offset * 2);
        sum += qwen35_dot_s8_f32_16(interleaved.val[1], values + offset * 2 + 16);
    }
    return sum;
}

static inline float qwen35_q8_group_dot(const int8_t *weights, const float *values)
{
    float sum = 0.0f;
    for (uint32_t offset = 0; offset < 128; offset += 16) {
        sum += qwen35_dot_s8_f32_16(vld1q_s8(weights + offset), values + offset);
    }
    return sum;
}

/* The generic loop walks state[ki][vi] with vi outside, producing a
 * 128-float stride. This form keeps vi contiguous, retains the same reduction
 * order for every output lane, and gives each independent head to one thread. */
static inline void qwen35_deltanet_head(float *state, const float *key,
                                       const float *query, const float *value,
                                       float *core, float decay, float beta)
{
    const float32x4_t decay4 = vdupq_n_f32(decay);
    for (uint32_t offset = 0; offset < LK_DIM * LV_DIM; offset += 4) {
        vst1q_f32(state + offset, vmulq_f32(vld1q_f32(state + offset), decay4));
    }

    for (uint32_t vi = 0; vi < LV_DIM; vi += 4) {
        vst1q_f32(core + vi, vdupq_n_f32(0.0f));
    }
    for (uint32_t ki = 0; ki < LK_DIM; ++ki) {
        const float32x4_t key4 = vdupq_n_f32(key[ki]);
        const float *row = state + (size_t)ki * LV_DIM;
        for (uint32_t vi = 0; vi < LV_DIM; vi += 4) {
            const float32x4_t accumulated = vld1q_f32(core + vi);
            vst1q_f32(core + vi,
                      vfmaq_f32(accumulated, vld1q_f32(row + vi), key4));
        }
    }

    const float32x4_t beta4 = vdupq_n_f32(beta);
    for (uint32_t vi = 0; vi < LV_DIM; vi += 4) {
        const float32x4_t delta = vmulq_f32(
            vsubq_f32(vld1q_f32(value + vi), vld1q_f32(core + vi)), beta4);
        vst1q_f32(core + vi, delta);
    }
    for (uint32_t ki = 0; ki < LK_DIM; ++ki) {
        const float32x4_t key4 = vdupq_n_f32(key[ki]);
        float *row = state + (size_t)ki * LV_DIM;
        for (uint32_t vi = 0; vi < LV_DIM; vi += 4) {
            vst1q_f32(row + vi,
                      vfmaq_f32(vld1q_f32(row + vi), vld1q_f32(core + vi), key4));
        }
    }

    for (uint32_t vi = 0; vi < LV_DIM; vi += 4) {
        vst1q_f32(core + vi, vdupq_n_f32(0.0f));
    }
    for (uint32_t ki = 0; ki < LK_DIM; ++ki) {
        const float32x4_t query4 = vdupq_n_f32(query[ki]);
        const float *row = state + (size_t)ki * LV_DIM;
        for (uint32_t vi = 0; vi < LV_DIM; vi += 4) {
            const float32x4_t accumulated = vld1q_f32(core + vi);
            vst1q_f32(core + vi,
                      vfmaq_f32(accumulated, vld1q_f32(row + vi), query4));
        }
    }
}

#endif
