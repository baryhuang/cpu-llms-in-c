#ifndef CLLM_WHISPER_SMALL_A113X_KERNELS_H
#define CLLM_WHISPER_SMALL_A113X_KERNELS_H

#include <arm_neon.h>
#include <stdint.h>

#define WHISPER_SMALL_HAVE_Q4_GROUP_DOT 1

static inline float whisper_small_dot_s8_f32_16(int8x16_t weights,
                                                 const float *values)
{
    const int16x8_t low = vmovl_s8(vget_low_s8(weights));
    const int16x8_t high = vmovl_s8(vget_high_s8(weights));
    float32x4_t sum = vmulq_f32(
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(low))), vld1q_f32(values));
    sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_high_s16(low))),
                    vld1q_f32(values + 4));
    sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_low_s16(high))),
                    vld1q_f32(values + 8));
    sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_high_s16(high))),
                    vld1q_f32(values + 12));
    return vaddvq_f32(sum);
}

static inline float whisper_small_q4_group_dot(const unsigned char *packed,
                                                const float *values)
{
    float sum = 0.0f;
    for (uint32_t offset = 0U; offset < 64U; offset += 16U) {
        const int8x16_t bytes = vreinterpretq_s8_u8(vld1q_u8(packed + offset));
        const int8x16_t low = vshrq_n_s8(vshlq_n_s8(bytes, 4), 4);
        const int8x16_t high = vshrq_n_s8(bytes, 4);
        const int8x16x2_t interleaved = vzipq_s8(low, high);
        sum += whisper_small_dot_s8_f32_16(interleaved.val[0], values + offset * 2U);
        sum += whisper_small_dot_s8_f32_16(interleaved.val[1], values + offset * 2U + 16U);
    }
    return sum;
}

#endif
