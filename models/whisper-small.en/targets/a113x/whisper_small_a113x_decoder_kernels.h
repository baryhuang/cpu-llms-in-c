#ifndef CLLM_WHISPER_SMALL_A113X_DECODER_KERNELS_H
#define CLLM_WHISPER_SMALL_A113X_DECODER_KERNELS_H

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

#define WHISPER_SMALL_HAVE_DECODER_Q8_DOT 1

static inline float whisper_small_decoder_s8_f32_16(int8x16_t weights,
                                                     const float *values)
{
    const int16x8_t low = vmovl_s8(vget_low_s8(weights));
    const int16x8_t high = vmovl_s8(vget_high_s8(weights));
    float32x4_t sum = vmulq_f32(
        vcvtq_f32_s32(vmovl_s16(vget_low_s16(low))), vld1q_f32(values));
    sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_high_s16(low))),
                    vld1q_f32(values + 4U));
    sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_low_s16(high))),
                    vld1q_f32(values + 8U));
    sum = vfmaq_f32(sum, vcvtq_f32_s32(vmovl_s16(vget_high_s16(high))),
                    vld1q_f32(values + 12U));
    return vaddvq_f32(sum);
}

static inline float whisper_small_decoder_q8_dot(const unsigned char *records,
                                                  const float *input,
                                                  size_t columns)
{
    float sum = 0.0f;
    for (size_t group = 0U; group < columns / 128U; ++group) {
        uint32_t bits = ((uint32_t)records[0] |
                         ((uint32_t)records[1] << 8U)) << 16U;
        float scale;
        memcpy(&scale, &bits, sizeof(scale));
        for (size_t index = 0U; index < 128U; index += 16U) {
            const int8x16_t weights = vld1q_s8(
                (const int8_t *)(const void *)(records + 2U + index));
            sum += scale * whisper_small_decoder_s8_f32_16(
                weights, input + group * 128U + index);
        }
        records += 130U;
    }
    return sum;
}

#endif
