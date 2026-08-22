#ifndef CLLM_WHISPER_TURBO_A113X_DECODER_KERNELS_H
#define CLLM_WHISPER_TURBO_A113X_DECODER_KERNELS_H

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

#define WHISPER_TURBO_HAVE_DECODER_Q4_DOT 1
#define WHISPER_TURBO_HAVE_DECODER_Q8_DOT 1
#define WHISPER_TURBO_HAVE_DECODER_Q5_DOT 1

static inline float whisper_turbo_decoder_s8_f32_16(int8x16_t weights,
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

static inline float whisper_turbo_decoder_q8_dot(const unsigned char *records,
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
            sum += scale * whisper_turbo_decoder_s8_f32_16(
                weights, input + group * 128U + index);
        }
        records += 130U;
    }
    return sum;
}

static inline float whisper_turbo_decoder_q4_dot(const unsigned char *records,
                                                 const float *input,
                                                 size_t columns)
{
    float sum = 0.0f;
    for (size_t group = 0U; group < columns / 128U; ++group) {
        uint32_t bits = ((uint32_t)records[0] |
                         ((uint32_t)records[1] << 8U)) << 16U;
        float scale;
        float group_sum = 0.0f;
        memcpy(&scale, &bits, sizeof(scale));
        const unsigned char *packed = records + 2U;
        const float *values = input + group * 128U;
        for (uint32_t offset = 0U; offset < 64U; offset += 16U) {
            const int8x16_t bytes = vreinterpretq_s8_u8(vld1q_u8(packed + offset));
            const int8x16_t low = vshrq_n_s8(vshlq_n_s8(bytes, 4), 4);
            const int8x16_t high = vshrq_n_s8(bytes, 4);
            const int8x16x2_t lanes = vzipq_s8(low, high);
            group_sum += whisper_turbo_decoder_s8_f32_16(
                lanes.val[0], values + offset * 2U);
            group_sum += whisper_turbo_decoder_s8_f32_16(
                lanes.val[1], values + offset * 2U + 16U);
        }
        sum += scale * group_sum;
        records += 66U;
    }
    return sum;
}

static inline int8x16_t whisper_turbo_decoder_q5_bits16(const unsigned char *plane_pair)
{
    static const uint8_t positions_array[16] = {
        1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U,
        1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U,
    };
    const uint8x16_t positions = vld1q_u8(positions_array);
    const uint8x16_t bytes = vcombine_u8(vdup_n_u8(plane_pair[0]),
                                         vdup_n_u8(plane_pair[1]));
    const uint8x16_t mask = vtstq_u8(bytes, positions);
    return vreinterpretq_s8_u8(vandq_u8(mask, vdupq_n_u8(0x10U)));
}

static inline float whisper_turbo_decoder_q5_dot(const unsigned char *records,
                                                 const float *input,
                                                 size_t columns)
{
    float sum = 0.0f;
    for (size_t group = 0U; group < columns / 128U; ++group) {
        uint32_t bits = ((uint32_t)records[0] |
                         ((uint32_t)records[1] << 8U)) << 16U;
        float scale;
        float group_sum = 0.0f;
        memcpy(&scale, &bits, sizeof(scale));
        const unsigned char *packed = records + 2U;
        const unsigned char *plane = packed + 64U;
        const float *values = input + group * 128U;
        for (uint32_t offset = 0U; offset < 64U; offset += 16U) {
            const uint8x16_t bytes = vld1q_u8(packed + offset);
            const int8x16_t low = vreinterpretq_s8_u8(
                vandq_u8(bytes, vdupq_n_u8(0x0FU)));
            const int8x16_t high = vreinterpretq_s8_u8(vshrq_n_u8(bytes, 4));
            const int8x16x2_t lanes = vzipq_s8(low, high);
            const uint32_t base = offset * 2U;
            const int8x16_t five0 = vorrq_s8(lanes.val[0],
                whisper_turbo_decoder_q5_bits16(plane + base / 8U));
            const int8x16_t five1 = vorrq_s8(lanes.val[1],
                whisper_turbo_decoder_q5_bits16(plane + base / 8U + 2U));
            group_sum += whisper_turbo_decoder_s8_f32_16(
                vshrq_n_s8(vshlq_n_s8(five0, 3), 3), values + base);
            group_sum += whisper_turbo_decoder_s8_f32_16(
                vshrq_n_s8(vshlq_n_s8(five1, 3), 3), values + base + 16U);
        }
        sum += scale * group_sum;
        records += 82U;
    }
    return sum;
}

#endif
