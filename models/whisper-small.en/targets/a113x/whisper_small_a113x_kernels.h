#ifndef CLLM_WHISPER_SMALL_A113X_KERNELS_H
#define CLLM_WHISPER_SMALL_A113X_KERNELS_H

#include <arm_neon.h>
#include <stdint.h>

#define WHISPER_SMALL_HAVE_Q4_GROUP_DOT 1
#define WHISPER_SMALL_HAVE_Q4_GEMM 1
#define WHISPER_SMALL_HAVE_SELF_ATTENTION 1

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

static inline void whisper_small_decode_q4_group(const unsigned char *packed,
                                                  float scale,
                                                  float *decoded)
{
    const float32x4_t scale4 = vdupq_n_f32(scale);
    for (uint32_t offset = 0U; offset < 64U; offset += 16U) {
        const int8x16_t bytes = vreinterpretq_s8_u8(vld1q_u8(packed + offset));
        const int8x16_t low = vshrq_n_s8(vshlq_n_s8(bytes, 4), 4);
        const int8x16_t high = vshrq_n_s8(bytes, 4);
        const int8x16x2_t lanes = vzipq_s8(low, high);
        const int8x16_t values[2] = {lanes.val[0], lanes.val[1]};
        for (uint32_t half = 0U; half < 2U; ++half) {
            const int16x8_t wide_low = vmovl_s8(vget_low_s8(values[half]));
            const int16x8_t wide_high = vmovl_s8(vget_high_s8(values[half]));
            float *destination = decoded + offset * 2U + half * 16U;
            vst1q_f32(destination,
                      vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(wide_low))), scale4));
            vst1q_f32(destination + 4U,
                      vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(wide_low))), scale4));
            vst1q_f32(destination + 8U,
                      vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(wide_high))), scale4));
            vst1q_f32(destination + 12U,
                      vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(wide_high))), scale4));
        }
    }
}

static inline float whisper_small_dot_f32_128(const float *left,
                                               const float *right)
{
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    for (uint32_t index = 0U; index < 128U; index += 8U) {
        sum0 = vfmaq_f32(sum0, vld1q_f32(left + index), vld1q_f32(right + index));
        sum1 = vfmaq_f32(sum1, vld1q_f32(left + index + 4U),
                        vld1q_f32(right + index + 4U));
    }
    return vaddvq_f32(vaddq_f32(sum0, sum1));
}

static inline void whisper_small_q4_gemm(const unsigned char *weights,
                                         const float *input,
                                         size_t rows,
                                         size_t input_width,
                                         size_t output_width,
                                         const float *bias,
                                         float *output)
{
    const size_t groups = input_width / 128U;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t output_column = 0U; output_column < output_width; ++output_column) {
        const unsigned char *record = weights + output_column * groups * 66U;
        float decoded[128];
        for (size_t row = 0U; row < rows; ++row)
            output[row * output_width + output_column] =
                bias == NULL ? 0.0f : bias[output_column];
        for (size_t group = 0U; group < groups; ++group) {
            uint32_t bits = ((uint32_t)record[0] | ((uint32_t)record[1] << 8U)) << 16U;
            float scale;
            memcpy(&scale, &bits, sizeof(scale));
            whisper_small_decode_q4_group(record + 2U, scale, decoded);
            for (size_t row = 0U; row < rows; ++row)
                output[row * output_width + output_column] +=
                    whisper_small_dot_f32_128(
                        input + row * input_width + group * 128U, decoded);
            record += 66U;
        }
    }
}

static inline float whisper_small_dot_f32_64(const float *left,
                                              const float *right)
{
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    for (uint32_t index = 0U; index < 64U; index += 8U) {
        sum0 = vfmaq_f32(sum0, vld1q_f32(left + index), vld1q_f32(right + index));
        sum1 = vfmaq_f32(sum1, vld1q_f32(left + index + 4U),
                        vld1q_f32(right + index + 4U));
    }
    return vaddvq_f32(vaddq_f32(sum0, sum1));
}

static inline void whisper_small_self_attention(size_t frames,
                                                 size_t n_state,
                                                 size_t n_heads,
                                                 const float *query,
                                                 const float *key,
                                                 const float *value,
                                                 float *scores,
                                                 float *context)
{
    const size_t head_width = n_state / n_heads;
    const float scale = 1.0f / sqrtf((float)head_width);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (size_t head = 0U; head < n_heads; ++head) {
        for (size_t query_frame = 0U; query_frame < frames; ++query_frame) {
#ifdef _OPENMP
            float *score_row = scores + (size_t)omp_get_thread_num() * frames;
#else
            float *score_row = scores;
#endif
            const size_t head_offset = head * head_width;
            const float *query_head = query + query_frame * n_state + head_offset;
            float *context_head = context + query_frame * n_state + head_offset;
            float maximum = -INFINITY;
            float denominator = 0.0f;

            for (size_t key_frame = 0U; key_frame < frames; ++key_frame) {
                score_row[key_frame] = scale * whisper_small_dot_f32_64(
                    query_head, key + key_frame * n_state + head_offset);
                if (score_row[key_frame] > maximum) maximum = score_row[key_frame];
            }
            for (size_t key_frame = 0U; key_frame < frames; ++key_frame) {
                score_row[key_frame] = expf(score_row[key_frame] - maximum);
                denominator += score_row[key_frame];
            }
            for (size_t channel = 0U; channel < head_width; channel += 4U)
                vst1q_f32(context_head + channel, vdupq_n_f32(0.0f));
            for (size_t key_frame = 0U; key_frame < frames; ++key_frame) {
                const float32x4_t probability =
                    vdupq_n_f32(score_row[key_frame] / denominator);
                const float *value_head = value + key_frame * n_state + head_offset;
                for (size_t channel = 0U; channel < head_width; channel += 4U) {
                    vst1q_f32(context_head + channel,
                              vfmaq_f32(vld1q_f32(context_head + channel),
                                        vld1q_f32(value_head + channel), probability));
                }
            }
        }
    }
}

#endif
