/* Precision-preserving encoder stem for Amlogic A113X / Cortex-A53. */

#include "../generic/whisper_turbo_frontend.h"

#include <arm_neon.h>
#include <math.h>

static float whisper_turbo_a113x_stem_gelu(float value)
{
    return 0.5f * value *
        (1.0f + erff(value * 0.70710678118654752440f));
}

static inline void whisper_turbo_a113x_fma_f32x4_f64(
    float64x2_t *low,
    float64x2_t *high,
    float32x4_t values,
    float weight)
{
    const float64x2_t weight2 = vdupq_n_f64((double)weight);
    *low = vfmaq_f64(*low, vcvt_f64_f32(vget_low_f32(values)), weight2);
    *high = vfmaq_f64(*high, vcvt_high_f64_f32(values), weight2);
}

static double whisper_turbo_a113x_conv1_scalar(
    const float *mel,
    size_t n_mels,
    size_t input_frames,
    const float *weights,
    float bias,
    size_t frame)
{
    double sum = bias;
    for (size_t input_channel = 0U; input_channel < n_mels; ++input_channel) {
        const float *kernel = weights + input_channel * 3U;
        for (size_t tap = 0U; tap < 3U; ++tap) {
            const long source = (long)frame + (long)tap - 1L;
            if (source >= 0L && (size_t)source < input_frames) {
                sum += (double)kernel[tap] *
                       (double)mel[input_channel * input_frames + (size_t)source];
            }
        }
    }
    return sum;
}

static double whisper_turbo_a113x_conv2_scalar(
    const float *scratch,
    size_t n_state,
    size_t input_frames,
    const float *weights,
    float bias,
    size_t output_frame)
{
    double sum = bias;
    for (size_t input_channel = 0U; input_channel < n_state; ++input_channel) {
        const float *kernel = weights + input_channel * 3U;
        for (size_t tap = 0U; tap < 3U; ++tap) {
            const long source = (long)(output_frame * 2U) + (long)tap - 1L;
            if (source >= 0L && (size_t)source < input_frames) {
                sum += (double)kernel[tap] *
                       (double)scratch[input_channel * input_frames + (size_t)source];
            }
        }
    }
    return sum;
}

static int whisper_turbo_a113x_encoder_stem(
    const float *mel,
    size_t n_mels,
    size_t input_frames,
    size_t n_state,
    const float *conv1_weight,
    const float *conv1_bias,
    const float *conv2_weight,
    const float *conv2_bias,
    const float *positions,
    float *output,
    float *scratch,
    size_t scratch_floats)
{
    const size_t output_frames =
        cllm_whisper_turbo_stem_output_frames(input_frames);
    const size_t required =
        cllm_whisper_turbo_stem_scratch_floats(input_frames, n_state);

    if (mel == NULL || conv1_weight == NULL || conv1_bias == NULL ||
        conv2_weight == NULL || conv2_bias == NULL || positions == NULL ||
        output == NULL || scratch == NULL || n_mels == 0U ||
        input_frames == 0U || n_state == 0U || required == 0U ||
        scratch_floats < required) {
        return -1;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t output_channel = 0U; output_channel < n_state;
         ++output_channel) {
        const float *weights = conv1_weight + output_channel * n_mels * 3U;
        size_t frame = 0U;
        if (frame < input_frames) {
            const double sum = whisper_turbo_a113x_conv1_scalar(
                mel, n_mels, input_frames, weights,
                conv1_bias[output_channel], frame);
            scratch[output_channel * input_frames + frame++] =
                whisper_turbo_a113x_stem_gelu((float)sum);
        }
        for (; frame + 4U < input_frames; frame += 4U) {
            float64x2_t sum_low = vdupq_n_f64(conv1_bias[output_channel]);
            float64x2_t sum_high = sum_low;
            for (size_t input_channel = 0U; input_channel < n_mels;
                 ++input_channel) {
                const float *source = mel + input_channel * input_frames + frame;
                const float *kernel = weights + input_channel * 3U;
                whisper_turbo_a113x_fma_f32x4_f64(
                    &sum_low, &sum_high, vld1q_f32(source - 1U), kernel[0]);
                whisper_turbo_a113x_fma_f32x4_f64(
                    &sum_low, &sum_high, vld1q_f32(source), kernel[1]);
                whisper_turbo_a113x_fma_f32x4_f64(
                    &sum_low, &sum_high, vld1q_f32(source + 1U), kernel[2]);
            }
            double values[4];
            vst1q_f64(values, sum_low);
            vst1q_f64(values + 2U, sum_high);
            for (size_t lane = 0U; lane < 4U; ++lane) {
                scratch[output_channel * input_frames + frame + lane] =
                    whisper_turbo_a113x_stem_gelu((float)values[lane]);
            }
        }
        for (; frame < input_frames; ++frame) {
            const double sum = whisper_turbo_a113x_conv1_scalar(
                mel, n_mels, input_frames, weights,
                conv1_bias[output_channel], frame);
            scratch[output_channel * input_frames + frame] =
                whisper_turbo_a113x_stem_gelu((float)sum);
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t output_channel = 0U; output_channel < n_state;
         ++output_channel) {
        const float *weights = conv2_weight + output_channel * n_state * 3U;
        size_t output_frame = 0U;
        if (output_frame < output_frames) {
            const double sum = whisper_turbo_a113x_conv2_scalar(
                scratch, n_state, input_frames, weights,
                conv2_bias[output_channel], output_frame);
            output[output_channel] = whisper_turbo_a113x_stem_gelu((float)sum) +
                positions[output_channel];
            ++output_frame;
        }
        for (; output_frame * 2U + 8U < input_frames;
             output_frame += 4U) {
            float64x2_t sum_low = vdupq_n_f64(conv2_bias[output_channel]);
            float64x2_t sum_high = sum_low;
            for (size_t input_channel = 0U; input_channel < n_state;
                 ++input_channel) {
                const float *source = scratch + input_channel * input_frames +
                                      output_frame * 2U - 1U;
                const float *kernel = weights + input_channel * 3U;
                const float32x4x2_t first = vld2q_f32(source);
                const float32x4x2_t shifted = vld2q_f32(source + 2U);
                whisper_turbo_a113x_fma_f32x4_f64(
                    &sum_low, &sum_high, first.val[0], kernel[0]);
                whisper_turbo_a113x_fma_f32x4_f64(
                    &sum_low, &sum_high, first.val[1], kernel[1]);
                whisper_turbo_a113x_fma_f32x4_f64(
                    &sum_low, &sum_high, shifted.val[0], kernel[2]);
            }
            double values[4];
            vst1q_f64(values, sum_low);
            vst1q_f64(values + 2U, sum_high);
            for (size_t lane = 0U; lane < 4U; ++lane) {
                const size_t frame = output_frame + lane;
                output[frame * n_state + output_channel] =
                    whisper_turbo_a113x_stem_gelu((float)values[lane]) +
                    positions[frame * n_state + output_channel];
            }
        }
        for (; output_frame < output_frames; ++output_frame) {
            const double sum = whisper_turbo_a113x_conv2_scalar(
                scratch, n_state, input_frames, weights,
                conv2_bias[output_channel], output_frame);
            output[output_frame * n_state + output_channel] =
                whisper_turbo_a113x_stem_gelu((float)sum) +
                positions[output_frame * n_state + output_channel];
        }
    }
    return 0;
}

#define WHISPER_TURBO_TARGET_ENCODER_STEM whisper_turbo_a113x_encoder_stem
#include "../generic/whisper_turbo_frontend.c"
