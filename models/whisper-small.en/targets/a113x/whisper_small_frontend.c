/* Whisper small.en front end specialized for Amlogic A113X / Cortex-A53. */

#include "../generic/whisper_small_frontend.h"

#include <arm_neon.h>
#include <math.h>

static float whisper_small_a113x_stem_gelu(float value)
{
    return 0.5f * value *
        (1.0f + erff(value * 0.70710678118654752440f));
}

static int whisper_small_a113x_encoder_stem(
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
        cllm_whisper_encoder_stem_output_frames(input_frames);
    const size_t required =
        cllm_whisper_encoder_stem_scratch_floats(input_frames, n_state);

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
        size_t frame = 0U;
        for (; frame < input_frames && frame < 1U; ++frame) {
            float sum = conv1_bias[output_channel];
            for (size_t input_channel = 0U; input_channel < n_mels;
                 ++input_channel) {
                const float *kernel = conv1_weight +
                    (output_channel * n_mels + input_channel) * 3U;
                sum += kernel[1] * mel[input_channel * input_frames];
                if (input_frames > 1U)
                    sum += kernel[2] *
                        mel[input_channel * input_frames + 1U];
            }
            scratch[output_channel * input_frames + frame] =
                whisper_small_a113x_stem_gelu(sum);
        }
        for (; frame + 4U < input_frames; frame += 4U) {
            float32x4_t sum = vdupq_n_f32(conv1_bias[output_channel]);
            for (size_t input_channel = 0U; input_channel < n_mels;
                 ++input_channel) {
                const float *source = mel + input_channel * input_frames + frame;
                const float *kernel = conv1_weight +
                    (output_channel * n_mels + input_channel) * 3U;
                sum = vfmaq_n_f32(sum, vld1q_f32(source - 1U), kernel[0]);
                sum = vfmaq_n_f32(sum, vld1q_f32(source), kernel[1]);
                sum = vfmaq_n_f32(sum, vld1q_f32(source + 1U), kernel[2]);
            }
            float values[4];
            vst1q_f32(values, sum);
            for (size_t lane = 0U; lane < 4U; ++lane)
                scratch[output_channel * input_frames + frame + lane] =
                    whisper_small_a113x_stem_gelu(values[lane]);
        }
        for (; frame < input_frames; ++frame) {
            float sum = conv1_bias[output_channel];
            for (size_t input_channel = 0U; input_channel < n_mels;
                 ++input_channel) {
                const float *kernel = conv1_weight +
                    (output_channel * n_mels + input_channel) * 3U;
                for (size_t tap = 0U; tap < 3U; ++tap) {
                    const long source = (long)frame + (long)tap - 1L;
                    if (source >= 0L && (size_t)source < input_frames)
                        sum += kernel[tap] *
                            mel[input_channel * input_frames + (size_t)source];
                }
            }
            scratch[output_channel * input_frames + frame] =
                whisper_small_a113x_stem_gelu(sum);
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t output_channel = 0U; output_channel < n_state;
         ++output_channel) {
        size_t output_frame = 0U;
        for (; output_frame < output_frames && output_frame < 1U;
             ++output_frame) {
            float sum = conv2_bias[output_channel];
            for (size_t input_channel = 0U; input_channel < n_state;
                 ++input_channel) {
                const float *kernel = conv2_weight +
                    (output_channel * n_state + input_channel) * 3U;
                const float *source = scratch + input_channel * input_frames;
                sum += kernel[1] * source[0];
                if (input_frames > 1U) sum += kernel[2] * source[1];
            }
            output[output_channel] = whisper_small_a113x_stem_gelu(sum) +
                positions[output_channel];
        }
        /* vld2q reads one lane past the last tap used by this four-frame tile. */
        for (; output_frame * 2U + 8U < input_frames;
             output_frame += 4U) {
            float32x4_t sum = vdupq_n_f32(conv2_bias[output_channel]);
            for (size_t input_channel = 0U; input_channel < n_state;
                 ++input_channel) {
                const float *source = scratch + input_channel * input_frames +
                    output_frame * 2U - 1U;
                const float *kernel = conv2_weight +
                    (output_channel * n_state + input_channel) * 3U;
                const float32x4x2_t first = vld2q_f32(source);
                const float32x4x2_t shifted = vld2q_f32(source + 2U);
                sum = vfmaq_n_f32(sum, first.val[0], kernel[0]);
                sum = vfmaq_n_f32(sum, first.val[1], kernel[1]);
                sum = vfmaq_n_f32(sum, shifted.val[0], kernel[2]);
            }
            float values[4];
            vst1q_f32(values, sum);
            for (size_t lane = 0U; lane < 4U; ++lane) {
                const size_t frame = output_frame + lane;
                output[frame * n_state + output_channel] =
                    whisper_small_a113x_stem_gelu(values[lane]) +
                    positions[frame * n_state + output_channel];
            }
        }
        for (; output_frame < output_frames; ++output_frame) {
            float sum = conv2_bias[output_channel];
            for (size_t input_channel = 0U; input_channel < n_state;
                 ++input_channel) {
                const float *kernel = conv2_weight +
                    (output_channel * n_state + input_channel) * 3U;
                const float *source_values =
                    scratch + input_channel * input_frames;
                for (size_t tap = 0U; tap < 3U; ++tap) {
                    const long source = (long)(output_frame * 2U) +
                                        (long)tap - 1L;
                    if (source >= 0L && (size_t)source < input_frames)
                        sum += kernel[tap] * source_values[source];
                }
            }
            output[output_frame * n_state + output_channel] =
                whisper_small_a113x_stem_gelu(sum) +
                positions[output_frame * n_state + output_channel];
        }
    }
    return 0;
}

#define WHISPER_SMALL_TARGET_ENCODER_STEM whisper_small_a113x_encoder_stem
#include "../generic/whisper_small_frontend.c"
