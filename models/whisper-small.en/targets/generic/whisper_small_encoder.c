#include "whisper_small_encoder.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef WHISPER_SMALL_TARGET_KERNELS
#include WHISPER_SMALL_TARGET_KERNELS
#endif

enum { Q4_GROUP_SIZE = 128, Q4_RECORD_BYTES = 66 };

static float bf16_to_float(const unsigned char *bytes)
{
    uint32_t bits = ((uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8U)) << 16U;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

#ifndef WHISPER_SMALL_HAVE_Q4_GROUP_DOT
static float whisper_small_q4_group_dot(const unsigned char *packed, const float *values)
{
    float sum = 0.0f;
    for (size_t index = 0U; index < Q4_GROUP_SIZE / 2U; ++index) {
        const unsigned char byte = packed[index];
        int low = byte & 0x0f;
        int high = byte >> 4;
        low = low >= 8 ? low - 16 : low;
        high = high >= 8 ? high - 16 : high;
        sum += values[index * 2U] * (float)low;
        sum += values[index * 2U + 1U] * (float)high;
    }
    return sum;
}
#endif

static float exact_gelu(float value)
{
    return 0.5f * value * (1.0f + erff(value * 0.70710678118654752440f));
}

static void layer_norm(const float *input,
                       size_t rows,
                       size_t width,
                       const float *weight,
                       const float *bias,
                       float *output)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t row = 0U; row < rows; ++row) {
        const float *source = input + row * width;
        float *destination = output + row * width;
        double sum = 0.0;
        double square_sum = 0.0;

        for (size_t column = 0U; column < width; ++column) {
            sum += source[column];
            square_sum += (double)source[column] * (double)source[column];
        }
        const double mean = sum / (double)width;
        double variance = square_sum / (double)width - mean * mean;
        if (variance < 0.0) variance = 0.0;
        const float inverse = 1.0f / sqrtf((float)variance + 1.0e-5f);
        for (size_t column = 0U; column < width; ++column) {
            destination[column] = (source[column] - (float)mean) * inverse *
                                  weight[column] + bias[column];
        }
    }
}

static void linear(const float *input,
                   size_t rows,
                   size_t input_width,
                   const float *weight,
                   const float *bias,
                   size_t output_width,
                   float *output)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t output_column = 0U; output_column < output_width; ++output_column) {
        for (size_t row = 0U; row < rows; ++row) {
            double sum = bias == NULL ? 0.0 : bias[output_column];
            const float *weight_row = weight + output_column * input_width;
            for (size_t input_column = 0U; input_column < input_width; ++input_column) {
                sum += (double)input[row * input_width + input_column] *
                       (double)weight_row[input_column];
            }
            output[row * output_width + output_column] = (float)sum;
        }
    }
}

static void linear_q4(const float *input,
                      size_t rows,
                      size_t input_width,
                      const unsigned char *weight,
                      const float *bias,
                      size_t output_width,
                      float *output)
{
#ifdef WHISPER_SMALL_HAVE_Q4_GEMM
    whisper_small_q4_gemm(weight, input, rows, input_width,
                          output_width, bias, output);
#else
    const size_t groups = input_width / Q4_GROUP_SIZE;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t output_column = 0U; output_column < output_width; ++output_column) {
        const unsigned char *weight_row = weight + output_column * groups * Q4_RECORD_BYTES;
        for (size_t row = 0U; row < rows; ++row) {
            float sum = bias == NULL ? 0.0f : bias[output_column];
            const float *input_row = input + row * input_width;
            const unsigned char *record = weight_row;
            for (size_t group = 0U; group < groups; ++group) {
                sum += bf16_to_float(record) *
                       whisper_small_q4_group_dot(record + 2U,
                           input_row + group * Q4_GROUP_SIZE);
                record += Q4_RECORD_BYTES;
            }
            output[row * output_width + output_column] = sum;
        }
    }
#endif
}

static void linear_any(const float *input,
                       size_t rows,
                       size_t input_width,
                       const float *weight_f32,
                       const unsigned char *weight_q4,
                       const float *bias,
                       size_t output_width,
                       float *output)
{
    if (weight_q4 != NULL)
        linear_q4(input, rows, input_width, weight_q4, bias, output_width, output);
    else
        linear(input, rows, input_width, weight_f32, bias, output_width, output);
}

static void self_attention(size_t frames,
                           size_t n_state,
                           size_t n_heads,
                           float *query,
                           const float *key,
                           const float *value,
                           float *scores,
                           float *context)
{
    const size_t head_width = n_state / n_heads;
    const float qk_scale = 1.0f / sqrtf((float)head_width);

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (size_t head = 0U; head < n_heads; ++head) {
        for (size_t query_frame = 0U; query_frame < frames; ++query_frame) {
            float maximum = -INFINITY;
            double denominator = 0.0;
#ifdef _OPENMP
            float *score_row = scores + (size_t)omp_get_thread_num() * frames;
#else
            float *score_row = scores;
#endif

            for (size_t key_frame = 0U; key_frame < frames; ++key_frame) {
                double dot = 0.0;
                for (size_t channel = 0U; channel < head_width; ++channel) {
                    const size_t state_channel = head * head_width + channel;
                    dot += (double)query[query_frame * n_state + state_channel] *
                           (double)key[key_frame * n_state + state_channel];
                }
                score_row[key_frame] = (float)dot * qk_scale;
                if (score_row[key_frame] > maximum) maximum = score_row[key_frame];
            }
            for (size_t key_frame = 0U; key_frame < frames; ++key_frame) {
                score_row[key_frame] = expf(score_row[key_frame] - maximum);
                denominator += score_row[key_frame];
            }
            for (size_t channel = 0U; channel < head_width; ++channel) {
                double sum = 0.0;
                const size_t state_channel = head * head_width + channel;
                for (size_t key_frame = 0U; key_frame < frames; ++key_frame) {
                    const float probability = (float)((double)score_row[key_frame] / denominator);
                    sum += (double)probability *
                           (double)value[key_frame * n_state + state_channel];
                }
                context[query_frame * n_state + state_channel] = (float)sum;
            }
        }
    }
}

int cllm_whisper_encoder_block(
    const float *input,
    size_t frames,
    const cllm_whisper_encoder_block_weights *weights,
    cllm_whisper_encoder_block_workspace *workspace,
    float *after_attention,
    float *output)
{
    size_t count;

    if (input == NULL || weights == NULL || workspace == NULL ||
        after_attention == NULL || output == NULL || frames == 0U ||
        weights->n_state == 0U || weights->n_heads == 0U || weights->n_mlp == 0U ||
        weights->n_state % weights->n_heads != 0U ||
        weights->attention_norm_weight == NULL || weights->attention_norm_bias == NULL ||
        (weights->query_weight == NULL && weights->query_weight_q4 == NULL) ||
        weights->query_bias == NULL ||
        (weights->key_weight == NULL && weights->key_weight_q4 == NULL) ||
        (weights->value_weight == NULL && weights->value_weight_q4 == NULL) ||
        weights->value_bias == NULL ||
        (weights->attention_output_weight == NULL && weights->attention_output_weight_q4 == NULL) ||
        weights->attention_output_bias == NULL || weights->mlp_norm_weight == NULL ||
        weights->mlp_norm_bias == NULL ||
        (weights->mlp_input_weight == NULL && weights->mlp_input_weight_q4 == NULL) ||
        weights->mlp_input_bias == NULL ||
        (weights->mlp_output_weight == NULL && weights->mlp_output_weight_q4 == NULL) ||
        weights->mlp_output_bias == NULL || workspace->normalized == NULL ||
        workspace->query == NULL || workspace->key == NULL || workspace->value == NULL ||
        workspace->attention_scores == NULL || workspace->attention_context == NULL ||
        workspace->mlp_hidden == NULL) {
        return -1;
    }

    layer_norm(input, frames, weights->n_state, weights->attention_norm_weight,
               weights->attention_norm_bias, workspace->normalized);
    linear_any(workspace->normalized, frames, weights->n_state, weights->query_weight,
               weights->query_weight_q4, weights->query_bias, weights->n_state, workspace->query);
    linear_any(workspace->normalized, frames, weights->n_state, weights->key_weight,
               weights->key_weight_q4, NULL, weights->n_state, workspace->key);
    linear_any(workspace->normalized, frames, weights->n_state, weights->value_weight,
               weights->value_weight_q4, weights->value_bias, weights->n_state, workspace->value);
    self_attention(frames, weights->n_state, weights->n_heads, workspace->query,
                   workspace->key, workspace->value, workspace->attention_scores,
                   workspace->attention_context);
    linear_any(workspace->attention_context, frames, weights->n_state,
               weights->attention_output_weight, weights->attention_output_weight_q4,
               weights->attention_output_bias, weights->n_state, workspace->query);

    count = frames * weights->n_state;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t index = 0U; index < count; ++index) {
        after_attention[index] = input[index] + workspace->query[index];
    }

    layer_norm(after_attention, frames, weights->n_state, weights->mlp_norm_weight,
               weights->mlp_norm_bias, workspace->normalized);
    linear_any(workspace->normalized, frames, weights->n_state, weights->mlp_input_weight,
               weights->mlp_input_weight_q4, weights->mlp_input_bias,
               weights->n_mlp, workspace->mlp_hidden);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t index = 0U; index < frames * weights->n_mlp; ++index) {
        workspace->mlp_hidden[index] = exact_gelu(workspace->mlp_hidden[index]);
    }
    linear_any(workspace->mlp_hidden, frames, weights->n_mlp, weights->mlp_output_weight,
               weights->mlp_output_weight_q4, weights->mlp_output_bias,
               weights->n_state, workspace->query);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t index = 0U; index < count; ++index) {
        output[index] = after_attention[index] + workspace->query[index];
    }
    return 0;
}
