#include "whisper_turbo_encoder.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef WHISPER_TURBO_TARGET_KERNELS
#include WHISPER_TURBO_TARGET_KERNELS
#endif

#if !defined(WHISPER_TURBO_HAVE_Q4_GROUP_DOT) && !defined(WHISPER_TURBO_HAVE_Q4_GEMM)
static float whisper_turbo_q4_group_dot(const unsigned char *packed, const float *values)
{
    float sum = 0.0f;
    for (size_t index = 0U; index < CLLM_WHISPER_TURBO_GROUP / 2U; ++index) {
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

#if !defined(WHISPER_TURBO_HAVE_Q8_GROUP_DOT) && !defined(WHISPER_TURBO_HAVE_Q8_GEMM)
static float whisper_turbo_q8_group_dot(const unsigned char *quantized, const float *values)
{
    const signed char *bytes = (const signed char *)(const void *)quantized;
    float sum = 0.0f;
    for (size_t index = 0U; index < CLLM_WHISPER_TURBO_GROUP; ++index)
        sum += values[index] * (float)bytes[index];
    return sum;
}
#endif

#if !defined(WHISPER_TURBO_HAVE_Q5_GROUP_DOT) && !defined(WHISPER_TURBO_HAVE_Q5_GEMM)
static float whisper_turbo_q5_group_dot(const unsigned char *packed, const float *values)
{
    /* 64 bytes of low nibbles, then a 16-byte plane of 5th bits. */
    const unsigned char *plane = packed + CLLM_WHISPER_TURBO_GROUP / 2U;
    float sum = 0.0f;
    for (size_t index = 0U; index < CLLM_WHISPER_TURBO_GROUP / 2U; ++index) {
        const unsigned char byte = packed[index];
        const size_t even = index * 2U;
        const size_t odd = even + 1U;
        int low = (byte & 0x0f) |
                  (((plane[even >> 3U] >> (even & 7U)) & 1U) << 4U);
        int high = (byte >> 4) |
                   (((plane[odd >> 3U] >> (odd & 7U)) & 1U) << 4U);
        low = low >= 16 ? low - 32 : low;
        high = high >= 16 ? high - 32 : high;
        sum += values[even] * (float)low;
        sum += values[odd] * (float)high;
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

static void linear_f32(const float *input,
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

#if !defined(WHISPER_TURBO_HAVE_Q4_GEMM) || !defined(WHISPER_TURBO_HAVE_Q8_GEMM) || \
    !defined(WHISPER_TURBO_HAVE_Q5_GEMM)
static void linear_grouped(const float *input,
                           size_t rows,
                           size_t input_width,
                           const unsigned char *weight,
                           size_t record_bytes,
                           float (*group_dot)(const unsigned char *, const float *),
                           const float *bias,
                           size_t output_width,
                           float *output)
{
    const size_t groups = input_width / CLLM_WHISPER_TURBO_GROUP;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t output_column = 0U; output_column < output_width; ++output_column) {
        const unsigned char *weight_row = weight + output_column * groups * record_bytes;
        for (size_t row = 0U; row < rows; ++row) {
            float sum = bias == NULL ? 0.0f : bias[output_column];
            const float *input_row = input + row * input_width;
            const unsigned char *record = weight_row;
            for (size_t group = 0U; group < groups; ++group) {
                sum += cllm_whisper_turbo_bf16(record) *
                       group_dot(record + 2U,
                                 input_row + group * CLLM_WHISPER_TURBO_GROUP);
                record += record_bytes;
            }
            output[row * output_width + output_column] = sum;
        }
    }
}
#endif

static void linear_any(const float *input,
                       size_t rows,
                       size_t input_width,
                       const cllm_whisper_turbo_matrix *weight,
                       const float *bias,
                       size_t output_width,
                       float *output)
{
    if (weight->q4 != NULL) {
#ifdef WHISPER_TURBO_HAVE_Q4_GEMM
        whisper_turbo_q4_gemm(weight->q4, input, rows, input_width,
                              output_width, bias, output);
#else
        linear_grouped(input, rows, input_width, weight->q4,
                       CLLM_WHISPER_TURBO_Q4_RECORD, whisper_turbo_q4_group_dot,
                       bias, output_width, output);
#endif
    } else if (weight->q8 != NULL) {
#ifdef WHISPER_TURBO_HAVE_Q8_GEMM
        whisper_turbo_q8_gemm(weight->q8, input, rows, input_width,
                              output_width, bias, output);
#else
        linear_grouped(input, rows, input_width, weight->q8,
                       CLLM_WHISPER_TURBO_Q8_RECORD, whisper_turbo_q8_group_dot,
                       bias, output_width, output);
#endif
    } else if (weight->q5 != NULL) {
#ifdef WHISPER_TURBO_HAVE_Q5_GEMM
        whisper_turbo_q5_gemm(weight->q5, input, rows, input_width,
                              output_width, bias, output);
#else
        linear_grouped(input, rows, input_width, weight->q5,
                       CLLM_WHISPER_TURBO_Q5_RECORD, whisper_turbo_q5_group_dot,
                       bias, output_width, output);
#endif
    } else {
        linear_f32(input, rows, input_width, weight->f32, bias,
                   output_width, output);
    }
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
#ifdef WHISPER_TURBO_HAVE_SELF_ATTENTION
    whisper_turbo_self_attention(frames, n_state, n_heads, query, key, value,
                                 scores, context);
#else
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
#endif
}

static int matrix_present(const cllm_whisper_turbo_matrix *matrix)
{
    return matrix->f32 != NULL || matrix->q4 != NULL ||
           matrix->q8 != NULL || matrix->q5 != NULL;
}

int cllm_whisper_turbo_encoder_block(
    const float *input,
    size_t frames,
    const cllm_whisper_turbo_encoder_block_weights *weights,
    cllm_whisper_turbo_encoder_block_workspace *workspace,
    float *after_attention,
    float *output)
{
    size_t count;

    if (input == NULL || weights == NULL || workspace == NULL ||
        after_attention == NULL || output == NULL || frames == 0U ||
        weights->n_state == 0U || weights->n_heads == 0U || weights->n_mlp == 0U ||
        weights->n_state % weights->n_heads != 0U ||
        weights->attention_norm_weight == NULL || weights->attention_norm_bias == NULL ||
        !matrix_present(&weights->query_weight) || weights->query_bias == NULL ||
        !matrix_present(&weights->key_weight) ||
        !matrix_present(&weights->value_weight) || weights->value_bias == NULL ||
        !matrix_present(&weights->attention_output_weight) ||
        weights->attention_output_bias == NULL || weights->mlp_norm_weight == NULL ||
        weights->mlp_norm_bias == NULL ||
        !matrix_present(&weights->mlp_input_weight) || weights->mlp_input_bias == NULL ||
        !matrix_present(&weights->mlp_output_weight) || weights->mlp_output_bias == NULL ||
        workspace->normalized == NULL ||
        workspace->query == NULL || workspace->key == NULL || workspace->value == NULL ||
        workspace->attention_scores == NULL || workspace->attention_context == NULL ||
        workspace->mlp_hidden == NULL) {
        return -1;
    }

    layer_norm(input, frames, weights->n_state, weights->attention_norm_weight,
               weights->attention_norm_bias, workspace->normalized);
    linear_any(workspace->normalized, frames, weights->n_state, &weights->query_weight,
               weights->query_bias, weights->n_state, workspace->query);
    linear_any(workspace->normalized, frames, weights->n_state, &weights->key_weight,
               NULL, weights->n_state, workspace->key);
    linear_any(workspace->normalized, frames, weights->n_state, &weights->value_weight,
               weights->value_bias, weights->n_state, workspace->value);
    self_attention(frames, weights->n_state, weights->n_heads, workspace->query,
                   workspace->key, workspace->value, workspace->attention_scores,
                   workspace->attention_context);
    linear_any(workspace->attention_context, frames, weights->n_state,
               &weights->attention_output_weight,
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
    linear_any(workspace->normalized, frames, weights->n_state, &weights->mlp_input_weight,
               weights->mlp_input_bias, weights->n_mlp, workspace->mlp_hidden);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t index = 0U; index < frames * weights->n_mlp; ++index) {
        workspace->mlp_hidden[index] = exact_gelu(workspace->mlp_hidden[index]);
    }
    linear_any(workspace->mlp_hidden, frames, weights->n_mlp, &weights->mlp_output_weight,
               weights->mlp_output_bias, weights->n_state, workspace->query);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t index = 0U; index < count; ++index) {
        output[index] = after_attention[index] + workspace->query[index];
    }
    return 0;
}
