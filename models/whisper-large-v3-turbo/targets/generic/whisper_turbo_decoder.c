#define _POSIX_C_SOURCE 200809L

#include "whisper_turbo_decoder.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef WHISPER_TURBO_TARGET_DECODER_KERNELS
#include WHISPER_TURBO_TARGET_DECODER_KERNELS
#endif

enum { WIDTH = 1280, HEADS = 20, HEAD_WIDTH = 64, MLP = 5120 };

static double monotonic_seconds(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0.0;
    return (double)now.tv_sec + (double)now.tv_nsec * 1.0e-9;
}

static float q4_dot(const unsigned char *records, const float *input, size_t columns)
{
#ifdef WHISPER_TURBO_HAVE_DECODER_Q4_DOT
    return whisper_turbo_decoder_q4_dot(records, input, columns);
#else
    float sum = 0.0f;
    for (size_t group = 0U; group < columns / CLLM_WHISPER_TURBO_GROUP; ++group) {
        const float scale = cllm_whisper_turbo_bf16(records);
        const unsigned char *packed = records + 2U;
        for (size_t index = 0U; index < CLLM_WHISPER_TURBO_GROUP / 2U; ++index) {
            int low = packed[index] & 15;
            int high = packed[index] >> 4;
            low = low >= 8 ? low - 16 : low;
            high = high >= 8 ? high - 16 : high;
            sum += scale * input[group * CLLM_WHISPER_TURBO_GROUP + index * 2U] * (float)low;
            sum += scale * input[group * CLLM_WHISPER_TURBO_GROUP + index * 2U + 1U] * (float)high;
        }
        records += CLLM_WHISPER_TURBO_Q4_RECORD;
    }
    return sum;
#endif
}

static void q4_dequantize(const unsigned char *records, float *output, size_t columns)
{
    for (size_t group = 0U; group < columns / CLLM_WHISPER_TURBO_GROUP; ++group) {
        const float scale = cllm_whisper_turbo_bf16(records);
        const unsigned char *packed = records + 2U;
        float *destination = output + group * CLLM_WHISPER_TURBO_GROUP;
        for (size_t index = 0U; index < CLLM_WHISPER_TURBO_GROUP / 2U; ++index) {
            int low = packed[index] & 15;
            int high = packed[index] >> 4;
            low = low >= 8 ? low - 16 : low;
            high = high >= 8 ? high - 16 : high;
            destination[index * 2U] = scale * (float)low;
            destination[index * 2U + 1U] = scale * (float)high;
        }
        records += CLLM_WHISPER_TURBO_Q4_RECORD;
    }
}

static float q8_dot(const unsigned char *records, const float *input, size_t columns)
{
#ifdef WHISPER_TURBO_HAVE_DECODER_Q8_DOT
    return whisper_turbo_decoder_q8_dot(records, input, columns);
#else
    float sum = 0.0f;
    for (size_t group = 0U; group < columns / CLLM_WHISPER_TURBO_GROUP; ++group) {
        const float scale = cllm_whisper_turbo_bf16(records);
        const signed char *quantized = (const signed char *)(const void *)(records + 2U);
        for (size_t index = 0U; index < CLLM_WHISPER_TURBO_GROUP; ++index)
            sum += scale * input[group * CLLM_WHISPER_TURBO_GROUP + index] * quantized[index];
        records += CLLM_WHISPER_TURBO_Q8_RECORD;
    }
    return sum;
#endif
}

static void q8_dequantize(const unsigned char *records, float *output, size_t columns)
{
    for (size_t group = 0U; group < columns / CLLM_WHISPER_TURBO_GROUP; ++group) {
        const float scale = cllm_whisper_turbo_bf16(records);
        const signed char *quantized = (const signed char *)(const void *)(records + 2U);
        for (size_t index = 0U; index < CLLM_WHISPER_TURBO_GROUP; ++index)
            output[group * CLLM_WHISPER_TURBO_GROUP + index] = scale * quantized[index];
        records += CLLM_WHISPER_TURBO_Q8_RECORD;
    }
}

static float q5_dot(const unsigned char *records, const float *input, size_t columns)
{
#ifdef WHISPER_TURBO_HAVE_DECODER_Q5_DOT
    return whisper_turbo_decoder_q5_dot(records, input, columns);
#else
    float sum = 0.0f;
    for (size_t group = 0U; group < columns / CLLM_WHISPER_TURBO_GROUP; ++group) {
        const float scale = cllm_whisper_turbo_bf16(records);
        const unsigned char *packed = records + 2U;
        const unsigned char *plane = packed + CLLM_WHISPER_TURBO_GROUP / 2U;
        const float *values = input + group * CLLM_WHISPER_TURBO_GROUP;
        for (size_t index = 0U; index < CLLM_WHISPER_TURBO_GROUP / 2U; ++index) {
            const size_t even = index * 2U;
            const size_t odd = even + 1U;
            int low = (packed[index] & 0x0f) |
                      (((plane[even >> 3U] >> (even & 7U)) & 1U) << 4U);
            int high = (packed[index] >> 4) |
                       (((plane[odd >> 3U] >> (odd & 7U)) & 1U) << 4U);
            low = low >= 16 ? low - 32 : low;
            high = high >= 16 ? high - 32 : high;
            sum += scale * values[even] * (float)low;
            sum += scale * values[odd] * (float)high;
        }
        records += CLLM_WHISPER_TURBO_Q5_RECORD;
    }
    return sum;
#endif
}

static void q5_dequantize(const unsigned char *records, float *output, size_t columns)
{
    for (size_t group = 0U; group < columns / CLLM_WHISPER_TURBO_GROUP; ++group) {
        const float scale = cllm_whisper_turbo_bf16(records);
        const unsigned char *packed = records + 2U;
        const unsigned char *plane = packed + CLLM_WHISPER_TURBO_GROUP / 2U;
        float *destination = output + group * CLLM_WHISPER_TURBO_GROUP;
        for (size_t index = 0U; index < CLLM_WHISPER_TURBO_GROUP / 2U; ++index) {
            const size_t even = index * 2U;
            const size_t odd = even + 1U;
            int low = (packed[index] & 0x0f) |
                      (((plane[even >> 3U] >> (even & 7U)) & 1U) << 4U);
            int high = (packed[index] >> 4) |
                       (((plane[odd >> 3U] >> (odd & 7U)) & 1U) << 4U);
            low = low >= 16 ? low - 32 : low;
            high = high >= 16 ? high - 32 : high;
            destination[even] = scale * (float)low;
            destination[odd] = scale * (float)high;
        }
        records += CLLM_WHISPER_TURBO_Q5_RECORD;
    }
}

static float output_embedding_logit(const cllm_whisper_turbo_matrix *embedding,
                                    size_t row,
                                    const float *hidden)
{
    if (embedding->f32 != NULL) {
        double sum = 0.0;
        const float *weights = embedding->f32 + row * WIDTH;
        for (size_t column = 0U; column < WIDTH; ++column)
            sum += (double)weights[column] * hidden[column];
        return (float)sum;
    }
    if (embedding->q4 != NULL) {
        const size_t row_bytes = (WIDTH / CLLM_WHISPER_TURBO_GROUP) * CLLM_WHISPER_TURBO_Q4_RECORD;
        return q4_dot(embedding->q4 + row * row_bytes, hidden, WIDTH);
    }
    if (embedding->q5 != NULL) {
        const size_t row_bytes = (WIDTH / CLLM_WHISPER_TURBO_GROUP) * CLLM_WHISPER_TURBO_Q5_RECORD;
        return q5_dot(embedding->q5 + row * row_bytes, hidden, WIDTH);
    }
    const size_t row_bytes = (WIDTH / CLLM_WHISPER_TURBO_GROUP) * CLLM_WHISPER_TURBO_Q8_RECORD;
    return q8_dot(embedding->q8 + row * row_bytes, hidden, WIDTH);
}

static float matrix_vector_row(const cllm_whisper_turbo_matrix *matrix,
                               const float *input,
                               const float *bias,
                               size_t row)
{
    double sum = bias == NULL ? 0.0 : bias[row];
    if (matrix->q4 != NULL) {
        const size_t row_bytes = (matrix->columns / CLLM_WHISPER_TURBO_GROUP) *
                                 CLLM_WHISPER_TURBO_Q4_RECORD;
        sum += q4_dot(matrix->q4 + row * row_bytes, input, matrix->columns);
    } else if (matrix->q8 != NULL) {
        const size_t row_bytes = (matrix->columns / CLLM_WHISPER_TURBO_GROUP) *
                                 CLLM_WHISPER_TURBO_Q8_RECORD;
        sum += q8_dot(matrix->q8 + row * row_bytes, input, matrix->columns);
    } else if (matrix->q5 != NULL) {
        const size_t row_bytes = (matrix->columns / CLLM_WHISPER_TURBO_GROUP) *
                                 CLLM_WHISPER_TURBO_Q5_RECORD;
        sum += q5_dot(matrix->q5 + row * row_bytes, input, matrix->columns);
    } else {
        const float *weights = matrix->f32 + row * matrix->columns;
        for (size_t column = 0U; column < matrix->columns; ++column)
            sum += (double)weights[column] * input[column];
    }
    return (float)sum;
}

static void matrix_vector(const cllm_whisper_turbo_matrix *matrix,
                          const float *input,
                          const float *bias,
                          float *output)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t row = 0U; row < matrix->rows; ++row)
        output[row] = matrix_vector_row(matrix, input, bias, row);
}

static void matrix_rows(const cllm_whisper_turbo_matrix *matrix,
                        const float *input,
                        size_t rows,
                        const float *bias,
                        float *output)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t row = 0U; row < rows; ++row)
        for (size_t output_row = 0U; output_row < matrix->rows; ++output_row)
            output[row * matrix->rows + output_row] = matrix_vector_row(
                matrix, input + row * matrix->columns, bias, output_row);
}

static void layer_norm(const float *input, const float *weight,
                       const float *bias, float *output)
{
    double sum = 0.0, square_sum = 0.0;
    for (size_t index = 0U; index < WIDTH; ++index) {
        sum += input[index];
        square_sum += (double)input[index] * input[index];
    }
    const double mean = sum / WIDTH;
    double variance = square_sum / WIDTH - mean * mean;
    if (variance < 0.0) variance = 0.0;
    const float inverse = 1.0f / sqrtf((float)variance + 1.0e-5f);
    for (size_t index = 0U; index < WIDTH; ++index)
        output[index] = (input[index] - (float)mean) * inverse * weight[index] + bias[index];
}

static void attention(const float *query,
                      const float *key,
                      const float *value,
                      size_t frames,
                      float *scores,
                      float *output)
{
#ifdef _OPENMP
    (void)scores;
#endif
    memset(output, 0, WIDTH * sizeof(float));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t head = 0U; head < HEADS; ++head) {
        const size_t offset = head * HEAD_WIDTH;
        float maximum = -INFINITY;
        double denominator = 0.0;
#ifdef _OPENMP
        float private_scores[1500];
        float *head_scores = private_scores;
#else
        float *head_scores = scores;
#endif
        for (size_t frame = 0U; frame < frames; ++frame) {
            double sum = 0.0;
            for (size_t channel = 0U; channel < HEAD_WIDTH; ++channel)
                sum += (double)query[offset + channel] *
                       key[frame * WIDTH + offset + channel];
            head_scores[frame] = (float)sum * 0.125f;
            if (head_scores[frame] > maximum) maximum = head_scores[frame];
        }
        for (size_t frame = 0U; frame < frames; ++frame) {
            head_scores[frame] = expf(head_scores[frame] - maximum);
            denominator += head_scores[frame];
        }
        for (size_t frame = 0U; frame < frames; ++frame) {
            const float probability = (float)(head_scores[frame] / denominator);
            for (size_t channel = 0U; channel < HEAD_WIDTH; ++channel)
                output[offset + channel] += probability *
                    value[frame * WIDTH + offset + channel];
        }
    }
}

static float exact_gelu(float value)
{
    return 0.5f * value * (1.0f + erff(value * 0.70710678118654752440f));
}

int cllm_whisper_turbo_decoder_state_init(const cllm_whisper_turbo_decoder_weights *weights,
                                          const float *encoder_output,
                                          size_t audio_frames,
                                          size_t maximum_tokens,
                                          cllm_whisper_turbo_decoder_state *state,
                                          cllm_whisper_turbo_decoder_metrics *metrics)
{
    size_t cross_count, self_count, workspace_count, total_count;
    float *cursor;
    double started;
    if (weights == NULL || encoder_output == NULL || state == NULL || metrics == NULL ||
        audio_frames == 0U || audio_frames > 1500U || maximum_tokens == 0U ||
        maximum_tokens > CLLM_WHISPER_TURBO_DECODER_CONTEXT)
        return -1;
    memset(state, 0, sizeof(*state));
    memset(metrics, 0, sizeof(*metrics));
    cross_count = CLLM_WHISPER_TURBO_DECODER_LAYERS * audio_frames * WIDTH;
    self_count = CLLM_WHISPER_TURBO_DECODER_LAYERS * maximum_tokens * WIDTH;
    workspace_count = MLP + WIDTH * 8U +
        (audio_frames > maximum_tokens ? audio_frames : maximum_tokens);
    total_count = cross_count * 2U + self_count * 2U + workspace_count;
    cursor = calloc(total_count, sizeof(float));
    if (cursor == NULL) return -1;
    state->cross_key = cursor; cursor += cross_count;
    state->cross_value = cursor; cursor += cross_count;
    state->self_key = cursor; cursor += self_count;
    state->self_value = cursor; cursor += self_count;
    state->workspace = cursor;
    state->audio_frames = audio_frames;
    state->maximum_tokens = maximum_tokens;
    state->allocated_bytes = total_count * sizeof(float);

    started = monotonic_seconds();
    for (size_t layer = 0U; layer < CLLM_WHISPER_TURBO_DECODER_LAYERS; ++layer) {
        matrix_rows(&weights->layers[layer].cross_key, encoder_output, audio_frames,
                    NULL, state->cross_key + layer * audio_frames * WIDTH);
        matrix_rows(&weights->layers[layer].cross_value, encoder_output, audio_frames,
                    weights->layers[layer].cross_value_bias,
                    state->cross_value + layer * audio_frames * WIDTH);
    }
    metrics->cross_cache_seconds = monotonic_seconds() - started;
    return 0;
}

void cllm_whisper_turbo_decoder_state_free(cllm_whisper_turbo_decoder_state *state)
{
    if (state == NULL) return;
    free(state->cross_key);
    memset(state, 0, sizeof(*state));
}

static int decoder_step_internal(const cllm_whisper_turbo_decoder_weights *weights,
                                 cllm_whisper_turbo_decoder_state *state,
                                 uint32_t token,
                                 const unsigned char *suppressed_tokens,
                                 uint32_t *next_token,
                                 float *next_logit,
                                 float *hidden_out,
                                 cllm_whisper_turbo_decoder_metrics *metrics)
{
    float *hidden, *norm, *query, *key, *value, *context, *temporary, *hidden_mlp, *scores;
    size_t position;
    double started;
    if (weights == NULL || state == NULL ||
        metrics == NULL || token >= CLLM_WHISPER_TURBO_VOCABULARY ||
        state->token_count >= state->maximum_tokens || state->workspace == NULL)
        return -1;
    position = state->token_count;
    hidden = state->workspace;
    norm = hidden + WIDTH;
    query = norm + WIDTH;
    key = query + WIDTH;
    value = key + WIDTH;
    context = value + WIDTH;
    temporary = context + WIDTH;
    hidden_mlp = temporary + WIDTH;
    scores = hidden_mlp + MLP;
    memset(metrics, 0, sizeof(*metrics));
    metrics->token_position = position;
    started = monotonic_seconds();

    if (weights->token_embedding.f32 != NULL) {
        memcpy(hidden, weights->token_embedding.f32 + (size_t)token * WIDTH,
               WIDTH * sizeof(float));
    } else if (weights->token_embedding.q4 != NULL) {
        const size_t row_bytes = (WIDTH / CLLM_WHISPER_TURBO_GROUP) * CLLM_WHISPER_TURBO_Q4_RECORD;
        const unsigned char *row = weights->token_embedding.q4 + (size_t)token * row_bytes;
        q4_dequantize(row, hidden, WIDTH);
    } else if (weights->token_embedding.q5 != NULL) {
        const size_t row_bytes = (WIDTH / CLLM_WHISPER_TURBO_GROUP) * CLLM_WHISPER_TURBO_Q5_RECORD;
        const unsigned char *row = weights->token_embedding.q5 + (size_t)token * row_bytes;
        q5_dequantize(row, hidden, WIDTH);
    } else {
        const size_t row_bytes = (WIDTH / CLLM_WHISPER_TURBO_GROUP) * CLLM_WHISPER_TURBO_Q8_RECORD;
        const unsigned char *row = weights->token_embedding.q8 + (size_t)token * row_bytes;
        q8_dequantize(row, hidden, WIDTH);
    }
    for (size_t index = 0U; index < WIDTH; ++index)
        hidden[index] += weights->positions[position * WIDTH + index];

    for (size_t layer = 0U; layer < CLLM_WHISPER_TURBO_DECODER_LAYERS; ++layer) {
        const cllm_whisper_turbo_decoder_layer_weights *w = &weights->layers[layer];
        float *self_key = state->self_key +
            (layer * state->maximum_tokens + position) * WIDTH;
        float *self_value = state->self_value +
            (layer * state->maximum_tokens + position) * WIDTH;
        layer_norm(hidden, w->self_norm_weight, w->self_norm_bias, norm);
        matrix_vector(&w->self_query, norm, w->self_query_bias, query);
        matrix_vector(&w->self_key, norm, NULL, key);
        matrix_vector(&w->self_value, norm, w->self_value_bias, value);
        memcpy(self_key, key, WIDTH * sizeof(float));
        memcpy(self_value, value, WIDTH * sizeof(float));
        attention(query,
                  state->self_key + layer * state->maximum_tokens * WIDTH,
                  state->self_value + layer * state->maximum_tokens * WIDTH,
                  position + 1U, scores, context);
        matrix_vector(&w->self_output, context, w->self_output_bias, temporary);
        for (size_t index = 0U; index < WIDTH; ++index) hidden[index] += temporary[index];

        layer_norm(hidden, w->cross_norm_weight, w->cross_norm_bias, norm);
        matrix_vector(&w->cross_query, norm, w->cross_query_bias, query);
        attention(query,
                  state->cross_key + layer * state->audio_frames * WIDTH,
                  state->cross_value + layer * state->audio_frames * WIDTH,
                  state->audio_frames, scores, context);
        matrix_vector(&w->cross_output, context, w->cross_output_bias, temporary);
        for (size_t index = 0U; index < WIDTH; ++index) hidden[index] += temporary[index];

        layer_norm(hidden, w->mlp_norm_weight, w->mlp_norm_bias, norm);
        matrix_vector(&w->mlp_input, norm, w->mlp_input_bias, hidden_mlp);
        for (size_t index = 0U; index < MLP; ++index) hidden_mlp[index] = exact_gelu(hidden_mlp[index]);
        matrix_vector(&w->mlp_output, hidden_mlp, w->mlp_output_bias, temporary);
        for (size_t index = 0U; index < WIDTH; ++index) hidden[index] += temporary[index];
    }
    layer_norm(hidden, weights->final_norm_weight, weights->final_norm_bias, norm);
    if (hidden_out != NULL) memcpy(hidden_out, norm, WIDTH * sizeof(float));
    metrics->step_seconds = monotonic_seconds() - started;

    if (next_token == NULL || next_logit == NULL) {
        state->token_count += 1U;
        return 0;
    }

    started = monotonic_seconds();
    *next_token = 0U;
    *next_logit = -INFINITY;
#ifdef _OPENMP
#pragma omp parallel
    {
        uint32_t local_token = 0U;
        float local_logit = -INFINITY;
#pragma omp for nowait schedule(static)
        for (size_t row = 0U; row < CLLM_WHISPER_TURBO_VOCABULARY; ++row) {
            if (suppressed_tokens != NULL && suppressed_tokens[row]) continue;
            const float logit = output_embedding_logit(
                &weights->token_embedding, row, norm);
            if (logit > local_logit ||
                (logit == local_logit && row < local_token)) {
                local_logit = logit;
                local_token = (uint32_t)row;
            }
        }
#pragma omp critical
        if (local_logit > *next_logit ||
            (local_logit == *next_logit && local_token < *next_token)) {
            *next_logit = local_logit;
            *next_token = local_token;
        }
    }
#else
    for (size_t row = 0U; row < CLLM_WHISPER_TURBO_VOCABULARY; ++row) {
        if (suppressed_tokens != NULL && suppressed_tokens[row]) continue;
        const float logit = output_embedding_logit(
            &weights->token_embedding, row, norm);
        if (logit > *next_logit) {
            *next_logit = logit;
            *next_token = (uint32_t)row;
        }
    }
#endif
    metrics->output_head_seconds = monotonic_seconds() - started;
    state->token_count += 1U;
    return 0;
}

int cllm_whisper_turbo_decoder_step(const cllm_whisper_turbo_decoder_weights *weights,
                                    cllm_whisper_turbo_decoder_state *state,
                                    uint32_t token,
                                    uint32_t *next_token,
                                    float *next_logit,
                                    float *hidden_out,
                                    cllm_whisper_turbo_decoder_metrics *metrics)
{
    return decoder_step_internal(weights, state, token, NULL, next_token,
                                 next_logit, hidden_out, metrics);
}

int cllm_whisper_turbo_decoder_consume(const cllm_whisper_turbo_decoder_weights *weights,
                                       cllm_whisper_turbo_decoder_state *state,
                                       uint32_t token,
                                       float *hidden_out,
                                       cllm_whisper_turbo_decoder_metrics *metrics)
{
    return decoder_step_internal(weights, state, token, NULL, NULL, NULL,
                                 hidden_out, metrics);
}

int cllm_whisper_turbo_decoder_step_filtered(const cllm_whisper_turbo_decoder_weights *weights,
                                             cllm_whisper_turbo_decoder_state *state,
                                             uint32_t token,
                                             const unsigned char *suppressed_tokens,
                                             uint32_t *next_token,
                                             float *next_logit,
                                             float *hidden_out,
                                             cllm_whisper_turbo_decoder_metrics *metrics)
{
    return decoder_step_internal(weights, state, token, suppressed_tokens,
                                 next_token, next_logit, hidden_out, metrics);
}
