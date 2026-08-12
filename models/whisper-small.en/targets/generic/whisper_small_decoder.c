#define _POSIX_C_SOURCE 200809L

#include "whisper_small_decoder.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { WIDTH = 768, HEADS = 12, HEAD_WIDTH = 64, MLP = 3072,
       Q4_GROUP = 128, Q4_RECORD = 66 };

static double monotonic_seconds(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0.0;
    return (double)now.tv_sec + (double)now.tv_nsec * 1.0e-9;
}

static float bf16_to_float(const unsigned char *bytes)
{
    uint32_t bits = ((uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8U)) << 16U;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float q4_dot(const unsigned char *records, const float *input, size_t columns)
{
    float sum = 0.0f;
    for (size_t group = 0U; group < columns / Q4_GROUP; ++group) {
        const float scale = bf16_to_float(records);
        const unsigned char *packed = records + 2U;
        for (size_t index = 0U; index < Q4_GROUP / 2U; ++index) {
            int low = packed[index] & 15;
            int high = packed[index] >> 4;
            low = low >= 8 ? low - 16 : low;
            high = high >= 8 ? high - 16 : high;
            sum += scale * input[group * Q4_GROUP + index * 2U] * (float)low;
            sum += scale * input[group * Q4_GROUP + index * 2U + 1U] * (float)high;
        }
        records += Q4_RECORD;
    }
    return sum;
}

static void q4_dequantize(const unsigned char *records, float *output, size_t columns)
{
    for (size_t group = 0U; group < columns / Q4_GROUP; ++group) {
        const float scale = bf16_to_float(records);
        const unsigned char *packed = records + 2U;
        float *destination = output + group * Q4_GROUP;
        for (size_t index = 0U; index < Q4_GROUP / 2U; ++index) {
            int low = packed[index] & 15;
            int high = packed[index] >> 4;
            low = low >= 8 ? low - 16 : low;
            high = high >= 8 ? high - 16 : high;
            destination[index * 2U] = scale * (float)low;
            destination[index * 2U + 1U] = scale * (float)high;
        }
        records += Q4_RECORD;
    }
}

static void matrix_vector(const cllm_whisper_matrix *matrix,
                          const float *input,
                          const float *bias,
                          float *output)
{
    for (size_t row = 0U; row < matrix->rows; ++row) {
        double sum = bias == NULL ? 0.0 : bias[row];
        if (matrix->q4 != NULL) {
            const size_t row_bytes = (matrix->columns / Q4_GROUP) * Q4_RECORD;
            sum += q4_dot(matrix->q4 + row * row_bytes, input, matrix->columns);
        } else {
            const float *weights = matrix->f32 + row * matrix->columns;
            for (size_t column = 0U; column < matrix->columns; ++column)
                sum += (double)weights[column] * input[column];
        }
        output[row] = (float)sum;
    }
}

static void matrix_rows(const cllm_whisper_matrix *matrix,
                        const float *input,
                        size_t rows,
                        const float *bias,
                        float *output)
{
    for (size_t row = 0U; row < rows; ++row)
        matrix_vector(matrix, input + row * matrix->columns, bias,
                      output + row * matrix->rows);
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
    memset(output, 0, WIDTH * sizeof(float));
    for (size_t head = 0U; head < HEADS; ++head) {
        const size_t offset = head * HEAD_WIDTH;
        float maximum = -INFINITY;
        double denominator = 0.0;
        for (size_t frame = 0U; frame < frames; ++frame) {
            double sum = 0.0;
            for (size_t channel = 0U; channel < HEAD_WIDTH; ++channel)
                sum += (double)query[offset + channel] *
                       key[frame * WIDTH + offset + channel];
            scores[frame] = (float)sum * 0.125f;
            if (scores[frame] > maximum) maximum = scores[frame];
        }
        for (size_t frame = 0U; frame < frames; ++frame) {
            scores[frame] = expf(scores[frame] - maximum);
            denominator += scores[frame];
        }
        for (size_t frame = 0U; frame < frames; ++frame) {
            const float probability = (float)(scores[frame] / denominator);
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

int cllm_whisper_decoder_state_init(const cllm_whisper_decoder_weights *weights,
                                     const float *encoder_output,
                                     size_t audio_frames,
                                     size_t maximum_tokens,
                                     cllm_whisper_decoder_state *state,
                                     cllm_whisper_decoder_metrics *metrics)
{
    size_t cross_count, self_count, workspace_count, total_count;
    float *cursor;
    double started;
    if (weights == NULL || encoder_output == NULL || state == NULL || metrics == NULL ||
        audio_frames == 0U || audio_frames > 1500U || maximum_tokens == 0U ||
        maximum_tokens > CLLM_WHISPER_SMALL_DECODER_CONTEXT)
        return -1;
    memset(state, 0, sizeof(*state));
    memset(metrics, 0, sizeof(*metrics));
    cross_count = CLLM_WHISPER_SMALL_DECODER_LAYERS * audio_frames * WIDTH;
    self_count = CLLM_WHISPER_SMALL_DECODER_LAYERS * maximum_tokens * WIDTH;
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
    for (size_t layer = 0U; layer < CLLM_WHISPER_SMALL_DECODER_LAYERS; ++layer) {
        matrix_rows(&weights->layers[layer].cross_key, encoder_output, audio_frames,
                    NULL, state->cross_key + layer * audio_frames * WIDTH);
        matrix_rows(&weights->layers[layer].cross_value, encoder_output, audio_frames,
                    weights->layers[layer].cross_value_bias,
                    state->cross_value + layer * audio_frames * WIDTH);
    }
    metrics->cross_cache_seconds = monotonic_seconds() - started;
    return 0;
}

void cllm_whisper_decoder_state_free(cllm_whisper_decoder_state *state)
{
    if (state == NULL) return;
    free(state->cross_key);
    memset(state, 0, sizeof(*state));
}

int cllm_whisper_decoder_step(const cllm_whisper_decoder_weights *weights,
                              cllm_whisper_decoder_state *state,
                              uint32_t token,
                              uint32_t *next_token,
                              float *next_logit,
                              float *hidden_out,
                              cllm_whisper_decoder_metrics *metrics)
{
    float *hidden, *norm, *query, *key, *value, *context, *temporary, *hidden_mlp, *scores;
    size_t position;
    double started;
    if (weights == NULL || state == NULL || next_token == NULL || next_logit == NULL ||
        metrics == NULL || token >= CLLM_WHISPER_SMALL_VOCABULARY ||
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
    } else {
        const size_t row_bytes = (WIDTH / Q4_GROUP) * Q4_RECORD;
        const unsigned char *row = weights->token_embedding.q4 + (size_t)token * row_bytes;
        q4_dequantize(row, hidden, WIDTH);
    }
    for (size_t index = 0U; index < WIDTH; ++index)
        hidden[index] += weights->positions[position * WIDTH + index];

    for (size_t layer = 0U; layer < CLLM_WHISPER_SMALL_DECODER_LAYERS; ++layer) {
        const cllm_whisper_decoder_layer_weights *w = &weights->layers[layer];
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

    started = monotonic_seconds();
    *next_token = 0U;
    *next_logit = -INFINITY;
    if (weights->token_embedding.f32 != NULL) {
        for (size_t row = 0U; row < CLLM_WHISPER_SMALL_VOCABULARY; ++row) {
            double logit = 0.0;
            const float *embedding = weights->token_embedding.f32 + row * WIDTH;
            for (size_t column = 0U; column < WIDTH; ++column)
                logit += (double)embedding[column] * norm[column];
            if ((float)logit > *next_logit) {
                *next_logit = (float)logit;
                *next_token = (uint32_t)row;
            }
        }
    } else {
        const size_t row_bytes = (WIDTH / Q4_GROUP) * Q4_RECORD;
        for (size_t row = 0U; row < CLLM_WHISPER_SMALL_VOCABULARY; ++row) {
            const float logit = q4_dot(weights->token_embedding.q4 + row * row_bytes, norm, WIDTH);
            if (logit > *next_logit) {
                *next_logit = logit;
                *next_token = (uint32_t)row;
            }
        }
    }
    metrics->output_head_seconds = monotonic_seconds() - started;
    state->token_count += 1U;
    return 0;
}
