#ifndef CLLM_WHISPER_TURBO_DECODER_H
#define CLLM_WHISPER_TURBO_DECODER_H

#include "whisper_turbo_quant.h"

#include <stddef.h>
#include <stdint.h>

#define CLLM_WHISPER_TURBO_DECODER_LAYERS 4U
#define CLLM_WHISPER_TURBO_DECODER_CONTEXT 448U
#define CLLM_WHISPER_TURBO_VOCABULARY 51866U

typedef struct {
    const float *self_norm_weight;
    const float *self_norm_bias;
    cllm_whisper_turbo_matrix self_query;
    const float *self_query_bias;
    cllm_whisper_turbo_matrix self_key;
    cllm_whisper_turbo_matrix self_value;
    const float *self_value_bias;
    cllm_whisper_turbo_matrix self_output;
    const float *self_output_bias;

    const float *cross_norm_weight;
    const float *cross_norm_bias;
    cllm_whisper_turbo_matrix cross_query;
    const float *cross_query_bias;
    cllm_whisper_turbo_matrix cross_key;
    cllm_whisper_turbo_matrix cross_value;
    const float *cross_value_bias;
    cllm_whisper_turbo_matrix cross_output;
    const float *cross_output_bias;

    const float *mlp_norm_weight;
    const float *mlp_norm_bias;
    cllm_whisper_turbo_matrix mlp_input;
    const float *mlp_input_bias;
    cllm_whisper_turbo_matrix mlp_output;
    const float *mlp_output_bias;
} cllm_whisper_turbo_decoder_layer_weights;

typedef struct {
    cllm_whisper_turbo_matrix token_embedding;
    const float *positions;
    const float *final_norm_weight;
    const float *final_norm_bias;
    cllm_whisper_turbo_decoder_layer_weights layers[CLLM_WHISPER_TURBO_DECODER_LAYERS];
    const uint32_t *token_offsets;
    const unsigned char *token_bytes;
    const unsigned char *token_special;
    const uint32_t *suppress_ids;
    size_t suppress_count;
} cllm_whisper_turbo_decoder_weights;

typedef struct {
    size_t audio_frames;
    size_t maximum_tokens;
    size_t token_count;
    float *self_key;
    float *self_value;
    float *cross_key;
    float *cross_value;
    float *workspace;
    size_t allocated_bytes;
} cllm_whisper_turbo_decoder_state;

typedef struct {
    double cross_cache_seconds;
    double step_seconds;
    double output_head_seconds;
    size_t token_position;
} cllm_whisper_turbo_decoder_metrics;

int cllm_whisper_turbo_decoder_state_init(const cllm_whisper_turbo_decoder_weights *weights,
                                          const float *encoder_output,
                                          size_t audio_frames,
                                          size_t maximum_tokens,
                                          cllm_whisper_turbo_decoder_state *state,
                                          cllm_whisper_turbo_decoder_metrics *metrics);
void cllm_whisper_turbo_decoder_state_free(cllm_whisper_turbo_decoder_state *state);

/*
 * Consume one token and return the greedy next token. The state owns cached
 * decoder self-attention K/V and precomputed encoder cross-attention K/V.
 * `hidden_out`, when non-null, receives the final normalized [1280] state.
 */
int cllm_whisper_turbo_decoder_step(const cllm_whisper_turbo_decoder_weights *weights,
                                    cllm_whisper_turbo_decoder_state *state,
                                    uint32_t token,
                                    uint32_t *next_token,
                                    float *next_logit,
                                    float *hidden_out,
                                    cllm_whisper_turbo_decoder_metrics *metrics);

int cllm_whisper_turbo_decoder_step_filtered(const cllm_whisper_turbo_decoder_weights *weights,
                                             cllm_whisper_turbo_decoder_state *state,
                                             uint32_t token,
                                             const unsigned char *suppressed_tokens,
                                             uint32_t *next_token,
                                             float *next_logit,
                                             float *hidden_out,
                                             cllm_whisper_turbo_decoder_metrics *metrics);

int cllm_whisper_turbo_decoder_consume(const cllm_whisper_turbo_decoder_weights *weights,
                                       cllm_whisper_turbo_decoder_state *state,
                                       uint32_t token,
                                       float *hidden_out,
                                       cllm_whisper_turbo_decoder_metrics *metrics);

#endif
