#ifndef CLLM_WHISPER_TURBO_ENCODER_H
#define CLLM_WHISPER_TURBO_ENCODER_H

#include "whisper_turbo_quant.h"

#include <stddef.h>

typedef struct {
    size_t n_state;
    size_t n_heads;
    size_t n_mlp;
    const float *attention_norm_weight;
    const float *attention_norm_bias;
    cllm_whisper_turbo_matrix query_weight;
    const float *query_bias;
    cllm_whisper_turbo_matrix key_weight;
    cllm_whisper_turbo_matrix value_weight;
    const float *value_bias;
    cllm_whisper_turbo_matrix attention_output_weight;
    const float *attention_output_bias;
    const float *mlp_norm_weight;
    const float *mlp_norm_bias;
    cllm_whisper_turbo_matrix mlp_input_weight;
    const float *mlp_input_bias;
    cllm_whisper_turbo_matrix mlp_output_weight;
    const float *mlp_output_bias;
} cllm_whisper_turbo_encoder_block_weights;

typedef struct {
    float *normalized;
    float *query;
    float *key;
    float *value;
    float *attention_scores;
    float *attention_context;
    float *mlp_hidden;
} cllm_whisper_turbo_encoder_block_workspace;

/*
 * Portable scalar OpenAI Whisper encoder block for large-v3-turbo.
 *
 * input, after_attention and output are [frames, n_state]. Linear weights
 * use PyTorch [output, input] layout and may be F32 or grouped Q4/Q8/Q5.
 * attention_scores is a reusable [frames] softmax row. All memory is
 * caller-owned.
 */
int cllm_whisper_turbo_encoder_block(
    const float *input,
    size_t frames,
    const cllm_whisper_turbo_encoder_block_weights *weights,
    cllm_whisper_turbo_encoder_block_workspace *workspace,
    float *after_attention,
    float *output);

#endif
