#ifndef CLLM_WHISPER_SMALL_ENCODER_H
#define CLLM_WHISPER_SMALL_ENCODER_H

#include <stddef.h>

typedef struct {
    size_t n_state;
    size_t n_heads;
    size_t n_mlp;
    const float *attention_norm_weight;
    const float *attention_norm_bias;
    const float *query_weight;
    const unsigned char *query_weight_q4;
    const float *query_bias;
    const float *key_weight;
    const unsigned char *key_weight_q4;
    const float *value_weight;
    const unsigned char *value_weight_q4;
    const float *value_bias;
    const float *attention_output_weight;
    const unsigned char *attention_output_weight_q4;
    const float *attention_output_bias;
    const float *mlp_norm_weight;
    const float *mlp_norm_bias;
    const float *mlp_input_weight;
    const unsigned char *mlp_input_weight_q4;
    const float *mlp_input_bias;
    const float *mlp_output_weight;
    const unsigned char *mlp_output_weight_q4;
    const float *mlp_output_bias;
} cllm_whisper_encoder_block_weights;

typedef struct {
    float *normalized;
    float *query;
    float *key;
    float *value;
    float *attention_scores;
    float *attention_context;
    float *mlp_hidden;
} cllm_whisper_encoder_block_workspace;

/*
 * Portable scalar OpenAI Whisper encoder block.
 *
 * input, after_attention and output are [frames, n_state]. Linear weights
 * use PyTorch [output, input] layout. attention_scores is a reusable
 * [frames] softmax row. All memory is caller-owned.
 */
int cllm_whisper_encoder_block(
    const float *input,
    size_t frames,
    const cllm_whisper_encoder_block_weights *weights,
    cllm_whisper_encoder_block_workspace *workspace,
    float *after_attention,
    float *output);

#endif
