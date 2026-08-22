#ifndef LLM_IN_C_MINIMINDO_LAYER_H
#define LLM_IN_C_MINIMINDO_LAYER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t sequence_length;
    uint32_t hidden_size;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t intermediate_size;
    uint32_t start_position;
    float rms_epsilon;
    float rope_theta;
} minimindo_layer_config;

typedef struct {
    const float *input_norm;
    const float *q_proj;
    const float *k_proj;
    const float *v_proj;
    const float *q_norm;
    const float *k_norm;
    const float *o_proj;
    const float *post_attention_norm;
    const float *gate_proj;
    const float *up_proj;
    const float *down_proj;
} minimindo_layer_weights;

typedef struct {
    float *normalized_input;
    float *query;
    float *key;
    float *value;
    float *attention;
    float *after_attention;
    float *normalized_mlp;
    float *output;
} minimindo_layer_outputs;

size_t minimindo_layer_workspace_floats(const minimindo_layer_config *config);
int minimindo_layer_forward(const minimindo_layer_config *config,
                            const minimindo_layer_weights *weights,
                            const float *input, float *workspace,
                            size_t workspace_floats,
                            const minimindo_layer_outputs *outputs);

#endif
