#ifndef CPU_LLMS_GEMMA4_LAYER_H
#define CPU_LLMS_GEMMA4_LAYER_H

#include <stddef.h>
#include <stdint.h>

struct cllm_gemma4_layer_config {
    uint32_t hidden_size;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t intermediate_size;
    uint32_t ple_size;
    uint32_t sliding_window;
    float rms_epsilon;
    float rope_theta;
    float layer_scalar;
};

struct cllm_gemma4_layer_weights {
    const float *input_norm;
    const float *q_proj;
    const float *k_proj;
    const float *v_proj;
    const float *q_norm;
    const float *k_norm;
    const float *o_proj;
    const float *post_attention_norm;
    const float *pre_feedforward_norm;
    const float *gate_proj;
    const float *up_proj;
    const float *down_proj;
    const float *post_feedforward_norm;
    const float *ple_gate;
    const float *ple_projection;
    const float *post_ple_norm;
};

struct cllm_gemma4_layer_outputs {
    float *normalized_input;
    float *query;
    float *key;
    float *value;
    float *attention;
    float *after_attention;
    float *mlp;
    float *after_mlp;
    float *ple;
    float *output;
};

size_t cllm_gemma4_layer_workspace_floats(const struct cllm_gemma4_layer_config *config,
                                          size_t sequence_length);

int cllm_gemma4_layer_forward(const struct cllm_gemma4_layer_config *config,
                              const struct cllm_gemma4_layer_weights *weights,
                              const float *input,
                              const float *per_layer_input,
                              size_t sequence_length,
                              float *workspace,
                              size_t workspace_floats,
                              const struct cllm_gemma4_layer_outputs *outputs);

#endif
