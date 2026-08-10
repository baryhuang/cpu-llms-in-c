#ifndef CPU_LLMS_QWEN35_LAYER_H
#define CPU_LLMS_QWEN35_LAYER_H

#include <stdint.h>

struct qwen35_layer_config {
    uint32_t sequence_length;
    uint32_t hidden_size;
    uint32_t linear_k_heads;
    uint32_t linear_v_heads;
    uint32_t linear_k_dim;
    uint32_t linear_v_dim;
    uint32_t conv_kernel;
    uint32_t query_heads;
    uint32_t kv_heads;
    uint32_t head_dim;
    uint32_t rotary_dim;
    uint32_t intermediate_size;
    float rms_epsilon;
    float rope_theta;
};

struct qwen35_deltanet_weights {
    const float *input_norm;
    const float *in_proj_qkv;
    const float *in_proj_z;
    const float *in_proj_b;
    const float *in_proj_a;
    const float *conv;
    const float *a_log;
    const float *dt_bias;
    const float *gated_norm;
    const float *out_proj;
    const float *post_norm;
    const float *gate_proj;
    const float *up_proj;
    const float *down_proj;
};

struct qwen35_attention_weights {
    const float *input_norm;
    const float *q_proj;
    const float *k_proj;
    const float *v_proj;
    const float *q_norm;
    const float *k_norm;
    const float *o_proj;
    const float *post_norm;
    const float *gate_proj;
    const float *up_proj;
    const float *down_proj;
};

struct qwen35_deltanet_outputs {
    float *normed;
    float *post_conv;
    float *gate;
    float *core_out;
    float *state;
    float *gated;
    float *mixer;
    float *after_mixer;
    float *after_mlp;
};

struct qwen35_attention_outputs {
    float *normed;
    float *query;
    float *key;
    float *attention;
    float *gated_attention;
    float *mixer;
    float *after_mixer;
    float *after_mlp;
};

int qwen35_deltanet_layer_forward(const struct qwen35_layer_config *config,
                                  const struct qwen35_deltanet_weights *weights,
                                  const float *input,
                                  const struct qwen35_deltanet_outputs *outputs);

int qwen35_attention_layer_forward(const struct qwen35_layer_config *config,
                                   const struct qwen35_attention_weights *weights,
                                   const float *input,
                                   const struct qwen35_attention_outputs *outputs);

#endif
