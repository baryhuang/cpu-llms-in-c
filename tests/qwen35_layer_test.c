#include "qwen35_layer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct fixture_header {
    char magic[8];
    uint32_t version;
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

static float *read_tensor(FILE *file, size_t count)
{
    float *data = malloc(count * sizeof(float));
    if (data == NULL || fread(data, sizeof(float), count, file) != count) {
        free(data);
        return NULL;
    }
    return data;
}

static int check(const char *name, const float *actual, const float *expected, size_t count)
{
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (size_t index = 0; index < count; ++index) {
        const double difference = fabs((double)actual[index] - expected[index]);
        const double magnitude = fabs((double)expected[index]);
        max_abs = difference > max_abs ? difference : max_abs;
        if (magnitude > 0.0) {
            const double relative = difference / magnitude;
            max_rel = relative > max_rel ? relative : max_rel;
        }
    }
    const int pass = max_abs <= 1.0e-6;
    printf("boundary=%s count=%zu max_abs=%g max_rel=%g status=%s\n",
           name, count, max_abs, max_rel, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s FIXTURE\n", argv[0]);
        return 2;
    }
    FILE *file = fopen(argv[1], "rb");
    if (file == NULL) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    struct fixture_header header;
    if (fread(&header, sizeof(header), 1, file) != 1 ||
        memcmp(header.magic, "QW35LYR1", 8) != 0 || header.version != 1U) {
        fprintf(stderr, "invalid fixture header\n");
        fclose(file);
        return 2;
    }

    struct qwen35_layer_config config = {
        .sequence_length = header.sequence_length,
        .hidden_size = header.hidden_size,
        .linear_k_heads = header.linear_k_heads,
        .linear_v_heads = header.linear_v_heads,
        .linear_k_dim = header.linear_k_dim,
        .linear_v_dim = header.linear_v_dim,
        .conv_kernel = header.conv_kernel,
        .query_heads = header.query_heads,
        .kv_heads = header.kv_heads,
        .head_dim = header.head_dim,
        .rotary_dim = header.rotary_dim,
        .intermediate_size = header.intermediate_size,
        .rms_epsilon = header.rms_epsilon,
        .rope_theta = header.rope_theta,
    };
    const uint32_t seq = config.sequence_length;
    const uint32_t hidden = config.hidden_size;
    const uint32_t key_dim = config.linear_k_heads * config.linear_k_dim;
    const uint32_t value_dim = config.linear_v_heads * config.linear_v_dim;
    const uint32_t conv_dim = 2U * key_dim + value_dim;
    const uint32_t query_size = config.query_heads * config.head_dim;
    const uint32_t kv_size = config.kv_heads * config.head_dim;
    const uint32_t inter = config.intermediate_size;

    enum { TENSOR_COUNT = 43 };
    const size_t sizes[TENSOR_COUNT] = {
        (size_t)seq * hidden,
        hidden,
        (size_t)conv_dim * hidden,
        (size_t)value_dim * hidden,
        (size_t)config.linear_v_heads * hidden,
        (size_t)config.linear_v_heads * hidden,
        (size_t)conv_dim * config.conv_kernel,
        config.linear_v_heads,
        config.linear_v_heads,
        config.linear_v_dim,
        (size_t)hidden * value_dim,
        hidden,
        (size_t)inter * hidden,
        (size_t)inter * hidden,
        (size_t)hidden * inter,
        hidden,
        (size_t)2U * query_size * hidden,
        (size_t)kv_size * hidden,
        (size_t)kv_size * hidden,
        config.head_dim,
        config.head_dim,
        (size_t)hidden * query_size,
        hidden,
        (size_t)inter * hidden,
        (size_t)inter * hidden,
        (size_t)hidden * inter,
        (size_t)seq * hidden,
        (size_t)seq * conv_dim,
        (size_t)seq * config.linear_v_heads,
        (size_t)seq * value_dim,
        (size_t)config.linear_v_heads * config.linear_k_dim * config.linear_v_dim,
        (size_t)seq * value_dim,
        (size_t)seq * hidden,
        (size_t)seq * hidden,
        (size_t)seq * hidden,
        (size_t)seq * hidden,
        (size_t)seq * query_size,
        (size_t)seq * kv_size,
        (size_t)seq * query_size,
        (size_t)seq * query_size,
        (size_t)seq * hidden,
        (size_t)seq * hidden,
        (size_t)seq * hidden,
    };
    float *tensors[TENSOR_COUNT] = {0};
    for (int index = 0; index < TENSOR_COUNT; ++index) {
        tensors[index] = read_tensor(file, sizes[index]);
        if (tensors[index] == NULL) {
            fprintf(stderr, "failed to read tensor %d\n", index);
            fclose(file);
            return 2;
        }
    }
    fclose(file);

    const struct qwen35_deltanet_weights deltanet_weights = {
        .input_norm = tensors[1],
        .in_proj_qkv = tensors[2],
        .in_proj_z = tensors[3],
        .in_proj_b = tensors[4],
        .in_proj_a = tensors[5],
        .conv = tensors[6],
        .a_log = tensors[7],
        .dt_bias = tensors[8],
        .gated_norm = tensors[9],
        .out_proj = tensors[10],
        .post_norm = tensors[11],
        .gate_proj = tensors[12],
        .up_proj = tensors[13],
        .down_proj = tensors[14],
    };
    const struct qwen35_attention_weights attention_weights = {
        .input_norm = tensors[15],
        .q_proj = tensors[16],
        .k_proj = tensors[17],
        .v_proj = tensors[18],
        .q_norm = tensors[19],
        .k_norm = tensors[20],
        .o_proj = tensors[21],
        .post_norm = tensors[22],
        .gate_proj = tensors[23],
        .up_proj = tensors[24],
        .down_proj = tensors[25],
    };

    struct qwen35_deltanet_outputs deltanet = {
        .normed = malloc((size_t)seq * hidden * sizeof(float)),
        .post_conv = malloc((size_t)seq * conv_dim * sizeof(float)),
        .gate = malloc((size_t)seq * config.linear_v_heads * sizeof(float)),
        .core_out = malloc((size_t)seq * value_dim * sizeof(float)),
        .state = malloc(sizes[30] * sizeof(float)),
        .gated = malloc((size_t)seq * value_dim * sizeof(float)),
        .mixer = malloc((size_t)seq * hidden * sizeof(float)),
        .after_mixer = malloc((size_t)seq * hidden * sizeof(float)),
        .after_mlp = malloc((size_t)seq * hidden * sizeof(float)),
    };
    struct qwen35_attention_outputs attention = {
        .normed = malloc((size_t)seq * hidden * sizeof(float)),
        .query = malloc((size_t)seq * query_size * sizeof(float)),
        .key = malloc((size_t)seq * kv_size * sizeof(float)),
        .attention = malloc((size_t)seq * query_size * sizeof(float)),
        .gated_attention = malloc((size_t)seq * query_size * sizeof(float)),
        .mixer = malloc((size_t)seq * hidden * sizeof(float)),
        .after_mixer = malloc((size_t)seq * hidden * sizeof(float)),
        .after_mlp = malloc((size_t)seq * hidden * sizeof(float)),
    };
    if (!deltanet.normed || !deltanet.post_conv || !deltanet.gate || !deltanet.core_out ||
        !deltanet.state || !deltanet.gated || !deltanet.mixer || !deltanet.after_mixer ||
        !deltanet.after_mlp || !attention.normed || !attention.query || !attention.key ||
        !attention.attention || !attention.gated_attention || !attention.mixer ||
        !attention.after_mixer || !attention.after_mlp) {
        fprintf(stderr, "allocation failed\n");
        return 2;
    }

    if (qwen35_deltanet_layer_forward(&config, &deltanet_weights, tensors[0], &deltanet) != 0 ||
        qwen35_attention_layer_forward(&config, &attention_weights, deltanet.after_mlp,
                                       &attention) != 0) {
        fprintf(stderr, "forward failed\n");
        return 2;
    }

    int failures = 0;
    failures += check("l0_normed", deltanet.normed, tensors[26], sizes[26]);
    failures += check("l0_post_conv", deltanet.post_conv, tensors[27], sizes[27]);
    failures += check("l0_gate", deltanet.gate, tensors[28], sizes[28]);
    failures += check("l0_core_out", deltanet.core_out, tensors[29], sizes[29]);
    failures += check("l0_state", deltanet.state, tensors[30], sizes[30]);
    failures += check("l0_gated", deltanet.gated, tensors[31], sizes[31]);
    failures += check("l0_mixer", deltanet.mixer, tensors[32], sizes[32]);
    failures += check("l0_after_mixer", deltanet.after_mixer, tensors[33], sizes[33]);
    failures += check("l0_after_mlp", deltanet.after_mlp, tensors[34], sizes[34]);
    failures += check("l1_normed", attention.normed, tensors[35], sizes[35]);
    failures += check("l1_query", attention.query, tensors[36], sizes[36]);
    failures += check("l1_key", attention.key, tensors[37], sizes[37]);
    failures += check("l1_attention", attention.attention, tensors[38], sizes[38]);
    failures += check("l1_gated_attention", attention.gated_attention, tensors[39], sizes[39]);
    failures += check("l1_mixer", attention.mixer, tensors[40], sizes[40]);
    failures += check("l1_after_mixer", attention.after_mixer, tensors[41], sizes[41]);
    failures += check("output", attention.after_mlp, tensors[42], sizes[42]);

    printf("VERDICT: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
