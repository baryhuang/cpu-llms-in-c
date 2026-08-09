#include "cpu_llms/gemma4_layer.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIXTURE_HEADER_BYTES 60U
#define FIXTURE_TENSOR_COUNT 28U

struct fixture_cursor {
    const unsigned char *data;
    size_t size;
    size_t offset;
};

static uint32_t read_u32(const unsigned char *bytes)
{
    return (uint32_t)bytes[0] |
           (uint32_t)bytes[1] << 8U |
           (uint32_t)bytes[2] << 16U |
           (uint32_t)bytes[3] << 24U;
}

static float read_f32(const unsigned char *bytes)
{
    const uint32_t bits = read_u32(bytes);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static const float *take_floats(struct fixture_cursor *cursor, size_t count)
{
    const size_t bytes = count * sizeof(float);
    const float *result;

    if (count > SIZE_MAX / sizeof(float) || cursor->offset > cursor->size ||
        bytes > cursor->size - cursor->offset) {
        return NULL;
    }
    result = (const float *)(const void *)(cursor->data + cursor->offset);
    cursor->offset += bytes;
    return result;
}

static unsigned char *read_file(const char *path, size_t *size)
{
    FILE *file = fopen(path, "rb");
    unsigned char *data;
    long length;

    if (file == NULL) {
        fprintf(stderr, "error: cannot open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "error: cannot size %s\n", path);
        fclose(file);
        return NULL;
    }

    data = malloc((size_t)length);
    if (data == NULL || fread(data, 1U, (size_t)length, file) != (size_t)length) {
        fprintf(stderr, "error: cannot read %s\n", path);
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *size = (size_t)length;
    return data;
}

static int compare_boundary(const char *name, const float *actual, const float *expected,
                            size_t count)
{
    double maximum_absolute = 0.0;
    double maximum_relative = 0.0;
    int passed = 1;

    for (size_t index = 0; index < count; ++index) {
        const double absolute = fabs((double)actual[index] - (double)expected[index]);
        const double denominator = fmax(fabs((double)expected[index]), 1.0e-12);
        const double relative = absolute / denominator;
        const double tolerance = 2.0e-5 + 2.0e-5 * fabs((double)expected[index]);

        if (absolute > maximum_absolute) {
            maximum_absolute = absolute;
        }
        if (relative > maximum_relative) {
            maximum_relative = relative;
        }
        if (!isfinite(actual[index]) || absolute > tolerance) {
            passed = 0;
        }
    }

    printf("boundary=%s count=%zu max_abs=%.9g max_rel=%.9g status=%s\n",
           name, count, maximum_absolute, maximum_relative, passed ? "PASS" : "FAIL");
    return passed;
}

int main(int argc, char **argv)
{
    unsigned char *file_data;
    size_t file_size;
    struct fixture_cursor cursor;
    struct cllm_gemma4_layer_config config;
    struct cllm_gemma4_layer_weights weights;
    struct cllm_gemma4_layer_outputs outputs;
    const float *input;
    const float *per_layer_input;
    const float *expected[10];
    float *actual[10] = {0};
    float *workspace = NULL;
    size_t workspace_count;
    size_t hidden_count;
    size_t query_count;
    size_t key_value_count;
    size_t output_counts[10];
    uint32_t tensor_count;
    uint32_t endian_test = 1U;
    int passed = 1;
    int result;

    if (argc != 2) {
        fprintf(stderr, "usage: %s FIXTURE\n", argv[0]);
        return 2;
    }
    if (sizeof(float) != 4U || *(const unsigned char *)(const void *)&endian_test != 1U) {
        fprintf(stderr, "error: fixture runner requires little-endian IEEE-754 float32\n");
        return 2;
    }

    file_data = read_file(argv[1], &file_size);
    if (file_data == NULL) {
        return 2;
    }
    if (file_size < FIXTURE_HEADER_BYTES || memcmp(file_data, "G4LYR001", 8U) != 0 ||
        read_u32(file_data + 8U) != 1U) {
        fprintf(stderr, "error: invalid Gemma 4 layer fixture header\n");
        free(file_data);
        return 2;
    }

    const size_t sequence_length = read_u32(file_data + 12U);
    config.hidden_size = read_u32(file_data + 16U);
    config.query_heads = read_u32(file_data + 20U);
    config.kv_heads = read_u32(file_data + 24U);
    config.head_dim = read_u32(file_data + 28U);
    config.intermediate_size = read_u32(file_data + 32U);
    config.ple_size = read_u32(file_data + 36U);
    config.sliding_window = read_u32(file_data + 40U);
    tensor_count = read_u32(file_data + 44U);
    config.rms_epsilon = read_f32(file_data + 48U);
    config.rope_theta = read_f32(file_data + 52U);
    config.layer_scalar = read_f32(file_data + 56U);

    if (tensor_count != FIXTURE_TENSOR_COUNT || sequence_length == 0U ||
        config.hidden_size == 0U || config.query_heads == 0U || config.kv_heads == 0U ||
        config.head_dim == 0U || config.intermediate_size == 0U || config.ple_size == 0U) {
        fprintf(stderr, "error: invalid fixture dimensions\n");
        free(file_data);
        return 2;
    }

    hidden_count = sequence_length * config.hidden_size;
    query_count = sequence_length * config.query_heads * config.head_dim;
    key_value_count = sequence_length * config.kv_heads * config.head_dim;
    cursor.data = file_data;
    cursor.size = file_size;
    cursor.offset = FIXTURE_HEADER_BYTES;

    input = take_floats(&cursor, hidden_count);
    per_layer_input = take_floats(&cursor, sequence_length * config.ple_size);
    weights.input_norm = take_floats(&cursor, config.hidden_size);
    weights.q_proj = take_floats(&cursor,
                                 (size_t)config.query_heads * config.head_dim * config.hidden_size);
    weights.k_proj = take_floats(&cursor,
                                 (size_t)config.kv_heads * config.head_dim * config.hidden_size);
    weights.v_proj = take_floats(&cursor,
                                 (size_t)config.kv_heads * config.head_dim * config.hidden_size);
    weights.q_norm = take_floats(&cursor, config.head_dim);
    weights.k_norm = take_floats(&cursor, config.head_dim);
    weights.o_proj = take_floats(&cursor,
                                 (size_t)config.hidden_size * config.query_heads * config.head_dim);
    weights.post_attention_norm = take_floats(&cursor, config.hidden_size);
    weights.pre_feedforward_norm = take_floats(&cursor, config.hidden_size);
    weights.gate_proj = take_floats(&cursor,
                                    (size_t)config.intermediate_size * config.hidden_size);
    weights.up_proj = take_floats(&cursor,
                                  (size_t)config.intermediate_size * config.hidden_size);
    weights.down_proj = take_floats(&cursor,
                                    (size_t)config.hidden_size * config.intermediate_size);
    weights.post_feedforward_norm = take_floats(&cursor, config.hidden_size);
    weights.ple_gate = take_floats(&cursor, (size_t)config.ple_size * config.hidden_size);
    weights.ple_projection = take_floats(&cursor, (size_t)config.hidden_size * config.ple_size);
    weights.post_ple_norm = take_floats(&cursor, config.hidden_size);

    output_counts[0] = hidden_count;
    output_counts[1] = query_count;
    output_counts[2] = key_value_count;
    output_counts[3] = key_value_count;
    output_counts[4] = hidden_count;
    output_counts[5] = hidden_count;
    output_counts[6] = hidden_count;
    output_counts[7] = hidden_count;
    output_counts[8] = hidden_count;
    output_counts[9] = hidden_count;

    for (size_t index = 0; index < 10U; ++index) {
        expected[index] = take_floats(&cursor, output_counts[index]);
        actual[index] = calloc(output_counts[index], sizeof(float));
        if (expected[index] == NULL || actual[index] == NULL) {
            fprintf(stderr, "error: fixture tensor allocation failed\n");
            passed = 0;
            goto cleanup;
        }
    }

    if (input == NULL || per_layer_input == NULL || weights.post_ple_norm == NULL ||
        cursor.offset != cursor.size) {
        fprintf(stderr, "error: fixture payload length mismatch\n");
        passed = 0;
        goto cleanup;
    }

    outputs.normalized_input = actual[0];
    outputs.query = actual[1];
    outputs.key = actual[2];
    outputs.value = actual[3];
    outputs.attention = actual[4];
    outputs.after_attention = actual[5];
    outputs.mlp = actual[6];
    outputs.after_mlp = actual[7];
    outputs.ple = actual[8];
    outputs.output = actual[9];

    workspace_count = cllm_gemma4_layer_workspace_floats(&config, sequence_length);
    workspace = calloc(workspace_count, sizeof(float));
    if (workspace == NULL) {
        fprintf(stderr, "error: workspace allocation failed\n");
        passed = 0;
        goto cleanup;
    }

    result = cllm_gemma4_layer_forward(&config, &weights, input, per_layer_input,
                                       sequence_length, workspace, workspace_count - 1U, &outputs);
    if (result != -2) {
        fprintf(stderr, "error: undersized workspace was not rejected: %d\n", result);
        passed = 0;
        goto cleanup;
    }

    result = cllm_gemma4_layer_forward(&config, &weights, input, per_layer_input,
                                       sequence_length, workspace, workspace_count, &outputs);
    if (result != 0) {
        fprintf(stderr, "error: layer execution failed: %d\n", result);
        passed = 0;
        goto cleanup;
    }

    const char *names[10] = {
        "normalized_input", "query", "key", "value", "attention",
        "after_attention", "mlp", "after_mlp", "ple", "output",
    };
    for (size_t index = 0; index < 10U; ++index) {
        passed &= compare_boundary(names[index], actual[index], expected[index], output_counts[index]);
    }

cleanup:
    for (size_t index = 0; index < 10U; ++index) {
        free(actual[index]);
    }
    free(workspace);
    free(file_data);

    printf("VERDICT: %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
