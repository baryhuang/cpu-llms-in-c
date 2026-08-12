#include "whisper_small_encoder.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HEADER_BYTES 32U

static uint32_t read_u32(const unsigned char *bytes)
{
    return (uint32_t)bytes[0] |
           (uint32_t)bytes[1] << 8U |
           (uint32_t)bytes[2] << 16U |
           (uint32_t)bytes[3] << 24U;
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
        fclose(file);
        return NULL;
    }
    data = malloc((size_t)length);
    if (data == NULL || fread(data, 1U, (size_t)length, file) != (size_t)length) {
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *size = (size_t)length;
    return data;
}

static void report_boundary(const char *name,
                            const float *actual,
                            const float *expected,
                            size_t count,
                            int *passed)
{
    double maximum_absolute = 0.0;
    double maximum_relative = 0.0;

    for (size_t index = 0U; index < count; ++index) {
        const double absolute = fabs((double)actual[index] - (double)expected[index]);
        const double relative = absolute / fmax(fabs((double)expected[index]), 1.0e-12);
        const double tolerance = 5.0e-6 + 5.0e-5 * fabs((double)expected[index]);
        if (absolute > maximum_absolute) maximum_absolute = absolute;
        if (relative > maximum_relative) maximum_relative = relative;
        if (!isfinite(actual[index]) || absolute > tolerance) *passed = 0;
    }
    printf("boundary=%s count=%zu max_abs=%.9g max_rel=%.9g status=%s\n",
           name, count, maximum_absolute, maximum_relative, *passed ? "PASS" : "FAIL");
}

int main(int argc, char **argv)
{
    unsigned char *data;
    size_t size;
    size_t offset = HEADER_BYTES;
    size_t frames;
    size_t n_state;
    size_t n_heads;
    size_t n_mlp;
    size_t state_count;
    cllm_whisper_encoder_block_weights weights;
    cllm_whisper_encoder_block_workspace workspace;
    const float *input;
    const float *expected_after_attention;
    const float *expected_output;
    float *after_attention;
    float *output;
    int passed = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s FIXTURE\n", argv[0]);
        return 2;
    }
    data = read_file(argv[1], &size);
    if (data == NULL || size < HEADER_BYTES || memcmp(data, "WHENCB01", 8U) != 0 ||
        read_u32(data + 8U) != 1U || read_u32(data + 28U) != 18U) {
        fprintf(stderr, "error: invalid Whisper encoder-block fixture\n");
        free(data);
        return 2;
    }
    frames = read_u32(data + 12U);
    n_state = read_u32(data + 16U);
    n_heads = read_u32(data + 20U);
    n_mlp = read_u32(data + 24U);
    state_count = frames * n_state;

#define TAKE_FLOATS(name, count) \
    do { \
        if ((count) > (size - offset) / sizeof(float)) { \
            fprintf(stderr, "error: truncated Whisper encoder-block fixture\n"); \
            free(data); \
            return 2; \
        } \
        name = (const float *)(const void *)(data + offset); \
        offset += (count) * sizeof(float); \
    } while (0)
    TAKE_FLOATS(input, state_count);
    TAKE_FLOATS(weights.attention_norm_weight, n_state);
    TAKE_FLOATS(weights.attention_norm_bias, n_state);
    TAKE_FLOATS(weights.query_weight, n_state * n_state);
    TAKE_FLOATS(weights.query_bias, n_state);
    TAKE_FLOATS(weights.key_weight, n_state * n_state);
    TAKE_FLOATS(weights.value_weight, n_state * n_state);
    TAKE_FLOATS(weights.value_bias, n_state);
    TAKE_FLOATS(weights.attention_output_weight, n_state * n_state);
    TAKE_FLOATS(weights.attention_output_bias, n_state);
    TAKE_FLOATS(weights.mlp_norm_weight, n_state);
    TAKE_FLOATS(weights.mlp_norm_bias, n_state);
    TAKE_FLOATS(weights.mlp_input_weight, n_mlp * n_state);
    TAKE_FLOATS(weights.mlp_input_bias, n_mlp);
    TAKE_FLOATS(weights.mlp_output_weight, n_state * n_mlp);
    TAKE_FLOATS(weights.mlp_output_bias, n_state);
    TAKE_FLOATS(expected_after_attention, state_count);
    TAKE_FLOATS(expected_output, state_count);
#undef TAKE_FLOATS
    if (offset != size) {
        fprintf(stderr, "error: extra Whisper encoder-block fixture bytes\n");
        free(data);
        return 2;
    }
    weights.n_state = n_state;
    weights.n_heads = n_heads;
    weights.n_mlp = n_mlp;

    workspace.normalized = malloc(state_count * sizeof(float));
    workspace.query = malloc(state_count * sizeof(float));
    workspace.key = malloc(state_count * sizeof(float));
    workspace.value = malloc(state_count * sizeof(float));
    workspace.attention_scores = malloc(frames * sizeof(float));
    workspace.attention_context = malloc(state_count * sizeof(float));
    workspace.mlp_hidden = malloc(frames * n_mlp * sizeof(float));
    after_attention = malloc(state_count * sizeof(float));
    output = malloc(state_count * sizeof(float));
    if (workspace.normalized == NULL || workspace.query == NULL || workspace.key == NULL ||
        workspace.value == NULL || workspace.attention_scores == NULL ||
        workspace.attention_context == NULL || workspace.mlp_hidden == NULL ||
        after_attention == NULL || output == NULL ||
        cllm_whisper_encoder_block(input, frames, &weights, &workspace,
                                   after_attention, output) != 0) {
        fprintf(stderr, "error: Whisper encoder-block execution failed\n");
        free(data);
        return 2;
    }

    report_boundary("whisper_encoder_after_attention", after_attention,
                    expected_after_attention, state_count, &passed);
    report_boundary("whisper_encoder_output", output, expected_output,
                    state_count, &passed);
    printf("VERDICT: %s\n", passed ? "PASS" : "FAIL");

    free(workspace.normalized);
    free(workspace.query);
    free(workspace.key);
    free(workspace.value);
    free(workspace.attention_scores);
    free(workspace.attention_context);
    free(workspace.mlp_hidden);
    free(after_attention);
    free(output);
    free(data);
    return passed ? 0 : 1;
}
