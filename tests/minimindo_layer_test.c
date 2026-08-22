#include "minimindo_layer.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { char *name; uint32_t count; float *values; } fixture_tensor;

static fixture_tensor *find_tensor(fixture_tensor *tensors, uint32_t count,
                                   const char *name)
{
    for (uint32_t index = 0; index < count; ++index)
        if (strcmp(tensors[index].name, name) == 0) return &tensors[index];
    return NULL;
}

static int compare(const char *name, const float *actual,
                   const fixture_tensor *expected)
{
    double maximum = 0.0;
    for (uint32_t index = 0; index < expected->count; ++index) {
        const double delta = fabs((double)actual[index] - expected->values[index]);
        if (delta > maximum) maximum = delta;
    }
    printf("%s max_abs=%.9g\n", name, maximum);
    return maximum <= 3.0e-5 ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc != 2) { fprintf(stderr, "usage: %s FIXTURE\n", argv[0]); return 2; }
    FILE *file = fopen(argv[1], "rb");
    if (file == NULL) return 3;
    unsigned char magic[8];
    uint32_t version, tensor_count;
    minimindo_layer_config config = {0};
    if (fread(magic, 1, 8, file) != 8 || fread(&version, 4, 1, file) != 1 ||
        fread(&config.sequence_length, 4, 1, file) != 1 ||
        fread(&config.hidden_size, 4, 1, file) != 1 ||
        fread(&config.query_heads, 4, 1, file) != 1 ||
        fread(&config.kv_heads, 4, 1, file) != 1 ||
        fread(&config.head_dim, 4, 1, file) != 1 ||
        fread(&config.intermediate_size, 4, 1, file) != 1 ||
        fread(&config.start_position, 4, 1, file) != 1 ||
        fread(&config.rms_epsilon, 4, 1, file) != 1 ||
        fread(&config.rope_theta, 4, 1, file) != 1 ||
        fread(&tensor_count, 4, 1, file) != 1 ||
        memcmp(magic, "MMOLYR1\0", 8) != 0 || version != 1) return 4;
    fixture_tensor *tensors = calloc(tensor_count, sizeof(*tensors));
    if (tensors == NULL) return 5;
    for (uint32_t index = 0; index < tensor_count; ++index) {
        uint32_t name_length;
        if (fread(&name_length, 4, 1, file) != 1 ||
            fread(&tensors[index].count, 4, 1, file) != 1) return 6;
        tensors[index].name = calloc(name_length + 1U, 1);
        tensors[index].values = calloc(tensors[index].count, sizeof(float));
        if (tensors[index].name == NULL || tensors[index].values == NULL ||
            fread(tensors[index].name, 1, name_length, file) != name_length ||
            fread(tensors[index].values, 4, tensors[index].count, file) != tensors[index].count)
            return 7;
    }
    fclose(file);
#define T(name) find_tensor(tensors, tensor_count, name)->values
    minimindo_layer_weights weights = {
        T("input_norm"), T("q_proj"), T("k_proj"), T("v_proj"),
        T("q_norm"), T("k_norm"), T("o_proj"), T("post_attention_norm"),
        T("gate_proj"), T("up_proj"), T("down_proj")
    };
    const uint32_t rows = config.sequence_length, hidden = config.hidden_size;
    const uint32_t kv_size = config.kv_heads * config.head_dim;
    const size_t workspace_count = minimindo_layer_workspace_floats(&config);
    float *workspace = calloc(workspace_count, sizeof(float));
    minimindo_layer_outputs outputs = {
        calloc((size_t)rows * hidden, sizeof(float)),
        calloc((size_t)rows * hidden, sizeof(float)),
        calloc((size_t)rows * kv_size, sizeof(float)),
        calloc((size_t)rows * kv_size, sizeof(float)),
        calloc((size_t)rows * hidden, sizeof(float)),
        calloc((size_t)rows * hidden, sizeof(float)),
        calloc((size_t)rows * hidden, sizeof(float)),
        calloc((size_t)rows * hidden, sizeof(float))
    };
    int failed = minimindo_layer_forward(&config, &weights, T("input"), workspace,
                                         workspace_count, &outputs) != 0;
#define CHECK(field, expected) failed |= compare(#field, outputs.field, find_tensor(tensors, tensor_count, expected))
    CHECK(normalized_input, "expected_normed");
    CHECK(query, "expected_query");
    CHECK(key, "expected_key");
    CHECK(value, "expected_value");
    CHECK(attention, "expected_attention");
    CHECK(after_attention, "expected_after_attention");
    CHECK(normalized_mlp, "expected_normalized_mlp");
    CHECK(output, "expected_output");
    printf("gate=%s\n", failed ? "FAIL" : "PASS");
    return failed ? 1 : 0;
}
