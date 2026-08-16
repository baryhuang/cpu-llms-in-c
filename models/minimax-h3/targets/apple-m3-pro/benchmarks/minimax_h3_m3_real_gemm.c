#include "minimax_h3_m3_gemm.h"
#include "minimax_h3_q4_layer.h"
#include "minimax_h3_remote_safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float bfloat_value(uint16_t bits) {
    uint32_t expanded = (uint32_t)bits << 16u;
    float value;
    memcpy(&value, &expanded, sizeof(value));
    return value;
}

static uint16_t bfloat_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t rounding = UINT32_C(0x7fff) + ((bits >> 16u) & 1u);
    return (uint16_t)((bits + rounding) >> 16u);
}

static float reference(const minimax_h3_q4_projection *projection,
                       const uint16_t *input,
                       size_t row) {
    float sum = 0.0f;
    for (size_t group = 0u; group < projection->groups_per_row; ++group) {
        size_t block = row * projection->groups_per_row + group;
        float scale = bfloat_value(projection->scales[block]);
        float bias = bfloat_value(projection->biases[block]);
        for (size_t within = 0u; within < 64u; ++within) {
            uint8_t packed = projection->weight[block * 32u + within / 2u];
            unsigned quant = within % 2u == 0u ? packed & 15u : packed >> 4u;
            float weight = bfloat_value(bfloat_bits(
                scale * (float)quant + bias));
            sum += bfloat_value(input[group * 64u + within]) * weight;
        }
    }
    return sum;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s TRANSFORMER_URL METALLIB\n", argv[0]);
        return 2;
    }
    minimax_h3_remote_safetensors file;
    minimax_h3_q4_layer layer;
    char error[512];
    if (minimax_h3_remote_safetensors_open(argv[1], &file, error,
                                            sizeof(error)) != 0 ||
        minimax_h3_q4_layer_load(&file, 0u, &layer, error, sizeof(error)) != 0) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    const minimax_h3_q4_projection *projection = &layer.projections[2];
    size_t batch = 11u;
    uint16_t *input = malloc(batch * projection->input_columns * 2u);
    uint16_t *output = malloc(batch * projection->output_rows * 2u);
    if (input == NULL || output == NULL) return 1;
    for (size_t index = 0u; index < batch * projection->input_columns; ++index)
        input[index] = bfloat_bits(
            (float)((int)(index % 29u) - 14) / 32.0f);
    double gpu_ms = 0.0;
    if (minimax_h3_m3_run_q4_projection_bf16(
            argv[2], projection, input, batch, output, &gpu_ms, error,
            sizeof(error)) != 0) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    float maximum = 0.0f;
    printf("{\n  \"real_weights\": true,\n  \"layer\": 0,\n"
           "  \"projection\": \"%s\",\n  \"batch_rows\": %zu,\n"
           "  \"gpu_ms\": %.6f,\n  \"first_8\": [\n",
           projection->suffix, batch, gpu_ms);
    for (size_t row = 0u; row < 8u; ++row) {
        float expected = reference(projection, input, row);
        float actual = bfloat_value(output[row]);
        float difference = fabsf(expected - actual);
        if (difference > maximum) maximum = difference;
        printf("    {\"row\": %zu, \"reference\": %.9g, "
               "\"metal\": %.9g, \"abs_error\": %.9g}%s\n",
               row, expected, actual, difference, row == 7u ? "" : ",");
    }
    printf("  ],\n  \"max_abs_error_first_8\": %.9g\n}\n", maximum);
    free(input);
    free(output);
    minimax_h3_q4_layer_free(&layer);
    minimax_h3_remote_safetensors_close(&file);
    return 0;
}
