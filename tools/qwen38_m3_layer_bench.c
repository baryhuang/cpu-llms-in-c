#include "qwen38_m3.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned parse_iterations(const char *text) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        value == 0 || value > 1000) {
        fprintf(stderr, "invalid iteration count: %s\n", text);
        exit(2);
    }
    return (unsigned)value;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        fprintf(stderr,
                "usage: %s IMAGE.q38m3 METALLIB [MEASURED] [WARMUP]\n",
                argv[0]);
        return 2;
    }
    unsigned measured = argc > 3 ? parse_iterations(argv[3]) : 20;
    unsigned warmup = argc > 4 ? parse_iterations(argv[4]) : 2;
    qwen38_m3_layer_result result;
    char error[512];
    int status = qwen38_m3_run_layer_benchmark(
        argv[2], argv[1], warmup, measured, &result, error, sizeof(error));
    if (status != 0) {
        fprintf(stderr, "qwen38 layer benchmark failed (%d): %s\n",
                status, error);
        return status;
    }
    printf("{\n");
    printf("  \"schema\": 1,\n");
    printf("  \"scope\": \"Qwen3.8-27B real layer-0 one-token decode; not full-model tokens/s\",\n");
    printf("  \"device\": \"%s\",\n", result.device_name);
    printf("  \"weight_source\": \"%s\",\n", result.weight_source);
    printf("  \"layer_type\": \"linear_attention\",\n");
    printf("  \"warmup_iterations\": %u,\n", result.warmup_iterations);
    printf("  \"measured_iterations\": %u,\n", result.measured_iterations);
    printf("  \"duration_ms_per_layer\": %.6f,\n", result.layer_ms);
    printf("  \"effective_weight_gbps\": %.6f,\n",
           result.effective_weight_gbps);
    printf("  \"mapped_image_bytes\": %zu,\n", result.mapped_image_bytes);
    printf("  \"recurrent_state_bytes\": %zu,\n",
           result.recurrent_state_bytes);
    printf("  \"convolution_state_bytes\": %zu,\n",
           result.convolution_state_bytes);
    printf("  \"metal_owned_buffer_bytes\": %zu,\n",
           result.metal_owned_buffer_bytes);
    printf("  \"footprint_before_bytes\": %zu,\n",
           result.footprint_before_bytes);
    printf("  \"footprint_peak_bytes\": %zu,\n",
           result.footprint_peak_bytes);
    printf("  \"max_abs_error_output_first_8_vs_c\": %.9g,\n",
           result.max_abs_error_output_first_8);
    printf("  \"max_abs_error_recurrent_state_vs_c\": %.9g,\n",
           result.max_abs_error_recurrent_state);
    printf("  \"max_abs_error_convolution_state_vs_c\": %.9g,\n",
           result.max_abs_error_convolution_state);
    printf("  \"c_reference_output_first_8\": [");
    for (unsigned index = 0; index < 8; ++index) {
        printf("%s%.9g", index == 0 ? "" : ", ",
               result.reference_output_first_8[index]);
    }
    printf("],\n  \"metal_output_first_8\": [");
    for (unsigned index = 0; index < 8; ++index) {
        printf("%s%.9g", index == 0 ? "" : ", ",
               result.metal_output_first_8[index]);
    }
    printf("]\n}\n");
    return 0;
}
