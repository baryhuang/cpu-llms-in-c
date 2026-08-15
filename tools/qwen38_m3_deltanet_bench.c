#include "qwen38_m3.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned parse_iterations(const char *text) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0 || value > 10000) {
        fprintf(stderr, "invalid iteration count: %s\n", text);
        exit(2);
    }
    return (unsigned)value;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s METALLIB [MEASURED] [WARMUP]\n", argv[0]);
        return 2;
    }
    unsigned measured = argc > 2 ? parse_iterations(argv[2]) : 100;
    unsigned warmup = argc > 3 ? parse_iterations(argv[3]) : 5;
    qwen38_m3_deltanet_core_result result;
    char error[512];
    int status = qwen38_m3_run_deltanet_core_benchmark(
        argv[1], warmup, measured, &result, error, sizeof(error));
    if (status != 0) {
        fprintf(stderr, "qwen38 DeltaNet benchmark failed (%d): %s\n",
                status, error);
        return status;
    }
    printf("{\n");
    printf("  \"schema\": 1,\n");
    printf("  \"scope\": \"Qwen3.8-27B one-token DeltaNet recurrent core; projections excluded\",\n");
    printf("  \"device\": \"%s\",\n", result.device_name);
    printf("  \"shape\": {\"key_heads\": %d, \"value_heads\": %d, \"key_dim\": %d, \"value_dim\": %d},\n",
           QWEN38_DELTA_KEY_HEADS, QWEN38_DELTA_VALUE_HEADS,
           QWEN38_DELTA_HEAD_SIZE, QWEN38_DELTA_HEAD_SIZE);
    printf("  \"state_bytes\": %zu,\n", result.state_bytes);
    printf("  \"warmup_iterations\": %u,\n", result.warmup_iterations);
    printf("  \"measured_iterations\": %u,\n", result.measured_iterations);
    printf("  \"direct_device_ms\": %.6f,\n", result.direct_device_ms);
    printf("  \"float2_vectorized_ms\": %.6f,\n", result.float2_vectorized_ms);
    printf("  \"speedup\": %.6f,\n", result.speedup);
    printf("  \"direct_effective_state_gbps\": %.6f,\n", result.direct_state_gbps);
    printf("  \"vectorized_effective_state_gbps\": %.6f,\n",
           result.vectorized_state_gbps);
    printf("  \"footprint_before_bytes\": %zu,\n", result.footprint_before_bytes);
    printf("  \"footprint_peak_bytes\": %zu,\n", result.footprint_peak_bytes);
    printf("  \"max_abs_error_both_outputs_vs_c\": %.9g,\n",
           result.max_abs_error_output);
    printf("  \"max_abs_error_both_states_vs_c\": %.9g,\n",
           result.max_abs_error_state);
    printf("  \"c_reference_output_first_8\": [");
    for (unsigned index = 0; index < 8; ++index) {
        printf("%s%.9g", index == 0 ? "" : ", ",
               result.reference_output_first_8[index]);
    }
    printf("],\n  \"vectorized_gpu_output_first_8\": [");
    for (unsigned index = 0; index < 8; ++index) {
        printf("%s%.9g", index == 0 ? "" : ", ",
               result.vectorized_output_first_8[index]);
    }
    printf("]\n}\n");
    return 0;
}
