#include "minimax_h3_m3_gemm.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned long parse_count(const char *text, unsigned long maximum) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0u ||
        value > maximum) {
        fprintf(stderr, "invalid positive count: %s\n", text);
        exit(2);
    }
    return value;
}

int main(int argc, char **argv) {
    minimax_h3_m3_gemm_result result;
    char error[512];
    size_t batch;
    unsigned measured;
    unsigned warmup;
    int status;

    if (argc < 2 || argc > 5) {
        fprintf(stderr,
                "usage: %s METALLIB [BATCH=15485] [MEASURED=3] [WARMUP=1]\n",
                argv[0]);
        return 2;
    }
    batch = argc > 2 ? parse_count(argv[2], 15485u) : 15485u;
    measured = argc > 3 ? (unsigned)parse_count(argv[3], 100u) : 3u;
    warmup = argc > 4 ? (unsigned)parse_count(argv[4], 100u) : 1u;
    status = minimax_h3_m3_run_gemm_benchmark(
        argv[1], batch, warmup, measured, &result, error, sizeof(error));
    if (status != 0) {
        fprintf(stderr, "MiniMax-H3 GEMM benchmark failed (%d): %s\n", status,
                error);
        return status;
    }
    printf("{\n");
    printf("  \"schema\": 1,\n");
    printf("  \"scope\": \"synthetic H3 5376-to-14336 W4 projection primitive\",\n");
    printf("  \"device\": \"%s\",\n", result.device_name);
    printf("  \"batch_rows\": %zu,\n", result.batch_rows);
    printf("  \"warmup_iterations\": %u,\n", result.warmup_iterations);
    printf("  \"measured_iterations\": %u,\n", result.measured_iterations);
    printf("  \"gpu_ms\": %.6f,\n", result.gpu_ms);
    printf("  \"wall_ms\": %.6f,\n", result.wall_ms);
    printf("  \"effective_tflops\": %.6f,\n", result.effective_tflops);
    printf("  \"weight_bytes\": %zu,\n", result.weight_bytes);
    printf("  \"metadata_bytes\": %zu,\n", result.metadata_bytes);
    printf("  \"input_bytes\": %zu,\n", result.input_bytes);
    printf("  \"output_bytes\": %zu,\n", result.output_bytes);
    printf("  \"metal_owned_buffer_bytes\": %zu,\n",
           result.metal_owned_buffer_bytes);
    printf("  \"footprint_before_bytes\": %zu,\n",
           result.footprint_before_bytes);
    printf("  \"footprint_peak_bytes\": %zu,\n", result.footprint_peak_bytes);
    printf("  \"max_abs_error_first_8\": %.9g,\n",
           result.max_abs_error_first_8);
    printf("  \"c_reference_first_8\": [");
    for (unsigned index = 0u; index < 8u; ++index)
        printf("%s%.9g", index == 0u ? "" : ", ",
               result.reference_first_8[index]);
    printf("],\n  \"metal_first_8\": [");
    for (unsigned index = 0u; index < 8u; ++index)
        printf("%s%.9g", index == 0u ? "" : ", ", result.metal_first_8[index]);
    printf("]\n}\n");
    return 0;
}
