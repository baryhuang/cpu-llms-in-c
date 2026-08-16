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
    if (argc < 5 || argc > 7) {
        fprintf(stderr,
                "usage: %s METALLIB COLUMNS ROWS BATCH [MEASURED=3] "
                "[WARMUP=1]\n",
                argv[0]);
        return 2;
    }
    size_t columns = parse_count(argv[2], 65536u);
    size_t rows = parse_count(argv[3], 65536u);
    size_t batch = parse_count(argv[4], 65536u);
    unsigned measured = argc > 5 ? (unsigned)parse_count(argv[5], 100u) : 3u;
    unsigned warmup = argc > 6 ? (unsigned)parse_count(argv[6], 100u) : 1u;
    minimax_h3_m3_gemm_result result;
    char error[512];
    int status = minimax_h3_m3_run_q8_gemm_benchmark(
        argv[1], columns, rows, batch, warmup, measured, &result, error,
        sizeof(error));
    if (status != 0) {
        fprintf(stderr, "MiniMax-H3 Q8 GEMM benchmark failed (%d): %s\n",
                status, error);
        return status;
    }
    printf("{\n"
           "  \"schema\": 1,\n"
           "  \"scope\": \"synthetic conditioner affine-Q8 projection\",\n"
           "  \"device\": \"%s\",\n"
           "  \"input_columns\": %zu,\n"
           "  \"output_rows\": %zu,\n"
           "  \"batch_rows\": %zu,\n"
           "  \"warmup_iterations\": %u,\n"
           "  \"measured_iterations\": %u,\n"
           "  \"gpu_ms\": %.6f,\n"
           "  \"wall_ms\": %.6f,\n"
           "  \"effective_tflops\": %.6f,\n"
           "  \"weight_bytes\": %zu,\n"
           "  \"metadata_bytes\": %zu,\n"
           "  \"metal_owned_buffer_bytes\": %zu,\n"
           "  \"max_abs_error_first_8\": %.9g\n"
           "}\n",
           result.device_name, result.input_columns, result.output_rows,
           result.batch_rows, result.warmup_iterations,
           result.measured_iterations, result.gpu_ms, result.wall_ms,
           result.effective_tflops, result.weight_bytes,
           result.metadata_bytes, result.metal_owned_buffer_bytes,
           result.max_abs_error_first_8);
    return 0;
}
