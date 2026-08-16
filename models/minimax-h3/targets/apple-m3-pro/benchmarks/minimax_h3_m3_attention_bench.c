#include "minimax_h3_m3_attention.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned parse_iterations(const char *text) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0u ||
        value > 100u) {
        fprintf(stderr, "invalid iteration count: %s\n", text);
        exit(2);
    }
    return (unsigned)value;
}

int main(int argc, char **argv) {
    minimax_h3_m3_attention_result result;
    char error[512];
    unsigned measured;
    unsigned warmup;
    int status;

    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s METALLIB [MEASURED] [WARMUP]\n", argv[0]);
        return 2;
    }
    measured = argc > 2 ? parse_iterations(argv[2]) : 3u;
    warmup = argc > 3 ? parse_iterations(argv[3]) : 1u;
    status = minimax_h3_m3_run_attention_benchmark(
        argv[1], warmup, measured, &result, error, sizeof(error));
    if (status != 0) {
        fprintf(stderr, "MiniMax-H3 attention benchmark failed (%d): %s\n",
                status, error);
        return status;
    }
    printf("{\n");
    printf("  \"schema\": 1,\n");
    printf("  \"scope\": \"synthetic exact-shape H3 hierarchical attention primitive; excludes QKV and output projections\",\n");
    printf("  \"device\": \"%s\",\n", result.device_name);
    printf("  \"warmup_iterations\": %u,\n", result.warmup_iterations);
    printf("  \"measured_iterations\": %u,\n", result.measured_iterations);
    printf("  \"sequence_rows\": %zu,\n", result.sequence_rows);
    printf("  \"exact_rows\": %zu,\n", result.exact_rows);
    printf("  \"video_rows\": %zu,\n", result.video_rows);
    printf("  \"tree_nodes\": %zu,\n", result.tree_nodes);
    printf("  \"query_blocks\": %zu,\n", result.query_blocks);
    printf("  \"route_entries\": %zu,\n", result.route_entries);
    printf("  \"maximum_route_entries\": %zu,\n", result.maximum_route_entries);
    printf("  \"summary_gpu_ms\": %.6f,\n", result.summary_gpu_ms);
    printf("  \"summary_wall_ms\": %.6f,\n", result.summary_wall_ms);
    printf("  \"attention_gpu_ms\": %.6f,\n", result.attention_gpu_ms);
    printf("  \"attention_wall_ms\": %.6f,\n", result.attention_wall_ms);
    printf("  \"qkv_bytes\": %zu,\n", result.qkv_bytes);
    printf("  \"summary_bytes\": %zu,\n", result.summary_bytes);
    printf("  \"output_bytes\": %zu,\n", result.output_bytes);
    printf("  \"lse_bytes\": %zu,\n", result.lse_bytes);
    printf("  \"metal_owned_buffer_bytes\": %zu,\n",
           result.metal_owned_buffer_bytes);
    printf("  \"footprint_before_bytes\": %zu,\n",
           result.footprint_before_bytes);
    printf("  \"footprint_peak_bytes\": %zu,\n", result.footprint_peak_bytes);
    printf("  \"max_abs_error_first_head\": %.9g,\n",
           result.max_abs_error_first_head);
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
