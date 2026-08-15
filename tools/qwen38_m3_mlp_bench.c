#include "qwen38_m3.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned parse_iterations(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0 || value > 10000) {
        fprintf(stderr, "invalid %s: %s\n", name, text);
        exit(2);
    }
    return (unsigned)value;
}

int main(int argc, char **argv) {
    const char *image_path = NULL;
    int first_number = 2;
    if (argc >= 4 && strcmp(argv[1], "--image") == 0) {
        image_path = argv[2];
        first_number = 4;
    }
    if (argc < first_number || argc > first_number + 2) {
        fprintf(stderr, "usage: %s [--image IMAGE.q38m3] METALLIB "
                        "[MEASURED_ITERATIONS] [WARMUP_ITERATIONS]\n", argv[0]);
        return 2;
    }
    const char *metallib_path = argv[first_number - 1];
    unsigned measured = argc > first_number ?
        parse_iterations(argv[first_number], "measured iterations") : 15;
    unsigned warmup = argc > first_number + 1 ?
        parse_iterations(argv[first_number + 1], "warmup iterations") : 2;

    qwen38_m3_mlp_result result;
    char error[512];
    int status = qwen38_m3_run_mlp_microbenchmark(metallib_path, image_path,
                                                  warmup, measured,
                                                  &result, error, sizeof(error));
    if (status != 0) {
        fprintf(stderr, "qwen38 M3 Pro benchmark failed (%d): %s\n", status, error);
        return status;
    }

    printf("{\n");
    printf("  \"schema\": 2,\n");
    printf("  \"scope\": \"Qwen3.8-27B layer-0 MLP primitive; not full-model tokens/s\",\n");
    printf("  \"device\": \"%s\",\n", result.device_name);
    printf("  \"weight_source\": \"%s\",\n", result.weight_source);
    printf("  \"shape\": {\"hidden\": %d, \"intermediate\": %d, \"group_size\": %d},\n",
           QWEN38_HIDDEN_SIZE, QWEN38_MLP_SIZE, QWEN38_Q4_GROUP_SIZE);
    printf("  \"warmup_iterations\": %u,\n", result.warmup_iterations);
    printf("  \"measured_iterations\": %u,\n", result.measured_iterations);
    printf("  \"bytes_per_matrix\": %zu,\n", result.bytes_per_matrix);
    printf("  \"metal_owned_buffer_bytes\": %zu,\n",
           result.metal_owned_buffer_bytes);
    printf("  \"mapped_image_bytes\": %zu,\n", result.mapped_image_bytes);
    printf("  \"footprint_before_bytes\": %zu,\n", result.footprint_before_bytes);
    printf("  \"footprint_peak_bytes\": %zu,\n", result.footprint_peak_bytes);
    printf("  \"generic_interleaved_ms\": %.6f,\n", result.generic_interleaved_ms);
    printf("  \"split_unfused_ms\": %.6f,\n", result.split_unfused_ms);
    printf("  \"fused_ms\": %.6f,\n", result.fused_ms);
    printf("  \"layout_vector_speedup\": %.6f,\n", result.layout_vector_speedup);
    printf("  \"fusion_speedup\": %.6f,\n", result.fusion_speedup);
    printf("  \"total_speedup\": %.6f,\n", result.total_speedup);
    printf("  \"fused_weight_gbps\": %.6f,\n", result.fused_weight_gbps);
    printf("  \"full_mlp_ms\": %.6f,\n", result.full_mlp_ms);
    printf("  \"full_mlp_effective_weight_gbps\": %.6f,\n",
           result.full_mlp_effective_weight_gbps);
    printf("  \"selected_threads_per_threadgroup\": %u,\n",
           result.selected_threads_per_threadgroup);
    printf("  \"schedule_search\": [\n");
    for (unsigned index = 0; index < result.schedule_candidate_count; ++index) {
        printf("    {\"threads\": %u, \"ms\": %.6f}%s\n",
               result.schedule_threads[index], result.schedule_ms[index],
               index + 1 == result.schedule_candidate_count ? "" : ",");
    }
    printf("  ],\n");
    printf("  \"max_abs_error_vs_unfused_gpu\": %.9g,\n", result.max_abs_error_vs_unfused_gpu);
    printf("  \"max_abs_error_vs_generic_gpu\": %.9g,\n", result.max_abs_error_vs_generic_gpu);
    printf("  \"max_abs_error_vs_cpu_first_8_rows\": %.9g,\n",
           result.max_abs_error_vs_cpu_first_8_rows);
    printf("  \"max_abs_error_vs_source_bf16_first_8_rows\": %.9g,\n",
           result.max_abs_error_vs_source_bf16_first_8_rows);
    printf("  \"source_bf16_reference_first_8\": [");
    for (unsigned index = 0; index < 8; ++index) {
        printf("%s%.9g", index == 0 ? "" : ", ",
               result.source_reference_first_8[index]);
    }
    printf("],\n");
    printf("  \"fused_gpu_output_first_8\": [");
    for (unsigned index = 0; index < 8; ++index) {
        printf("%s%.9g", index == 0 ? "" : ", ",
               result.fused_output_first_8[index]);
    }
    printf("],\n");
    printf("  \"max_abs_error_full_mlp_vs_source_bf16_first_8\": %.9g,\n",
           result.max_abs_error_full_mlp_vs_source_bf16_first_8);
    printf("  \"source_bf16_full_mlp_reference_first_8\": [");
    for (unsigned index = 0; index < 8; ++index) {
        printf("%s%.9g", index == 0 ? "" : ", ",
               result.source_mlp_reference_first_8[index]);
    }
    printf("],\n");
    printf("  \"full_mlp_gpu_output_first_8\": [");
    for (unsigned index = 0; index < 8; ++index) {
        printf("%s%.9g", index == 0 ? "" : ", ",
               result.full_mlp_output_first_8[index]);
    }
    printf("]\n");
    printf("}\n");
    return 0;
}
