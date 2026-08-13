#ifndef QWEN36_M3_H
#define QWEN36_M3_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    QWEN36_HIDDEN_SIZE = 5120,
    QWEN36_MLP_SIZE = 17408,
    QWEN36_Q4_GROUP_SIZE = 64,
    QWEN36_Q4_GROUP_BYTES = 36
};

typedef struct {
    char device_name[128];
    char weight_source[128];
    unsigned warmup_iterations;
    unsigned measured_iterations;
    size_t bytes_per_matrix;
    size_t metal_owned_buffer_bytes;
    size_t mapped_image_bytes;
    size_t footprint_before_bytes;
    size_t footprint_peak_bytes;
    double generic_interleaved_ms;
    double split_unfused_ms;
    double fused_ms;
    double layout_vector_speedup;
    double fusion_speedup;
    double total_speedup;
    double fused_weight_gbps;
    double max_abs_error_vs_unfused_gpu;
    double max_abs_error_vs_generic_gpu;
    double max_abs_error_vs_cpu_first_8_rows;
    double max_abs_error_vs_source_bf16_first_8_rows;
    double full_mlp_ms;
    double full_mlp_effective_weight_gbps;
    double max_abs_error_full_mlp_vs_source_bf16_first_8;
    unsigned selected_threads_per_threadgroup;
    unsigned schedule_candidate_count;
    unsigned schedule_threads[4];
    double schedule_ms[4];
    float source_reference_first_8[8];
    float fused_output_first_8[8];
    float source_mlp_reference_first_8[8];
    float full_mlp_output_first_8[8];
} qwen36_m3_mlp_result;

/*
 * Runs the real Qwen3.6-27B layer-0 MLP shape. With image_path set, weights
 * come from the pinned checkpoint image; otherwise deterministic packed
 * weights exercise the exact runtime layout. The result is a primitive
 * benchmark, not an end-to-end token-generation measurement.
 */
int qwen36_m3_run_mlp_microbenchmark(const char *metallib_path,
                                    const char *image_path,
                                    unsigned warmup_iterations,
                                    unsigned measured_iterations,
                                    qwen36_m3_mlp_result *result,
                                    char *error_message,
                                    size_t error_message_capacity);

#ifdef __cplusplus
}
#endif

#endif
