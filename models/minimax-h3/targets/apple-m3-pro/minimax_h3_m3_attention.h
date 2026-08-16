#ifndef MINIMAX_H3_M3_ATTENTION_H
#define MINIMAX_H3_M3_ATTENTION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char device_name[128];
    unsigned warmup_iterations;
    unsigned measured_iterations;
    size_t sequence_rows;
    size_t exact_rows;
    size_t video_rows;
    size_t tree_nodes;
    size_t query_blocks;
    size_t route_entries;
    size_t maximum_route_entries;
    size_t qkv_bytes;
    size_t summary_bytes;
    size_t output_bytes;
    size_t lse_bytes;
    size_t metal_owned_buffer_bytes;
    size_t footprint_before_bytes;
    size_t footprint_peak_bytes;
    double summary_gpu_ms;
    double summary_wall_ms;
    double attention_gpu_ms;
    double attention_wall_ms;
    double max_abs_error_first_head;
    float reference_first_8[8];
    float metal_first_8[8];
} minimax_h3_m3_attention_result;

/*
 * Synthetic exact-shape primitive benchmark. It measures H3's 15,485-row,
 * 56-head, 128-channel attention boundary with the compiled hierarchy. Q/K/V
 * projection, transformer weights and output projection are outside scope.
 */
int minimax_h3_m3_run_attention_benchmark(
    const char *metallib_path,
    unsigned warmup_iterations,
    unsigned measured_iterations,
    minimax_h3_m3_attention_result *result,
    char *error_message,
    size_t error_message_capacity);

#ifdef __cplusplus
}
#endif

#endif
