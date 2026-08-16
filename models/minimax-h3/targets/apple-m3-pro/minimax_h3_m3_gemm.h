#ifndef MINIMAX_H3_M3_GEMM_H
#define MINIMAX_H3_M3_GEMM_H

#include <stddef.h>
#include <stdint.h>

struct minimax_h3_q4_projection;

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char device_name[128];
    unsigned warmup_iterations;
    unsigned measured_iterations;
    size_t batch_rows;
    size_t input_columns;
    size_t output_rows;
    size_t weight_bytes;
    size_t metadata_bytes;
    size_t input_bytes;
    size_t output_bytes;
    size_t metal_owned_buffer_bytes;
    size_t footprint_before_bytes;
    size_t footprint_peak_bytes;
    double gpu_ms;
    double wall_ms;
    double effective_tflops;
    double max_abs_error_first_8;
    float reference_first_8[8];
    float metal_first_8[8];
} minimax_h3_m3_gemm_result;

/* Synthetic H3 5376 -> 14336 W4 projection at caller-selected token count. */
int minimax_h3_m3_run_gemm_benchmark(
    const char *metallib_path,
    size_t batch_rows,
    unsigned warmup_iterations,
    unsigned measured_iterations,
    minimax_h3_m3_gemm_result *result,
    char *error_message,
    size_t error_message_capacity);

/* Synthetic affine-Q8 projection used by the streamed Qwen conditioner. */
int minimax_h3_m3_run_q8_gemm_benchmark(
    const char *metallib_path,
    size_t input_columns,
    size_t output_rows,
    size_t batch_rows,
    unsigned warmup_iterations,
    unsigned measured_iterations,
    minimax_h3_m3_gemm_result *result,
    char *error_message,
    size_t error_message_capacity);

/* Execute the real BF16 activation path used by H3 affine-Q4 projections. */
int minimax_h3_m3_run_q4_projection_bf16(
    const char *metallib_path,
    const struct minimax_h3_q4_projection *projection,
    const uint16_t *input_bf16,
    size_t batch_rows,
    uint16_t *output_bf16,
    double *gpu_ms,
    char *error_message,
    size_t error_message_capacity);

#ifdef __cplusplus
}
#endif

#endif
