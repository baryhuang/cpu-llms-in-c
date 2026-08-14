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
    QWEN36_Q4_GROUP_BYTES = 36,
    QWEN36_DELTA_KEY_HEADS = 16,
    QWEN36_DELTA_VALUE_HEADS = 48,
    QWEN36_DELTA_HEAD_SIZE = 128,
    QWEN36_DELTA_QKV_ROWS = 10240,
    QWEN36_DELTA_Z_ROWS = 6144,
    QWEN36_DELTA_INPUT_ROWS = 16480,
    QWEN36_DELTA_OUTPUT_INPUTS = 6144,
    QWEN36_DELTA_CONV_SIZE = 4
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

typedef struct {
    char device_name[128];
    unsigned warmup_iterations;
    unsigned measured_iterations;
    size_t state_bytes;
    size_t footprint_before_bytes;
    size_t footprint_peak_bytes;
    double direct_device_ms;
    double float2_vectorized_ms;
    double speedup;
    double direct_state_gbps;
    double vectorized_state_gbps;
    double max_abs_error_output;
    double max_abs_error_state;
    float reference_output_first_8[8];
    float vectorized_output_first_8[8];
} qwen36_m3_deltanet_core_result;

typedef struct {
    char device_name[128];
    char weight_source[256];
    unsigned warmup_iterations;
    unsigned measured_iterations;
    size_t mapped_image_bytes;
    size_t recurrent_state_bytes;
    size_t convolution_state_bytes;
    size_t metal_owned_buffer_bytes;
    size_t footprint_before_bytes;
    size_t footprint_peak_bytes;
    double layer_ms;
    double effective_weight_gbps;
    double max_abs_error_output_first_8;
    double max_abs_error_recurrent_state;
    double max_abs_error_convolution_state;
    float reference_output_first_8[8];
    float metal_output_first_8[8];
} qwen36_m3_layer_result;

typedef struct {
    char device_name[128];
    char weight_source[256];
    unsigned warmup_iterations;
    unsigned measured_iterations;
    size_t mapped_image_bytes;
    size_t kv_cache_bytes_at_context_1;
    size_t metal_owned_buffer_bytes;
    size_t footprint_before_bytes;
    size_t footprint_peak_bytes;
    double layer_ms_at_context_1;
    double effective_weight_gbps_at_context_1;
    double max_abs_error_output_first_8;
    double max_abs_error_query;
    double max_abs_error_key_cache;
    double max_abs_error_value_cache;
    float reference_output_first_8[8];
    float metal_output_first_8[8];
} qwen36_m3_attention_result;

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

/*
 * Measures one exact-shape Qwen3.6 recurrent DeltaNet state update. Inputs are
 * deterministic normalized projected tensors. This does not include Q4
 * projections, convolution, normalization or the output projection.
 */
int qwen36_m3_run_deltanet_core_benchmark(
    const char *metallib_path,
    unsigned warmup_iterations,
    unsigned measured_iterations,
    qwen36_m3_deltanet_core_result *result,
    char *error_message,
    size_t error_message_capacity);

/*
 * Runs a complete real layer-0 decode step:
 * RMSNorm -> DeltaNet projections/convolution/recurrent update/gated norm/
 * output projection -> residual -> RMSNorm -> MLP -> residual.
 * The first correctness run starts from zero recurrent and convolution state.
 */
int qwen36_m3_run_layer_benchmark(
    const char *metallib_path,
    const char *image_path,
    unsigned warmup_iterations,
    unsigned measured_iterations,
    qwen36_m3_layer_result *result,
    char *error_message,
    size_t error_message_capacity);

int qwen36_m3_run_attention_layer_benchmark(
    const char *metallib_path,
    const char *image_path,
    unsigned warmup_iterations,
    unsigned measured_iterations,
    qwen36_m3_attention_result *result,
    char *error_message,
    size_t error_message_capacity);

#ifdef __cplusplus
}
#endif

#endif
