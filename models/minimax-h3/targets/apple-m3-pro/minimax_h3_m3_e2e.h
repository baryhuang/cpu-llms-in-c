#ifndef MINIMAX_H3_M3_E2E_H
#define MINIMAX_H3_M3_E2E_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t frames;
    uint64_t seed;
    const char *prompt;
    const char *tokenizer_image_path;
    const char *metallib_path;
    const char *output_directory;
} minimax_h3_m3_e2e_options;

typedef struct {
    size_t prompt_tokens;
    size_t sequence_rows;
    double tokenizer_seconds;
    double metal_setup_seconds;
    double text_download_seconds;
    double text_encode_seconds;
    double transformer_download_seconds;
    double turbo_compile_seconds;
    double rope_precompute_seconds;
    double denoise_seconds;
    double video_download_seconds;
    double video_precompute_seconds;
    double video_decode_seconds;
    double audio_download_seconds;
    double audio_decode_seconds;
    double mux_seconds;
    double total_seconds;
    double process_user_seconds;
    double process_system_seconds;
    size_t peak_footprint_bytes;
    int pipeline_archive_hit;
    uint32_t sampling_steps;
    int turbo_adapter_enabled;
    char video_path[1024];
    char audio_path[1024];
    char output_path[1024];
} minimax_h3_m3_e2e_result;

int minimax_h3_m3_run_e2e(const minimax_h3_m3_e2e_options *options,
                          minimax_h3_m3_e2e_result *result,
                          char *error,
                          size_t error_capacity);

/* Component validation only: bypasses the Qwen conditioner with one fixed
 * state row.  A passing result must never be reported as prompt-to-media E2E. */
int minimax_h3_m3_run_downstream_smoke(
    const minimax_h3_m3_e2e_options *options,
    minimax_h3_m3_e2e_result *result,
    char *error,
    size_t error_capacity);

/* Deterministic real-weight Video VAE benchmark.  This bypasses the text
 * encoder and denoiser and must not be reported as prompt-to-media E2E. */
int minimax_h3_m3_run_video_vae_smoke(
    const minimax_h3_m3_e2e_options *options,
    minimax_h3_m3_e2e_result *result,
    char *error,
    size_t error_capacity);

#endif
