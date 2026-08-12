#ifndef CLLM_WHISPER_FRONTEND_H
#define CLLM_WHISPER_FRONTEND_H

#include <stddef.h>

#define CLLM_WHISPER_SAMPLE_RATE 16000U
#define CLLM_WHISPER_N_FFT 400U
#define CLLM_WHISPER_HOP_LENGTH 160U
#define CLLM_WHISPER_N_FREQUENCIES 201U
#define CLLM_WHISPER_LARGE_V3_N_MELS 128U

size_t cllm_whisper_log_mel_frames(size_t sample_count);
size_t cllm_whisper_log_mel_workspace_floats(size_t sample_count);

/*
 * Scalar M1 correctness baseline for the pinned OpenAI front end.
 *
 * mel_filters is [128, 201], row-major. output is [128, frames],
 * row-major. workspace holds [frames, 201] power values. The caller owns
 * every buffer; this function allocates no memory.
 */
int cllm_whisper_log_mel_128(const float *audio,
                             size_t sample_count,
                             const float *mel_filters,
                             float *output,
                             float *workspace,
                             size_t workspace_floats);

#endif
