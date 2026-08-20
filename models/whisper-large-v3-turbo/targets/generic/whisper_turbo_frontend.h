#ifndef CLLM_WHISPER_TURBO_FRONTEND_H
#define CLLM_WHISPER_TURBO_FRONTEND_H

#include <stddef.h>

#define CLLM_WHISPER_SAMPLE_RATE 16000U
#define CLLM_WHISPER_N_FFT 400U
#define CLLM_WHISPER_HOP_LENGTH 160U
#define CLLM_WHISPER_N_FREQUENCIES 201U
#define CLLM_WHISPER_TURBO_N_MELS 128U
#define CLLM_WHISPER_TURBO_N_STATE 1280U
#define CLLM_WHISPER_TURBO_AUDIO_CONTEXT 1500U
#define CLLM_WHISPER_TURBO_ENCODER_LAYERS 32U

size_t cllm_whisper_turbo_log_mel_frames(size_t sample_count);
size_t cllm_whisper_turbo_log_mel_workspace_floats(size_t sample_count);

/*
 * Portable scalar correctness boundary for the pinned large-v3-turbo front
 * end. Identical STFT to small.en; the filterbank is the 128-bin large-v3
 * bank. mel_filters is [128, 201], output is [128, frames], and workspace is
 * [frames, 201], all row-major. The caller owns every buffer.
 */
int cllm_whisper_turbo_log_mel(const float *audio,
                               size_t sample_count,
                               const float *mel_filters,
                               float *output,
                               float *workspace,
                               size_t workspace_floats);

/*
 * OpenAI Whisper audio encoder stem:
 *
 *   Conv1D(kernel=3, padding=1) -> exact GELU
 *   Conv1D(kernel=3, stride=2, padding=1) -> exact GELU
 *   transpose channel/time -> add sinusoidal positions
 *
 * Input is [n_mels, input_frames]. Convolution weights are
 * [output_channel, input_channel, 3]. Positions and output are
 * [ceil(input_frames / 2), n_state]. scratch is [n_state, input_frames].
 */
size_t cllm_whisper_turbo_stem_output_frames(size_t input_frames);
size_t cllm_whisper_turbo_stem_scratch_floats(size_t input_frames,
                                              size_t n_state);
int cllm_whisper_turbo_stem(const float *mel,
                            size_t n_mels,
                            size_t input_frames,
                            size_t n_state,
                            const float *conv1_weight,
                            const float *conv1_bias,
                            const float *conv2_weight,
                            const float *conv2_bias,
                            const float *positions,
                            float *output,
                            float *scratch,
                            size_t scratch_floats);

#endif
