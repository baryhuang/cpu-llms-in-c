#ifndef CLLM_WHISPER_TURBO_IMAGE_H
#define CLLM_WHISPER_TURBO_IMAGE_H

#include "whisper_turbo_encoder.h"
#include "whisper_turbo_decoder.h"

#include <stddef.h>
#include <stdint.h>

#define CLLM_WHISPER_TURBO_IMAGE_LAYERS 32U

typedef struct {
    int fd;
    size_t bytes;
    const unsigned char *base;
    const void *header;
    const void *descriptors;
} cllm_whisper_turbo_image;

typedef struct {
    cllm_whisper_turbo_image image;
    uint32_t image_version;
    const float *mel_filters;
    const float *conv1_weight;
    const float *conv1_bias;
    const float *conv2_weight;
    const float *conv2_bias;
    const float *positions;
    const float *final_norm_weight;
    const float *final_norm_bias;
    cllm_whisper_turbo_encoder_block_weights layers[CLLM_WHISPER_TURBO_IMAGE_LAYERS];
    cllm_whisper_turbo_decoder_weights decoder;
} cllm_whisper_turbo_model;

typedef struct {
    double stem_seconds;
    double layer_seconds[CLLM_WHISPER_TURBO_IMAGE_LAYERS];
    double final_norm_seconds;
    size_t input_frames;
    size_t output_frames;
    size_t executed_layers;
    size_t workspace_bytes;
} cllm_whisper_turbo_encoder_metrics;

int cllm_whisper_turbo_model_open(const char *path, cllm_whisper_turbo_model *model);
void cllm_whisper_turbo_model_close(cllm_whisper_turbo_model *model);

/*
 * Execute the encoder on [128, input_frames] log-Mel input. `layer_count`
 * may be 0..32 for boundary verification. Final LayerNorm is applied only
 * after all 32 blocks. The caller owns [output_frames, 1280].
 */
int cllm_whisper_turbo_encode_mel(const cllm_whisper_turbo_model *model,
                                  const float *mel,
                                  size_t input_frames,
                                  size_t layer_count,
                                  float *output,
                                  cllm_whisper_turbo_encoder_metrics *metrics);

#endif
