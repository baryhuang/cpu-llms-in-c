#ifndef CLLM_WHISPER_SMALL_IMAGE_H
#define CLLM_WHISPER_SMALL_IMAGE_H

#include "whisper_small_encoder.h"

#include <stddef.h>
#include <stdint.h>

#define CLLM_WHISPER_SMALL_IMAGE_LAYERS 12U

typedef struct {
    int fd;
    size_t bytes;
    const unsigned char *base;
    const void *header;
    const void *descriptors;
} cllm_whisper_small_image;

typedef struct {
    cllm_whisper_small_image image;
    const float *mel_filters;
    const float *conv1_weight;
    const float *conv1_bias;
    const float *conv2_weight;
    const float *conv2_bias;
    const float *positions;
    const float *final_norm_weight;
    const float *final_norm_bias;
    cllm_whisper_encoder_block_weights layers[CLLM_WHISPER_SMALL_IMAGE_LAYERS];
} cllm_whisper_small_model;

typedef struct {
    double stem_seconds;
    double layer_seconds[CLLM_WHISPER_SMALL_IMAGE_LAYERS];
    double final_norm_seconds;
    size_t input_frames;
    size_t output_frames;
    size_t executed_layers;
    size_t workspace_bytes;
} cllm_whisper_small_encoder_metrics;

int cllm_whisper_small_model_open(const char *path, cllm_whisper_small_model *model);
void cllm_whisper_small_model_close(cllm_whisper_small_model *model);

/*
 * Execute the exact F32 encoder image on [80, input_frames] log-Mel input.
 * `layer_count` may be 0..12 for boundary verification. Final LayerNorm is
 * applied only after all 12 blocks. The caller owns [output_frames, 768].
 */
int cllm_whisper_small_encode_mel(const cllm_whisper_small_model *model,
                                  const float *mel,
                                  size_t input_frames,
                                  size_t layer_count,
                                  float *output,
                                  cllm_whisper_small_encoder_metrics *metrics);

#endif
