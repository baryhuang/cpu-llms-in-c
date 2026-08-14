#ifndef QWEN36_M3_DECODE_H
#define QWEN36_M3_DECODE_H

#include <stddef.h>
#include <stdint.h>

typedef struct qwen36_m3_model qwen36_m3_model;

typedef struct {
    uint32_t input_token;
    uint32_t next_token;
    uint32_t position;
    double duration_ms;
    size_t mapped_weight_bytes;
    size_t state_bytes;
    size_t kv_cache_bytes;
    size_t physical_footprint_bytes;
} qwen36_m3_decode_result;

qwen36_m3_model *qwen36_m3_model_open(
    const char *model_directory,
    const char *metallib_path,
    uint32_t context_capacity,
    char *error_message,
    size_t error_message_capacity);

void qwen36_m3_model_reset(qwen36_m3_model *model);

int qwen36_m3_model_decode(
    qwen36_m3_model *model,
    uint32_t token_id,
    uint32_t position,
    qwen36_m3_decode_result *result,
    char *error_message,
    size_t error_message_capacity);

/* Runs one token through the static graph and exposes read-only logits until
 * the next call. This is the C sampling interface; the model owns the data. */
int qwen36_m3_model_forward(
    qwen36_m3_model *model,
    uint32_t token_id,
    uint32_t position,
    qwen36_m3_decode_result *result,
    const float **logits,
    size_t *logit_count,
    char *error_message,
    size_t error_message_capacity);

void qwen36_m3_model_close(qwen36_m3_model *model);

#endif
