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

typedef struct {
    uint32_t token_count;
    uint32_t chunk32_count;
    uint32_t chunk16_count;
    uint32_t single_count;
    double duration_ms;
    double first_chunk_ms;
} qwen36_m3_prefill_result;

/* Process a run of prompt tokens through batched S32/S16 graphs, falling
 * back to one-token forwards for a tail shorter than 16. Layer state after
 * prefill is bitwise-identical to the same tokens pushed one at a time; no
 * logits are produced, so the caller forwards the final prompt token through
 * qwen36_m3_model_forward for sampling. Synchronous; on error the layer
 * state is partially advanced and the caller should reset the model. */
int qwen36_m3_model_prefill(
    qwen36_m3_model *model,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t start_position,
    qwen36_m3_prefill_result *result,
    char *error_message,
    size_t error_message_capacity);

/* Submit one token without waiting for Metal completion. The model owns one
 * workspace and therefore permits exactly one in-flight forward. This split
 * lets the caller overlap CPU detokenization and output with GPU execution. */
int qwen36_m3_model_forward_submit(
    qwen36_m3_model *model,
    uint32_t token_id,
    uint32_t position,
    char *error_message,
    size_t error_message_capacity);

/* Wait for the submitted forward and expose its read-only logits until the
 * next submit. Calling this without an in-flight forward is an error. */
int qwen36_m3_model_forward_wait(
    qwen36_m3_model *model,
    qwen36_m3_decode_result *result,
    const float **logits,
    size_t *logit_count,
    char *error_message,
    size_t error_message_capacity);

enum {
    QWEN36_M3_STATE_RECURRENT = 0,
    QWEN36_M3_STATE_CONVOLUTION = 1,
    QWEN36_M3_STATE_KEY_CACHE = 2,
    QWEN36_M3_STATE_VALUE_CACHE = 3
};

/* Verification support: copy one layer's persistent state buffer. Returns
 * the state byte count, or 0 if the layer/kind combination does not exist
 * or the destination is too small. Recurrent and convolution state exist on
 * DeltaNet layers; key/value caches exist on attention layers. */
size_t qwen36_m3_model_copy_state(
    qwen36_m3_model *model,
    uint32_t layer_index,
    uint32_t kind,
    void *destination,
    size_t destination_capacity);

void qwen36_m3_model_close(qwen36_m3_model *model);

#endif
