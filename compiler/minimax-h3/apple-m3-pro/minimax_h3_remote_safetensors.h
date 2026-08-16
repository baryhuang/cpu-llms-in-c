#ifndef MINIMAX_H3_REMOTE_SAFETENSORS_H
#define MINIMAX_H3_REMOTE_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MINIMAX_H3_REMOTE_MAX_URL = 2048,
    MINIMAX_H3_REMOTE_MAX_DTYPE = 16,
    MINIMAX_H3_REMOTE_MAX_RANK = 8
};

typedef struct {
    char dtype[MINIMAX_H3_REMOTE_MAX_DTYPE];
    uint64_t shape[MINIMAX_H3_REMOTE_MAX_RANK];
    size_t rank;
    uint64_t data_start;
    uint64_t data_length;
} minimax_h3_remote_tensor;

typedef struct {
    char source_url[MINIMAX_H3_REMOTE_MAX_URL];
    char url[MINIMAX_H3_REMOTE_MAX_URL];
    char *header;
    size_t header_length;
    uint64_t payload_start;
    uint64_t file_length;
} minimax_h3_remote_safetensors;

typedef void (*minimax_h3_remote_progress_fn)(uint64_t completed,
                                               uint64_t total,
                                               void *context);

int minimax_h3_remote_safetensors_open(
    const char *url,
    minimax_h3_remote_safetensors *file,
    char *error,
    size_t error_capacity);

void minimax_h3_remote_safetensors_close(
    minimax_h3_remote_safetensors *file);

int minimax_h3_remote_safetensors_find(
    const minimax_h3_remote_safetensors *file,
    const char *tensor_name,
    minimax_h3_remote_tensor *tensor,
    char *error,
    size_t error_capacity);

int minimax_h3_remote_safetensors_fetch(
    const minimax_h3_remote_safetensors *file,
    const minimax_h3_remote_tensor *tensor,
    void *output,
    size_t output_capacity,
    char *error,
    size_t error_capacity);

/* Fetch one tensor with independently validated parallel HTTP ranges. */
int minimax_h3_remote_safetensors_fetch_parallel(
    const minimax_h3_remote_safetensors *file,
    const minimax_h3_remote_tensor *tensor,
    void *output,
    size_t output_capacity,
    size_t chunk_bytes,
    char *error,
    size_t error_capacity);

/* Bounded absolute read for tensor slices or staged full-file materialization. */
int minimax_h3_remote_safetensors_read(
    const minimax_h3_remote_safetensors *file,
    uint64_t absolute_start,
    void *output,
    size_t length,
    char *error,
    size_t error_capacity);

/*
 * Materialize the exact remote safetensors byte stream into caller-owned
 * memory.  Reads are independently bounded and validated HTTP 206 ranges;
 * no temporary model file is created.
 */
int minimax_h3_remote_safetensors_materialize(
    const minimax_h3_remote_safetensors *file,
    void *output,
    size_t output_capacity,
    size_t chunk_bytes,
    minimax_h3_remote_progress_fn progress,
    void *progress_context,
    char *error,
    size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
