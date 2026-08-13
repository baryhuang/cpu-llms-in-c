#ifndef QWEN36_SAFETENSORS_H
#define QWEN36_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    char dtype[16];
    uint64_t shape[8];
    size_t rank;
    uint64_t data_start;
    uint64_t data_length;
} qwen36_tensor_view;

/*
 * Reads one exact tensor descriptor. With require_payload != 0, the complete
 * tensor byte range must be present in the file. Header-only inspection of a
 * partial range is permitted only when require_payload == 0.
 */
int qwen36_safetensors_find(const char *path,
                            const char *tensor_name,
                            int require_payload,
                            qwen36_tensor_view *view,
                            char *error,
                            size_t error_capacity);

#endif
