#ifndef MINIMAX_H3_Q4_LAYER_H
#define MINIMAX_H3_Q4_LAYER_H

#include "minimax_h3_remote_safetensors.h"

#include <stddef.h>
#include <stdint.h>

enum {
    MINIMAX_H3_Q4_PROJECTION_COUNT = 4,
    MINIMAX_H3_Q4_NORM_COUNT = 4
};

typedef struct minimax_h3_q4_projection {
    const char *suffix;
    uint32_t output_rows;
    uint32_t input_columns;
    uint32_t groups_per_row;
    unsigned char *weight;
    uint16_t *scales;
    uint16_t *biases;
    size_t weight_bytes;
    size_t metadata_elements;
} minimax_h3_q4_projection;

typedef struct {
    unsigned layer_index;
    minimax_h3_q4_projection projections[MINIMAX_H3_Q4_PROJECTION_COUNT];
    uint16_t *norms[MINIMAX_H3_Q4_NORM_COUNT];
    size_t norm_elements[MINIMAX_H3_Q4_NORM_COUNT];
    size_t allocated_bytes;
    uint64_t fingerprint;
} minimax_h3_q4_layer;

int minimax_h3_q4_layer_load(const minimax_h3_remote_safetensors *file,
                             unsigned layer_index,
                             minimax_h3_q4_layer *layer,
                             char *error,
                             size_t error_capacity);

void minimax_h3_q4_layer_free(minimax_h3_q4_layer *layer);

#endif
