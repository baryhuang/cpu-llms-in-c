#ifndef MINIMAX_H3_M3_CACHE_H
#define MINIMAX_H3_M3_CACHE_H

#include <stddef.h>
#include <stdint.h>

#include "minimax_h3.h"
#include "minimax_h3_m3_aot.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MINIMAX_H3_M3_CACHE_LAYER_BAND = 10,
    MINIMAX_H3_M3_CACHE_BAND_COUNT = 5,
    MINIMAX_H3_M3_CACHE_BRANCH_COUNT = 2,
    MINIMAX_H3_M3_CACHE_Q4_GROUP_SIZE = 64,
    MINIMAX_H3_M3_CACHE_Q4_QUANT_BYTES = 32,
    MINIMAX_H3_M3_CACHE_Q4_META_BYTES = 4
};

typedef struct {
    /* 1..1000. A larger value computes more rows and reuses fewer rows. */
    uint16_t query_keep_per_mille[MINIMAX_H3_M3_TURBO_EVALUATIONS]
                                    [MINIMAX_H3_M3_CACHE_BAND_COUNT]
                                    [MINIMAX_H3_MODALITY_COUNT];
    uint16_t mlp_keep_per_mille[MINIMAX_H3_M3_TURBO_EVALUATIONS]
                                  [MINIMAX_H3_M3_CACHE_BAND_COUNT]
                                  [MINIMAX_H3_MODALITY_COUNT];
} minimax_h3_m3_cache_schedule;

typedef struct {
    size_t sequence_rows;
    size_t hidden_size;
    size_t layer_count;
    size_t branch_count;
    size_t groups_per_vector;
    size_t bytes_per_vector_per_branch;
    size_t bytes_per_layer;
    size_t file_bytes;
} minimax_h3_m3_cache_layout;

size_t minimax_h3_m3_cache_layer_band(size_t layer_index);

/*
 * Evaluation zero has no prior branch cache and must keep every row. The
 * first reusable evaluation may drop its keep ratio; from evaluation one
 * onward, high-noise -> low-noise keep ratios must be nondecreasing.
 */
minimax_h3_status minimax_h3_m3_cache_schedule_validate(
    const minimax_h3_m3_cache_schedule *schedule);

/* Full-capacity Q4 image for two ungated branch outputs at every H3 block. */
minimax_h3_status minimax_h3_m3_cache_layout_make(
    size_t sequence_rows,
    minimax_h3_m3_cache_layout *layout);

/* Byte offset of one branch vector; every value is checked. */
minimax_h3_status minimax_h3_m3_cache_vector_offset(
    const minimax_h3_m3_cache_layout *layout,
    size_t layer,
    size_t branch,
    size_t row,
    size_t *byte_offset);

#ifdef __cplusplus
}
#endif

#endif
