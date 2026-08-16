#ifndef MINIMAX_H3_M3_SELECTOR_H
#define MINIMAX_H3_M3_SELECTOR_H

#include <stddef.h>
#include <stdint.h>

#include "minimax_h3.h"
#include "minimax_h3_m3_tree.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* 0..1; the remainder is split across available adjacent frames. */
    float temporal_center_weight;
    float lse_weight;
    float input_delta_weight;
    float reuse_age_weight;
} minimax_h3_m3_selector_weights;

/*
 * Select complete spatial leaves, never isolated video rows. One leaf is
 * retained per latent frame before the remaining global budget is assigned.
 * Caller-owned scratch_priority and selected_flags each need leaf_count
 * entries. selected_leaf_indices is emitted in physical leaf order.
 */
minimax_h3_status minimax_h3_m3_select_video_leaves(
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    const float *previous_lse_score,
    const float *input_delta_score,
    const uint16_t *reuse_age,
    uint16_t keep_per_mille,
    const minimax_h3_m3_selector_weights *weights,
    float *scratch_priority,
    uint8_t *selected_flags,
    uint32_t *selected_leaf_indices,
    size_t selected_capacity,
    size_t *selected_count,
    size_t *selected_video_rows);

#ifdef __cplusplus
}
#endif

#endif
