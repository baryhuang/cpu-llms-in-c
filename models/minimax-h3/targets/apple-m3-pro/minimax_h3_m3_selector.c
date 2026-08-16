#include "minimax_h3_m3_selector.h"

#include <math.h>
#include <string.h>

static int finite_nonnegative(float value) {
    return isfinite(value) && value >= 0.0f;
}

static int better_leaf(float candidate,
                       size_t candidate_index,
                       float incumbent,
                       size_t incumbent_index) {
    return candidate > incumbent ||
           (candidate == incumbent && candidate_index < incumbent_index);
}

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
    size_t *selected_video_rows) {
    size_t leaves_per_frame;
    size_t target;
    size_t chosen = 0u;
    size_t rows = 0u;
    size_t leaf;

    if (nodes == NULL || plan == NULL || previous_lse_score == NULL ||
        input_delta_score == NULL || reuse_age == NULL || weights == NULL ||
        scratch_priority == NULL || selected_flags == NULL ||
        selected_leaf_indices == NULL || selected_count == NULL ||
        selected_video_rows == NULL || keep_per_mille == 0u ||
        keep_per_mille > 1000u || plan->leaf_count == 0u ||
        plan->frame_node_count == 0u ||
        plan->leaf_count % plan->frame_node_count != 0u ||
        weights->temporal_center_weight < 0.0f ||
        weights->temporal_center_weight > 1.0f ||
        !finite_nonnegative(weights->temporal_center_weight) ||
        !finite_nonnegative(weights->lse_weight) ||
        !finite_nonnegative(weights->input_delta_weight) ||
        !finite_nonnegative(weights->reuse_age_weight)) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    leaves_per_frame = plan->leaf_count / plan->frame_node_count;
    target = ((size_t)keep_per_mille * plan->leaf_count + 999u) / 1000u;
    if (target < plan->frame_node_count) target = plan->frame_node_count;
    if (target > plan->leaf_count) target = plan->leaf_count;
    if (selected_capacity < target) return MINIMAX_H3_INSUFFICIENT_CAPACITY;

    memset(selected_flags, 0, plan->leaf_count * sizeof(*selected_flags));
    for (leaf = 0u; leaf < plan->leaf_count; ++leaf) {
        size_t frame = leaf / leaves_per_frame;
        size_t spatial = leaf % leaves_per_frame;
        float neighbor_sum = 0.0f;
        size_t neighbor_count = 0u;
        float smoothed;
        if (!isfinite(previous_lse_score[leaf]) ||
            !finite_nonnegative(input_delta_score[leaf])) {
            return MINIMAX_H3_INVALID_ARGUMENT;
        }
        if (frame > 0u) {
            neighbor_sum += previous_lse_score[
                (frame - 1u) * leaves_per_frame + spatial];
            ++neighbor_count;
        }
        if (frame + 1u < plan->frame_node_count) {
            neighbor_sum += previous_lse_score[
                (frame + 1u) * leaves_per_frame + spatial];
            ++neighbor_count;
        }
        smoothed = weights->temporal_center_weight *
                   previous_lse_score[leaf];
        if (neighbor_count != 0u) {
            smoothed += (1.0f - weights->temporal_center_weight) *
                        neighbor_sum / (float)neighbor_count;
        }
        scratch_priority[leaf] =
            weights->lse_weight * smoothed +
            weights->input_delta_weight * input_delta_score[leaf] +
            weights->reuse_age_weight * (float)reuse_age[leaf];
    }

    /* Temporal coverage is a hard constraint, not a score penalty. */
    for (size_t frame = 0u; frame < plan->frame_node_count; ++frame) {
        size_t first = frame * leaves_per_frame;
        size_t best = first;
        for (leaf = first + 1u; leaf < first + leaves_per_frame; ++leaf) {
            if (better_leaf(scratch_priority[leaf], leaf,
                            scratch_priority[best], best)) {
                best = leaf;
            }
        }
        selected_flags[best] = 1u;
        ++chosen;
    }

    while (chosen < target) {
        size_t best = SIZE_MAX;
        for (leaf = 0u; leaf < plan->leaf_count; ++leaf) {
            if (selected_flags[leaf] != 0u) continue;
            if (best == SIZE_MAX ||
                better_leaf(scratch_priority[leaf], leaf,
                            scratch_priority[best], best)) {
                best = leaf;
            }
        }
        if (best == SIZE_MAX) return MINIMAX_H3_INVALID_ARGUMENT;
        selected_flags[best] = 1u;
        ++chosen;
    }

    chosen = 0u;
    for (leaf = 0u; leaf < plan->leaf_count; ++leaf) {
        if (selected_flags[leaf] == 0u) continue;
        selected_leaf_indices[chosen++] = (uint32_t)leaf;
        if (SIZE_MAX - rows < nodes[leaf].token_count)
            return MINIMAX_H3_OVERFLOW;
        rows += nodes[leaf].token_count;
    }
    *selected_count = chosen;
    *selected_video_rows = rows;
    return MINIMAX_H3_OK;
}
