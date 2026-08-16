#include "minimax_h3_m3_cache.h"

#include <limits.h>

static int checked_add_size(size_t left, size_t right, size_t *result) {
    if (SIZE_MAX - left < right) return 0;
    *result = left + right;
    return 1;
}

static int checked_mul_size(size_t left, size_t right, size_t *result) {
    if (left != 0u && right > SIZE_MAX / left) return 0;
    *result = left * right;
    return 1;
}

size_t minimax_h3_m3_cache_layer_band(size_t layer_index) {
    return layer_index < MINIMAX_H3_M3_BLOCK_COUNT
               ? layer_index / MINIMAX_H3_M3_CACHE_LAYER_BAND
               : SIZE_MAX;
}

minimax_h3_status minimax_h3_m3_cache_schedule_validate(
    const minimax_h3_m3_cache_schedule *schedule) {
    size_t evaluation;
    size_t band;
    size_t modality;

    if (schedule == NULL) return MINIMAX_H3_INVALID_ARGUMENT;
    for (evaluation = 0u;
         evaluation < MINIMAX_H3_M3_TURBO_EVALUATIONS; ++evaluation) {
        for (band = 0u; band < MINIMAX_H3_M3_CACHE_BAND_COUNT; ++band) {
            for (modality = 0u; modality < MINIMAX_H3_MODALITY_COUNT;
                 ++modality) {
                uint16_t query =
                    schedule->query_keep_per_mille[evaluation][band][modality];
                uint16_t mlp =
                    schedule->mlp_keep_per_mille[evaluation][band][modality];
                if (query == 0u || query > 1000u || mlp == 0u || mlp > 1000u)
                    return MINIMAX_H3_INVALID_ARGUMENT;
                if (evaluation == 0u && (query != 1000u || mlp != 1000u))
                    return MINIMAX_H3_INVALID_ARGUMENT;
                if (evaluation > 1u &&
                    (query < schedule->query_keep_per_mille[evaluation - 1u]
                                                               [band][modality] ||
                     mlp < schedule->mlp_keep_per_mille[evaluation - 1u]
                                                           [band][modality])) {
                    return MINIMAX_H3_INVALID_ARGUMENT;
                }
            }
        }
    }
    return MINIMAX_H3_OK;
}

minimax_h3_status minimax_h3_m3_cache_layout_make(
    size_t sequence_rows,
    minimax_h3_m3_cache_layout *layout) {
    minimax_h3_m3_cache_layout result;
    size_t group_bytes;
    size_t vectors_per_layer;

    if (sequence_rows == 0u || layout == NULL) return MINIMAX_H3_INVALID_ARGUMENT;
    result.sequence_rows = sequence_rows;
    result.hidden_size = MINIMAX_H3_M3_HIDDEN_SIZE;
    result.layer_count = MINIMAX_H3_M3_BLOCK_COUNT;
    result.branch_count = MINIMAX_H3_M3_CACHE_BRANCH_COUNT;
    result.groups_per_vector =
        (result.hidden_size + MINIMAX_H3_M3_CACHE_Q4_GROUP_SIZE - 1u) /
        MINIMAX_H3_M3_CACHE_Q4_GROUP_SIZE;
    if (!checked_add_size(MINIMAX_H3_M3_CACHE_Q4_QUANT_BYTES,
                          MINIMAX_H3_M3_CACHE_Q4_META_BYTES, &group_bytes) ||
        !checked_mul_size(result.groups_per_vector, group_bytes,
                          &result.bytes_per_vector_per_branch) ||
        !checked_mul_size(result.sequence_rows, result.branch_count,
                          &vectors_per_layer) ||
        !checked_mul_size(vectors_per_layer,
                          result.bytes_per_vector_per_branch,
                          &result.bytes_per_layer) ||
        !checked_mul_size(result.bytes_per_layer, result.layer_count,
                          &result.file_bytes)) {
        return MINIMAX_H3_OVERFLOW;
    }
    *layout = result;
    return MINIMAX_H3_OK;
}

minimax_h3_status minimax_h3_m3_cache_vector_offset(
    const minimax_h3_m3_cache_layout *layout,
    size_t layer,
    size_t branch,
    size_t row,
    size_t *byte_offset) {
    size_t vector_index;
    size_t offset;

    if (layout == NULL || byte_offset == NULL ||
        layer >= layout->layer_count || branch >= layout->branch_count ||
        row >= layout->sequence_rows) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    if (!checked_mul_size(layer, layout->branch_count, &vector_index) ||
        !checked_add_size(vector_index, branch, &vector_index) ||
        !checked_mul_size(vector_index, layout->sequence_rows, &vector_index) ||
        !checked_add_size(vector_index, row, &vector_index) ||
        !checked_mul_size(vector_index, layout->bytes_per_vector_per_branch,
                          &offset) ||
        offset >= layout->file_bytes) {
        return MINIMAX_H3_OVERFLOW;
    }
    *byte_offset = offset;
    return MINIMAX_H3_OK;
}
