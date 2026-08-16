#include "minimax_h3_m3_sparse.h"

#include <limits.h>
#include <string.h>

static size_t selected_rows(size_t rows, uint16_t keep_per_mille) {
    uint64_t numerator = (uint64_t)rows * keep_per_mille + 999u;
    size_t selected = (size_t)(numerator / 1000u);
    if (selected == 0u && rows != 0u) selected = 1u;
    return selected < rows ? selected : rows;
}

static int add_u64(uint64_t left, uint64_t right, uint64_t *result) {
    if (UINT64_MAX - left < right) return 0;
    *result = left + right;
    return 1;
}

static int mul_u64(uint64_t left, uint64_t right, uint64_t *result) {
    if (left != 0u && right > UINT64_MAX / left) return 0;
    *result = left * right;
    return 1;
}

static int projection_macs(uint64_t query_rows,
                           uint64_t kv_rows,
                           uint64_t mlp_rows,
                           uint64_t *result) {
    uint64_t query_and_output;
    uint64_t key_and_value;
    uint64_t mlp;
    uint64_t temporary;

    if (!mul_u64(MINIMAX_H3_M3_HIDDEN_SIZE,
                 MINIMAX_H3_M3_ATTENTION_WIDTH, &temporary) ||
        !mul_u64(temporary, 2u, &query_and_output) ||
        !mul_u64(query_and_output, query_rows, &query_and_output) ||
        !mul_u64(temporary, 2u, &key_and_value) ||
        !mul_u64(key_and_value, kv_rows, &key_and_value) ||
        !mul_u64(MINIMAX_H3_M3_HIDDEN_SIZE,
                 MINIMAX_H3_M3_FFN_SIZE, &temporary) ||
        !mul_u64(temporary, 3u, &mlp) ||
        !mul_u64(mlp, mlp_rows, &mlp) ||
        !add_u64(query_and_output, key_and_value, &temporary) ||
        !add_u64(temporary, mlp, result)) {
        return 0;
    }
    return 1;
}

void minimax_h3_m3_cache_schedule_seed(
    minimax_h3_m3_cache_schedule *schedule) {
    static const uint16_t video_query[3] = {180u, 300u, 500u};
    static const uint16_t video_mlp[3] = {150u, 250u, 450u};
    static const uint16_t audio_query[3] = {650u, 800u, 1000u};
    static const uint16_t audio_mlp[3] = {600u, 750u, 900u};
    static const uint16_t band_add[MINIMAX_H3_M3_CACHE_BAND_COUNT] = {
        70u, 20u, 0u, 20u, 70u
    };
    size_t evaluation;
    size_t band;

    if (schedule == NULL) return;
    memset(schedule, 0, sizeof(*schedule));
    for (band = 0u; band < MINIMAX_H3_M3_CACHE_BAND_COUNT; ++band) {
        for (size_t modality = 0u; modality < MINIMAX_H3_MODALITY_COUNT;
             ++modality) {
            schedule->query_keep_per_mille[0][band][modality] = 1000u;
            schedule->mlp_keep_per_mille[0][band][modality] = 1000u;
        }
        for (evaluation = 1u;
             evaluation < MINIMAX_H3_M3_TURBO_EVALUATIONS; ++evaluation) {
            size_t index = evaluation - 1u;
            schedule->query_keep_per_mille[evaluation][band]
                                                   [MINIMAX_H3_TEXT_TAG] = 1000u;
            schedule->mlp_keep_per_mille[evaluation][band]
                                                 [MINIMAX_H3_TEXT_TAG] = 1000u;
            schedule->query_keep_per_mille[evaluation][band]
                                                   [MINIMAX_H3_AUDIO_TAG] =
                audio_query[index];
            schedule->mlp_keep_per_mille[evaluation][band]
                                                 [MINIMAX_H3_AUDIO_TAG] =
                audio_mlp[index];
            schedule->query_keep_per_mille[evaluation][band]
                                                   [MINIMAX_H3_VIDEO_TAG] =
                (uint16_t)(video_query[index] + band_add[band]);
            schedule->mlp_keep_per_mille[evaluation][band]
                                                 [MINIMAX_H3_VIDEO_TAG] =
                (uint16_t)(video_mlp[index] + band_add[band]);
        }
    }
}

minimax_h3_status minimax_h3_m3_sparse_work_make(
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_cache_schedule *schedule,
    size_t evaluation,
    size_t layer_band,
    size_t summary_rows,
    minimax_h3_m3_sparse_work *work) {
    minimax_h3_m3_sparse_work result;
    size_t modality_rows[MINIMAX_H3_MODALITY_COUNT];
    size_t modality;

    if (layout == NULL || schedule == NULL || work == NULL ||
        evaluation >= MINIMAX_H3_M3_TURBO_EVALUATIONS ||
        layer_band >= MINIMAX_H3_M3_CACHE_BAND_COUNT ||
        minimax_h3_m3_cache_schedule_validate(schedule) != MINIMAX_H3_OK) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    memset(&result, 0, sizeof(result));
    modality_rows[MINIMAX_H3_VIDEO_TAG] = layout->video_count;
    modality_rows[MINIMAX_H3_TEXT_TAG] = layout->text_count;
    modality_rows[MINIMAX_H3_AUDIO_TAG] = layout->audio_count;
    for (modality = 0u; modality < MINIMAX_H3_MODALITY_COUNT; ++modality) {
        result.query_rows[modality] = selected_rows(
            modality_rows[modality],
            schedule->query_keep_per_mille[evaluation][layer_band][modality]);
        result.mlp_rows[modality] = selected_rows(
            modality_rows[modality],
            schedule->mlp_keep_per_mille[evaluation][layer_band][modality]);
        if (SIZE_MAX - result.total_query_rows < result.query_rows[modality] ||
            SIZE_MAX - result.total_mlp_rows < result.mlp_rows[modality]) {
            return MINIMAX_H3_OVERFLOW;
        }
        result.total_query_rows += result.query_rows[modality];
        result.total_mlp_rows += result.mlp_rows[modality];
    }
    result.summary_rows = result.total_query_rows == layout->sequence_rows
                              ? 0u
                              : summary_rows;
    if (SIZE_MAX - result.total_query_rows < result.summary_rows)
        return MINIMAX_H3_OVERFLOW;
    result.kv_projection_rows = result.total_query_rows + result.summary_rows;
    if (result.kv_projection_rows > layout->sequence_rows)
        result.kv_projection_rows = layout->sequence_rows;
    if (!projection_macs(layout->sequence_rows, layout->sequence_rows,
                         layout->sequence_rows,
                         &result.dense_projection_macs) ||
        !projection_macs(result.total_query_rows, result.kv_projection_rows,
                         result.total_mlp_rows,
                         &result.sparse_projection_macs)) {
        return MINIMAX_H3_OVERFLOW;
    }
    *work = result;
    return MINIMAX_H3_OK;
}
