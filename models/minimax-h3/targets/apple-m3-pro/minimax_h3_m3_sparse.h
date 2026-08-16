#ifndef MINIMAX_H3_M3_SPARSE_H
#define MINIMAX_H3_M3_SPARSE_H

#include <stddef.h>
#include <stdint.h>

#include "minimax_h3.h"
#include "minimax_h3_m3_cache.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MINIMAX_H3_M3_ATTENTION_WIDTH = 56 * 128,
    MINIMAX_H3_M3_FFN_SIZE = 14336,
    MINIMAX_H3_M3_NATIVE_SUMMARY_ROWS = 366
};

typedef struct {
    size_t query_rows[MINIMAX_H3_MODALITY_COUNT];
    size_t mlp_rows[MINIMAX_H3_MODALITY_COUNT];
    size_t total_query_rows;
    size_t total_mlp_rows;
    size_t kv_projection_rows;
    size_t summary_rows;
    uint64_t dense_projection_macs;
    uint64_t sparse_projection_macs;
} minimax_h3_m3_sparse_work;

/*
 * An unvalidated search seed, not a release policy. Evaluation zero is dense
 * because no prior branch output exists. Later video ratios are smallest in
 * the middle layer bands and relax toward the final denoise evaluation.
 */
void minimax_h3_m3_cache_schedule_seed(
    minimax_h3_m3_cache_schedule *schedule);

/*
 * Count projection work for the sparse execution contract. Selected rows
 * produce Q, attention output and MLP output. K/V are produced for selected
 * exact rows plus tree summaries. H3's K/V maps have no bias, so projecting a
 * mean normalized-hidden vector is exactly equal to averaging projected K/V:
 * W * mean(x) == mean(W * x). This removes dense K/V projection from pruned
 * branches; it does not make the summary-attention approximation exact.
 */
minimax_h3_status minimax_h3_m3_sparse_work_make(
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_cache_schedule *schedule,
    size_t evaluation,
    size_t layer_band,
    size_t summary_rows,
    minimax_h3_m3_sparse_work *work);

#ifdef __cplusplus
}
#endif

#endif
