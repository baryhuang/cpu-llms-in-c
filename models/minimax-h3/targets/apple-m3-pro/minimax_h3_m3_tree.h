#ifndef MINIMAX_H3_M3_TREE_H
#define MINIMAX_H3_M3_TREE_H

#include <stddef.h>
#include <stdint.h>

#include "minimax_h3.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    MINIMAX_H3_M3_TREE_TILE_H = 8,
    MINIMAX_H3_M3_TREE_TILE_W = 8,
    MINIMAX_H3_M3_TREE_TEMPORAL_GROUP = 5,
    MINIMAX_H3_M3_TREE_TEXT_SUMMARY_ROWS = 8,
    MINIMAX_H3_M3_TREE_AUDIO_SUMMARY_ROWS = 32
};

#define MINIMAX_H3_M3_TREE_NO_PARENT UINT32_MAX
#define MINIMAX_H3_M3_TREE_SUMMARY_ENTRY UINT32_C(0x80000000)

typedef enum {
    MINIMAX_H3_M3_TREE_LEAF = 0,
    MINIMAX_H3_M3_TREE_FRAME = 1,
    MINIMAX_H3_M3_TREE_TEMPORAL = 2,
    MINIMAX_H3_M3_TREE_ROOT = 3
} minimax_h3_m3_tree_kind;

typedef struct {
    uint32_t parent;
    uint32_t first_child;
    uint16_t child_count;
    uint8_t kind;
    uint8_t reserved;
    uint32_t first_frame;
    uint16_t frame_count;
    uint16_t patch_y;
    uint16_t patch_x;
    uint16_t patch_h;
    uint16_t patch_w;
    uint32_t token_count;
} minimax_h3_m3_tree_node;

typedef struct {
    size_t exact_start;
    size_t exact_count;
    size_t video_start;
    size_t video_count;
    size_t leaf_count;
    size_t frame_node_start;
    size_t frame_node_count;
    size_t temporal_node_start;
    size_t temporal_node_count;
    size_t root_index;
    size_t node_count;
    uint16_t patch_rows;
    uint16_t patch_columns;
    uint16_t tile_rows;
    uint16_t tile_columns;
} minimax_h3_m3_tree_plan;

typedef struct {
    uint32_t first_row;
    uint16_t row_count;
    uint16_t reserved;
    uint32_t route_index;
} minimax_h3_m3_query_block;

/* Return the exact node capacity required for the fixed 8x8 / 5-frame tree. */
minimax_h3_status minimax_h3_m3_tree_node_count(
    const minimax_h3_geometry *geometry,
    size_t *node_count);

/*
 * Build static topology only. Text and stereo-audio rows form one exact sink;
 * video rows are leaves -> frame summaries -> five-frame summaries -> root.
 * The five-frame level follows H3's Video VAE latent cadence.
 */
minimax_h3_status minimax_h3_m3_tree_plan_make(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    minimax_h3_m3_tree_node *nodes,
    size_t node_capacity,
    minimax_h3_m3_tree_plan *plan);

/* Map one coordinate inside a leaf back to its packed sequence row. */
minimax_h3_status minimax_h3_m3_tree_leaf_row(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *leaf,
    uint16_t local_y,
    uint16_t local_x,
    size_t *sequence_row);

/*
 * Compile one locality seed route per video leaf. Each route partitions the
 * complete video domain into exact token rows and non-overlapping summary
 * nodes. Runtime centroid checks may replace these routes without changing the
 * attention ABI. Video queries use routes [0, leaf_count). Conditioning
 * queries use route leaf_count, which contains every video row exactly so
 * joint text/audio state is never conditioned on an arbitrary video leaf.
 * Offsets therefore has leaf_count + 2 entries.
 */
minimax_h3_status minimax_h3_m3_tree_route_entry_count(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    size_t *entry_count,
    size_t *maximum_route_entries);

minimax_h3_status minimax_h3_m3_tree_routes_make(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    uint32_t *offsets,
    size_t offset_capacity,
    uint32_t *entries,
    size_t entry_capacity);

/*
 * Quality-first route: never average keys or values across frames.  For every
 * query leaf, the two nearest spatial leaves are expanded exactly in every
 * latent frame; every other region remains a separate per-frame leaf
 * summary.  The conditioning route is the same exact all-video route used by
 * the hierarchical plan.  This costs more than the temporal tree but removes
 * its cross-frame value mixing, the primary source of double exposure.
 */
minimax_h3_status minimax_h3_m3_tree_frame_safe_route_entry_count(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    size_t *entry_count,
    size_t *maximum_route_entries);

minimax_h3_status minimax_h3_m3_tree_frame_safe_routes_make(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    uint32_t *offsets,
    size_t offset_capacity,
    uint32_t *entries,
    size_t entry_capacity);

/* Query blocks never cross a route boundary and contain at most eight rows. */
minimax_h3_status minimax_h3_m3_tree_query_block_count(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    size_t *block_count);

minimax_h3_status minimax_h3_m3_tree_query_blocks_make(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    minimax_h3_m3_query_block *blocks,
    size_t block_capacity);

/*
 * Aggressive H3-native route used by the M3 target experiment: text and audio
 * are represented by contiguous seed summaries, and only the query's own
 * video leaf is expanded. The offline compiler will replace the seed text
 * groups with teacher-selected groups before this path can pass quality.
 */
minimax_h3_status minimax_h3_m3_tree_native_route_entry_count(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    size_t *text_summary_count,
    size_t *audio_summary_count,
    size_t *entry_count,
    size_t *maximum_route_entries);

minimax_h3_status minimax_h3_m3_tree_native_routes_make(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    size_t text_summary_node_start,
    size_t audio_summary_node_start,
    uint32_t *offsets,
    size_t offset_capacity,
    uint32_t *entries,
    size_t entry_capacity);

#ifdef __cplusplus
}
#endif

#endif
