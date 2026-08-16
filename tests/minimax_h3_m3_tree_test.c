#include "minimax_h3.h"
#include "minimax_h3_m3_tree.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

#define CHECK(expression)                                                        \
    do {                                                                         \
        if (!(expression)) {                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,            \
                    #expression);                                                \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

int main(void) {
    minimax_h3_geometry geometry;
    minimax_h3_t2va_layout layout;
    minimax_h3_m3_tree_plan plan;
    minimax_h3_m3_tree_node *nodes;
    double *positions;
    uint8_t *tags;
    uint8_t *seen;
    size_t node_count;
    size_t leaf_index;
    size_t seen_count = 0u;
    size_t route_entry_count;
    size_t maximum_route_entries;
    uint32_t *route_offsets;
    uint32_t *route_entries;
    minimax_h3_m3_query_block *query_blocks;
    size_t query_block_count;
    size_t native_text_summary_count;
    size_t native_audio_summary_count;
    size_t native_route_entry_count;
    size_t native_maximum_route_entries;
    uint32_t *native_route_offsets;
    uint32_t *native_route_entries;

    CHECK(minimax_h3_geometry_init(&geometry, 864, 480, 124, 86) ==
          MINIMAX_H3_OK);
    positions = calloc(geometry.sequence_rows * 3u, sizeof(*positions));
    tags = calloc(geometry.sequence_rows, sizeof(*tags));
    CHECK(positions != NULL && tags != NULL);
    if (positions == NULL || tags == NULL) {
        free(positions);
        free(tags);
        return 1;
    }
    CHECK(minimax_h3_build_t2va_layout(&geometry, NULL, positions, tags,
                                       geometry.sequence_rows, &layout) ==
          MINIMAX_H3_OK);
    CHECK(minimax_h3_m3_tree_node_count(&geometry, &node_count) == MINIMAX_H3_OK);
    CHECK(node_count == 342u);
    nodes = calloc(node_count, sizeof(*nodes));
    seen = calloc(geometry.video_rows, sizeof(*seen));
    CHECK(nodes != NULL && seen != NULL);
    if (nodes == NULL || seen == NULL) {
        free(positions);
        free(tags);
        free(nodes);
        free(seen);
        return 1;
    }
    CHECK(minimax_h3_m3_tree_plan_make(&geometry, &layout, nodes, node_count,
                                       &plan) == MINIMAX_H3_OK);
    CHECK(plan.exact_count == 500u);
    CHECK(plan.video_start == 500u && plan.video_count == 14985u);
    CHECK(plan.leaf_count == 296u);
    CHECK(plan.frame_node_start == 296u && plan.frame_node_count == 37u);
    CHECK(plan.temporal_node_start == 333u && plan.temporal_node_count == 8u);
    CHECK(plan.root_index == 341u && plan.node_count == 342u);
    CHECK(plan.patch_rows == 15u && plan.patch_columns == 27u);
    CHECK(plan.tile_rows == 2u && plan.tile_columns == 4u);

    CHECK(nodes[0].kind == MINIMAX_H3_M3_TREE_LEAF);
    CHECK(nodes[0].patch_h == 8u && nodes[0].patch_w == 8u);
    CHECK(nodes[3].patch_x == 24u && nodes[3].patch_w == 3u);
    CHECK(nodes[4].patch_y == 8u && nodes[4].patch_h == 7u);
    CHECK(nodes[296].kind == MINIMAX_H3_M3_TREE_FRAME);
    CHECK(nodes[296].first_child == 0u && nodes[296].child_count == 8u);
    CHECK(nodes[296].token_count == 405u);
    CHECK(nodes[333].kind == MINIMAX_H3_M3_TREE_TEMPORAL);
    CHECK(nodes[333].first_child == 296u && nodes[333].child_count == 5u);
    CHECK(nodes[333].token_count == 2025u);
    CHECK(nodes[340].first_frame == 35u && nodes[340].frame_count == 2u);
    CHECK(nodes[340].token_count == 810u);
    CHECK(nodes[341].kind == MINIMAX_H3_M3_TREE_ROOT);
    CHECK(nodes[341].parent == MINIMAX_H3_M3_TREE_NO_PARENT);
    CHECK(nodes[341].token_count == 14985u);

    for (leaf_index = 0u; leaf_index < plan.leaf_count; ++leaf_index) {
        const minimax_h3_m3_tree_node *leaf = &nodes[leaf_index];
        uint16_t y;
        uint16_t x;
        for (y = 0u; y < leaf->patch_h; ++y) {
            for (x = 0u; x < leaf->patch_w; ++x) {
                size_t row;
                size_t video_row;
                CHECK(minimax_h3_m3_tree_leaf_row(&geometry, &layout, leaf, y, x,
                                                   &row) == MINIMAX_H3_OK);
                video_row = row - layout.video_start;
                CHECK(video_row < geometry.video_rows);
                CHECK(seen[video_row] == 0u);
                seen[video_row] = 1u;
                ++seen_count;
            }
        }
    }
    CHECK(seen_count == geometry.video_rows);
    for (leaf_index = 0u; leaf_index < geometry.video_rows; ++leaf_index) {
        CHECK(seen[leaf_index] == 1u);
    }

    CHECK(minimax_h3_m3_tree_route_entry_count(
              &geometry, &layout, nodes, &plan, &route_entry_count,
              &maximum_route_entries) == MINIMAX_H3_OK);
    CHECK(maximum_route_entries <= 548u);
    route_offsets = calloc(plan.leaf_count + 1u, sizeof(*route_offsets));
    route_entries = calloc(route_entry_count, sizeof(*route_entries));
    CHECK(route_offsets != NULL && route_entries != NULL);
    if (route_offsets != NULL && route_entries != NULL) {
        CHECK(minimax_h3_m3_tree_routes_make(
                  &geometry, &layout, nodes, &plan, route_offsets,
                  plan.leaf_count + 1u, route_entries, route_entry_count) ==
              MINIMAX_H3_OK);
        CHECK(route_offsets[plan.leaf_count] == route_entry_count);
        for (leaf_index = 0u; leaf_index < plan.leaf_count; ++leaf_index) {
            size_t entry;
            size_t represented_video_rows = 0u;
            for (entry = route_offsets[leaf_index];
                 entry < route_offsets[leaf_index + 1u]; ++entry) {
                uint32_t encoded = route_entries[entry];
                if ((encoded & MINIMAX_H3_M3_TREE_SUMMARY_ENTRY) != 0u) {
                    size_t node = encoded & ~MINIMAX_H3_M3_TREE_SUMMARY_ENTRY;
                    CHECK(node < plan.node_count);
                    represented_video_rows += nodes[node].token_count;
                } else {
                    CHECK(encoded >= layout.video_start &&
                          encoded < layout.sequence_rows);
                    ++represented_video_rows;
                }
            }
            CHECK(represented_video_rows == geometry.video_rows);
        }
    }

    CHECK(minimax_h3_m3_tree_query_block_count(
              &geometry, &layout, nodes, &plan, &query_block_count) ==
          MINIMAX_H3_OK);
    CHECK(query_block_count == 2283u);
    query_blocks = calloc(query_block_count, sizeof(*query_blocks));
    CHECK(query_blocks != NULL);
    if (query_blocks != NULL) {
        memset(seen, 0, geometry.video_rows * sizeof(*seen));
        CHECK(minimax_h3_m3_tree_query_blocks_make(
                  &geometry, &layout, nodes, &plan, query_blocks,
                  query_block_count) == MINIMAX_H3_OK);
        CHECK(query_blocks[0].first_row == 0u &&
              query_blocks[0].row_count == 8u &&
              query_blocks[0].route_index == 0u);
        CHECK(query_blocks[62].first_row == 496u &&
              query_blocks[62].row_count == 4u);
        for (size_t query_block = 63u; query_block < query_block_count;
             ++query_block) {
            for (size_t local = 0u;
                 local < query_blocks[query_block].row_count; ++local) {
                size_t video_row = query_blocks[query_block].first_row + local -
                                   layout.video_start;
                CHECK(video_row < geometry.video_rows);
                CHECK(seen[video_row] == 0u);
                seen[video_row] = 1u;
            }
        }
        for (leaf_index = 0u; leaf_index < geometry.video_rows; ++leaf_index)
            CHECK(seen[leaf_index] == 1u);
    }

    CHECK(minimax_h3_m3_tree_native_route_entry_count(
              &geometry, &layout, nodes, &plan, &native_text_summary_count,
              &native_audio_summary_count, &native_route_entry_count,
              &native_maximum_route_entries) ==
          MINIMAX_H3_OK);
    CHECK(native_text_summary_count == 11u);
    CHECK(native_audio_summary_count == 13u);
    CHECK(native_maximum_route_entries <= 106u);
    native_route_offsets = calloc(plan.leaf_count + 1u,
                                  sizeof(*native_route_offsets));
    native_route_entries = calloc(native_route_entry_count,
                                  sizeof(*native_route_entries));
    CHECK(native_route_offsets != NULL && native_route_entries != NULL);
    if (native_route_offsets != NULL && native_route_entries != NULL) {
        CHECK(minimax_h3_m3_tree_native_routes_make(
                  &geometry, &layout, nodes, &plan, plan.node_count,
                  plan.node_count + native_text_summary_count,
                  native_route_offsets, plan.leaf_count + 1u,
                  native_route_entries, native_route_entry_count) ==
              MINIMAX_H3_OK);
        for (leaf_index = 0u; leaf_index < plan.leaf_count; ++leaf_index) {
            size_t entry;
            size_t represented_rows = 0u;
            for (entry = native_route_offsets[leaf_index];
                 entry < native_route_offsets[leaf_index + 1u]; ++entry) {
                uint32_t encoded = native_route_entries[entry];
                if ((encoded & MINIMAX_H3_M3_TREE_SUMMARY_ENTRY) == 0u) {
                    ++represented_rows;
                } else {
                    size_t node = encoded & ~MINIMAX_H3_M3_TREE_SUMMARY_ENTRY;
                    if (node < plan.node_count) {
                        represented_rows += nodes[node].token_count;
                    } else if (node < plan.node_count +
                                      native_text_summary_count) {
                        size_t text_index = node - plan.node_count;
                        size_t text_start = text_index *
                                            MINIMAX_H3_M3_TREE_TEXT_SUMMARY_ROWS;
                        size_t text_rows = layout.text_count - text_start;
                        if (text_rows > MINIMAX_H3_M3_TREE_TEXT_SUMMARY_ROWS)
                            text_rows = MINIMAX_H3_M3_TREE_TEXT_SUMMARY_ROWS;
                        represented_rows += text_rows;
                    } else {
                        size_t audio_index = node - plan.node_count -
                                             native_text_summary_count;
                        size_t audio_start = audio_index *
                                             MINIMAX_H3_M3_TREE_AUDIO_SUMMARY_ROWS;
                        size_t audio_rows = layout.audio_count - audio_start;
                        if (audio_rows > MINIMAX_H3_M3_TREE_AUDIO_SUMMARY_ROWS)
                            audio_rows = MINIMAX_H3_M3_TREE_AUDIO_SUMMARY_ROWS;
                        represented_rows += audio_rows;
                    }
                }
            }
            CHECK(represented_rows == layout.sequence_rows);
        }
    }

    free(positions);
    free(tags);
    free(nodes);
    free(seen);
    free(route_offsets);
    free(route_entries);
    free(query_blocks);
    free(native_route_offsets);
    free(native_route_entries);
    if (failures != 0) {
        fprintf(stderr, "%d MiniMax-H3 M3 tree checks failed\n", failures);
        return 1;
    }
    puts("MiniMax-H3 M3 hierarchical video tree: PASS");
    return 0;
}
