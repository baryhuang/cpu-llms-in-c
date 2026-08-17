#include "minimax_h3_m3_tree.h"

#include <limits.h>
#include <string.h>

static size_t ceil_div_size(size_t value, size_t divisor) {
    return value / divisor + (value % divisor != 0u);
}

static int checked_add_size(size_t left, size_t right, size_t *result) {
    if (SIZE_MAX - left < right) {
        return 0;
    }
    *result = left + right;
    return 1;
}

static int checked_mul_size(size_t left, size_t right, size_t *result) {
    if (left != 0u && right > SIZE_MAX / left) {
        return 0;
    }
    *result = left * right;
    return 1;
}

static minimax_h3_status tree_counts(const minimax_h3_geometry *geometry,
                                     size_t *patch_rows,
                                     size_t *patch_columns,
                                     size_t *tile_rows,
                                     size_t *tile_columns,
                                     size_t *leaves_per_frame,
                                     size_t *leaf_count,
                                     size_t *temporal_count,
                                     size_t *node_count) {
    size_t count;

    if (geometry == NULL || patch_rows == NULL || patch_columns == NULL ||
        tile_rows == NULL || tile_columns == NULL || leaves_per_frame == NULL ||
        leaf_count == NULL || temporal_count == NULL || node_count == NULL ||
        geometry->video_latent_frames == 0u || geometry->latent_height == 0u ||
        geometry->latent_width == 0u ||
        geometry->latent_height % MINIMAX_H3_VIDEO_PATCH_H != 0u ||
        geometry->latent_width % MINIMAX_H3_VIDEO_PATCH_W != 0u) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    *patch_rows = geometry->latent_height / MINIMAX_H3_VIDEO_PATCH_H;
    *patch_columns = geometry->latent_width / MINIMAX_H3_VIDEO_PATCH_W;
    *tile_rows = ceil_div_size(*patch_rows, MINIMAX_H3_M3_TREE_TILE_H);
    *tile_columns = ceil_div_size(*patch_columns, MINIMAX_H3_M3_TREE_TILE_W);
    if (!checked_mul_size(*tile_rows, *tile_columns, leaves_per_frame) ||
        !checked_mul_size(*leaves_per_frame, geometry->video_latent_frames,
                          leaf_count)) {
        return MINIMAX_H3_OVERFLOW;
    }
    *temporal_count = ceil_div_size(geometry->video_latent_frames,
                                    MINIMAX_H3_M3_TREE_TEMPORAL_GROUP);
    if (!checked_add_size(*leaf_count, geometry->video_latent_frames, &count) ||
        !checked_add_size(count, *temporal_count, &count) ||
        !checked_add_size(count, 1u, node_count)) {
        return MINIMAX_H3_OVERFLOW;
    }
    if (*patch_rows > UINT16_MAX || *patch_columns > UINT16_MAX ||
        *tile_rows > UINT16_MAX || *tile_columns > UINT16_MAX ||
        *node_count > UINT32_MAX) {
        return MINIMAX_H3_OVERFLOW;
    }
    return MINIMAX_H3_OK;
}

minimax_h3_status minimax_h3_m3_tree_node_count(
    const minimax_h3_geometry *geometry,
    size_t *node_count) {
    size_t patch_rows;
    size_t patch_columns;
    size_t tile_rows;
    size_t tile_columns;
    size_t leaves_per_frame;
    size_t leaf_count;
    size_t temporal_count;

    return tree_counts(geometry, &patch_rows, &patch_columns, &tile_rows,
                       &tile_columns, &leaves_per_frame, &leaf_count,
                       &temporal_count, node_count);
}

minimax_h3_status minimax_h3_m3_tree_plan_make(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    minimax_h3_m3_tree_node *nodes,
    size_t node_capacity,
    minimax_h3_m3_tree_plan *plan) {
    minimax_h3_m3_tree_plan result;
    size_t patch_rows;
    size_t patch_columns;
    size_t tile_rows;
    size_t tile_columns;
    size_t leaves_per_frame;
    size_t leaf_count;
    size_t temporal_count;
    size_t node_count;
    size_t frame_node_start;
    size_t temporal_node_start;
    size_t root_index;
    size_t frame;
    size_t tile_y;
    size_t tile_x;
    size_t temporal;
    minimax_h3_status status;

    if (layout == NULL || nodes == NULL || plan == NULL) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    status = tree_counts(geometry, &patch_rows, &patch_columns, &tile_rows,
                         &tile_columns, &leaves_per_frame, &leaf_count,
                         &temporal_count, &node_count);
    if (status != MINIMAX_H3_OK) {
        return status;
    }
    if (node_capacity < node_count) {
        return MINIMAX_H3_INSUFFICIENT_CAPACITY;
    }
    if (layout->text_start != 0u ||
        layout->audio_start != layout->text_count ||
        layout->video_start != layout->text_count + layout->audio_count ||
        layout->video_count != geometry->video_rows ||
        layout->sequence_rows != geometry->sequence_rows) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }

    frame_node_start = leaf_count;
    temporal_node_start = frame_node_start + geometry->video_latent_frames;
    root_index = temporal_node_start + temporal_count;
    memset(nodes, 0, node_count * sizeof(*nodes));

    for (frame = 0u; frame < geometry->video_latent_frames; ++frame) {
        size_t frame_leaf_start = frame * leaves_per_frame;
        size_t frame_node_index = frame_node_start + frame;
        size_t frame_tokens = 0u;
        size_t leaf_in_frame = 0u;
        for (tile_y = 0u; tile_y < tile_rows; ++tile_y) {
            for (tile_x = 0u; tile_x < tile_columns; ++tile_x) {
                size_t leaf_index = frame_leaf_start + leaf_in_frame;
                minimax_h3_m3_tree_node *leaf = &nodes[leaf_index];
                size_t patch_y = tile_y * MINIMAX_H3_M3_TREE_TILE_H;
                size_t patch_x = tile_x * MINIMAX_H3_M3_TREE_TILE_W;
                size_t patch_h = patch_rows - patch_y;
                size_t patch_w = patch_columns - patch_x;
                if (patch_h > MINIMAX_H3_M3_TREE_TILE_H) {
                    patch_h = MINIMAX_H3_M3_TREE_TILE_H;
                }
                if (patch_w > MINIMAX_H3_M3_TREE_TILE_W) {
                    patch_w = MINIMAX_H3_M3_TREE_TILE_W;
                }
                leaf->parent = (uint32_t)frame_node_index;
                leaf->first_child = MINIMAX_H3_M3_TREE_NO_PARENT;
                leaf->kind = MINIMAX_H3_M3_TREE_LEAF;
                leaf->first_frame = (uint32_t)frame;
                leaf->frame_count = 1u;
                leaf->patch_y = (uint16_t)patch_y;
                leaf->patch_x = (uint16_t)patch_x;
                leaf->patch_h = (uint16_t)patch_h;
                leaf->patch_w = (uint16_t)patch_w;
                leaf->token_count = (uint32_t)(patch_h * patch_w);
                frame_tokens += leaf->token_count;
                ++leaf_in_frame;
            }
        }
        nodes[frame_node_index].parent =
            (uint32_t)(temporal_node_start +
                       frame / MINIMAX_H3_M3_TREE_TEMPORAL_GROUP);
        nodes[frame_node_index].first_child = (uint32_t)frame_leaf_start;
        nodes[frame_node_index].child_count = (uint16_t)leaves_per_frame;
        nodes[frame_node_index].kind = MINIMAX_H3_M3_TREE_FRAME;
        nodes[frame_node_index].first_frame = (uint32_t)frame;
        nodes[frame_node_index].frame_count = 1u;
        nodes[frame_node_index].patch_h = (uint16_t)patch_rows;
        nodes[frame_node_index].patch_w = (uint16_t)patch_columns;
        nodes[frame_node_index].token_count = (uint32_t)frame_tokens;
    }

    for (temporal = 0u; temporal < temporal_count; ++temporal) {
        minimax_h3_m3_tree_node *node = &nodes[temporal_node_start + temporal];
        size_t first_frame = temporal * MINIMAX_H3_M3_TREE_TEMPORAL_GROUP;
        size_t frame_count = geometry->video_latent_frames - first_frame;
        size_t token_count;
        if (frame_count > MINIMAX_H3_M3_TREE_TEMPORAL_GROUP) {
            frame_count = MINIMAX_H3_M3_TREE_TEMPORAL_GROUP;
        }
        if (!checked_mul_size(frame_count, geometry->rows_per_video_frame,
                              &token_count) || token_count > UINT32_MAX) {
            return MINIMAX_H3_OVERFLOW;
        }
        node->parent = (uint32_t)root_index;
        node->first_child = (uint32_t)(frame_node_start + first_frame);
        node->child_count = (uint16_t)frame_count;
        node->kind = MINIMAX_H3_M3_TREE_TEMPORAL;
        node->first_frame = (uint32_t)first_frame;
        node->frame_count = (uint16_t)frame_count;
        node->patch_h = (uint16_t)patch_rows;
        node->patch_w = (uint16_t)patch_columns;
        node->token_count = (uint32_t)token_count;
    }

    nodes[root_index].parent = MINIMAX_H3_M3_TREE_NO_PARENT;
    nodes[root_index].first_child = (uint32_t)temporal_node_start;
    nodes[root_index].child_count = (uint16_t)temporal_count;
    nodes[root_index].kind = MINIMAX_H3_M3_TREE_ROOT;
    nodes[root_index].first_frame = 0u;
    nodes[root_index].frame_count = (uint16_t)geometry->video_latent_frames;
    nodes[root_index].patch_h = (uint16_t)patch_rows;
    nodes[root_index].patch_w = (uint16_t)patch_columns;
    nodes[root_index].token_count = (uint32_t)geometry->video_rows;

    memset(&result, 0, sizeof(result));
    result.exact_start = 0u;
    result.exact_count = layout->video_start;
    result.video_start = layout->video_start;
    result.video_count = layout->video_count;
    result.leaf_count = leaf_count;
    result.frame_node_start = frame_node_start;
    result.frame_node_count = geometry->video_latent_frames;
    result.temporal_node_start = temporal_node_start;
    result.temporal_node_count = temporal_count;
    result.root_index = root_index;
    result.node_count = node_count;
    result.patch_rows = (uint16_t)patch_rows;
    result.patch_columns = (uint16_t)patch_columns;
    result.tile_rows = (uint16_t)tile_rows;
    result.tile_columns = (uint16_t)tile_columns;
    *plan = result;
    return MINIMAX_H3_OK;
}

minimax_h3_status minimax_h3_m3_tree_leaf_row(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *leaf,
    uint16_t local_y,
    uint16_t local_x,
    size_t *sequence_row) {
    size_t patch_columns;
    size_t frame_offset;
    size_t row_offset;
    size_t result;

    if (geometry == NULL || layout == NULL || leaf == NULL ||
        sequence_row == NULL || leaf->kind != MINIMAX_H3_M3_TREE_LEAF ||
        leaf->frame_count != 1u || local_y >= leaf->patch_h ||
        local_x >= leaf->patch_w ||
        leaf->first_frame >= geometry->video_latent_frames) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    patch_columns = geometry->latent_width / MINIMAX_H3_VIDEO_PATCH_W;
    if (!checked_mul_size(leaf->first_frame, geometry->rows_per_video_frame,
                          &frame_offset) ||
        !checked_mul_size((size_t)leaf->patch_y + local_y, patch_columns,
                          &row_offset) ||
        !checked_add_size(row_offset, (size_t)leaf->patch_x + local_x,
                          &row_offset) ||
        !checked_add_size(layout->video_start, frame_offset, &result) ||
        !checked_add_size(result, row_offset, &result) ||
        result >= layout->sequence_rows) {
        return MINIMAX_H3_OVERFLOW;
    }
    *sequence_row = result;
    return MINIMAX_H3_OK;
}

typedef struct {
    uint32_t *entries;
    size_t capacity;
    size_t count;
} route_writer;

static minimax_h3_status route_emit(route_writer *writer, uint32_t value) {
    if (writer->entries != NULL) {
        if (writer->count >= writer->capacity) {
            return MINIMAX_H3_INSUFFICIENT_CAPACITY;
        }
        writer->entries[writer->count] = value;
    }
    ++writer->count;
    return MINIMAX_H3_OK;
}

static int route_selected(size_t value, const size_t *selected, size_t count) {
    size_t index;
    for (index = 0u; index < count; ++index) {
        if (selected[index] == value) {
            return 1;
        }
    }
    return 0;
}

static size_t select_nearest(size_t begin,
                             size_t count,
                             size_t query,
                             size_t maximum,
                             size_t selected[2]) {
    size_t selected_count = 0u;
    while (selected_count < maximum && selected_count < count) {
        size_t candidate;
        size_t best = SIZE_MAX;
        size_t best_distance = SIZE_MAX;
        for (candidate = begin; candidate < begin + count; ++candidate) {
            size_t distance = candidate > query ? candidate - query : query - candidate;
            if (!route_selected(candidate, selected, selected_count) &&
                (distance < best_distance ||
                 (distance == best_distance && candidate < best))) {
                best = candidate;
                best_distance = distance;
            }
        }
        if (best == SIZE_MAX) {
            break;
        }
        selected[selected_count++] = best;
    }
    return selected_count;
}

static size_t select_nearest_leaf(size_t frame_leaf_start,
                                  size_t leaves_per_frame,
                                  size_t tile_columns,
                                  size_t query_tile_y,
                                  size_t query_tile_x,
                                  size_t selected[2]) {
    size_t selected_count = 0u;
    while (selected_count < 2u && selected_count < leaves_per_frame) {
        size_t local;
        size_t best = SIZE_MAX;
        size_t best_distance = SIZE_MAX;
        for (local = 0u; local < leaves_per_frame; ++local) {
            size_t leaf = frame_leaf_start + local;
            size_t tile_y = local / tile_columns;
            size_t tile_x = local - tile_y * tile_columns;
            size_t y_distance = tile_y > query_tile_y
                                    ? tile_y - query_tile_y
                                    : query_tile_y - tile_y;
            size_t x_distance = tile_x > query_tile_x
                                    ? tile_x - query_tile_x
                                    : query_tile_x - tile_x;
            size_t distance = y_distance + x_distance;
            if (!route_selected(leaf, selected, selected_count) &&
                (distance < best_distance ||
                 (distance == best_distance && leaf < best))) {
                best = leaf;
                best_distance = distance;
            }
        }
        if (best == SIZE_MAX) {
            break;
        }
        selected[selected_count++] = best;
    }
    return selected_count;
}

static minimax_h3_status emit_exact_leaf(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *leaf,
    route_writer *writer) {
    uint16_t y;
    uint16_t x;
    for (y = 0u; y < leaf->patch_h; ++y) {
        for (x = 0u; x < leaf->patch_w; ++x) {
            size_t row;
            minimax_h3_status status = minimax_h3_m3_tree_leaf_row(
                geometry, layout, leaf, y, x, &row);
            if (status != MINIMAX_H3_OK) return status;
            if (row >= MINIMAX_H3_M3_TREE_SUMMARY_ENTRY) {
                return MINIMAX_H3_OVERFLOW;
            }
            status = route_emit(writer, (uint32_t)row);
            if (status != MINIMAX_H3_OK) return status;
        }
    }
    return MINIMAX_H3_OK;
}

static minimax_h3_status build_one_route(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    size_t query_leaf,
    route_writer *writer) {
    size_t leaves_per_frame = (size_t)plan->tile_rows * plan->tile_columns;
    size_t query_frame = query_leaf / leaves_per_frame;
    size_t query_local_leaf = query_leaf - query_frame * leaves_per_frame;
    size_t query_tile_y = query_local_leaf / plan->tile_columns;
    size_t query_tile_x = query_local_leaf - query_tile_y * plan->tile_columns;
    size_t own_temporal = query_frame / MINIMAX_H3_M3_TREE_TEMPORAL_GROUP;
    size_t selected_temporals[2];
    size_t selected_temporal_count = 1u;
    size_t temporal;

    selected_temporals[0] = own_temporal;
    if (plan->temporal_node_count > 1u) {
        selected_temporals[1] = own_temporal + 1u < plan->temporal_node_count
                                    ? own_temporal + 1u
                                    : own_temporal - 1u;
        selected_temporal_count = 2u;
    }

    for (temporal = 0u; temporal < plan->temporal_node_count; ++temporal) {
        size_t temporal_node_index = plan->temporal_node_start + temporal;
        const minimax_h3_m3_tree_node *temporal_node = &nodes[temporal_node_index];
        if (!route_selected(temporal, selected_temporals,
                            selected_temporal_count)) {
            minimax_h3_status status = route_emit(
                writer, MINIMAX_H3_M3_TREE_SUMMARY_ENTRY |
                            (uint32_t)temporal_node_index);
            if (status != MINIMAX_H3_OK) return status;
        } else {
            size_t selected_frames[2];
            size_t selected_frame_count = select_nearest(
                temporal_node->first_frame, temporal_node->frame_count,
                query_frame, 2u, selected_frames);
            size_t frame;
            for (frame = temporal_node->first_frame;
                 frame < (size_t)temporal_node->first_frame +
                             temporal_node->frame_count;
                 ++frame) {
                size_t frame_node_index = plan->frame_node_start + frame;
                if (!route_selected(frame, selected_frames, selected_frame_count)) {
                    minimax_h3_status status = route_emit(
                        writer, MINIMAX_H3_M3_TREE_SUMMARY_ENTRY |
                                    (uint32_t)frame_node_index);
                    if (status != MINIMAX_H3_OK) return status;
                } else {
                    size_t frame_leaf_start = frame * leaves_per_frame;
                    size_t selected_leaves[2];
                    size_t selected_leaf_count = select_nearest_leaf(
                        frame_leaf_start, leaves_per_frame, plan->tile_columns,
                        query_tile_y, query_tile_x, selected_leaves);
                    size_t leaf;
                    for (leaf = frame_leaf_start;
                         leaf < frame_leaf_start + leaves_per_frame; ++leaf) {
                        if (!route_selected(leaf, selected_leaves,
                                            selected_leaf_count)) {
                            minimax_h3_status status = route_emit(
                                writer, MINIMAX_H3_M3_TREE_SUMMARY_ENTRY |
                                            (uint32_t)leaf);
                            if (status != MINIMAX_H3_OK) return status;
                        } else {
                            minimax_h3_status status = emit_exact_leaf(
                                geometry, layout, &nodes[leaf], writer);
                            if (status != MINIMAX_H3_OK) return status;
                        }
                    }
                }
            }
        }
    }
    return MINIMAX_H3_OK;
}

static minimax_h3_status build_one_frame_safe_route(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    size_t query_leaf,
    route_writer *writer) {
    size_t leaves_per_frame = (size_t)plan->tile_rows * plan->tile_columns;
    size_t query_local_leaf = query_leaf % leaves_per_frame;
    size_t query_tile_y = query_local_leaf / plan->tile_columns;
    size_t query_tile_x = query_local_leaf % plan->tile_columns;
    size_t frame;

    for (frame = 0u; frame < geometry->video_latent_frames; ++frame) {
        size_t frame_leaf_start = frame * leaves_per_frame;
        size_t selected_leaves[2];
        size_t selected_leaf_count = select_nearest_leaf(
            frame_leaf_start, leaves_per_frame, plan->tile_columns,
            query_tile_y, query_tile_x, selected_leaves);
        size_t leaf;
        for (leaf = frame_leaf_start;
             leaf < frame_leaf_start + leaves_per_frame; ++leaf) {
            minimax_h3_status status;
            if (route_selected(leaf, selected_leaves, selected_leaf_count)) {
                status = emit_exact_leaf(geometry, layout, &nodes[leaf], writer);
            } else {
                status = route_emit(writer,
                    MINIMAX_H3_M3_TREE_SUMMARY_ENTRY | (uint32_t)leaf);
            }
            if (status != MINIMAX_H3_OK) return status;
        }
    }
    return MINIMAX_H3_OK;
}

static minimax_h3_status route_entry_count_with_builder(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    minimax_h3_status (*builder)(const minimax_h3_geometry *,
                                const minimax_h3_t2va_layout *,
                                const minimax_h3_m3_tree_node *,
                                const minimax_h3_m3_tree_plan *, size_t,
                                route_writer *),
    size_t *entry_count,
    size_t *maximum_route_entries) {
    size_t leaf;
    size_t total = 0u;
    size_t maximum = 0u;

    if (geometry == NULL || layout == NULL || nodes == NULL || plan == NULL ||
        builder == NULL || entry_count == NULL ||
        maximum_route_entries == NULL) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    for (leaf = 0u; leaf < plan->leaf_count; ++leaf) {
        route_writer writer = {NULL, 0u, 0u};
        minimax_h3_status status = builder(
            geometry, layout, nodes, plan, leaf, &writer);
        if (status != MINIMAX_H3_OK) return status;
        if (!checked_add_size(total, writer.count, &total)) {
            return MINIMAX_H3_OVERFLOW;
        }
        if (writer.count > maximum) maximum = writer.count;
    }
    if (!checked_add_size(total, geometry->video_rows, &total)) {
        return MINIMAX_H3_OVERFLOW;
    }
    if (geometry->video_rows > maximum) maximum = geometry->video_rows;
    *entry_count = total;
    *maximum_route_entries = maximum;
    return MINIMAX_H3_OK;
}

static minimax_h3_status routes_make_with_builder(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    minimax_h3_status (*builder)(const minimax_h3_geometry *,
                                const minimax_h3_t2va_layout *,
                                const minimax_h3_m3_tree_node *,
                                const minimax_h3_m3_tree_plan *, size_t,
                                route_writer *),
    uint32_t *offsets,
    size_t offset_capacity,
    uint32_t *entries,
    size_t entry_capacity) {
    route_writer writer = {entries, entry_capacity, 0u};
    size_t leaf;

    if (geometry == NULL || layout == NULL || nodes == NULL || plan == NULL ||
        builder == NULL || offsets == NULL || entries == NULL) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    if (offset_capacity < plan->leaf_count + 2u) {
        return MINIMAX_H3_INSUFFICIENT_CAPACITY;
    }
    for (leaf = 0u; leaf < plan->leaf_count; ++leaf) {
        minimax_h3_status status;
        if (writer.count > UINT32_MAX) return MINIMAX_H3_OVERFLOW;
        offsets[leaf] = (uint32_t)writer.count;
        status = builder(geometry, layout, nodes, plan, leaf, &writer);
        if (status != MINIMAX_H3_OK) return status;
    }
    if (writer.count > UINT32_MAX) return MINIMAX_H3_OVERFLOW;
    offsets[plan->leaf_count] = (uint32_t)writer.count;
    for (size_t row = layout->video_start; row < layout->sequence_rows; ++row) {
        minimax_h3_status status;
        if (row >= MINIMAX_H3_M3_TREE_SUMMARY_ENTRY) {
            return MINIMAX_H3_OVERFLOW;
        }
        status = route_emit(&writer, (uint32_t)row);
        if (status != MINIMAX_H3_OK) return status;
    }
    if (writer.count > UINT32_MAX) return MINIMAX_H3_OVERFLOW;
    offsets[plan->leaf_count + 1u] = (uint32_t)writer.count;
    return MINIMAX_H3_OK;
}

minimax_h3_status minimax_h3_m3_tree_route_entry_count(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    size_t *entry_count,
    size_t *maximum_route_entries) {
    return route_entry_count_with_builder(
        geometry, layout, nodes, plan, build_one_route, entry_count,
        maximum_route_entries);
}

minimax_h3_status minimax_h3_m3_tree_routes_make(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    uint32_t *offsets,
    size_t offset_capacity,
    uint32_t *entries,
    size_t entry_capacity) {
    return routes_make_with_builder(
        geometry, layout, nodes, plan, build_one_route, offsets,
        offset_capacity, entries, entry_capacity);
}

minimax_h3_status minimax_h3_m3_tree_frame_safe_route_entry_count(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    size_t *entry_count,
    size_t *maximum_route_entries) {
    return route_entry_count_with_builder(
        geometry, layout, nodes, plan, build_one_frame_safe_route,
        entry_count, maximum_route_entries);
}

minimax_h3_status minimax_h3_m3_tree_frame_safe_routes_make(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    uint32_t *offsets,
    size_t offset_capacity,
    uint32_t *entries,
    size_t entry_capacity) {
    return routes_make_with_builder(
        geometry, layout, nodes, plan, build_one_frame_safe_route, offsets,
        offset_capacity, entries, entry_capacity);
}

minimax_h3_status minimax_h3_m3_tree_query_block_count(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    size_t *block_count) {
    size_t count;
    size_t leaf;

    if (geometry == NULL || layout == NULL || nodes == NULL || plan == NULL ||
        block_count == NULL || layout->video_start != plan->exact_count) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    count = ceil_div_size(plan->exact_count, 8u);
    for (leaf = 0u; leaf < plan->leaf_count; ++leaf) {
        if (!checked_add_size(count, nodes[leaf].patch_h, &count)) {
            return MINIMAX_H3_OVERFLOW;
        }
    }
    *block_count = count;
    return MINIMAX_H3_OK;
}

minimax_h3_status minimax_h3_m3_tree_query_blocks_make(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    minimax_h3_m3_query_block *blocks,
    size_t block_capacity) {
    size_t required;
    size_t block = 0u;
    size_t row;
    size_t leaf;
    minimax_h3_status status = minimax_h3_m3_tree_query_block_count(
        geometry, layout, nodes, plan, &required);

    if (status != MINIMAX_H3_OK) return status;
    if (blocks == NULL) return MINIMAX_H3_INVALID_ARGUMENT;
    if (block_capacity < required) return MINIMAX_H3_INSUFFICIENT_CAPACITY;

    for (row = 0u; row < plan->exact_count; row += 8u) {
        size_t count = plan->exact_count - row;
        if (count > 8u) count = 8u;
        blocks[block].first_row = (uint32_t)row;
        blocks[block].row_count = (uint16_t)count;
        blocks[block].route_index = (uint32_t)plan->leaf_count;
        ++block;
    }
    for (leaf = 0u; leaf < plan->leaf_count; ++leaf) {
        const minimax_h3_m3_tree_node *node = &nodes[leaf];
        uint16_t local_y;
        for (local_y = 0u; local_y < node->patch_h; ++local_y) {
            size_t first_row;
            status = minimax_h3_m3_tree_leaf_row(
                geometry, layout, node, local_y, 0u, &first_row);
            if (status != MINIMAX_H3_OK) return status;
            blocks[block].first_row = (uint32_t)first_row;
            blocks[block].row_count = node->patch_w;
            blocks[block].route_index = (uint32_t)leaf;
            ++block;
        }
    }
    return block == required ? MINIMAX_H3_OK : MINIMAX_H3_OVERFLOW;
}

static minimax_h3_status build_one_native_route(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    size_t text_summary_node_start,
    size_t audio_summary_node_start,
    size_t query_leaf,
    route_writer *writer) {
    size_t leaves_per_frame = (size_t)plan->tile_rows * plan->tile_columns;
    size_t query_frame = query_leaf / leaves_per_frame;
    size_t own_temporal = query_frame / MINIMAX_H3_M3_TREE_TEMPORAL_GROUP;
    size_t text_summary_count = ceil_div_size(
        layout->text_count, MINIMAX_H3_M3_TREE_TEXT_SUMMARY_ROWS);
    size_t audio_summary_count = ceil_div_size(
        layout->audio_count, MINIMAX_H3_M3_TREE_AUDIO_SUMMARY_ROWS);
    size_t summary;
    size_t temporal;

    for (summary = 0u; summary < text_summary_count; ++summary) {
        minimax_h3_status status = route_emit(
            writer, MINIMAX_H3_M3_TREE_SUMMARY_ENTRY |
                        (uint32_t)(text_summary_node_start + summary));
        if (status != MINIMAX_H3_OK) return status;
    }
    for (summary = 0u; summary < audio_summary_count; ++summary) {
        minimax_h3_status status = route_emit(
            writer, MINIMAX_H3_M3_TREE_SUMMARY_ENTRY |
                        (uint32_t)(audio_summary_node_start + summary));
        if (status != MINIMAX_H3_OK) return status;
    }
    for (temporal = 0u; temporal < plan->temporal_node_count; ++temporal) {
        size_t temporal_node_index = plan->temporal_node_start + temporal;
        const minimax_h3_m3_tree_node *temporal_node = &nodes[temporal_node_index];
        if (temporal != own_temporal) {
            minimax_h3_status status = route_emit(
                writer, MINIMAX_H3_M3_TREE_SUMMARY_ENTRY |
                            (uint32_t)temporal_node_index);
            if (status != MINIMAX_H3_OK) return status;
        } else {
            size_t frame;
            for (frame = temporal_node->first_frame;
                 frame < (size_t)temporal_node->first_frame +
                             temporal_node->frame_count;
                 ++frame) {
                size_t frame_node_index = plan->frame_node_start + frame;
                if (frame != query_frame) {
                    minimax_h3_status status = route_emit(
                        writer, MINIMAX_H3_M3_TREE_SUMMARY_ENTRY |
                                    (uint32_t)frame_node_index);
                    if (status != MINIMAX_H3_OK) return status;
                } else {
                    size_t frame_leaf_start = frame * leaves_per_frame;
                    size_t leaf;
                    for (leaf = frame_leaf_start;
                         leaf < frame_leaf_start + leaves_per_frame; ++leaf) {
                        if (leaf != query_leaf) {
                            minimax_h3_status status = route_emit(
                                writer, MINIMAX_H3_M3_TREE_SUMMARY_ENTRY |
                                            (uint32_t)leaf);
                            if (status != MINIMAX_H3_OK) return status;
                        } else {
                            const minimax_h3_m3_tree_node *leaf_node = &nodes[leaf];
                            uint16_t y;
                            uint16_t x;
                            for (y = 0u; y < leaf_node->patch_h; ++y) {
                                for (x = 0u; x < leaf_node->patch_w; ++x) {
                                    size_t row;
                                    minimax_h3_status status =
                                        minimax_h3_m3_tree_leaf_row(
                                            geometry, layout, leaf_node, y, x,
                                            &row);
                                    if (status != MINIMAX_H3_OK) return status;
                                    status = route_emit(writer, (uint32_t)row);
                                    if (status != MINIMAX_H3_OK) return status;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return MINIMAX_H3_OK;
}

minimax_h3_status minimax_h3_m3_tree_native_route_entry_count(
    const minimax_h3_geometry *geometry,
    const minimax_h3_t2va_layout *layout,
    const minimax_h3_m3_tree_node *nodes,
    const minimax_h3_m3_tree_plan *plan,
    size_t *text_summary_count,
    size_t *audio_summary_count,
    size_t *entry_count,
    size_t *maximum_route_entries) {
    size_t leaf;
    size_t total = 0u;
    size_t maximum = 0u;
    size_t text_count;
    size_t audio_count;

    if (geometry == NULL || layout == NULL || nodes == NULL || plan == NULL ||
        text_summary_count == NULL || audio_summary_count == NULL ||
        entry_count == NULL ||
        maximum_route_entries == NULL) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    text_count = ceil_div_size(layout->text_count,
                               MINIMAX_H3_M3_TREE_TEXT_SUMMARY_ROWS);
    audio_count = ceil_div_size(layout->audio_count,
                                MINIMAX_H3_M3_TREE_AUDIO_SUMMARY_ROWS);
    for (leaf = 0u; leaf < plan->leaf_count; ++leaf) {
        route_writer writer = {NULL, 0u, 0u};
        minimax_h3_status status = build_one_native_route(
            geometry, layout, nodes, plan, plan->node_count,
            plan->node_count + text_count, leaf, &writer);
        if (status != MINIMAX_H3_OK) return status;
        if (!checked_add_size(total, writer.count, &total))
            return MINIMAX_H3_OVERFLOW;
        if (writer.count > maximum) maximum = writer.count;
    }
    *text_summary_count = text_count;
    *audio_summary_count = audio_count;
    *entry_count = total;
    *maximum_route_entries = maximum;
    return MINIMAX_H3_OK;
}

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
    size_t entry_capacity) {
    route_writer writer = {entries, entry_capacity, 0u};
    size_t text_summary_count = ceil_div_size(
        layout != NULL ? layout->text_count : 0u,
        MINIMAX_H3_M3_TREE_TEXT_SUMMARY_ROWS);
    size_t audio_summary_count = ceil_div_size(
        layout != NULL ? layout->audio_count : 0u,
        MINIMAX_H3_M3_TREE_AUDIO_SUMMARY_ROWS);
    size_t leaf;

    if (geometry == NULL || layout == NULL || nodes == NULL || plan == NULL ||
        offsets == NULL || entries == NULL ||
        text_summary_node_start > UINT32_MAX ||
        audio_summary_node_start > UINT32_MAX ||
        text_summary_count > UINT32_MAX - text_summary_node_start ||
        audio_summary_count > UINT32_MAX - audio_summary_node_start) {
        return MINIMAX_H3_INVALID_ARGUMENT;
    }
    if (offset_capacity < plan->leaf_count + 1u)
        return MINIMAX_H3_INSUFFICIENT_CAPACITY;
    for (leaf = 0u; leaf < plan->leaf_count; ++leaf) {
        minimax_h3_status status;
        if (writer.count > UINT32_MAX) return MINIMAX_H3_OVERFLOW;
        offsets[leaf] = (uint32_t)writer.count;
        status = build_one_native_route(
            geometry, layout, nodes, plan, text_summary_node_start,
            audio_summary_node_start, leaf, &writer);
        if (status != MINIMAX_H3_OK) return status;
    }
    if (writer.count > UINT32_MAX) return MINIMAX_H3_OVERFLOW;
    offsets[plan->leaf_count] = (uint32_t)writer.count;
    return MINIMAX_H3_OK;
}
