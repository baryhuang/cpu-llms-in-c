#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "minimax_h3.h"
#include "minimax_h3_m3_attention.h"
#include "minimax_h3_m3_tree.h"

#include <mach/mach.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    H3_M3_HEADS = 56,
    H3_M3_HEAD_DIM = 128,
    H3_M3_Q_SEED = 1,
    H3_M3_K_SEED = 5,
    H3_M3_V_SEED = 9
};

typedef struct {
    uint32_t parent;
    uint32_t first_child;
    uint32_t child_count;
    uint32_t kind;
    uint32_t first_frame;
    uint32_t frame_count;
    uint32_t patch_y;
    uint32_t patch_x;
    uint32_t patch_h;
    uint32_t patch_w;
    uint32_t token_count;
    uint32_t physical_start;
} h3_tree_node_gpu;

typedef struct {
    uint32_t sequence_rows;
    uint32_t exact_rows;
    uint32_t video_start;
    uint32_t rows_per_video_frame;
    uint32_t patch_columns;
    uint32_t tile_columns;
    uint32_t leaves_per_frame;
    uint32_t leaf_count;
    uint32_t aggregate_start;
    uint32_t aggregate_count;
} h3_tree_parameters;

typedef struct {
    uint32_t first_row;
    uint32_t row_count;
    uint32_t route_index;
} h3_query_block_gpu;

typedef struct {
    id<MTLComputePipelineState> initialize;
    id<MTLComputePipelineState> leaf_summary;
    id<MTLComputePipelineState> parent_summary;
    id<MTLComputePipelineState> attention;
    id<MTLComputePipelineState> attention_mma;
    id<MTLComputePipelineState> attention_mma64;
    id<MTLComputePipelineState> copy_output;
} h3_attention_pipelines;

typedef struct {
    id<MTLBuffer> query;
    id<MTLBuffer> key;
    id<MTLBuffer> value;
    id<MTLBuffer> output;
    id<MTLBuffer> summary_key;
    id<MTLBuffer> summary_value;
    id<MTLBuffer> nodes;
    id<MTLBuffer> summary_log_counts;
    id<MTLBuffer> route_offsets;
    id<MTLBuffer> route_entries;
    id<MTLBuffer> first_output;
    id<MTLBuffer> query_blocks;
    id<MTLBuffer> lse;
} h3_attention_buffers;

static void h3_error(char *message, size_t capacity, NSString *text) {
    if (message == NULL || capacity == 0u) return;
    const char *utf8 = text != nil ? text.UTF8String : "unknown Metal error";
    snprintf(message, capacity, "%s", utf8 != NULL ? utf8 : "unknown Metal error");
}

static size_t h3_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t status = task_info(mach_task_self(), TASK_VM_INFO,
                                     (task_info_t)&info, &count);
    return status == KERN_SUCCESS ? (size_t)info.phys_footprint : 0u;
}

static double h3_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC_RAW, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static id<MTLComputePipelineState> h3_pipeline(id<MTLDevice> device,
                                               id<MTLLibrary> library,
                                               NSString *name,
                                               NSError **error) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) return nil;
    return [device newComputePipelineStateWithFunction:function error:error];
}

static float h3_half_value(uint16_t bits) {
    __fp16 value;
    memcpy(&value, &bits, sizeof(value));
    return (float)value;
}

static float h3_synthetic_value(size_t index, uint32_t seed) {
    int value = (int)(((uint32_t)index * 13u + seed) % 17u) - 8;
    return (float)value * 0.0625f;
}

static void h3_encode_summary_build(id<MTLComputeCommandEncoder> encoder,
                                    const h3_attention_pipelines *pipelines,
                                    const h3_attention_buffers *buffers,
                                    h3_tree_parameters parameters,
                                    const minimax_h3_m3_tree_plan *plan,
                                    size_t audio_summary_start,
                                    size_t audio_summary_count) {
    parameters.aggregate_start = 0u;
    parameters.aggregate_count = (uint32_t)plan->leaf_count;
    [encoder setComputePipelineState:pipelines->leaf_summary];
    [encoder setBuffer:buffers->key offset:0 atIndex:0];
    [encoder setBuffer:buffers->value offset:0 atIndex:1];
    [encoder setBuffer:buffers->nodes offset:0 atIndex:2];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder setBuffer:buffers->summary_key offset:0 atIndex:4];
    [encoder setBuffer:buffers->summary_value offset:0 atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake(plan->leaf_count * H3_M3_HEADS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    parameters.aggregate_start = (uint32_t)audio_summary_start;
    parameters.aggregate_count = (uint32_t)audio_summary_count;
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder dispatchThreadgroups:
        MTLSizeMake(audio_summary_count * H3_M3_HEADS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    [encoder setComputePipelineState:pipelines->parent_summary];
    [encoder setBuffer:buffers->key offset:0 atIndex:0];
    [encoder setBuffer:buffers->value offset:0 atIndex:1];
    [encoder setBuffer:buffers->nodes offset:0 atIndex:2];
    [encoder setBuffer:buffers->summary_key offset:0 atIndex:4];
    [encoder setBuffer:buffers->summary_value offset:0 atIndex:5];

    parameters.aggregate_start = (uint32_t)plan->frame_node_start;
    parameters.aggregate_count = (uint32_t)plan->frame_node_count;
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder dispatchThreadgroups:
        MTLSizeMake(plan->frame_node_count * H3_M3_HEADS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    parameters.aggregate_start = (uint32_t)plan->temporal_node_start;
    parameters.aggregate_count = (uint32_t)plan->temporal_node_count;
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder dispatchThreadgroups:
        MTLSizeMake(plan->temporal_node_count * H3_M3_HEADS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];

    parameters.aggregate_start = (uint32_t)plan->root_index;
    parameters.aggregate_count = 1u;
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(H3_M3_HEADS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
}

static void h3_encode_attention(id<MTLComputeCommandEncoder> encoder,
                                const h3_attention_pipelines *pipelines,
                                const h3_attention_buffers *buffers,
                                h3_tree_parameters parameters,
                                size_t query_block_count) {
    [encoder setComputePipelineState:pipelines->attention_mma64];
    [encoder setBuffer:buffers->query offset:0 atIndex:0];
    [encoder setBuffer:buffers->key offset:0 atIndex:1];
    [encoder setBuffer:buffers->value offset:0 atIndex:2];
    [encoder setBuffer:buffers->summary_key offset:0 atIndex:3];
    [encoder setBuffer:buffers->summary_value offset:0 atIndex:4];
    [encoder setBuffer:buffers->summary_log_counts offset:0 atIndex:5];
    [encoder setBuffer:buffers->route_offsets offset:0 atIndex:6];
    [encoder setBuffer:buffers->route_entries offset:0 atIndex:7];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:8];
    [encoder setBuffer:buffers->output offset:0 atIndex:9];
    [encoder setBuffer:buffers->query_blocks offset:0 atIndex:10];
    [encoder setBuffer:buffers->lse offset:0 atIndex:11];
    [encoder dispatchThreadgroups:MTLSizeMake(query_block_count, H3_M3_HEADS, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static int h3_wait(id<MTLCommandBuffer> command,
                   char *error_message,
                   size_t error_capacity) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
        h3_error(error_message, error_capacity, command.error.localizedDescription);
        return -1;
    }
    return 0;
}

static float h3_reference_component(
    size_t dimension,
    const uint32_t *route_offsets,
    const uint32_t *route_entries,
    const uint16_t *summary_keys,
    const uint16_t *summary_values,
    const float *summary_log_counts) {
    float q[H3_M3_HEAD_DIM];
    float maximum = -INFINITY;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    size_t d;
    size_t route;

    for (d = 0u; d < H3_M3_HEAD_DIM; ++d) {
        q[d] = h3_synthetic_value(d, H3_M3_Q_SEED);
    }
    for (route = route_offsets[0]; route < route_offsets[1]; ++route) {
        uint32_t encoded = route_entries[route];
        int is_summary =
            (encoded & MINIMAX_H3_M3_TREE_SUMMARY_ENTRY) != 0u;
        size_t index = encoded & ~MINIMAX_H3_M3_TREE_SUMMARY_ENTRY;
        float score = 0.0f;
        size_t base = (index * H3_M3_HEADS) * H3_M3_HEAD_DIM;
        for (d = 0u; d < H3_M3_HEAD_DIM; ++d) {
            float key = is_summary ? h3_half_value(summary_keys[base + d])
                                   : h3_synthetic_value(base + d,
                                                        H3_M3_K_SEED);
            score += q[d] * key;
        }
        score *= 0.08838834764831845f;
        if (is_summary) score += summary_log_counts[index];
        float value = is_summary
                          ? h3_half_value(summary_values[base + dimension])
                          : h3_synthetic_value(base + dimension, H3_M3_V_SEED);
        float next_maximum = fmaxf(maximum, score);
        float old_scale = expf(maximum - next_maximum);
        float new_scale = expf(score - next_maximum);
        accumulator = accumulator * old_scale + value * new_scale;
        denominator = denominator * old_scale + new_scale;
        maximum = next_maximum;
    }
    return accumulator / denominator;
}

int minimax_h3_m3_run_attention_benchmark(
    const char *metallib_path,
    unsigned warmup_iterations,
    unsigned measured_iterations,
    minimax_h3_m3_attention_result *result,
    char *error_message,
    size_t error_message_capacity) {
    if (metallib_path == NULL || result == NULL || warmup_iterations == 0u ||
        measured_iterations == 0u) {
        h3_error(error_message, error_message_capacity, @"invalid argument");
        return 2;
    }

    @autoreleasepool {
        minimax_h3_geometry geometry;
        minimax_h3_t2va_layout layout;
        minimax_h3_m3_tree_plan tree_plan;
        minimax_h3_m3_tree_node *tree_nodes = NULL;
        double *positions = NULL;
        uint8_t *tags = NULL;
        uint32_t *route_offsets = NULL;
        uint32_t *route_entries = NULL;
        h3_query_block_gpu *query_blocks = NULL;
        uint32_t *logical_to_physical = NULL;
        size_t tree_node_count;
        size_t route_entry_count;
        size_t maximum_route_entries;
        size_t text_summary_count;
        size_t audio_summary_count;
        size_t summary_node_count;
        size_t query_block_count;
        size_t tensor_elements;
        size_t tensor_bytes;
        size_t summary_elements;
        size_t summary_bytes;
        minimax_h3_m3_attention_result measured;
        id<MTLDevice> device = nil;
        id<MTLLibrary> library = nil;
        id<MTLCommandQueue> queue = nil;
        h3_attention_pipelines pipelines = {0};
        h3_attention_buffers buffers = {0};
        h3_tree_parameters parameters = {0};
        NSError *error = nil;
        unsigned iteration;
        double summary_gpu_seconds = 0.0;
        double summary_wall_seconds = 0.0;
        double attention_gpu_seconds = 0.0;
        double attention_wall_seconds = 0.0;
        int return_status = 1;

        memset(&measured, 0, sizeof(measured));
        measured.footprint_before_bytes = h3_footprint();
        if (minimax_h3_geometry_init(&geometry, 864, 480, 124, 86) !=
            MINIMAX_H3_OK) {
            h3_error(error_message, error_message_capacity, @"geometry failure");
            goto cleanup;
        }
        positions = calloc(geometry.sequence_rows * 3u, sizeof(*positions));
        tags = calloc(geometry.sequence_rows, sizeof(*tags));
        if (positions == NULL || tags == NULL ||
            minimax_h3_build_t2va_layout(&geometry, NULL, positions, tags,
                                         geometry.sequence_rows, &layout) !=
                MINIMAX_H3_OK ||
            minimax_h3_m3_tree_node_count(&geometry, &tree_node_count) !=
                MINIMAX_H3_OK) {
            h3_error(error_message, error_message_capacity,
                     @"layout allocation failure");
            goto cleanup;
        }
        tree_nodes = calloc(tree_node_count, sizeof(*tree_nodes));
        if (tree_nodes == NULL ||
            minimax_h3_m3_tree_plan_make(&geometry, &layout, tree_nodes,
                                         tree_node_count, &tree_plan) !=
                MINIMAX_H3_OK ||
            minimax_h3_m3_tree_native_route_entry_count(
                &geometry, &layout, tree_nodes, &tree_plan,
                &text_summary_count, &audio_summary_count, &route_entry_count,
                &maximum_route_entries) != MINIMAX_H3_OK) {
            h3_error(error_message, error_message_capacity, @"tree plan failure");
            goto cleanup;
        }
        query_block_count = (tree_plan.exact_count + 63u) / 64u +
                            tree_plan.leaf_count;
        summary_node_count = tree_plan.node_count + text_summary_count +
                             audio_summary_count;
        route_offsets = calloc(tree_plan.leaf_count + 1u,
                               sizeof(*route_offsets));
        route_entries = calloc(route_entry_count, sizeof(*route_entries));
        query_blocks = calloc(query_block_count, sizeof(*query_blocks));
        logical_to_physical = calloc(geometry.sequence_rows,
                                     sizeof(*logical_to_physical));
        if (route_offsets == NULL || route_entries == NULL ||
            query_blocks == NULL || logical_to_physical == NULL ||
            minimax_h3_m3_tree_native_routes_make(
                &geometry, &layout, tree_nodes, &tree_plan,
                tree_plan.node_count,
                tree_plan.node_count + text_summary_count, route_offsets,
                tree_plan.leaf_count + 1u, route_entries, route_entry_count) !=
                MINIMAX_H3_OK) {
            h3_error(error_message, error_message_capacity,
                     @"route allocation failure");
            goto cleanup;
        }
        {
            size_t block = 0u;
            size_t physical_row = tree_plan.video_start;
            for (size_t row = 0u; row < tree_plan.exact_count; ++row)
                logical_to_physical[row] = (uint32_t)row;
            for (size_t row = 0u; row < tree_plan.exact_count; row += 64u) {
                size_t count = tree_plan.exact_count - row;
                if (count > 64u) count = 64u;
                query_blocks[block++] = (h3_query_block_gpu){
                    .first_row = (uint32_t)row,
                    .row_count = (uint32_t)count,
                    .route_index = 0u
                };
            }
            for (size_t leaf = 0u; leaf < tree_plan.leaf_count; ++leaf) {
                const minimax_h3_m3_tree_node *node = &tree_nodes[leaf];
                query_blocks[block++] = (h3_query_block_gpu){
                    .first_row = (uint32_t)physical_row,
                    .row_count = node->token_count,
                    .route_index = (uint32_t)leaf
                };
                for (uint16_t y = 0u; y < node->patch_h; ++y) {
                    for (uint16_t x = 0u; x < node->patch_w; ++x) {
                        size_t logical_row;
                        if (minimax_h3_m3_tree_leaf_row(
                                &geometry, &layout, node, y, x,
                                &logical_row) != MINIMAX_H3_OK) {
                            h3_error(error_message, error_message_capacity,
                                     @"physical row mapping failure");
                            goto cleanup;
                        }
                        logical_to_physical[logical_row] =
                            (uint32_t)physical_row++;
                    }
                }
            }
            if (block != query_block_count ||
                physical_row != geometry.sequence_rows) {
                h3_error(error_message, error_message_capacity,
                         @"physical row mapping is incomplete");
                goto cleanup;
            }
            for (size_t entry = 0u; entry < route_entry_count; ++entry) {
                if ((route_entries[entry] &
                     MINIMAX_H3_M3_TREE_SUMMARY_ENTRY) == 0u) {
                    route_entries[entry] =
                        logical_to_physical[route_entries[entry]];
                }
            }
        }

        tensor_elements = geometry.sequence_rows * H3_M3_HEADS * H3_M3_HEAD_DIM;
        tensor_bytes = tensor_elements * sizeof(uint16_t);
        summary_elements = summary_node_count * H3_M3_HEADS * H3_M3_HEAD_DIM;
        summary_bytes = summary_elements * sizeof(uint16_t);

        device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            h3_error(error_message, error_message_capacity, @"Metal unavailable");
            goto cleanup;
        }
        library = [device newLibraryWithURL:[NSURL fileURLWithPath:
                    [NSString stringWithUTF8String:metallib_path]] error:&error];
        if (library == nil) {
            h3_error(error_message, error_message_capacity,
                     error.localizedDescription);
            goto cleanup;
        }
        queue = [device newCommandQueue];
        pipelines.initialize = h3_pipeline(device, library,
                                            @"minimax_h3_initialize_half", &error);
        pipelines.leaf_summary = h3_pipeline(
            device, library, @"minimax_h3_build_leaf_summaries", &error);
        pipelines.parent_summary = h3_pipeline(
            device, library, @"minimax_h3_build_parent_summaries", &error);
        pipelines.attention = h3_pipeline(
            device, library, @"minimax_h3_hierarchical_attention", &error);
        pipelines.attention_mma = h3_pipeline(
            device, library, @"minimax_h3_hierarchical_attention_mma", &error);
        pipelines.attention_mma64 = h3_pipeline(
            device, library, @"minimax_h3_hierarchical_attention_mma64", &error);
        pipelines.copy_output = h3_pipeline(
            device, library, @"minimax_h3_copy_first_output", &error);
        if (queue == nil || pipelines.initialize == nil ||
            pipelines.leaf_summary == nil || pipelines.parent_summary == nil ||
            pipelines.attention == nil || pipelines.attention_mma == nil ||
            pipelines.attention_mma64 == nil || pipelines.copy_output == nil) {
            h3_error(error_message, error_message_capacity,
                     error.localizedDescription);
            goto cleanup;
        }

        buffers.query = [device newBufferWithLength:tensor_bytes
                                             options:MTLResourceStorageModePrivate];
        buffers.key = [device newBufferWithLength:tensor_bytes
                                           options:MTLResourceStorageModePrivate];
        buffers.value = [device newBufferWithLength:tensor_bytes
                                             options:MTLResourceStorageModePrivate];
        buffers.output = [device newBufferWithLength:tensor_bytes
                                              options:MTLResourceStorageModePrivate];
        buffers.summary_key = [device newBufferWithLength:summary_bytes
                                                   options:MTLResourceStorageModeShared];
        buffers.summary_value = [device newBufferWithLength:summary_bytes
                                                     options:MTLResourceStorageModeShared];
        buffers.nodes = [device newBufferWithLength:summary_node_count *
                                                 sizeof(h3_tree_node_gpu)
                                            options:MTLResourceStorageModeShared];
        buffers.summary_log_counts = [device newBufferWithLength:
                                                 summary_node_count * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
        buffers.route_offsets = [device newBufferWithBytes:route_offsets
                                                    length:(tree_plan.leaf_count + 1u) *
                                                           sizeof(*route_offsets)
                                                   options:MTLResourceStorageModeShared];
        buffers.route_entries = [device newBufferWithBytes:route_entries
                                                    length:route_entry_count *
                                                           sizeof(*route_entries)
                                                   options:MTLResourceStorageModeShared];
        buffers.first_output = [device newBufferWithLength:H3_M3_HEAD_DIM *
                                                            sizeof(uint16_t)
                                                   options:MTLResourceStorageModeShared];
        buffers.query_blocks = [device newBufferWithBytes:query_blocks
                                                   length:query_block_count *
                                                          sizeof(h3_query_block_gpu)
                                                  options:MTLResourceStorageModeShared];
        buffers.lse = [device newBufferWithLength:geometry.sequence_rows *
                                                H3_M3_HEADS * sizeof(float)
                                           options:MTLResourceStorageModePrivate];
        if (buffers.query == nil || buffers.key == nil || buffers.value == nil ||
            buffers.output == nil || buffers.summary_key == nil ||
            buffers.summary_value == nil || buffers.nodes == nil ||
            buffers.summary_log_counts == nil || buffers.route_offsets == nil ||
            buffers.route_entries == nil || buffers.first_output == nil ||
            buffers.query_blocks == nil || buffers.lse == nil) {
            h3_error(error_message, error_message_capacity,
                     @"Metal buffer allocation failure");
            goto cleanup;
        }

        h3_tree_node_gpu *gpu_nodes = buffers.nodes.contents;
        float *gpu_log_counts = buffers.summary_log_counts.contents;
        size_t leaf_physical_start = tree_plan.video_start;
        for (size_t index = 0u; index < tree_plan.node_count; ++index) {
            gpu_nodes[index] = (h3_tree_node_gpu){
                .parent = tree_nodes[index].parent,
                .first_child = tree_nodes[index].first_child,
                .child_count = tree_nodes[index].child_count,
                .kind = tree_nodes[index].kind,
                .first_frame = tree_nodes[index].first_frame,
                .frame_count = tree_nodes[index].frame_count,
                .patch_y = tree_nodes[index].patch_y,
                .patch_x = tree_nodes[index].patch_x,
                .patch_h = tree_nodes[index].patch_h,
                .patch_w = tree_nodes[index].patch_w,
                .token_count = tree_nodes[index].token_count,
                .physical_start = index < tree_plan.leaf_count
                                      ? (uint32_t)leaf_physical_start
                                      : 0u
            };
            if (index < tree_plan.leaf_count)
                leaf_physical_start += tree_nodes[index].token_count;
            gpu_log_counts[index] = logf((float)tree_nodes[index].token_count);
        }
        for (size_t text_index = 0u; text_index < text_summary_count;
             ++text_index) {
            size_t node_index = tree_plan.node_count + text_index;
            size_t text_start = text_index *
                                MINIMAX_H3_M3_TREE_TEXT_SUMMARY_ROWS;
            size_t text_rows = layout.text_count - text_start;
            if (text_rows > MINIMAX_H3_M3_TREE_TEXT_SUMMARY_ROWS)
                text_rows = MINIMAX_H3_M3_TREE_TEXT_SUMMARY_ROWS;
            gpu_nodes[node_index] = (h3_tree_node_gpu){
                .parent = MINIMAX_H3_M3_TREE_NO_PARENT,
                .kind = 5u,
                .token_count = (uint32_t)text_rows,
                .physical_start = (uint32_t)(layout.text_start + text_start)
            };
            gpu_log_counts[node_index] = logf((float)text_rows);
        }
        for (size_t audio_index = 0u; audio_index < audio_summary_count;
             ++audio_index) {
            size_t node_index = tree_plan.node_count + text_summary_count +
                                audio_index;
            size_t audio_start = audio_index *
                                 MINIMAX_H3_M3_TREE_AUDIO_SUMMARY_ROWS;
            size_t audio_rows = layout.audio_count - audio_start;
            if (audio_rows > MINIMAX_H3_M3_TREE_AUDIO_SUMMARY_ROWS)
                audio_rows = MINIMAX_H3_M3_TREE_AUDIO_SUMMARY_ROWS;
            gpu_nodes[node_index] = (h3_tree_node_gpu){
                .parent = MINIMAX_H3_M3_TREE_NO_PARENT,
                .kind = 4u,
                .token_count = (uint32_t)audio_rows,
                .physical_start = (uint32_t)(layout.audio_start + audio_start)
            };
            gpu_log_counts[node_index] = logf((float)audio_rows);
        }
        parameters.sequence_rows = (uint32_t)geometry.sequence_rows;
        parameters.exact_rows = 0u;
        parameters.video_start = (uint32_t)layout.video_start;
        parameters.rows_per_video_frame =
            (uint32_t)geometry.rows_per_video_frame;
        parameters.patch_columns = tree_plan.patch_columns;
        parameters.tile_columns = tree_plan.tile_columns;
        parameters.leaves_per_frame =
            (uint32_t)tree_plan.tile_rows * tree_plan.tile_columns;
        parameters.leaf_count = (uint32_t)tree_plan.leaf_count;

        {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            uint32_t count = (uint32_t)tensor_elements;
            const id<MTLBuffer> tensors[3] = {
                buffers.query, buffers.key, buffers.value
            };
            const uint32_t seeds[3] = {H3_M3_Q_SEED, H3_M3_K_SEED,
                                       H3_M3_V_SEED};
            [encoder setComputePipelineState:pipelines.initialize];
            for (size_t tensor = 0u; tensor < 3u; ++tensor) {
                [encoder setBuffer:tensors[tensor] offset:0 atIndex:0];
                [encoder setBytes:&count length:sizeof(count) atIndex:1];
                [encoder setBytes:&seeds[tensor] length:sizeof(seeds[tensor])
                           atIndex:2];
                [encoder dispatchThreads:MTLSizeMake(tensor_elements, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            [encoder endEncoding];
            if (h3_wait(command, error_message, error_message_capacity) != 0)
                goto cleanup;
        }

        for (iteration = 0u;
             iteration < warmup_iterations + measured_iterations; ++iteration) {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            double start = h3_seconds();
            h3_encode_summary_build(encoder, &pipelines, &buffers, parameters,
                                    &tree_plan, tree_plan.node_count,
                                    text_summary_count + audio_summary_count);
            [encoder endEncoding];
            if (h3_wait(command, error_message, error_message_capacity) != 0)
                goto cleanup;
            if (iteration >= warmup_iterations) {
                summary_wall_seconds += h3_seconds() - start;
                summary_gpu_seconds += command.GPUEndTime - command.GPUStartTime;
            }
        }

        for (iteration = 0u;
             iteration < warmup_iterations + measured_iterations; ++iteration) {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            double start = h3_seconds();
            h3_encode_attention(encoder, &pipelines, &buffers, parameters,
                                query_block_count);
            [encoder endEncoding];
            if (h3_wait(command, error_message, error_message_capacity) != 0)
                goto cleanup;
            if (iteration >= warmup_iterations) {
                attention_wall_seconds += h3_seconds() - start;
                attention_gpu_seconds += command.GPUEndTime - command.GPUStartTime;
            }
        }

        {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:pipelines.copy_output];
            [encoder setBuffer:buffers.output offset:0 atIndex:0];
            [encoder setBuffer:buffers.first_output offset:0 atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(H3_M3_HEAD_DIM, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
            if (h3_wait(command, error_message, error_message_capacity) != 0)
                goto cleanup;
        }

        measured.warmup_iterations = warmup_iterations;
        measured.measured_iterations = measured_iterations;
        measured.sequence_rows = geometry.sequence_rows;
        measured.exact_rows = 0u;
        measured.video_rows = geometry.video_rows;
        measured.tree_nodes = summary_node_count;
        measured.query_blocks = query_block_count;
        measured.route_entries = route_entry_count;
        measured.maximum_route_entries = maximum_route_entries;
        measured.qkv_bytes = tensor_bytes * 3u;
        measured.summary_bytes = summary_bytes * 2u;
        measured.output_bytes = tensor_bytes;
        measured.lse_bytes = geometry.sequence_rows * H3_M3_HEADS * sizeof(float);
        measured.metal_owned_buffer_bytes = tensor_bytes * 4u +
            summary_bytes * 2u + summary_node_count *
                (sizeof(h3_tree_node_gpu) + sizeof(float)) +
            (tree_plan.leaf_count + 1u) * sizeof(uint32_t) +
            route_entry_count * sizeof(uint32_t) +
            H3_M3_HEAD_DIM * sizeof(uint16_t) +
            query_block_count * sizeof(h3_query_block_gpu) + measured.lse_bytes;
        measured.summary_gpu_ms =
            summary_gpu_seconds * 1000.0 / measured_iterations;
        measured.summary_wall_ms =
            summary_wall_seconds * 1000.0 / measured_iterations;
        measured.attention_gpu_ms =
            attention_gpu_seconds * 1000.0 / measured_iterations;
        measured.attention_wall_ms =
            attention_wall_seconds * 1000.0 / measured_iterations;
        measured.footprint_peak_bytes = h3_footprint();
        snprintf(measured.device_name, sizeof(measured.device_name), "%s",
                 device.name.UTF8String);

        const uint16_t *metal_output = buffers.first_output.contents;
        const uint16_t *summary_keys = buffers.summary_key.contents;
        const uint16_t *summary_values = buffers.summary_value.contents;
        for (size_t dimension = 0u; dimension < H3_M3_HEAD_DIM; ++dimension) {
            float reference = h3_reference_component(
                dimension, route_offsets, route_entries,
                summary_keys, summary_values,
                buffers.summary_log_counts.contents);
            float actual = h3_half_value(metal_output[dimension]);
            float error_value = fabsf(reference - actual);
            if (error_value > measured.max_abs_error_first_head)
                measured.max_abs_error_first_head = error_value;
            if (dimension < 8u) {
                measured.reference_first_8[dimension] = reference;
                measured.metal_first_8[dimension] = actual;
            }
        }
        *result = measured;
        return_status = 0;

cleanup:
        free(positions);
        free(tags);
        free(tree_nodes);
        free(route_offsets);
        free(route_entries);
        free(query_blocks);
        free(logical_to_physical);
        return return_status;
    }
}
