#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "minimax_h3.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    H3_HEADS = 56,
    H3_HEAD_DIM = 128,
    H3_QKV_WIDTH = 3 * H3_HEADS * H3_HEAD_DIM,
    H3_ROPE_SLOTS = 48,
    H3_TEXT_ROWS = 240,
    H3_TEST_ROWS = 8
};

typedef struct {
    id<MTLComputePipelineState> initialize_bf16;
    id<MTLComputePipelineState> build_rope;
    id<MTLComputePipelineState> reference_prepare;
    id<MTLComputePipelineState> precomputed_prepare;
} h3_rope_pipelines;

typedef struct {
    id<MTLBuffer> projected;
    id<MTLBuffer> positions;
    id<MTLBuffer> rotary;
    id<MTLBuffer> norm;
    id<MTLBuffer> query;
    id<MTLBuffer> key;
    id<MTLBuffer> value;
} h3_rope_buffers;

static double h3_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC_RAW, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static unsigned h3_iterations(const char *text) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value == 0u ||
        value > 100u) {
        fprintf(stderr, "invalid iteration count: %s\n", text);
        exit(2);
    }
    return (unsigned)value;
}

static id<MTLComputePipelineState> h3_pipeline(id<MTLDevice> device,
                                               id<MTLLibrary> library,
                                               NSString *name,
                                               NSError **error) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    return function == nil
        ? nil
        : [device newComputePipelineStateWithFunction:function error:error];
}

static int h3_wait(id<MTLCommandBuffer> command) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "Metal command failed: %s\n",
                command.error.localizedDescription.UTF8String);
        return 1;
    }
    return 0;
}

static uint16_t h3_bfloat_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16u);
}

static float h3_bfloat_value(uint16_t value) {
    uint32_t bits = (uint32_t)value << 16u;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static int h3_build_rope(id<MTLCommandQueue> queue,
                         const h3_rope_pipelines *pipelines,
                         const h3_rope_buffers *buffers,
                         uint32_t rows,
                         double *wall_ms,
                         double *gpu_ms) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    double started = h3_seconds();
    [encoder setComputePipelineState:pipelines->build_rope];
    [encoder setBuffer:buffers->positions offset:0 atIndex:0];
    [encoder setBuffer:buffers->rotary offset:0 atIndex:1];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake((size_t)rows * H3_ROPE_SLOTS, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [encoder endEncoding];
    if (h3_wait(command) != 0) return 1;
    *wall_ms = (h3_seconds() - started) * 1000.0;
    *gpu_ms = (command.GPUEndTime - command.GPUStartTime) * 1000.0;
    return 0;
}

static int h3_prepare(id<MTLCommandQueue> queue,
                      id<MTLComputePipelineState> pipeline,
                      const h3_rope_buffers *buffers,
                      uint32_t rows,
                      int precomputed,
                      double *wall_ms,
                      double *gpu_ms) {
    int status = 0;
    @autoreleasepool {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        uint32_t apply_rotary = 1u;
        double started = h3_seconds();
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffers->projected offset:0 atIndex:0];
        [encoder setBuffer:buffers->norm offset:0 atIndex:1];
        [encoder setBuffer:buffers->norm offset:0 atIndex:2];
        [encoder setBuffer:precomputed ? buffers->rotary : buffers->positions
                     offset:0 atIndex:3];
        [encoder setBuffer:buffers->query offset:0 atIndex:4];
        [encoder setBuffer:buffers->key offset:0 atIndex:5];
        [encoder setBuffer:buffers->value offset:0 atIndex:6];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:7];
        if (precomputed)
            [encoder setBytes:&apply_rotary length:sizeof(apply_rotary)
                       atIndex:8];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 2u * H3_HEADS, 1u)
                    threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        [encoder endEncoding];
        if (h3_wait(command) != 0) {
            status = 1;
        } else {
            *wall_ms = (h3_seconds() - started) * 1000.0;
            *gpu_ms = (command.GPUEndTime - command.GPUStartTime) * 1000.0;
        }
    }
    return status;
}

static void h3_fill_projected(uint16_t *values, size_t count, uint32_t seed) {
    for (size_t index = 0u; index < count; ++index) {
        int value = (int)(((uint32_t)index * 13u + seed) % 17u) - 8;
        values[index] = h3_bfloat_bits((float)value * 0.0625f);
    }
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s METALLIB [MEASURED] [WARMUP]\n", argv[0]);
        return 2;
    }
    unsigned measured = argc > 2 ? h3_iterations(argv[2]) : 3u;
    unsigned warmup = argc > 3 ? h3_iterations(argv[3]) : 1u;

    @autoreleasepool {
        minimax_h3_geometry geometry;
        minimax_h3_t2va_layout layout;
        double *positions64 = NULL;
        uint8_t *tags = NULL;
        float *positions32 = NULL;
        NSError *error = nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            fprintf(stderr, "Metal is unavailable\n");
            return 1;
        }
        id<MTLLibrary> library = [device newLibraryWithURL:
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
                                                    error:&error];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        h3_rope_pipelines pipelines = {
            .initialize_bf16 = h3_pipeline(
                device, library, @"minimax_h3_initialize_bf16", &error),
            .build_rope = h3_pipeline(
                device, library, @"minimax_h3_build_rope_f32", &error),
            .reference_prepare = h3_pipeline(
                device, library,
                @"minimax_h3_prepare_qkv_bf16_reference", &error),
            .precomputed_prepare = h3_pipeline(
                device, library, @"minimax_h3_prepare_qkv_bf16", &error),
        };
        if (library == nil || queue == nil || pipelines.initialize_bf16 == nil ||
            pipelines.build_rope == nil || pipelines.reference_prepare == nil ||
            pipelines.precomputed_prepare == nil) {
            fprintf(stderr, "Metal setup failed: %s\n",
                    error.localizedDescription.UTF8String);
            return 1;
        }

        if (minimax_h3_geometry_init(&geometry, 864u, 480u, 124u,
                                     H3_TEXT_ROWS) != MINIMAX_H3_OK) {
            fprintf(stderr, "H3 geometry failed\n");
            return 1;
        }
        positions64 = calloc(geometry.sequence_rows * 3u, sizeof(*positions64));
        tags = calloc(geometry.sequence_rows, sizeof(*tags));
        positions32 = calloc(geometry.sequence_rows * 3u, sizeof(*positions32));
        if (positions64 == NULL || tags == NULL || positions32 == NULL ||
            minimax_h3_build_t2va_layout(&geometry, NULL, positions64, tags,
                                         geometry.sequence_rows,
                                         &layout) != MINIMAX_H3_OK) {
            fprintf(stderr, "H3 layout allocation failed\n");
            free(positions64); free(tags); free(positions32);
            return 1;
        }
        for (size_t index = 0u; index < geometry.sequence_rows * 3u; ++index)
            positions32[index] = (float)positions64[index];

        size_t rows = geometry.sequence_rows;
        size_t projected_elements = rows * H3_QKV_WIDTH;
        size_t head_elements = rows * H3_HEADS * H3_HEAD_DIM;
        h3_rope_buffers buffers = {
            .projected = [device newBufferWithLength:projected_elements * 2u
                                             options:MTLResourceStorageModeShared],
            .positions = [device newBufferWithBytes:positions32
                                             length:rows * 3u * sizeof(float)
                                            options:MTLResourceStorageModeShared],
            .rotary = [device newBufferWithLength:rows * H3_ROPE_SLOTS *
                                                   2u * sizeof(float)
                                          options:MTLResourceStorageModePrivate],
            .norm = [device newBufferWithLength:H3_HEAD_DIM * sizeof(uint16_t)
                                          options:MTLResourceStorageModeShared],
            .query = [device newBufferWithLength:head_elements * 2u
                                           options:MTLResourceStorageModeShared],
            .key = [device newBufferWithLength:head_elements * 2u
                                         options:MTLResourceStorageModeShared],
            .value = [device newBufferWithLength:head_elements * 2u
                                           options:MTLResourceStorageModeShared],
        };
        if (buffers.projected == nil || buffers.positions == nil ||
            buffers.rotary == nil || buffers.norm == nil ||
            buffers.query == nil || buffers.key == nil || buffers.value == nil) {
            fprintf(stderr, "full-size Metal allocation failed\n");
            free(positions64); free(tags); free(positions32);
            return 1;
        }
        uint16_t *norm = buffers.norm.contents;
        for (size_t index = 0u; index < H3_HEAD_DIM; ++index)
            norm[index] = h3_bfloat_bits(1.0f);

        {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            uint32_t count = (uint32_t)projected_elements;
            uint32_t seed = 3u;
            [encoder setComputePipelineState:pipelines.initialize_bf16];
            [encoder setBuffer:buffers.projected offset:0 atIndex:0];
            [encoder setBytes:&count length:sizeof(count) atIndex:1];
            [encoder setBytes:&seed length:sizeof(seed) atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(projected_elements, 1u, 1u)
                     threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
            [encoder endEncoding];
            if (h3_wait(command) != 0) {
                free(positions64); free(tags); free(positions32);
                return 1;
            }
        }

        double rope_wall_ms = 0.0;
        double rope_gpu_ms = 0.0;
        if (h3_build_rope(queue, &pipelines, &buffers, (uint32_t)rows,
                          &rope_wall_ms, &rope_gpu_ms) != 0) {
            free(positions64); free(tags); free(positions32);
            return 1;
        }

        size_t test_projected_elements = H3_TEST_ROWS * H3_QKV_WIDTH;
        size_t test_head_elements = H3_TEST_ROWS * H3_HEADS * H3_HEAD_DIM;
        h3_rope_buffers reference = {
            .projected = [device newBufferWithLength:test_projected_elements * 2u
                                               options:MTLResourceStorageModeShared],
            .positions = buffers.positions,
            .rotary = buffers.rotary,
            .norm = buffers.norm,
            .query = [device newBufferWithLength:test_head_elements * 2u
                                           options:MTLResourceStorageModeShared],
            .key = [device newBufferWithLength:test_head_elements * 2u
                                         options:MTLResourceStorageModeShared],
            .value = [device newBufferWithLength:test_head_elements * 2u
                                           options:MTLResourceStorageModeShared],
        };
        h3_rope_buffers optimized = reference;
        optimized.query = [device newBufferWithLength:test_head_elements * 2u
                                                options:MTLResourceStorageModeShared];
        optimized.key = [device newBufferWithLength:test_head_elements * 2u
                                              options:MTLResourceStorageModeShared];
        optimized.value = [device newBufferWithLength:test_head_elements * 2u
                                                options:MTLResourceStorageModeShared];
        if (reference.projected == nil || reference.query == nil ||
            reference.key == nil || reference.value == nil ||
            optimized.query == nil || optimized.key == nil ||
            optimized.value == nil) {
            fprintf(stderr, "differential allocation failed\n");
            free(positions64); free(tags); free(positions32);
            return 1;
        }
        h3_fill_projected(reference.projected.contents,
                          test_projected_elements, 3u);
        double ignored_wall = 0.0;
        double ignored_gpu = 0.0;
        if (h3_prepare(queue, pipelines.reference_prepare, &reference,
                       H3_TEST_ROWS, 0, &ignored_wall, &ignored_gpu) != 0 ||
            h3_prepare(queue, pipelines.precomputed_prepare, &optimized,
                       H3_TEST_ROWS, 1, &ignored_wall, &ignored_gpu) != 0) {
            free(positions64); free(tags); free(positions32);
            return 1;
        }
        size_t differing = 0u;
        float maximum_error = 0.0f;
        const id<MTLBuffer> reference_outputs[3] = {
            reference.query, reference.key, reference.value
        };
        const id<MTLBuffer> optimized_outputs[3] = {
            optimized.query, optimized.key, optimized.value
        };
        for (size_t output = 0u; output < 3u; ++output) {
            const uint16_t *expected = reference_outputs[output].contents;
            const uint16_t *actual = optimized_outputs[output].contents;
            for (size_t index = 0u; index < test_head_elements; ++index) {
                if (expected[index] != actual[index]) ++differing;
                float difference = fabsf(h3_bfloat_value(expected[index]) -
                                          h3_bfloat_value(actual[index]));
                if (difference > maximum_error) maximum_error = difference;
            }
        }

        double reference_gpu_total = 0.0;
        double reference_wall_total = 0.0;
        double optimized_gpu_total = 0.0;
        double optimized_wall_total = 0.0;
        for (unsigned iteration = 0u; iteration < warmup + measured; ++iteration) {
            double wall_ms;
            double gpu_ms;
            int optimized_first = (iteration & 1u) != 0u;
            if (h3_prepare(queue,
                           optimized_first ? pipelines.precomputed_prepare
                                           : pipelines.reference_prepare,
                           &buffers, (uint32_t)rows, optimized_first,
                           &wall_ms, &gpu_ms) != 0) {
                free(positions64); free(tags); free(positions32);
                return 1;
            }
            if (iteration >= warmup) {
                if (optimized_first) {
                    optimized_wall_total += wall_ms;
                    optimized_gpu_total += gpu_ms;
                } else {
                    reference_wall_total += wall_ms;
                    reference_gpu_total += gpu_ms;
                }
            }
            if (h3_prepare(queue,
                           optimized_first ? pipelines.reference_prepare
                                           : pipelines.precomputed_prepare,
                           &buffers, (uint32_t)rows, !optimized_first,
                           &wall_ms, &gpu_ms) != 0) {
                free(positions64); free(tags); free(positions32);
                return 1;
            }
            if (iteration >= warmup) {
                if (optimized_first) {
                    reference_wall_total += wall_ms;
                    reference_gpu_total += gpu_ms;
                } else {
                    optimized_wall_total += wall_ms;
                    optimized_gpu_total += gpu_ms;
                }
            }
        }

        double reference_gpu_ms = reference_gpu_total / measured;
        double reference_wall_ms = reference_wall_total / measured;
        double optimized_gpu_ms = optimized_gpu_total / measured;
        double optimized_wall_ms = optimized_wall_total / measured;
        double saved_for_200_ms =
            200.0 * (reference_gpu_ms - optimized_gpu_ms) - rope_gpu_ms;
        size_t owned_bytes = projected_elements * 2u + rows * 3u * sizeof(float) +
            rows * H3_ROPE_SLOTS * 2u * sizeof(float) +
            H3_HEAD_DIM * sizeof(uint16_t) + 3u * head_elements * 2u;

        printf("{\n");
        printf("  \"schema\": 1,\n");
        printf("  \"scope\": \"H3 Q/K RMSNorm plus MM-RoPE preparation; excludes QKV projection and attention\",\n");
        printf("  \"device\": \"%s\",\n", device.name.UTF8String);
        printf("  \"sequence_rows\": %zu,\n", rows);
        printf("  \"rope_pairs\": %zu,\n", rows * H3_ROPE_SLOTS);
        printf("  \"rope_bytes\": %zu,\n",
               rows * H3_ROPE_SLOTS * 2u * sizeof(float));
        printf("  \"warmup_iterations\": %u,\n", warmup);
        printf("  \"measured_iterations\": %u,\n", measured);
        printf("  \"precompute_gpu_ms\": %.6f,\n", rope_gpu_ms);
        printf("  \"precompute_wall_ms\": %.6f,\n", rope_wall_ms);
        printf("  \"reference_prepare_gpu_ms\": %.6f,\n", reference_gpu_ms);
        printf("  \"reference_prepare_wall_ms\": %.6f,\n", reference_wall_ms);
        printf("  \"precomputed_prepare_gpu_ms\": %.6f,\n", optimized_gpu_ms);
        printf("  \"precomputed_prepare_wall_ms\": %.6f,\n", optimized_wall_ms);
        printf("  \"prepare_speedup\": %.6f,\n",
               reference_gpu_ms / optimized_gpu_ms);
        printf("  \"estimated_saved_gpu_ms_200_calls_after_precompute\": %.6f,\n",
               saved_for_200_ms);
        printf("  \"differential_rows\": %u,\n", H3_TEST_ROWS);
        printf("  \"differing_bf16_values\": %zu,\n", differing);
        printf("  \"maximum_absolute_error\": %.9g,\n", maximum_error);
        printf("  \"metal_owned_buffer_bytes\": %zu\n", owned_bytes);
        printf("}\n");

        free(positions64);
        free(tags);
        free(positions32);
        return differing == 0u ? 0 : 3;
    }
}
