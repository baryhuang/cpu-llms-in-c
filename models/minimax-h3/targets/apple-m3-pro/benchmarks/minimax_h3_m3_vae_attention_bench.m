#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double h3_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC_RAW, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static uint32_t h3_u32(const char *text, uint32_t minimum,
                       uint32_t maximum) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < minimum ||
        value > maximum) {
        fprintf(stderr, "invalid integer: %s\n", text);
        exit(2);
    }
    return (uint32_t)value;
}

static id<MTLComputePipelineState> h3_pipeline(id<MTLDevice> device,
                                               id<MTLLibrary> library,
                                               NSString *name,
                                               NSError **error) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    return function == nil
               ? nil
               : [device newComputePipelineStateWithFunction:function
                                                        error:error];
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

static int h3_initialize(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> pipeline,
                         id<MTLBuffer> buffer,
                         uint32_t count,
                         uint32_t seed) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:&count length:sizeof(count) atIndex:1];
    [encoder setBytes:&seed length:sizeof(seed) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [encoder endEncoding];
    return h3_wait(command);
}

typedef struct {
    double gpu_ms;
    double wall_ms;
} h3_time;

static int h3_run_attention(id<MTLCommandQueue> queue,
                            id<MTLComputePipelineState> pipeline,
                            id<MTLBuffer> query,
                            id<MTLBuffer> key,
                            id<MTLBuffer> value,
                            id<MTLBuffer> output,
                            uint32_t rows,
                            uint32_t runs,
                            uint32_t queries_per_group,
                            h3_time *result) {
    memset(result, 0, sizeof(*result));
    for (uint32_t run = 0u; run <= runs; ++run) {
        @autoreleasepool {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            double started = h3_seconds();
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:query offset:0 atIndex:0];
            [encoder setBuffer:key offset:0 atIndex:1];
            [encoder setBuffer:value offset:0 atIndex:2];
            [encoder setBuffer:output offset:0 atIndex:3];
            [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
            [encoder dispatchThreadgroups:
                MTLSizeMake(((rows + queries_per_group - 1u) /
                             queries_per_group) * 32u,
                            1u, 1u)
                    threadsPerThreadgroup:
                        MTLSizeMake(queries_per_group * 32u, 1u, 1u)];
            [encoder endEncoding];
            if (h3_wait(command) != 0) return 1;
            if (run != 0u) {
                result->wall_ms += (h3_seconds() - started) * 1000.0;
                result->gpu_ms +=
                    (command.GPUEndTime - command.GPUStartTime) * 1000.0;
            }
        }
    }
    result->wall_ms /= runs;
    result->gpu_ms /= runs;
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "usage: %s METALLIB [ROWS=1797] [RUNS=5]\n",
                argv[0]);
        return 2;
    }
    uint32_t rows = argc > 2 ? h3_u32(argv[2], 1u, 8192u) : 1797u;
    uint32_t runs = argc > 3 ? h3_u32(argv[3], 1u, 100u) : 5u;
    @autoreleasepool {
        NSError *error = nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLLibrary> library = [device newLibraryWithURL:
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]
                                                    error:&error];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLComputePipelineState> initialize = h3_pipeline(
            device, library, @"minimax_h3_initialize_half_irregular", &error);
        id<MTLComputePipelineState> reference = h3_pipeline(
            device, library, @"minimax_h3_video_attention", &error);
        id<MTLComputePipelineState> tiled = h3_pipeline(
            device, library, @"minimax_h3_video_attention_tiled8", &error);
        id<MTLComputePipelineState> tiled16 = h3_pipeline(
            device, library, @"minimax_h3_video_attention_tiled16", &error);
        uint64_t values = (uint64_t)rows * 32u * 64u;
        if (values > UINT32_MAX) {
            fprintf(stderr, "attention tensor exceeds initializer range\n");
            return 2;
        }
        id<MTLBuffer> query = [device newBufferWithLength:values * 2u
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> key = [device newBufferWithLength:values * 2u
                                                options:MTLResourceStorageModeShared];
        id<MTLBuffer> value = [device newBufferWithLength:values * 2u
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> expected = [device newBufferWithLength:values * 2u
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> actual = [device newBufferWithLength:values * 2u
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> actual16 = [device newBufferWithLength:values * 2u
                                                     options:MTLResourceStorageModeShared];
        if (device == nil || library == nil || queue == nil ||
            initialize == nil || reference == nil || tiled == nil ||
            tiled16 == nil ||
            query == nil || key == nil || value == nil || expected == nil ||
            actual == nil || actual16 == nil ||
            h3_initialize(queue, initialize, query, (uint32_t)values, 3u) ||
            h3_initialize(queue, initialize, key, (uint32_t)values, 5u) ||
            h3_initialize(queue, initialize, value, (uint32_t)values, 7u)) {
            fprintf(stderr, "attention benchmark setup failed: %s\n",
                    error.localizedDescription.UTF8String);
            return 1;
        }
        h3_time reference_time;
        h3_time tiled_time;
        h3_time tiled16_time;
        if (h3_run_attention(queue, reference, query, key, value, expected,
                             rows, runs, 1u, &reference_time) != 0 ||
            h3_run_attention(queue, tiled, query, key, value, actual, rows,
                             runs, 8u, &tiled_time) != 0 ||
            h3_run_attention(queue, tiled16, query, key, value, actual16,
                             rows, runs, 16u, &tiled16_time) != 0)
            return 1;
        uint16_t *expected_values = expected.contents;
        uint16_t *actual_values = actual.contents;
        uint16_t *actual16_values = actual16.contents;
        uint64_t non_finite = 0u;
        uint64_t differing = 0u;
        uint64_t differing16 = 0u;
        for (uint64_t index = 0u; index < values; ++index) {
            __fp16 half_value;
            uint16_t bits = actual_values[index];
            memcpy(&half_value, &bits, sizeof(bits));
            if (!isfinite((float)half_value)) ++non_finite;
            if (expected_values[index] != actual_values[index]) ++differing;
            if (expected_values[index] != actual16_values[index])
                ++differing16;
        }
        printf("{\n");
        printf("  \"schema\": 1,\n");
        printf("  \"scope\": \"one MiniMax-H3 ViT3D dense-attention call\",\n");
        printf("  \"device\": \"%s\",\n", device.name.UTF8String);
        printf("  \"sequence_rows\": %u,\n", rows);
        printf("  \"heads\": 32,\n");
        printf("  \"head_dimension\": 64,\n");
        printf("  \"warmup_runs\": 1,\n");
        printf("  \"measured_runs\": %u,\n", runs);
        printf("  \"reference_gpu_ms\": %.6f,\n", reference_time.gpu_ms);
        printf("  \"reference_wall_ms\": %.6f,\n", reference_time.wall_ms);
        printf("  \"tiled8_gpu_ms\": %.6f,\n", tiled_time.gpu_ms);
        printf("  \"tiled8_wall_ms\": %.6f,\n", tiled_time.wall_ms);
        printf("  \"speedup\": %.6f,\n",
               reference_time.gpu_ms / tiled_time.gpu_ms);
        printf("  \"tiled16_gpu_ms\": %.6f,\n", tiled16_time.gpu_ms);
        printf("  \"tiled16_wall_ms\": %.6f,\n", tiled16_time.wall_ms);
        printf("  \"tiled16_speedup\": %.6f,\n",
               reference_time.gpu_ms / tiled16_time.gpu_ms);
        printf("  \"differing_fp16_values\": %llu,\n",
               (unsigned long long)differing);
        printf("  \"tiled16_differing_fp16_values\": %llu,\n",
               (unsigned long long)differing16);
        printf("  \"non_finite_fp16_values\": %llu,\n",
               (unsigned long long)non_finite);
        printf("  \"metal_owned_buffer_bytes\": %llu\n",
               (unsigned long long)(values * 12u));
        printf("}\n");
        return non_finite == 0u && differing == 0u && differing16 == 0u
                   ? 0
                   : 1;
    }
}
