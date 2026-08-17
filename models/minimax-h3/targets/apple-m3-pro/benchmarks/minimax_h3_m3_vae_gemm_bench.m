#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    uint32_t rows;
    uint32_t columns;
    uint32_t batch;
    uint32_t input_stride;
} h3_dense_parameters;

typedef struct {
    double wall_ms;
    double gpu_ms;
} h3_time;

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

static int h3_initialize(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> pipeline,
                         id<MTLBuffer> buffer, uint32_t count,
                         uint32_t seed) {
    @autoreleasepool {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer offset:0 atIndex:0];
        [encoder setBytes:&count length:sizeof(count) atIndex:1];
        [encoder setBytes:&seed length:sizeof(seed) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(count, 1u, 1u)
                 threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        [encoder endEncoding];
        return h3_wait(command);
    }
}

static int h3_run(id<MTLCommandQueue> queue,
                  id<MTLComputePipelineState> pipeline,
                  id<MTLBuffer> input, id<MTLBuffer> weights,
                  id<MTLBuffer> bias, id<MTLBuffer> output,
                  h3_dense_parameters parameters, int mode,
                  h3_time *result) {
    int status = 0;
    @autoreleasepool {
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        uint32_t has_bias = 1u;
        double started = h3_seconds();
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:weights offset:0 atIndex:1];
        [encoder setBuffer:bias offset:0 atIndex:2];
        [encoder setBuffer:output offset:0 atIndex:3];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
        [encoder setBytes:&has_bias length:sizeof(has_bias) atIndex:5];
        if (mode != 0) {
            uint32_t batch_tile = mode == 2 ? 64u : 32u;
            uint32_t threads = mode == 2 ? 256u : 128u;
            [encoder dispatchThreadgroups:
                MTLSizeMake(parameters.rows / 64u,
                            (parameters.batch + batch_tile - 1u) / batch_tile,
                            1u)
                        threadsPerThreadgroup:MTLSizeMake(threads, 1u, 1u)];
        } else {
            uint64_t outputs = (uint64_t)parameters.batch * parameters.rows;
            [encoder dispatchThreadgroups:
                MTLSizeMake((outputs + 3u) / 4u, 1u, 1u)
                        threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
        }
        [encoder endEncoding];
        if (h3_wait(command) != 0) {
            status = 1;
        } else {
            result->wall_ms = (h3_seconds() - started) * 1000.0;
            result->gpu_ms =
                (command.GPUEndTime - command.GPUStartTime) * 1000.0;
        }
    }
    return status;
}

static float h3_half_value(uint16_t bits) {
    __fp16 value;
    memcpy(&value, &bits, sizeof(bits));
    return (float)value;
}

int main(int argc, char **argv) {
    if (argc < 2 || argc > 6) {
        fprintf(stderr,
                "usage: %s METALLIB [BATCH=1797] [ROWS=16384] "
                "[COLUMNS=2048] [MMA_RUNS=3]\n",
                argv[0]);
        return 2;
    }
    uint32_t batch = argc > 2 ? h3_u32(argv[2], 1u, 16384u) : 1797u;
    uint32_t rows = argc > 3 ? h3_u32(argv[3], 64u, 32768u) : 16384u;
    uint32_t columns = argc > 4 ? h3_u32(argv[4], 64u, 16384u) : 2048u;
    uint32_t mma_runs = argc > 5 ? h3_u32(argv[5], 1u, 100u) : 3u;
    if (rows % 64u != 0u || columns % 64u != 0u) {
        fprintf(stderr, "ROWS and COLUMNS must be multiples of 64\n");
        return 2;
    }

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
            device, library, @"minimax_h3_dense_f16", &error);
        id<MTLComputePipelineState> mma = h3_pipeline(
            device, library, @"minimax_h3_dense_f16_mma", &error);
        id<MTLComputePipelineState> weight_tiled = h3_pipeline(
            device, library,
            @"minimax_h3_dense_f16_mma_weight_tiled", &error);
        id<MTLComputePipelineState> weight_tiled_b64 = h3_pipeline(
            device, library,
            @"minimax_h3_dense_f16_mma_weight_tiled_b64", &error);
        if (device == nil || library == nil || queue == nil ||
            initialize == nil || reference == nil || mma == nil ||
            weight_tiled == nil || weight_tiled_b64 == nil) {
            fprintf(stderr, "Metal setup failed: %s\n",
                    error.localizedDescription.UTF8String);
            return 1;
        }

        uint32_t padded_batch = (batch + 63u) & ~63u;
        uint64_t input_count64 = (uint64_t)padded_batch * columns;
        uint64_t weight_count64 = (uint64_t)rows * columns;
        uint64_t output_count64 = (uint64_t)padded_batch * rows;
        if (input_count64 > UINT32_MAX || weight_count64 > UINT32_MAX ||
            output_count64 > UINT32_MAX) {
            fprintf(stderr, "benchmark tensor exceeds initializer range\n");
            return 2;
        }
        uint32_t input_count = (uint32_t)input_count64;
        uint32_t weight_count = (uint32_t)weight_count64;
        id<MTLBuffer> input = [device newBufferWithLength:input_count64 * 2u
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> weights = [device newBufferWithLength:weight_count64 * 2u
                                                    options:MTLResourceStorageModeShared];
        id<MTLBuffer> bias = [device newBufferWithLength:(uint64_t)rows * 2u
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> expected = [device newBufferWithLength:output_count64 * 2u
                                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> actual = [device newBufferWithLength:output_count64 * 2u
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> tiled_actual = [device
            newBufferWithLength:output_count64 * 2u
                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> tiled_b64_actual = [device
            newBufferWithLength:output_count64 * 2u
                        options:MTLResourceStorageModeShared];
        if (input == nil || weights == nil || bias == nil || expected == nil ||
            actual == nil || tiled_actual == nil || tiled_b64_actual == nil ||
            h3_initialize(queue, initialize, input,
                                            input_count, 3u) != 0 ||
            h3_initialize(queue, initialize, weights, weight_count, 7u) != 0 ||
            h3_initialize(queue, initialize, bias, rows, 11u) != 0) {
            fprintf(stderr, "benchmark allocation or initialization failed\n");
            return 1;
        }

        h3_dense_parameters parameters = {
            .rows = rows,
            .columns = columns,
            .batch = batch,
            .input_stride = columns,
        };
        h3_time reference_time = {0};
        if (h3_run(queue, reference, input, weights, bias, expected,
                   parameters, 0, &reference_time) != 0)
            return 1;
        h3_time mma_total = {0};
        for (uint32_t run = 0u; run < mma_runs; ++run) {
            h3_time current = {0};
            if (h3_run(queue, mma, input, weights, bias, actual, parameters,
                       1, &current) != 0)
                return 1;
            mma_total.wall_ms += current.wall_ms;
            mma_total.gpu_ms += current.gpu_ms;
        }
        mma_total.wall_ms /= mma_runs;
        mma_total.gpu_ms /= mma_runs;
        h3_time tiled_total = {0};
        for (uint32_t run = 0u; run < mma_runs; ++run) {
            h3_time current = {0};
            if (h3_run(queue, weight_tiled, input, weights, bias,
                       tiled_actual, parameters, 1, &current) != 0)
                return 1;
            tiled_total.wall_ms += current.wall_ms;
            tiled_total.gpu_ms += current.gpu_ms;
        }
        tiled_total.wall_ms /= mma_runs;
        tiled_total.gpu_ms /= mma_runs;
        h3_time tiled_b64_total = {0};
        for (uint32_t run = 0u; run < mma_runs; ++run) {
            h3_time current = {0};
            if (h3_run(queue, weight_tiled_b64, input, weights, bias,
                       tiled_b64_actual, parameters, 2, &current) != 0)
                return 1;
            tiled_b64_total.wall_ms += current.wall_ms;
            tiled_b64_total.gpu_ms += current.gpu_ms;
        }
        tiled_b64_total.wall_ms /= mma_runs;
        tiled_b64_total.gpu_ms /= mma_runs;

        const uint16_t *expected_values = expected.contents;
        const uint16_t *mma_values = actual.contents;
        const uint16_t *actual_values = tiled_actual.contents;
        const uint16_t *b64_values = tiled_b64_actual.contents;
        uint64_t compared = (uint64_t)batch * rows;
        uint64_t differing = 0u;
        uint64_t differing_from_mma = 0u;
        uint64_t b64_differing_from_mma = 0u;
        double squared_error = 0.0;
        float maximum_error = 0.0f;
        for (uint64_t index = 0u; index < compared; ++index) {
            if (expected_values[index] != actual_values[index]) ++differing;
            if (mma_values[index] != actual_values[index])
                ++differing_from_mma;
            if (mma_values[index] != b64_values[index])
                ++b64_differing_from_mma;
            float difference = fabsf(h3_half_value(expected_values[index]) -
                                      h3_half_value(actual_values[index]));
            squared_error += (double)difference * difference;
            if (difference > maximum_error) maximum_error = difference;
        }
        double operations = 2.0 * (double)batch * rows * columns;
        size_t metal_bytes = (size_t)(input_count64 + weight_count64 + rows +
                                      4u * output_count64) * 2u;
        printf("{\n");
        printf("  \"schema\": 1,\n");
        printf("  \"scope\": \"one MiniMax-H3 ViT3D FP16 dense projection\",\n");
        printf("  \"device\": \"%s\",\n", device.name.UTF8String);
        printf("  \"batch_rows\": %u,\n", batch);
        printf("  \"output_rows\": %u,\n", rows);
        printf("  \"input_columns\": %u,\n", columns);
        printf("  \"reference_gpu_ms\": %.6f,\n", reference_time.gpu_ms);
        printf("  \"reference_wall_ms\": %.6f,\n", reference_time.wall_ms);
        printf("  \"mma_runs\": %u,\n", mma_runs);
        printf("  \"mma_gpu_ms\": %.6f,\n", mma_total.gpu_ms);
        printf("  \"mma_wall_ms\": %.6f,\n", mma_total.wall_ms);
        printf("  \"mma_speedup\": %.6f,\n",
               reference_time.gpu_ms / mma_total.gpu_ms);
        printf("  \"mma_effective_tflops\": %.6f,\n",
               operations / (mma_total.gpu_ms * 1.0e9));
        printf("  \"weight_tiled_gpu_ms\": %.6f,\n",
               tiled_total.gpu_ms);
        printf("  \"weight_tiled_wall_ms\": %.6f,\n",
               tiled_total.wall_ms);
        printf("  \"weight_tiled_vs_mma\": %.6f,\n",
               mma_total.gpu_ms / tiled_total.gpu_ms);
        printf("  \"weight_tiled_effective_tflops\": %.6f,\n",
               operations / (tiled_total.gpu_ms * 1.0e9));
        printf("  \"weight_tiled_b64_gpu_ms\": %.6f,\n",
               tiled_b64_total.gpu_ms);
        printf("  \"weight_tiled_b64_wall_ms\": %.6f,\n",
               tiled_b64_total.wall_ms);
        printf("  \"weight_tiled_b64_vs_mma\": %.6f,\n",
               mma_total.gpu_ms / tiled_b64_total.gpu_ms);
        printf("  \"compared_fp16_values\": %llu,\n",
               (unsigned long long)compared);
        printf("  \"differing_fp16_values\": %llu,\n",
               (unsigned long long)differing);
        printf("  \"differing_from_mma_fp16_values\": %llu,\n",
               (unsigned long long)differing_from_mma);
        printf("  \"b64_differing_from_mma_fp16_values\": %llu,\n",
               (unsigned long long)b64_differing_from_mma);
        printf("  \"maximum_absolute_error\": %.9g,\n", maximum_error);
        printf("  \"root_mean_square_error\": %.9g,\n",
               sqrt(squared_error / (double)compared));
        printf("  \"metal_owned_buffer_bytes\": %zu\n", metal_bytes);
        printf("}\n");
        return 0;
    }
}
