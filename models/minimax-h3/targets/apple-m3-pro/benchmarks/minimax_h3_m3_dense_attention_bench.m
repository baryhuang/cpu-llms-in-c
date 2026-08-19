#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>\n#include <string.h>

enum {
    H3_HEADS = 56,
    H3_HEAD_DIM = 128,
};

static id<MTLComputePipelineState> pipeline(id<MTLDevice> device,
                                             id<MTLLibrary> library,
                                             NSString *name,
                                             NSError **error) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    return function == nil
               ? nil
               : [device newComputePipelineStateWithFunction:function
                                                        error:error];
}

static int wait_for(id<MTLCommandBuffer> command) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "%s\n", command.error.localizedDescription.UTF8String);
        return 1;
    }
    return 0;
}

static double run_reference(id<MTLCommandQueue> queue,
                            id<MTLComputePipelineState> attention,
                            id<MTLComputePipelineState> convert,
                            id<MTLBuffer> query,
                            id<MTLBuffer> key,
                            id<MTLBuffer> value,
                            id<MTLBuffer> scratch,
                            id<MTLBuffer> output,
                            uint32_t rows,
                            uint32_t elements) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:attention];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:key offset:0 atIndex:1];
    [encoder setBuffer:value offset:0 atIndex:2];
    [encoder setBuffer:scratch offset:0 atIndex:3];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake((rows + 63u) / 64u,
                                               H3_HEADS, 1u)
                threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [encoder setComputePipelineState:convert];
    [encoder setBuffer:scratch offset:0 atIndex:0];
    [encoder setBuffer:output offset:0 atIndex:1];
    [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(elements, 1u, 1u)
             threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [encoder endEncoding];
    if (wait_for(command) != 0) return -1.0;
    return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
}

static double run_direct(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> attention,
                         id<MTLBuffer> query,
                         id<MTLBuffer> key,
                         id<MTLBuffer> value,
                         id<MTLBuffer> output,
                         uint32_t rows) {
    return -2.0;
}

static double run_block(id<MTLCommandQueue> queue,
                        id<MTLComputePipelineState> attention,
                        id<MTLBuffer> query,
                        id<MTLBuffer> key,
                        id<MTLBuffer> value,
                        id<MTLBuffer> output,
                        uint32_t rows,
                        uint32_t block_rows) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:attention];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:key offset:0 atIndex:1];
    [encoder setBuffer:value offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(
            (rows + block_rows - 1u) / block_rows, H3_HEADS, 1u)
                threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
    [encoder endEncoding];
    if (wait_for(command) != 0) return -1.0;
    return (command.GPUEndTime - command.GPUStartTime) * 1000.0;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *metallib_path = argc > 1
            ? argv[1]
            : "build/minimax-h3-m3-attention.metallib";
        unsigned long parsed_rows = argc > 2 ? strtoul(argv[2], NULL, 10)
                                              : 15639ul;
        if (parsed_rows == 0ul || parsed_rows > UINT32_MAX) {
            fprintf(stderr, "invalid row count\n");
            return 2;
        }
        uint32_t rows = (uint32_t)parsed_rows;
        uint64_t element_count64 =
            (uint64_t)rows * H3_HEADS * H3_HEAD_DIM;
        if (element_count64 > UINT32_MAX ||
            element_count64 > SIZE_MAX / sizeof(uint32_t)) {
            fprintf(stderr, "attention image exceeds benchmark limits\n");
            return 2;
        }
        uint32_t elements = (uint32_t)element_count64;
        size_t bf16_bytes = (size_t)elements * sizeof(uint16_t);
        size_t fp32_bytes = (size_t)elements * sizeof(float);

        NSError *error = nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLLibrary> library = [device
            newLibraryWithURL:[NSURL fileURLWithPath:
                [NSString stringWithUTF8String:metallib_path]]
                        error:&error];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLComputePipelineState> initialize = pipeline(
            device, library, @"minimax_h3_initialize_bf16_irregular", &error);
        id<MTLComputePipelineState> reference = pipeline(
            device, library, @"minimax_h3_dense_attention_mma64_bf16", &error);
        id<MTLComputePipelineState> direct = pipeline(
            device, library,
            @"minimax_h3_dense_attention_mma64_bf16_direct", &error);
        id<MTLComputePipelineState> convert = pipeline(
            device, library, @"minimax_h3_f32_to_bf16", &error);
        id<MTLComputePipelineState> flash = pipeline(
            device, library,
            @"minimax_h3_dense_attention_mma64_bf16_flash16", &error);
        if (device == nil || library == nil || queue == nil ||
            initialize == nil || reference == nil || direct == nil ||
            convert == nil || flash == nil) {
            const char *message = error != nil
                ? error.localizedDescription.UTF8String
                : "Metal setup failed";
            fprintf(stderr, "%s\n",
                    message != NULL ? message : "Metal setup failed");
            return 1;
        }

        id<MTLBuffer> query = [device newBufferWithLength:bf16_bytes
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> key = [device newBufferWithLength:bf16_bytes
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> value = [device newBufferWithLength:bf16_bytes
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> scratch = [device newBufferWithLength:fp32_bytes
                                                     options:MTLResourceStorageModePrivate];
        id<MTLBuffer> reference_output = [device newBufferWithLength:bf16_bytes
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> direct_output = [device newBufferWithLength:bf16_bytes
                                                           options:MTLResourceStorageModeShared];
        id<MTLBuffer> flash_output = [device newBufferWithLength:bf16_bytes
                                                          options:MTLResourceStorageModeShared];
        if (query == nil || key == nil || value == nil || scratch == nil ||
            reference_output == nil || direct_output == nil ||
            flash_output == nil) {
            fprintf(stderr, "Metal allocation failed\n");
            return 1;
        }

        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        const id<MTLBuffer> tensors[3] = {query, key, value};
        const uint32_t seeds[3] = {3u, 5u, 9u};
        [encoder setComputePipelineState:initialize];
        for (unsigned index = 0u; index < 3u; ++index) {
            [encoder setBuffer:tensors[index] offset:0 atIndex:0];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:1];
            [encoder setBytes:&seeds[index] length:sizeof(seeds[index])
                       atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(elements, 1u, 1u)
                 threadsPerThreadgroup:MTLSizeMake(256u, 1u, 1u)];
        }
        [encoder endEncoding];
        if (wait_for(command) != 0) return 1;

        double reference_ms = run_reference(
            queue, reference, convert, query, key, value, scratch,
            reference_output, rows, elements);
        double direct_ms = run_block(
            queue, direct, query, key, value, direct_output, rows, 64u);
        double flash_ms = run_block(
            queue, flash, query, key, value, flash_output, rows, 64u);
        if (reference_ms < 0.0 || direct_ms < 0.0 || flash_ms < 0.0)
            return 1;

        const uint16_t *expected = reference_output.contents;
        const uint16_t *actual = direct_output.contents;
        size_t differing = 0u;
        for (size_t index = 0u; index < elements; ++index)
            differing += expected[index] != actual[index];

        /* The single-pass kernel accumulates in a different order, so it
         * is compared by magnitude, not bitwise. */
        const uint16_t *flash_actual = flash_output.contents;
        size_t flash_differing = 0u;
        double flash_max_abs = 0.0;
        double reference_max_abs = 0.0;
        for (size_t index = 0u; index < elements; ++index) {
            uint32_t expected_bits = (uint32_t)expected[index] << 16u;
            uint32_t flash_bits = (uint32_t)flash_actual[index] << 16u;
            float expected_value;
            float flash_value;
            memcpy(&expected_value, &expected_bits, sizeof(expected_value));
            memcpy(&flash_value, &flash_bits, sizeof(flash_value));
            flash_differing += expected[index] != flash_actual[index];
            double difference = fabs((double)expected_value -
                                     (double)flash_value);
            if (difference > flash_max_abs) flash_max_abs = difference;
            double magnitude = fabs((double)expected_value);
            if (magnitude > reference_max_abs) reference_max_abs = magnitude;
        }

        printf("{\n");
        printf("  \"device\": \"%s\",\n", device.name.UTF8String);
        printf("  \"rows\": %u,\n", rows);
        printf("  \"elements\": %u,\n", elements);
        printf("  \"reference_fp32_scratch_bytes\": %zu,\n", fp32_bytes);
        printf("  \"reference_gpu_ms\": %.6f,\n", reference_ms);
        printf("  \"direct_bf16_gpu_ms\": %.6f,\n", direct_ms);
        printf("  \"flash_bf16_gpu_ms\": %.6f,\n", flash_ms);
        printf("  \"speedup\": %.6f,\n", reference_ms / direct_ms);
        printf("  \"flash_speedup\": %.6f,\n", reference_ms / flash_ms);
        printf("  \"differing_bf16_values\": %zu,\n", differing);
        printf("  \"flash_differing_bf16_values\": %zu,\n",
               flash_differing);
        printf("  \"flash_max_abs_diff\": %.9g,\n", flash_max_abs);
        printf("  \"reference_max_abs\": %.9g\n", reference_max_abs);
        printf("}\n");
        return differing == 0u ? 0 : 3;
    }
}
