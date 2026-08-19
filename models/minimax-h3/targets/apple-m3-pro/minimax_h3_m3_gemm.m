#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "minimax_h3_m3_gemm.h"
#include "minimax_h3_q4_layer.h"

#include <mach/mach.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

enum {
    H3_GEMM_COLUMNS = 5376,
    H3_GEMM_ROWS = 14336,
    H3_GEMM_GROUPS = 84,
    H3_GEMM_INPUT_SEED = 17
};

typedef struct {
    uint32_t rows;
    uint32_t groups_per_row;
    uint32_t batch;
} h3_gemm_parameters;

static void gemm_error(char *message, size_t capacity, NSString *text) {
    if (message == NULL || capacity == 0u) return;
    const char *utf8 = text != nil ? text.UTF8String : "unknown Metal error";
    snprintf(message, capacity, "%s", utf8 != NULL ? utf8 : "unknown Metal error");
}

static size_t gemm_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t status = task_info(mach_task_self(), TASK_VM_INFO,
                                     (task_info_t)&info, &count);
    return status == KERN_SUCCESS ? (size_t)info.phys_footprint : 0u;
}

static double gemm_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC_RAW, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static id<MTLComputePipelineState> gemm_pipeline(id<MTLDevice> device,
                                                 id<MTLLibrary> library,
                                                 NSString *name,
                                                 NSError **error) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    return function != nil
               ? [device newComputePipelineStateWithFunction:function error:error]
               : nil;
}

static int gemm_wait(id<MTLCommandBuffer> command,
                     char *error_message,
                     size_t error_capacity) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
        gemm_error(error_message, error_capacity, command.error.localizedDescription);
        return -1;
    }
    return 0;
}

static float gemm_half_value(uint16_t bits) {
    __fp16 value;
    memcpy(&value, &bits, sizeof(value));
    return (float)value;
}

static float gemm_input_value(size_t index) {
    int value = (int)(((uint32_t)index * 13u + H3_GEMM_INPUT_SEED) % 17u) - 8;
    return (float)value * 0.0625f;
}

static uint8_t gemm_quant_byte(size_t index) {
    uint32_t block = (uint32_t)index >> 5u;
    uint32_t low = ((uint32_t)index * 5u + block * 3u + 3u) & 15u;
    uint32_t high = ((uint32_t)index * 7u + block * 11u + 11u) & 15u;
    return (uint8_t)(low | (high << 4u));
}

static float gemm_reference(size_t output_row) {
    float result = 0.0f;
    for (size_t group = 0u; group < H3_GEMM_GROUPS; ++group) {
        size_t block = output_row * H3_GEMM_GROUPS + group;
        __fp16 scale = (__fp16)(0.0175f + (float)(block % 7u) * 0.0017f);
        __fp16 bias = (__fp16)(-0.1875f + (float)(block % 7u) * 0.0078125f);
        for (size_t within = 0u; within < 64u; ++within) {
            size_t byte_index = block * 32u + (within >> 1u);
            uint8_t bits = gemm_quant_byte(byte_index);
            uint32_t quant = (within & 1u) == 0u ? bits & 15u : bits >> 4u;
            __fp16 weight_half =
                (__fp16)(scale * (__fp16)quant + bias);
            float weight = (float)weight_half;
            result += gemm_input_value(group * 64u + within) * weight;
        }
    }
    return result;
}

int minimax_h3_m3_run_gemm_benchmark(
    const char *metallib_path,
    size_t batch_rows,
    unsigned warmup_iterations,
    unsigned measured_iterations,
    minimax_h3_m3_gemm_result *result,
    char *error_message,
    size_t error_message_capacity) {
    if (metallib_path == NULL || batch_rows == 0u || batch_rows > UINT32_MAX ||
        warmup_iterations == 0u || measured_iterations == 0u || result == NULL) {
        gemm_error(error_message, error_message_capacity, @"invalid argument");
        return 2;
    }

    @autoreleasepool {
        size_t padded_batch = (batch_rows + 31u) & ~(size_t)31u;
        size_t input_elements = padded_batch * H3_GEMM_COLUMNS;
        size_t output_elements = padded_batch * H3_GEMM_ROWS;
        size_t quant_bytes = (size_t)H3_GEMM_ROWS * H3_GEMM_GROUPS * 32u;
        size_t meta_count = (size_t)H3_GEMM_ROWS * H3_GEMM_GROUPS;
        size_t meta_bytes = meta_count * 4u;
        size_t input_bytes = input_elements * sizeof(uint16_t);
        size_t output_bytes = output_elements * sizeof(uint16_t);
        h3_gemm_parameters parameters = {
            .rows = H3_GEMM_ROWS,
            .groups_per_row = H3_GEMM_GROUPS,
            .batch = (uint32_t)batch_rows
        };
        minimax_h3_m3_gemm_result measured;
        NSError *error = nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLLibrary> library = nil;
        id<MTLCommandQueue> queue = nil;
        id<MTLComputePipelineState> initialize_half = nil;
        id<MTLComputePipelineState> initialize_q4 = nil;
        id<MTLComputePipelineState> initialize_meta = nil;
        id<MTLComputePipelineState> gemm = nil;
        id<MTLComputePipelineState> copy = nil;
        id<MTLBuffer> input = nil;
        id<MTLBuffer> quants = nil;
        id<MTLBuffer> metadata = nil;
        id<MTLBuffer> output = nil;
        id<MTLBuffer> first_output = nil;
        double gpu_seconds = 0.0;
        double wall_seconds = 0.0;

        memset(&measured, 0, sizeof(measured));
        measured.footprint_before_bytes = gemm_footprint();
        if (device == nil) {
            gemm_error(error_message, error_message_capacity, @"Metal unavailable");
            return 1;
        }
        library = [device newLibraryWithURL:[NSURL fileURLWithPath:
                    [NSString stringWithUTF8String:metallib_path]] error:&error];
        if (library == nil) {
            gemm_error(error_message, error_message_capacity,
                       error.localizedDescription);
            return 1;
        }
        queue = [device newCommandQueue];
        initialize_half = gemm_pipeline(device, library,
                                        @"minimax_h3_initialize_half", &error);
        initialize_q4 = gemm_pipeline(device, library,
                                      @"minimax_h3_initialize_q4", &error);
        initialize_meta = gemm_pipeline(device, library,
                                        @"minimax_h3_initialize_q4_meta", &error);
        gemm = gemm_pipeline(device, library,
                             @"minimax_h3_q4_gemm_f16_mma_wide", &error);
        copy = gemm_pipeline(device, library,
                             @"minimax_h3_copy_gemm_first_output", &error);
        if (queue == nil || initialize_half == nil || initialize_q4 == nil ||
            initialize_meta == nil || gemm == nil || copy == nil) {
            gemm_error(error_message, error_message_capacity,
                       error.localizedDescription);
            return 1;
        }

        input = [device newBufferWithLength:input_bytes
                                    options:MTLResourceStorageModePrivate];
        quants = [device newBufferWithLength:quant_bytes
                                     options:MTLResourceStorageModePrivate];
        metadata = [device newBufferWithLength:meta_bytes
                                       options:MTLResourceStorageModePrivate];
        output = [device newBufferWithLength:output_bytes
                                     options:MTLResourceStorageModePrivate];
        first_output = [device newBufferWithLength:8u * sizeof(uint16_t)
                                           options:MTLResourceStorageModeShared];
        if (input == nil || quants == nil || metadata == nil || output == nil ||
            first_output == nil) {
            gemm_error(error_message, error_message_capacity,
                       @"Metal buffer allocation failure");
            return 1;
        }

        {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            uint32_t input_count = (uint32_t)input_elements;
            uint32_t quant_count = (uint32_t)quant_bytes;
            uint32_t metadata_count = (uint32_t)meta_count;
            uint32_t seed = H3_GEMM_INPUT_SEED;
            [encoder setComputePipelineState:initialize_half];
            [encoder setBuffer:input offset:0 atIndex:0];
            [encoder setBytes:&input_count length:sizeof(input_count) atIndex:1];
            [encoder setBytes:&seed length:sizeof(seed) atIndex:2];
            [encoder dispatchThreads:MTLSizeMake(input_elements, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder setComputePipelineState:initialize_q4];
            [encoder setBuffer:quants offset:0 atIndex:0];
            [encoder setBytes:&quant_count length:sizeof(quant_count) atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(quant_bytes, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder setComputePipelineState:initialize_meta];
            [encoder setBuffer:metadata offset:0 atIndex:0];
            [encoder setBytes:&metadata_count length:sizeof(metadata_count)
                       atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(meta_count, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder endEncoding];
            if (gemm_wait(command, error_message, error_message_capacity) != 0)
                return 1;
        }

        for (unsigned iteration = 0u;
             iteration < warmup_iterations + measured_iterations; ++iteration) {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            double start = gemm_seconds();
            [encoder setComputePipelineState:gemm];
            [encoder setBuffer:input offset:0 atIndex:0];
            [encoder setBuffer:quants offset:0 atIndex:1];
            [encoder setBuffer:metadata offset:0 atIndex:2];
            [encoder setBuffer:output offset:0 atIndex:3];
            [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
            [encoder dispatchThreadgroups:
                MTLSizeMake((H3_GEMM_ROWS + 63u) / 64u,
                            (batch_rows + 63u) / 64u, 1)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
            if (gemm_wait(command, error_message, error_message_capacity) != 0)
                return 1;
            if (iteration >= warmup_iterations) {
                wall_seconds += gemm_seconds() - start;
                gpu_seconds += command.GPUEndTime - command.GPUStartTime;
            }
        }

        {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            [encoder setComputePipelineState:copy];
            [encoder setBuffer:output offset:0 atIndex:0];
            [encoder setBuffer:first_output offset:0 atIndex:1];
            [encoder dispatchThreads:MTLSizeMake(8, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(8, 1, 1)];
            [encoder endEncoding];
            if (gemm_wait(command, error_message, error_message_capacity) != 0)
                return 1;
        }

        measured.warmup_iterations = warmup_iterations;
        measured.measured_iterations = measured_iterations;
        measured.batch_rows = batch_rows;
        measured.input_columns = H3_GEMM_COLUMNS;
        measured.output_rows = H3_GEMM_ROWS;
        measured.weight_bytes = quant_bytes;
        measured.metadata_bytes = meta_bytes;
        measured.input_bytes = input_bytes;
        measured.output_bytes = output_bytes;
        measured.metal_owned_buffer_bytes = input_bytes + quant_bytes +
                                            meta_bytes + output_bytes + 16u;
        measured.gpu_ms = gpu_seconds * 1000.0 / measured_iterations;
        measured.wall_ms = wall_seconds * 1000.0 / measured_iterations;
        measured.effective_tflops =
            (2.0 * (double)batch_rows * H3_GEMM_COLUMNS * H3_GEMM_ROWS) /
            (measured.gpu_ms * 1.0e9);
        measured.footprint_peak_bytes = gemm_footprint();
        snprintf(measured.device_name, sizeof(measured.device_name), "%s",
                 device.name.UTF8String);
        const uint16_t *actual = first_output.contents;
        for (size_t index = 0u; index < 8u; ++index) {
            float reference = gemm_reference(index);
            float value = gemm_half_value(actual[index]);
            float difference = fabsf(reference - value);
            measured.reference_first_8[index] = reference;
            measured.metal_first_8[index] = value;
            if (difference > measured.max_abs_error_first_8)
                measured.max_abs_error_first_8 = difference;
        }
        *result = measured;
        return 0;
    }
}

int minimax_h3_m3_run_q4_projection_bf16(
    const char *metallib_path,
    const minimax_h3_q4_projection *projection,
    const uint16_t *input_bf16,
    size_t batch_rows,
    uint16_t *output_bf16,
    double *gpu_ms,
    char *error_message,
    size_t error_message_capacity) {
    if (metallib_path == NULL || projection == NULL || input_bf16 == NULL ||
        output_bf16 == NULL || batch_rows == 0u || batch_rows > UINT32_MAX ||
        projection->weight == NULL || projection->scales == NULL ||
        projection->biases == NULL || projection->input_columns % 64u != 0u) {
        gemm_error(error_message, error_message_capacity, @"invalid projection");
        return 2;
    }
    @autoreleasepool {
        NSError *error = nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLLibrary> library = device != nil
            ? [device newLibraryWithURL:[NSURL fileURLWithPath:
                [NSString stringWithUTF8String:metallib_path]] error:&error]
            : nil;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLComputePipelineState> pipeline =
            library != nil ? gemm_pipeline(device, library,
                @"minimax_h3_q4_gemm_bf16_activation", &error) : nil;
        if (device == nil || library == nil || queue == nil || pipeline == nil) {
            gemm_error(error_message, error_message_capacity,
                       error.localizedDescription);
            return 1;
        }
        size_t padded_batch = (batch_rows + 31u) & ~(size_t)31u;
        size_t input_bytes = padded_batch * projection->input_columns * 2u;
        size_t output_bytes = padded_batch * projection->output_rows * 2u;
        size_t metadata_bytes = projection->metadata_elements * 2u;
        id<MTLBuffer> input = [device newBufferWithLength:input_bytes
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> weights = [device newBufferWithBytes:projection->weight
                                                    length:projection->weight_bytes
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> scales = [device newBufferWithBytes:projection->scales
                                                   length:metadata_bytes
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> biases = [device newBufferWithBytes:projection->biases
                                                   length:metadata_bytes
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> output = [device newBufferWithLength:output_bytes
                                                   options:MTLResourceStorageModeShared];
        if (input == nil || weights == nil || scales == nil || biases == nil ||
            output == nil) {
            gemm_error(error_message, error_message_capacity,
                       @"real projection Metal allocation failed");
            return 1;
        }
        memset(input.contents, 0, input_bytes);
        memcpy(input.contents, input_bf16,
               batch_rows * projection->input_columns * 2u);
        h3_gemm_parameters parameters = {
            .rows = projection->output_rows,
            .groups_per_row = projection->groups_per_row,
            .batch = (uint32_t)batch_rows,
        };
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:input offset:0 atIndex:0];
        [encoder setBuffer:weights offset:0 atIndex:1];
        [encoder setBuffer:scales offset:0 atIndex:2];
        [encoder setBuffer:biases offset:0 atIndex:3];
        [encoder setBuffer:output offset:0 atIndex:4];
        [encoder setBytes:&parameters length:sizeof(parameters) atIndex:5];
        [encoder dispatchThreadgroups:
            MTLSizeMake((projection->output_rows + 63u) / 64u,
                        (batch_rows + 31u) / 32u, 1u)
                threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
        [encoder endEncoding];
        if (gemm_wait(command, error_message, error_message_capacity) != 0)
            return 1;
        memcpy(output_bf16, output.contents,
               batch_rows * projection->output_rows * 2u);
        if (gpu_ms != NULL)
            *gpu_ms = (command.GPUEndTime - command.GPUStartTime) * 1000.0;
        return 0;
    }
}

int minimax_h3_m3_run_q8_gemm_benchmark(
    const char *metallib_path,
    size_t input_columns,
    size_t output_rows,
    size_t batch_rows,
    unsigned warmup_iterations,
    unsigned measured_iterations,
    minimax_h3_m3_gemm_result *result,
    char *error_message,
    size_t error_message_capacity) {
    if (metallib_path == NULL || input_columns == 0u ||
        input_columns % 64u != 0u || input_columns > UINT32_MAX ||
        output_rows == 0u || output_rows > UINT32_MAX || batch_rows == 0u ||
        batch_rows > UINT32_MAX || warmup_iterations == 0u ||
        measured_iterations == 0u || result == NULL) {
        gemm_error(error_message, error_message_capacity, @"invalid Q8 benchmark argument");
        return 2;
    }

    @autoreleasepool {
        const size_t padded_batch = (batch_rows + 31u) & ~(size_t)31u;
        const size_t groups = input_columns / 64u;
        const size_t input_bytes = padded_batch * input_columns * 2u;
        const size_t output_bytes = padded_batch * output_rows * 2u;
        const size_t quant_bytes = output_rows * groups * 64u;
        const size_t metadata_bytes = output_rows * groups * 2u;
        const size_t total_bytes = input_bytes + output_bytes + quant_bytes +
                                   metadata_bytes * 2u;
        if (input_bytes / 2u / input_columns != padded_batch ||
            output_bytes / 2u / output_rows != padded_batch ||
            quant_bytes / 64u / groups != output_rows ||
            metadata_bytes / 2u / groups != output_rows) {
            gemm_error(error_message, error_message_capacity, @"Q8 benchmark size overflow");
            return 2;
        }
        h3_gemm_parameters parameters = {
            .rows = (uint32_t)output_rows,
            .groups_per_row = (uint32_t)groups,
            .batch = (uint32_t)batch_rows,
        };
        NSError *error = nil;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        id<MTLLibrary> library = device != nil
            ? [device newLibraryWithURL:[NSURL fileURLWithPath:
                [NSString stringWithUTF8String:metallib_path]] error:&error]
            : nil;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLComputePipelineState> pipeline = library != nil
            ? gemm_pipeline(device, library,
                            @"minimax_h3_q8_gemm_bf16_meta", &error)
            : nil;
        if (device == nil || library == nil || queue == nil || pipeline == nil ||
            input_bytes > device.maxBufferLength ||
            output_bytes > device.maxBufferLength ||
            quant_bytes > device.maxBufferLength ||
            metadata_bytes > device.maxBufferLength) {
            gemm_error(error_message, error_message_capacity,
                       error != nil ? error.localizedDescription
                                    : @"Q8 benchmark Metal setup failed");
            return 1;
        }
        id<MTLBuffer> input = [device newBufferWithLength:input_bytes
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> quants = [device newBufferWithLength:quant_bytes
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> scales = [device newBufferWithLength:metadata_bytes
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> biases = [device newBufferWithLength:metadata_bytes
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> output = [device newBufferWithLength:output_bytes
                                                   options:MTLResourceStorageModeShared];
        if (input == nil || quants == nil || scales == nil || biases == nil ||
            output == nil) {
            gemm_error(error_message, error_message_capacity,
                       @"Q8 benchmark buffer allocation failed");
            return 1;
        }
        memset(input.contents, 0, input_bytes);
        memset(quants.contents, 0, quant_bytes);
        memset(scales.contents, 0, metadata_bytes);
        memset(biases.contents, 0, metadata_bytes);
        memset(output.contents, 0, output_bytes);

        double gpu_seconds = 0.0;
        double wall_seconds = 0.0;
        for (unsigned iteration = 0u;
             iteration < warmup_iterations + measured_iterations; ++iteration) {
            id<MTLCommandBuffer> command = [queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
            const double started = gemm_seconds();
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:input offset:0 atIndex:0];
            [encoder setBuffer:quants offset:0 atIndex:1];
            [encoder setBuffer:scales offset:0 atIndex:2];
            [encoder setBuffer:biases offset:0 atIndex:3];
            [encoder setBuffer:output offset:0 atIndex:4];
            [encoder setBytes:&parameters length:sizeof(parameters) atIndex:5];
            [encoder dispatchThreadgroups:
                MTLSizeMake((output_rows + 63u) / 64u,
                            (batch_rows + 31u) / 32u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(128u, 1u, 1u)];
            [encoder endEncoding];
            if (gemm_wait(command, error_message, error_message_capacity) != 0)
                return 1;
            if (iteration >= warmup_iterations) {
                wall_seconds += gemm_seconds() - started;
                gpu_seconds += command.GPUEndTime - command.GPUStartTime;
            }
        }

        minimax_h3_m3_gemm_result measured;
        memset(&measured, 0, sizeof(measured));
        measured.warmup_iterations = warmup_iterations;
        measured.measured_iterations = measured_iterations;
        measured.batch_rows = batch_rows;
        measured.input_columns = input_columns;
        measured.output_rows = output_rows;
        measured.weight_bytes = quant_bytes;
        measured.metadata_bytes = metadata_bytes * 2u;
        measured.input_bytes = input_bytes;
        measured.output_bytes = output_bytes;
        measured.metal_owned_buffer_bytes = total_bytes;
        measured.gpu_ms = gpu_seconds * 1000.0 / measured_iterations;
        measured.wall_ms = wall_seconds * 1000.0 / measured_iterations;
        measured.effective_tflops =
            (2.0 * (double)batch_rows * (double)input_columns *
             (double)output_rows) / (measured.gpu_ms * 1.0e9);
        measured.footprint_peak_bytes = gemm_footprint();
        snprintf(measured.device_name, sizeof(measured.device_name), "%s",
                 device.name.UTF8String);
        const uint16_t *values = output.contents;
        for (size_t index = 0u; index < 8u && index < output_rows; ++index) {
            measured.reference_first_8[index] = 0.0f;
            measured.metal_first_8[index] = gemm_half_value(values[index]);
            measured.max_abs_error_first_8 = fmax(
                measured.max_abs_error_first_8,
                fabs((double)measured.metal_first_8[index]));
        }
        *result = measured;
        return 0;
    }
}
