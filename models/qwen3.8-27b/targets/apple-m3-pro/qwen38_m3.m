#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "qwen38_m3.h"
#include "qwen38_m3_image.h"

#include <mach/mach.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct __attribute__((packed, aligned(2))) {
    uint16_t scale;
    uint16_t bias;
} qwen38_q4_meta;

typedef struct __attribute__((packed, aligned(2))) {
    uint8_t q[32];
    uint16_t scale;
    uint16_t bias;
} qwen38_q4_interleaved_group;

_Static_assert(sizeof(qwen38_q4_meta) == 4,
               "CPU and Metal Q4 metadata strides must match");
_Static_assert(sizeof(qwen38_q4_interleaved_group) == QWEN38_Q4_GROUP_BYTES,
               "CPU and Metal interleaved Q4 strides must match");

static void set_error(char *message, size_t capacity, NSString *text) {
    if (message == NULL || capacity == 0) {
        return;
    }
    const char *utf8 = text != nil ? text.UTF8String : "unknown Metal error";
    snprintf(message, capacity, "%s", utf8 != NULL ? utf8 : "unknown Metal error");
}

static uint16_t half_bits(float value) {
    __fp16 converted = (__fp16)value;
    uint16_t bits;
    memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

static float half_value(uint16_t bits) {
    __fp16 converted;
    memcpy(&converted, &bits, sizeof(converted));
    return (float)converted;
}

static size_t physical_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t status = task_info(mach_task_self(), TASK_VM_INFO,
                                     (task_info_t)&info, &count);
    return status == KERN_SUCCESS ? (size_t)info.phys_footprint : 0;
}

static uint32_t next_random(uint32_t *state) {
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

static void fill_weights(uint8_t *quants, qwen38_q4_meta *metadata,
                         size_t group_count,
                         uint32_t seed) {
    uint32_t state = seed;
    for (size_t index = 0; index < group_count; ++index) {
        for (size_t byte = 0; byte < QWEN38_Q4_GROUP_SIZE / 2; ++byte) {
            quants[index * (QWEN38_Q4_GROUP_SIZE / 2) + byte] =
                (uint8_t)next_random(&state);
        }
        float scale = 0.0035f + 0.00025f * (float)(next_random(&state) & 7u);
        float bias = -7.5f * scale + 0.0002f * (float)((int)(index % 5) - 2);
        metadata[index].scale = half_bits(scale);
        metadata[index].bias = half_bits(bias);
    }
}

static void interleave_weights(qwen38_q4_interleaved_group *output,
                               const uint8_t *quants,
                               const qwen38_q4_meta *metadata,
                               size_t group_count) {
    for (size_t index = 0; index < group_count; ++index) {
        memcpy(output[index].q,
               quants + index * (QWEN38_Q4_GROUP_SIZE / 2),
               QWEN38_Q4_GROUP_SIZE / 2);
        output[index].scale = metadata[index].scale;
        output[index].bias = metadata[index].bias;
    }
}

static float cpu_row_dot(const uint8_t *quants,
                         const qwen38_q4_meta *metadata,
                         const uint16_t *x, size_t row) {
    const size_t groups_per_row = QWEN38_HIDDEN_SIZE / QWEN38_Q4_GROUP_SIZE;
    float total = 0.0f;
    for (size_t group = 0; group < groups_per_row; ++group) {
        size_t block = row * groups_per_row + group;
        const uint8_t *packed =
            quants + block * (QWEN38_Q4_GROUP_SIZE / 2);
        float scale = half_value(metadata[block].scale);
        float bias = half_value(metadata[block].bias);
        for (size_t index = 0; index < QWEN38_Q4_GROUP_SIZE; ++index) {
            uint8_t byte = packed[index >> 1];
            unsigned quant = (index & 1u) == 0 ? byte & 0x0fu : byte >> 4;
            float weight = scale * (float)quant + bias;
            total += weight * half_value(x[group * QWEN38_Q4_GROUP_SIZE + index]);
        }
    }
    return total;
}

static double monotonic_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC_RAW, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static double median_five(double values[5]) {
    for (size_t index = 1; index < 5; ++index) {
        double value = values[index];
        size_t position = index;
        while (position > 0 && values[position - 1] > value) {
            values[position] = values[position - 1];
            --position;
        }
        values[position] = value;
    }
    return values[2];
}

static id<MTLComputePipelineState> make_pipeline(id<MTLLibrary> library,
                                                  NSString *name,
                                                  NSError **error) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        if (error != NULL) {
            *error = [NSError errorWithDomain:@"cpu-llms-in-c"
                                         code:1
                                     userInfo:@{NSLocalizedDescriptionKey:
                                         [NSString stringWithFormat:@"missing Metal function %@", name]}];
        }
        return nil;
    }
    return [library.device newComputePipelineStateWithFunction:function error:error];
}

static void encode_matvec(id<MTLCommandBuffer> command,
                          id<MTLComputePipelineState> pipeline,
                          id<MTLBuffer> x,
                          id<MTLBuffer> quants,
                          id<MTLBuffer> metadata,
                          id<MTLBuffer> output,
                          NSUInteger threads) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:x offset:0 atIndex:0];
    [encoder setBuffer:quants offset:0 atIndex:1];
    [encoder setBuffer:metadata offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    NSUInteger simdgroups = threads / 32;
    MTLSize groups = MTLSizeMake((QWEN38_MLP_SIZE + simdgroups - 1) / simdgroups, 1, 1);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
}

static void encode_interleaved_matvec(id<MTLCommandBuffer> command,
                                      id<MTLComputePipelineState> pipeline,
                                      id<MTLBuffer> x,
                                      id<MTLBuffer> weights,
                                      id<MTLBuffer> output,
                                      NSUInteger threads) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:x offset:0 atIndex:0];
    [encoder setBuffer:weights offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    NSUInteger simdgroups = threads / 32;
    MTLSize groups = MTLSizeMake((QWEN38_MLP_SIZE + simdgroups - 1) / simdgroups, 1, 1);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
}

static void encode_silu(id<MTLCommandBuffer> command,
                        id<MTLComputePipelineState> pipeline,
                        id<MTLBuffer> gate,
                        id<MTLBuffer> up,
                        id<MTLBuffer> output) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:gate offset:0 atIndex:0];
    [encoder setBuffer:up offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(QWEN38_MLP_SIZE, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
}

static void encode_fused(id<MTLCommandBuffer> command,
                         id<MTLComputePipelineState> pipeline,
                         id<MTLBuffer> x,
                         id<MTLBuffer> gate_quants,
                         id<MTLBuffer> gate_metadata,
                         id<MTLBuffer> up_quants,
                         id<MTLBuffer> up_metadata,
                         id<MTLBuffer> output,
                         NSUInteger threads) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:x offset:0 atIndex:0];
    [encoder setBuffer:gate_quants offset:0 atIndex:1];
    [encoder setBuffer:gate_metadata offset:0 atIndex:2];
    [encoder setBuffer:up_quants offset:0 atIndex:3];
    [encoder setBuffer:up_metadata offset:0 atIndex:4];
    [encoder setBuffer:output offset:0 atIndex:5];
    NSUInteger simdgroups = threads / 32;
    MTLSize groups = MTLSizeMake((QWEN38_MLP_SIZE + simdgroups - 1) / simdgroups, 1, 1);
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
}

static void encode_down(id<MTLCommandBuffer> command,
                        id<MTLComputePipelineState> pipeline,
                        id<MTLBuffer> intermediate,
                        id<MTLBuffer> quants,
                        id<MTLBuffer> metadata,
                        id<MTLBuffer> residual,
                        id<MTLBuffer> output,
                        NSUInteger threads) {
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:intermediate offset:0 atIndex:0];
    [encoder setBuffer:quants offset:0 atIndex:1];
    [encoder setBuffer:metadata offset:0 atIndex:2];
    [encoder setBuffer:residual offset:0 atIndex:3];
    [encoder setBuffer:output offset:0 atIndex:4];
    NSUInteger simdgroups = threads / 32;
    MTLSize groups = MTLSizeMake((QWEN38_HIDDEN_SIZE + simdgroups - 1) /
                                 simdgroups, 1, 1);
    [encoder dispatchThreadgroups:groups
             threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    [encoder endEncoding];
}

static BOOL run_baseline(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> matvec,
                         id<MTLComputePipelineState> silu,
                         id<MTLBuffer> x,
                         id<MTLBuffer> gate_quants,
                         id<MTLBuffer> gate_metadata,
                         id<MTLBuffer> up_quants,
                         id<MTLBuffer> up_metadata,
                         id<MTLBuffer> gate,
                         id<MTLBuffer> up,
                         id<MTLBuffer> output,
                         NSUInteger threads) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    encode_matvec(command, matvec, x, gate_quants, gate_metadata, gate, threads);
    encode_matvec(command, matvec, x, up_quants, up_metadata, up, threads);
    encode_silu(command, silu, gate, up, output);
    [command commit];
    [command waitUntilCompleted];
    return command.status == MTLCommandBufferStatusCompleted;
}

static BOOL run_fused(id<MTLCommandQueue> queue,
                      id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> x,
                      id<MTLBuffer> gate_quants,
                      id<MTLBuffer> gate_metadata,
                      id<MTLBuffer> up_quants,
                      id<MTLBuffer> up_metadata,
                      id<MTLBuffer> output,
                      NSUInteger threads) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    encode_fused(command, pipeline, x, gate_quants, gate_metadata,
                 up_quants, up_metadata, output, threads);
    [command commit];
    [command waitUntilCompleted];
    return command.status == MTLCommandBufferStatusCompleted;
}

static BOOL run_interleaved(id<MTLCommandQueue> queue,
                            id<MTLComputePipelineState> matvec,
                            id<MTLComputePipelineState> silu,
                            id<MTLBuffer> x,
                            id<MTLBuffer> gate_weights,
                            id<MTLBuffer> up_weights,
                            id<MTLBuffer> gate,
                            id<MTLBuffer> up,
                            id<MTLBuffer> output,
                            NSUInteger threads) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    encode_interleaved_matvec(command, matvec, x, gate_weights, gate, threads);
    encode_interleaved_matvec(command, matvec, x, up_weights, up, threads);
    encode_silu(command, silu, gate, up, output);
    [command commit];
    [command waitUntilCompleted];
    return command.status == MTLCommandBufferStatusCompleted;
}

static BOOL run_full_mlp(id<MTLCommandQueue> queue,
                         id<MTLComputePipelineState> fused_pipeline,
                         id<MTLComputePipelineState> down_pipeline,
                         id<MTLBuffer> x,
                         id<MTLBuffer> gate_quants,
                         id<MTLBuffer> gate_metadata,
                         id<MTLBuffer> up_quants,
                         id<MTLBuffer> up_metadata,
                         id<MTLBuffer> down_quants,
                         id<MTLBuffer> down_metadata,
                         id<MTLBuffer> intermediate,
                         id<MTLBuffer> output,
                         NSUInteger threads) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    encode_fused(command, fused_pipeline, x, gate_quants, gate_metadata,
                 up_quants, up_metadata, intermediate, threads);
    encode_down(command, down_pipeline, intermediate, down_quants,
                down_metadata, x, output, threads);
    [command commit];
    [command waitUntilCompleted];
    return command.status == MTLCommandBufferStatusCompleted;
}

int qwen38_m3_run_mlp_microbenchmark(const char *metallib_path,
                                    const char *image_path,
                                    unsigned warmup_iterations,
                                    unsigned measured_iterations,
                                    qwen38_m3_mlp_result *result,
                                    char *error_message,
                                    size_t error_message_capacity) {
    if (metallib_path == NULL || result == NULL || measured_iterations == 0) {
        set_error(error_message, error_message_capacity, @"invalid benchmark arguments");
        return 1;
    }
    memset(result, 0, sizeof(*result));

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            set_error(error_message, error_message_capacity, @"no Metal device");
            return 2;
        }
        snprintf(result->device_name, sizeof(result->device_name), "%s", device.name.UTF8String);
        snprintf(result->weight_source, sizeof(result->weight_source), "%s",
                 image_path != NULL ? image_path : "deterministic synthetic Q4");

        NSError *error = nil;
        NSURL *library_url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:metallib_path]];
        id<MTLLibrary> library = [device newLibraryWithURL:library_url error:&error];
        if (library == nil) {
            set_error(error_message, error_message_capacity, error.localizedDescription);
            return 3;
        }
        id<MTLComputePipelineState> matvec = make_pipeline(library, @"qwen38_q4_matvec", &error);
        id<MTLComputePipelineState> interleaved =
            make_pipeline(library, @"qwen38_q4_interleaved_matvec", &error);
        id<MTLComputePipelineState> fused = make_pipeline(library, @"qwen38_q4_gate_up_silu", &error);
        id<MTLComputePipelineState> silu = make_pipeline(library, @"qwen38_silu_mul", &error);
        id<MTLComputePipelineState> down =
            make_pipeline(library, @"qwen38_q4_down_residual", &error);
        if (matvec == nil || interleaved == nil || fused == nil || silu == nil ||
            down == nil) {
            set_error(error_message, error_message_capacity, error.localizedDescription);
            return 4;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            set_error(error_message, error_message_capacity, @"cannot create Metal command queue");
            return 5;
        }

        const size_t groups_per_row = QWEN38_HIDDEN_SIZE / QWEN38_Q4_GROUP_SIZE;
        const size_t group_count = (size_t)QWEN38_MLP_SIZE * groups_per_row;
        const size_t quant_bytes = group_count * (QWEN38_Q4_GROUP_SIZE / 2);
        const size_t metadata_bytes = group_count * sizeof(qwen38_q4_meta);
        const size_t matrix_bytes = quant_bytes + metadata_bytes;
        const size_t down_group_count = (size_t)QWEN38_HIDDEN_SIZE *
                                        (QWEN38_MLP_SIZE /
                                         QWEN38_Q4_GROUP_SIZE);
        const size_t down_quant_bytes = down_group_count *
                                        (QWEN38_Q4_GROUP_SIZE / 2);
        const size_t down_metadata_bytes = down_group_count *
                                           sizeof(qwen38_q4_meta);
        const size_t down_matrix_bytes = down_quant_bytes + down_metadata_bytes;
        const size_t vector_bytes = (size_t)QWEN38_HIDDEN_SIZE * sizeof(uint16_t);
        const size_t output_bytes = (size_t)QWEN38_MLP_SIZE * sizeof(float);
        result->bytes_per_matrix = matrix_bytes;
        result->metal_owned_buffer_bytes = 2 * matrix_bytes + vector_bytes +
                                           5 * output_bytes +
                                           QWEN38_HIDDEN_SIZE * sizeof(float);
        result->warmup_iterations = warmup_iterations;
        result->measured_iterations = measured_iterations;
        result->footprint_before_bytes = physical_footprint();

        MTLResourceOptions options = MTLResourceStorageModeShared |
                                     MTLResourceCPUCacheModeWriteCombined;
        id<MTLBuffer> x = [device newBufferWithLength:vector_bytes options:options];
        id<MTLBuffer> gate_quants = nil;
        id<MTLBuffer> gate_metadata = nil;
        id<MTLBuffer> up_quants = nil;
        id<MTLBuffer> up_metadata = nil;
        id<MTLBuffer> down_quants = nil;
        id<MTLBuffer> down_metadata = nil;
        id<MTLBuffer> gate_interleaved = [device newBufferWithLength:matrix_bytes options:options];
        id<MTLBuffer> up_interleaved = [device newBufferWithLength:matrix_bytes options:options];
        id<MTLBuffer> gate = [device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> up = [device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> baseline_output = [device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> fused_output = [device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> generic_output = [device newBufferWithLength:output_bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> full_mlp_output =
            [device newBufferWithLength:QWEN38_HIDDEN_SIZE * sizeof(float)
                                options:MTLResourceStorageModeShared];
        int image_file = -1;
        void *image_mapping = MAP_FAILED;
        size_t image_length = 0;
        const qwen38_m3_image_header *image_header = NULL;
        if (image_path != NULL) {
            image_file = open(image_path, O_RDONLY);
            struct stat image_status;
            if (image_file < 0 || fstat(image_file, &image_status) != 0 ||
                image_status.st_size < QWEN38_M3_IMAGE_HEADER_BYTES) {
                set_error(error_message, error_message_capacity,
                          @"cannot open Qwen3.8 M3 image");
                if (image_file >= 0) close(image_file);
                return 7;
            }
            image_length = (size_t)image_status.st_size;
            result->mapped_image_bytes = image_length;
            image_mapping = mmap(NULL, image_length, PROT_READ, MAP_PRIVATE,
                                 image_file, 0);
            if (image_mapping == MAP_FAILED) {
                set_error(error_message, error_message_capacity,
                          @"cannot map Qwen3.8 M3 image");
                close(image_file);
                return 7;
            }
            image_header = image_mapping;
            uint64_t expected_gate_quants_offset =
                QWEN38_M3_IMAGE_HEADER_BYTES;
            uint64_t expected_gate_metadata_offset =
                expected_gate_quants_offset + quant_bytes;
            uint64_t expected_up_quants_offset =
                expected_gate_metadata_offset + metadata_bytes;
            uint64_t expected_up_metadata_offset =
                expected_up_quants_offset + quant_bytes;
            uint64_t expected_down_quants_offset =
                expected_up_metadata_offset + metadata_bytes;
            uint64_t expected_down_metadata_offset =
                expected_down_quants_offset + down_quant_bytes;
            uint64_t image_end =
                image_header->version == QWEN38_M3_MLP_ONLY_IMAGE_VERSION ?
                image_header->down_metadata_offset +
                    image_header->down_metadata_bytes :
                image_header->delta_output_metadata_offset +
                    image_header->delta_output_metadata_bytes;
            if (memcmp(image_header->magic, QWEN38_M3_IMAGE_MAGIC, 8) != 0 ||
                (image_header->version != QWEN38_M3_IMAGE_VERSION &&
                 image_header->version !=
                    QWEN38_M3_MLP_ONLY_IMAGE_VERSION) ||
                image_header->header_bytes != QWEN38_M3_IMAGE_HEADER_BYTES ||
                image_header->hidden_size != QWEN38_HIDDEN_SIZE ||
                image_header->rows != QWEN38_MLP_SIZE ||
                image_header->group_size != QWEN38_Q4_GROUP_SIZE ||
                image_header->source_reference_count != 8 ||
                image_header->source_mlp_reference_count != 8 ||
                memcmp(image_header->source_sha256,
                       QWEN38_M3_EXPECTED_SOURCE_SHA256,
                       QWEN38_M3_SOURCE_SHA256_LENGTH) != 0 ||
                image_header->gate_quants_offset !=
                    expected_gate_quants_offset ||
                image_header->gate_quants_bytes != quant_bytes ||
                image_header->gate_metadata_offset !=
                    expected_gate_metadata_offset ||
                image_header->gate_metadata_bytes != metadata_bytes ||
                image_header->up_quants_offset != expected_up_quants_offset ||
                image_header->up_quants_bytes != quant_bytes ||
                image_header->up_metadata_offset !=
                    expected_up_metadata_offset ||
                image_header->up_metadata_bytes != metadata_bytes ||
                image_header->down_rows != QWEN38_HIDDEN_SIZE ||
                image_header->down_groups_per_row !=
                    QWEN38_MLP_SIZE / QWEN38_Q4_GROUP_SIZE ||
                image_header->down_quants_offset !=
                    expected_down_quants_offset ||
                image_header->down_quants_bytes != down_quant_bytes ||
                image_header->down_metadata_offset !=
                    expected_down_metadata_offset ||
                image_header->down_metadata_bytes != down_metadata_bytes ||
                image_end != image_length) {
                set_error(error_message, error_message_capacity,
                          @"invalid Qwen3.8 M3 image header or length");
                munmap(image_mapping, image_length);
                close(image_file);
                return 7;
            }
        }

        if (image_header == NULL) {
            gate_quants = [device newBufferWithLength:quant_bytes options:options];
            gate_metadata = [device newBufferWithLength:metadata_bytes options:options];
            up_quants = [device newBufferWithLength:quant_bytes options:options];
            up_metadata = [device newBufferWithLength:metadata_bytes options:options];
            down_quants = [device newBufferWithLength:down_quant_bytes options:options];
            down_metadata = [device newBufferWithLength:down_metadata_bytes options:options];
            result->metal_owned_buffer_bytes += 2 * matrix_bytes +
                                                down_matrix_bytes;
        } else {
            unsigned char *image_bytes = image_mapping;
            MTLResourceOptions mapped_options = MTLResourceStorageModeShared |
                                                MTLResourceHazardTrackingModeUntracked;
            gate_quants = [device newBufferWithBytesNoCopy:
                               image_bytes + image_header->gate_quants_offset
                                                    length:quant_bytes
                                                   options:mapped_options
                                               deallocator:nil];
            gate_metadata = [device newBufferWithBytesNoCopy:
                                 image_bytes + image_header->gate_metadata_offset
                                                      length:metadata_bytes
                                                     options:mapped_options
                                                 deallocator:nil];
            up_quants = [device newBufferWithBytesNoCopy:
                             image_bytes + image_header->up_quants_offset
                                                  length:quant_bytes
                                                 options:mapped_options
                                             deallocator:nil];
            up_metadata = [device newBufferWithBytesNoCopy:
                               image_bytes + image_header->up_metadata_offset
                                                    length:metadata_bytes
                                                   options:mapped_options
                                               deallocator:nil];
            down_quants = [device newBufferWithBytesNoCopy:
                               image_bytes + image_header->down_quants_offset
                                                    length:down_quant_bytes
                                                   options:mapped_options
                                               deallocator:nil];
            down_metadata = [device newBufferWithBytesNoCopy:
                                 image_bytes + image_header->down_metadata_offset
                                                      length:down_metadata_bytes
                                                     options:mapped_options
                                                 deallocator:nil];
        }
        if (x == nil || gate_quants == nil || gate_metadata == nil ||
            up_quants == nil || up_metadata == nil || gate == nil || up == nil ||
            down_quants == nil || down_metadata == nil ||
            gate_interleaved == nil || up_interleaved == nil ||
            baseline_output == nil || fused_output == nil || generic_output == nil ||
            full_mlp_output == nil) {
            set_error(error_message, error_message_capacity,
                      @"Metal buffer allocation or no-copy mapping failed");
            if (image_mapping != MAP_FAILED) munmap(image_mapping, image_length);
            if (image_file >= 0) close(image_file);
            return 6;
        }

        uint16_t *x_values = x.contents;
        for (size_t index = 0; index < QWEN38_HIDDEN_SIZE; ++index) {
            float value = 0.75f * sinf((float)index * 0.013f) +
                          0.20f * cosf((float)index * 0.031f);
            x_values[index] = half_bits(value);
        }
        if (image_header == NULL) {
            fill_weights(gate_quants.contents, gate_metadata.contents,
                         group_count, 0x36a110cu);
            fill_weights(up_quants.contents, up_metadata.contents,
                         group_count, 0x27b4c51u);
            fill_weights(down_quants.contents, down_metadata.contents,
                         down_group_count, 0x36d027bu);
        }
        interleave_weights(gate_interleaved.contents, gate_quants.contents,
                           gate_metadata.contents, group_count);
        interleave_weights(up_interleaved.contents, up_quants.contents,
                           up_metadata.contents, group_count);
        static const unsigned schedule_threads[] = {64, 128, 256, 512};
        const unsigned schedule_count =
            (unsigned)(sizeof(schedule_threads) / sizeof(schedule_threads[0]));
        const unsigned search_iterations = measured_iterations < 10 ?
                                           measured_iterations : 10;
        result->schedule_candidate_count = schedule_count;
        double best_schedule_ms = HUGE_VAL;
        NSUInteger selected_threads = 0;
        for (unsigned candidate = 0; candidate < schedule_count; ++candidate) {
            NSUInteger threads = schedule_threads[candidate];
            result->schedule_threads[candidate] = (unsigned)threads;
            if (threads > fused.maxTotalThreadsPerThreadgroup ||
                !run_fused(queue, fused, x, gate_quants, gate_metadata,
                           up_quants, up_metadata, fused_output, threads)) {
                result->schedule_ms[candidate] = 0.0;
                continue;
            }
            double samples[5];
            BOOL schedule_ok = YES;
            for (unsigned sample = 0; sample < 5 && schedule_ok; ++sample) {
                double schedule_start = monotonic_seconds();
                for (unsigned iteration = 0; iteration < search_iterations; ++iteration) {
                    schedule_ok = run_fused(queue, fused, x,
                                            gate_quants, gate_metadata,
                                            up_quants, up_metadata,
                                            fused_output, threads);
                    if (!schedule_ok) {
                        break;
                    }
                }
                samples[sample] = (monotonic_seconds() - schedule_start) *
                                  1000.0 / search_iterations;
            }
            if (!schedule_ok) {
                result->schedule_ms[candidate] = 0.0;
                continue;
            }
            double schedule_ms = median_five(samples);
            result->schedule_ms[candidate] = schedule_ms;
            if (schedule_ms < best_schedule_ms) {
                best_schedule_ms = schedule_ms;
                selected_threads = threads;
            }
        }
        if (selected_threads == 0) {
            set_error(error_message, error_message_capacity, @"no valid Metal schedule");
            return 7;
        }
        result->selected_threads_per_threadgroup = (unsigned)selected_threads;

        for (unsigned iteration = 0; iteration < warmup_iterations; ++iteration) {
            if (!run_baseline(queue, matvec, silu, x,
                              gate_quants, gate_metadata, up_quants, up_metadata,
                              gate, up, baseline_output, selected_threads) ||
                !run_fused(queue, fused, x,
                           gate_quants, gate_metadata, up_quants, up_metadata,
                           fused_output, selected_threads)) {
                set_error(error_message, error_message_capacity, @"Metal warmup failed");
                return 8;
            }
        }

        for (unsigned iteration = 0; iteration < warmup_iterations; ++iteration) {
            if (!run_full_mlp(queue, fused, down, x,
                              gate_quants, gate_metadata, up_quants, up_metadata,
                              down_quants, down_metadata, fused_output,
                              full_mlp_output, selected_threads)) {
                set_error(error_message, error_message_capacity,
                          @"full MLP Metal warmup failed");
                return 9;
            }
        }

        if (!run_interleaved(queue, interleaved, silu, x,
                             gate_interleaved, up_interleaved,
                             gate, up, generic_output, selected_threads)) {
            set_error(error_message, error_message_capacity,
                      @"interleaved Metal warmup failed");
            return 9;
        }

        double start = monotonic_seconds();
        for (unsigned iteration = 0; iteration < measured_iterations; ++iteration) {
            if (!run_interleaved(queue, interleaved, silu, x,
                                 gate_interleaved, up_interleaved,
                                 gate, up, generic_output, selected_threads)) {
                set_error(error_message, error_message_capacity,
                          @"interleaved Metal dispatch failed");
                return 10;
            }
        }
        double interleaved_seconds = monotonic_seconds() - start;

        start = monotonic_seconds();
        for (unsigned iteration = 0; iteration < measured_iterations; ++iteration) {
            if (!run_baseline(queue, matvec, silu, x,
                              gate_quants, gate_metadata, up_quants, up_metadata,
                              gate, up, baseline_output, selected_threads)) {
                set_error(error_message, error_message_capacity, @"baseline Metal dispatch failed");
                return 11;
            }
        }
        double baseline_seconds = monotonic_seconds() - start;

        start = monotonic_seconds();
        for (unsigned iteration = 0; iteration < measured_iterations; ++iteration) {
            if (!run_fused(queue, fused, x,
                           gate_quants, gate_metadata, up_quants, up_metadata,
                           fused_output, selected_threads)) {
                set_error(error_message, error_message_capacity, @"fused Metal dispatch failed");
                return 12;
            }
        }
        double fused_seconds = monotonic_seconds() - start;

        start = monotonic_seconds();
        for (unsigned iteration = 0; iteration < measured_iterations; ++iteration) {
            if (!run_full_mlp(queue, fused, down, x,
                              gate_quants, gate_metadata, up_quants, up_metadata,
                              down_quants, down_metadata, fused_output,
                              full_mlp_output, selected_threads)) {
                set_error(error_message, error_message_capacity,
                          @"full MLP Metal dispatch failed");
                return 13;
            }
        }
        double full_mlp_seconds = monotonic_seconds() - start;

        result->generic_interleaved_ms =
            interleaved_seconds * 1000.0 / measured_iterations;
        result->split_unfused_ms = baseline_seconds * 1000.0 / measured_iterations;
        result->fused_ms = fused_seconds * 1000.0 / measured_iterations;
        result->layout_vector_speedup =
            result->generic_interleaved_ms / result->split_unfused_ms;
        result->fusion_speedup = result->split_unfused_ms / result->fused_ms;
        result->total_speedup = result->generic_interleaved_ms / result->fused_ms;
        result->fused_weight_gbps = (2.0 * (double)matrix_bytes) /
                                    (result->fused_ms * 1.0e6);
        result->full_mlp_ms = full_mlp_seconds * 1000.0 / measured_iterations;
        result->full_mlp_effective_weight_gbps =
            (2.0 * (double)matrix_bytes + (double)down_matrix_bytes) /
            (result->full_mlp_ms * 1.0e6);
        result->footprint_peak_bytes = physical_footprint();

        const float *baseline_values = baseline_output.contents;
        const float *fused_values = fused_output.contents;
        const float *generic_values = generic_output.contents;
        double max_gpu_error = 0.0;
        double max_generic_error = 0.0;
        for (size_t row = 0; row < QWEN38_MLP_SIZE; ++row) {
            double difference = fabs((double)baseline_values[row] - (double)fused_values[row]);
            if (difference > max_gpu_error) {
                max_gpu_error = difference;
            }
            double generic_difference =
                fabs((double)generic_values[row] - (double)fused_values[row]);
            if (generic_difference > max_generic_error) {
                max_generic_error = generic_difference;
            }
        }
        result->max_abs_error_vs_unfused_gpu = max_gpu_error;
        result->max_abs_error_vs_generic_gpu = max_generic_error;

        const uint8_t *gate_quant_data = gate_quants.contents;
        const qwen38_q4_meta *gate_meta_data = gate_metadata.contents;
        const uint8_t *up_quant_data = up_quants.contents;
        const qwen38_q4_meta *up_meta_data = up_metadata.contents;
        double max_cpu_error = 0.0;
        for (size_t row = 0; row < 8; ++row) {
            float gate_value = cpu_row_dot(gate_quant_data, gate_meta_data,
                                           x_values, row);
            float up_value = cpu_row_dot(up_quant_data, up_meta_data,
                                         x_values, row);
            float reference = (gate_value / (1.0f + expf(-gate_value))) * up_value;
            double difference = fabs((double)reference - (double)fused_values[row]);
            if (difference > max_cpu_error) {
                max_cpu_error = difference;
            }
        }
        result->max_abs_error_vs_cpu_first_8_rows = max_cpu_error;
        if (image_header != NULL && image_header->source_reference_count == 8) {
            double max_source_error = 0.0;
            for (size_t row = 0; row < 8; ++row) {
                result->source_reference_first_8[row] =
                    image_header->source_reference_first_8[row];
                result->fused_output_first_8[row] = fused_values[row];
                double difference = fabs(
                    (double)image_header->source_reference_first_8[row] -
                    (double)fused_values[row]);
                if (difference > max_source_error) {
                    max_source_error = difference;
                }
            }
            result->max_abs_error_vs_source_bf16_first_8_rows =
                max_source_error;
        }
        if (image_header != NULL &&
            image_header->source_mlp_reference_count == 8) {
            const float *full_mlp_values = full_mlp_output.contents;
            double max_full_mlp_error = 0.0;
            for (size_t row = 0; row < 8; ++row) {
                result->source_mlp_reference_first_8[row] =
                    image_header->source_mlp_reference_first_8[row];
                result->full_mlp_output_first_8[row] = full_mlp_values[row];
                double difference = fabs(
                    (double)image_header->source_mlp_reference_first_8[row] -
                    (double)full_mlp_values[row]);
                if (difference > max_full_mlp_error) {
                    max_full_mlp_error = difference;
                }
            }
            result->max_abs_error_full_mlp_vs_source_bf16_first_8 =
                max_full_mlp_error;
        }
        if (image_mapping != MAP_FAILED) {
            munmap(image_mapping, image_length);
            close(image_file);
        }
    }
    return 0;
}
