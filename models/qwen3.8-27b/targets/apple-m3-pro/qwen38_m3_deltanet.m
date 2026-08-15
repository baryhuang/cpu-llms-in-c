#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "qwen38_m3.h"

#include <mach/mach.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void set_error(char *message, size_t capacity, NSString *text) {
    if (message == NULL || capacity == 0) {
        return;
    }
    const char *utf8 = text != nil ? text.UTF8String : "unknown Metal error";
    snprintf(message, capacity, "%s", utf8 != NULL ? utf8 : "unknown Metal error");
}

static size_t physical_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t status = task_info(mach_task_self(), TASK_VM_INFO,
                                     (task_info_t)&info, &count);
    return status == KERN_SUCCESS ? (size_t)info.phys_footprint : 0;
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
                                         [NSString stringWithFormat:
                                             @"missing Metal function %@", name]}];
        }
        return nil;
    }
    return [library.device newComputePipelineStateWithFunction:function error:error];
}

static BOOL run_direct(id<MTLCommandQueue> queue,
                       id<MTLComputePipelineState> pipeline,
                       id<MTLBuffer> query, id<MTLBuffer> key,
                       id<MTLBuffer> value, id<MTLBuffer> decay,
                       id<MTLBuffer> beta, id<MTLBuffer> state,
                       id<MTLBuffer> output) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:key offset:0 atIndex:1];
    [encoder setBuffer:value offset:0 atIndex:2];
    [encoder setBuffer:decay offset:0 atIndex:3];
    [encoder setBuffer:beta offset:0 atIndex:4];
    [encoder setBuffer:state offset:0 atIndex:5];
    [encoder setBuffer:output offset:0 atIndex:6];
    NSUInteger count = QWEN38_DELTA_VALUE_HEADS * QWEN38_DELTA_HEAD_SIZE;
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    return command.status == MTLCommandBufferStatusCompleted;
}

static BOOL run_vectorized(id<MTLCommandQueue> queue,
                      id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> query, id<MTLBuffer> key,
                      id<MTLBuffer> value, id<MTLBuffer> decay,
                      id<MTLBuffer> beta, id<MTLBuffer> state,
                      id<MTLBuffer> output) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:query offset:0 atIndex:0];
    [encoder setBuffer:key offset:0 atIndex:1];
    [encoder setBuffer:value offset:0 atIndex:2];
    [encoder setBuffer:decay offset:0 atIndex:3];
    [encoder setBuffer:beta offset:0 atIndex:4];
    [encoder setBuffer:state offset:0 atIndex:5];
    [encoder setBuffer:output offset:0 atIndex:6];
    NSUInteger count = QWEN38_DELTA_VALUE_HEADS *
                       (QWEN38_DELTA_HEAD_SIZE / 2);
    [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    return command.status == MTLCommandBufferStatusCompleted;
}

static void fill_inputs(float *query, float *key, float *value,
                        float *decay, float *beta, float *state) {
    const size_t vector_count =
        QWEN38_DELTA_VALUE_HEADS * QWEN38_DELTA_HEAD_SIZE;
    for (unsigned head = 0; head < QWEN38_DELTA_VALUE_HEADS; ++head) {
        double query_squared = 0.0;
        double key_squared = 0.0;
        for (unsigned index = 0; index < QWEN38_DELTA_HEAD_SIZE; ++index) {
            size_t position = (size_t)head * QWEN38_DELTA_HEAD_SIZE + index;
            query[position] = 0.7f * sinf((float)position * 0.017f) +
                              0.2f * cosf((float)index * 0.071f);
            key[position] = 0.6f * cosf((float)position * 0.019f) -
                            0.15f * sinf((float)index * 0.043f);
            value[position] = 0.35f * sinf((float)position * 0.011f) +
                              0.1f * cosf((float)position * 0.029f);
            query_squared += (double)query[position] * query[position];
            key_squared += (double)key[position] * key[position];
        }
        float query_scale = 1.0f /
            sqrtf((float)(query_squared + 1.0e-6) * QWEN38_DELTA_HEAD_SIZE);
        float key_scale = 1.0f / sqrtf((float)(key_squared + 1.0e-6));
        for (unsigned index = 0; index < QWEN38_DELTA_HEAD_SIZE; ++index) {
            size_t position = (size_t)head * QWEN38_DELTA_HEAD_SIZE + index;
            query[position] *= query_scale;
            key[position] *= key_scale;
        }
        decay[head] = expf(-0.02f - 0.001f * (float)(head % 11));
        beta[head] = 1.0f /
            (1.0f + expf(0.35f - 0.025f * (float)(head % 13)));
    }
    const size_t state_count =
        (size_t)QWEN38_DELTA_VALUE_HEADS * QWEN38_DELTA_HEAD_SIZE *
        QWEN38_DELTA_HEAD_SIZE;
    for (size_t index = 0; index < state_count; ++index) {
        state[index] = 0.008f * sinf((float)index * 0.0031f) +
                       0.003f * cosf((float)index * 0.0073f);
    }
    (void)vector_count;
}

static void cpu_reference(const float *query, const float *key,
                          const float *value, const float *decay,
                          const float *beta, float *state, float *output) {
    for (unsigned head = 0; head < QWEN38_DELTA_VALUE_HEADS; ++head) {
        size_t vector_base = (size_t)head * QWEN38_DELTA_HEAD_SIZE;
        size_t state_head_base = vector_base * QWEN38_DELTA_HEAD_SIZE;
        for (unsigned value_index = 0; value_index < QWEN38_DELTA_HEAD_SIZE;
             ++value_index) {
            float kv_memory = 0.0f;
            float previous_output = 0.0f;
            float key_query = 0.0f;
            for (unsigned key_index = 0; key_index < QWEN38_DELTA_HEAD_SIZE;
                 ++key_index) {
                size_t state_index = state_head_base +
                    (size_t)key_index * QWEN38_DELTA_HEAD_SIZE + value_index;
                float decayed = state[state_index] * decay[head];
                kv_memory += decayed * key[vector_base + key_index];
                previous_output += decayed * query[vector_base + key_index];
                key_query += key[vector_base + key_index] *
                             query[vector_base + key_index];
            }
            float delta = (value[vector_base + value_index] - kv_memory) *
                          beta[head];
            for (unsigned key_index = 0; key_index < QWEN38_DELTA_HEAD_SIZE;
                 ++key_index) {
                size_t state_index = state_head_base +
                    (size_t)key_index * QWEN38_DELTA_HEAD_SIZE + value_index;
                state[state_index] = state[state_index] * decay[head] +
                    key[vector_base + key_index] * delta;
            }
            output[vector_base + value_index] =
                previous_output + key_query * delta;
        }
    }
}

int qwen38_m3_run_deltanet_core_benchmark(
    const char *metallib_path,
    unsigned warmup_iterations,
    unsigned measured_iterations,
    qwen38_m3_deltanet_core_result *result,
    char *error_message,
    size_t error_message_capacity) {
    if (metallib_path == NULL || result == NULL || measured_iterations == 0) {
        set_error(error_message, error_message_capacity,
                  @"invalid DeltaNet benchmark arguments");
        return 1;
    }
    memset(result, 0, sizeof(*result));
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            set_error(error_message, error_message_capacity, @"no Metal device");
            return 2;
        }
        snprintf(result->device_name, sizeof(result->device_name), "%s",
                 device.name.UTF8String);
        NSError *error = nil;
        NSURL *url = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:metallib_path]];
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
        id<MTLComputePipelineState> direct = library != nil ?
            make_pipeline(library, @"qwen38_deltanet_direct", &error) : nil;
        id<MTLComputePipelineState> vectorized = library != nil ?
            make_pipeline(library, @"qwen38_deltanet_float2_vectorized", &error) : nil;
        if (direct == nil || vectorized == nil) {
            set_error(error_message, error_message_capacity,
                      error.localizedDescription);
            return 3;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            set_error(error_message, error_message_capacity,
                      @"cannot create Metal command queue");
            return 4;
        }

        const size_t vector_count =
            QWEN38_DELTA_VALUE_HEADS * QWEN38_DELTA_HEAD_SIZE;
        const size_t vector_bytes = vector_count * sizeof(float);
        const size_t state_count = vector_count * QWEN38_DELTA_HEAD_SIZE;
        const size_t state_bytes = state_count * sizeof(float);
        const size_t scalar_bytes = QWEN38_DELTA_VALUE_HEADS * sizeof(float);
        result->warmup_iterations = warmup_iterations;
        result->measured_iterations = measured_iterations;
        result->state_bytes = state_bytes;
        result->footprint_before_bytes = physical_footprint();
        MTLResourceOptions options = MTLResourceStorageModeShared;
        id<MTLBuffer> query = [device newBufferWithLength:vector_bytes options:options];
        id<MTLBuffer> key = [device newBufferWithLength:vector_bytes options:options];
        id<MTLBuffer> value = [device newBufferWithLength:vector_bytes options:options];
        id<MTLBuffer> decay = [device newBufferWithLength:scalar_bytes options:options];
        id<MTLBuffer> beta = [device newBufferWithLength:scalar_bytes options:options];
        id<MTLBuffer> initial = [device newBufferWithLength:state_bytes options:options];
        id<MTLBuffer> direct_state = [device newBufferWithLength:state_bytes options:options];
        id<MTLBuffer> vectorized_state = [device newBufferWithLength:state_bytes options:options];
        id<MTLBuffer> direct_output = [device newBufferWithLength:vector_bytes options:options];
        id<MTLBuffer> vectorized_output = [device newBufferWithLength:vector_bytes options:options];
        if (query == nil || key == nil || value == nil || decay == nil || beta == nil ||
            initial == nil || direct_state == nil || vectorized_state == nil ||
            direct_output == nil || vectorized_output == nil) {
            set_error(error_message, error_message_capacity,
                      @"DeltaNet Metal allocation failed");
            return 5;
        }
        fill_inputs(query.contents, key.contents, value.contents,
                    decay.contents, beta.contents, initial.contents);
        memcpy(direct_state.contents, initial.contents, state_bytes);
        memcpy(vectorized_state.contents, initial.contents, state_bytes);

        float *reference_state = malloc(state_bytes);
        float *reference_output = malloc(vector_bytes);
        if (reference_state == NULL || reference_output == NULL) {
            free(reference_state);
            free(reference_output);
            set_error(error_message, error_message_capacity,
                      @"DeltaNet reference allocation failed");
            return 6;
        }
        memcpy(reference_state, initial.contents, state_bytes);
        cpu_reference(query.contents, key.contents, value.contents,
                      decay.contents, beta.contents,
                      reference_state, reference_output);
        if (!run_direct(queue, direct, query, key, value, decay, beta,
                        direct_state, direct_output) ||
            !run_vectorized(queue, vectorized, query, key, value, decay, beta,
                            vectorized_state, vectorized_output)) {
            free(reference_state);
            free(reference_output);
            set_error(error_message, error_message_capacity,
                      @"DeltaNet correctness dispatch failed");
            return 7;
        }
        double max_output_error = 0.0;
        double max_state_error = 0.0;
        const float *direct_output_values = direct_output.contents;
        const float *direct_state_values = direct_state.contents;
        const float *vectorized_output_values = vectorized_output.contents;
        const float *vectorized_state_values = vectorized_state.contents;
        for (size_t index = 0; index < vector_count; ++index) {
            double direct_error = fabs((double)reference_output[index] -
                                       direct_output_values[index]);
            double vectorized_error = fabs((double)reference_output[index] -
                                           vectorized_output_values[index]);
            if (direct_error > max_output_error) max_output_error = direct_error;
            if (vectorized_error > max_output_error) {
                max_output_error = vectorized_error;
            }
        }
        for (size_t index = 0; index < state_count; ++index) {
            double direct_error = fabs((double)reference_state[index] -
                                       direct_state_values[index]);
            double vectorized_error = fabs((double)reference_state[index] -
                                           vectorized_state_values[index]);
            if (direct_error > max_state_error) max_state_error = direct_error;
            if (vectorized_error > max_state_error) {
                max_state_error = vectorized_error;
            }
        }
        for (size_t index = 0; index < 8; ++index) {
            result->reference_output_first_8[index] = reference_output[index];
            result->vectorized_output_first_8[index] =
                vectorized_output_values[index];
        }
        result->max_abs_error_output = max_output_error;
        result->max_abs_error_state = max_state_error;
        free(reference_state);
        free(reference_output);

        memcpy(direct_state.contents, initial.contents, state_bytes);
        memcpy(vectorized_state.contents, initial.contents, state_bytes);
        for (unsigned iteration = 0; iteration < warmup_iterations; ++iteration) {
            if (!run_direct(queue, direct, query, key, value, decay, beta,
                            direct_state, direct_output) ||
                !run_vectorized(queue, vectorized, query, key, value,
                                decay, beta, vectorized_state,
                                vectorized_output)) {
                set_error(error_message, error_message_capacity,
                          @"DeltaNet warmup failed");
                return 8;
            }
        }
        double direct_samples[5];
        double vectorized_samples[5];
        for (unsigned sample = 0; sample < 5; ++sample) {
            memcpy(direct_state.contents, initial.contents, state_bytes);
            memcpy(vectorized_state.contents, initial.contents, state_bytes);
            BOOL vectorized_first = (sample & 1u) != 0;
            for (unsigned path = 0; path < 2; ++path) {
                BOOL measure_vectorized = (path == 0) == vectorized_first;
                double start = monotonic_seconds();
                for (unsigned iteration = 0; iteration < measured_iterations;
                     ++iteration) {
                    BOOL ok = measure_vectorized ?
                        run_vectorized(queue, vectorized, query, key, value,
                                       decay, beta, vectorized_state,
                                       vectorized_output) :
                        run_direct(queue, direct, query, key, value, decay,
                                   beta, direct_state, direct_output);
                    if (!ok) {
                        set_error(error_message, error_message_capacity,
                                  @"DeltaNet measured dispatch failed");
                        return 9;
                    }
                }
                double milliseconds = (monotonic_seconds() - start) * 1000.0 /
                                      measured_iterations;
                if (measure_vectorized) {
                    vectorized_samples[sample] = milliseconds;
                } else {
                    direct_samples[sample] = milliseconds;
                }
            }
        }
        result->direct_device_ms = median_five(direct_samples);
        result->float2_vectorized_ms = median_five(vectorized_samples);
        result->speedup = result->direct_device_ms /
                          result->float2_vectorized_ms;
        result->direct_state_gbps = (3.0 * state_bytes) /
                                    (result->direct_device_ms * 1.0e6);
        result->vectorized_state_gbps = (3.0 * state_bytes) /
                                        (result->float2_vectorized_ms * 1.0e6);
        result->footprint_peak_bytes = physical_footprint();
    }
    return 0;
}
