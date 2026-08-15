#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "qwen38_m3.h"
#include "qwen38_m3_image.h"

#include <fcntl.h>
#include <mach/mach.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef struct __attribute__((packed, aligned(2))) {
    uint16_t scale;
    uint16_t bias;
} qwen38_layer_q4_meta;

_Static_assert(sizeof(qwen38_layer_q4_meta) == 4,
               "CPU and Metal metadata strides must match");

enum {
    QWEN38_LAYER_CONSTANTS = 51424,
    QWEN38_LAYER_THREADS = 256,
    QWEN38_LAYER_SIMDGROUPS = QWEN38_LAYER_THREADS / 32
};

typedef struct {
    id<MTLComputePipelineState> rms_half;
    id<MTLComputePipelineState> rms_float;
    id<MTLComputePipelineState> delta_inputs;
    id<MTLComputePipelineState> conv;
    id<MTLComputePipelineState> prepare;
    id<MTLComputePipelineState> recurrent;
    id<MTLComputePipelineState> gated_norm;
    id<MTLComputePipelineState> delta_output;
    id<MTLComputePipelineState> mlp_gate_up;
    id<MTLComputePipelineState> mlp_down;
} qwen38_layer_pipelines;

typedef struct {
    id<MTLBuffer> input;
    id<MTLBuffer> normalized;
    id<MTLBuffer> constants;
    id<MTLBuffer> delta_input_quants;
    id<MTLBuffer> delta_input_metadata;
    id<MTLBuffer> projected;
    id<MTLBuffer> conv_state;
    id<MTLBuffer> convolved;
    id<MTLBuffer> query;
    id<MTLBuffer> key;
    id<MTLBuffer> value;
    id<MTLBuffer> decay;
    id<MTLBuffer> beta;
    id<MTLBuffer> recurrent_state;
    id<MTLBuffer> core;
    id<MTLBuffer> gated;
    id<MTLBuffer> delta_output_quants;
    id<MTLBuffer> delta_output_metadata;
    id<MTLBuffer> mixer_output;
    id<MTLBuffer> post_normalized;
    id<MTLBuffer> gate_quants;
    id<MTLBuffer> gate_metadata;
    id<MTLBuffer> up_quants;
    id<MTLBuffer> up_metadata;
    id<MTLBuffer> down_quants;
    id<MTLBuffer> down_metadata;
    id<MTLBuffer> mlp_intermediate;
    id<MTLBuffer> output;
} qwen38_layer_buffers;

static void layer_set_error(char *message, size_t capacity, NSString *text) {
    if (message == NULL || capacity == 0) {
        return;
    }
    const char *utf8 = text != nil ? text.UTF8String : "unknown Metal error";
    snprintf(message, capacity, "%s",
             utf8 != NULL ? utf8 : "unknown Metal error");
}

static size_t layer_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t status = task_info(mach_task_self(), TASK_VM_INFO,
                                     (task_info_t)&info, &count);
    return status == KERN_SUCCESS ? (size_t)info.phys_footprint : 0;
}

static double layer_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC_RAW, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static id<MTLComputePipelineState>
layer_pipeline(id<MTLDevice> device, id<MTLLibrary> library,
               NSString *name, NSError **error) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        return nil;
    }
    return [device newComputePipelineStateWithFunction:function error:error];
}

static uint16_t layer_half_bits(float value) {
    __fp16 converted = (__fp16)value;
    uint16_t bits;
    memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

static float layer_half_value(uint16_t bits) {
    __fp16 converted;
    memcpy(&converted, &bits, sizeof(converted));
    return (float)converted;
}

static uint64_t layer_align_page(uint64_t value) {
    return (value + QWEN38_M3_IMAGE_HEADER_BYTES - 1) &
           ~(uint64_t)(QWEN38_M3_IMAGE_HEADER_BYTES - 1);
}

static int validate_layer_image(const qwen38_m3_image_header *header,
                                size_t length) {
    const uint64_t mlp_weight_bytes =
        (uint64_t)QWEN38_MLP_SIZE * QWEN38_HIDDEN_SIZE / 2;
    const uint64_t mlp_metadata_bytes =
        (uint64_t)QWEN38_MLP_SIZE *
        (QWEN38_HIDDEN_SIZE / QWEN38_Q4_GROUP_SIZE) * 4;
    const uint64_t delta_input_quants =
        (uint64_t)QWEN38_DELTA_INPUT_ROWS * QWEN38_HIDDEN_SIZE / 2;
    const uint64_t delta_input_metadata_payload =
        (uint64_t)QWEN38_DELTA_INPUT_ROWS *
        (QWEN38_HIDDEN_SIZE / QWEN38_Q4_GROUP_SIZE) * 4;
    const uint64_t delta_output_quants =
        (uint64_t)QWEN38_HIDDEN_SIZE *
        QWEN38_DELTA_OUTPUT_INPUTS / 2;
    const uint64_t delta_output_metadata =
        (uint64_t)QWEN38_HIDDEN_SIZE *
        (QWEN38_DELTA_OUTPUT_INPUTS / QWEN38_Q4_GROUP_SIZE) * 4;
    uint64_t expected = QWEN38_M3_IMAGE_HEADER_BYTES;
#define Q38_CHECK_SEGMENT(name, bytes) do { \
    if (header->name##_offset != expected || \
        header->name##_bytes != (bytes)) return -1; \
    expected += (bytes); \
} while (0)
    if (memcmp(header->magic, QWEN38_M3_IMAGE_MAGIC, 8) != 0 ||
        header->version != QWEN38_M3_IMAGE_VERSION ||
        header->header_bytes != QWEN38_M3_IMAGE_HEADER_BYTES ||
        header->hidden_size != QWEN38_HIDDEN_SIZE ||
        header->rows != QWEN38_MLP_SIZE ||
        header->group_size != QWEN38_Q4_GROUP_SIZE ||
        header->layer_index != 0 ||
        header->source_reference_count != 8 ||
        header->source_mlp_reference_count != 8 ||
        header->delta_input_rows != QWEN38_DELTA_INPUT_ROWS ||
        header->delta_input_groups_per_row !=
            QWEN38_HIDDEN_SIZE / QWEN38_Q4_GROUP_SIZE ||
        header->delta_output_rows != QWEN38_HIDDEN_SIZE ||
        header->delta_output_groups_per_row !=
            QWEN38_DELTA_OUTPUT_INPUTS / QWEN38_Q4_GROUP_SIZE ||
        header->constants_f32_count != QWEN38_LAYER_CONSTANTS ||
        memcmp(header->source_sha256, QWEN38_M3_EXPECTED_SOURCE_SHA256,
               QWEN38_M3_SOURCE_SHA256_LENGTH) != 0) {
        return -1;
    }
    Q38_CHECK_SEGMENT(gate_quants, mlp_weight_bytes);
    Q38_CHECK_SEGMENT(gate_metadata, mlp_metadata_bytes);
    Q38_CHECK_SEGMENT(up_quants, mlp_weight_bytes);
    Q38_CHECK_SEGMENT(up_metadata, mlp_metadata_bytes);
    Q38_CHECK_SEGMENT(down_quants, mlp_weight_bytes);
    Q38_CHECK_SEGMENT(down_metadata, mlp_metadata_bytes);
    expected = layer_align_page(expected);
    if (header->constants_offset != expected ||
        header->constants_bytes !=
            layer_align_page(QWEN38_LAYER_CONSTANTS * sizeof(float)) ||
        header->input_norm_constants_index != 0 ||
        header->post_norm_constants_index != QWEN38_HIDDEN_SIZE ||
        header->conv_constants_index != 2 * QWEN38_HIDDEN_SIZE ||
        header->a_log_constants_index != 51200 ||
        header->dt_bias_constants_index != 51248 ||
        header->recurrent_norm_constants_index != 51296) {
        return -1;
    }
    expected += header->constants_bytes;
    Q38_CHECK_SEGMENT(delta_input_quants, delta_input_quants);
    Q38_CHECK_SEGMENT(delta_input_metadata,
                      layer_align_page(delta_input_metadata_payload));
    Q38_CHECK_SEGMENT(delta_output_quants, delta_output_quants);
    Q38_CHECK_SEGMENT(delta_output_metadata, delta_output_metadata);
#undef Q38_CHECK_SEGMENT
    return expected == length ? 0 : -1;
}

static void encode_rms_half(id<MTLComputeCommandEncoder> encoder,
                            id<MTLComputePipelineState> pipeline,
                            id<MTLBuffer> input, id<MTLBuffer> weight,
                            NSUInteger weight_offset,
                            id<MTLBuffer> output) {
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:weight offset:weight_offset atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void encode_rms_float(id<MTLComputeCommandEncoder> encoder,
                             id<MTLComputePipelineState> pipeline,
                             id<MTLBuffer> input, id<MTLBuffer> weight,
                             NSUInteger weight_offset,
                             id<MTLBuffer> output) {
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:weight offset:weight_offset atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void encode_q4_rows(id<MTLComputeCommandEncoder> encoder,
                           id<MTLComputePipelineState> pipeline,
                           id<MTLBuffer> input, id<MTLBuffer> quants,
                           id<MTLBuffer> metadata, id<MTLBuffer> output,
                           NSUInteger rows) {
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:quants offset:0 atIndex:1];
    [encoder setBuffer:metadata offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    NSUInteger groups = (rows + QWEN38_LAYER_SIMDGROUPS - 1) /
                        QWEN38_LAYER_SIMDGROUPS;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(QWEN38_LAYER_THREADS, 1, 1)];
}

static void encode_one_layer(id<MTLComputeCommandEncoder> encoder,
                             const qwen38_layer_pipelines *p,
                             const qwen38_layer_buffers *b,
                             const qwen38_m3_image_header *h) {
    encode_rms_half(encoder, p->rms_half, b->input, b->constants,
                    h->input_norm_constants_index * sizeof(float),
                    b->normalized);
    encode_q4_rows(encoder, p->delta_inputs, b->normalized,
                   b->delta_input_quants, b->delta_input_metadata,
                   b->projected, QWEN38_DELTA_INPUT_ROWS);

    [encoder setComputePipelineState:p->conv];
    [encoder setBuffer:b->projected offset:0 atIndex:0];
    [encoder setBuffer:b->constants
                 offset:h->conv_constants_index * sizeof(float) atIndex:1];
    [encoder setBuffer:b->conv_state offset:0 atIndex:2];
    [encoder setBuffer:b->convolved offset:0 atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(QWEN38_DELTA_QKV_ROWS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [encoder setComputePipelineState:p->prepare];
    [encoder setBuffer:b->convolved offset:0 atIndex:0];
    [encoder setBuffer:b->projected offset:0 atIndex:1];
    [encoder setBuffer:b->constants
                 offset:h->a_log_constants_index * sizeof(float) atIndex:2];
    [encoder setBuffer:b->constants
                 offset:h->dt_bias_constants_index * sizeof(float) atIndex:3];
    [encoder setBuffer:b->query offset:0 atIndex:4];
    [encoder setBuffer:b->key offset:0 atIndex:5];
    [encoder setBuffer:b->value offset:0 atIndex:6];
    [encoder setBuffer:b->decay offset:0 atIndex:7];
    [encoder setBuffer:b->beta offset:0 atIndex:8];
    [encoder dispatchThreadgroups:MTLSizeMake(QWEN38_DELTA_VALUE_HEADS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    [encoder setComputePipelineState:p->recurrent];
    [encoder setBuffer:b->query offset:0 atIndex:0];
    [encoder setBuffer:b->key offset:0 atIndex:1];
    [encoder setBuffer:b->value offset:0 atIndex:2];
    [encoder setBuffer:b->decay offset:0 atIndex:3];
    [encoder setBuffer:b->beta offset:0 atIndex:4];
    [encoder setBuffer:b->recurrent_state offset:0 atIndex:5];
    [encoder setBuffer:b->core offset:0 atIndex:6];
    [encoder dispatchThreads:
        MTLSizeMake(QWEN38_DELTA_VALUE_HEADS * QWEN38_DELTA_HEAD_SIZE, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [encoder setComputePipelineState:p->gated_norm];
    [encoder setBuffer:b->core offset:0 atIndex:0];
    [encoder setBuffer:b->projected offset:0 atIndex:1];
    [encoder setBuffer:b->constants
                 offset:h->recurrent_norm_constants_index * sizeof(float)
                 atIndex:2];
    [encoder setBuffer:b->gated offset:0 atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(QWEN38_DELTA_VALUE_HEADS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];

    [encoder setComputePipelineState:p->delta_output];
    [encoder setBuffer:b->gated offset:0 atIndex:0];
    [encoder setBuffer:b->delta_output_quants offset:0 atIndex:1];
    [encoder setBuffer:b->delta_output_metadata offset:0 atIndex:2];
    [encoder setBuffer:b->input offset:0 atIndex:3];
    [encoder setBuffer:b->mixer_output offset:0 atIndex:4];
    [encoder dispatchThreadgroups:
        MTLSizeMake((QWEN38_HIDDEN_SIZE + QWEN38_LAYER_SIMDGROUPS - 1) /
                    QWEN38_LAYER_SIMDGROUPS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(QWEN38_LAYER_THREADS, 1, 1)];

    encode_rms_float(encoder, p->rms_float, b->mixer_output, b->constants,
                     h->post_norm_constants_index * sizeof(float),
                     b->post_normalized);

    [encoder setComputePipelineState:p->mlp_gate_up];
    [encoder setBuffer:b->post_normalized offset:0 atIndex:0];
    [encoder setBuffer:b->gate_quants offset:0 atIndex:1];
    [encoder setBuffer:b->gate_metadata offset:0 atIndex:2];
    [encoder setBuffer:b->up_quants offset:0 atIndex:3];
    [encoder setBuffer:b->up_metadata offset:0 atIndex:4];
    [encoder setBuffer:b->mlp_intermediate offset:0 atIndex:5];
    [encoder dispatchThreadgroups:
        MTLSizeMake((QWEN38_MLP_SIZE + QWEN38_LAYER_SIMDGROUPS - 1) /
                    QWEN38_LAYER_SIMDGROUPS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(QWEN38_LAYER_THREADS, 1, 1)];

    [encoder setComputePipelineState:p->mlp_down];
    [encoder setBuffer:b->mlp_intermediate offset:0 atIndex:0];
    [encoder setBuffer:b->down_quants offset:0 atIndex:1];
    [encoder setBuffer:b->down_metadata offset:0 atIndex:2];
    [encoder setBuffer:b->mixer_output offset:0 atIndex:3];
    [encoder setBuffer:b->output offset:0 atIndex:4];
    [encoder dispatchThreadgroups:
        MTLSizeMake((QWEN38_HIDDEN_SIZE + QWEN38_LAYER_SIMDGROUPS - 1) /
                    QWEN38_LAYER_SIMDGROUPS, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(QWEN38_LAYER_THREADS, 1, 1)];
}

static BOOL submit_layers(id<MTLCommandQueue> queue,
                          const qwen38_layer_pipelines *pipelines,
                          const qwen38_layer_buffers *buffers,
                          const qwen38_m3_image_header *header,
                          unsigned iterations) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
        encode_one_layer(encoder, pipelines, buffers, header);
    }
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    return command.status == MTLCommandBufferStatusCompleted;
}

static float cpu_q4_dot_half(const uint8_t *quants,
                             const qwen38_layer_q4_meta *metadata,
                             const uint16_t *input, size_t row,
                             size_t groups) {
    float total = 0.0f;
    for (size_t group = 0; group < groups; ++group) {
        size_t block = row * groups + group;
        float scale = layer_half_value(metadata[block].scale);
        float bias = layer_half_value(metadata[block].bias);
        for (size_t lane = 0; lane < 32; ++lane) {
            uint8_t bits = quants[block * 32 + lane];
            size_t index = group * 64 + lane * 2;
            total += (scale * (float)(bits & 15u) + bias) *
                     layer_half_value(input[index]);
            total += (scale * (float)(bits >> 4) + bias) *
                     layer_half_value(input[index + 1]);
        }
    }
    return total;
}

static float cpu_q4_dot_float(const uint8_t *quants,
                              const qwen38_layer_q4_meta *metadata,
                              const float *input, size_t row,
                              size_t groups) {
    float total = 0.0f;
    for (size_t group = 0; group < groups; ++group) {
        size_t block = row * groups + group;
        float scale = layer_half_value(metadata[block].scale);
        float bias = layer_half_value(metadata[block].bias);
        for (size_t lane = 0; lane < 32; ++lane) {
            uint8_t bits = quants[block * 32 + lane];
            size_t index = group * 64 + lane * 2;
            total += (scale * (float)(bits & 15u) + bias) * input[index];
            total += (scale * (float)(bits >> 4) + bias) *
                     input[index + 1];
        }
    }
    return total;
}

static void cpu_rms_half(const uint16_t *input, const float *weight,
                         uint16_t *output) {
    double squared = 0.0;
    for (size_t index = 0; index < QWEN38_HIDDEN_SIZE; ++index) {
        float value = layer_half_value(input[index]);
        squared += (double)value * value;
    }
    float inv = 1.0f /
        sqrtf((float)(squared / QWEN38_HIDDEN_SIZE) + 1.0e-6f);
    for (size_t index = 0; index < QWEN38_HIDDEN_SIZE; ++index) {
        output[index] = layer_half_bits(layer_half_value(input[index]) * inv *
                                        weight[index]);
    }
}

static void cpu_rms_float(const float *input, const float *weight,
                          uint16_t *output) {
    double squared = 0.0;
    for (size_t index = 0; index < QWEN38_HIDDEN_SIZE; ++index) {
        squared += (double)input[index] * input[index];
    }
    float inv = 1.0f /
        sqrtf((float)(squared / QWEN38_HIDDEN_SIZE) + 1.0e-6f);
    for (size_t index = 0; index < QWEN38_HIDDEN_SIZE; ++index) {
        output[index] = layer_half_bits(input[index] * inv * weight[index]);
    }
}

static int cpu_layer_reference(const unsigned char *image,
                               const qwen38_m3_image_header *h,
                               const uint16_t *input,
                               float output_first_8[8],
                               float *state_out, float *conv_state_out) {
    const float *constants =
        (const float *)(image + h->constants_offset);
    uint16_t *normalized = malloc(QWEN38_HIDDEN_SIZE * sizeof(uint16_t));
    float *projected = malloc(QWEN38_DELTA_INPUT_ROWS * sizeof(float));
    float *convolved = malloc(QWEN38_DELTA_QKV_ROWS * sizeof(float));
    const size_t delta_vector_count =
        QWEN38_DELTA_VALUE_HEADS * QWEN38_DELTA_HEAD_SIZE;
    float *query = malloc(delta_vector_count * sizeof(float));
    float *key = malloc(delta_vector_count * sizeof(float));
    float *value = malloc(delta_vector_count * sizeof(float));
    float *core = malloc(delta_vector_count * sizeof(float));
    float *gated = malloc(delta_vector_count * sizeof(float));
    float *mixer_output = malloc(QWEN38_HIDDEN_SIZE * sizeof(float));
    uint16_t *post = malloc(QWEN38_HIDDEN_SIZE * sizeof(uint16_t));
    float *intermediate = malloc(QWEN38_MLP_SIZE * sizeof(float));
    if (normalized == NULL || projected == NULL || convolved == NULL ||
        query == NULL || key == NULL || value == NULL || core == NULL ||
        gated == NULL || mixer_output == NULL || post == NULL ||
        intermediate == NULL) {
        free(normalized); free(projected); free(convolved);
        free(query); free(key); free(value); free(core); free(gated);
        free(mixer_output); free(post); free(intermediate);
        return -1;
    }
    cpu_rms_half(input, constants + h->input_norm_constants_index,
                 normalized);
    const uint8_t *delta_input_quants =
        image + h->delta_input_quants_offset;
    const qwen38_layer_q4_meta *delta_input_meta =
        (const qwen38_layer_q4_meta *)(
            image + h->delta_input_metadata_offset);
    for (size_t row = 0; row < QWEN38_DELTA_INPUT_ROWS; ++row) {
        projected[row] = cpu_q4_dot_half(
            delta_input_quants, delta_input_meta, normalized, row, 80);
    }
    const float *conv_weight = constants + h->conv_constants_index;
    memset(conv_state_out, 0,
           QWEN38_DELTA_QKV_ROWS * QWEN38_DELTA_CONV_SIZE * sizeof(float));
    for (size_t channel = 0; channel < QWEN38_DELTA_QKV_ROWS; ++channel) {
        size_t base = channel * 4;
        conv_state_out[base + 3] = projected[channel];
        float sum = conv_state_out[base] * conv_weight[base] +
                    conv_state_out[base + 1] * conv_weight[base + 1] +
                    conv_state_out[base + 2] * conv_weight[base + 2] +
                    conv_state_out[base + 3] * conv_weight[base + 3];
        convolved[channel] = sum / (1.0f + expf(-sum));
    }
    float decay[QWEN38_DELTA_VALUE_HEADS];
    float beta[QWEN38_DELTA_VALUE_HEADS];
    const float *a_log = constants + h->a_log_constants_index;
    const float *dt_bias = constants + h->dt_bias_constants_index;
    for (size_t head = 0; head < QWEN38_DELTA_VALUE_HEADS; ++head) {
        size_t key_head = head / 3;
        double q_squared = 0.0;
        double k_squared = 0.0;
        for (size_t index = 0; index < 128; ++index) {
            float q = convolved[key_head * 128 + index];
            float k = convolved[2048 + key_head * 128 + index];
            q_squared += (double)q * q;
            k_squared += (double)k * k;
        }
        float q_inv = 1.0f /
            sqrtf((float)q_squared + 128.0e-6f) / sqrtf(128.0f);
        float k_inv = 1.0f / sqrtf((float)k_squared + 128.0e-6f);
        for (size_t index = 0; index < 128; ++index) {
            size_t out = head * 128 + index;
            query[out] = convolved[key_head * 128 + index] * q_inv;
            key[out] = convolved[2048 + key_head * 128 + index] * k_inv;
            value[out] = convolved[4096 + out];
        }
        float a = projected[16384 + head] + dt_bias[head];
        float softplus = fmaxf(a, 0.0f) + log1pf(expf(-fabsf(a)));
        decay[head] = expf(-expf(a_log[head]) * softplus);
        float b = projected[16432 + head];
        beta[head] = 1.0f / (1.0f + expf(-b));
    }
    const size_t state_count = delta_vector_count * 128;
    memset(state_out, 0, state_count * sizeof(float));
    for (size_t head = 0; head < QWEN38_DELTA_VALUE_HEADS; ++head) {
        size_t vector_base = head * 128;
        size_t state_head = vector_base * 128;
        for (size_t v_index = 0; v_index < 128; ++v_index) {
            float kv_memory = 0.0f;
            float prior = 0.0f;
            float key_query = 0.0f;
            for (size_t k_index = 0; k_index < 128; ++k_index) {
                size_t state_index = state_head + k_index * 128 + v_index;
                float decayed = state_out[state_index] * decay[head];
                kv_memory += decayed * key[vector_base + k_index];
                prior += decayed * query[vector_base + k_index];
                key_query += key[vector_base + k_index] *
                             query[vector_base + k_index];
            }
            float delta =
                (value[vector_base + v_index] - kv_memory) * beta[head];
            for (size_t k_index = 0; k_index < 128; ++k_index) {
                size_t state_index = state_head + k_index * 128 + v_index;
                state_out[state_index] =
                    state_out[state_index] * decay[head] +
                    key[vector_base + k_index] * delta;
            }
            core[vector_base + v_index] = prior + key_query * delta;
        }
    }
    const float *gated_weight =
        constants + h->recurrent_norm_constants_index;
    for (size_t head = 0; head < QWEN38_DELTA_VALUE_HEADS; ++head) {
        double squared = 0.0;
        for (size_t index = 0; index < 128; ++index) {
            float v = core[head * 128 + index];
            squared += (double)v * v;
        }
        float inv = 1.0f / sqrtf((float)(squared / 128.0) + 1.0e-6f);
        for (size_t index = 0; index < 128; ++index) {
            size_t at = head * 128 + index;
            float z = projected[10240 + at];
            gated[at] = core[at] * inv * gated_weight[index] *
                        (z / (1.0f + expf(-z)));
        }
    }
    const uint8_t *delta_output_quants =
        image + h->delta_output_quants_offset;
    const qwen38_layer_q4_meta *delta_output_meta =
        (const qwen38_layer_q4_meta *)(
            image + h->delta_output_metadata_offset);
    for (size_t row = 0; row < QWEN38_HIDDEN_SIZE; ++row) {
        mixer_output[row] = cpu_q4_dot_float(
            delta_output_quants, delta_output_meta, gated, row, 96) +
            layer_half_value(input[row]);
    }
    cpu_rms_float(mixer_output,
                  constants + h->post_norm_constants_index, post);
    const uint8_t *gate_quants = image + h->gate_quants_offset;
    const qwen38_layer_q4_meta *gate_meta =
        (const qwen38_layer_q4_meta *)(image + h->gate_metadata_offset);
    const uint8_t *up_quants = image + h->up_quants_offset;
    const qwen38_layer_q4_meta *up_meta =
        (const qwen38_layer_q4_meta *)(image + h->up_metadata_offset);
    for (size_t row = 0; row < QWEN38_MLP_SIZE; ++row) {
        float gate = cpu_q4_dot_half(gate_quants, gate_meta, post, row, 80);
        float up = cpu_q4_dot_half(up_quants, up_meta, post, row, 80);
        intermediate[row] = (gate / (1.0f + expf(-gate))) * up;
    }
    const uint8_t *down_quants = image + h->down_quants_offset;
    const qwen38_layer_q4_meta *down_meta =
        (const qwen38_layer_q4_meta *)(image + h->down_metadata_offset);
    for (size_t row = 0; row < 8; ++row) {
        output_first_8[row] = cpu_q4_dot_float(
            down_quants, down_meta, intermediate, row, 272) +
            mixer_output[row];
    }
    free(normalized); free(projected); free(convolved);
    free(query); free(key); free(value); free(core); free(gated);
    free(mixer_output); free(post); free(intermediate);
    return 0;
}

int qwen38_m3_run_layer_benchmark(
    const char *metallib_path,
    const char *image_path,
    unsigned warmup_iterations,
    unsigned measured_iterations,
    qwen38_m3_layer_result *result,
    char *error_message,
    size_t error_message_capacity) {
    if (metallib_path == NULL || image_path == NULL || result == NULL ||
        measured_iterations == 0) {
        layer_set_error(error_message, error_message_capacity,
                        @"invalid layer benchmark arguments");
        return 1;
    }
    memset(result, 0, sizeof(*result));
    @autoreleasepool {
        int image_file = open(image_path, O_RDONLY);
        struct stat image_status;
        if (image_file < 0 || fstat(image_file, &image_status) != 0 ||
            image_status.st_size < QWEN38_M3_IMAGE_HEADER_BYTES) {
            if (image_file >= 0) close(image_file);
            layer_set_error(error_message, error_message_capacity,
                            @"cannot open layer image");
            return 2;
        }
        size_t image_length = (size_t)image_status.st_size;
        unsigned char *image = mmap(NULL, image_length, PROT_READ,
                                    MAP_PRIVATE, image_file, 0);
        if (image == MAP_FAILED) {
            close(image_file);
            layer_set_error(error_message, error_message_capacity,
                            @"cannot map layer image");
            return 2;
        }
        const qwen38_m3_image_header *header =
            (const qwen38_m3_image_header *)image;
        if (validate_layer_image(header, image_length) != 0) {
            munmap(image, image_length);
            close(image_file);
            layer_set_error(error_message, error_message_capacity,
                            @"invalid layer image header or length");
            return 2;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        NSURL *url = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:metallib_path]];
        id<MTLLibrary> library =
            device != nil ? [device newLibraryWithURL:url error:&error] : nil;
        if (device == nil || library == nil) {
            munmap(image, image_length);
            close(image_file);
            layer_set_error(error_message, error_message_capacity,
                            error != nil ? error.localizedDescription :
                                           @"no Metal device");
            return 3;
        }
        qwen38_layer_pipelines p = {
            .rms_half = layer_pipeline(device, library,
                @"qwen38_rmsnorm_f16_to_f16", &error),
            .rms_float = layer_pipeline(device, library,
                @"qwen38_rmsnorm_f32_to_f16", &error),
            .delta_inputs = layer_pipeline(device, library,
                @"qwen38_q4_delta_inputs", &error),
            .conv = layer_pipeline(device, library,
                @"qwen38_delta_causal_conv_silu", &error),
            .prepare = layer_pipeline(device, library,
                @"qwen38_delta_prepare", &error),
            .recurrent = layer_pipeline(device, library,
                @"qwen38_deltanet_direct", &error),
            .gated_norm = layer_pipeline(device, library,
                @"qwen38_delta_gated_rmsnorm", &error),
            .delta_output = layer_pipeline(device, library,
                @"qwen38_q4_delta_output_residual", &error),
            .mlp_gate_up = layer_pipeline(device, library,
                @"qwen38_q4_gate_up_silu", &error),
            .mlp_down = layer_pipeline(device, library,
                @"qwen38_q4_mlp_down_residual_f32", &error)
        };
        if (p.rms_half == nil || p.rms_float == nil ||
            p.delta_inputs == nil || p.conv == nil || p.prepare == nil ||
            p.recurrent == nil || p.gated_norm == nil ||
            p.delta_output == nil || p.mlp_gate_up == nil ||
            p.mlp_down == nil) {
            munmap(image, image_length);
            close(image_file);
            layer_set_error(error_message, error_message_capacity,
                            error.localizedDescription);
            return 4;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (queue == nil) {
            munmap(image, image_length);
            close(image_file);
            layer_set_error(error_message, error_message_capacity,
                            @"cannot create Metal command queue");
            return 4;
        }
        const size_t half_hidden_bytes =
            QWEN38_HIDDEN_SIZE * sizeof(uint16_t);
        const size_t float_hidden_bytes =
            QWEN38_HIDDEN_SIZE * sizeof(float);
        const size_t projected_bytes =
            QWEN38_DELTA_INPUT_ROWS * sizeof(float);
        const size_t conv_state_bytes =
            QWEN38_DELTA_QKV_ROWS * QWEN38_DELTA_CONV_SIZE * sizeof(float);
        const size_t convolved_bytes =
            QWEN38_DELTA_QKV_ROWS * sizeof(float);
        const size_t vector_bytes =
            QWEN38_DELTA_VALUE_HEADS * QWEN38_DELTA_HEAD_SIZE *
            sizeof(float);
        const size_t state_bytes =
            QWEN38_DELTA_VALUE_HEADS * QWEN38_DELTA_HEAD_SIZE *
            QWEN38_DELTA_HEAD_SIZE * sizeof(float);
        const size_t scalar_bytes =
            QWEN38_DELTA_VALUE_HEADS * sizeof(float);
        const size_t intermediate_bytes =
            QWEN38_MLP_SIZE * sizeof(float);
        result->footprint_before_bytes = layer_footprint();
        MTLResourceOptions shared = MTLResourceStorageModeShared;
        MTLResourceOptions mapped = MTLResourceStorageModeShared |
                                    MTLResourceHazardTrackingModeUntracked;
#define Q38_MAPPED_BUFFER(field) \
        [device newBufferWithBytesNoCopy:image + header->field##_offset \
                                  length:header->field##_bytes \
                                 options:mapped deallocator:nil]
        qwen38_layer_buffers b = {
            .input = [device newBufferWithLength:half_hidden_bytes
                                         options:shared],
            .normalized = [device newBufferWithLength:half_hidden_bytes
                                              options:shared],
            .constants = [device newBufferWithLength:
                header->constants_f32_count * sizeof(float) options:shared],
            .delta_input_quants = Q38_MAPPED_BUFFER(delta_input_quants),
            .delta_input_metadata = Q38_MAPPED_BUFFER(delta_input_metadata),
            .projected = [device newBufferWithLength:projected_bytes
                                             options:shared],
            .conv_state = [device newBufferWithLength:conv_state_bytes
                                              options:shared],
            .convolved = [device newBufferWithLength:convolved_bytes
                                              options:shared],
            .query = [device newBufferWithLength:vector_bytes options:shared],
            .key = [device newBufferWithLength:vector_bytes options:shared],
            .value = [device newBufferWithLength:vector_bytes options:shared],
            .decay = [device newBufferWithLength:scalar_bytes options:shared],
            .beta = [device newBufferWithLength:scalar_bytes options:shared],
            .recurrent_state = [device newBufferWithLength:state_bytes
                                                   options:shared],
            .core = [device newBufferWithLength:vector_bytes options:shared],
            .gated = [device newBufferWithLength:vector_bytes options:shared],
            .delta_output_quants = Q38_MAPPED_BUFFER(delta_output_quants),
            .delta_output_metadata = Q38_MAPPED_BUFFER(delta_output_metadata),
            .mixer_output = [device newBufferWithLength:float_hidden_bytes
                                                options:shared],
            .post_normalized = [device newBufferWithLength:half_hidden_bytes
                                                   options:shared],
            .gate_quants = Q38_MAPPED_BUFFER(gate_quants),
            .gate_metadata = Q38_MAPPED_BUFFER(gate_metadata),
            .up_quants = Q38_MAPPED_BUFFER(up_quants),
            .up_metadata = Q38_MAPPED_BUFFER(up_metadata),
            .down_quants = Q38_MAPPED_BUFFER(down_quants),
            .down_metadata = Q38_MAPPED_BUFFER(down_metadata),
            .mlp_intermediate = [device newBufferWithLength:
                intermediate_bytes options:shared],
            .output = [device newBufferWithLength:float_hidden_bytes
                                          options:shared]
        };
#undef Q38_MAPPED_BUFFER
        if (b.input == nil || b.normalized == nil || b.constants == nil ||
            b.delta_input_quants == nil || b.delta_input_metadata == nil ||
            b.projected == nil || b.conv_state == nil || b.convolved == nil ||
            b.query == nil || b.key == nil || b.value == nil ||
            b.decay == nil || b.beta == nil || b.recurrent_state == nil ||
            b.core == nil || b.gated == nil ||
            b.delta_output_quants == nil ||
            b.delta_output_metadata == nil || b.mixer_output == nil ||
            b.post_normalized == nil || b.gate_quants == nil ||
            b.gate_metadata == nil || b.up_quants == nil ||
            b.up_metadata == nil || b.down_quants == nil ||
            b.down_metadata == nil || b.mlp_intermediate == nil ||
            b.output == nil) {
            munmap(image, image_length);
            close(image_file);
            layer_set_error(error_message, error_message_capacity,
                            @"Metal layer allocation failed");
            return 5;
        }
        memcpy(b.constants.contents, image + header->constants_offset,
               header->constants_f32_count * sizeof(float));
        uint16_t *input = b.input.contents;
        for (size_t index = 0; index < QWEN38_HIDDEN_SIZE; ++index) {
            input[index] = layer_half_bits(
                0.75f * sinf((float)index * 0.013f) +
                0.20f * cosf((float)index * 0.031f));
        }
        memset(b.conv_state.contents, 0, conv_state_bytes);
        memset(b.recurrent_state.contents, 0, state_bytes);

        float reference_first_8[8];
        float *reference_state = malloc(state_bytes);
        float *reference_conv = malloc(conv_state_bytes);
        if (reference_state == NULL || reference_conv == NULL ||
            cpu_layer_reference(image, header, input, reference_first_8,
                                reference_state, reference_conv) != 0) {
            free(reference_state);
            free(reference_conv);
            munmap(image, image_length);
            close(image_file);
            layer_set_error(error_message, error_message_capacity,
                            @"CPU layer reference failed");
            return 6;
        }
        if (!submit_layers(queue, &p, &b, header, 1)) {
            free(reference_state);
            free(reference_conv);
            munmap(image, image_length);
            close(image_file);
            layer_set_error(error_message, error_message_capacity,
                            @"Metal layer correctness dispatch failed");
            return 7;
        }
        const float *metal_output = b.output.contents;
        const float *metal_state = b.recurrent_state.contents;
        const float *metal_conv = b.conv_state.contents;
        for (size_t index = 0; index < 8; ++index) {
            result->reference_output_first_8[index] =
                reference_first_8[index];
            result->metal_output_first_8[index] = metal_output[index];
            double error_value = fabs((double)reference_first_8[index] -
                                      metal_output[index]);
            if (error_value > result->max_abs_error_output_first_8) {
                result->max_abs_error_output_first_8 = error_value;
            }
        }
        size_t state_count = state_bytes / sizeof(float);
        for (size_t index = 0; index < state_count; ++index) {
            double error_value = fabs((double)reference_state[index] -
                                      metal_state[index]);
            if (error_value > result->max_abs_error_recurrent_state) {
                result->max_abs_error_recurrent_state = error_value;
            }
        }
        size_t conv_count = conv_state_bytes / sizeof(float);
        for (size_t index = 0; index < conv_count; ++index) {
            double error_value = fabs((double)reference_conv[index] -
                                      metal_conv[index]);
            if (error_value > result->max_abs_error_convolution_state) {
                result->max_abs_error_convolution_state = error_value;
            }
        }
        free(reference_state);
        free(reference_conv);

        memset(b.conv_state.contents, 0, conv_state_bytes);
        memset(b.recurrent_state.contents, 0, state_bytes);
        if (warmup_iterations != 0 &&
            !submit_layers(queue, &p, &b, header, warmup_iterations)) {
            munmap(image, image_length);
            close(image_file);
            layer_set_error(error_message, error_message_capacity,
                            @"Metal layer warmup failed");
            return 8;
        }
        double start = layer_seconds();
        if (!submit_layers(queue, &p, &b, header, measured_iterations)) {
            munmap(image, image_length);
            close(image_file);
            layer_set_error(error_message, error_message_capacity,
                            @"Metal layer measurement failed");
            return 9;
        }
        double elapsed = layer_seconds() - start;

        size_t owned = half_hidden_bytes * 2 +
            header->constants_f32_count * sizeof(float) +
            projected_bytes + conv_state_bytes + convolved_bytes +
            vector_bytes * 5 + scalar_bytes * 2 + state_bytes +
            float_hidden_bytes * 2 + half_hidden_bytes +
            intermediate_bytes;
        uint64_t effective_weight_bytes =
            header->gate_quants_bytes + header->gate_metadata_bytes +
            header->up_quants_bytes + header->up_metadata_bytes +
            header->down_quants_bytes + header->down_metadata_bytes +
            header->constants_f32_count * sizeof(float) +
            header->delta_input_quants_bytes +
            (uint64_t)header->delta_input_rows *
                header->delta_input_groups_per_row * 4 +
            header->delta_output_quants_bytes +
            header->delta_output_metadata_bytes;
        snprintf(result->device_name, sizeof(result->device_name), "%s",
                 device.name.UTF8String);
        snprintf(result->weight_source, sizeof(result->weight_source), "%s",
                 image_path);
        result->warmup_iterations = warmup_iterations;
        result->measured_iterations = measured_iterations;
        result->mapped_image_bytes = image_length;
        result->recurrent_state_bytes = state_bytes;
        result->convolution_state_bytes = conv_state_bytes;
        result->metal_owned_buffer_bytes = owned;
        result->layer_ms = elapsed * 1000.0 / measured_iterations;
        result->effective_weight_gbps =
            (double)effective_weight_bytes /
            (result->layer_ms * 1.0e6);
        result->footprint_peak_bytes = layer_footprint();
        munmap(image, image_length);
        close(image_file);
        return 0;
    }
}
