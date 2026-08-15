#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "qwen38_m3.h"
#include "qwen38_m3_attention_image.h"

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
} qwen38_attention_q4_meta;

typedef struct {
    uint32_t position;
    uint32_t context_length;
    uint32_t cache_capacity;
    uint32_t reserved;
} qwen38_attention_parameters;

typedef struct {
    id<MTLComputePipelineState> rms_half;
    id<MTLComputePipelineState> rms_float;
    id<MTLComputePipelineState> input_projection;
    id<MTLComputePipelineState> prepare_query;
    id<MTLComputePipelineState> prepare_key_value;
    id<MTLComputePipelineState> scores;
    id<MTLComputePipelineState> softmax_value;
    id<MTLComputePipelineState> output_projection;
    id<MTLComputePipelineState> mlp_gate_up;
    id<MTLComputePipelineState> mlp_down;
} qwen38_attention_pipelines;

typedef struct {
    id<MTLBuffer> input;
    id<MTLBuffer> normalized;
    id<MTLBuffer> constants;
    id<MTLBuffer> input_quants;
    id<MTLBuffer> input_metadata;
    id<MTLBuffer> projected;
    id<MTLBuffer> query;
    id<MTLBuffer> query_gate;
    id<MTLBuffer> key_cache;
    id<MTLBuffer> value_cache;
    id<MTLBuffer> scores;
    id<MTLBuffer> attention_output;
    id<MTLBuffer> output_quants;
    id<MTLBuffer> output_metadata;
    id<MTLBuffer> mixer_output;
    id<MTLBuffer> post_normalized;
    id<MTLBuffer> gate_quants;
    id<MTLBuffer> gate_metadata;
    id<MTLBuffer> up_quants;
    id<MTLBuffer> up_metadata;
    id<MTLBuffer> down_quants;
    id<MTLBuffer> down_metadata;
    id<MTLBuffer> intermediate;
    id<MTLBuffer> output;
} qwen38_attention_buffers;

static void attention_error(char *message, size_t capacity, NSString *text) {
    if (message == NULL || capacity == 0) return;
    const char *utf8 = text != nil ? text.UTF8String : "unknown Metal error";
    snprintf(message, capacity, "%s",
             utf8 != NULL ? utf8 : "unknown Metal error");
}

static size_t attention_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t status = task_info(mach_task_self(), TASK_VM_INFO,
                                     (task_info_t)&info, &count);
    return status == KERN_SUCCESS ? (size_t)info.phys_footprint : 0;
}

static double attention_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC_RAW, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static uint16_t attention_half_bits(float value) {
    __fp16 converted = (__fp16)value;
    uint16_t bits;
    memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

static float attention_half_value(uint16_t bits) {
    __fp16 converted;
    memcpy(&converted, &bits, sizeof(converted));
    return (float)converted;
}

static id<MTLComputePipelineState>
attention_pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                   NSString *name, NSError **error) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    return function != nil ?
        [device newComputePipelineStateWithFunction:function error:error] :
        nil;
}

static uint64_t attention_align(uint64_t value) {
    return (value + 4095) & ~(uint64_t)4095;
}

static int attention_pinned_sha(const char *sha) {
    return memcmp(sha, QWEN38_M3_EXPECTED_SOURCE_SHA256, 64) == 0 ||
           memcmp(sha, QWEN38_M3_EXPECTED_SOURCE_SHA256_2, 64) == 0 ||
           memcmp(sha, QWEN38_M3_EXPECTED_SOURCE_SHA256_3, 64) == 0;
}

static int validate_attention_image(
    const qwen38_m3_attention_image_header *h, size_t length) {
    const uint64_t mlp_weight =
        (uint64_t)QWEN38_MLP_SIZE * QWEN38_HIDDEN_SIZE / 2;
    const uint64_t mlp_meta =
        (uint64_t)QWEN38_MLP_SIZE * 80 * 4;
    const uint64_t attention_input_q =
        (uint64_t)QWEN38_ATTENTION_INPUT_ROWS * QWEN38_HIDDEN_SIZE / 2;
    const uint64_t attention_input_m =
        (uint64_t)QWEN38_ATTENTION_INPUT_ROWS * 80 * 4;
    const uint64_t attention_output_q =
        (uint64_t)QWEN38_HIDDEN_SIZE * 6144 / 2;
    const uint64_t attention_output_m =
        (uint64_t)QWEN38_HIDDEN_SIZE * 96 * 4;
    if (memcmp(h->magic, QWEN38_M3_ATTENTION_IMAGE_MAGIC, 8) != 0 ||
        h->version != QWEN38_M3_ATTENTION_IMAGE_VERSION ||
        h->header_bytes != 4096 || h->layer_index >= 64 ||
        h->layer_index % 4 != 3 || h->hidden_size != QWEN38_HIDDEN_SIZE ||
        h->intermediate_size != QWEN38_MLP_SIZE ||
        h->group_size != 64 || h->q_heads != 24 || h->kv_heads != 4 ||
        h->head_size != 256 || h->rotary_size != 64 ||
        h->input_rows != QWEN38_ATTENTION_INPUT_ROWS ||
        h->input_groups_per_row != 80 || h->output_rows != 5120 ||
        h->output_groups_per_row != 96 || h->constants_f32_count != 10752 ||
        !attention_pinned_sha(h->source_sha256)) {
        return -1;
    }
    uint64_t offset = 4096;
#define CHECK(field, bytes) do { \
    if (h->field##_offset != offset || h->field##_bytes != (bytes)) \
        return -1; \
    offset += (bytes); \
} while (0)
    CHECK(gate_quants, mlp_weight);
    CHECK(gate_metadata, mlp_meta);
    CHECK(up_quants, mlp_weight);
    CHECK(up_metadata, mlp_meta);
    CHECK(down_quants, mlp_weight);
    CHECK(down_metadata, mlp_meta);
    offset = attention_align(offset);
    if (h->constants_offset != offset ||
        h->constants_bytes != attention_align(10752 * sizeof(float)) ||
        h->input_norm_constants_index != 0 ||
        h->post_norm_constants_index != 5120 ||
        h->q_norm_constants_index != 10240 ||
        h->k_norm_constants_index != 10496) return -1;
    offset += h->constants_bytes;
    CHECK(attention_input_quants, attention_input_q);
    CHECK(attention_input_metadata, attention_input_m);
    CHECK(attention_output_quants, attention_output_q);
    CHECK(attention_output_metadata, attention_output_m);
#undef CHECK
    return offset == length ? 0 : -1;
}

static void encode_attention_layer(
    id<MTLComputeCommandEncoder> encoder,
    const qwen38_attention_pipelines *p,
    const qwen38_attention_buffers *b,
    const qwen38_m3_attention_image_header *h) {
    qwen38_attention_parameters parameters = {
        .position = 0, .context_length = 1, .cache_capacity = 1, .reserved = 0
    };
    [encoder setComputePipelineState:p->rms_half];
    [encoder setBuffer:b->input offset:0 atIndex:0];
    [encoder setBuffer:b->constants
                 offset:h->input_norm_constants_index * 4 atIndex:1];
    [encoder setBuffer:b->normalized offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [encoder setComputePipelineState:p->input_projection];
    [encoder setBuffer:b->normalized offset:0 atIndex:0];
    [encoder setBuffer:b->input_quants offset:0 atIndex:1];
    [encoder setBuffer:b->input_metadata offset:0 atIndex:2];
    [encoder setBuffer:b->projected offset:0 atIndex:3];
    [encoder dispatchThreadgroups:
        MTLSizeMake((QWEN38_ATTENTION_INPUT_ROWS + 7) / 8, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [encoder setComputePipelineState:p->prepare_query];
    [encoder setBuffer:b->projected offset:0 atIndex:0];
    [encoder setBuffer:b->constants
                 offset:h->q_norm_constants_index * 4 atIndex:1];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
    [encoder setBuffer:b->query offset:0 atIndex:3];
    [encoder setBuffer:b->query_gate offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(24, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [encoder setComputePipelineState:p->prepare_key_value];
    [encoder setBuffer:b->projected offset:0 atIndex:0];
    [encoder setBuffer:b->constants
                 offset:h->k_norm_constants_index * 4 atIndex:1];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
    [encoder setBuffer:b->key_cache offset:0 atIndex:3];
    [encoder setBuffer:b->value_cache offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(4, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [encoder setComputePipelineState:p->scores];
    [encoder setBuffer:b->query offset:0 atIndex:0];
    [encoder setBuffer:b->key_cache offset:0 atIndex:1];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
    [encoder setBuffer:b->scores offset:0 atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(3, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [encoder setComputePipelineState:p->softmax_value];
    [encoder setBuffer:b->scores offset:0 atIndex:0];
    [encoder setBuffer:b->value_cache offset:0 atIndex:1];
    [encoder setBuffer:b->query_gate offset:0 atIndex:2];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder setBuffer:b->attention_output offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(24, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [encoder setComputePipelineState:p->output_projection];
    [encoder setBuffer:b->attention_output offset:0 atIndex:0];
    [encoder setBuffer:b->output_quants offset:0 atIndex:1];
    [encoder setBuffer:b->output_metadata offset:0 atIndex:2];
    [encoder setBuffer:b->input offset:0 atIndex:3];
    [encoder setBuffer:b->mixer_output offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake((5120 + 7) / 8, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [encoder setComputePipelineState:p->rms_float];
    [encoder setBuffer:b->mixer_output offset:0 atIndex:0];
    [encoder setBuffer:b->constants
                 offset:h->post_norm_constants_index * 4 atIndex:1];
    [encoder setBuffer:b->post_normalized offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [encoder setComputePipelineState:p->mlp_gate_up];
    [encoder setBuffer:b->post_normalized offset:0 atIndex:0];
    [encoder setBuffer:b->gate_quants offset:0 atIndex:1];
    [encoder setBuffer:b->gate_metadata offset:0 atIndex:2];
    [encoder setBuffer:b->up_quants offset:0 atIndex:3];
    [encoder setBuffer:b->up_metadata offset:0 atIndex:4];
    [encoder setBuffer:b->intermediate offset:0 atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake((17408 + 7) / 8, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

    [encoder setComputePipelineState:p->mlp_down];
    [encoder setBuffer:b->intermediate offset:0 atIndex:0];
    [encoder setBuffer:b->down_quants offset:0 atIndex:1];
    [encoder setBuffer:b->down_metadata offset:0 atIndex:2];
    [encoder setBuffer:b->mixer_output offset:0 atIndex:3];
    [encoder setBuffer:b->output offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake((5120 + 7) / 8, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static BOOL submit_attention(
    id<MTLCommandQueue> queue, const qwen38_attention_pipelines *p,
    const qwen38_attention_buffers *b,
    const qwen38_m3_attention_image_header *h, unsigned iterations) {
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
        encode_attention_layer(encoder, p, b, h);
    }
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    return command.status == MTLCommandBufferStatusCompleted;
}

static float attention_q4_half(const uint8_t *q,
                               const qwen38_attention_q4_meta *m,
                               const uint16_t *input, size_t row,
                               size_t groups) {
    float total = 0.0f;
    for (size_t group = 0; group < groups; ++group) {
        size_t block = row * groups + group;
        float scale = attention_half_value(m[block].scale);
        float bias = attention_half_value(m[block].bias);
        for (size_t lane = 0; lane < 32; ++lane) {
            uint8_t bits = q[block * 32 + lane];
            size_t index = group * 64 + lane * 2;
            total += (scale * (float)(bits & 15u) + bias) *
                     attention_half_value(input[index]);
            total += (scale * (float)(bits >> 4) + bias) *
                     attention_half_value(input[index + 1]);
        }
    }
    return total;
}

static float attention_q4_float(const uint8_t *q,
                                const qwen38_attention_q4_meta *m,
                                const float *input, size_t row,
                                size_t groups) {
    float total = 0.0f;
    for (size_t group = 0; group < groups; ++group) {
        size_t block = row * groups + group;
        float scale = attention_half_value(m[block].scale);
        float bias = attention_half_value(m[block].bias);
        for (size_t lane = 0; lane < 32; ++lane) {
            uint8_t bits = q[block * 32 + lane];
            size_t index = group * 64 + lane * 2;
            total += (scale * (float)(bits & 15u) + bias) * input[index];
            total += (scale * (float)(bits >> 4) + bias) * input[index + 1];
        }
    }
    return total;
}

static void attention_rms_half(const uint16_t *input, const float *weight,
                               uint16_t *output, size_t count) {
    double squared = 0.0;
    for (size_t i = 0; i < count; ++i) {
        float value = attention_half_value(input[i]);
        squared += (double)value * value;
    }
    float inv = 1.0f / sqrtf((float)(squared / count) + 1.0e-6f);
    for (size_t i = 0; i < count; ++i) {
        output[i] = attention_half_bits(
            attention_half_value(input[i]) * inv * weight[i]);
    }
}

static void attention_rms_float(const float *input, const float *weight,
                                uint16_t *output, size_t count) {
    double squared = 0.0;
    for (size_t i = 0; i < count; ++i) {
        squared += (double)input[i] * input[i];
    }
    float inv = 1.0f / sqrtf((float)(squared / count) + 1.0e-6f);
    for (size_t i = 0; i < count; ++i) {
        output[i] = attention_half_bits(input[i] * inv * weight[i]);
    }
}

static int attention_cpu_reference(
    const unsigned char *image,
    const qwen38_m3_attention_image_header *h,
    const uint16_t *input, float output_first_8[8],
    float *query, float *key, float *value) {
    const float *constants =
        (const float *)(image + h->constants_offset);
    uint16_t *normalized = malloc(5120 * sizeof(uint16_t));
    float *projected = malloc(14336 * sizeof(float));
    float *attention_output = malloc(6144 * sizeof(float));
    float *mixer_output = malloc(5120 * sizeof(float));
    uint16_t *post = malloc(5120 * sizeof(uint16_t));
    float *intermediate = malloc(17408 * sizeof(float));
    if (normalized == NULL || projected == NULL ||
        attention_output == NULL || mixer_output == NULL || post == NULL ||
        intermediate == NULL) {
        free(normalized); free(projected); free(attention_output);
        free(mixer_output); free(post); free(intermediate);
        return -1;
    }
    attention_rms_half(input, constants + h->input_norm_constants_index,
                       normalized, 5120);
    const uint8_t *input_q = image + h->attention_input_quants_offset;
    const qwen38_attention_q4_meta *input_m =
        (const qwen38_attention_q4_meta *)(
            image + h->attention_input_metadata_offset);
    for (size_t row = 0; row < 14336; ++row) {
        projected[row] =
            attention_q4_half(input_q, input_m, normalized, row, 80);
    }
    const float *q_weight = constants + h->q_norm_constants_index;
    const float *k_weight = constants + h->k_norm_constants_index;
    for (size_t head = 0; head < 24; ++head) {
        double squared = 0.0;
        for (size_t d = 0; d < 256; ++d) {
            float x = projected[head * 512 + d];
            squared += (double)x * x;
        }
        float inv = 1.0f / sqrtf((float)(squared / 256.0) + 1.0e-6f);
        size_t kv_head = head / 6;
        for (size_t d = 0; d < 256; ++d) {
            size_t at = head * 256 + d;
            query[at] = projected[head * 512 + d] * inv * q_weight[d];
            float gate = projected[head * 512 + 256 + d];
            float gate_value = 1.0f / (1.0f + expf(-gate));
            attention_output[at] =
                projected[13312 + kv_head * 256 + d] * gate_value;
        }
    }
    for (size_t head = 0; head < 4; ++head) {
        double squared = 0.0;
        for (size_t d = 0; d < 256; ++d) {
            float x = projected[12288 + head * 256 + d];
            squared += (double)x * x;
        }
        float inv = 1.0f / sqrtf((float)(squared / 256.0) + 1.0e-6f);
        for (size_t d = 0; d < 256; ++d) {
            size_t at = head * 256 + d;
            key[at] = projected[12288 + at] * inv * k_weight[d];
            value[at] = projected[13312 + at];
        }
    }
    const uint8_t *output_q = image + h->attention_output_quants_offset;
    const qwen38_attention_q4_meta *output_m =
        (const qwen38_attention_q4_meta *)(
            image + h->attention_output_metadata_offset);
    for (size_t row = 0; row < 5120; ++row) {
        mixer_output[row] =
            attention_q4_float(output_q, output_m,
                               attention_output, row, 96) +
            attention_half_value(input[row]);
    }
    attention_rms_float(mixer_output,
                        constants + h->post_norm_constants_index,
                        post, 5120);
    const uint8_t *gate_q = image + h->gate_quants_offset;
    const qwen38_attention_q4_meta *gate_m =
        (const qwen38_attention_q4_meta *)(image + h->gate_metadata_offset);
    const uint8_t *up_q = image + h->up_quants_offset;
    const qwen38_attention_q4_meta *up_m =
        (const qwen38_attention_q4_meta *)(image + h->up_metadata_offset);
    for (size_t row = 0; row < 17408; ++row) {
        float gate = attention_q4_half(gate_q, gate_m, post, row, 80);
        float up = attention_q4_half(up_q, up_m, post, row, 80);
        intermediate[row] = gate / (1.0f + expf(-gate)) * up;
    }
    const uint8_t *down_q = image + h->down_quants_offset;
    const qwen38_attention_q4_meta *down_m =
        (const qwen38_attention_q4_meta *)(image + h->down_metadata_offset);
    for (size_t row = 0; row < 8; ++row) {
        output_first_8[row] =
            attention_q4_float(down_q, down_m, intermediate, row, 272) +
            mixer_output[row];
    }
    free(normalized); free(projected); free(attention_output);
    free(mixer_output); free(post); free(intermediate);
    return 0;
}

int qwen38_m3_run_attention_layer_benchmark(
    const char *metallib_path, const char *image_path,
    unsigned warmup_iterations, unsigned measured_iterations,
    qwen38_m3_attention_result *result,
    char *error_message, size_t error_message_capacity) {
    if (metallib_path == NULL || image_path == NULL || result == NULL ||
        measured_iterations == 0) {
        attention_error(error_message, error_message_capacity,
                        @"invalid attention benchmark arguments");
        return 1;
    }
    memset(result, 0, sizeof(*result));
    @autoreleasepool {
        int file = open(image_path, O_RDONLY);
        struct stat status;
        if (file < 0 || fstat(file, &status) != 0 ||
            status.st_size < 4096) {
            if (file >= 0) close(file);
            attention_error(error_message, error_message_capacity,
                            @"cannot open attention image");
            return 2;
        }
        size_t image_length = (size_t)status.st_size;
        unsigned char *image = mmap(NULL, image_length, PROT_READ,
                                    MAP_PRIVATE, file, 0);
        const qwen38_m3_attention_image_header *h =
            (const qwen38_m3_attention_image_header *)image;
        if (image == MAP_FAILED ||
            validate_attention_image(h, image_length) != 0) {
            if (image != MAP_FAILED) munmap(image, image_length);
            close(file);
            attention_error(error_message, error_message_capacity,
                            @"invalid attention image");
            return 2;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        NSURL *url = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:metallib_path]];
        id<MTLLibrary> library =
            device != nil ? [device newLibraryWithURL:url error:&error] : nil;
        if (library == nil) {
            munmap(image, image_length); close(file);
            attention_error(error_message, error_message_capacity,
                            error.localizedDescription);
            return 3;
        }
#define PIPE(field, name) \
        .field = attention_pipeline(device, library, name, &error)
        qwen38_attention_pipelines p = {
            PIPE(rms_half, @"qwen38_rmsnorm_f16_to_f16"),
            PIPE(rms_float, @"qwen38_rmsnorm_f32_to_f16"),
            PIPE(input_projection, @"qwen38_q4_attention_inputs"),
            PIPE(prepare_query, @"qwen38_attention_prepare_query"),
            PIPE(prepare_key_value, @"qwen38_attention_prepare_key_value"),
            PIPE(scores, @"qwen38_attention_scores"),
            PIPE(softmax_value, @"qwen38_attention_softmax_value_gate"),
            PIPE(output_projection,
                 @"qwen38_q4_attention_output_residual"),
            PIPE(mlp_gate_up, @"qwen38_q4_gate_up_silu"),
            PIPE(mlp_down, @"qwen38_q4_mlp_down_residual_f32")
        };
#undef PIPE
        if (p.rms_half == nil || p.rms_float == nil ||
            p.input_projection == nil || p.prepare_query == nil ||
            p.prepare_key_value == nil || p.scores == nil ||
            p.softmax_value == nil || p.output_projection == nil ||
            p.mlp_gate_up == nil || p.mlp_down == nil) {
            munmap(image, image_length); close(file);
            attention_error(error_message, error_message_capacity,
                            error.localizedDescription);
            return 4;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        MTLResourceOptions shared = MTLResourceStorageModeShared;
        MTLResourceOptions mapped = shared |
            MTLResourceHazardTrackingModeUntracked;
#define MAP_BUFFER(field) \
        [device newBufferWithBytesNoCopy:image + h->field##_offset \
                                  length:h->field##_bytes \
                                 options:mapped deallocator:nil]
        qwen38_attention_buffers b = {
            .input = [device newBufferWithLength:5120 * 2 options:shared],
            .normalized =
                [device newBufferWithLength:5120 * 2 options:shared],
            .constants = [device newBufferWithLength:10752 * 4
                                              options:shared],
            .input_quants = MAP_BUFFER(attention_input_quants),
            .input_metadata = MAP_BUFFER(attention_input_metadata),
            .projected = [device newBufferWithLength:14336 * 4
                                              options:shared],
            .query = [device newBufferWithLength:6144 * 4 options:shared],
            .query_gate =
                [device newBufferWithLength:6144 * 4 options:shared],
            .key_cache = [device newBufferWithLength:1024 * 4 options:shared],
            .value_cache =
                [device newBufferWithLength:1024 * 4 options:shared],
            .scores = [device newBufferWithLength:24 * 4 options:shared],
            .attention_output =
                [device newBufferWithLength:6144 * 4 options:shared],
            .output_quants = MAP_BUFFER(attention_output_quants),
            .output_metadata = MAP_BUFFER(attention_output_metadata),
            .mixer_output =
                [device newBufferWithLength:5120 * 4 options:shared],
            .post_normalized =
                [device newBufferWithLength:5120 * 2 options:shared],
            .gate_quants = MAP_BUFFER(gate_quants),
            .gate_metadata = MAP_BUFFER(gate_metadata),
            .up_quants = MAP_BUFFER(up_quants),
            .up_metadata = MAP_BUFFER(up_metadata),
            .down_quants = MAP_BUFFER(down_quants),
            .down_metadata = MAP_BUFFER(down_metadata),
            .intermediate =
                [device newBufferWithLength:17408 * 4 options:shared],
            .output = [device newBufferWithLength:5120 * 4 options:shared]
        };
#undef MAP_BUFFER
        if (queue == nil || b.input == nil || b.normalized == nil ||
            b.constants == nil || b.input_quants == nil ||
            b.input_metadata == nil || b.projected == nil ||
            b.query == nil || b.query_gate == nil || b.key_cache == nil ||
            b.value_cache == nil || b.scores == nil ||
            b.attention_output == nil || b.output_quants == nil ||
            b.output_metadata == nil || b.mixer_output == nil ||
            b.post_normalized == nil || b.gate_quants == nil ||
            b.gate_metadata == nil || b.up_quants == nil ||
            b.up_metadata == nil || b.down_quants == nil ||
            b.down_metadata == nil || b.intermediate == nil ||
            b.output == nil) {
            munmap(image, image_length); close(file);
            attention_error(error_message, error_message_capacity,
                            @"attention Metal allocation failed");
            return 5;
        }
        result->footprint_before_bytes = attention_footprint();
        memcpy(b.constants.contents, image + h->constants_offset, 10752 * 4);
        uint16_t *input = b.input.contents;
        for (size_t index = 0; index < 5120; ++index) {
            input[index] = attention_half_bits(
                0.75f * sinf((float)index * 0.013f) +
                0.20f * cosf((float)index * 0.031f));
        }
        float reference_first_8[8];
        float *reference_q = malloc(6144 * 4);
        float *reference_k = malloc(1024 * 4);
        float *reference_v = malloc(1024 * 4);
        if (reference_q == NULL || reference_k == NULL ||
            reference_v == NULL ||
            attention_cpu_reference(image, h, input, reference_first_8,
                                    reference_q, reference_k,
                                    reference_v) != 0) {
            free(reference_q); free(reference_k); free(reference_v);
            munmap(image, image_length); close(file);
            attention_error(error_message, error_message_capacity,
                            @"attention CPU reference failed");
            return 6;
        }
        if (!submit_attention(queue, &p, &b, h, 1)) {
            free(reference_q); free(reference_k); free(reference_v);
            munmap(image, image_length); close(file);
            attention_error(error_message, error_message_capacity,
                            @"attention correctness dispatch failed");
            return 7;
        }
        const float *metal_output = b.output.contents;
        const float *metal_q = b.query.contents;
        const float *metal_k = b.key_cache.contents;
        const float *metal_v = b.value_cache.contents;
        for (size_t index = 0; index < 8; ++index) {
            result->reference_output_first_8[index] =
                reference_first_8[index];
            result->metal_output_first_8[index] = metal_output[index];
            double e = fabs((double)reference_first_8[index] -
                            metal_output[index]);
            if (e > result->max_abs_error_output_first_8)
                result->max_abs_error_output_first_8 = e;
        }
        for (size_t index = 0; index < 6144; ++index) {
            double e = fabs((double)reference_q[index] - metal_q[index]);
            if (e > result->max_abs_error_query)
                result->max_abs_error_query = e;
        }
        for (size_t index = 0; index < 1024; ++index) {
            double ek = fabs((double)reference_k[index] - metal_k[index]);
            double ev = fabs((double)reference_v[index] - metal_v[index]);
            if (ek > result->max_abs_error_key_cache)
                result->max_abs_error_key_cache = ek;
            if (ev > result->max_abs_error_value_cache)
                result->max_abs_error_value_cache = ev;
        }
        free(reference_q); free(reference_k); free(reference_v);
        if (warmup_iterations != 0 &&
            !submit_attention(queue, &p, &b, h, warmup_iterations)) {
            munmap(image, image_length); close(file);
            attention_error(error_message, error_message_capacity,
                            @"attention warmup failed");
            return 8;
        }
        double start = attention_seconds();
        if (!submit_attention(queue, &p, &b, h, measured_iterations)) {
            munmap(image, image_length); close(file);
            attention_error(error_message, error_message_capacity,
                            @"attention measurement failed");
            return 9;
        }
        double ms = (attention_seconds() - start) * 1000.0 /
                    measured_iterations;
        uint64_t weights =
            h->gate_quants_bytes + h->gate_metadata_bytes +
            h->up_quants_bytes + h->up_metadata_bytes +
            h->down_quants_bytes + h->down_metadata_bytes +
            h->constants_f32_count * 4 +
            h->attention_input_quants_bytes +
            h->attention_input_metadata_bytes +
            h->attention_output_quants_bytes +
            h->attention_output_metadata_bytes;
        snprintf(result->device_name, sizeof(result->device_name), "%s",
                 device.name.UTF8String);
        snprintf(result->weight_source, sizeof(result->weight_source), "%s",
                 image_path);
        result->warmup_iterations = warmup_iterations;
        result->measured_iterations = measured_iterations;
        result->mapped_image_bytes = image_length;
        result->kv_cache_bytes_at_context_1 = 2 * 1024 * 4;
        result->metal_owned_buffer_bytes =
            5120 * 2 * 3 + 10752 * 4 + 14336 * 4 +
            6144 * 4 * 3 + 1024 * 4 * 2 + 24 * 4 +
            5120 * 4 * 2 + 17408 * 4;
        result->layer_ms_at_context_1 = ms;
        result->effective_weight_gbps_at_context_1 =
            (double)weights / (ms * 1.0e6);
        result->footprint_peak_bytes = attention_footprint();
        munmap(image, image_length); close(file);
        return 0;
    }
}
