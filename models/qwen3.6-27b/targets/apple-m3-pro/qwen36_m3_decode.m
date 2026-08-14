#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "qwen36_m3_decode.h"
#include "qwen36_m3.h"
#include "qwen36_m3_attention_image.h"
#include "qwen36_m3_global_image.h"
#include "qwen36_m3_image.h"

#include <fcntl.h>
#include <mach/mach.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    uint32_t position;
    uint32_t context_length;
    uint32_t cache_capacity;
    uint32_t reserved;
} q36_decode_attention_parameters;

@interface Q36DecodeLayer : NSObject {
@public
    int file;
    void *mapping;
    size_t length;
    BOOL attention;
    const qwen36_m3_image_header *delta_header;
    const qwen36_m3_attention_image_header *attention_header;
    id<MTLBuffer> constants;
    id<MTLBuffer> gate_quants;
    id<MTLBuffer> gate_metadata;
    id<MTLBuffer> up_quants;
    id<MTLBuffer> up_metadata;
    id<MTLBuffer> down_quants;
    id<MTLBuffer> down_metadata;
    id<MTLBuffer> input_quants;
    id<MTLBuffer> input_metadata;
    id<MTLBuffer> output_quants;
    id<MTLBuffer> output_metadata;
    id<MTLBuffer> recurrent_state;
    id<MTLBuffer> convolution_state;
    id<MTLBuffer> key_cache;
    id<MTLBuffer> value_cache;
}
@end

@implementation Q36DecodeLayer
- (instancetype)init {
    self = [super init];
    if (self != nil) {
        file = -1;
        mapping = MAP_FAILED;
    }
    return self;
}
- (void)dealloc {
    if (mapping != MAP_FAILED) munmap(mapping, length);
    if (file >= 0) close(file);
}
@end

@interface Q36DecodeGlobal : NSObject {
@public
    int file;
    void *mapping;
    size_t length;
    const qwen36_m3_global_image_header *header;
    id<MTLBuffer> embedding_quants;
    id<MTLBuffer> embedding_metadata;
    id<MTLBuffer> lm_head_quants;
    id<MTLBuffer> lm_head_metadata;
    id<MTLBuffer> constants;
}
@end

@implementation Q36DecodeGlobal
- (instancetype)init {
    self = [super init];
    if (self != nil) {
        file = -1;
        mapping = MAP_FAILED;
    }
    return self;
}
- (void)dealloc {
    if (mapping != MAP_FAILED) munmap(mapping, length);
    if (file >= 0) close(file);
}
@end

@interface Q36DecodeRuntime : NSObject {
@public
    id<MTLDevice> device;
    id<MTLLibrary> library;
    id<MTLCommandQueue> queue;
    NSMutableArray<Q36DecodeLayer *> *layers;
    Q36DecodeGlobal *global;
    uint32_t capacity;
    size_t mapped_bytes;
    size_t state_bytes;
    size_t kv_bytes;

    id<MTLComputePipelineState> rms_half;
    id<MTLComputePipelineState> rms_float;
    id<MTLComputePipelineState> delta_inputs;
    id<MTLComputePipelineState> delta_conv;
    id<MTLComputePipelineState> delta_prepare;
    id<MTLComputePipelineState> delta_recurrent;
    id<MTLComputePipelineState> delta_gated_norm;
    id<MTLComputePipelineState> delta_output;
    id<MTLComputePipelineState> attention_inputs;
    id<MTLComputePipelineState> attention_query;
    id<MTLComputePipelineState> attention_key_value;
    id<MTLComputePipelineState> attention_scores;
    id<MTLComputePipelineState> attention_softmax_value;
    id<MTLComputePipelineState> attention_output;
    id<MTLComputePipelineState> mlp_gate_up;
    id<MTLComputePipelineState> mlp_down;
    id<MTLComputePipelineState> embedding;
    id<MTLComputePipelineState> convert_hidden;
    id<MTLComputePipelineState> lm_head;

    id<MTLBuffer> hidden_half;
    id<MTLBuffer> normalized;
    id<MTLBuffer> projected;
    id<MTLBuffer> convolved;
    id<MTLBuffer> query;
    id<MTLBuffer> query_gate;
    id<MTLBuffer> key;
    id<MTLBuffer> value;
    id<MTLBuffer> decay;
    id<MTLBuffer> beta;
    id<MTLBuffer> core;
    id<MTLBuffer> gated;
    id<MTLBuffer> attention_scores_buffer;
    id<MTLBuffer> mixer_output;
    id<MTLBuffer> post_normalized;
    id<MTLBuffer> intermediate;
    id<MTLBuffer> layer_output;
    id<MTLBuffer> final_normalized;
    id<MTLBuffer> logits;
}
@end

@implementation Q36DecodeRuntime
@end

struct qwen36_m3_model {
    void *runtime;
};

static void decode_error(char *message, size_t capacity_value,
                         NSString *text) {
    if (message == NULL || capacity_value == 0) return;
    const char *utf8 = text != nil ? text.UTF8String : "unknown error";
    snprintf(message, capacity_value, "%s",
             utf8 != NULL ? utf8 : "unknown error");
}

static size_t decode_footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t status = task_info(mach_task_self(), TASK_VM_INFO,
                                     (task_info_t)&info, &count);
    return status == KERN_SUCCESS ? (size_t)info.phys_footprint : 0;
}

static double decode_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC_RAW, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static int pinned_sha(const char *sha) {
    return memcmp(sha, QWEN36_M3_EXPECTED_SOURCE_SHA256, 64) == 0 ||
           memcmp(sha, QWEN36_M3_EXPECTED_SOURCE_SHA256_2, 64) == 0 ||
           memcmp(sha, QWEN36_M3_EXPECTED_SOURCE_SHA256_3, 64) == 0;
}

static id<MTLComputePipelineState>
decode_pipeline(Q36DecodeRuntime *runtime, NSString *name, NSError **error) {
    id<MTLFunction> function = [runtime->library newFunctionWithName:name];
    return function != nil ?
        [runtime->device newComputePipelineStateWithFunction:function
                                                        error:error] : nil;
}

static int map_file(const char *path, int *file, void **mapping,
                    size_t *length, NSString **error) {
    *file = open(path, O_RDONLY);
    struct stat status;
    if (*file < 0 || fstat(*file, &status) != 0 ||
        status.st_size < 4096) {
        if (*file >= 0) close(*file);
        *file = -1;
        if (error != NULL)
            *error = [NSString stringWithFormat:@"cannot open %s", path];
        return -1;
    }
    *length = (size_t)status.st_size;
    *mapping = mmap(NULL, *length, PROT_READ, MAP_PRIVATE, *file, 0);
    if (*mapping == MAP_FAILED) {
        close(*file);
        *file = -1;
        if (error != NULL)
            *error = [NSString stringWithFormat:@"cannot map %s", path];
        return -1;
    }
    return 0;
}

static id<MTLBuffer> mapped_buffer(id<MTLDevice> device, void *mapping,
                                   uint64_t offset, uint64_t bytes) {
    if ((offset & 4095u) != 0 || (bytes & 4095u) != 0) return nil;
    return [device newBufferWithBytesNoCopy:
                (unsigned char *)mapping + offset
                                      length:(NSUInteger)bytes
                                     options:MTLResourceStorageModeShared |
            MTLResourceHazardTrackingModeUntracked
                                 deallocator:nil];
}

static int validate_delta_header(const qwen36_m3_image_header *h,
                                 size_t length, unsigned layer) {
    if (memcmp(h->magic, QWEN36_M3_IMAGE_MAGIC, 8) != 0 ||
        h->version != QWEN36_M3_IMAGE_VERSION ||
        h->header_bytes != 4096 || h->hidden_size != 5120 ||
        h->rows != 17408 || h->group_size != 64 ||
        h->layer_index != layer || layer % 4 == 3 ||
        h->delta_input_rows != 16480 ||
        h->delta_input_groups_per_row != 80 ||
        h->delta_output_rows != 5120 ||
        h->delta_output_groups_per_row != 96 ||
        h->constants_f32_count != 51424 ||
        !pinned_sha(h->source_sha256) ||
        (h->mlp_source_sha256[0] != '\0' &&
         !pinned_sha(h->mlp_source_sha256))) return -1;
    uint64_t end = h->delta_output_metadata_offset +
                   h->delta_output_metadata_bytes;
    return end == length ? 0 : -1;
}

static int validate_attention_header(
    const qwen36_m3_attention_image_header *h,
    size_t length, unsigned layer) {
    if (memcmp(h->magic, QWEN36_M3_ATTENTION_IMAGE_MAGIC, 8) != 0 ||
        h->version != 1 || h->header_bytes != 4096 ||
        h->layer_index != layer || layer % 4 != 3 ||
        h->hidden_size != 5120 || h->intermediate_size != 17408 ||
        h->group_size != 64 || h->q_heads != 24 || h->kv_heads != 4 ||
        h->head_size != 256 || h->rotary_size != 64 ||
        h->input_rows != 14336 || h->input_groups_per_row != 80 ||
        h->output_rows != 5120 || h->output_groups_per_row != 96 ||
        h->constants_f32_count != 10752 ||
        !pinned_sha(h->source_sha256)) return -1;
    return h->attention_output_metadata_offset +
           h->attention_output_metadata_bytes == length ? 0 : -1;
}

static int validate_global_header(const qwen36_m3_global_image_header *h,
                                  size_t length) {
    if (memcmp(h->magic, QWEN36_M3_GLOBAL_IMAGE_MAGIC, 8) != 0 ||
        h->version != 1 || h->header_bytes != 4096 ||
        h->vocab_size != QWEN36_VOCAB_SIZE ||
        h->hidden_size != 5120 || h->group_size != 64 ||
        h->constants_f32_count != 5120 ||
        memcmp(h->embedding_source_sha256,
               QWEN36_M3_EXPECTED_SOURCE_SHA256, 64) != 0 ||
        memcmp(h->lm_head_source_sha256,
               QWEN36_M3_EXPECTED_SOURCE_SHA256_3, 64) != 0)
        return -1;
    return h->constants_offset + h->constants_bytes == length ? 0 : -1;
}

static Q36DecodeGlobal *load_global(Q36DecodeRuntime *runtime,
                                    NSString *path, NSString **error) {
    Q36DecodeGlobal *item = [Q36DecodeGlobal new];
    if (map_file(path.fileSystemRepresentation, &item->file,
                 &item->mapping, &item->length, error) != 0) return nil;
    item->header = item->mapping;
    if (validate_global_header(item->header, item->length) != 0) {
        if (error != NULL) *error = @"invalid global image";
        return nil;
    }
    const qwen36_m3_global_image_header *h = item->header;
    item->embedding_quants = mapped_buffer(
        runtime->device, item->mapping, h->embedding_quants_offset,
        h->embedding_quants_bytes);
    item->embedding_metadata = mapped_buffer(
        runtime->device, item->mapping, h->embedding_metadata_offset,
        h->embedding_metadata_bytes);
    item->lm_head_quants = mapped_buffer(
        runtime->device, item->mapping, h->lm_head_quants_offset,
        h->lm_head_quants_bytes);
    item->lm_head_metadata = mapped_buffer(
        runtime->device, item->mapping, h->lm_head_metadata_offset,
        h->lm_head_metadata_bytes);
    item->constants = [runtime->device
        newBufferWithLength:h->constants_f32_count * sizeof(float)
                   options:MTLResourceStorageModeShared];
    if (item->embedding_quants == nil || item->embedding_metadata == nil ||
        item->lm_head_quants == nil || item->lm_head_metadata == nil ||
        item->constants == nil) {
        if (error != NULL) *error = @"cannot create global Metal buffers";
        return nil;
    }
    memcpy(item->constants.contents,
           (unsigned char *)item->mapping + h->constants_offset,
           h->constants_f32_count * sizeof(float));
    return item;
}

static Q36DecodeLayer *load_delta_layer(Q36DecodeRuntime *runtime,
                                        NSString *path, unsigned index,
                                        NSString **error) {
    Q36DecodeLayer *item = [Q36DecodeLayer new];
    if (map_file(path.fileSystemRepresentation, &item->file,
                 &item->mapping, &item->length, error) != 0) return nil;
    item->delta_header = item->mapping;
    if (validate_delta_header(item->delta_header, item->length, index) != 0) {
        if (error != NULL)
            *error = [NSString stringWithFormat:
                @"invalid DeltaNet image at layer %u", index];
        return nil;
    }
    const qwen36_m3_image_header *h = item->delta_header;
#define DELTA_MAP(field) \
    mapped_buffer(runtime->device, item->mapping, h->field##_offset, \
                  h->field##_bytes)
    item->gate_quants = DELTA_MAP(gate_quants);
    item->gate_metadata = DELTA_MAP(gate_metadata);
    item->up_quants = DELTA_MAP(up_quants);
    item->up_metadata = DELTA_MAP(up_metadata);
    item->down_quants = DELTA_MAP(down_quants);
    item->down_metadata = DELTA_MAP(down_metadata);
    item->input_quants = DELTA_MAP(delta_input_quants);
    item->input_metadata = DELTA_MAP(delta_input_metadata);
    item->output_quants = DELTA_MAP(delta_output_quants);
    item->output_metadata = DELTA_MAP(delta_output_metadata);
#undef DELTA_MAP
    item->constants = [runtime->device
        newBufferWithLength:h->constants_f32_count * sizeof(float)
                   options:MTLResourceStorageModeShared];
    item->recurrent_state = [runtime->device
        newBufferWithLength:(size_t)48 * 128 * 128 * sizeof(float)
                   options:MTLResourceStorageModeShared];
    item->convolution_state = [runtime->device
        newBufferWithLength:(size_t)10240 * 4 * sizeof(float)
                   options:MTLResourceStorageModeShared];
    if (item->gate_quants == nil || item->gate_metadata == nil ||
        item->up_quants == nil || item->up_metadata == nil ||
        item->down_quants == nil || item->down_metadata == nil ||
        item->input_quants == nil || item->input_metadata == nil ||
        item->output_quants == nil || item->output_metadata == nil ||
        item->constants == nil || item->recurrent_state == nil ||
        item->convolution_state == nil) {
        if (error != NULL)
            *error = [NSString stringWithFormat:
                @"cannot allocate DeltaNet layer %u", index];
        return nil;
    }
    memcpy(item->constants.contents,
           (unsigned char *)item->mapping + h->constants_offset,
           h->constants_f32_count * sizeof(float));
    memset(item->recurrent_state.contents, 0,
           item->recurrent_state.length);
    memset(item->convolution_state.contents, 0,
           item->convolution_state.length);
    item->attention = NO;
    runtime->state_bytes += item->recurrent_state.length +
                            item->convolution_state.length;
    return item;
}

static Q36DecodeLayer *load_attention_layer(Q36DecodeRuntime *runtime,
                                            NSString *path, unsigned index,
                                            NSString **error) {
    Q36DecodeLayer *item = [Q36DecodeLayer new];
    if (map_file(path.fileSystemRepresentation, &item->file,
                 &item->mapping, &item->length, error) != 0) return nil;
    item->attention_header = item->mapping;
    if (validate_attention_header(item->attention_header,
                                  item->length, index) != 0) {
        if (error != NULL)
            *error = [NSString stringWithFormat:
                @"invalid attention image at layer %u", index];
        return nil;
    }
    const qwen36_m3_attention_image_header *h = item->attention_header;
#define ATTENTION_MAP(field) \
    mapped_buffer(runtime->device, item->mapping, h->field##_offset, \
                  h->field##_bytes)
    item->gate_quants = ATTENTION_MAP(gate_quants);
    item->gate_metadata = ATTENTION_MAP(gate_metadata);
    item->up_quants = ATTENTION_MAP(up_quants);
    item->up_metadata = ATTENTION_MAP(up_metadata);
    item->down_quants = ATTENTION_MAP(down_quants);
    item->down_metadata = ATTENTION_MAP(down_metadata);
    item->input_quants = ATTENTION_MAP(attention_input_quants);
    item->input_metadata = ATTENTION_MAP(attention_input_metadata);
    item->output_quants = ATTENTION_MAP(attention_output_quants);
    item->output_metadata = ATTENTION_MAP(attention_output_metadata);
#undef ATTENTION_MAP
    item->constants = [runtime->device
        newBufferWithLength:h->constants_f32_count * sizeof(float)
                   options:MTLResourceStorageModeShared];
    size_t cache_bytes = (size_t)runtime->capacity * 4 * 256 *
                         sizeof(float);
    item->key_cache = [runtime->device newBufferWithLength:cache_bytes
                                                   options:
        MTLResourceStorageModeShared];
    item->value_cache = [runtime->device newBufferWithLength:cache_bytes
                                                     options:
        MTLResourceStorageModeShared];
    if (item->gate_quants == nil || item->gate_metadata == nil ||
        item->up_quants == nil || item->up_metadata == nil ||
        item->down_quants == nil || item->down_metadata == nil ||
        item->input_quants == nil || item->input_metadata == nil ||
        item->output_quants == nil || item->output_metadata == nil ||
        item->constants == nil || item->key_cache == nil ||
        item->value_cache == nil) {
        if (error != NULL)
            *error = [NSString stringWithFormat:
                @"cannot allocate attention layer %u", index];
        return nil;
    }
    memcpy(item->constants.contents,
           (unsigned char *)item->mapping + h->constants_offset,
           h->constants_f32_count * sizeof(float));
    memset(item->key_cache.contents, 0, cache_bytes);
    memset(item->value_cache.contents, 0, cache_bytes);
    item->attention = YES;
    runtime->kv_bytes += cache_bytes * 2;
    return item;
}

static void encode_rms_half(Q36DecodeRuntime *r,
                            id<MTLComputeCommandEncoder> encoder,
                            id<MTLBuffer> input, id<MTLBuffer> constants,
                            NSUInteger offset) {
    [encoder setComputePipelineState:r->rms_half];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:constants offset:offset atIndex:1];
    [encoder setBuffer:r->normalized offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void encode_mlp(Q36DecodeRuntime *r,
                       id<MTLComputeCommandEncoder> encoder,
                       Q36DecodeLayer *layer,
                       id<MTLBuffer> residual) {
    [encoder setComputePipelineState:r->mlp_gate_up];
    [encoder setBuffer:r->post_normalized offset:0 atIndex:0];
    [encoder setBuffer:layer->gate_quants offset:0 atIndex:1];
    [encoder setBuffer:layer->gate_metadata offset:0 atIndex:2];
    [encoder setBuffer:layer->up_quants offset:0 atIndex:3];
    [encoder setBuffer:layer->up_metadata offset:0 atIndex:4];
    [encoder setBuffer:r->intermediate offset:0 atIndex:5];
    [encoder dispatchThreadgroups:MTLSizeMake((17408 + 7) / 8, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder setComputePipelineState:r->mlp_down];
    [encoder setBuffer:r->intermediate offset:0 atIndex:0];
    [encoder setBuffer:layer->down_quants offset:0 atIndex:1];
    [encoder setBuffer:layer->down_metadata offset:0 atIndex:2];
    [encoder setBuffer:residual offset:0 atIndex:3];
    [encoder setBuffer:r->layer_output offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake((5120 + 7) / 8, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void encode_delta(Q36DecodeRuntime *r,
                         id<MTLComputeCommandEncoder> encoder,
                         Q36DecodeLayer *layer) {
    const qwen36_m3_image_header *h = layer->delta_header;
    encode_rms_half(r, encoder, r->hidden_half, layer->constants,
                    h->input_norm_constants_index * sizeof(float));
    [encoder setComputePipelineState:r->delta_inputs];
    [encoder setBuffer:r->normalized offset:0 atIndex:0];
    [encoder setBuffer:layer->input_quants offset:0 atIndex:1];
    [encoder setBuffer:layer->input_metadata offset:0 atIndex:2];
    [encoder setBuffer:r->projected offset:0 atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake((16480 + 7) / 8, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder setComputePipelineState:r->delta_conv];
    [encoder setBuffer:r->projected offset:0 atIndex:0];
    [encoder setBuffer:layer->constants
                 offset:h->conv_constants_index * sizeof(float) atIndex:1];
    [encoder setBuffer:layer->convolution_state offset:0 atIndex:2];
    [encoder setBuffer:r->convolved offset:0 atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(10240, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder setComputePipelineState:r->delta_prepare];
    [encoder setBuffer:r->convolved offset:0 atIndex:0];
    [encoder setBuffer:r->projected offset:0 atIndex:1];
    [encoder setBuffer:layer->constants
                 offset:h->a_log_constants_index * sizeof(float) atIndex:2];
    [encoder setBuffer:layer->constants
                 offset:h->dt_bias_constants_index * sizeof(float) atIndex:3];
    [encoder setBuffer:r->query offset:0 atIndex:4];
    [encoder setBuffer:r->key offset:0 atIndex:5];
    [encoder setBuffer:r->value offset:0 atIndex:6];
    [encoder setBuffer:r->decay offset:0 atIndex:7];
    [encoder setBuffer:r->beta offset:0 atIndex:8];
    [encoder dispatchThreadgroups:MTLSizeMake(48, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder setComputePipelineState:r->delta_recurrent];
    [encoder setBuffer:r->query offset:0 atIndex:0];
    [encoder setBuffer:r->key offset:0 atIndex:1];
    [encoder setBuffer:r->value offset:0 atIndex:2];
    [encoder setBuffer:r->decay offset:0 atIndex:3];
    [encoder setBuffer:r->beta offset:0 atIndex:4];
    [encoder setBuffer:layer->recurrent_state offset:0 atIndex:5];
    [encoder setBuffer:r->core offset:0 atIndex:6];
    [encoder dispatchThreads:MTLSizeMake(48 * 128, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder setComputePipelineState:r->delta_gated_norm];
    [encoder setBuffer:r->core offset:0 atIndex:0];
    [encoder setBuffer:r->projected offset:0 atIndex:1];
    [encoder setBuffer:layer->constants
                 offset:h->recurrent_norm_constants_index * sizeof(float)
                 atIndex:2];
    [encoder setBuffer:r->gated offset:0 atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(48, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder setComputePipelineState:r->delta_output];
    [encoder setBuffer:r->gated offset:0 atIndex:0];
    [encoder setBuffer:layer->output_quants offset:0 atIndex:1];
    [encoder setBuffer:layer->output_metadata offset:0 atIndex:2];
    [encoder setBuffer:r->hidden_half offset:0 atIndex:3];
    [encoder setBuffer:r->mixer_output offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake((5120 + 7) / 8, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder setComputePipelineState:r->rms_float];
    [encoder setBuffer:r->mixer_output offset:0 atIndex:0];
    [encoder setBuffer:layer->constants
                 offset:h->post_norm_constants_index * sizeof(float)
                 atIndex:1];
    [encoder setBuffer:r->post_normalized offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    encode_mlp(r, encoder, layer, r->mixer_output);
}

static void encode_attention(Q36DecodeRuntime *r,
                             id<MTLComputeCommandEncoder> encoder,
                             Q36DecodeLayer *layer, uint32_t position) {
    const qwen36_m3_attention_image_header *h = layer->attention_header;
    q36_decode_attention_parameters parameters = {
        .position = position,
        .context_length = position + 1,
        .cache_capacity = r->capacity,
        .reserved = 0
    };
    encode_rms_half(r, encoder, r->hidden_half, layer->constants,
                    h->input_norm_constants_index * sizeof(float));
    [encoder setComputePipelineState:r->attention_inputs];
    [encoder setBuffer:r->normalized offset:0 atIndex:0];
    [encoder setBuffer:layer->input_quants offset:0 atIndex:1];
    [encoder setBuffer:layer->input_metadata offset:0 atIndex:2];
    [encoder setBuffer:r->projected offset:0 atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake((14336 + 7) / 8, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder setComputePipelineState:r->attention_query];
    [encoder setBuffer:r->projected offset:0 atIndex:0];
    [encoder setBuffer:layer->constants
                 offset:h->q_norm_constants_index * sizeof(float) atIndex:1];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
    [encoder setBuffer:r->query offset:0 atIndex:3];
    [encoder setBuffer:r->query_gate offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(24, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder setComputePipelineState:r->attention_key_value];
    [encoder setBuffer:r->projected offset:0 atIndex:0];
    [encoder setBuffer:layer->constants
                 offset:h->k_norm_constants_index * sizeof(float) atIndex:1];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
    [encoder setBuffer:layer->key_cache offset:0 atIndex:3];
    [encoder setBuffer:layer->value_cache offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(4, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder setComputePipelineState:r->attention_scores];
    [encoder setBuffer:r->query offset:0 atIndex:0];
    [encoder setBuffer:layer->key_cache offset:0 atIndex:1];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
    [encoder setBuffer:r->attention_scores_buffer offset:0 atIndex:3];
    NSUInteger score_count = 24 * (position + 1);
    [encoder dispatchThreadgroups:MTLSizeMake((score_count + 7) / 8, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder setComputePipelineState:r->attention_softmax_value];
    [encoder setBuffer:r->attention_scores_buffer offset:0 atIndex:0];
    [encoder setBuffer:layer->value_cache offset:0 atIndex:1];
    [encoder setBuffer:r->query_gate offset:0 atIndex:2];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder setBuffer:r->gated offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(24, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder setComputePipelineState:r->attention_output];
    [encoder setBuffer:r->gated offset:0 atIndex:0];
    [encoder setBuffer:layer->output_quants offset:0 atIndex:1];
    [encoder setBuffer:layer->output_metadata offset:0 atIndex:2];
    [encoder setBuffer:r->hidden_half offset:0 atIndex:3];
    [encoder setBuffer:r->mixer_output offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake((5120 + 7) / 8, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder setComputePipelineState:r->rms_float];
    [encoder setBuffer:r->mixer_output offset:0 atIndex:0];
    [encoder setBuffer:layer->constants
                 offset:h->post_norm_constants_index * sizeof(float)
                 atIndex:1];
    [encoder setBuffer:r->post_normalized offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    encode_mlp(r, encoder, layer, r->mixer_output);
}

static int initialize_pipelines(Q36DecodeRuntime *r, NSString **message) {
    NSError *error = nil;
#define MAKE(field, name) do { \
    r->field = decode_pipeline(r, name, &error); \
    if (r->field == nil) { \
        if (message != NULL) *message = error.localizedDescription; \
        return -1; \
    } \
} while (0)
    MAKE(rms_half, @"qwen36_rmsnorm_f16_to_f16");
    MAKE(rms_float, @"qwen36_rmsnorm_f32_to_f16");
    MAKE(delta_inputs, @"qwen36_q4_delta_inputs");
    MAKE(delta_conv, @"qwen36_delta_causal_conv_silu");
    MAKE(delta_prepare, @"qwen36_delta_prepare");
    MAKE(delta_recurrent, @"qwen36_deltanet_direct");
    MAKE(delta_gated_norm, @"qwen36_delta_gated_rmsnorm");
    MAKE(delta_output, @"qwen36_q4_delta_output_residual");
    MAKE(attention_inputs, @"qwen36_q4_attention_inputs");
    MAKE(attention_query, @"qwen36_attention_prepare_query");
    MAKE(attention_key_value, @"qwen36_attention_prepare_key_value");
    MAKE(attention_scores, @"qwen36_attention_scores");
    MAKE(attention_softmax_value,
         @"qwen36_attention_softmax_value_gate");
    MAKE(attention_output, @"qwen36_q4_attention_output_residual");
    MAKE(mlp_gate_up, @"qwen36_q4_gate_up_silu");
    MAKE(mlp_down, @"qwen36_q4_mlp_down_residual_f32");
    MAKE(embedding, @"qwen36_q4_embedding_lookup");
    MAKE(convert_hidden, @"qwen36_f32_to_f16_hidden");
    MAKE(lm_head, @"qwen36_q4_lm_head");
#undef MAKE
    return 0;
}

static int allocate_workspace(Q36DecodeRuntime *r, NSString **message) {
    MTLResourceOptions options = MTLResourceStorageModeShared;
#define ALLOC(field, count, type) \
    r->field = [r->device newBufferWithLength:(size_t)(count) * sizeof(type) \
                                       options:options]
    ALLOC(hidden_half, 5120, uint16_t);
    ALLOC(normalized, 5120, uint16_t);
    ALLOC(projected, 16480, float);
    ALLOC(convolved, 10240, float);
    ALLOC(query, 6144, float);
    ALLOC(query_gate, 6144, float);
    ALLOC(key, 6144, float);
    ALLOC(value, 6144, float);
    ALLOC(decay, 48, float);
    ALLOC(beta, 48, float);
    ALLOC(core, 6144, float);
    ALLOC(gated, 6144, float);
    ALLOC(attention_scores_buffer, (size_t)24 * r->capacity, float);
    ALLOC(mixer_output, 5120, float);
    ALLOC(post_normalized, 5120, uint16_t);
    ALLOC(intermediate, 17408, float);
    ALLOC(layer_output, 5120, float);
    ALLOC(final_normalized, 5120, uint16_t);
    ALLOC(logits, QWEN36_VOCAB_SIZE, float);
#undef ALLOC
    if (r->hidden_half == nil || r->normalized == nil ||
        r->projected == nil || r->convolved == nil || r->query == nil ||
        r->query_gate == nil || r->key == nil || r->value == nil ||
        r->decay == nil || r->beta == nil || r->core == nil ||
        r->gated == nil || r->attention_scores_buffer == nil ||
        r->mixer_output == nil || r->post_normalized == nil ||
        r->intermediate == nil || r->layer_output == nil ||
        r->final_normalized == nil || r->logits == nil) {
        if (message != NULL) *message = @"cannot allocate decode workspace";
        return -1;
    }
    return 0;
}

qwen36_m3_model *qwen36_m3_model_open(
    const char *model_directory, const char *metallib_path,
    uint32_t context_capacity, char *error_message,
    size_t error_message_capacity) {
    if (model_directory == NULL || metallib_path == NULL ||
        context_capacity == 0) {
        decode_error(error_message, error_message_capacity,
                     @"invalid model open arguments");
        return NULL;
    }
    @autoreleasepool {
        Q36DecodeRuntime *r = [Q36DecodeRuntime new];
        r->capacity = context_capacity;
        r->layers = [NSMutableArray arrayWithCapacity:64];
        r->device = MTLCreateSystemDefaultDevice();
        NSError *metal_error = nil;
        NSURL *url = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:metallib_path]];
        r->library = r->device != nil ?
            [r->device newLibraryWithURL:url error:&metal_error] : nil;
        r->queue = [r->device newCommandQueue];
        NSString *message = nil;
        if (r->device == nil || r->library == nil || r->queue == nil ||
            initialize_pipelines(r, &message) != 0 ||
            allocate_workspace(r, &message) != 0) {
            decode_error(error_message, error_message_capacity,
                message != nil ? message :
                (metal_error != nil ? metal_error.localizedDescription :
                                      @"cannot initialize Metal runtime"));
            return NULL;
        }
        NSString *directory =
            [NSString stringWithUTF8String:model_directory];
        NSString *global_path =
            [directory stringByAppendingPathComponent:@"global.q36global"];
        r->global = load_global(r, global_path, &message);
        if (r->global == nil) {
            decode_error(error_message, error_message_capacity, message);
            return NULL;
        }
        r->mapped_bytes = r->global->length;
        for (unsigned index = 0; index < 64; ++index) {
            NSString *filename = index % 4 == 3 ?
                [NSString stringWithFormat:@"layer-%02u.q36att", index] :
                [NSString stringWithFormat:@"layer-%02u.q36delta", index];
            NSString *path =
                [directory stringByAppendingPathComponent:filename];
            Q36DecodeLayer *layer = index % 4 == 3 ?
                load_attention_layer(r, path, index, &message) :
                load_delta_layer(r, path, index, &message);
            if (layer == nil) {
                decode_error(error_message, error_message_capacity, message);
                return NULL;
            }
            [r->layers addObject:layer];
            r->mapped_bytes += layer->length;
        }
        qwen36_m3_model *model = calloc(1, sizeof(*model));
        if (model == NULL) {
            decode_error(error_message, error_message_capacity,
                         @"cannot allocate model handle");
            return NULL;
        }
        model->runtime = (__bridge_retained void *)r;
        return model;
    }
}

void qwen36_m3_model_reset(qwen36_m3_model *model) {
    if (model == NULL || model->runtime == NULL) return;
    Q36DecodeRuntime *r = (__bridge Q36DecodeRuntime *)model->runtime;
    for (Q36DecodeLayer *layer in r->layers) {
        if (layer->attention) {
            memset(layer->key_cache.contents, 0, layer->key_cache.length);
            memset(layer->value_cache.contents, 0,
                   layer->value_cache.length);
        } else {
            memset(layer->recurrent_state.contents, 0,
                   layer->recurrent_state.length);
            memset(layer->convolution_state.contents, 0,
                   layer->convolution_state.length);
        }
    }
}

int qwen36_m3_model_forward(
    qwen36_m3_model *model, uint32_t token_id, uint32_t position,
    qwen36_m3_decode_result *result, const float **logits,
    size_t *logit_count, char *error_message, size_t error_message_capacity) {
    if (model == NULL || model->runtime == NULL || result == NULL ||
        logits == NULL || logit_count == NULL ||
        token_id >= QWEN36_VOCAB_SIZE) {
        decode_error(error_message, error_message_capacity,
                     @"invalid decode arguments");
        return 1;
    }
    Q36DecodeRuntime *r = (__bridge Q36DecodeRuntime *)model->runtime;
    if (position >= r->capacity) {
        decode_error(error_message, error_message_capacity,
                     @"position exceeds configured context capacity");
        return 1;
    }
    memset(result, 0, sizeof(*result));
    @autoreleasepool {
        id<MTLCommandBuffer> command = [r->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        [encoder setComputePipelineState:r->embedding];
        [encoder setBuffer:r->global->embedding_quants offset:0 atIndex:0];
        [encoder setBuffer:r->global->embedding_metadata offset:0 atIndex:1];
        [encoder setBytes:&token_id length:sizeof(token_id) atIndex:2];
        [encoder setBuffer:r->hidden_half offset:0 atIndex:3];
        [encoder dispatchThreads:MTLSizeMake(5120, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        for (unsigned index = 0; index < 64; ++index) {
            Q36DecodeLayer *layer = r->layers[index];
            if (layer->attention)
                encode_attention(r, encoder, layer, position);
            else
                encode_delta(r, encoder, layer);
            if (index != 63) {
                [encoder setComputePipelineState:r->convert_hidden];
                [encoder setBuffer:r->layer_output offset:0 atIndex:0];
                [encoder setBuffer:r->hidden_half offset:0 atIndex:1];
                [encoder dispatchThreads:MTLSizeMake(5120, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }
        [encoder setComputePipelineState:r->rms_float];
        [encoder setBuffer:r->layer_output offset:0 atIndex:0];
        [encoder setBuffer:r->global->constants offset:0 atIndex:1];
        [encoder setBuffer:r->final_normalized offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder setComputePipelineState:r->lm_head];
        [encoder setBuffer:r->final_normalized offset:0 atIndex:0];
        [encoder setBuffer:r->global->lm_head_quants offset:0 atIndex:1];
        [encoder setBuffer:r->global->lm_head_metadata offset:0 atIndex:2];
        [encoder setBuffer:r->logits offset:0 atIndex:3];
        [encoder dispatchThreadgroups:
            MTLSizeMake((QWEN36_VOCAB_SIZE + 7) / 8, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        double start = decode_seconds();
        [command commit];
        [command waitUntilCompleted];
        double duration = decode_seconds() - start;
        if (command.status != MTLCommandBufferStatusCompleted) {
            decode_error(error_message, error_message_capacity,
                command.error != nil ? command.error.localizedDescription :
                                       @"Metal decode failed");
            return 2;
        }
        result->input_token = token_id;
        result->next_token = 0;
        result->position = position;
        result->duration_ms = duration * 1000.0;
        result->mapped_weight_bytes = r->mapped_bytes;
        result->state_bytes = r->state_bytes;
        result->kv_cache_bytes = r->kv_bytes;
        result->physical_footprint_bytes = decode_footprint();
        *logits = r->logits.contents;
        *logit_count = QWEN36_VOCAB_SIZE;
        return 0;
    }
}

int qwen36_m3_model_decode(
    qwen36_m3_model *model, uint32_t token_id, uint32_t position,
    qwen36_m3_decode_result *result, char *error_message,
    size_t error_message_capacity) {
    const float *logits = NULL;
    size_t logit_count = 0;
    int status = qwen36_m3_model_forward(
        model, token_id, position, result, &logits, &logit_count,
        error_message, error_message_capacity);
    if (status != 0) return status;
    uint32_t best = 0;
    float best_value = logits[0];
    for (uint32_t index = 1; index < logit_count; ++index) {
        if (logits[index] > best_value) {
            best_value = logits[index];
            best = index;
        }
    }
    result->next_token = best;
    return 0;
}

void qwen36_m3_model_close(qwen36_m3_model *model) {
    if (model == NULL) return;
    if (model->runtime != NULL) {
        CFBridgingRelease(model->runtime);
        model->runtime = NULL;
    }
    free(model);
}
