#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "qwen36_m3_decode.h"
#include "qwen36_m3.h"
#include "qwen36_m3_attention_image.h"
#include "qwen36_m3_global_image.h"
#include "qwen36_m3_mtp_image.h"
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

typedef struct {
    uint32_t rows;
    uint32_t groups_per_row;
} q36_prefill_gemm_parameters;

typedef struct {
    uint32_t start_position;
    uint32_t cache_capacity;
} q36_prefill_attention_parameters;

enum { Q36_PREFILL_MAX_BATCH = 32 };

/* One specialized pipeline set per compiled prefill shape bucket. */
@interface Q36PrefillPipelines : NSObject {
@public
    uint32_t batch;
    id<MTLComputePipelineState> embedding;
    id<MTLComputePipelineState> rms_f16;
    id<MTLComputePipelineState> rms_f32;
    id<MTLComputePipelineState> convert_hidden;
    id<MTLComputePipelineState> gemm_f16;
    id<MTLComputePipelineState> gemm_f32_residual_f16;
    id<MTLComputePipelineState> gemm_f32_residual_f32;
    id<MTLComputePipelineState> gemm_f16_mma;
    id<MTLComputePipelineState> gemm_f32_residual_f16_mma;
    id<MTLComputePipelineState> gemm_f32_residual_f32_mma;
    id<MTLComputePipelineState> gemm_f16_mma2;
    id<MTLComputePipelineState> convert_x;
    id<MTLComputePipelineState> gemm_f32_residual_f16_mma2;
    id<MTLComputePipelineState> gemm_f32_residual_f32_mma2;
    id<MTLComputePipelineState> silu_mul;
    id<MTLComputePipelineState> delta_conv;
    id<MTLComputePipelineState> delta_prepare;
    id<MTLComputePipelineState> delta_recurrent;
    id<MTLComputePipelineState> delta_gated_norm;
    id<MTLComputePipelineState> attention_query;
    id<MTLComputePipelineState> attention_key_value;
    id<MTLComputePipelineState> attention_scores;
    id<MTLComputePipelineState> attention_softmax_value;
    id<MTLComputePipelineState> mtp_fuse;
}
@end

@implementation Q36PrefillPipelines
@end

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
    int prefill_mma_level;
    id<MTLResidencySet> residency API_AVAILABLE(macos(15.0));

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

    Q36PrefillPipelines *prefill1;
    Q36PrefillPipelines *prefill2;
    Q36PrefillPipelines *prefill3;
    Q36PrefillPipelines *prefill4;
    uint32_t mtp_depth;
    Q36PrefillPipelines *prefill16;
    Q36PrefillPipelines *prefill32;

    Q36DecodeLayer *mtp_layer;
    int mtp_file;
    void *mtp_mapping;
    size_t mtp_mapping_length;
    const qwen36_m3_mtp_image_header *mtp_header;
    id<MTLBuffer> mtp_fc_quants;
    id<MTLBuffer> mtp_fc_metadata;
    id<MTLBuffer> mtp_constants;
    id<MTLBuffer> mtp_fused;
    id<MTLBuffer> mtp_logits;
    id<MTLBuffer> p_logits2;
    id<MTLBuffer> snapshot_recurrent;
    id<MTLBuffer> snapshot_convolution;
    id<MTLBuffer> p_token_ids;
    id<MTLBuffer> p_hidden_half;
    id<MTLBuffer> p_normalized;
    id<MTLBuffer> p_projected;
    id<MTLBuffer> p_convolved;
    id<MTLBuffer> p_query;
    id<MTLBuffer> p_query_gate;
    id<MTLBuffer> p_key;
    id<MTLBuffer> p_value;
    id<MTLBuffer> p_decay;
    id<MTLBuffer> p_beta;
    id<MTLBuffer> p_core;
    id<MTLBuffer> p_gated;
    id<MTLBuffer> p_scores;
    id<MTLBuffer> p_mixer_output;
    id<MTLBuffer> p_post_normalized;
    id<MTLBuffer> p_mlp_gate;
    id<MTLBuffer> p_mlp_up;
    id<MTLBuffer> p_mlp_activated;
    id<MTLBuffer> p_layer_output;
    id<MTLBuffer> p_x_half;
}
@end

@implementation Q36DecodeRuntime
@end

struct qwen36_m3_model {
    void *runtime;
    void *pending_command;
    uint32_t pending_token;
    uint32_t pending_position;
    double pending_start;
    /* Source of the hidden state feeding the next MTP draft:
     * 0 = decode workspace, 1 = verify batch row 0, 2 = row 1. */
    int mtp_hidden_source;
    uint32_t mtp_adaptive_depth;
    float mtp_accept_ema;
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
           memcmp(sha, QWEN36_M3_EXPECTED_SOURCE_SHA256_3, 64) == 0 ||
           memcmp(sha, QWEN36_M3_EXPECTED_MTP_SHA256, 64) == 0;
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
        h->layer_index != layer ||
        (layer % 4 != 3 && layer != QWEN36_M3_MTP_LAYER_INDEX) ||
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

static Q36PrefillPipelines *prefill_pipelines(
    Q36DecodeRuntime *r, uint32_t batch, NSString **message) {
    MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
    [values setConstantValue:&batch type:MTLDataTypeUInt atIndex:0];
    Q36PrefillPipelines *p = [Q36PrefillPipelines new];
    p->batch = batch;
    NSError *error = nil;
#define MAKE_PREFILL(field, name) do { \
    id<MTLFunction> function = \
        [r->library newFunctionWithName:name constantValues:values \
                                  error:&error]; \
    p->field = function != nil ? \
        [r->device newComputePipelineStateWithFunction:function \
                                                 error:&error] : nil; \
    if (p->field == nil) { \
        if (message != NULL) *message = error != nil ? \
            error.localizedDescription : @"cannot build prefill pipeline"; \
        return nil; \
    } \
} while (0)
    MAKE_PREFILL(embedding, @"qwen36_prefill_embedding");
    MAKE_PREFILL(rms_f16, @"qwen36_prefill_rmsnorm_f16");
    MAKE_PREFILL(rms_f32, @"qwen36_prefill_rmsnorm_f32");
    MAKE_PREFILL(convert_hidden, @"qwen36_prefill_convert_hidden");
    MAKE_PREFILL(gemm_f16, @"qwen36_prefill_q4_gemm_f16");
    MAKE_PREFILL(gemm_f32_residual_f16,
                 @"qwen36_prefill_q4_gemm_f32_residual_f16");
    MAKE_PREFILL(gemm_f32_residual_f32,
                 @"qwen36_prefill_q4_gemm_f32_residual_f32");
    MAKE_PREFILL(gemm_f16_mma, @"qwen36_prefill_q4_gemm_f16_mma");
    MAKE_PREFILL(gemm_f32_residual_f16_mma,
                 @"qwen36_prefill_q4_gemm_f32_residual_f16_mma");
    MAKE_PREFILL(gemm_f32_residual_f32_mma,
                 @"qwen36_prefill_q4_gemm_f32_residual_f32_mma");
    MAKE_PREFILL(gemm_f16_mma2, @"qwen36_prefill_q4_gemm_f16_mma2");
    MAKE_PREFILL(convert_x, @"qwen36_prefill_convert_x");
    MAKE_PREFILL(gemm_f32_residual_f16_mma2,
                 @"qwen36_prefill_q4_gemm_f32_residual_f16_mma2");
    MAKE_PREFILL(gemm_f32_residual_f32_mma2,
                 @"qwen36_prefill_q4_gemm_f32_residual_f32_mma2");
    MAKE_PREFILL(silu_mul, @"qwen36_prefill_silu_mul");
    MAKE_PREFILL(delta_conv, @"qwen36_prefill_delta_conv");
    MAKE_PREFILL(delta_prepare, @"qwen36_prefill_delta_prepare");
    MAKE_PREFILL(delta_recurrent, @"qwen36_prefill_delta_recurrent");
    MAKE_PREFILL(delta_gated_norm, @"qwen36_prefill_delta_gated_norm");
    MAKE_PREFILL(attention_query, @"qwen36_prefill_attention_query");
    MAKE_PREFILL(attention_key_value,
                 @"qwen36_prefill_attention_key_value");
    MAKE_PREFILL(attention_scores, @"qwen36_prefill_attention_scores");
    MAKE_PREFILL(attention_softmax_value,
                 @"qwen36_prefill_attention_softmax_value");
    MAKE_PREFILL(mtp_fuse, @"qwen36_prefill_mtp_fuse");
#undef MAKE_PREFILL
    return p;
}

static void encode_prefill_gemm_f16(
    Q36DecodeRuntime *r, Q36PrefillPipelines *p,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> x, id<MTLBuffer> quants, id<MTLBuffer> metadata,
    id<MTLBuffer> output, uint32_t rows, uint32_t groups_per_row) {
    q36_prefill_gemm_parameters parameters = {rows, groups_per_row};
    int use_mma = r->prefill_mma_level > 0 && p->batch > 8;
    [encoder setComputePipelineState:
        use_mma ? (r->prefill_mma_level >= 2 ?
                   p->gemm_f16_mma2 : p->gemm_f16_mma) : p->gemm_f16];
    [encoder setBuffer:x offset:0 atIndex:0];
    [encoder setBuffer:quants offset:0 atIndex:1];
    [encoder setBuffer:metadata offset:0 atIndex:2];
    [encoder setBuffer:output offset:0 atIndex:3];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:4];
    if (use_mma)
        [encoder dispatchThreadgroups:MTLSizeMake(rows / 32, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    else
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 7) / 8, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void encode_prefill_gemm_residual(
    Q36DecodeRuntime *r, Q36PrefillPipelines *p,
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> x, id<MTLBuffer> quants, id<MTLBuffer> metadata,
    id<MTLBuffer> residual, id<MTLBuffer> output, uint32_t rows,
    uint32_t groups_per_row, BOOL residual_is_half) {
    q36_prefill_gemm_parameters parameters = {rows, groups_per_row};
    int use_mma = r->prefill_mma_level > 0 && p->batch > 8;
    id<MTLComputePipelineState> pipeline;
    if (residual_is_half)
        pipeline = !use_mma ? p->gemm_f32_residual_f16 :
            (r->prefill_mma_level >= 2 ?
             p->gemm_f32_residual_f16_mma2 : p->gemm_f32_residual_f16_mma);
    else
        pipeline = !use_mma ? p->gemm_f32_residual_f32 :
            (r->prefill_mma_level >= 2 ?
             p->gemm_f32_residual_f32_mma2 : p->gemm_f32_residual_f32_mma);
    if (use_mma && r->prefill_mma_level >= 2) {
        uint32_t columns = groups_per_row * 64;
        [encoder setComputePipelineState:p->convert_x];
        [encoder setBuffer:x offset:0 atIndex:0];
        [encoder setBuffer:r->p_x_half offset:0 atIndex:1];
        [encoder setBytes:&columns length:sizeof(columns) atIndex:2];
        [encoder dispatchThreads:MTLSizeMake(columns, p->batch, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        x = r->p_x_half;
    }
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:x offset:0 atIndex:0];
    [encoder setBuffer:quants offset:0 atIndex:1];
    [encoder setBuffer:metadata offset:0 atIndex:2];
    [encoder setBuffer:residual offset:0 atIndex:3];
    [encoder setBuffer:output offset:0 atIndex:4];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:5];
    if (use_mma)
        [encoder dispatchThreadgroups:MTLSizeMake(rows / 32, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    else
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 7) / 8, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void encode_prefill_mlp(Q36DecodeRuntime *r, Q36PrefillPipelines *p,
                               id<MTLComputeCommandEncoder> encoder,
                               Q36DecodeLayer *layer) {
    encode_prefill_gemm_f16(r, p, encoder, r->p_post_normalized,
                            layer->gate_quants, layer->gate_metadata,
                            r->p_mlp_gate, 17408, 80);
    encode_prefill_gemm_f16(r, p, encoder, r->p_post_normalized,
                            layer->up_quants, layer->up_metadata,
                            r->p_mlp_up, 17408, 80);
    [encoder setComputePipelineState:p->silu_mul];
    [encoder setBuffer:r->p_mlp_gate offset:0 atIndex:0];
    [encoder setBuffer:r->p_mlp_up offset:0 atIndex:1];
    [encoder setBuffer:r->p_mlp_activated offset:0 atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(17408, p->batch, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    encode_prefill_gemm_residual(r, p, encoder, r->p_mlp_activated,
                                 layer->down_quants, layer->down_metadata,
                                 r->p_mixer_output, r->p_layer_output,
                                 5120, 272, NO);
}

static void encode_prefill_delta(Q36DecodeRuntime *r,
                                 Q36PrefillPipelines *p,
                                 id<MTLComputeCommandEncoder> encoder,
                                 Q36DecodeLayer *layer) {
    const qwen36_m3_image_header *h = layer->delta_header;
    [encoder setComputePipelineState:p->rms_f16];
    [encoder setBuffer:r->p_hidden_half offset:0 atIndex:0];
    [encoder setBuffer:layer->constants
                 offset:h->input_norm_constants_index * sizeof(float)
                atIndex:1];
    [encoder setBuffer:r->p_normalized offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(p->batch, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    encode_prefill_gemm_f16(r, p, encoder, r->p_normalized,
                            layer->input_quants, layer->input_metadata,
                            r->p_projected, 16480, 80);
    [encoder setComputePipelineState:p->delta_conv];
    [encoder setBuffer:r->p_projected offset:0 atIndex:0];
    [encoder setBuffer:layer->constants
                 offset:h->conv_constants_index * sizeof(float) atIndex:1];
    [encoder setBuffer:layer->convolution_state offset:0 atIndex:2];
    [encoder setBuffer:r->p_convolved offset:0 atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(10240, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder setComputePipelineState:p->delta_prepare];
    [encoder setBuffer:r->p_convolved offset:0 atIndex:0];
    [encoder setBuffer:r->p_projected offset:0 atIndex:1];
    [encoder setBuffer:layer->constants
                 offset:h->a_log_constants_index * sizeof(float) atIndex:2];
    [encoder setBuffer:layer->constants
                 offset:h->dt_bias_constants_index * sizeof(float) atIndex:3];
    [encoder setBuffer:r->p_query offset:0 atIndex:4];
    [encoder setBuffer:r->p_key offset:0 atIndex:5];
    [encoder setBuffer:r->p_value offset:0 atIndex:6];
    [encoder setBuffer:r->p_decay offset:0 atIndex:7];
    [encoder setBuffer:r->p_beta offset:0 atIndex:8];
    [encoder dispatchThreadgroups:MTLSizeMake(48, p->batch, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [encoder setComputePipelineState:p->delta_recurrent];
    [encoder setBuffer:r->p_query offset:0 atIndex:0];
    [encoder setBuffer:r->p_key offset:0 atIndex:1];
    [encoder setBuffer:r->p_value offset:0 atIndex:2];
    [encoder setBuffer:r->p_decay offset:0 atIndex:3];
    [encoder setBuffer:r->p_beta offset:0 atIndex:4];
    [encoder setBuffer:layer->recurrent_state offset:0 atIndex:5];
    [encoder setBuffer:r->p_core offset:0 atIndex:6];
    [encoder dispatchThreads:MTLSizeMake(48 * 128, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder setComputePipelineState:p->delta_gated_norm];
    [encoder setBuffer:r->p_core offset:0 atIndex:0];
    [encoder setBuffer:r->p_projected offset:0 atIndex:1];
    [encoder setBuffer:layer->constants
                 offset:h->recurrent_norm_constants_index * sizeof(float)
                atIndex:2];
    [encoder setBuffer:r->p_gated offset:0 atIndex:3];
    [encoder dispatchThreadgroups:MTLSizeMake(48, p->batch, 1)
            threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    encode_prefill_gemm_residual(r, p, encoder, r->p_gated,
                                 layer->output_quants,
                                 layer->output_metadata,
                                 r->p_hidden_half, r->p_mixer_output,
                                 5120, 96, YES);
    [encoder setComputePipelineState:p->rms_f32];
    [encoder setBuffer:r->p_mixer_output offset:0 atIndex:0];
    [encoder setBuffer:layer->constants
                 offset:h->post_norm_constants_index * sizeof(float)
                atIndex:1];
    [encoder setBuffer:r->p_post_normalized offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(p->batch, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    encode_prefill_mlp(r, p, encoder, layer);
}

static void encode_prefill_attention(Q36DecodeRuntime *r,
                                     Q36PrefillPipelines *p,
                                     id<MTLComputeCommandEncoder> encoder,
                                     Q36DecodeLayer *layer,
                                     uint32_t start_position) {
    const qwen36_m3_attention_image_header *h = layer->attention_header;
    q36_prefill_attention_parameters parameters = {
        start_position, r->capacity
    };
    [encoder setComputePipelineState:p->rms_f16];
    [encoder setBuffer:r->p_hidden_half offset:0 atIndex:0];
    [encoder setBuffer:layer->constants
                 offset:h->input_norm_constants_index * sizeof(float)
                atIndex:1];
    [encoder setBuffer:r->p_normalized offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(p->batch, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    encode_prefill_gemm_f16(r, p, encoder, r->p_normalized,
                            layer->input_quants, layer->input_metadata,
                            r->p_projected, 14336, 80);
    [encoder setComputePipelineState:p->attention_query];
    [encoder setBuffer:r->p_projected offset:0 atIndex:0];
    [encoder setBuffer:layer->constants
                 offset:h->q_norm_constants_index * sizeof(float) atIndex:1];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
    [encoder setBuffer:r->p_query offset:0 atIndex:3];
    [encoder setBuffer:r->p_query_gate offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(24, p->batch, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder setComputePipelineState:p->attention_key_value];
    [encoder setBuffer:r->p_projected offset:0 atIndex:0];
    [encoder setBuffer:layer->constants
                 offset:h->k_norm_constants_index * sizeof(float) atIndex:1];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
    [encoder setBuffer:layer->key_cache offset:0 atIndex:3];
    [encoder setBuffer:layer->value_cache offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(4, p->batch, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder setComputePipelineState:p->attention_scores];
    [encoder setBuffer:r->p_query offset:0 atIndex:0];
    [encoder setBuffer:layer->key_cache offset:0 atIndex:1];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:2];
    [encoder setBuffer:r->p_scores offset:0 atIndex:3];
    NSUInteger score_count = 24 * (start_position + p->batch);
    [encoder dispatchThreadgroups:
        MTLSizeMake((score_count + 7) / 8, p->batch, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder setComputePipelineState:p->attention_softmax_value];
    [encoder setBuffer:r->p_scores offset:0 atIndex:0];
    [encoder setBuffer:layer->value_cache offset:0 atIndex:1];
    [encoder setBuffer:r->p_query_gate offset:0 atIndex:2];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder setBuffer:r->p_gated offset:0 atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake(24, p->batch, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    encode_prefill_gemm_residual(r, p, encoder, r->p_gated,
                                 layer->output_quants,
                                 layer->output_metadata,
                                 r->p_hidden_half, r->p_mixer_output,
                                 5120, 96, YES);
    [encoder setComputePipelineState:p->rms_f32];
    [encoder setBuffer:r->p_mixer_output offset:0 atIndex:0];
    [encoder setBuffer:layer->constants
                 offset:h->post_norm_constants_index * sizeof(float)
                atIndex:1];
    [encoder setBuffer:r->p_post_normalized offset:0 atIndex:2];
    [encoder dispatchThreadgroups:MTLSizeMake(p->batch, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    encode_prefill_mlp(r, p, encoder, layer);
}

/* Kernel-class GPU profiling (QWEN36_PROFILE=1, =2 adds per-layer lines).
 * Apple GPUs sample counters only at encoder boundaries, so the profile
 * mode splits the one-command-buffer graph into one command buffer per
 * region on the same serial queue — execution order and results are
 * unchanged — and attributes each region's GPUStartTime..GPUEndTime span.
 * Sub-millisecond dispatch gaps between buffers are excluded, so class
 * sums slightly undercount wall time. */
static int decode_profile_level(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *value = getenv("QWEN36_PROFILE");
        cached = value == NULL ? 0 : atoi(value);
    }
    return cached;
}

typedef struct {
    NSMutableArray *commands;
    NSMutableArray *tags;
} Q36Profile;

enum { Q36_PROFILE_TAG_SNAPSHOT = -2, Q36_PROFILE_TAG_EMBED = -1,
       Q36_PROFILE_TAG_HEAD = 64 };

static void profile_split(Q36DecodeRuntime *r, Q36Profile *profile, int tag,
                          id<MTLCommandBuffer> *command,
                          id<MTLComputeCommandEncoder> *encoder) {
    [*encoder endEncoding];
    [*command commit];
    [profile->commands addObject:*command];
    [profile->tags addObject:@(tag)];
    *command = [r->queue commandBuffer];
    *encoder = [*command computeCommandEncoder];
}

/* The final command buffer must be complete before calling. */
static void profile_report(Q36DecodeRuntime *r, Q36Profile *profile,
                           const char *label) {
    double snapshot = 0, embed = 0, delta = 0, attention = 0, head = 0;
    double per_layer[64] = {0};
    for (NSUInteger index = 0; index < profile->commands.count; ++index) {
        id<MTLCommandBuffer> command = profile->commands[index];
        int tag = [profile->tags[index] intValue];
        double ms = (command.GPUEndTime - command.GPUStartTime) * 1000.0;
        if (tag == Q36_PROFILE_TAG_SNAPSHOT) snapshot += ms;
        else if (tag == Q36_PROFILE_TAG_EMBED) embed += ms;
        else if (tag == Q36_PROFILE_TAG_HEAD) head += ms;
        else {
            per_layer[tag] += ms;
            Q36DecodeLayer *layer = r->layers[(NSUInteger)tag];
            if (layer->attention) attention += ms;
            else delta += ms;
        }
    }
    fprintf(stderr,
            "[profile %s] gpu %.2f ms: snapshot %.2f, embed %.2f, "
            "delta48 %.2f (avg %.3f), attn16 %.2f (avg %.3f), head %.2f\n",
            label, snapshot + embed + delta + attention + head, snapshot,
            embed, delta, delta / 48.0, attention, attention / 16.0, head);
    if (decode_profile_level() >= 2)
        for (int index = 0; index < 64; ++index) {
            Q36DecodeLayer *layer = r->layers[(NSUInteger)index];
            fprintf(stderr, "[profile %s] layer %02d %s %.3f ms\n", label,
                    index, layer->attention ? "attn " : "delta",
                    per_layer[index]);
        }
}

static void encode_prefill_chunk(Q36DecodeRuntime *r,
                                 Q36PrefillPipelines *p,
                                 id<MTLCommandBuffer> *command,
                                 id<MTLComputeCommandEncoder> *encoder,
                                 uint32_t start_position,
                                 Q36Profile *profile) {
    [*encoder setComputePipelineState:p->embedding];
    [*encoder setBuffer:r->global->embedding_quants offset:0 atIndex:0];
    [*encoder setBuffer:r->global->embedding_metadata offset:0 atIndex:1];
    [*encoder setBuffer:r->p_token_ids offset:0 atIndex:2];
    [*encoder setBuffer:r->p_hidden_half offset:0 atIndex:3];
    [*encoder dispatchThreads:MTLSizeMake(5120, p->batch, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (profile != NULL)
        profile_split(r, profile, Q36_PROFILE_TAG_EMBED, command, encoder);
    for (unsigned index = 0; index < 64; ++index) {
        Q36DecodeLayer *layer = r->layers[index];
        if (layer->attention)
            encode_prefill_attention(r, p, *encoder, layer, start_position);
        else
            encode_prefill_delta(r, p, *encoder, layer);
        if (index != 63) {
            [*encoder setComputePipelineState:p->convert_hidden];
            [*encoder setBuffer:r->p_layer_output offset:0 atIndex:0];
            [*encoder setBuffer:r->p_hidden_half offset:0 atIndex:1];
            [*encoder dispatchThreads:MTLSizeMake(5120, p->batch, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        if (profile != NULL)
            profile_split(r, profile, (int)index, command, encoder);
    }
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
    r->prefill16 = prefill_pipelines(r, 16, message);
    if (r->prefill16 == nil) return -1;
    r->prefill32 = prefill_pipelines(r, 32, message);
    if (r->prefill32 == nil) return -1;
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
    ALLOC(p_token_ids, Q36_PREFILL_MAX_BATCH, uint32_t);
    ALLOC(p_hidden_half, Q36_PREFILL_MAX_BATCH * 5120, uint16_t);
    ALLOC(p_normalized, Q36_PREFILL_MAX_BATCH * 5120, uint16_t);
    ALLOC(p_projected, Q36_PREFILL_MAX_BATCH * 16480, float);
    ALLOC(p_convolved, Q36_PREFILL_MAX_BATCH * 10240, float);
    ALLOC(p_query, Q36_PREFILL_MAX_BATCH * 6144, float);
    ALLOC(p_query_gate, Q36_PREFILL_MAX_BATCH * 6144, float);
    ALLOC(p_key, Q36_PREFILL_MAX_BATCH * 6144, float);
    ALLOC(p_value, Q36_PREFILL_MAX_BATCH * 6144, float);
    ALLOC(p_decay, Q36_PREFILL_MAX_BATCH * 48, float);
    ALLOC(p_beta, Q36_PREFILL_MAX_BATCH * 48, float);
    ALLOC(p_core, Q36_PREFILL_MAX_BATCH * 6144, float);
    ALLOC(p_gated, Q36_PREFILL_MAX_BATCH * 6144, float);
    ALLOC(p_scores, (size_t)Q36_PREFILL_MAX_BATCH * 24 * r->capacity,
          float);
    ALLOC(p_mixer_output, Q36_PREFILL_MAX_BATCH * 5120, float);
    ALLOC(p_post_normalized, Q36_PREFILL_MAX_BATCH * 5120, uint16_t);
    ALLOC(p_mlp_gate, Q36_PREFILL_MAX_BATCH * 17408, float);
    ALLOC(p_mlp_up, Q36_PREFILL_MAX_BATCH * 17408, float);
    ALLOC(p_mlp_activated, Q36_PREFILL_MAX_BATCH * 17408, float);
    ALLOC(p_layer_output, Q36_PREFILL_MAX_BATCH * 5120, float);
    ALLOC(p_x_half, Q36_PREFILL_MAX_BATCH * 17408, uint16_t);
#undef ALLOC
    if (r->hidden_half == nil || r->normalized == nil ||
        r->projected == nil || r->convolved == nil || r->query == nil ||
        r->query_gate == nil || r->key == nil || r->value == nil ||
        r->decay == nil || r->beta == nil || r->core == nil ||
        r->gated == nil || r->attention_scores_buffer == nil ||
        r->mixer_output == nil || r->post_normalized == nil ||
        r->intermediate == nil || r->layer_output == nil ||
        r->final_normalized == nil || r->logits == nil ||
        r->p_token_ids == nil || r->p_hidden_half == nil ||
        r->p_normalized == nil || r->p_projected == nil ||
        r->p_convolved == nil || r->p_query == nil ||
        r->p_query_gate == nil || r->p_key == nil || r->p_value == nil ||
        r->p_decay == nil || r->p_beta == nil || r->p_core == nil ||
        r->p_gated == nil || r->p_scores == nil ||
        r->p_mixer_output == nil || r->p_post_normalized == nil ||
        r->p_mlp_gate == nil || r->p_mlp_up == nil ||
        r->p_mlp_activated == nil || r->p_layer_output == nil) {
        if (message != NULL) *message = @"cannot allocate decode workspace";
        return -1;
    }
    return 0;
}

/* Ask Metal to wire the mapped weight buffers up front instead of paying
 * first-use residency cost inside the first prefill chunk. Gated by
 * QWEN36_RESIDENCY=0 for controlled comparison. */
static void request_weight_residency(Q36DecodeRuntime *r) {
    const char *env = getenv("QWEN36_RESIDENCY");
    if (env != NULL && strcmp(env, "0") == 0) return;
    if (@available(macOS 15.0, *)) {
        MTLResidencySetDescriptor *descriptor =
            [MTLResidencySetDescriptor new];
        descriptor.initialCapacity = 644;
        NSError *error = nil;
        id<MTLResidencySet> set =
            [r->device newResidencySetWithDescriptor:descriptor
                                               error:&error];
        if (set == nil) return;
        [set addAllocation:r->global->embedding_quants];
        [set addAllocation:r->global->embedding_metadata];
        [set addAllocation:r->global->lm_head_quants];
        [set addAllocation:r->global->lm_head_metadata];
        for (Q36DecodeLayer *layer in r->layers) {
            [set addAllocation:layer->gate_quants];
            [set addAllocation:layer->gate_metadata];
            [set addAllocation:layer->up_quants];
            [set addAllocation:layer->up_metadata];
            [set addAllocation:layer->down_quants];
            [set addAllocation:layer->down_metadata];
            [set addAllocation:layer->input_quants];
            [set addAllocation:layer->input_metadata];
            [set addAllocation:layer->output_quants];
            [set addAllocation:layer->output_metadata];
        }
        [set commit];
        [set requestResidency];
        [r->queue addResidencySet:set];
        r->residency = set;
    }
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
        request_weight_residency(r);
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
    if (model->pending_command != NULL) {
        id<MTLCommandBuffer> pending =
            (__bridge id<MTLCommandBuffer>)model->pending_command;
        [pending waitUntilCompleted];
        CFBridgingRelease(model->pending_command);
        model->pending_command = NULL;
    }
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
    model->mtp_adaptive_depth = 1;
    model->mtp_accept_ema = 0.85f;
    if (r->mtp_layer != nil) {
        memset(r->mtp_layer->key_cache.contents, 0,
               r->mtp_layer->key_cache.length);
        memset(r->mtp_layer->value_cache.contents, 0,
               r->mtp_layer->value_cache.length);
    }
    model->mtp_hidden_source = 0;
}

int qwen36_m3_model_forward_submit(
    qwen36_m3_model *model, uint32_t token_id, uint32_t position,
    char *error_message, size_t error_message_capacity) {
    if (model == NULL || model->runtime == NULL ||
        token_id >= QWEN36_VOCAB_SIZE) {
        decode_error(error_message, error_message_capacity,
                     @"invalid decode submit arguments");
        return 1;
    }
    if (model->pending_command != NULL) {
        decode_error(error_message, error_message_capacity,
                     @"a decode forward is already in flight");
        return 1;
    }
    Q36DecodeRuntime *r = (__bridge Q36DecodeRuntime *)model->runtime;
    if (position >= r->capacity) {
        decode_error(error_message, error_message_capacity,
                     @"position exceeds configured context capacity");
        return 1;
    }
    @autoreleasepool {
        Q36Profile profile_storage;
        Q36Profile *profile = NULL;
        if (decode_profile_level() > 0) {
            profile_storage.commands = [NSMutableArray array];
            profile_storage.tags = [NSMutableArray array];
            profile = &profile_storage;
        }
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
        if (profile != NULL)
            profile_split(r, profile, Q36_PROFILE_TAG_EMBED, &command,
                          &encoder);
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
            if (profile != NULL)
                profile_split(r, profile, (int)index, &command, &encoder);
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
        model->pending_token = token_id;
        model->pending_position = position;
        model->pending_start = decode_seconds();
        model->pending_command = (__bridge_retained void *)command;
        [command commit];
        if (profile != NULL) {
            [command waitUntilCompleted];
            [profile->commands addObject:command];
            [profile->tags addObject:@(Q36_PROFILE_TAG_HEAD)];
            profile_report(r, profile, "fwd1");
        }
        return 0;
    }
}

int qwen36_m3_model_forward_wait(
    qwen36_m3_model *model, qwen36_m3_decode_result *result,
    const float **logits, size_t *logit_count, char *error_message,
    size_t error_message_capacity) {
    if (model == NULL || model->runtime == NULL || result == NULL ||
        logits == NULL || logit_count == NULL ||
        model->pending_command == NULL) {
        decode_error(error_message, error_message_capacity,
                     @"invalid decode wait arguments or no forward in flight");
        return 1;
    }
    Q36DecodeRuntime *r = (__bridge Q36DecodeRuntime *)model->runtime;
    @autoreleasepool {
        id<MTLCommandBuffer> command =
            (__bridge id<MTLCommandBuffer>)model->pending_command;
        [command waitUntilCompleted];
        double duration = decode_seconds() - model->pending_start;
        MTLCommandBufferStatus status = command.status;
        NSString *message = command.error != nil ?
            command.error.localizedDescription : @"Metal decode failed";
        CFBridgingRelease(model->pending_command);
        model->pending_command = NULL;
        if (status != MTLCommandBufferStatusCompleted) {
            decode_error(error_message, error_message_capacity, message);
            return 2;
        }
        memset(result, 0, sizeof(*result));
        result->input_token = model->pending_token;
        result->next_token = 0;
        result->position = model->pending_position;
        result->duration_ms = duration * 1000.0;
        result->mapped_weight_bytes = r->mapped_bytes;
        result->state_bytes = r->state_bytes;
        result->kv_cache_bytes = r->kv_bytes;
        result->physical_footprint_bytes = decode_footprint();
        *logits = r->logits.contents;
        *logit_count = QWEN36_VOCAB_SIZE;
        model->mtp_hidden_source = 0;
        return 0;
    }
}

int qwen36_m3_model_forward(
    qwen36_m3_model *model, uint32_t token_id, uint32_t position,
    qwen36_m3_decode_result *result, const float **logits,
    size_t *logit_count, char *error_message, size_t error_message_capacity) {
    int status = qwen36_m3_model_forward_submit(
        model, token_id, position, error_message, error_message_capacity);
    if (status != 0) return status;
    return qwen36_m3_model_forward_wait(
        model, result, logits, logit_count, error_message,
        error_message_capacity);
}

static Q36PrefillPipelines *prefill_set_for_batch(
    Q36DecodeRuntime *r, uint32_t batch) {
    switch (batch) {
    case 1: return r->prefill1;
    case 2: return r->prefill2;
    case 3: return r->prefill3;
    case 4: return r->prefill4;
    case 16: return r->prefill16;
    default: return r->prefill32;
    }
}

/* Encode one MTP pass: fuse(embed(token), hidden) -> fc -> the MTP
 * transformer layer at rope position start_j (batched: start_j + s). The
 * hidden input is the target model's POST-final-norm hidden state (the
 * value its lm_head consumes), matching the reference implementation.
 * The pass writes the MTP layer's KV cache as its side effect. with_head
 * adds the final MTP norm and the shared language-model head into
 * mtp_logits (single-token drafting only); normalize_hidden first applies
 * the global final norm over p_layer_output rows (chunk-fill path). */
static void encode_mtp_pass(Q36DecodeRuntime *r,
                            id<MTLComputeCommandEncoder> encoder,
                            uint32_t batch, uint32_t start_j,
                            id<MTLBuffer> hidden, NSUInteger hidden_offset,
                            int with_head, int normalize_hidden) {
    Q36PrefillPipelines *p = prefill_set_for_batch(r, batch);
    if (normalize_hidden) {
        [encoder setComputePipelineState:p->rms_f32];
        [encoder setBuffer:r->p_layer_output offset:0 atIndex:0];
        [encoder setBuffer:r->global->constants offset:0 atIndex:1];
        [encoder setBuffer:r->p_post_normalized offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(batch, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        hidden = r->p_post_normalized;
        hidden_offset = 0;
    }
    [encoder setComputePipelineState:p->mtp_fuse];
    [encoder setBuffer:r->global->embedding_quants offset:0 atIndex:0];
    [encoder setBuffer:r->global->embedding_metadata offset:0 atIndex:1];
    [encoder setBuffer:r->p_token_ids offset:0 atIndex:2];
    [encoder setBuffer:hidden offset:hidden_offset atIndex:3];
    [encoder setBuffer:r->mtp_constants
                 offset:r->mtp_header->embedding_norm_constants_index *
                        sizeof(float)
                atIndex:4];
    [encoder setBuffer:r->mtp_constants
                 offset:r->mtp_header->hidden_norm_constants_index *
                        sizeof(float)
                atIndex:5];
    [encoder setBuffer:r->mtp_fused offset:0 atIndex:6];
    [encoder dispatchThreadgroups:MTLSizeMake(batch, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    if (batch == 1) {
        encode_prefill_gemm_f16(r, p, encoder, r->mtp_fused,
                                r->mtp_fc_quants, r->mtp_fc_metadata,
                                r->mixer_output, 5120,
                                r->mtp_header->fc_groups_per_row);
        [encoder setComputePipelineState:r->convert_hidden];
        [encoder setBuffer:r->mixer_output offset:0 atIndex:0];
        [encoder setBuffer:r->hidden_half offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(5120, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        encode_attention(r, encoder, r->mtp_layer, start_j);
        if (with_head) {
            [encoder setComputePipelineState:r->rms_float];
            [encoder setBuffer:r->layer_output offset:0 atIndex:0];
            [encoder setBuffer:r->mtp_constants
                         offset:r->mtp_header->final_norm_constants_index *
                                sizeof(float)
                        atIndex:1];
            [encoder setBuffer:r->final_normalized offset:0 atIndex:2];
            [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [encoder setComputePipelineState:r->lm_head];
            [encoder setBuffer:r->final_normalized offset:0 atIndex:0];
            [encoder setBuffer:r->global->lm_head_quants offset:0
                    atIndex:1];
            [encoder setBuffer:r->global->lm_head_metadata offset:0
                    atIndex:2];
            [encoder setBuffer:r->mtp_logits offset:0 atIndex:3];
            [encoder dispatchThreadgroups:
                MTLSizeMake((QWEN36_VOCAB_SIZE + 7) / 8, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
    } else {
        encode_prefill_gemm_f16(r, p, encoder, r->mtp_fused,
                                r->mtp_fc_quants, r->mtp_fc_metadata,
                                r->p_mixer_output, 5120,
                                r->mtp_header->fc_groups_per_row);
        [encoder setComputePipelineState:p->convert_hidden];
        [encoder setBuffer:r->p_mixer_output offset:0 atIndex:0];
        [encoder setBuffer:r->p_hidden_half offset:0 atIndex:1];
        [encoder dispatchThreads:MTLSizeMake(5120, batch, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        encode_prefill_attention(r, p, encoder, r->mtp_layer, start_j);
    }
}

/* Run one synchronous MTP pass; batch 1 with with_head produces the draft
 * token in *draft via a CPU argmax over the full masked head. */
static int run_mtp_pass(qwen36_m3_model *model, uint32_t batch,
                        const uint32_t *tokens, uint32_t start_j,
                        id<MTLBuffer> hidden, NSUInteger hidden_offset,
                        int with_head, int normalize_hidden,
                        uint32_t *draft, char *error_message,
                        size_t error_message_capacity) {
    Q36DecodeRuntime *r = (__bridge Q36DecodeRuntime *)model->runtime;
    MTLCommandBufferStatus status;
    NSString *failure = nil;
    @autoreleasepool {
        memcpy(r->p_token_ids.contents, tokens,
               (size_t)batch * sizeof(uint32_t));
        id<MTLCommandBuffer> command = [r->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        encode_mtp_pass(r, encoder, batch, start_j, hidden, hidden_offset,
                        with_head, normalize_hidden);
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        status = command.status;
        if (command.error != nil)
            failure = command.error.localizedDescription;
    }
    if (status != MTLCommandBufferStatusCompleted) {
        decode_error(error_message, error_message_capacity,
            failure != nil ? failure : @"Metal MTP pass failed");
        return 2;
    }
    if (with_head && draft != NULL) {
        const float *logits = r->mtp_logits.contents;
        uint32_t best = 0;
        float best_value = logits[0];
        for (uint32_t index = 1; index < QWEN36_VOCAB_SIZE; ++index) {
            if (logits[index] > best_value) {
                best_value = logits[index];
                best = index;
            }
        }
        *draft = best;
    }
    return 0;
}

/* Batched forward with logits at every position: the verification step of
 * speculative decoding. Layer states advance by batch positions. */
static int model_forward_batch(qwen36_m3_model *model,
                               const uint32_t *tokens, uint32_t batch,
                               uint32_t position, int snapshot_first,
                               char *error_message,
                               size_t error_message_capacity) {
    Q36DecodeRuntime *r = (__bridge Q36DecodeRuntime *)model->runtime;
    if ((uint64_t)position + batch > r->capacity) {
        decode_error(error_message, error_message_capacity,
                     @"verification exceeds context capacity");
        return 1;
    }
    MTLCommandBufferStatus status;
    NSString *failure = nil;
    @autoreleasepool {
        Q36Profile profile_storage;
        Q36Profile *profile = NULL;
        if (decode_profile_level() > 0) {
            profile_storage.commands = [NSMutableArray array];
            profile_storage.tags = [NSMutableArray array];
            profile = &profile_storage;
        }
        Q36PrefillPipelines *pipelines = prefill_set_for_batch(r, batch);
        memcpy(r->p_token_ids.contents, tokens,
               (size_t)batch * sizeof(uint32_t));
        id<MTLCommandBuffer> command = [r->queue commandBuffer];
        if (snapshot_first) {
            id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
            NSUInteger recurrent_offset = 0;
            NSUInteger convolution_offset = 0;
            for (Q36DecodeLayer *layer in r->layers) {
                if (layer->attention) continue;
                [blit copyFromBuffer:layer->recurrent_state
                        sourceOffset:0
                            toBuffer:r->snapshot_recurrent
                   destinationOffset:recurrent_offset
                                size:layer->recurrent_state.length];
                [blit copyFromBuffer:layer->convolution_state
                        sourceOffset:0
                            toBuffer:r->snapshot_convolution
                   destinationOffset:convolution_offset
                                size:layer->convolution_state.length];
                recurrent_offset += layer->recurrent_state.length;
                convolution_offset += layer->convolution_state.length;
            }
            [blit endEncoding];
            if (profile != NULL) {
                [command commit];
                [profile->commands addObject:command];
                [profile->tags addObject:@(Q36_PROFILE_TAG_SNAPSHOT)];
                command = [r->queue commandBuffer];
            }
        }
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        encode_prefill_chunk(r, pipelines, &command, &encoder, position,
                             profile);
        [encoder setComputePipelineState:pipelines->rms_f32];
        [encoder setBuffer:r->p_layer_output offset:0 atIndex:0];
        [encoder setBuffer:r->global->constants offset:0 atIndex:1];
        [encoder setBuffer:r->p_post_normalized offset:0 atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(batch, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        encode_prefill_gemm_f16(r, pipelines, encoder,
                                r->p_post_normalized,
                                r->global->lm_head_quants,
                                r->global->lm_head_metadata,
                                r->p_logits2, QWEN36_VOCAB_SIZE, 80);
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        status = command.status;
        if (command.error != nil)
            failure = command.error.localizedDescription;
        if (profile != NULL && status == MTLCommandBufferStatusCompleted) {
            [profile->commands addObject:command];
            [profile->tags addObject:@(Q36_PROFILE_TAG_HEAD)];
            profile_report(r, profile, "verify");
        }
    }
    if (status != MTLCommandBufferStatusCompleted) {
        decode_error(error_message, error_message_capacity,
            failure != nil ? failure : @"Metal verification failed");
        return 2;
    }
    return 0;
}

static void gdn_state_copy(Q36DecodeRuntime *r, int restore) {
    @autoreleasepool {
        id<MTLCommandBuffer> command = [r->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        NSUInteger recurrent_offset = 0;
        NSUInteger convolution_offset = 0;
        for (Q36DecodeLayer *layer in r->layers) {
            if (layer->attention) continue;
            if (restore) {
                [blit copyFromBuffer:r->snapshot_recurrent
                        sourceOffset:recurrent_offset
                            toBuffer:layer->recurrent_state
                   destinationOffset:0
                                size:layer->recurrent_state.length];
                [blit copyFromBuffer:r->snapshot_convolution
                        sourceOffset:convolution_offset
                            toBuffer:layer->convolution_state
                   destinationOffset:0
                                size:layer->convolution_state.length];
            } else {
                [blit copyFromBuffer:layer->recurrent_state
                        sourceOffset:0
                            toBuffer:r->snapshot_recurrent
                   destinationOffset:recurrent_offset
                                size:layer->recurrent_state.length];
                [blit copyFromBuffer:layer->convolution_state
                        sourceOffset:0
                            toBuffer:r->snapshot_convolution
                   destinationOffset:convolution_offset
                                size:layer->convolution_state.length];
            }
            recurrent_offset += layer->recurrent_state.length;
            convolution_offset += layer->convolution_state.length;
        }
        [blit endEncoding];
        [command commit];
        [command waitUntilCompleted];
    }
}

static uint32_t argmax_f32(const float *values, uint32_t count) {
    uint32_t best = 0;
    for (uint32_t index = 1; index < count; ++index)
        if (values[index] > values[best]) best = index;
    return best;
}

int qwen36_m3_model_mtp_open(
    qwen36_m3_model *model, const char *layer_image_path,
    const char *extras_image_path, char *error_message,
    size_t error_message_capacity) {
    if (model == NULL || model->runtime == NULL ||
        layer_image_path == NULL || extras_image_path == NULL) {
        decode_error(error_message, error_message_capacity,
                     @"invalid MTP open arguments");
        return 1;
    }
    Q36DecodeRuntime *r = (__bridge Q36DecodeRuntime *)model->runtime;
    if (r->mtp_layer != nil) return 0;
    @autoreleasepool {
        NSString *message = nil;
        Q36DecodeLayer *layer = load_attention_layer(
            r, [NSString stringWithUTF8String:layer_image_path],
            QWEN36_M3_MTP_LAYER_INDEX, &message);
        if (layer == nil) {
            decode_error(error_message, error_message_capacity, message);
            return 3;
        }
        int file = -1;
        void *mapping = MAP_FAILED;
        size_t length = 0;
        if (map_file(extras_image_path, &file, &mapping, &length,
                     &message) != 0) {
            decode_error(error_message, error_message_capacity, message);
            return 3;
        }
        const qwen36_m3_mtp_image_header *h = mapping;
        if (memcmp(h->magic, QWEN36_M3_MTP_IMAGE_MAGIC, 8) != 0 ||
            h->version != QWEN36_M3_MTP_IMAGE_VERSION ||
            h->header_bytes != QWEN36_M3_MTP_HEADER_BYTES ||
            h->hidden_size != 5120 || h->fc_rows != 5120 ||
            h->fc_groups_per_row != 160 || h->group_size != 64 ||
            h->constants_f32_count != 3 * 5120 ||
            memcmp(h->source_sha256, QWEN36_M3_EXPECTED_MTP_SHA256,
                   64) != 0 ||
            h->constants_offset + h->constants_bytes != length) {
            munmap(mapping, length);
            close(file);
            decode_error(error_message, error_message_capacity,
                         @"invalid MTP extras image");
            return 3;
        }
        r->mtp_fc_quants = mapped_buffer(r->device, mapping,
                                         h->fc_quants_offset,
                                         h->fc_quants_bytes);
        r->mtp_fc_metadata = mapped_buffer(r->device, mapping,
                                           h->fc_metadata_offset,
                                           h->fc_metadata_bytes);
        r->mtp_constants = [r->device
            newBufferWithLength:h->constants_f32_count * sizeof(float)
                        options:MTLResourceStorageModeShared];
        r->mtp_fused = [r->device
            newBufferWithLength:(size_t)32 * 10240 * 2
                        options:MTLResourceStorageModeShared];
        r->mtp_logits = [r->device
            newBufferWithLength:(size_t)QWEN36_VOCAB_SIZE * sizeof(float)
                        options:MTLResourceStorageModeShared];
        /* 1..3 fixes the draft depth; 0 (the default) adapts it per
         * step: a fully accepted step deepens the next draft chain, any
         * reject resets it to one. */
        const char *depth_env = getenv("QWEN36_MTP_DEPTH");
        int depth_value = depth_env != NULL ? atoi(depth_env) : 0;
        if (depth_value < 0) depth_value = 0;
        if (depth_value > 3) depth_value = 3;
        r->mtp_depth = (uint32_t)depth_value;
        uint32_t max_depth = r->mtp_depth == 0 ? 3 : r->mtp_depth;
        r->p_logits2 = [r->device
            newBufferWithLength:(size_t)(max_depth + 1) *
                                QWEN36_VOCAB_SIZE * sizeof(float)
                        options:MTLResourceStorageModeShared];
        size_t recurrent_total = 0;
        size_t convolution_total = 0;
        for (Q36DecodeLayer *existing in r->layers) {
            if (existing->attention) continue;
            recurrent_total += existing->recurrent_state.length;
            convolution_total += existing->convolution_state.length;
        }
        r->snapshot_recurrent = [r->device
            newBufferWithLength:recurrent_total
                        options:MTLResourceStorageModeShared];
        r->snapshot_convolution = [r->device
            newBufferWithLength:convolution_total
                        options:MTLResourceStorageModeShared];
        NSString *pipeline_message = nil;
        if (r->prefill1 == nil)
            r->prefill1 = prefill_pipelines(r, 1, &pipeline_message);
        if (r->prefill2 == nil)
            r->prefill2 = prefill_pipelines(r, 2, &pipeline_message);
        int deep = r->mtp_depth == 0 || r->mtp_depth >= 2;
        if (deep && r->prefill3 == nil)
            r->prefill3 = prefill_pipelines(r, 3, &pipeline_message);
        if ((r->mtp_depth == 0 || r->mtp_depth >= 3) &&
            r->prefill4 == nil)
            r->prefill4 = prefill_pipelines(r, 4, &pipeline_message);
        if (r->mtp_fc_quants == nil || r->mtp_fc_metadata == nil ||
            r->mtp_constants == nil || r->mtp_fused == nil ||
            r->mtp_logits == nil || r->p_logits2 == nil ||
            r->snapshot_recurrent == nil ||
            r->snapshot_convolution == nil || r->prefill1 == nil ||
            r->prefill2 == nil || (deep && r->prefill3 == nil) ||
            ((r->mtp_depth == 0 || r->mtp_depth >= 3) &&
             r->prefill4 == nil)) {
            munmap(mapping, length);
            close(file);
            decode_error(error_message, error_message_capacity,
                pipeline_message != nil ? pipeline_message :
                @"cannot allocate MTP runtime state");
            return 3;
        }
        memcpy(r->mtp_constants.contents,
               (unsigned char *)mapping + h->constants_offset,
               h->constants_f32_count * sizeof(float));
        r->mtp_file = file;
        r->mtp_mapping = mapping;
        r->mtp_mapping_length = length;
        r->mtp_header = h;
        r->mtp_layer = layer;
        r->mapped_bytes += layer->length + length;
        return 0;
    }
}

int qwen36_m3_model_mtp_step(
    qwen36_m3_model *model, uint32_t *current_token, uint32_t *position,
    uint32_t emitted[8], uint32_t *emitted_count, int *accepted,
    char *error_message, size_t error_message_capacity) {
    if (model == NULL || model->runtime == NULL ||
        current_token == NULL || position == NULL || emitted == NULL ||
        emitted_count == NULL || accepted == NULL) {
        decode_error(error_message, error_message_capacity,
                     @"invalid MTP step arguments");
        return 1;
    }
    Q36DecodeRuntime *r = (__bridge Q36DecodeRuntime *)model->runtime;
    if (r->mtp_layer == nil) {
        decode_error(error_message, error_message_capacity,
                     @"MTP images are not loaded");
        return 1;
    }
    uint32_t j = *position;
    uint32_t depth = r->mtp_depth;
    if (depth == 0) {
        /* Adaptive: a running per-draft acceptance estimate picks the
         * chain length. Around 0.83 (free prose) a deeper chain loses
         * to the extra verify rows and replays, so the thresholds sit
         * where measured break-even is. */
        float ema = model->mtp_accept_ema;
        depth = ema > 0.95f ? 3 : (ema > 0.90f ? 2 : 1);
    }
    while (depth > 1 && (uint64_t)j + depth + 1 > r->capacity) --depth;
    id<MTLBuffer> hidden = model->mtp_hidden_source == 0 ?
        r->final_normalized : r->p_post_normalized;
    NSUInteger hidden_offset = model->mtp_hidden_source >= 1 ?
        (NSUInteger)(model->mtp_hidden_source - 1) * 5120 *
            sizeof(uint16_t) : 0;

    uint32_t tokens[8];
    tokens[0] = *current_token;
    double phase0 = decode_seconds();
    /* Chained drafting: step 1 consumes the target model's
     * post-final-norm hidden; later steps consume the previous MTP
     * pass's own post-norm hidden, which the with_head encode leaves in
     * final_normalized (the reference implementation returns
     * self.norm(hidden) as the next spec step's hidden). */
    for (uint32_t i = 0; i < depth; ++i) {
        uint32_t draft = 0;
        int draft_status = run_mtp_pass(
            model, 1, &tokens[i], j + i,
            i == 0 ? hidden : r->final_normalized,
            i == 0 ? hidden_offset : 0, 1, 0, &draft, error_message,
            error_message_capacity);
        if (draft_status != 0) return draft_status;
        tokens[i + 1] = draft;
    }
    double phase1 = decode_seconds();
    int status = model_forward_batch(model, tokens, depth + 1, j, 1,
                                     error_message,
                                     error_message_capacity);
    if (status != 0) return status;
    double phase2 = decode_seconds();
    if (getenv("QWEN36_MTP_DEBUG") != NULL)
        fprintf(stderr, "[mtp-time] depth %u draft %.1f ms, "
                "snapshot+verify %.1f ms\n", depth,
                (phase1 - phase0) * 1000.0, (phase2 - phase1) * 1000.0);

    const float *logits = r->p_logits2.contents;
    uint32_t true_next[8];
    for (uint32_t i = 0; i <= depth; ++i)
        true_next[i] = argmax_f32(
            logits + (size_t)i * QWEN36_VOCAB_SIZE, QWEN36_VOCAB_SIZE);
    uint32_t accepted_count = 0;
    while (accepted_count < depth &&
           tokens[accepted_count + 1] == true_next[accepted_count])
        ++accepted_count;

    uint32_t emit_count;
    uint32_t next_pending;
    if (accepted_count == depth) {
        emit_count = depth + 1;
        next_pending = true_next[depth];
    } else {
        /* Restore the pre-verify GDN state, then re-verify the accepted
         * prefix plus the corrected token; that both repairs the layer
         * state and advances it one position past the correction. The
         * attention KV rows are simply overwritten. */
        gdn_state_copy(r, 1);
        uint32_t replay_tokens[8];
        for (uint32_t i = 0; i <= accepted_count; ++i)
            replay_tokens[i] = tokens[i];
        replay_tokens[accepted_count + 1] = true_next[accepted_count];
        status = model_forward_batch(model, replay_tokens,
                                     accepted_count + 2, j, 0,
                                     error_message,
                                     error_message_capacity);
        if (status != 0) return status;
        logits = r->p_logits2.contents;
        emit_count = accepted_count + 2;
        next_pending = argmax_f32(
            logits + (size_t)(accepted_count + 1) * QWEN36_VOCAB_SIZE,
            QWEN36_VOCAB_SIZE);
    }
    *accepted = (int)accepted_count;

    emitted[0] = tokens[0];
    for (uint32_t i = 1; i < emit_count; ++i)
        emitted[i] = true_next[i - 1];
    *emitted_count = emit_count;

    /* Refresh the draft cache for every confirmed successor with the
     * true post-final-norm hidden rows the last forward produced; the
     * rows written speculatively during chained drafting are among the
     * ones overwritten. */
    status = run_mtp_pass(model, emit_count - 1, emitted + 1, j + 1,
                          r->p_post_normalized, 0, 0, 0, NULL,
                          error_message, error_message_capacity);
    if (status != 0) return status;

    *current_token = next_pending;
    *position = j + emit_count;
    model->mtp_hidden_source = (int)emit_count;
    if (r->mtp_depth == 0)
        for (uint32_t i = 0; i < depth; ++i)
            model->mtp_accept_ema = 0.85f * model->mtp_accept_ema +
                (i < accepted_count ? 0.15f : 0.0f);
    return 0;
}

int qwen36_m3_model_prefill(
    qwen36_m3_model *model, const uint32_t *token_ids,
    uint32_t token_count, uint32_t start_position,
    qwen36_m3_prefill_result *result, char *error_message,
    size_t error_message_capacity) {
    if (model == NULL || model->runtime == NULL || token_ids == NULL ||
        token_count == 0 || result == NULL) {
        decode_error(error_message, error_message_capacity,
                     @"invalid prefill arguments");
        return 1;
    }
    if (model->pending_command != NULL) {
        decode_error(error_message, error_message_capacity,
                     @"a decode forward is already in flight");
        return 1;
    }
    Q36DecodeRuntime *r = (__bridge Q36DecodeRuntime *)model->runtime;
    if ((uint64_t)start_position + token_count > r->capacity) {
        decode_error(error_message, error_message_capacity,
                     @"prefill run exceeds configured context capacity");
        return 1;
    }
    for (uint32_t index = 0; index < token_count; ++index) {
        if (token_ids[index] >= QWEN36_VOCAB_SIZE) {
            decode_error(error_message, error_message_capacity,
                         @"prefill token id exceeds vocabulary");
            return 1;
        }
    }
    memset(result, 0, sizeof(*result));
    result->token_count = token_count;
    const char *mma_env = getenv("QWEN36_PREFILL_MMA");
    r->prefill_mma_level = mma_env == NULL ? 2 : atoi(mma_env);
    double begin = decode_seconds();
    const char *max_chunk_env = getenv("QWEN36_PREFILL_MAX_CHUNK");
    uint32_t max_chunk = max_chunk_env != NULL ?
        (uint32_t)atoi(max_chunk_env) : 32;
    uint32_t offset = 0;
    while (token_count - offset >= 16) {
        uint32_t chunk = token_count - offset >= 32 && max_chunk >= 32 ?
            32 : 16;
        Q36PrefillPipelines *pipelines =
            chunk == 32 ? r->prefill32 : r->prefill16;
        double chunk_start = decode_seconds();
        MTLCommandBufferStatus status;
        NSString *failure = nil;
        @autoreleasepool {
            Q36Profile profile_storage;
            Q36Profile *profile = NULL;
            if (decode_profile_level() > 0) {
                profile_storage.commands = [NSMutableArray array];
                profile_storage.tags = [NSMutableArray array];
                profile = &profile_storage;
            }
            memcpy(r->p_token_ids.contents, token_ids + offset,
                   (size_t)chunk * sizeof(uint32_t));
            id<MTLCommandBuffer> command = [r->queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder =
                [command computeCommandEncoder];
            encode_prefill_chunk(r, pipelines, &command, &encoder,
                                 start_position + offset, profile);
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            status = command.status;
            if (command.error != nil)
                failure = command.error.localizedDescription;
            if (profile != NULL &&
                status == MTLCommandBufferStatusCompleted) {
                [profile->commands addObject:command];
                [profile->tags addObject:@(Q36_PROFILE_TAG_HEAD)];
                profile_report(r, profile,
                               chunk == 32 ? "chunk32" : "chunk16");
            }
        }
        if (status != MTLCommandBufferStatusCompleted) {
            decode_error(error_message, error_message_capacity,
                failure != nil ? failure : @"Metal prefill failed");
            return 2;
        }
        if (result->first_chunk_ms == 0.0)
            result->first_chunk_ms =
                (decode_seconds() - chunk_start) * 1000.0;
        if (r->mtp_layer != nil) {
            /* Fill the draft layer's cache for this chunk: input token
             * j pairs with the main hidden at j - 1, so the shifted ids
             * rely on the documented token_count + 1 contract. */
            int mtp_status = run_mtp_pass(
                model, chunk, token_ids + offset + 1,
                start_position + offset + 1, nil, 0, 0, 1,
                NULL, error_message, error_message_capacity);
            if (mtp_status != 0) return mtp_status;
        }
        if (chunk == 32) ++result->chunk32_count;
        else ++result->chunk16_count;
        offset += chunk;
    }
    while (offset < token_count) {
        qwen36_m3_decode_result single;
        const float *single_logits = NULL;
        size_t single_count = 0;
        double single_start = decode_seconds();
        int status = qwen36_m3_model_forward(
            model, token_ids[offset], start_position + offset, &single,
            &single_logits, &single_count, error_message,
            error_message_capacity);
        if (status != 0) return status;
        if (r->mtp_layer != nil) {
            status = run_mtp_pass(
                model, 1, token_ids + offset + 1,
                start_position + offset + 1, r->final_normalized, 0, 0,
                0, NULL, error_message, error_message_capacity);
            if (status != 0) return status;
        }
        if (result->first_chunk_ms == 0.0)
            result->first_chunk_ms =
                (decode_seconds() - single_start) * 1000.0;
        ++result->single_count;
        ++offset;
    }
    result->duration_ms = (decode_seconds() - begin) * 1000.0;
    return 0;
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

size_t qwen36_m3_model_copy_state(
    qwen36_m3_model *model, uint32_t layer_index, uint32_t kind,
    void *destination, size_t destination_capacity) {
    if (model == NULL || model->runtime == NULL || destination == NULL ||
        layer_index >= 64) return 0;
    Q36DecodeRuntime *r = (__bridge Q36DecodeRuntime *)model->runtime;
    Q36DecodeLayer *layer = r->layers[layer_index];
    id<MTLBuffer> source = nil;
    switch (kind) {
    case QWEN36_M3_STATE_RECURRENT:
        source = layer->attention ? nil : layer->recurrent_state;
        break;
    case QWEN36_M3_STATE_CONVOLUTION:
        source = layer->attention ? nil : layer->convolution_state;
        break;
    case QWEN36_M3_STATE_KEY_CACHE:
        source = layer->attention ? layer->key_cache : nil;
        break;
    case QWEN36_M3_STATE_VALUE_CACHE:
        source = layer->attention ? layer->value_cache : nil;
        break;
    default:
        break;
    }
    if (source == nil || source.length > destination_capacity) return 0;
    memcpy(destination, source.contents, source.length);
    return source.length;
}

void qwen36_m3_model_close(qwen36_m3_model *model) {
    if (model == NULL) return;
    if (model->pending_command != NULL) {
        id<MTLCommandBuffer> pending =
            (__bridge id<MTLCommandBuffer>)model->pending_command;
        [pending waitUntilCompleted];
        CFBridgingRelease(model->pending_command);
        model->pending_command = NULL;
    }
    if (model->runtime != NULL) {
        Q36DecodeRuntime *r = (__bridge Q36DecodeRuntime *)model->runtime;
        if (r->mtp_mapping != NULL && r->mtp_mapping != MAP_FAILED)
            munmap(r->mtp_mapping, r->mtp_mapping_length);
        if (r->mtp_file > 0) close(r->mtp_file);
        CFBridgingRelease(model->runtime);
        model->runtime = NULL;
    }
    free(model);
}
