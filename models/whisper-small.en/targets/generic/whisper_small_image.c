#define _POSIX_C_SOURCE 200809L

#include "whisper_small_image.h"
#include "whisper_small_frontend.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum { IMAGE_KIND_F32 = 1 };

#pragma pack(push, 1)
struct image_header {
    char magic[8];
    uint32_t version;
    uint32_t descriptor_count;
    uint32_t layer_count;
    uint32_t n_mels;
    uint32_t n_state;
    uint32_t n_heads;
    uint32_t n_mlp;
    uint32_t maximum_frames;
    uint64_t directory_offset;
    uint64_t data_offset;
    uint64_t file_bytes;
    uint64_t reserved;
};

struct descriptor {
    char name[96];
    uint32_t kind;
    uint32_t rank;
    uint32_t shape[4];
    uint64_t offset;
    uint64_t byte_count;
    uint64_t reserved;
};
#pragma pack(pop)

static double monotonic_seconds(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0.0;
    return (double)now.tv_sec + (double)now.tv_nsec * 1.0e-9;
}

static int multiply_size(size_t left, size_t right, size_t *result)
{
    if (left != 0U && right > SIZE_MAX / left) return -1;
    *result = left * right;
    return 0;
}

static const struct descriptor *find_descriptor(const cllm_whisper_small_image *image,
                                                const char *name)
{
    const struct image_header *header = image->header;
    const struct descriptor *entries = image->descriptors;
    for (uint32_t index = 0U; index < header->descriptor_count; ++index) {
        if (strncmp(entries[index].name, name, sizeof(entries[index].name)) == 0)
            return &entries[index];
    }
    return NULL;
}

static const float *bind_f32(const cllm_whisper_small_image *image,
                             const char *name,
                             uint32_t rank,
                             uint32_t d0,
                             uint32_t d1,
                             uint32_t d2)
{
    const struct descriptor *entry = find_descriptor(image, name);
    size_t count = 1U;
    size_t bytes;
    const uint32_t expected[3] = {d0, d1, d2};

    if (entry == NULL || entry->kind != IMAGE_KIND_F32 || entry->rank != rank)
        return NULL;
    for (uint32_t dimension = 0U; dimension < rank; ++dimension) {
        if (entry->shape[dimension] != expected[dimension] ||
            multiply_size(count, entry->shape[dimension], &count) != 0)
            return NULL;
    }
    if (multiply_size(count, sizeof(float), &bytes) != 0 || entry->byte_count != bytes ||
        entry->offset % _Alignof(float) != 0U)
        return NULL;
    return (const float *)(const void *)(image->base + entry->offset);
}

static int map_image(const char *path, cllm_whisper_small_image *image)
{
    struct stat status;
    const struct image_header *header;
    const struct descriptor *entries;
    size_t directory_bytes;

    memset(image, 0, sizeof(*image));
    image->fd = -1;
    image->fd = open(path, O_RDONLY);
    if (image->fd < 0 || fstat(image->fd, &status) != 0 ||
        status.st_size < (off_t)sizeof(struct image_header))
        goto fail;
    image->bytes = (size_t)status.st_size;
    image->base = mmap(NULL, image->bytes, PROT_READ, MAP_PRIVATE, image->fd, 0);
    if (image->base == MAP_FAILED) {
        image->base = NULL;
        goto fail;
    }
    header = (const struct image_header *)(const void *)image->base;
    image->header = header;
    if (memcmp(header->magic, "WHSENC01", 8U) != 0 || header->version != 1U ||
        header->layer_count != 12U || header->n_mels != 80U ||
        header->n_state != 768U || header->n_heads != 12U ||
        header->n_mlp != 3072U || header->maximum_frames != 1500U ||
        header->file_bytes != image->bytes ||
        multiply_size(header->descriptor_count, sizeof(struct descriptor), &directory_bytes) != 0 ||
        header->directory_offset > image->bytes ||
        directory_bytes > image->bytes - header->directory_offset)
        goto fail;
    entries = (const struct descriptor *)(const void *)(image->base + header->directory_offset);
    image->descriptors = entries;
    for (uint32_t index = 0U; index < header->descriptor_count; ++index) {
        if (memchr(entries[index].name, '\0', sizeof(entries[index].name)) == NULL ||
            entries[index].rank > 4U || entries[index].offset > image->bytes ||
            entries[index].byte_count > image->bytes - entries[index].offset)
            goto fail;
    }
    return 0;

fail:
    if (image->base != NULL) munmap((void *)image->base, image->bytes);
    if (image->fd >= 0) close(image->fd);
    memset(image, 0, sizeof(*image));
    image->fd = -1;
    return -1;
}

static int bind_layer(const cllm_whisper_small_image *image,
                      size_t index,
                      cllm_whisper_encoder_block_weights *weights)
{
    char prefix[64];
    char name[128];
#define BIND(field, suffix, rank, d0, d1, d2) do { \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix); \
    weights->field = bind_f32(image, name, rank, d0, d1, d2); \
    if (weights->field == NULL) return -1; \
} while (0)
    snprintf(prefix, sizeof(prefix), "encoder.layers.%zu.", index);
    memset(weights, 0, sizeof(*weights));
    weights->n_state = 768U;
    weights->n_heads = 12U;
    weights->n_mlp = 3072U;
    BIND(attention_norm_weight, "self_attn_layer_norm.weight", 1U, 768U, 0U, 0U);
    BIND(attention_norm_bias, "self_attn_layer_norm.bias", 1U, 768U, 0U, 0U);
    BIND(query_weight, "self_attn.q_proj.weight", 2U, 768U, 768U, 0U);
    BIND(query_bias, "self_attn.q_proj.bias", 1U, 768U, 0U, 0U);
    BIND(key_weight, "self_attn.k_proj.weight", 2U, 768U, 768U, 0U);
    BIND(value_weight, "self_attn.v_proj.weight", 2U, 768U, 768U, 0U);
    BIND(value_bias, "self_attn.v_proj.bias", 1U, 768U, 0U, 0U);
    BIND(attention_output_weight, "self_attn.out_proj.weight", 2U, 768U, 768U, 0U);
    BIND(attention_output_bias, "self_attn.out_proj.bias", 1U, 768U, 0U, 0U);
    BIND(mlp_norm_weight, "final_layer_norm.weight", 1U, 768U, 0U, 0U);
    BIND(mlp_norm_bias, "final_layer_norm.bias", 1U, 768U, 0U, 0U);
    BIND(mlp_input_weight, "fc1.weight", 2U, 3072U, 768U, 0U);
    BIND(mlp_input_bias, "fc1.bias", 1U, 3072U, 0U, 0U);
    BIND(mlp_output_weight, "fc2.weight", 2U, 768U, 3072U, 0U);
    BIND(mlp_output_bias, "fc2.bias", 1U, 768U, 0U, 0U);
#undef BIND
    return 0;
}

int cllm_whisper_small_model_open(const char *path, cllm_whisper_small_model *model)
{
    if (path == NULL || model == NULL) return -1;
    memset(model, 0, sizeof(*model));
    model->image.fd = -1;
    if (map_image(path, &model->image) != 0) return -1;
#define BIND_MODEL(field, name, rank, d0, d1, d2) do { \
    model->field = bind_f32(&model->image, name, rank, d0, d1, d2); \
    if (model->field == NULL) goto fail; \
} while (0)
    BIND_MODEL(mel_filters, "frontend.mel_80", 2U, 80U, 201U, 0U);
    BIND_MODEL(conv1_weight, "encoder.conv1.weight", 3U, 768U, 80U, 3U);
    BIND_MODEL(conv1_bias, "encoder.conv1.bias", 1U, 768U, 0U, 0U);
    BIND_MODEL(conv2_weight, "encoder.conv2.weight", 3U, 768U, 768U, 3U);
    BIND_MODEL(conv2_bias, "encoder.conv2.bias", 1U, 768U, 0U, 0U);
    BIND_MODEL(positions, "encoder.embed_positions.weight", 2U, 1500U, 768U, 0U);
    BIND_MODEL(final_norm_weight, "encoder.layer_norm.weight", 1U, 768U, 0U, 0U);
    BIND_MODEL(final_norm_bias, "encoder.layer_norm.bias", 1U, 768U, 0U, 0U);
#undef BIND_MODEL
    for (size_t layer = 0U; layer < 12U; ++layer)
        if (bind_layer(&model->image, layer, &model->layers[layer]) != 0) goto fail;
    return 0;
fail:
    cllm_whisper_small_model_close(model);
    return -1;
}

void cllm_whisper_small_model_close(cllm_whisper_small_model *model)
{
    if (model == NULL) return;
    if (model->image.base != NULL) munmap((void *)model->image.base, model->image.bytes);
    if (model->image.fd >= 0) close(model->image.fd);
    memset(model, 0, sizeof(*model));
    model->image.fd = -1;
}

static void final_layer_norm(const float *input, size_t rows,
                             const float *weight, const float *bias, float *output)
{
    for (size_t row = 0U; row < rows; ++row) {
        const float *source = input + row * 768U;
        float *destination = output + row * 768U;
        double sum = 0.0, square_sum = 0.0;
        for (size_t column = 0U; column < 768U; ++column) {
            sum += source[column];
            square_sum += (double)source[column] * source[column];
        }
        const double mean = sum / 768.0;
        double variance = square_sum / 768.0 - mean * mean;
        if (variance < 0.0) variance = 0.0;
        const float inverse = 1.0f / sqrtf((float)variance + 1.0e-5f);
        for (size_t column = 0U; column < 768U; ++column)
            destination[column] = (source[column] - (float)mean) * inverse * weight[column] + bias[column];
    }
}

int cllm_whisper_small_encode_mel(const cllm_whisper_small_model *model,
                                  const float *mel,
                                  size_t input_frames,
                                  size_t layer_count,
                                  float *output,
                                  cllm_whisper_small_encoder_metrics *metrics)
{
    const size_t frames = cllm_whisper_encoder_stem_output_frames(input_frames);
    const size_t state_count = frames * 768U;
    const size_t stem_count = input_frames * 768U;
    cllm_whisper_encoder_block_workspace workspace;
    float *allocation = NULL;
    float *cursor;
    float *state_a;
    float *state_b;
    float *after_attention;
    float *stem_scratch;
    size_t total_floats;
    double started;
    int result = -1;

    if (model == NULL || model->image.base == NULL || mel == NULL || output == NULL ||
        metrics == NULL || input_frames == 0U || input_frames > 3000U ||
        layer_count > 12U || frames > 1500U)
        return -1;
    if (state_count > SIZE_MAX / 8U || stem_count > SIZE_MAX - state_count * 8U ||
        frames * 3072U > SIZE_MAX - stem_count - state_count * 8U)
        return -1;
    total_floats = stem_count + state_count * 8U + frames + frames * 3072U;
    allocation = malloc(total_floats * sizeof(float));
    if (allocation == NULL) return -1;
    cursor = allocation;
    workspace.normalized = cursor; cursor += state_count;
    workspace.query = cursor; cursor += state_count;
    workspace.key = cursor; cursor += state_count;
    workspace.value = cursor; cursor += state_count;
    workspace.attention_context = cursor; cursor += state_count;
    after_attention = cursor; cursor += state_count;
    state_a = cursor; cursor += state_count;
    state_b = cursor; cursor += state_count;
    workspace.attention_scores = cursor; cursor += frames;
    workspace.mlp_hidden = cursor; cursor += frames * 3072U;
    stem_scratch = cursor;

    memset(metrics, 0, sizeof(*metrics));
    metrics->input_frames = input_frames;
    metrics->output_frames = frames;
    metrics->executed_layers = layer_count;
    metrics->workspace_bytes = total_floats * sizeof(float);
    started = monotonic_seconds();
    if (cllm_whisper_encoder_stem(mel, 80U, input_frames, 768U,
            model->conv1_weight, model->conv1_bias, model->conv2_weight,
            model->conv2_bias, model->positions, state_a,
            stem_scratch, stem_count) != 0)
        goto cleanup;
    metrics->stem_seconds = monotonic_seconds() - started;

    for (size_t layer = 0U; layer < layer_count; ++layer) {
        started = monotonic_seconds();
        if (cllm_whisper_encoder_block(state_a, frames, &model->layers[layer],
                &workspace, after_attention, state_b) != 0)
            goto cleanup;
        metrics->layer_seconds[layer] = monotonic_seconds() - started;
        { float *temporary = state_a; state_a = state_b; state_b = temporary; }
    }
    if (layer_count == 12U) {
        started = monotonic_seconds();
        final_layer_norm(state_a, frames, model->final_norm_weight,
                         model->final_norm_bias, output);
        metrics->final_norm_seconds = monotonic_seconds() - started;
    } else {
        memcpy(output, state_a, state_count * sizeof(float));
    }
    result = 0;
cleanup:
    free(allocation);
    return result;
}
