#define _POSIX_C_SOURCE 200809L

#include "gemma4_task.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

enum {
    KIND_F32 = 1,
    KIND_U32 = 2,
    KIND_U8 = 3,
    KIND_Q4_GROUPED = 4,
    HIDDEN = 1536,
    LAYERS = 35,
    QUERY_HEADS = 8,
    PLE_SIZE = 256,
    EARLY_LAYERS = 15,
    LOCAL_SOURCE_LAYER = 13,
    GLOBAL_SOURCE_LAYER = 14,
    GROUP_SIZE = 128,
    Q4_RECORD_BYTES = 66
};

#pragma pack(push, 1)
struct image_header {
    char magic[8];
    uint32_t version;
    uint32_t descriptor_count;
    uint32_t token_count;
    uint32_t case_count;
    uint32_t label_count;
    uint32_t maximum_sequence;
    uint32_t group_size;
    uint32_t reserved;
    uint64_t directory_offset;
    uint64_t data_offset;
    uint64_t file_bytes;
    uint64_t reserved64;
};

struct image_descriptor {
    char name[96];
    uint32_t kind;
    uint32_t rank;
    uint32_t shape[4];
    uint64_t offset;
    uint64_t byte_count;
    uint64_t reserved;
};
#pragma pack(pop)

struct mapped_image {
    int fd;
    size_t bytes;
    const uint8_t *base;
    const struct image_header *header;
    const struct image_descriptor *descriptors;
};

struct layer_weights {
    const struct image_descriptor *input_norm;
    const struct image_descriptor *q_proj;
    const struct image_descriptor *k_proj;
    const struct image_descriptor *v_proj;
    const struct image_descriptor *q_norm;
    const struct image_descriptor *k_norm;
    const struct image_descriptor *o_proj;
    const struct image_descriptor *post_attention_norm;
    const struct image_descriptor *pre_feedforward_norm;
    const struct image_descriptor *gate_proj;
    const struct image_descriptor *up_proj;
    const struct image_descriptor *down_proj;
    const struct image_descriptor *post_feedforward_norm;
    const struct image_descriptor *ple_gate;
    const struct image_descriptor *ple_projection;
    const struct image_descriptor *post_ple_norm;
    const struct image_descriptor *layer_scalar;
};

struct model {
    struct mapped_image image;
    struct layer_weights layers[LAYERS];
    const struct image_descriptor *final_norm;
    const uint32_t *token_ids;
    const float *embeddings;
    const float *folded_ple;
    const uint32_t *label_token_ids;
    const float *label_weights;
    const uint32_t *case_offsets;
    const uint32_t *case_token_rows;
    const uint32_t *case_expected;
};

struct kv_cache {
    float *key[EARLY_LAYERS];
    float *value[EARLY_LAYERS];
    size_t maximum_sequence;
};

struct workspace {
    float *normalized;
    float *query;
    float *key;
    float *value;
    float *attention_heads;
    float *attention;
    float *gate;
    float *up;
    float *mlp;
    float *ple_gate;
    float *ple;
    float *scores;
};

static double monotonic_seconds(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return 0.0;
    }
    return (double)now.tv_sec + (double)now.tv_nsec * 1.0e-9;
}

static const void *entry_data(const struct mapped_image *image,
                              const struct image_descriptor *entry)
{
    return image->base + entry->offset;
}

static const struct image_descriptor *find_entry(const struct mapped_image *image,
                                                  const char *name)
{
    for (uint32_t index = 0; index < image->header->descriptor_count; ++index) {
        if (strncmp(image->descriptors[index].name, name,
                    sizeof(image->descriptors[index].name)) == 0) {
            return &image->descriptors[index];
        }
    }
    return NULL;
}

static int map_image(const char *path, struct mapped_image *image)
{
    struct stat status;
    memset(image, 0, sizeof(*image));
    image->fd = -1;
    image->fd = open(path, O_RDONLY);
    if (image->fd < 0 || fstat(image->fd, &status) != 0 || status.st_size < (off_t)sizeof(struct image_header)) {
        return -1;
    }
    image->bytes = (size_t)status.st_size;
    image->base = mmap(NULL, image->bytes, PROT_READ, MAP_PRIVATE, image->fd, 0);
    if (image->base == MAP_FAILED) {
        image->base = NULL;
        return -1;
    }
    image->header = (const struct image_header *)image->base;
    if (memcmp(image->header->magic, CLLM_GEMMA4_TASK_MAGIC, 8) != 0 ||
        image->header->version != CLLM_GEMMA4_TASK_VERSION ||
        image->header->group_size != CLLM_GEMMA4_TASK_GROUP_SIZE ||
        image->header->file_bytes != image->bytes ||
        image->header->directory_offset > image->bytes ||
        (uint64_t)image->header->descriptor_count * sizeof(struct image_descriptor) >
            image->bytes - image->header->directory_offset) {
        return -1;
    }
    image->descriptors = (const struct image_descriptor *)(
        image->base + image->header->directory_offset);
    for (uint32_t index = 0; index < image->header->descriptor_count; ++index) {
        const struct image_descriptor *entry = &image->descriptors[index];
        if (entry->offset > image->bytes || entry->byte_count > image->bytes - entry->offset ||
            memchr(entry->name, '\0', sizeof(entry->name)) == NULL) {
            return -1;
        }
    }
    return 0;
}

static void unmap_image(struct mapped_image *image)
{
    if (image->base != NULL) {
        munmap((void *)image->base, image->bytes);
    }
    if (image->fd >= 0) {
        close(image->fd);
    }
    memset(image, 0, sizeof(*image));
    image->fd = -1;
}

static float bf16_to_float(const uint8_t *bytes)
{
    uint16_t half = (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8U);
    uint32_t bits = (uint32_t)half << 16U;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int q4_matrix_valid(const struct image_descriptor *matrix)
{
    uint64_t expected;
    if (matrix == NULL || matrix->kind != KIND_Q4_GROUPED || matrix->rank != 2 ||
        matrix->shape[1] == 0U || matrix->shape[1] % GROUP_SIZE != 0U) {
        return 0;
    }
    expected = (uint64_t)matrix->shape[0] * (matrix->shape[1] / GROUP_SIZE) * Q4_RECORD_BYTES;
    return expected == matrix->byte_count;
}

static int q4_shape(const struct image_descriptor *matrix, uint32_t rows, uint32_t columns)
{
    return q4_matrix_valid(matrix) && matrix->shape[0] == rows && matrix->shape[1] == columns;
}

static int f32_vector_valid(const struct image_descriptor *entry, uint32_t count)
{
    return entry != NULL && entry->kind == KIND_F32 && entry->rank == 1 &&
           entry->shape[0] == count && entry->byte_count == (uint64_t)count * sizeof(float);
}

static void q4_gemv(const struct mapped_image *image,
                    const struct image_descriptor *matrix,
                    const float *input,
                    float *output)
{
    const uint8_t *data = entry_data(image, matrix);
    const uint32_t rows = matrix->shape[0];
    const uint32_t columns = matrix->shape[1];
    const uint32_t groups = columns / GROUP_SIZE;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (uint32_t row = 0; row < rows; ++row) {
        const uint8_t *record = data + (uint64_t)row * groups * Q4_RECORD_BYTES;
        float sum = 0.0f;
        for (uint32_t group = 0; group < groups; ++group) {
            const float scale = bf16_to_float(record);
            const uint8_t *packed = record + 2;
            const float *values = input + (uint64_t)group * GROUP_SIZE;
            float group_sum = 0.0f;
            for (uint32_t index = 0; index < GROUP_SIZE / 2; ++index) {
                const uint8_t byte = packed[index];
                int low = byte & 0x0F;
                int high = byte >> 4;
                low = low >= 8 ? low - 16 : low;
                high = high >= 8 ? high - 16 : high;
                group_sum += values[index * 2] * (float)low;
                group_sum += values[index * 2 + 1] * (float)high;
            }
            sum += group_sum * scale;
            record += Q4_RECORD_BYTES;
        }
        output[row] = sum;
    }
}

static void rms_norm(const float *input, const float *scale, float *output, size_t width)
{
    double squares = 0.0;
    for (size_t index = 0; index < width; ++index) {
        squares += (double)input[index] * input[index];
    }
    const float inverse = (float)pow(squares / (double)width + 1.0e-6, -0.5);
    if (scale == NULL) {
        for (size_t index = 0; index < width; ++index) {
            output[index] = input[index] * inverse;
        }
    } else {
        for (size_t index = 0; index < width; ++index) {
            output[index] = input[index] * inverse * scale[index];
        }
    }
}

static float gelu_pytorch_tanh(float value)
{
    const float inner = 0.7978845608028654f * (value + 0.044715f * value * value * value);
    return 0.5f * value * (1.0f + tanhf(inner));
}

static void apply_rope(float *states, size_t heads, size_t head_dim, size_t position,
                       float theta, float rotary_fraction)
{
    const size_t half = head_dim / 2U;
    const size_t rotated = (size_t)(rotary_fraction * (float)half);
    for (size_t head = 0; head < heads; ++head) {
        float *vector = states + head * head_dim;
        for (size_t index = 0; index < rotated; ++index) {
            const double frequency = 1.0 / pow((double)theta, (double)(2U * index) / (double)head_dim);
            const double angle = (double)position * frequency;
            const float cosine = (float)cos(angle);
            const float sine = (float)sin(angle);
            const float first = vector[index];
            const float second = vector[index + half];
            vector[index] = first * cosine - second * sine;
            vector[index + half] = second * cosine + first * sine;
        }
    }
}

static void attention_one(const float *query, const float *keys, const float *values,
                          size_t position, size_t heads, size_t head_dim,
                          size_t window, float *scores, float *output)
{
    const size_t first = window != 0U && position + 1U > window ? position + 1U - window : 0U;
    for (size_t head = 0; head < heads; ++head) {
        const float *query_head = query + head * head_dim;
        float maximum = -INFINITY;
        float denominator = 0.0f;
        for (size_t key_position = first; key_position <= position; ++key_position) {
            const float *key = keys + key_position * head_dim;
            float score = 0.0f;
            for (size_t index = 0; index < head_dim; ++index) {
                score += query_head[index] * key[index];
            }
            scores[key_position] = score;
            maximum = score > maximum ? score : maximum;
        }
        for (size_t key_position = first; key_position <= position; ++key_position) {
            scores[key_position] = expf(scores[key_position] - maximum);
            denominator += scores[key_position];
        }
        for (size_t index = 0; index < head_dim; ++index) {
            float sum = 0.0f;
            for (size_t key_position = first; key_position <= position; ++key_position) {
                sum += scores[key_position] / denominator * values[key_position * head_dim + index];
            }
            output[head * head_dim + index] = sum;
        }
    }
}

static const float *f32_data(const struct mapped_image *image,
                             const struct image_descriptor *entry)
{
    return (const float *)entry_data(image, entry);
}

static int resolve_layer(const struct mapped_image *image, uint32_t layer,
                         struct layer_weights *weights)
{
    char name[160];
    const uint32_t head_dim = layer % 5U == 4U ? 512U : 256U;
    const uint32_t query_size = QUERY_HEADS * head_dim;
    const uint32_t intermediate = layer >= 15U ? 12288U : 6144U;
#define RESOLVE(field, suffix) do { \
    snprintf(name, sizeof(name), "model.language_model.layers.%u.%s", layer, suffix); \
    weights->field = find_entry(image, name); \
} while (0)
    RESOLVE(input_norm, "input_layernorm.weight");
    RESOLVE(q_proj, "self_attn.q_proj.weight");
    RESOLVE(k_proj, "self_attn.k_proj.weight");
    RESOLVE(v_proj, "self_attn.v_proj.weight");
    RESOLVE(q_norm, "self_attn.q_norm.weight");
    RESOLVE(k_norm, "self_attn.k_norm.weight");
    RESOLVE(o_proj, "self_attn.o_proj.weight");
    RESOLVE(post_attention_norm, "post_attention_layernorm.weight");
    RESOLVE(pre_feedforward_norm, "pre_feedforward_layernorm.weight");
    RESOLVE(gate_proj, "mlp.gate_proj.weight");
    RESOLVE(up_proj, "mlp.up_proj.weight");
    RESOLVE(down_proj, "mlp.down_proj.weight");
    RESOLVE(post_feedforward_norm, "post_feedforward_layernorm.weight");
    RESOLVE(ple_gate, "per_layer_input_gate.weight");
    RESOLVE(ple_projection, "per_layer_projection.weight");
    RESOLVE(post_ple_norm, "post_per_layer_input_norm.weight");
    RESOLVE(layer_scalar, "layer_scalar");
#undef RESOLVE
    if (!q4_shape(weights->q_proj, query_size, HIDDEN) ||
        !q4_shape(weights->o_proj, HIDDEN, query_size) ||
        !q4_shape(weights->gate_proj, intermediate, HIDDEN) ||
        !q4_shape(weights->up_proj, intermediate, HIDDEN) ||
        !q4_shape(weights->down_proj, HIDDEN, intermediate) ||
        !q4_shape(weights->ple_gate, PLE_SIZE, HIDDEN) ||
        !q4_shape(weights->ple_projection, HIDDEN, PLE_SIZE) ||
        !f32_vector_valid(weights->input_norm, HIDDEN) ||
        !f32_vector_valid(weights->post_attention_norm, HIDDEN) ||
        !f32_vector_valid(weights->pre_feedforward_norm, HIDDEN) ||
        !f32_vector_valid(weights->post_feedforward_norm, HIDDEN) ||
        !f32_vector_valid(weights->post_ple_norm, HIDDEN) ||
        !f32_vector_valid(weights->q_norm, head_dim) ||
        !f32_vector_valid(weights->layer_scalar, 1U)) {
        return -1;
    }
    if (layer < EARLY_LAYERS &&
        (!q4_shape(weights->k_proj, head_dim, HIDDEN) ||
         !q4_shape(weights->v_proj, head_dim, HIDDEN) ||
         !f32_vector_valid(weights->k_norm, head_dim))) {
        return -1;
    }
    return 0;
}

static int resolve_model(const char *path, struct model *model)
{
    memset(model, 0, sizeof(*model));
    if (map_image(path, &model->image) != 0 || model->image.header->label_count != 2U) {
        return -1;
    }
    for (uint32_t layer = 0; layer < LAYERS; ++layer) {
        if (resolve_layer(&model->image, layer, &model->layers[layer]) != 0) {
            return -1;
        }
    }
    model->final_norm = find_entry(&model->image, "model.language_model.norm.weight");
#define PROFILE(pointer, name, expected_kind) do { \
    const struct image_descriptor *entry = find_entry(&model->image, name); \
    if (entry == NULL || entry->kind != expected_kind) return -1; \
    model->pointer = entry_data(&model->image, entry); \
} while (0)
    PROFILE(token_ids, "profile.token_ids", KIND_U32);
    PROFILE(embeddings, "profile.embeddings", KIND_F32);
    PROFILE(folded_ple, "profile.folded_ple", KIND_F32);
    PROFILE(label_token_ids, "profile.label_token_ids", KIND_U32);
    PROFILE(label_weights, "profile.label_weights", KIND_F32);
    PROFILE(case_offsets, "profile.case_offsets", KIND_U32);
    PROFILE(case_token_rows, "profile.case_token_rows", KIND_U32);
    PROFILE(case_expected, "profile.case_expected", KIND_U32);
#undef PROFILE
    return f32_vector_valid(model->final_norm, HIDDEN) ? 0 : -1;
}

static void free_cache(struct kv_cache *cache);
static void free_workspace(struct workspace *workspace);

static int allocate_cache(struct kv_cache *cache, size_t maximum_sequence)
{
    memset(cache, 0, sizeof(*cache));
    cache->maximum_sequence = maximum_sequence;
    for (size_t layer = 0; layer < EARLY_LAYERS; ++layer) {
        const size_t head_dim = layer % 5U == 4U ? 512U : 256U;
        cache->key[layer] = calloc(maximum_sequence * head_dim, sizeof(float));
        cache->value[layer] = calloc(maximum_sequence * head_dim, sizeof(float));
        if (cache->key[layer] == NULL || cache->value[layer] == NULL) {
            free_cache(cache);
            return -1;
        }
    }
    return 0;
}

static void free_cache(struct kv_cache *cache)
{
    for (size_t layer = 0; layer < EARLY_LAYERS; ++layer) {
        free(cache->key[layer]);
        free(cache->value[layer]);
    }
    memset(cache, 0, sizeof(*cache));
}

static int allocate_workspace(struct workspace *workspace, size_t maximum_sequence)
{
    memset(workspace, 0, sizeof(*workspace));
    workspace->normalized = malloc(HIDDEN * sizeof(float));
    workspace->query = malloc(QUERY_HEADS * 512U * sizeof(float));
    workspace->key = malloc(512U * sizeof(float));
    workspace->value = malloc(512U * sizeof(float));
    workspace->attention_heads = malloc(QUERY_HEADS * 512U * sizeof(float));
    workspace->attention = malloc(HIDDEN * sizeof(float));
    workspace->gate = malloc(12288U * sizeof(float));
    workspace->up = malloc(12288U * sizeof(float));
    workspace->mlp = malloc(HIDDEN * sizeof(float));
    workspace->ple_gate = malloc(PLE_SIZE * sizeof(float));
    workspace->ple = malloc(HIDDEN * sizeof(float));
    workspace->scores = malloc(maximum_sequence * sizeof(float));
    if (workspace->normalized && workspace->query && workspace->key && workspace->value &&
        workspace->attention_heads && workspace->attention && workspace->gate && workspace->up &&
        workspace->mlp && workspace->ple_gate && workspace->ple && workspace->scores) {
        return 0;
    }
    free_workspace(workspace);
    return -1;
}

static void free_workspace(struct workspace *workspace)
{
    free(workspace->normalized);
    free(workspace->query);
    free(workspace->key);
    free(workspace->value);
    free(workspace->attention_heads);
    free(workspace->attention);
    free(workspace->gate);
    free(workspace->up);
    free(workspace->mlp);
    free(workspace->ple_gate);
    free(workspace->ple);
    free(workspace->scores);
    memset(workspace, 0, sizeof(*workspace));
}

static int forward_token(const struct model *model, struct kv_cache *cache,
                         struct workspace *workspace, uint32_t token_row,
                         size_t position, float *hidden)
{
    if (token_row >= model->image.header->token_count || position >= cache->maximum_sequence) {
        return -1;
    }
    memcpy(hidden, model->embeddings + (uint64_t)token_row * HIDDEN, HIDDEN * sizeof(float));
    for (uint32_t layer = 0; layer < LAYERS; ++layer) {
        const struct layer_weights *weights = &model->layers[layer];
        const int global = layer % 5U == 4U;
        const size_t head_dim = global ? 512U : 256U;
        const size_t intermediate = layer >= 15U ? 12288U : 6144U;
        const size_t source_layer = global ? GLOBAL_SOURCE_LAYER : LOCAL_SOURCE_LAYER;
        const float *keys;
        const float *values;

        rms_norm(hidden, f32_data(&model->image, weights->input_norm), workspace->normalized, HIDDEN);
        q4_gemv(&model->image, weights->q_proj, workspace->normalized, workspace->query);
        for (size_t head = 0; head < QUERY_HEADS; ++head) {
            rms_norm(workspace->query + head * head_dim,
                     f32_data(&model->image, weights->q_norm),
                     workspace->query + head * head_dim, head_dim);
        }
        apply_rope(workspace->query, QUERY_HEADS, head_dim, position,
                   global ? 1000000.0f : 10000.0f, global ? 0.25f : 1.0f);

        if (layer < EARLY_LAYERS) {
            q4_gemv(&model->image, weights->k_proj, workspace->normalized, workspace->key);
            q4_gemv(&model->image, weights->v_proj, workspace->normalized, workspace->value);
            rms_norm(workspace->key, f32_data(&model->image, weights->k_norm), workspace->key, head_dim);
            rms_norm(workspace->value, NULL, workspace->value, head_dim);
            apply_rope(workspace->key, 1U, head_dim, position,
                       global ? 1000000.0f : 10000.0f, global ? 0.25f : 1.0f);
            memcpy(cache->key[layer] + position * head_dim, workspace->key, head_dim * sizeof(float));
            memcpy(cache->value[layer] + position * head_dim, workspace->value, head_dim * sizeof(float));
            keys = cache->key[layer];
            values = cache->value[layer];
        } else {
            keys = cache->key[source_layer];
            values = cache->value[source_layer];
        }
        attention_one(workspace->query, keys, values, position, QUERY_HEADS, head_dim,
                      global ? 0U : 512U, workspace->scores, workspace->attention_heads);
        q4_gemv(&model->image, weights->o_proj, workspace->attention_heads, workspace->attention);
        rms_norm(workspace->attention,
                 f32_data(&model->image, weights->post_attention_norm),
                 workspace->normalized, HIDDEN);
        for (size_t index = 0; index < HIDDEN; ++index) {
            hidden[index] += workspace->normalized[index];
        }

        rms_norm(hidden, f32_data(&model->image, weights->pre_feedforward_norm),
                 workspace->normalized, HIDDEN);
        q4_gemv(&model->image, weights->gate_proj, workspace->normalized, workspace->gate);
        q4_gemv(&model->image, weights->up_proj, workspace->normalized, workspace->up);
        for (size_t index = 0; index < intermediate; ++index) {
            workspace->gate[index] = gelu_pytorch_tanh(workspace->gate[index]) * workspace->up[index];
        }
        q4_gemv(&model->image, weights->down_proj, workspace->gate, workspace->mlp);
        rms_norm(workspace->mlp,
                 f32_data(&model->image, weights->post_feedforward_norm),
                 workspace->normalized, HIDDEN);
        for (size_t index = 0; index < HIDDEN; ++index) {
            hidden[index] += workspace->normalized[index];
        }

        q4_gemv(&model->image, weights->ple_gate, hidden, workspace->ple_gate);
        const float *token_ple = model->folded_ple +
            ((uint64_t)token_row * LAYERS + layer) * PLE_SIZE;
        for (size_t index = 0; index < PLE_SIZE; ++index) {
            workspace->ple_gate[index] = gelu_pytorch_tanh(workspace->ple_gate[index]) * token_ple[index];
        }
        q4_gemv(&model->image, weights->ple_projection, workspace->ple_gate, workspace->ple);
        rms_norm(workspace->ple, f32_data(&model->image, weights->post_ple_norm),
                 workspace->normalized, HIDDEN);
        const float scalar = f32_data(&model->image, weights->layer_scalar)[0];
        for (size_t index = 0; index < HIDDEN; ++index) {
            hidden[index] = (hidden[index] + workspace->normalized[index]) * scalar;
        }
    }
    return 0;
}

static uint32_t token_row_for_id(const struct model *model, uint32_t token_id)
{
    for (uint32_t row = 0; row < model->image.header->token_count; ++row) {
        if (model->token_ids[row] == token_id) {
            return row;
        }
    }
    return UINT32_MAX;
}

static int run_case(const struct model *model, uint32_t case_index)
{
    struct kv_cache cache;
    struct workspace workspace;
    float hidden[HIDDEN];
    float normalized[HIDDEN];
    float logits[2];
    const uint32_t begin = model->case_offsets[case_index];
    const uint32_t end = model->case_offsets[case_index + 1U];
    const uint32_t length = end - begin;
    const double start = monotonic_seconds();
    double prompt_end;
    double decode_end;
    uint32_t predicted;
    int result = -1;

    memset(&cache, 0, sizeof(cache));
    memset(&workspace, 0, sizeof(workspace));
    if (allocate_cache(&cache, model->image.header->maximum_sequence + 1U) != 0) {
        goto cleanup;
    }
    if (allocate_workspace(&workspace, model->image.header->maximum_sequence + 1U) != 0) {
        goto cleanup;
    }
    for (uint32_t position = 0; position < length; ++position) {
        if (forward_token(model, &cache, &workspace,
                          model->case_token_rows[begin + position], position, hidden) != 0) {
            goto cleanup;
        }
    }
    rms_norm(hidden, f32_data(&model->image, model->final_norm), normalized, HIDDEN);
    for (uint32_t label = 0; label < 2U; ++label) {
        double sum = 0.0;
        const float *row = model->label_weights + (uint64_t)label * HIDDEN;
        for (size_t index = 0; index < HIDDEN; ++index) {
            sum += (double)normalized[index] * row[index];
        }
        logits[label] = 30.0f * tanhf((float)sum / 30.0f);
    }
    prompt_end = monotonic_seconds();
    predicted = logits[1] > logits[0] ? 1U : 0U;
    const uint32_t decode_row = token_row_for_id(model, model->label_token_ids[predicted]);
    if (decode_row == UINT32_MAX || forward_token(model, &cache, &workspace, decode_row, length, hidden) != 0) {
        goto cleanup;
    }
    decode_end = monotonic_seconds();
    printf("{\"case\":%u,\"tokens\":%u,\"expected\":%u,\"predicted\":%u,"
           "\"safe_logit\":%.9g,\"danger_logit\":%.9g,\"prompt_seconds\":%.6f,"
           "\"prefill_tokens_per_second\":%.6f,\"decode_seconds\":%.6f,"
           "\"decode_tokens_per_second\":%.6f}\n",
           case_index, length, model->case_expected[case_index], predicted,
           logits[0], logits[1], prompt_end - start,
           (double)length / (prompt_end - start), decode_end - prompt_end,
           1.0 / (decode_end - prompt_end));
    fflush(stdout);
    result = predicted == model->case_expected[case_index] ? 0 : 1;
cleanup:
    free_workspace(&workspace);
    free_cache(&cache);
    return result;
}

int main(int argc, char **argv)
{
    struct model model;
    uint32_t first = 0;
    uint32_t end;
    unsigned correct = 0;
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "usage: %s IMAGE [CASE_INDEX|all]\n", argv[0]);
        return 2;
    }
    if (resolve_model(argv[1], &model) != 0) {
        fprintf(stderr, "invalid or unsupported task image: %s\n", argv[1]);
        return 2;
    }
#ifdef _OPENMP
    omp_set_dynamic(0);
#endif
    end = model.image.header->case_count;
    if (argc == 3 && strcmp(argv[2], "all") != 0) {
        char *tail = NULL;
        unsigned long parsed = strtoul(argv[2], &tail, 10);
        if (tail == argv[2] || *tail != '\0' || parsed >= end) {
            fprintf(stderr, "case index is outside the image\n");
            return 2;
        }
        first = (uint32_t)parsed;
        end = first + 1U;
    }
    for (uint32_t index = first; index < end; ++index) {
        const int result = run_case(&model, index);
        if (result < 0) {
            fprintf(stderr, "case %u failed during execution\n", index);
            return 2;
        }
        correct += result == 0;
    }
    printf("{\"summary\":true,\"cases\":%u,\"correct\":%u,\"accuracy\":%.9g}\n",
           end - first, correct, (double)correct / (double)(end - first));
    unmap_image(&model.image);
    return 0;
}
