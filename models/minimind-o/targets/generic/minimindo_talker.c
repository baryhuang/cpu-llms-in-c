#define _POSIX_C_SOURCE 200809L

#include "minimindo_talker.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

enum { MMO_VERSION = 1, MMO_HEADER_BYTES = 4096, MMO_F32 = 1, MMO_Q8_ROW = 2 };

typedef struct {
    unsigned char magic[8];
    uint32_t version, header_bytes, layers, hidden, heads, kv_heads;
    uint32_t head_dim, intermediate, vocab, adapters, rank, pad_token;
    uint32_t tensor_count;
    float rms_epsilon, rope_theta;
    uint64_t tensors_offset, file_bytes;
    char source_sha256[64];
} mmo_talker_header;

typedef struct {
    uint32_t type, rows, cols, reserved;
    uint64_t data_bytes;
} mmo_tensor_header;

typedef struct {
    const mmo_tensor_header *header;
    const unsigned char *data;
} mmo_tensor;

typedef struct {
    mmo_tensor input_norm, q_norm, k_norm, post_norm;
    mmo_tensor q_proj, k_proj, v_proj, o_proj;
    mmo_tensor gate_proj, up_proj, down_proj;
} mmo_layer;

typedef struct {
    mmo_tensor first_weight, first_bias, second_weight, second_bias, norm;
} mmo_projection;

typedef struct { mmo_tensor input, output; } mmo_adapter;

struct minimindo_talker {
    int file;
    const unsigned char *mapping;
    size_t mapped_bytes;
    const mmo_talker_header *header;
    mmo_tensor base_embedding, final_norm, base_head, speaker_projection;
    mmo_tensor text_scale, audio_scale;
    mmo_adapter embedding_adapters[MINIMINDO_AUDIO_CODEBOOKS];
    mmo_adapter head_adapters[MINIMINDO_AUDIO_CODEBOOKS];
    mmo_projection codec_projection, bridge_projection;
    mmo_layer *layers;
    uint32_t max_context, position;
    float *key_cache, *value_cache;
    float *hidden, *normed, *query, *key, *value;
    float *attention, *projected, *mlp_normed, *gate, *up;
    float *rank_buffer, *codec_state, *bridge_state, *base_logits;
    double *scores;
};

_Static_assert(sizeof(mmo_talker_header) == 152, "MiniMind-O Talker header ABI");
_Static_assert(sizeof(mmo_tensor_header) == 24, "MiniMind-O tensor ABI");

static void set_error(char *error, size_t capacity, const char *format, ...)
{
    if (error == NULL || capacity == 0) return;
    va_list args;
    va_start(args, format);
    vsnprintf(error, capacity, format, args);
    va_end(args);
}

static uint64_t align64(uint64_t value) { return (value + 63U) & ~UINT64_C(63); }
static int range_ok(uint64_t offset, uint64_t length, uint64_t total)
{
    return offset <= total && length <= total - offset;
}

static int take_tensor(const unsigned char *mapping, uint64_t file_bytes,
                       uint64_t *cursor, uint32_t type, uint32_t rows,
                       uint32_t cols, mmo_tensor *tensor)
{
    if (!range_ok(*cursor, sizeof(mmo_tensor_header), file_bytes)) return -1;
    const mmo_tensor_header *header =
        (const mmo_tensor_header *)(mapping + *cursor);
    const uint64_t expected = type == MMO_F32
        ? (uint64_t)rows * cols * sizeof(float)
        : (uint64_t)rows * (sizeof(float) + cols);
    if (header->type != type || header->rows != rows || header->cols != cols ||
        header->reserved != 0 || header->data_bytes != expected) return -1;
    const uint64_t data_offset = *cursor + sizeof(*header);
    if (!range_ok(data_offset, expected, file_bytes)) return -1;
    tensor->header = header;
    tensor->data = mapping + data_offset;
    *cursor = align64(data_offset + expected);
    return *cursor <= file_bytes ? 0 : -1;
}

static const float *f32_data(const mmo_tensor *tensor)
{
    return (const float *)tensor->data;
}

static void q8_row_to_float(const mmo_tensor *tensor, uint32_t row, float *output)
{
    const size_t stride = sizeof(float) + tensor->header->cols;
    const unsigned char *record = tensor->data + (size_t)row * stride;
    float scale;
    memcpy(&scale, record, sizeof(scale));
    const int8_t *values = (const int8_t *)(record + sizeof(scale));
    for (uint32_t index = 0; index < tensor->header->cols; ++index)
        output[index] = scale * values[index];
}

static float q8_f32_dot(const int8_t *weights, const float *input, uint32_t count)
{
#if defined(__aarch64__)
    float32x4_t sum0 = vdupq_n_f32(0.0f), sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f), sum3 = vdupq_n_f32(0.0f);
    uint32_t index = 0;
    for (; index + 16 <= count; index += 16) {
        const int8x16_t packed = vld1q_s8(weights + index);
        const int16x8_t low = vmovl_s8(vget_low_s8(packed));
        const int16x8_t high = vmovl_s8(vget_high_s8(packed));
        sum0 = vfmaq_f32(sum0, vld1q_f32(input + index),
                         vcvtq_f32_s32(vmovl_s16(vget_low_s16(low))));
        sum1 = vfmaq_f32(sum1, vld1q_f32(input + index + 4),
                         vcvtq_f32_s32(vmovl_s16(vget_high_s16(low))));
        sum2 = vfmaq_f32(sum2, vld1q_f32(input + index + 8),
                         vcvtq_f32_s32(vmovl_s16(vget_low_s16(high))));
        sum3 = vfmaq_f32(sum3, vld1q_f32(input + index + 12),
                         vcvtq_f32_s32(vmovl_s16(vget_high_s16(high))));
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3)));
    for (; index < count; ++index) sum += input[index] * weights[index];
    return sum;
#else
    double sum = 0.0;
    for (uint32_t index = 0; index < count; ++index)
        sum += (double)input[index] * weights[index];
    return (float)sum;
#endif
}

static void q8_matvec(const mmo_tensor *tensor, const float *input, float *output)
{
    const size_t stride = sizeof(float) + tensor->header->cols;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (uint32_t row = 0; row < tensor->header->rows; ++row) {
        const unsigned char *record = tensor->data + (size_t)row * stride;
        float scale;
        memcpy(&scale, record, sizeof(scale));
        output[row] = q8_f32_dot((const int8_t *)(record + sizeof(scale)), input,
                                tensor->header->cols) * scale;
    }
}

static void rms_norm(const float *input, const float *weight, float *output,
                     uint32_t width, float epsilon)
{
    double squares = 0.0;
    for (uint32_t index = 0; index < width; ++index)
        squares += (double)input[index] * input[index];
    const double scale = 1.0 / sqrt(squares / width + epsilon);
    for (uint32_t index = 0; index < width; ++index)
        output[index] = (float)((double)input[index] * scale * weight[index]);
}

static void apply_rope(float *states, uint32_t heads, uint32_t head_dim,
                       uint32_t position, float theta)
{
    const uint32_t half = head_dim / 2U;
    for (uint32_t head = 0; head < heads; ++head) {
        float *vector = states + (size_t)head * head_dim;
        for (uint32_t index = 0; index < half; ++index) {
            const double angle = position / pow((double)theta, (2.0 * index) / head_dim);
            const double cosine = cos(angle), sine = sin(angle);
            const double first = vector[index], second = vector[index + half];
            vector[index] = (float)(first * cosine - second * sine);
            vector[index + half] = (float)(second * cosine + first * sine);
        }
    }
}

static float gelu(float value)
{
    return 0.5f * value * (1.0f + erff(value * 0.7071067811865475244f));
}

static double silu(double value) { return value / (1.0 + exp(-value)); }

static int allocate_buffers(minimindo_talker *model)
{
    const size_t hidden = model->header->hidden;
    const size_t kv = (size_t)model->header->kv_heads * model->header->head_dim;
    const size_t intermediate = model->header->intermediate;
    const size_t cache = (size_t)model->header->layers * model->max_context * kv;
#define ALLOC(field, count) do { model->field = calloc((count), sizeof(*model->field)); if (model->field == NULL) return -1; } while (0)
    ALLOC(key_cache, cache); ALLOC(value_cache, cache);
    ALLOC(hidden, hidden); ALLOC(normed, hidden); ALLOC(query, hidden);
    ALLOC(key, kv); ALLOC(value, kv); ALLOC(attention, hidden);
    ALLOC(projected, hidden); ALLOC(mlp_normed, hidden);
    ALLOC(gate, intermediate); ALLOC(up, intermediate);
    ALLOC(rank_buffer, model->header->rank); ALLOC(codec_state, hidden);
    ALLOC(bridge_state, hidden); ALLOC(base_logits, model->header->vocab);
    ALLOC(scores, model->max_context);
#undef ALLOC
    return 0;
}

static int take_projection(minimindo_talker *model, uint64_t *cursor,
                           mmo_projection *projection)
{
    const uint32_t h = model->header->hidden;
#define TAKE(member, type, rows, cols) if (take_tensor(model->mapping, model->mapped_bytes, cursor, type, rows, cols, &projection->member) != 0) return -1
    TAKE(first_weight, MMO_Q8_ROW, h, h); TAKE(first_bias, MMO_F32, 1, h);
    TAKE(second_weight, MMO_Q8_ROW, h, h); TAKE(second_bias, MMO_F32, 1, h);
    TAKE(norm, MMO_F32, 1, h);
#undef TAKE
    return 0;
}

minimindo_talker *minimindo_talker_open(const char *image_path,
                                        uint32_t max_context,
                                        char *error, size_t error_capacity)
{
    if (image_path == NULL || max_context == 0) {
        set_error(error, error_capacity, "invalid image path or context"); return NULL;
    }
    minimindo_talker *model = calloc(1, sizeof(*model));
    if (model == NULL) return NULL;
    model->file = -1;
    model->file = open(image_path, O_RDONLY);
    struct stat status;
    if (model->file < 0 || fstat(model->file, &status) != 0 ||
        status.st_size < MMO_HEADER_BYTES) {
        set_error(error, error_capacity, "cannot open image: %s", strerror(errno));
        minimindo_talker_close(model); return NULL;
    }
    model->mapped_bytes = (size_t)status.st_size;
    model->mapping = mmap(NULL, model->mapped_bytes, PROT_READ, MAP_PRIVATE,
                          model->file, 0);
    if (model->mapping == MAP_FAILED) {
        model->mapping = NULL;
        set_error(error, error_capacity, "cannot mmap image: %s", strerror(errno));
        minimindo_talker_close(model); return NULL;
    }
    model->header = (const mmo_talker_header *)model->mapping;
    static const unsigned char magic[8] = {'M','M','O','T','A','L','K','1'};
    if (memcmp(model->header->magic, magic, 8) != 0 ||
        model->header->version != MMO_VERSION ||
        model->header->header_bytes != MMO_HEADER_BYTES ||
        model->header->file_bytes != model->mapped_bytes ||
        model->header->tensors_offset != MMO_HEADER_BYTES ||
        model->header->hidden != model->header->heads * model->header->head_dim ||
        model->header->heads % model->header->kv_heads != 0 ||
        model->header->adapters != MINIMINDO_AUDIO_CODEBOOKS ||
        model->header->tensor_count != 1U + 16U + 10U +
            model->header->layers * 11U + 1U + 1U + 16U + 1U + 2U) {
        set_error(error, error_capacity, "invalid MiniMind-O Talker image header");
        minimindo_talker_close(model); return NULL;
    }
    model->layers = calloc(model->header->layers, sizeof(*model->layers));
    if (model->layers == NULL) { minimindo_talker_close(model); return NULL; }
    uint64_t cursor = model->header->tensors_offset;
    const uint32_t h = model->header->hidden, d = model->header->head_dim;
    const uint32_t kv = model->header->kv_heads * d, m = model->header->intermediate;
    const uint32_t v = model->header->vocab, r = model->header->rank;
#define TAKE(target, type, rows, cols) if (take_tensor(model->mapping, model->mapped_bytes, &cursor, type, rows, cols, target) != 0) goto bad_tensors
    TAKE(&model->base_embedding, MMO_Q8_ROW, v, h);
    for (uint32_t i = 0; i < MINIMINDO_AUDIO_CODEBOOKS; ++i) {
        TAKE(&model->embedding_adapters[i].input, MMO_Q8_ROW, v, r);
        TAKE(&model->embedding_adapters[i].output, MMO_Q8_ROW, h, r);
    }
    if (take_projection(model, &cursor, &model->codec_projection) != 0 ||
        take_projection(model, &cursor, &model->bridge_projection) != 0) goto bad_tensors;
    for (uint32_t index = 0; index < model->header->layers; ++index) {
        mmo_layer *layer = &model->layers[index];
        TAKE(&layer->input_norm, MMO_F32, 1, h);
        TAKE(&layer->q_norm, MMO_F32, 1, d); TAKE(&layer->k_norm, MMO_F32, 1, d);
        TAKE(&layer->post_norm, MMO_F32, 1, h);
        TAKE(&layer->q_proj, MMO_Q8_ROW, h, h); TAKE(&layer->k_proj, MMO_Q8_ROW, kv, h);
        TAKE(&layer->v_proj, MMO_Q8_ROW, kv, h); TAKE(&layer->o_proj, MMO_Q8_ROW, h, h);
        TAKE(&layer->gate_proj, MMO_Q8_ROW, m, h); TAKE(&layer->up_proj, MMO_Q8_ROW, m, h);
        TAKE(&layer->down_proj, MMO_Q8_ROW, h, m);
    }
    TAKE(&model->final_norm, MMO_F32, 1, h);
    TAKE(&model->base_head, MMO_Q8_ROW, v, h);
    for (uint32_t i = 0; i < MINIMINDO_AUDIO_CODEBOOKS; ++i) {
        TAKE(&model->head_adapters[i].input, MMO_Q8_ROW, r, h);
        TAKE(&model->head_adapters[i].output, MMO_Q8_ROW, v, r);
    }
    TAKE(&model->speaker_projection, MMO_Q8_ROW, h, 192);
    TAKE(&model->text_scale, MMO_F32, 1, 1); TAKE(&model->audio_scale, MMO_F32, 1, 1);
#undef TAKE
    if (cursor != model->mapped_bytes) goto bad_tensors;
    model->max_context = max_context;
    if (allocate_buffers(model) != 0) {
        set_error(error, error_capacity, "cannot allocate Talker runtime buffers");
        minimindo_talker_close(model); return NULL;
    }
    return model;

bad_tensors:
    set_error(error, error_capacity, "invalid MiniMind-O Talker tensor sequence");
    minimindo_talker_close(model);
    return NULL;
}

void minimindo_talker_close(minimindo_talker *model)
{
    if (model == NULL) return;
    free(model->layers); free(model->key_cache); free(model->value_cache);
    free(model->hidden); free(model->normed); free(model->query);
    free(model->key); free(model->value); free(model->attention);
    free(model->projected); free(model->mlp_normed); free(model->gate); free(model->up);
    free(model->rank_buffer); free(model->codec_state); free(model->bridge_state);
    free(model->base_logits); free(model->scores);
    if (model->mapping != NULL) munmap((void *)model->mapping, model->mapped_bytes);
    if (model->file >= 0) close(model->file);
    free(model);
}

void minimindo_talker_reset(minimindo_talker *model)
{
    if (model != NULL) model->position = 0;
}

uint32_t minimindo_talker_hidden_size(const minimindo_talker *model)
{ return model == NULL ? 0 : model->header->hidden; }
uint32_t minimindo_talker_vocab_size(const minimindo_talker *model)
{ return model == NULL ? 0 : model->header->vocab; }
uint32_t minimindo_talker_pad_token(const minimindo_talker *model)
{ return model == NULL ? 0 : model->header->pad_token; }
uint32_t minimindo_talker_position(const minimindo_talker *model)
{ return model == NULL ? 0 : model->position; }

static void projection_forward(const mmo_projection *projection,
                               const float *input, float *scratch, float *output,
                               uint32_t hidden, float epsilon)
{
    q8_matvec(&projection->first_weight, input, scratch);
    const float *first_bias = f32_data(&projection->first_bias);
    for (uint32_t i = 0; i < hidden; ++i) scratch[i] = gelu(scratch[i] + first_bias[i]);
    q8_matvec(&projection->second_weight, scratch, output);
    const float *second_bias = f32_data(&projection->second_bias);
    for (uint32_t i = 0; i < hidden; ++i) output[i] += second_bias[i];
    rms_norm(output, f32_data(&projection->norm), output, hidden, epsilon);
}

int minimindo_talker_forward(minimindo_talker *model,
                             const float *bridge_states, size_t bridge_count,
                             const uint32_t audio_ids[MINIMINDO_AUDIO_CODEBOOKS],
                             const float *speaker_embedding, size_t speaker_count,
                             float *logits, size_t logits_count,
                             char *error, size_t error_capacity)
{
    if (model == NULL || bridge_states == NULL || audio_ids == NULL || logits == NULL ||
        bridge_count < model->header->hidden ||
        logits_count < (size_t)MINIMINDO_AUDIO_CODEBOOKS * model->header->vocab) {
        set_error(error, error_capacity, "invalid Talker forward arguments"); return -1;
    }
    if (model->position >= model->max_context) {
        set_error(error, error_capacity, "Talker context limit reached"); return -2;
    }
    for (uint32_t i = 0; i < MINIMINDO_AUDIO_CODEBOOKS; ++i) {
        if (audio_ids[i] >= model->header->vocab) {
            set_error(error, error_capacity, "Talker audio token is out of range"); return -1;
        }
    }
    const uint32_t h = model->header->hidden, d = model->header->head_dim;
    const uint32_t heads = model->header->heads, kv_heads = model->header->kv_heads;
    const uint32_t kv_size = kv_heads * d, groups = heads / kv_heads;
    const uint32_t pos = model->position;

    memset(model->hidden, 0, h * sizeof(float));
    for (uint32_t codebook = 0; codebook < MINIMINDO_AUDIO_CODEBOOKS; ++codebook) {
        q8_row_to_float(&model->base_embedding, audio_ids[codebook], model->projected);
        q8_row_to_float(&model->embedding_adapters[codebook].input,
                        audio_ids[codebook], model->rank_buffer);
        for (uint32_t i = 0; i < model->header->rank; ++i)
            model->rank_buffer[i] = gelu(model->rank_buffer[i]);
        q8_matvec(&model->embedding_adapters[codebook].output,
                  model->rank_buffer, model->normed);
        for (uint32_t i = 0; i < h; ++i)
            model->hidden[i] += (model->projected[i] + model->normed[i]) /
                                (float)MINIMINDO_AUDIO_CODEBOOKS;
    }
    if (audio_ids[0] == 2051U && speaker_embedding != NULL && speaker_count >= 192U)
        q8_matvec(&model->speaker_projection, speaker_embedding, model->hidden);

    projection_forward(&model->codec_projection, model->hidden, model->normed,
                       model->codec_state, h, model->header->rms_epsilon);
    projection_forward(&model->bridge_projection, bridge_states, model->normed,
                       model->bridge_state, h, model->header->rms_epsilon);
    const float text_scale = f32_data(&model->text_scale)[0];
    const float audio_scale = f32_data(&model->audio_scale)[0];
    for (uint32_t i = 0; i < h; ++i)
        model->hidden[i] = model->bridge_state[i] * text_scale +
                           model->codec_state[i] * audio_scale;

    for (uint32_t layer_index = 0; layer_index < model->header->layers; ++layer_index) {
        const mmo_layer *layer = &model->layers[layer_index];
        rms_norm(model->hidden, f32_data(&layer->input_norm), model->normed,
                 h, model->header->rms_epsilon);
        q8_matvec(&layer->q_proj, model->normed, model->query);
        q8_matvec(&layer->k_proj, model->normed, model->key);
        q8_matvec(&layer->v_proj, model->normed, model->value);
        for (uint32_t head = 0; head < heads; ++head)
            rms_norm(model->query + (size_t)head * d, f32_data(&layer->q_norm),
                     model->query + (size_t)head * d, d, model->header->rms_epsilon);
        for (uint32_t head = 0; head < kv_heads; ++head)
            rms_norm(model->key + (size_t)head * d, f32_data(&layer->k_norm),
                     model->key + (size_t)head * d, d, model->header->rms_epsilon);
        apply_rope(model->query, heads, d, pos, model->header->rope_theta);
        apply_rope(model->key, kv_heads, d, pos, model->header->rope_theta);
        const size_t cache_base =
            ((size_t)layer_index * model->max_context + pos) * kv_size;
        memcpy(model->key_cache + cache_base, model->key, kv_size * sizeof(float));
        memcpy(model->value_cache + cache_base, model->value, kv_size * sizeof(float));
        for (uint32_t head = 0; head < heads; ++head) {
            const uint32_t kv_head = head / groups;
            const float *query = model->query + (size_t)head * d;
            double maximum = -INFINITY;
            for (uint32_t source = 0; source <= pos; ++source) {
                const size_t base = ((size_t)layer_index * model->max_context + source) *
                                    kv_size + (size_t)kv_head * d;
                double score = 0.0;
                for (uint32_t i = 0; i < d; ++i)
                    score += (double)query[i] * model->key_cache[base + i];
                model->scores[source] = score / sqrt((double)d);
                if (model->scores[source] > maximum) maximum = model->scores[source];
            }
            double denominator = 0.0;
            for (uint32_t source = 0; source <= pos; ++source) {
                model->scores[source] = exp(model->scores[source] - maximum);
                denominator += model->scores[source];
            }
            float *attention = model->attention + (size_t)head * d;
            for (uint32_t i = 0; i < d; ++i) {
                double sum = 0.0;
                for (uint32_t source = 0; source <= pos; ++source) {
                    const size_t base = ((size_t)layer_index * model->max_context + source) *
                                        kv_size + (size_t)kv_head * d;
                    sum += model->scores[source] * model->value_cache[base + i];
                }
                attention[i] = (float)(sum / denominator);
            }
        }
        q8_matvec(&layer->o_proj, model->attention, model->projected);
        for (uint32_t i = 0; i < h; ++i) model->hidden[i] += model->projected[i];
        rms_norm(model->hidden, f32_data(&layer->post_norm), model->mlp_normed,
                 h, model->header->rms_epsilon);
        q8_matvec(&layer->gate_proj, model->mlp_normed, model->gate);
        q8_matvec(&layer->up_proj, model->mlp_normed, model->up);
        for (uint32_t i = 0; i < model->header->intermediate; ++i)
            model->gate[i] = (float)(silu(model->gate[i]) * model->up[i]);
        q8_matvec(&layer->down_proj, model->gate, model->projected);
        for (uint32_t i = 0; i < h; ++i) model->hidden[i] += model->projected[i];
    }
    rms_norm(model->hidden, f32_data(&model->final_norm), model->normed,
             h, model->header->rms_epsilon);
    q8_matvec(&model->base_head, model->normed, model->base_logits);
    for (uint32_t codebook = 0; codebook < MINIMINDO_AUDIO_CODEBOOKS; ++codebook) {
        float *output = logits + (size_t)codebook * model->header->vocab;
        q8_matvec(&model->head_adapters[codebook].input,
                  model->normed, model->rank_buffer);
        for (uint32_t i = 0; i < model->header->rank; ++i)
            model->rank_buffer[i] = gelu(model->rank_buffer[i]);
        q8_matvec(&model->head_adapters[codebook].output, model->rank_buffer, output);
        for (uint32_t i = 0; i < model->header->vocab; ++i)
            output[i] += model->base_logits[i];
    }
    model->position++;
    return 0;
}
