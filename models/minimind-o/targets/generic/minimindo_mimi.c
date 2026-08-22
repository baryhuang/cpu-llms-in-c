#define _POSIX_C_SOURCE 200809L

#include "minimindo_mimi.h"
#include "minimindo_parallel.h"

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
#include <time.h>
#include <unistd.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

enum { MMO_VERSION = 1, MMO_HEADER_BYTES = 4096, MMO_F32 = 1, MMO_Q8_ROW = 2 };

typedef struct {
    unsigned char magic[8];
    uint32_t version, header_bytes, layers, hidden, heads, head_dim;
    uint32_t intermediate, codebooks, codebook_size, codebook_dim;
    uint32_t sample_rate, frame_rate_x10, tensor_count, sliding_window;
    uint32_t ratio_count, reserved;
    float norm_epsilon, rope_theta;
    uint32_t ratios[4];
    uint64_t tensors_offset, file_bytes;
    char source_sha256[64];
} mimi_header;

typedef struct {
    uint32_t type, rows, cols, reserved;
    uint64_t data_bytes;
} tensor_header;

typedef struct { const tensor_header *header; const unsigned char *data; } tensor;

typedef struct {
    tensor input_weight, input_bias, post_weight, post_bias;
    tensor attention_scale, mlp_scale;
    tensor q, k, v, o, fc1, fc2;
} transformer_layer;

typedef struct {
    tensor weight, bias;
    uint32_t in_channels, out_channels, kernel, stride;
    int transpose;
    int8_t *phase_weights;
} decoder_conv;

struct minimindo_mimi {
    int file;
    const unsigned char *mapping;
    size_t mapped_bytes;
    const mimi_header *header;
    tensor codebooks[MINIMINDO_MIMI_CODEBOOKS];
    tensor semantic_output, acoustic_output, upsample;
    transformer_layer *layers;
    decoder_conv convs[14];
    uint32_t max_frames;
    float *key_cache, *value_cache;
    float *hidden, *normed, *query, *key, *value, *attention;
    float *projected, *mlp, *scores;
};

typedef struct {
    float *history;
} mimi_conv_stream;

struct minimindo_mimi_stream {
    minimindo_mimi *model;
    uint32_t frames;
    uint32_t transformer_positions;
    float *upsample_tail;
    float *transformer_pair;
    mimi_conv_stream convs[14];
};

_Static_assert(sizeof(mimi_header) == 176, "MiniMind-O Mimi header ABI");
_Static_assert(sizeof(tensor_header) == 24, "MiniMind-O tensor ABI");

static void set_error(char *error, size_t capacity, const char *format, ...)
{
    if (error == NULL || capacity == 0) return;
    va_list args;
    va_start(args, format); vsnprintf(error, capacity, format, args); va_end(args);
}

static uint64_t align64(uint64_t value) { return (value + 63U) & ~UINT64_C(63); }
static int range_ok(uint64_t offset, uint64_t length, uint64_t total)
{ return offset <= total && length <= total - offset; }

static double seconds(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

static int take_tensor(const unsigned char *mapping, uint64_t file_bytes,
                       uint64_t *cursor, uint32_t type, uint32_t rows,
                       uint32_t cols, tensor *output)
{
    if (!range_ok(*cursor, sizeof(tensor_header), file_bytes)) return -1;
    const tensor_header *header = (const tensor_header *)(mapping + *cursor);
    const uint64_t expected = type == MMO_F32
        ? (uint64_t)rows * cols * sizeof(float)
        : (uint64_t)rows * (sizeof(float) + cols);
    if (header->type != type || header->rows != rows || header->cols != cols ||
        header->reserved != 0 || header->data_bytes != expected) return -1;
    const uint64_t data_offset = *cursor + sizeof(*header);
    if (!range_ok(data_offset, expected, file_bytes)) return -1;
    output->header = header; output->data = mapping + data_offset;
    *cursor = align64(data_offset + expected);
    return *cursor <= file_bytes ? 0 : -1;
}

static const float *f32_data(const tensor *value) { return (const float *)value->data; }

static float q8_dot(const int8_t *weights, const float *input, uint32_t count)
{
#if defined(__aarch64__)
    float32x4_t a = vdupq_n_f32(0), b = a, c = a, d = a;
    uint32_t i = 0;
    for (; i + 16 <= count; i += 16) {
        int8x16_t packed = vld1q_s8(weights + i);
        int16x8_t lo = vmovl_s8(vget_low_s8(packed));
        int16x8_t hi = vmovl_s8(vget_high_s8(packed));
        a = vfmaq_f32(a, vld1q_f32(input + i), vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))));
        b = vfmaq_f32(b, vld1q_f32(input + i + 4), vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))));
        c = vfmaq_f32(c, vld1q_f32(input + i + 8), vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))));
        d = vfmaq_f32(d, vld1q_f32(input + i + 12), vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))));
    }
    for (; i + 4 <= count; i += 4) {
        uint32_t packed4;
        memcpy(&packed4, weights + i, sizeof(packed4));
        const int16x8_t packed = vmovl_s8(vcreate_s8(packed4));
        a = vfmaq_f32(a, vld1q_f32(input + i),
                      vcvtq_f32_s32(vmovl_s16(vget_low_s16(packed))));
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d)));
    for (; i < count; ++i) sum += weights[i] * input[i];
    return sum;
#else
    double sum = 0;
    for (uint32_t i = 0; i < count; ++i) sum += (double)weights[i] * input[i];
    return (float)sum;
#endif
}

typedef struct {
    int8_t *values;
    float *scales;
} q8_windows;

static int32_t q8_i8_dot(const int8_t *weights, const int8_t *input,
                         uint32_t count)
{
#if defined(__aarch64__)
    int32x4_t sum0 = vdupq_n_s32(0);
    int32x4_t sum1 = vdupq_n_s32(0);
    uint32_t index = 0;
    for (; index + 16U <= count; index += 16U) {
        const int8x16_t w = vld1q_s8(weights + index);
        const int8x16_t x = vld1q_s8(input + index);
        sum0 = vpadalq_s16(sum0,
                           vmull_s8(vget_low_s8(w), vget_low_s8(x)));
        sum1 = vpadalq_s16(sum1,
                           vmull_s8(vget_high_s8(w), vget_high_s8(x)));
    }
    int32_t sum = vaddvq_s32(vaddq_s32(sum0, sum1));
    for (; index < count; ++index) sum += weights[index] * input[index];
    return sum;
#else
    int32_t sum = 0;
    for (uint32_t index = 0; index < count; ++index)
        sum += weights[index] * input[index];
    return sum;
#endif
}

static float quantize_i8(const float *input, int8_t *output, uint32_t count)
{
    float maximum = 0.0f;
#if defined(__aarch64__)
    float32x4_t vmax = vdupq_n_f32(0.0f);
    uint32_t index = 0;
    for (; index + 4U <= count; index += 4U)
        vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(input + index)));
    maximum = vmaxvq_f32(vmax);
    for (; index < count; ++index) {
        const float value = fabsf(input[index]);
        if (value > maximum) maximum = value;
    }
#else
    for (uint32_t index = 0; index < count; ++index) {
        const float value = fabsf(input[index]);
        if (value > maximum) maximum = value;
    }
#endif
    if (!(maximum > 0.0f)) {
        memset(output, 0, count);
        return 0.0f;
    }
    const float scale = maximum / 127.0f;
    const float inverse = 1.0f / scale;
#if defined(__aarch64__)
    const float32x4_t vinverse = vdupq_n_f32(inverse);
    uint32_t output_index = 0;
    for (; output_index + 16U <= count; output_index += 16U) {
        const int32x4_t q0 = vcvtnq_s32_f32(
            vmulq_f32(vld1q_f32(input + output_index), vinverse));
        const int32x4_t q1 = vcvtnq_s32_f32(
            vmulq_f32(vld1q_f32(input + output_index + 4U), vinverse));
        const int32x4_t q2 = vcvtnq_s32_f32(
            vmulq_f32(vld1q_f32(input + output_index + 8U), vinverse));
        const int32x4_t q3 = vcvtnq_s32_f32(
            vmulq_f32(vld1q_f32(input + output_index + 12U), vinverse));
        const int16x8_t lo = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        const int16x8_t hi = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
        vst1q_s8(output + output_index,
                 vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi)));
    }
    for (; output_index < count; ++output_index)
        output[output_index] =
            (int8_t)lrintf(input[output_index] * inverse);
#else
    for (uint32_t index = 0; index < count; ++index)
        output[index] = (int8_t)lrintf(input[index] * inverse);
#endif
    return scale;
}

static void q8_row(const tensor *value, uint32_t row,
                   const int8_t **weights, float *scale)
{
    const size_t stride = sizeof(float) + value->header->cols;
    const unsigned char *record = value->data + (size_t)row * stride;
    memcpy(scale, record, sizeof(*scale));
    *weights = (const int8_t *)(record + sizeof(*scale));
}

typedef struct { const tensor *matrix; const float *input; float *output; } matvec_parallel_context;

static void matvec_rows(void *opaque, size_t begin, size_t end)
{
    matvec_parallel_context *context = opaque;
    const tensor *matrix = context->matrix;
    for (size_t row = begin; row < end; ++row) {
        const int8_t *weights; float scale;
        q8_row(matrix, (uint32_t)row, &weights, &scale);
        context->output[row] = q8_dot(
            weights, context->input, matrix->header->cols) * scale;
    }
}

static void matvec(const tensor *matrix, const float *input, float *output)
{
    matvec_parallel_context context = {matrix,input,output};
    minimindo_parallel_for(matrix->header->rows,matvec_rows,&context);
}

static void matvec_pair_rows(void *opaque, size_t begin, size_t end)
{
    matvec_parallel_context *context = opaque;
    const tensor *matrix = context->matrix;
    const uint32_t rows = matrix->header->rows;
    const uint32_t columns = matrix->header->cols;
    for (size_t row = begin; row < end; ++row) {
        const int8_t *weights; float scale;
        q8_row(matrix,(uint32_t)row,&weights,&scale);
        context->output[row]=q8_dot(weights,context->input,columns)*scale;
        context->output[rows+row]=q8_dot(weights,context->input+columns,columns)*scale;
    }
}

static void matvec_pair(const tensor *matrix, const float *input,
                        float *output)
{
    matvec_parallel_context context = {matrix,input,output};
    minimindo_parallel_for(matrix->header->rows,matvec_pair_rows,&context);
}

static void layer_norm(const float *input, const float *weight, const float *bias,
                       float *output, uint32_t width, float epsilon)
{
    double mean = 0, squares = 0;
    for (uint32_t i = 0; i < width; ++i) { mean += input[i]; squares += (double)input[i] * input[i]; }
    mean /= width;
    const double scale = 1.0 / sqrt(squares / width - mean * mean + epsilon);
    for (uint32_t i = 0; i < width; ++i)
        output[i] = (float)((input[i] - mean) * scale * weight[i] + bias[i]);
}

static float gelu(float x) { return 0.5f * x * (1.0f + erff(x * 0.7071067811865475f)); }
static float elu(float x) { return x >= 0 ? x : expf(x) - 1.0f; }

static void rope(float *states, uint32_t heads, uint32_t dim,
                 uint32_t position, float theta)
{
    const uint32_t half = dim / 2;
    for (uint32_t head = 0; head < heads; ++head) {
        float *v = states + (size_t)head * dim;
        for (uint32_t i = 0; i < half; ++i) {
            const double angle = position / pow((double)theta, (2.0 * i) / dim);
            const double co = cos(angle), si = sin(angle), first = v[i], second = v[i + half];
            v[i] = (float)(first * co - second * si);
            v[i + half] = (float)(second * co + first * si);
        }
    }
}

static int allocate_runtime(minimindo_mimi *model)
{
    const size_t h = model->header->hidden, m = model->header->intermediate;
    const size_t positions = (size_t)model->max_frames * 2;
    const size_t cache = (size_t)model->header->layers * positions * h;
#define ALLOC(field, count) do { model->field = calloc((count), sizeof(*model->field)); if (model->field == NULL) return -1; } while (0)
    ALLOC(key_cache, cache); ALLOC(value_cache, cache);
    ALLOC(hidden, h); ALLOC(normed, h); ALLOC(query, h); ALLOC(key, h);
    ALLOC(value, h); ALLOC(attention, h); ALLOC(projected, h);
    ALLOC(mlp, m); ALLOC(scores, positions);
#undef ALLOC
    return 0;
}

minimindo_mimi *minimindo_mimi_open(const char *image_path, uint32_t max_frames,
                                    char *error, size_t error_capacity)
{
    if (image_path == NULL || max_frames == 0) {
        set_error(error, error_capacity, "invalid Mimi image or frame limit"); return NULL;
    }
    minimindo_mimi *model = calloc(1, sizeof(*model));
    if (model == NULL) return NULL;
    model->file = -1; model->file = open(image_path, O_RDONLY);
    struct stat status;
    if (model->file < 0 || fstat(model->file, &status) != 0 || status.st_size < MMO_HEADER_BYTES) {
        set_error(error, error_capacity, "cannot open Mimi image: %s", strerror(errno));
        minimindo_mimi_close(model); return NULL;
    }
    model->mapped_bytes = (size_t)status.st_size;
    model->mapping = mmap(NULL, model->mapped_bytes, PROT_READ, MAP_PRIVATE, model->file, 0);
    if (model->mapping == MAP_FAILED) {
        model->mapping = NULL; set_error(error, error_capacity, "cannot mmap Mimi image: %s", strerror(errno));
        minimindo_mimi_close(model); return NULL;
    }
    model->header = (const mimi_header *)model->mapping;
    static const unsigned char magic[8] = {'M','M','O','M','I','M','I','1'};
    if (memcmp(model->header->magic, magic, 8) != 0 || model->header->version != 1 ||
        model->header->header_bytes != MMO_HEADER_BYTES ||
        model->header->tensors_offset != MMO_HEADER_BYTES ||
        model->header->file_bytes != model->mapped_bytes ||
        model->header->codebooks != MINIMINDO_MIMI_CODEBOOKS ||
        model->header->hidden != model->header->heads * model->header->head_dim ||
        model->header->ratio_count != 4 || model->header->tensor_count != 135) {
        set_error(error, error_capacity, "invalid MiniMind-O Mimi image header");
        minimindo_mimi_close(model); return NULL;
    }
    model->layers = calloc(model->header->layers, sizeof(*model->layers));
    if (model->layers == NULL) { minimindo_mimi_close(model); return NULL; }
    uint64_t cursor = model->header->tensors_offset;
    const uint32_t h = model->header->hidden, m = model->header->intermediate;
    const uint32_t cb = model->header->codebook_size, cd = model->header->codebook_dim;
#define TAKE(target, type, rows, cols) if (take_tensor(model->mapping, model->mapped_bytes, &cursor, type, rows, cols, target) != 0) goto bad
    for (uint32_t i = 0; i < MINIMINDO_MIMI_CODEBOOKS; ++i) TAKE(&model->codebooks[i], MMO_F32, cb, cd);
    TAKE(&model->semantic_output, MMO_Q8_ROW, h, cd);
    TAKE(&model->acoustic_output, MMO_Q8_ROW, h, cd);
    TAKE(&model->upsample, MMO_Q8_ROW, h, 4);
    for (uint32_t i = 0; i < model->header->layers; ++i) {
        transformer_layer *l = &model->layers[i];
        TAKE(&l->input_weight, MMO_F32, 1, h); TAKE(&l->input_bias, MMO_F32, 1, h);
        TAKE(&l->post_weight, MMO_F32, 1, h); TAKE(&l->post_bias, MMO_F32, 1, h);
        TAKE(&l->attention_scale, MMO_F32, 1, h); TAKE(&l->mlp_scale, MMO_F32, 1, h);
        TAKE(&l->q, MMO_Q8_ROW, h, h); TAKE(&l->k, MMO_Q8_ROW, h, h);
        TAKE(&l->v, MMO_Q8_ROW, h, h); TAKE(&l->o, MMO_Q8_ROW, h, h);
        TAKE(&l->fc1, MMO_Q8_ROW, m, h); TAKE(&l->fc2, MMO_Q8_ROW, h, m);
    }
    static const uint32_t ins[14] = {512,1024,512,256,512,256,128,256,128,64,128,64,32,64};
    static const uint32_t outs[14] = {1024,512,256,512,256,128,256,128,64,128,64,32,64,1};
    static const uint32_t kernels[14] = {7,16,3,1,12,3,1,10,3,1,8,3,1,3};
    static const uint32_t strides[14] = {1,8,1,1,6,1,1,5,1,1,4,1,1,1};
    for (uint32_t i = 0; i < 14; ++i) {
        decoder_conv *conv = &model->convs[i];
        conv->in_channels = ins[i]; conv->out_channels = outs[i];
        conv->kernel = kernels[i]; conv->stride = strides[i];
        conv->transpose = i == 1 || i == 4 || i == 7 || i == 10;
        TAKE(&conv->weight, MMO_Q8_ROW, outs[i], ins[i] * kernels[i]);
        TAKE(&conv->bias, MMO_F32, 1, outs[i]);
        if (conv->transpose) {
            const size_t columns = (size_t)conv->in_channels * 2U;
            const size_t count = (size_t)conv->out_channels * conv->stride *
                                 columns;
            conv->phase_weights = malloc(count * sizeof(*conv->phase_weights));
            if (conv->phase_weights == NULL) goto bad;
            for (uint32_t out = 0; out < conv->out_channels; ++out) {
                const int8_t *weights;
                float unused_scale;
                q8_row(&conv->weight, out, &weights, &unused_scale);
                for (uint32_t phase = 0; phase < conv->stride; ++phase) {
                    int8_t *packed = conv->phase_weights +
                        ((size_t)out * conv->stride + phase) * columns;
                    for (uint32_t in = 0; in < conv->in_channels; ++in) {
                        packed[in] = weights[(size_t)in * conv->kernel + phase];
                        packed[conv->in_channels + in] =
                            weights[(size_t)in * conv->kernel +
                                    conv->stride + phase];
                    }
                }
            }
        }
    }
#undef TAKE
    if (cursor != model->mapped_bytes) goto bad;
    model->max_frames = max_frames;
    if (allocate_runtime(model) != 0) {
        set_error(error, error_capacity, "cannot allocate Mimi runtime");
        minimindo_mimi_close(model); return NULL;
    }
    return model;
bad:
    set_error(error, error_capacity, "invalid MiniMind-O Mimi tensor sequence");
    minimindo_mimi_close(model); return NULL;
}

void minimindo_mimi_close(minimindo_mimi *model)
{
    if (model == NULL) return;
    free(model->layers); free(model->key_cache); free(model->value_cache);
    free(model->hidden); free(model->normed); free(model->query); free(model->key);
    free(model->value); free(model->attention); free(model->projected);
    free(model->mlp); free(model->scores);
    for (uint32_t index = 0; index < 14; ++index)
        free(model->convs[index].phase_weights);
    if (model->mapping != NULL) munmap((void *)model->mapping, model->mapped_bytes);
    if (model->file >= 0) close(model->file);
    free(model);
}

uint32_t minimindo_mimi_sample_rate(const minimindo_mimi *model)
{ return model == NULL ? 0 : model->header->sample_rate; }

size_t minimindo_mimi_samples_for_frames(const minimindo_mimi *model, size_t frames)
{
    if (model == NULL) return 0;
    size_t samples = frames * 2;
    for (uint32_t i = 0; i < 4; ++i) samples *= model->header->ratios[i];
    return samples;
}

static int transformer(minimindo_mimi *model, float *sequence, uint32_t length)
{
    const uint32_t h = model->header->hidden, heads = model->header->heads;
    const uint32_t d = model->header->head_dim, window = model->header->sliding_window;
    memset(model->key_cache, 0, (size_t)model->header->layers * length * h * sizeof(float));
    memset(model->value_cache, 0, (size_t)model->header->layers * length * h * sizeof(float));
    for (uint32_t pos = 0; pos < length; ++pos) {
        for (uint32_t i = 0; i < h; ++i) model->hidden[i] = sequence[(size_t)i * length + pos];
        for (uint32_t li = 0; li < model->header->layers; ++li) {
            const transformer_layer *l = &model->layers[li];
            layer_norm(model->hidden, f32_data(&l->input_weight), f32_data(&l->input_bias),
                       model->normed, h, model->header->norm_epsilon);
            matvec(&l->q, model->normed, model->query);
            matvec(&l->k, model->normed, model->key);
            matvec(&l->v, model->normed, model->value);
            rope(model->query, heads, d, pos, model->header->rope_theta);
            rope(model->key, heads, d, pos, model->header->rope_theta);
            const size_t cache = ((size_t)li * length + pos) * h;
            memcpy(model->key_cache + cache, model->key, h * sizeof(float));
            memcpy(model->value_cache + cache, model->value, h * sizeof(float));
            const uint32_t first = pos + 1 > window ? pos + 1 - window : 0;
            for (uint32_t head = 0; head < heads; ++head) {
                const float *q = model->query + (size_t)head * d;
                double maximum = -INFINITY;
                for (uint32_t source = first; source <= pos; ++source) {
                    const size_t base = ((size_t)li * length + source) * h + (size_t)head * d;
                    double score = 0;
                    for (uint32_t j = 0; j < d; ++j) score += (double)q[j] * model->key_cache[base + j];
                    model->scores[source] = score / sqrt((double)d);
                    if (model->scores[source] > maximum) maximum = model->scores[source];
                }
                double denominator = 0;
                for (uint32_t source = first; source <= pos; ++source) {
                    model->scores[source] = exp(model->scores[source] - maximum);
                    denominator += model->scores[source];
                }
                for (uint32_t j = 0; j < d; ++j) {
                    double sum = 0;
                    for (uint32_t source = first; source <= pos; ++source) {
                        const size_t base = ((size_t)li * length + source) * h + (size_t)head * d;
                        sum += model->scores[source] * model->value_cache[base + j];
                    }
                    model->attention[(size_t)head * d + j] = (float)(sum / denominator);
                }
            }
            matvec(&l->o, model->attention, model->projected);
            const float *attention_scale = f32_data(&l->attention_scale);
            for (uint32_t i = 0; i < h; ++i) model->hidden[i] += model->projected[i] * attention_scale[i];
            layer_norm(model->hidden, f32_data(&l->post_weight), f32_data(&l->post_bias),
                       model->normed, h, model->header->norm_epsilon);
            matvec(&l->fc1, model->normed, model->mlp);
            for (uint32_t i = 0; i < model->header->intermediate; ++i) model->mlp[i] = gelu(model->mlp[i]);
            matvec(&l->fc2, model->mlp, model->projected);
            const float *mlp_scale = f32_data(&l->mlp_scale);
            for (uint32_t i = 0; i < h; ++i) model->hidden[i] += model->projected[i] * mlp_scale[i];
        }
        for (uint32_t i = 0; i < h; ++i) sequence[(size_t)i * length + pos] = model->hidden[i];
    }
    return 0;
}

static float *causal_windows(const float *input, const float *history,
                             uint32_t channels, uint32_t kernel,
                             uint32_t length)
{
    const size_t columns = (size_t)channels * kernel;
    float *windows = calloc((size_t)length * columns, sizeof(float));
    if (windows == NULL) return NULL;
    const uint32_t history_length = kernel - 1U;
    for (uint32_t t = 0; t < length; ++t) {
        float *window = windows + (size_t)t * columns;
        for (uint32_t in = 0; in < channels; ++in) {
            for (uint32_t k = 0; k < kernel; ++k) {
                const int32_t source =
                    (int32_t)t - (int32_t)history_length + (int32_t)k;
                if (source >= 0) {
                    window[(size_t)in * kernel + k] =
                        input[(size_t)in * length + (uint32_t)source];
                } else if (history != NULL) {
                    window[(size_t)in * kernel + k] =
                        history[(size_t)in * history_length +
                                (uint32_t)((int32_t)history_length + source)];
                }
            }
        }
    }
    return windows;
}

typedef struct {
    const float *input, *history;
    float *scratch;
    q8_windows output;
    uint32_t channels, kernel, length;
} causal_q8_context;

static void causal_q8_positions(void *opaque,size_t begin,size_t end)
{
    causal_q8_context *context=opaque;
    const uint32_t columns=context->channels*context->kernel;
    const uint32_t history_length=context->kernel-1U;
    for(size_t t=begin;t<end;++t){
        float *window=context->scratch+t*columns;
        for(uint32_t in=0;in<context->channels;++in)
            for(uint32_t k=0;k<context->kernel;++k){
                const int32_t source=(int32_t)t-(int32_t)history_length+(int32_t)k;
                float value=0.0f;
                if(source>=0)value=context->input[(size_t)in*context->length+(uint32_t)source];
                else if(context->history)value=context->history[(size_t)in*history_length+(uint32_t)((int32_t)history_length+source)];
                window[(size_t)in*context->kernel+k]=value;
            }
        context->output.scales[t]=quantize_i8(window,context->output.values+t*columns,columns);
    }
}

static q8_windows causal_q8_windows(const float *input, const float *history,
                                    uint32_t channels, uint32_t kernel,
                                    uint32_t length)
{
    const uint32_t columns = channels * kernel;
    q8_windows result = {
        .values = malloc((size_t)length * columns),
        .scales = malloc((size_t)length * sizeof(float))
    };
    float *scratch = malloc((size_t)length * columns * sizeof(float));
    if (result.values == NULL || result.scales == NULL || scratch == NULL) {
        free(result.values);
        free(result.scales);
        free(scratch);
        return (q8_windows){0};
    }
    causal_q8_context context={input,history,scratch,result,channels,kernel,length};
    minimindo_parallel_for(length,causal_q8_positions,&context);
    free(scratch);
    return result;
}

static float *conv1d(const decoder_conv *conv, const float *input, uint32_t length)
{
    float *output = malloc((size_t)conv->out_channels * length * sizeof(float));
    if (output == NULL) return NULL;
    float *windows = causal_windows(input, NULL, conv->in_channels,
                                    conv->kernel, length);
    if (windows == NULL) {
        free(output);
        return NULL;
    }
    const uint32_t columns = conv->in_channels * conv->kernel;
    const float *bias = f32_data(&conv->bias);
    for (uint32_t out = 0; out < conv->out_channels; ++out) {
        for (uint32_t t = 0; t < length; ++t) {
            const int8_t *weights;
            float scale;
            q8_row(&conv->weight, out, &weights, &scale);
            output[(size_t)out * length + t] = bias[out] + scale *
                q8_dot(weights, windows + (size_t)t * columns, columns);
        }
    }
    free(windows);
    return output;
}

static float *deconv_windows(const float *input, const float *history,
                             uint32_t channels, uint32_t length)
{
    const size_t columns = (size_t)channels * 2U;
    float *windows = calloc((size_t)length * columns, sizeof(float));
    if (windows == NULL) return NULL;
    for (uint32_t t = 0; t < length; ++t) {
        float *window = windows + (size_t)t * columns;
        for (uint32_t in = 0; in < channels; ++in) {
            window[in] = input[(size_t)in * length + t];
            if (t != 0U)
                window[channels + in] =
                    input[(size_t)in * length + t - 1U];
            else if (history != NULL)
                window[channels + in] = history[in];
        }
    }
    return windows;
}

typedef struct {
    const float *input, *history;
    float *scratch;
    q8_windows output;
    uint32_t channels, length;
} deconv_q8_context;

static void deconv_q8_positions(void *opaque,size_t begin,size_t end)
{
    deconv_q8_context *context=opaque;const uint32_t columns=context->channels*2U;
    for(size_t t=begin;t<end;++t){float *window=context->scratch+t*columns;
        for(uint32_t in=0;in<context->channels;++in){window[in]=context->input[(size_t)in*context->length+t];
            window[context->channels+in]=t?context->input[(size_t)in*context->length+t-1U]:context->history?context->history[in]:0.0f;}
        context->output.scales[t]=quantize_i8(window,context->output.values+t*columns,columns);}
}

static q8_windows deconv_q8_windows(const float *input, const float *history,
                                    uint32_t channels, uint32_t length)
{
    const uint32_t columns = channels * 2U;
    q8_windows result = {
        .values = malloc((size_t)length * columns),
        .scales = malloc((size_t)length * sizeof(float))
    };
    float *scratch = malloc((size_t)length * columns * sizeof(float));
    if (result.values == NULL || result.scales == NULL || scratch == NULL) {
        free(result.values);
        free(result.scales);
        free(scratch);
        return (q8_windows){0};
    }
    deconv_q8_context context={input,history,scratch,result,channels,length};
    minimindo_parallel_for(length,deconv_q8_positions,&context);
    free(scratch);
    return result;
}

static float *conv_transpose(const decoder_conv *conv, const float *input,
                             uint32_t length, uint32_t *output_length)
{
    const uint32_t result_length = length * conv->stride;
    float *output = malloc((size_t)conv->out_channels * result_length * sizeof(float));
    if (output == NULL) return NULL;
    const uint32_t columns = conv->in_channels * 2U;
    float *windows = deconv_windows(input, NULL, conv->in_channels, length);
    if (windows == NULL || conv->phase_weights == NULL ||
        conv->kernel != conv->stride * 2U) {
        free(windows);
        free(output);
        return NULL;
    }
    const float *bias = f32_data(&conv->bias);
    for (uint32_t out = 0; out < conv->out_channels; ++out) {
        for (uint32_t t = 0; t < length; ++t) {
            const int8_t *unused_weights;
            float scale;
            q8_row(&conv->weight, out, &unused_weights, &scale);
            const float *window = windows + (size_t)t * columns;
            float *row = output + (size_t)out * result_length +
                         (size_t)t * conv->stride;
            for (uint32_t phase = 0; phase < conv->stride; ++phase) {
                const int8_t *weights = conv->phase_weights +
                    ((size_t)out * conv->stride + phase) * columns;
                row[phase] = bias[out] + scale *
                    q8_dot(weights, window, columns);
            }
        }
    }
    free(windows);
    *output_length = result_length;
    return output;
}

static void activate_elu_range(void *opaque,size_t begin,size_t end)
{
    float *values=opaque;for(size_t index=begin;index<end;++index)values[index]=elu(values[index]);
}

static void activate_elu(float *values, size_t count)
{
    minimindo_parallel_for(count,activate_elu_range,values);
}

static float *residual_block(minimindo_mimi *model, uint32_t first_index,
                             float *input, uint32_t channels, uint32_t length)
{
    const size_t count = (size_t)channels * length;
    float *skip = malloc(count * sizeof(float));
    if (skip == NULL) return NULL;
    memcpy(skip, input, count * sizeof(float));
    activate_elu(input, (size_t)channels * length);
    float *first = conv1d(&model->convs[first_index], input, length);
    if (first == NULL) { free(skip); return NULL; }
    activate_elu(first, (size_t)model->convs[first_index].out_channels * length);
    float *second = conv1d(&model->convs[first_index + 1], first, length);
    free(first);
    if (second == NULL) { free(skip); return NULL; }
    for (size_t i = 0; i < count; ++i) second[i] += skip[i];
    free(skip);
    return second;
}

static int transformer_stream_pair(minimindo_mimi *model, uint32_t position,
                                   const float *input, float *output,
                                   float *scratch)
{
    const uint32_t h = model->header->hidden;
    const uint32_t heads = model->header->heads;
    const uint32_t d = model->header->head_dim;
    const uint32_t window = model->header->sliding_window;
    const uint32_t cache_stride = model->max_frames * 2U;
    const uint32_t m = model->header->intermediate;
    if (position + 1U >= cache_stride) return -1;
    float *hidden = scratch;
    float *normed = hidden + (size_t)h * 2U;
    float *query = normed + (size_t)h * 2U;
    float *key = query + (size_t)h * 2U;
    float *value = key + (size_t)h * 2U;
    float *attention = value + (size_t)h * 2U;
    float *projected = attention + (size_t)h * 2U;
    float *mlp = projected + (size_t)h * 2U;
    for (uint32_t index = 0; index < h; ++index) {
        hidden[index] = input[(size_t)index * 2U];
        hidden[h + index] = input[(size_t)index * 2U + 1U];
    }
    for (uint32_t li = 0; li < model->header->layers; ++li) {
        const transformer_layer *layer = &model->layers[li];
        for (uint32_t slot = 0; slot < 2U; ++slot)
            layer_norm(hidden + (size_t)slot * h,
                       f32_data(&layer->input_weight),
                       f32_data(&layer->input_bias),
                       normed + (size_t)slot * h, h,
                       model->header->norm_epsilon);
        matvec_pair(&layer->q, normed, query);
        matvec_pair(&layer->k, normed, key);
        matvec_pair(&layer->v, normed, value);
        for (uint32_t slot = 0; slot < 2U; ++slot) {
            const uint32_t absolute = position + slot;
            rope(query + (size_t)slot * h, heads, d, absolute,
                 model->header->rope_theta);
            rope(key + (size_t)slot * h, heads, d, absolute,
                 model->header->rope_theta);
            const size_t cache =
                ((size_t)li * cache_stride + absolute) * h;
            memcpy(model->key_cache + cache, key + (size_t)slot * h,
                   (size_t)h * sizeof(float));
            memcpy(model->value_cache + cache, value + (size_t)slot * h,
                   (size_t)h * sizeof(float));
            const uint32_t first = absolute + 1U > window
                ? absolute + 1U - window : 0U;
            for (uint32_t head = 0; head < heads; ++head) {
                const float *head_query = query + (size_t)slot * h +
                                          (size_t)head * d;
                double maximum = -INFINITY;
                for (uint32_t source = first; source <= absolute; ++source) {
                    const size_t base =
                        ((size_t)li * cache_stride + source) * h +
                        (size_t)head * d;
                    double score = 0.0;
                    for (uint32_t index = 0; index < d; ++index)
                        score += (double)head_query[index] *
                                 model->key_cache[base + index];
                    model->scores[source] =
                        (float)(score / sqrt((double)d));
                    if (model->scores[source] > maximum)
                        maximum = model->scores[source];
                }
                double denominator = 0.0;
                for (uint32_t source = first; source <= absolute; ++source) {
                    model->scores[source] =
                        (float)exp(model->scores[source] - maximum);
                    denominator += model->scores[source];
                }
                for (uint32_t index = 0; index < d; ++index) {
                    double sum = 0.0;
                    for (uint32_t source = first; source <= absolute; ++source) {
                        const size_t base =
                            ((size_t)li * cache_stride + source) * h +
                            (size_t)head * d;
                        sum += model->scores[source] *
                               model->value_cache[base + index];
                    }
                    attention[(size_t)slot * h + (size_t)head * d + index] =
                        (float)(sum / denominator);
                }
            }
        }
        matvec_pair(&layer->o, attention, projected);
        const float *attention_scale = f32_data(&layer->attention_scale);
        for (uint32_t slot = 0; slot < 2U; ++slot) {
            for (uint32_t index = 0; index < h; ++index)
                hidden[(size_t)slot * h + index] +=
                    projected[(size_t)slot * h + index] *
                    attention_scale[index];
            layer_norm(hidden + (size_t)slot * h,
                       f32_data(&layer->post_weight),
                       f32_data(&layer->post_bias),
                       normed + (size_t)slot * h, h,
                       model->header->norm_epsilon);
        }
        matvec_pair(&layer->fc1, normed, mlp);
        for (uint32_t index = 0; index < m * 2U; ++index)
            mlp[index] = gelu(mlp[index]);
        matvec_pair(&layer->fc2, mlp, projected);
        const float *mlp_scale = f32_data(&layer->mlp_scale);
        for (uint32_t slot = 0; slot < 2U; ++slot)
            for (uint32_t index = 0; index < h; ++index)
                hidden[(size_t)slot * h + index] +=
                    projected[(size_t)slot * h + index] * mlp_scale[index];
    }
    for (uint32_t index = 0; index < h; ++index) {
        output[(size_t)index * 2U] = hidden[index];
        output[(size_t)index * 2U + 1U] = hidden[h + index];
    }
    return 0;
}

typedef struct {
    const decoder_conv *conv;
    q8_windows windows;
    float *output;
    uint32_t length;
} stream_conv_context;

static void stream_conv_cells(void *opaque,size_t begin,size_t end)
{
    stream_conv_context *context=opaque;const decoder_conv *conv=context->conv;
    const uint32_t columns=conv->in_channels*conv->kernel;const float *bias=f32_data(&conv->bias);
    for(size_t cell=begin;cell<end;++cell){const uint32_t out=(uint32_t)(cell/context->length),t=(uint32_t)(cell%context->length);
        const int8_t *weights;float scale;q8_row(&conv->weight,out,&weights,&scale);
        context->output[(size_t)out*context->length+t]=bias[out]+scale*context->windows.scales[t]*
            q8_i8_dot(weights,context->windows.values+(size_t)t*columns,columns);}
}

static float *stream_conv1d(mimi_conv_stream *state,
                            const decoder_conv *conv, const float *input,
                            uint32_t length)
{
    float *output = malloc((size_t)conv->out_channels * length * sizeof(float));
    if (output == NULL) return NULL;
    const uint32_t history_length = conv->kernel - 1U;
    q8_windows windows = causal_q8_windows(
        input, state->history, conv->in_channels, conv->kernel, length);
    if (windows.values == NULL) {
        free(output);
        return NULL;
    }
    stream_conv_context context={conv,windows,output,length};
    minimindo_parallel_for((size_t)conv->out_channels*length,stream_conv_cells,&context);
    free(windows.values);
    free(windows.scales);
    if (history_length != 0U) {
        for (uint32_t in = 0; in < conv->in_channels; ++in) {
            float *history = state->history + (size_t)in * history_length;
            const float *row = input + (size_t)in * length;
            if (length >= history_length) {
                memcpy(history, row + length - history_length,
                       (size_t)history_length * sizeof(float));
            } else {
                memmove(history, history + length,
                        (size_t)(history_length - length) * sizeof(float));
                memcpy(history + history_length - length, row,
                       (size_t)length * sizeof(float));
            }
        }
    }
    return output;
}

static void stream_deconv_cells(void *opaque,size_t begin,size_t end)
{
    stream_conv_context *context=opaque;const decoder_conv *conv=context->conv;
    const uint32_t columns=conv->in_channels*2U;const float *bias=f32_data(&conv->bias);
    const uint32_t result_length=context->length*conv->stride;
    for(size_t cell=begin;cell<end;++cell){const uint32_t out=(uint32_t)(cell/context->length),t=(uint32_t)(cell%context->length);
        const int8_t *unused;float scale;q8_row(&conv->weight,out,&unused,&scale);
        const int8_t *window=context->windows.values+(size_t)t*columns;
        float *row=context->output+(size_t)out*result_length+(size_t)t*conv->stride;
        for(uint32_t phase=0;phase<conv->stride;++phase){const int8_t *weights=conv->phase_weights+
            ((size_t)out*conv->stride+phase)*columns;row[phase]=bias[out]+scale*context->windows.scales[t]*q8_i8_dot(weights,window,columns);}}
}

static float *stream_conv_transpose(mimi_conv_stream *state,
                                    const decoder_conv *conv,
                                    const float *input, uint32_t length,
                                    uint32_t *output_length)
{
    const uint32_t result_length = length * conv->stride;
    float *output = malloc((size_t)conv->out_channels * result_length *
                           sizeof(float));
    if (output == NULL) return NULL;
    q8_windows windows = deconv_q8_windows(
        input, state->history, conv->in_channels, length);
    if (windows.values == NULL || conv->phase_weights == NULL ||
        conv->kernel != conv->stride * 2U) {
        free(windows.values);
        free(windows.scales);
        free(output);
        return NULL;
    }
    stream_conv_context context={conv,windows,output,length};
    minimindo_parallel_for((size_t)conv->out_channels*length,stream_deconv_cells,&context);
    free(windows.values);
    free(windows.scales);
    for (uint32_t in = 0; in < conv->in_channels; ++in)
        state->history[in] = input[(size_t)in * length + length - 1U];
    *output_length = result_length;
    return output;
}

static float *stream_residual_block(minimindo_mimi_stream *stream,
                                    uint32_t first_index, float *input,
                                    uint32_t channels, uint32_t length)
{
    const size_t count = (size_t)channels * length;
    float *skip = malloc(count * sizeof(float));
    if (skip == NULL) return NULL;
    memcpy(skip, input, count * sizeof(float));
    activate_elu(input, count);
    float *first = stream_conv1d(&stream->convs[first_index],
                                 &stream->model->convs[first_index],
                                 input, length);
    if (first == NULL) {
        free(skip);
        return NULL;
    }
    activate_elu(first,
                 (size_t)stream->model->convs[first_index].out_channels *
                 length);
    float *second = stream_conv1d(&stream->convs[first_index + 1U],
                                  &stream->model->convs[first_index + 1U],
                                  first, length);
    free(first);
    if (second == NULL) {
        free(skip);
        return NULL;
    }
    for (size_t index = 0; index < count; ++index)
        second[index] += skip[index];
    free(skip);
    return second;
}

minimindo_mimi_stream *minimindo_mimi_stream_open(
    minimindo_mimi *model, char *error, size_t error_capacity)
{
    if (model == NULL) {
        set_error(error, error_capacity, "invalid Mimi stream model");
        return NULL;
    }
    minimindo_mimi_stream *stream = calloc(1, sizeof(*stream));
    if (stream == NULL) return NULL;
    stream->model = model;
    const uint32_t h = model->header->hidden;
    stream->upsample_tail = calloc((size_t)h * 2U, sizeof(float));
    stream->transformer_pair = malloc(
        ((size_t)h * 16U + (size_t)model->header->intermediate * 2U) *
        sizeof(float));
    if (stream->upsample_tail == NULL || stream->transformer_pair == NULL)
        goto oom;
    for (uint32_t index = 0; index < 14; ++index) {
        const decoder_conv *conv = &model->convs[index];
        if (conv->transpose) {
            stream->convs[index].history =
                calloc(conv->in_channels, sizeof(float));
            if (stream->convs[index].history == NULL) goto oom;
        } else if (conv->kernel > 1U) {
            const size_t count = (size_t)conv->in_channels *
                                 (conv->kernel - 1U);
            stream->convs[index].history = calloc(count, sizeof(float));
            if (stream->convs[index].history == NULL) goto oom;
        }
    }
    minimindo_mimi_stream_reset(stream);
    return stream;
oom:
    set_error(error, error_capacity, "cannot allocate Mimi stream state");
    minimindo_mimi_stream_close(stream);
    return NULL;
}

void minimindo_mimi_stream_reset(minimindo_mimi_stream *stream)
{
    if (stream == NULL) return;
    minimindo_mimi *model = stream->model;
    const uint32_t h = model->header->hidden;
    stream->frames = 0;
    stream->transformer_positions = 0;
    memset(stream->upsample_tail, 0, (size_t)h * 2U * sizeof(float));
    const size_t positions = (size_t)model->max_frames * 2U;
    const size_t cache = (size_t)model->header->layers * positions * h;
    memset(model->key_cache, 0, cache * sizeof(float));
    memset(model->value_cache, 0, cache * sizeof(float));
    for (uint32_t index = 0; index < 14; ++index) {
        const decoder_conv *conv = &model->convs[index];
        if (conv->transpose) {
            memset(stream->convs[index].history, 0,
                   (size_t)conv->in_channels * sizeof(float));
        } else if (conv->kernel > 1U) {
            memset(stream->convs[index].history, 0,
                   (size_t)conv->in_channels *
                   (conv->kernel - 1U) * sizeof(float));
        }
    }
}

void minimindo_mimi_stream_close(minimindo_mimi_stream *stream)
{
    if (stream == NULL) return;
    free(stream->upsample_tail);
    free(stream->transformer_pair);
    for (uint32_t index = 0; index < 14; ++index) {
        free(stream->convs[index].history);
    }
    free(stream);
}

typedef struct {
    minimindo_mimi *model;
    minimindo_mimi_stream *stream;
    const float *semantic,*acoustic;
    float *sequence;
    uint32_t frames,sequence_length;
} upsample_parallel_context;

static void upsample_channels(void *opaque,size_t begin,size_t end)
{
    upsample_parallel_context *context=opaque;
    for(size_t channel=begin;channel<end;++channel){const int8_t *weights;float scale;
        q8_row(&context->model->upsample,(uint32_t)channel,&weights,&scale);
        float *row=context->sequence+channel*context->sequence_length;
        float *tail=context->stream->upsample_tail+channel*2U;
        row[0]+=tail[0];row[1]+=tail[1];tail[0]=0.0f;tail[1]=0.0f;
        for(uint32_t t=0;t<context->frames;++t){const float value=
            context->semantic[channel*context->frames+t]+context->acoustic[channel*context->frames+t];
            for(uint32_t k=0;k<4U;++k){const uint32_t target=t*2U+k;const float contribution=value*weights[k]*scale;
                if(target<context->sequence_length)row[target]+=contribution;else tail[target-context->sequence_length]+=contribution;}}
    }
}

int minimindo_mimi_stream_decode(minimindo_mimi_stream *stream,
                                 const uint32_t *codes, size_t frames,
                                 float *audio, size_t audio_capacity,
                                 size_t *audio_samples,
                                 char *error, size_t error_capacity)
{
    const int profile = getenv("MINIMINDO_MIMI_STREAM_PROFILE") != NULL;
    const double profile_start = profile ? seconds() : 0.0;
    double profile_previous = profile_start;
    double profile_codebooks = 0.0, profile_transformer = 0.0;
    double profile_conv0 = 0.0, profile_stages[4] = {0};
    if (stream == NULL || codes == NULL || audio == NULL || frames == 0U ||
        stream->frames + frames > stream->model->max_frames ||
        audio_capacity < minimindo_mimi_samples_for_frames(stream->model,
                                                           frames)) {
        set_error(error, error_capacity, "invalid Mimi stream arguments");
        return -1;
    }
    minimindo_mimi *model = stream->model;
    const uint32_t h = model->header->hidden;
    const uint32_t cd = model->header->codebook_dim;
    const uint32_t length = (uint32_t)frames;
    float *semantic = calloc((size_t)h * frames, sizeof(float));
    float *acoustic = calloc((size_t)h * frames, sizeof(float));
    float *projected = malloc((size_t)h * sizeof(float));
    if (semantic == NULL || acoustic == NULL || projected == NULL) goto oom;
    for (uint32_t codebook = 0; codebook < MINIMINDO_MIMI_CODEBOOKS;
         ++codebook) {
        const float *book = f32_data(&model->codebooks[codebook]);
        const tensor *projection = codebook == 0U
            ? &model->semantic_output : &model->acoustic_output;
        float *destination = codebook == 0U ? semantic : acoustic;
        for (uint32_t t = 0; t < length; ++t) {
            const uint32_t code = codes[(size_t)codebook * frames + t];
            if (code >= model->header->codebook_size) {
                set_error(error, error_capacity, "Mimi code is out of range");
                free(semantic);
                free(acoustic);
                free(projected);
                return -1;
            }
            matvec(projection, book + (size_t)code * cd, projected);
            for (uint32_t index = 0; index < h; ++index) {
                float *slot = destination + (size_t)index * frames + t;
                if (codebook <= 1U) *slot = projected[index];
                else *slot += projected[index];
            }
        }
    }
    free(projected);
    projected = NULL;
    if (profile) {
        const double now = seconds();
        profile_codebooks = now - profile_previous;
        profile_previous = now;
    }
    const uint32_t sequence_length = length * 2U;
    float *sequence = calloc((size_t)h * sequence_length, sizeof(float));
    if (sequence == NULL) goto oom;
    upsample_parallel_context upsample_context={model,stream,semantic,acoustic,sequence,length,sequence_length};
    minimindo_parallel_for(h,upsample_channels,&upsample_context);
    free(semantic);
    free(acoustic);
    semantic = NULL;
    acoustic = NULL;
    const size_t scratch_floats = (size_t)h * 14U +
                                  (size_t)model->header->intermediate * 2U;
    float *position_pair = stream->transformer_pair + scratch_floats;
    for (uint32_t t = 0; t < sequence_length; t += 2U) {
        for (uint32_t channel = 0; channel < h; ++channel) {
            position_pair[(size_t)channel * 2U] =
                sequence[(size_t)channel * sequence_length + t];
            position_pair[(size_t)channel * 2U + 1U] =
                sequence[(size_t)channel * sequence_length + t + 1U];
        }
        if (transformer_stream_pair(model, stream->transformer_positions,
                                    position_pair, position_pair,
                                    stream->transformer_pair) != 0) {
            free(sequence);
            set_error(error, error_capacity,
                      "Mimi stream transformer context exhausted");
            return -1;
        }
        stream->transformer_positions += 2U;
        for (uint32_t channel = 0; channel < h; ++channel) {
            sequence[(size_t)channel * sequence_length + t] =
                position_pair[(size_t)channel * 2U];
            sequence[(size_t)channel * sequence_length + t + 1U] =
                position_pair[(size_t)channel * 2U + 1U];
        }
    }
    if (profile) {
        const double now = seconds();
        profile_transformer = now - profile_previous;
        profile_previous = now;
    }
    float *current = stream_conv1d(&stream->convs[0], &model->convs[0],
                                   sequence, sequence_length);
    free(sequence);
    if (current == NULL) goto oom_simple;
    if (profile) {
        const double now = seconds();
        profile_conv0 = now - profile_previous;
        profile_previous = now;
    }
    uint32_t current_length = sequence_length;
    static const uint32_t deconvs[4] = {1, 4, 7, 10};
    static const uint32_t residuals[4] = {2, 5, 8, 11};
    for (uint32_t stage = 0; stage < 4; ++stage) {
        const uint32_t deconv = deconvs[stage];
        activate_elu(current,
                     (size_t)model->convs[deconv].in_channels *
                     current_length);
        uint32_t next_length = 0;
        float *next = stream_conv_transpose(&stream->convs[deconv],
                                            &model->convs[deconv], current,
                                            current_length, &next_length);
        free(current);
        if (next == NULL) goto oom_simple;
        float *residual = stream_residual_block(
            stream, residuals[stage], next,
            model->convs[deconv].out_channels, next_length);
        free(next);
        if (residual == NULL) goto oom_simple;
        current = residual;
        current_length = next_length;
        if (profile) {
            const double now = seconds();
            profile_stages[stage] = now - profile_previous;
            profile_previous = now;
        }
    }
    activate_elu(current,
                 (size_t)model->convs[13].in_channels * current_length);
    float *final = stream_conv1d(&stream->convs[13], &model->convs[13],
                                 current, current_length);
    free(current);
    if (final == NULL) goto oom_simple;
    memcpy(audio, final, (size_t)current_length * sizeof(float));
    free(final);
    stream->frames += length;
    if (audio_samples != NULL) *audio_samples = current_length;
    if (profile) {
        fprintf(stderr,
                "MIMI_STREAM_PROFILE frame=%u codebooks_ms=%.3f "
                "transformer_ms=%.3f conv0_ms=%.3f stage1_ms=%.3f "
                "stage2_ms=%.3f stage3_ms=%.3f stage4_ms=%.3f "
                "final_ms=%.3f total_ms=%.3f\n",
                stream->frames, profile_codebooks * 1000.0,
                profile_transformer * 1000.0, profile_conv0 * 1000.0,
                profile_stages[0] * 1000.0, profile_stages[1] * 1000.0,
                profile_stages[2] * 1000.0, profile_stages[3] * 1000.0,
                (seconds() - profile_previous) * 1000.0,
                (seconds() - profile_start) * 1000.0);
    }
    return 0;
oom:
    free(semantic);
    free(acoustic);
    free(projected);
oom_simple:
    set_error(error, error_capacity, "out of memory in Mimi stream decoder");
    return -3;
}

int minimindo_mimi_decode(minimindo_mimi *model, const uint32_t *codes,
                          size_t frames, float *audio, size_t audio_capacity,
                          size_t *audio_samples, char *error, size_t error_capacity)
{
    const int profile = getenv("MINIMINDO_MIMI_PROFILE") != NULL;
    const double profile_start = profile ? seconds() : 0.0;
    double profile_previous = profile_start;
    double profile_codebooks = 0.0, profile_transformer = 0.0;
    double profile_conv0 = 0.0, profile_stages[4] = {0};
    if (model == NULL || codes == NULL || audio == NULL || frames == 0 ||
        frames > model->max_frames ||
        audio_capacity < minimindo_mimi_samples_for_frames(model, frames)) {
        set_error(error, error_capacity, "invalid Mimi decode arguments"); return -1;
    }
    const uint32_t h = model->header->hidden, cd = model->header->codebook_dim;
    const uint32_t length = (uint32_t)frames;
    float *quantized = calloc((size_t)cd * frames, sizeof(float));
    float *semantic = malloc((size_t)h * frames * sizeof(float));
    float *acoustic = malloc((size_t)h * frames * sizeof(float));
    float *vector = malloc(cd * sizeof(float));
    float *projected = malloc(h * sizeof(float));
    if (!quantized || !semantic || !acoustic || !vector || !projected) goto oom;
    for (uint32_t codebook = 0; codebook < MINIMINDO_MIMI_CODEBOOKS; ++codebook) {
        memset(quantized, 0, (size_t)cd * frames * sizeof(float));
        const float *book = f32_data(&model->codebooks[codebook]);
        for (uint32_t t = 0; t < length; ++t) {
            const uint32_t code = codes[(size_t)codebook * frames + t];
            if (code >= model->header->codebook_size) {
                set_error(error, error_capacity, "Mimi code is out of range");
                free(quantized); free(semantic); free(acoustic); free(vector); free(projected); return -1;
            }
            const float *row = book + (size_t)code * cd;
            for (uint32_t i = 0; i < cd; ++i) quantized[(size_t)i * frames + t] = row[i];
        }
        const tensor *projection = codebook == 0 ? &model->semantic_output : &model->acoustic_output;
        float *destination = codebook == 0 ? semantic : acoustic;
        if (codebook == 1) memset(acoustic, 0, (size_t)h * frames * sizeof(float));
        for (uint32_t t = 0; t < length; ++t) {
            for (uint32_t i = 0; i < cd; ++i) vector[i] = quantized[(size_t)i * frames + t];
            matvec(projection, vector, projected);
            for (uint32_t i = 0; i < h; ++i) {
                float *slot = destination + (size_t)i * frames + t;
                if (codebook <= 1) *slot = projected[i]; else *slot += projected[i];
            }
        }
    }
    if (profile) {
        const double now = seconds();
        profile_codebooks = now - profile_previous;
        profile_previous = now;
    }
    free(quantized); free(vector); free(projected);
    float *sequence = calloc((size_t)h * frames * 2, sizeof(float));
    if (sequence == NULL) { free(semantic); free(acoustic); goto oom_simple; }
    for (uint32_t channel = 0; channel < h; ++channel) {
        const int8_t *weights; float scale;
        q8_row(&model->upsample, channel, &weights, &scale);
        float *out = sequence + (size_t)channel * frames * 2;
        for (uint32_t t = 0; t < length; ++t) {
            const float value = semantic[(size_t)channel * frames + t] +
                                acoustic[(size_t)channel * frames + t];
            for (uint32_t k = 0; k < 4; ++k) {
                const uint32_t n = t * 2 + k;
                if (n < frames * 2) out[n] += value * weights[k] * scale;
            }
        }
    }
    free(semantic); free(acoustic);
    transformer(model, sequence, length * 2);
    if (profile) {
        const double now = seconds();
        profile_transformer = now - profile_previous;
        profile_previous = now;
    }
    uint32_t current_length = length * 2;
    float *current = conv1d(&model->convs[0], sequence, current_length);
    free(sequence);
    if (current == NULL) goto oom_simple;
    if (profile) {
        const double now = seconds();
        profile_conv0 = now - profile_previous;
        profile_previous = now;
    }
    const uint32_t deconvs[4] = {1,4,7,10};
    const uint32_t residuals[4] = {2,5,8,11};
    for (uint32_t stage = 0; stage < 4; ++stage) {
        activate_elu(current, (size_t)model->convs[deconvs[stage]].in_channels * current_length);
        uint32_t next_length = 0;
        float *next = conv_transpose(&model->convs[deconvs[stage]], current,
                                     current_length, &next_length);
        free(current); if (next == NULL) goto oom_simple;
        float *residual = residual_block(model, residuals[stage], next,
                                         model->convs[deconvs[stage]].out_channels,
                                         next_length);
        free(next); if (residual == NULL) goto oom_simple;
        current = residual; current_length = next_length;
        if (profile) {
            const double now = seconds();
            profile_stages[stage] = now - profile_previous;
            profile_previous = now;
        }
    }
    activate_elu(current, (size_t)model->convs[13].in_channels * current_length);
    float *final = conv1d(&model->convs[13], current, current_length);
    free(current); if (final == NULL) goto oom_simple;
    memcpy(audio, final, current_length * sizeof(float)); free(final);
    if (audio_samples != NULL) *audio_samples = current_length;
    if (profile) {
        fprintf(stderr,
                "MIMI_PROFILE frames=%zu codebooks_ms=%.3f "
                "transformer_ms=%.3f conv0_ms=%.3f stage1_ms=%.3f "
                "stage2_ms=%.3f stage3_ms=%.3f stage4_ms=%.3f "
                "final_ms=%.3f total_ms=%.3f\n",
                frames, profile_codebooks * 1000.0,
                profile_transformer * 1000.0, profile_conv0 * 1000.0,
                profile_stages[0] * 1000.0, profile_stages[1] * 1000.0,
                profile_stages[2] * 1000.0, profile_stages[3] * 1000.0,
                (seconds() - profile_previous) * 1000.0,
                (seconds() - profile_start) * 1000.0);
    }
    return 0;
oom:
    free(quantized); free(semantic); free(acoustic); free(vector); free(projected);
oom_simple:
    set_error(error, error_capacity, "out of memory in Mimi decoder"); return -3;
}
