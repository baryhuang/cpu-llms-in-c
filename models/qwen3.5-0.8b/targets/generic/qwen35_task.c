/* Qwen3.5-0.8B prompt-defined task runtime (generic target).
 *
 * Executes the QW35TSK1 image: full 24-layer hybrid text graph with
 * Gated-DeltaNet linear layers, gated full attention, and the tied
 * Q4 embedding/output table. Inputs are token ids until the C
 * tokenizer lands; the caller may pass an answer set whose rows are
 * scored after prefill (prompt-defined outputs), and no other head
 * rows are touched.
 */

#define _POSIX_C_SOURCE 200809L

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
    KIND_Q4 = 4,
    KIND_Q8 = 5,
    GROUP = 128,
    Q4_RECORD = 66,
    Q8_RECORD = 130,
    MAX_ANSWERS = 64,
    MAX_TOKENS = 4096
};

#pragma pack(push, 1)
struct image_header {
    char magic[8];
    uint32_t version;
    uint32_t descriptor_count;
    uint32_t layer_count;
    uint32_t full_interval;
    uint32_t vocab_rows;
    uint32_t hidden;
    uint32_t group_size;
    uint32_t reserved;
    uint64_t directory_offset;
    uint64_t data_offset;
    uint64_t file_bytes;
    uint64_t reserved64;
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

/* Graph constants for the pinned Qwen3.5-0.8B checkpoint. */
enum {
    HIDDEN = 1024,
    LAYERS = 24,
    FULL_INTERVAL = 4,
    LK_HEADS = 16,
    LV_HEADS = 16,
    LK_DIM = 128,
    LV_DIM = 128,
    KEY_DIM = LK_HEADS * LK_DIM,
    VALUE_DIM = LV_HEADS * LV_DIM,
    CONV_DIM = 2 * KEY_DIM + VALUE_DIM,
    CONV_KERNEL = 4,
    Q_HEADS = 8,
    KV_HEADS = 2,
    HEAD_DIM = 256,
    QUERY_SIZE = Q_HEADS * HEAD_DIM,
    KV_SIZE = KV_HEADS * HEAD_DIM,
    ROTARY_DIM = 64,
    INTERMEDIATE = 3584
};

struct mapped_image {
    int fd;
    size_t bytes;
    const uint8_t *base;
    const struct image_header *header;
    const struct descriptor *entries;
};

struct linear_layer {
    const struct descriptor *input_norm;
    const struct descriptor *in_proj_qkv;
    const struct descriptor *in_proj_z;
    const struct descriptor *in_proj_b;
    const struct descriptor *in_proj_a;
    const struct descriptor *conv;
    const struct descriptor *a_log;
    const struct descriptor *dt_bias;
    const struct descriptor *gated_norm;
    const struct descriptor *out_proj;
};

struct full_layer {
    const struct descriptor *input_norm;
    const struct descriptor *q_proj;
    const struct descriptor *k_proj;
    const struct descriptor *v_proj;
    const struct descriptor *q_norm;
    const struct descriptor *k_norm;
    const struct descriptor *o_proj;
};

struct layer {
    int is_full;
    struct linear_layer linear;
    struct full_layer full;
    const struct descriptor *post_norm;
    const struct descriptor *gate_proj;
    const struct descriptor *up_proj;
    const struct descriptor *down_proj;
};

struct tokenizer {
    int present;
    uint32_t token_count;
    const uint32_t *token_offsets;
    const uint8_t *token_bytes;
    const uint32_t *byte_tokens;
    const uint32_t *merges; /* [merge_count][4] = left,right,merged,rank sorted by (l,r) */
    uint32_t merge_count;
    const uint32_t *special_ids;
    const uint32_t *special_offsets;
    const uint8_t *special_bytes;
    uint32_t special_count;
};

struct model {
    struct mapped_image image;
    struct layer layers[LAYERS];
    const struct descriptor *embed;
    const struct descriptor *final_norm;
    struct tokenizer tokenizer;
};

struct state {
    /* DeltaNet per linear layer: recurrent state and conv ring window. */
    float *recurrent[LAYERS]; /* [LV_HEADS][LK_DIM][LV_DIM] */
    float *conv_window[LAYERS]; /* [CONV_DIM][CONV_KERNEL] rolling */
    /* KV cache per full layer. */
    float *key_cache[LAYERS];   /* [max_seq][KV_HEADS][HEAD_DIM] */
    float *value_cache[LAYERS];
    float *rope_cos; /* [max_seq][ROTARY_DIM] */
    float *rope_sin;
    size_t max_seq;
};

struct workspace {
    float hidden[HIDDEN];
    float normed[HIDDEN];
    float mixed[CONV_DIM];
    float post_conv[CONV_DIM];
    float z[VALUE_DIM];
    float b[LV_HEADS];
    float a[LV_HEADS];
    float queries[KEY_DIM];
    float keys[KEY_DIM];
    float core[VALUE_DIM];
    float gated[VALUE_DIM];
    float q_and_gate[2 * QUERY_SIZE];
    float query[QUERY_SIZE];
    float attn_gate[QUERY_SIZE];
    float key_now[KV_SIZE];
    float value_now[KV_SIZE];
    float attention[QUERY_SIZE];
    float mixer[HIDDEN];
    float gate_buf[INTERMEDIATE];
    float up_buf[INTERMEDIATE];
    float *scores; /* [max_seq] */
};

static double monotonic_seconds(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return 0.0;
    }
    return (double)now.tv_sec + (double)now.tv_nsec * 1.0e-9;
}

static float bf16_to_float(const uint8_t *bytes)
{
    uint32_t bits = ((uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8U)) << 16U;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static const void *entry_data(const struct mapped_image *image, const struct descriptor *entry)
{
    return image->base + entry->offset;
}

static const float *f32_data(const struct mapped_image *image, const struct descriptor *entry)
{
    return (const float *)entry_data(image, entry);
}

static const struct descriptor *find_entry(const struct mapped_image *image, const char *name)
{
    for (uint32_t index = 0; index < image->header->descriptor_count; ++index) {
        if (strncmp(image->entries[index].name, name, sizeof(image->entries[index].name)) == 0) {
            return &image->entries[index];
        }
    }
    return NULL;
}

static int map_image(const char *path, struct mapped_image *image)
{
    struct stat status;
    memset(image, 0, sizeof(*image));
    image->fd = open(path, O_RDONLY);
    if (image->fd < 0 || fstat(image->fd, &status) != 0 ||
        status.st_size < (off_t)sizeof(struct image_header)) {
        return -1;
    }
    image->bytes = (size_t)status.st_size;
    image->base = mmap(NULL, image->bytes, PROT_READ, MAP_PRIVATE, image->fd, 0);
    if (image->base == MAP_FAILED) {
        image->base = NULL;
        return -1;
    }
    image->header = (const struct image_header *)image->base;
    if (memcmp(image->header->magic, "QW35TSK1", 8) != 0 ||
        (image->header->version != 1U && image->header->version != 2U) ||
        image->header->group_size != GROUP || image->header->file_bytes != image->bytes ||
        image->header->hidden != HIDDEN || image->header->layer_count != LAYERS ||
        image->header->full_interval != FULL_INTERVAL) {
        return -1;
    }
    image->entries = (const struct descriptor *)(image->base + image->header->directory_offset);
    for (uint32_t index = 0; index < image->header->descriptor_count; ++index) {
        const struct descriptor *entry = &image->entries[index];
        if (entry->offset > image->bytes || entry->byte_count > image->bytes - entry->offset ||
            memchr(entry->name, '\0', sizeof(entry->name)) == NULL) {
            return -1;
        }
    }
    return 0;
}

/* --- kernels ------------------------------------------------------------ */

static void q4_gemv(const struct mapped_image *image, const struct descriptor *matrix,
                    const float *input, float *output)
{
    const uint8_t *data = entry_data(image, matrix);
    const uint32_t rows = matrix->shape[0];
    const uint32_t groups = matrix->shape[1] / GROUP;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (uint32_t row = 0; row < rows; ++row) {
        const uint8_t *record = data + (uint64_t)row * groups * Q4_RECORD;
        float sum = 0.0f;
        for (uint32_t group = 0; group < groups; ++group) {
            const float scale = bf16_to_float(record);
            const uint8_t *packed = record + 2;
            const float *values = input + (uint64_t)group * GROUP;
            float group_sum = 0.0f;
            for (uint32_t index = 0; index < GROUP / 2; ++index) {
                const uint8_t byte = packed[index];
                int low = byte & 0x0F;
                int high = byte >> 4;
                low = low >= 8 ? low - 16 : low;
                high = high >= 8 ? high - 16 : high;
                group_sum += values[index * 2] * (float)low;
                group_sum += values[index * 2 + 1] * (float)high;
            }
            sum += group_sum * scale;
            record += Q4_RECORD;
        }
        output[row] = sum;
    }
}

static void q8_gemv(const struct mapped_image *image, const struct descriptor *matrix,
                    const float *input, float *output)
{
    const uint8_t *data = entry_data(image, matrix);
    const uint32_t rows = matrix->shape[0];
    const uint32_t groups = matrix->shape[1] / GROUP;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (uint32_t row = 0; row < rows; ++row) {
        const uint8_t *record = data + (uint64_t)row * groups * Q8_RECORD;
        float sum = 0.0f;
        for (uint32_t group = 0; group < groups; ++group) {
            const float scale = bf16_to_float(record);
            const int8_t *values8 = (const int8_t *)(record + 2);
            const float *values = input + (uint64_t)group * GROUP;
            float group_sum = 0.0f;
            for (uint32_t index = 0; index < GROUP; ++index) {
                group_sum += values[index] * (float)values8[index];
            }
            sum += group_sum * scale;
            record += Q8_RECORD;
        }
        output[row] = sum;
    }
}

static void f32_gemv(const struct mapped_image *image, const struct descriptor *matrix,
                     const float *input, float *output)
{
    const float *data = f32_data(image, matrix);
    const uint32_t rows = matrix->shape[0];
    const uint32_t columns = matrix->shape[1];
    for (uint32_t row = 0; row < rows; ++row) {
        const float *weight = data + (uint64_t)row * columns;
        float sum = 0.0f;
        for (uint32_t index = 0; index < columns; ++index) {
            sum += input[index] * weight[index];
        }
        output[row] = sum;
    }
}

static void dequantize_q4_row(const struct mapped_image *image, const struct descriptor *matrix,
                              uint32_t row, float *output)
{
    const uint32_t groups = matrix->shape[1] / GROUP;
    const uint8_t *record = (const uint8_t *)entry_data(image, matrix) +
                            (uint64_t)row * groups * Q4_RECORD;
    for (uint32_t group = 0; group < groups; ++group) {
        const float scale = bf16_to_float(record);
        const uint8_t *packed = record + 2;
        float *values = output + (uint64_t)group * GROUP;
        for (uint32_t index = 0; index < GROUP / 2; ++index) {
            const uint8_t byte = packed[index];
            int low = byte & 0x0F;
            int high = byte >> 4;
            low = low >= 8 ? low - 16 : low;
            high = high >= 8 ? high - 16 : high;
            values[index * 2] = (float)low * scale;
            values[index * 2 + 1] = (float)high * scale;
        }
        record += Q4_RECORD;
    }
}

/* --- math helpers ------------------------------------------------------- */

static void rms_norm_one_plus(const float *input, const float *weight, float *output,
                              uint32_t width)
{
    double squares = 0.0;
    for (uint32_t index = 0; index < width; ++index) {
        squares += (double)input[index] * input[index];
    }
    const float inverse = (float)pow(squares / (double)width + 1.0e-6, -0.5);
    for (uint32_t index = 0; index < width; ++index) {
        output[index] = input[index] * inverse * (1.0f + weight[index]);
    }
}

static float silu_f(float value)
{
    return value / (1.0f + expf(-value));
}

/* --- resolution --------------------------------------------------------- */

static const struct descriptor *need(const struct mapped_image *image, const char *format,
                                     uint32_t layer, const char *suffix)
{
    char name[96];
    snprintf(name, sizeof(name), format, layer, suffix);
    return find_entry(image, name);
}

static void resolve_tokenizer(struct model *model)
{
    const struct mapped_image *image = &model->image;
    const struct descriptor *offsets = find_entry(image, "tokenizer.token_offsets");
    const struct descriptor *blob = find_entry(image, "tokenizer.token_bytes");
    const struct descriptor *bytes_map = find_entry(image, "tokenizer.byte_tokens");
    const struct descriptor *merges = find_entry(image, "tokenizer.merges");
    const struct descriptor *special_ids = find_entry(image, "tokenizer.special_ids");
    const struct descriptor *special_offsets = find_entry(image, "tokenizer.special_offsets");
    const struct descriptor *special_bytes = find_entry(image, "tokenizer.special_bytes");
    if (!offsets || !blob || !bytes_map || !merges || !special_ids || !special_offsets ||
        !special_bytes || merges->rank != 2U || merges->shape[1] != 4U) {
        return;
    }
    model->tokenizer.present = 1;
    model->tokenizer.token_count = offsets->shape[0] - 1U;
    model->tokenizer.token_offsets = entry_data(image, offsets);
    model->tokenizer.token_bytes = entry_data(image, blob);
    model->tokenizer.byte_tokens = entry_data(image, bytes_map);
    model->tokenizer.merges = entry_data(image, merges);
    model->tokenizer.merge_count = merges->shape[0];
    model->tokenizer.special_ids = entry_data(image, special_ids);
    model->tokenizer.special_offsets = entry_data(image, special_offsets);
    model->tokenizer.special_bytes = entry_data(image, special_bytes);
    model->tokenizer.special_count = special_ids->shape[0];
}

static int resolve_model(const char *path, struct model *model)
{
    memset(model, 0, sizeof(*model));
    if (map_image(path, &model->image) != 0) {
        return -1;
    }
    model->embed = find_entry(&model->image, "embed_tokens.weight");
    model->final_norm = find_entry(&model->image, "norm.weight");
    if (model->embed == NULL || model->embed->kind != KIND_Q4 ||
        model->final_norm == NULL || model->final_norm->kind != KIND_F32) {
        return -1;
    }
    resolve_tokenizer(model);
    for (uint32_t index = 0; index < LAYERS; ++index) {
        struct layer *layer = &model->layers[index];
        layer->is_full = (index + 1U) % FULL_INTERVAL == 0U;
        layer->post_norm = need(&model->image, "layers.%u.%s", index,
                                "post_attention_layernorm.weight");
        layer->gate_proj = need(&model->image, "layers.%u.mlp.%s", index, "gate_proj.weight");
        layer->up_proj = need(&model->image, "layers.%u.mlp.%s", index, "up_proj.weight");
        layer->down_proj = need(&model->image, "layers.%u.mlp.%s", index, "down_proj.weight");
        if (!layer->post_norm || !layer->gate_proj || !layer->up_proj || !layer->down_proj) {
            return -1;
        }
        if (layer->is_full) {
            struct full_layer *full = &layer->full;
            full->input_norm = need(&model->image, "layers.%u.%s", index,
                                    "input_layernorm.weight");
            full->q_proj = need(&model->image, "layers.%u.self_attn.%s", index, "q_proj.weight");
            full->k_proj = need(&model->image, "layers.%u.self_attn.%s", index, "k_proj.weight");
            full->v_proj = need(&model->image, "layers.%u.self_attn.%s", index, "v_proj.weight");
            full->q_norm = need(&model->image, "layers.%u.self_attn.%s", index, "q_norm.weight");
            full->k_norm = need(&model->image, "layers.%u.self_attn.%s", index, "k_norm.weight");
            full->o_proj = need(&model->image, "layers.%u.self_attn.%s", index, "o_proj.weight");
            if (!full->input_norm || !full->q_proj || !full->k_proj || !full->v_proj ||
                !full->q_norm || !full->k_norm || !full->o_proj) {
                return -1;
            }
        } else {
            struct linear_layer *linear = &layer->linear;
            linear->input_norm = need(&model->image, "layers.%u.%s", index,
                                      "input_layernorm.weight");
            linear->in_proj_qkv = need(&model->image, "layers.%u.linear_attn.%s", index,
                                       "in_proj_qkv.weight");
            linear->in_proj_z = need(&model->image, "layers.%u.linear_attn.%s", index,
                                     "in_proj_z.weight");
            linear->in_proj_b = need(&model->image, "layers.%u.linear_attn.%s", index,
                                     "in_proj_b.weight");
            linear->in_proj_a = need(&model->image, "layers.%u.linear_attn.%s", index,
                                     "in_proj_a.weight");
            linear->conv = need(&model->image, "layers.%u.linear_attn.%s", index,
                                "conv1d.weight");
            linear->a_log = need(&model->image, "layers.%u.linear_attn.%s", index, "A_log");
            linear->dt_bias = need(&model->image, "layers.%u.linear_attn.%s", index, "dt_bias");
            linear->gated_norm = need(&model->image, "layers.%u.linear_attn.%s", index,
                                      "norm.weight");
            linear->out_proj = need(&model->image, "layers.%u.linear_attn.%s", index,
                                    "out_proj.weight");
            if (!linear->input_norm || !linear->in_proj_qkv || !linear->in_proj_z ||
                !linear->in_proj_b || !linear->in_proj_a || !linear->conv || !linear->a_log ||
                !linear->dt_bias || !linear->gated_norm || !linear->out_proj) {
                return -1;
            }
        }
    }
    return 0;
}

/* --- forward ------------------------------------------------------------ */

static void forward_token(const struct model *model, struct state *state,
                          struct workspace *ws, uint32_t token_id, size_t position)
{
    dequantize_q4_row(&model->image, model->embed, token_id, ws->hidden);

    for (uint32_t index = 0; index < LAYERS; ++index) {
        const struct layer *layer = &model->layers[index];

        if (!layer->is_full) {
            const struct linear_layer *lin = &layer->linear;
            rms_norm_one_plus(ws->hidden, f32_data(&model->image, lin->input_norm), ws->normed,
                              HIDDEN);
            q8_gemv(&model->image, lin->in_proj_qkv, ws->normed, ws->mixed);
            q8_gemv(&model->image, lin->in_proj_z, ws->normed, ws->z);
            f32_gemv(&model->image, lin->in_proj_b, ws->normed, ws->b);
            f32_gemv(&model->image, lin->in_proj_a, ws->normed, ws->a);

            /* Causal depthwise conv over the rolling window, then SiLU. */
            const float *conv_weight = f32_data(&model->image, lin->conv);
            float *window = state->conv_window[index];
            for (uint32_t channel = 0; channel < CONV_DIM; ++channel) {
                float *lane = window + (size_t)channel * CONV_KERNEL;
                lane[0] = lane[1];
                lane[1] = lane[2];
                lane[2] = lane[3];
                lane[3] = ws->mixed[channel];
                const float *taps = conv_weight + (size_t)channel * CONV_KERNEL;
                float total = 0.0f;
                for (uint32_t tap = 0; tap < CONV_KERNEL; ++tap) {
                    total += lane[tap] * taps[tap];
                }
                ws->post_conv[channel] = silu_f(total);
            }

            /* Split and L2-normalize q/k per head; scale q. */
            const float query_scale = 1.0f / sqrtf((float)LK_DIM);
            for (uint32_t head = 0; head < LK_HEADS; ++head) {
                const float *q_in = ws->post_conv + (size_t)head * LK_DIM;
                const float *k_in = ws->post_conv + KEY_DIM + (size_t)head * LK_DIM;
                float q_squares = 0.0f;
                float k_squares = 0.0f;
                for (uint32_t i = 0; i < LK_DIM; ++i) {
                    q_squares += q_in[i] * q_in[i];
                    k_squares += k_in[i] * k_in[i];
                }
                const float q_inverse = 1.0f / sqrtf(q_squares + 1.0e-6f);
                const float k_inverse = 1.0f / sqrtf(k_squares + 1.0e-6f);
                for (uint32_t i = 0; i < LK_DIM; ++i) {
                    ws->queries[(size_t)head * LK_DIM + i] = q_in[i] * q_inverse * query_scale;
                    ws->keys[(size_t)head * LK_DIM + i] = k_in[i] * k_inverse;
                }
            }
            const float *values = ws->post_conv + 2 * KEY_DIM;

            const float *a_log = f32_data(&model->image, lin->a_log);
            const float *dt_bias = f32_data(&model->image, lin->dt_bias);
            for (uint32_t head = 0; head < LV_HEADS; ++head) {
                const float gate = -expf(a_log[head]) *
                                   log1pf(expf(ws->a[head] + dt_bias[head]));
                const float decay = expf(gate);
                const float beta = 1.0f / (1.0f + expf(-ws->b[head]));
                const float *k_vec = ws->keys + (size_t)head * LK_DIM;
                const float *q_vec = ws->queries + (size_t)head * LK_DIM;
                const float *v_vec = values + (size_t)head * LV_DIM;
                float *head_state = state->recurrent[index] + (size_t)head * LK_DIM * LV_DIM;

                for (uint32_t i = 0; i < LK_DIM * LV_DIM; ++i) {
                    head_state[i] *= decay;
                }
                for (uint32_t vi = 0; vi < LV_DIM; ++vi) {
                    float kv_mem = 0.0f;
                    for (uint32_t ki = 0; ki < LK_DIM; ++ki) {
                        kv_mem += head_state[(size_t)ki * LV_DIM + vi] * k_vec[ki];
                    }
                    const float delta = (v_vec[vi] - kv_mem) * beta;
                    for (uint32_t ki = 0; ki < LK_DIM; ++ki) {
                        head_state[(size_t)ki * LV_DIM + vi] += k_vec[ki] * delta;
                    }
                }
                for (uint32_t vi = 0; vi < LV_DIM; ++vi) {
                    float total = 0.0f;
                    for (uint32_t ki = 0; ki < LK_DIM; ++ki) {
                        total += head_state[(size_t)ki * LV_DIM + vi] * q_vec[ki];
                    }
                    ws->core[(size_t)head * LV_DIM + vi] = total;
                }
            }

            /* Gated RMSNorm per value head (plain weight), then out_proj. */
            const float *gated_weight = f32_data(&model->image, lin->gated_norm);
            for (uint32_t head = 0; head < LV_HEADS; ++head) {
                const float *core = ws->core + (size_t)head * LV_DIM;
                const float *z = ws->z + (size_t)head * LV_DIM;
                double squares = 0.0;
                for (uint32_t i = 0; i < LV_DIM; ++i) {
                    squares += (double)core[i] * core[i];
                }
                const float inverse = (float)pow(squares / LV_DIM + 1.0e-6, -0.5);
                for (uint32_t i = 0; i < LV_DIM; ++i) {
                    ws->gated[(size_t)head * LV_DIM + i] =
                        core[i] * inverse * gated_weight[i] * silu_f(z[i]);
                }
            }
            q8_gemv(&model->image, lin->out_proj, ws->gated, ws->mixer);
        } else {
            const struct full_layer *full = &layer->full;
            rms_norm_one_plus(ws->hidden, f32_data(&model->image, full->input_norm), ws->normed,
                              HIDDEN);
            q4_gemv(&model->image, full->q_proj, ws->normed, ws->q_and_gate);
            q4_gemv(&model->image, full->k_proj, ws->normed, ws->key_now);
            q4_gemv(&model->image, full->v_proj, ws->normed, ws->value_now);

            for (uint32_t head = 0; head < Q_HEADS; ++head) {
                memcpy(ws->query + (size_t)head * HEAD_DIM,
                       ws->q_and_gate + (size_t)head * 2 * HEAD_DIM, HEAD_DIM * sizeof(float));
                memcpy(ws->attn_gate + (size_t)head * HEAD_DIM,
                       ws->q_and_gate + (size_t)head * 2 * HEAD_DIM + HEAD_DIM,
                       HEAD_DIM * sizeof(float));
            }
            const float *q_norm = f32_data(&model->image, full->q_norm);
            const float *k_norm = f32_data(&model->image, full->k_norm);
            for (uint32_t head = 0; head < Q_HEADS; ++head) {
                rms_norm_one_plus(ws->query + (size_t)head * HEAD_DIM, q_norm,
                                  ws->query + (size_t)head * HEAD_DIM, HEAD_DIM);
            }
            for (uint32_t head = 0; head < KV_HEADS; ++head) {
                rms_norm_one_plus(ws->key_now + (size_t)head * HEAD_DIM, k_norm,
                                  ws->key_now + (size_t)head * HEAD_DIM, HEAD_DIM);
            }
            /* Partial RoPE over the first ROTARY_DIM dims of each head. */
            const float *cos_row = state->rope_cos + position * ROTARY_DIM;
            const float *sin_row = state->rope_sin + position * ROTARY_DIM;
            for (uint32_t head = 0; head < Q_HEADS + KV_HEADS; ++head) {
                float *vector = head < Q_HEADS
                                    ? ws->query + (size_t)head * HEAD_DIM
                                    : ws->key_now + (size_t)(head - Q_HEADS) * HEAD_DIM;
                for (uint32_t i = 0; i < ROTARY_DIM / 2; ++i) {
                    const float first = vector[i];
                    const float second = vector[i + ROTARY_DIM / 2];
                    vector[i] = first * cos_row[i] - second * sin_row[i];
                    vector[i + ROTARY_DIM / 2] = second * cos_row[i] + first * sin_row[i];
                }
            }

            memcpy(state->key_cache[index] + position * KV_SIZE, ws->key_now,
                   KV_SIZE * sizeof(float));
            memcpy(state->value_cache[index] + position * KV_SIZE, ws->value_now,
                   KV_SIZE * sizeof(float));

            const float scale = 1.0f / sqrtf((float)HEAD_DIM);
            for (uint32_t head = 0; head < Q_HEADS; ++head) {
                const uint32_t kv_head = head / (Q_HEADS / KV_HEADS);
                const float *query = ws->query + (size_t)head * HEAD_DIM;
                float maximum = -INFINITY;
                for (size_t pos = 0; pos <= position; ++pos) {
                    const float *key = state->key_cache[index] + pos * KV_SIZE +
                                       (size_t)kv_head * HEAD_DIM;
                    float raw = 0.0f;
                    for (uint32_t i = 0; i < HEAD_DIM; ++i) {
                        raw += query[i] * key[i];
                    }
                    raw *= scale;
                    ws->scores[pos] = raw;
                    maximum = raw > maximum ? raw : maximum;
                }
                float denominator = 0.0f;
                for (size_t pos = 0; pos <= position; ++pos) {
                    ws->scores[pos] = expf(ws->scores[pos] - maximum);
                    denominator += ws->scores[pos];
                }
                for (uint32_t i = 0; i < HEAD_DIM; ++i) {
                    float total = 0.0f;
                    for (size_t pos = 0; pos <= position; ++pos) {
                        total += ws->scores[pos] / denominator *
                                 state->value_cache[index][pos * KV_SIZE +
                                                           (size_t)kv_head * HEAD_DIM + i];
                    }
                    ws->attention[(size_t)head * HEAD_DIM + i] = total;
                }
            }
            for (uint32_t i = 0; i < QUERY_SIZE; ++i) {
                ws->attention[i] *= 1.0f / (1.0f + expf(-ws->attn_gate[i]));
            }
            q4_gemv(&model->image, full->o_proj, ws->attention, ws->mixer);
        }

        for (uint32_t i = 0; i < HIDDEN; ++i) {
            ws->hidden[i] += ws->mixer[i];
        }

        rms_norm_one_plus(ws->hidden, f32_data(&model->image, layer->post_norm), ws->normed,
                          HIDDEN);
        q4_gemv(&model->image, layer->gate_proj, ws->normed, ws->gate_buf);
        q4_gemv(&model->image, layer->up_proj, ws->normed, ws->up_buf);
        for (uint32_t i = 0; i < INTERMEDIATE; ++i) {
            ws->gate_buf[i] = silu_f(ws->gate_buf[i]) * ws->up_buf[i];
        }
        q4_gemv(&model->image, layer->down_proj, ws->gate_buf, ws->mixer);
        for (uint32_t i = 0; i < HIDDEN; ++i) {
            ws->hidden[i] += ws->mixer[i];
        }
    }
}

static int allocate_state(struct state *state, size_t max_seq)
{
    memset(state, 0, sizeof(*state));
    state->max_seq = max_seq;
    state->rope_cos = malloc(max_seq * ROTARY_DIM * sizeof(float));
    state->rope_sin = malloc(max_seq * ROTARY_DIM * sizeof(float));
    if (state->rope_cos == NULL || state->rope_sin == NULL) {
        return -1;
    }
    for (size_t position = 0; position < max_seq; ++position) {
        for (uint32_t i = 0; i < ROTARY_DIM / 2; ++i) {
            const double inv_freq = 1.0 / pow(1.0e7, (2.0 * i) / (double)ROTARY_DIM);
            const double angle = (double)position * inv_freq;
            state->rope_cos[position * ROTARY_DIM + i] = (float)cos(angle);
            state->rope_sin[position * ROTARY_DIM + i] = (float)sin(angle);
            state->rope_cos[position * ROTARY_DIM + ROTARY_DIM / 2 + i] = (float)cos(angle);
            state->rope_sin[position * ROTARY_DIM + ROTARY_DIM / 2 + i] = (float)sin(angle);
        }
    }
    for (uint32_t index = 0; index < LAYERS; ++index) {
        if ((index + 1U) % FULL_INTERVAL == 0U) {
            state->key_cache[index] = calloc(max_seq * KV_SIZE, sizeof(float));
            state->value_cache[index] = calloc(max_seq * KV_SIZE, sizeof(float));
            if (state->key_cache[index] == NULL || state->value_cache[index] == NULL) {
                return -1;
            }
        } else {
            state->recurrent[index] = calloc((size_t)LV_HEADS * LK_DIM * LV_DIM, sizeof(float));
            state->conv_window[index] = calloc((size_t)CONV_DIM * CONV_KERNEL, sizeof(float));
            if (state->recurrent[index] == NULL || state->conv_window[index] == NULL) {
                return -1;
            }
        }
    }
    return 0;
}

static void free_state(struct state *state)
{
    for (uint32_t index = 0; index < LAYERS; ++index) {
        free(state->recurrent[index]);
        free(state->conv_window[index]);
        free(state->key_cache[index]);
        free(state->value_cache[index]);
    }
    free(state->rope_cos);
    free(state->rope_sin);
    memset(state, 0, sizeof(*state));
}

/* --- tokenizer ---------------------------------------------------------- */

static int is_ws(uint8_t c)
{
    return c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '\f' || c == '\v';
}

static int is_digit_ascii(uint8_t c)
{
    return c >= '0' && c <= '9';
}

/* ASCII letters plus every non-ASCII byte. This approximates \p{L}\p{M} from
 * the pinned pretokenizer regex: multi-byte letters and combining marks land
 * in letter runs, at the cost of classifying non-ASCII punctuation the same
 * way. The parity test against the pinned tokenizer bounds the deviation. */
static int is_letterish(uint8_t c)
{
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c >= 0x80;
}

static size_t contraction_length(const uint8_t *text, size_t length, size_t p)
{
    if (text[p] != '\'' || p + 1 >= length) {
        return 0;
    }
    const uint8_t one = text[p + 1] | 0x20;
    if (one == 's' || one == 't' || one == 'm' || one == 'd') {
        return 2;
    }
    if (p + 2 < length) {
        const uint8_t two = text[p + 2] | 0x20;
        if ((one == 'r' && two == 'e') || (one == 'v' && two == 'e') ||
            (one == 'l' && two == 'l')) {
            return 3;
        }
    }
    return 0;
}

/* Length of the next pretokenizer chunk starting at p. Mirrors the ordered
 * alternatives of the pinned Split regex. */
static size_t pretoken_length(const uint8_t *text, size_t length, size_t p)
{
    size_t span = contraction_length(text, length, p);
    if (span != 0) {
        return span;
    }
    /* [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ */
    {
        size_t start = p;
        const uint8_t c = text[start];
        if (!is_letterish(c) && !is_digit_ascii(c) && c != '\r' && c != '\n') {
            start += 1;
        }
        if (start < length && is_letterish(text[start])) {
            size_t end = start;
            while (end < length && is_letterish(text[end])) {
                end += 1;
            }
            return end - p;
        }
    }
    /* \p{N} — a single digit */
    if (is_digit_ascii(text[p])) {
        return 1;
    }
    /* " ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*" */
    {
        size_t cursor = p;
        if (text[cursor] == ' ' && cursor + 1 < length) {
            cursor += 1;
        }
        size_t run = cursor;
        while (run < length && !is_ws(text[run]) && !is_letterish(text[run]) &&
               !is_digit_ascii(text[run])) {
            run += 1;
        }
        if (run > cursor) {
            while (run < length && (text[run] == '\r' || text[run] == '\n')) {
                run += 1;
            }
            return run - p;
        }
    }
    /* whitespace alternatives */
    if (is_ws(text[p])) {
        size_t end = p;
        size_t last_newline = 0;
        int saw_newline = 0;
        while (end < length && is_ws(text[end])) {
            if (text[end] == '\r' || text[end] == '\n') {
                last_newline = end;
                saw_newline = 1;
            }
            end += 1;
        }
        if (saw_newline) {
            return last_newline + 1 - p; /* \s*[\r\n]+ */
        }
        if (end == length) {
            return end - p; /* \s+(?!\S) at end of input */
        }
        if (end - p >= 2) {
            return end - p - 1; /* \s+(?!\S) keeps the final space for the next word */
        }
        return 1; /* \s+ */
    }
    return 1; /* lone CR/LF or unmatched byte falls through one at a time */
}

static uint32_t merge_lookup(const struct tokenizer *tokenizer, uint32_t left, uint32_t right,
                             uint32_t *merged)
{
    size_t low = 0;
    size_t high = tokenizer->merge_count;
    while (low < high) {
        const size_t middle = (low + high) / 2;
        const uint32_t *entry = tokenizer->merges + middle * 4;
        if (entry[0] < left || (entry[0] == left && entry[1] < right)) {
            low = middle + 1;
        } else if (entry[0] == left && entry[1] == right) {
            *merged = entry[2];
            return entry[3];
        } else {
            high = middle;
        }
    }
    return UINT32_MAX;
}

enum { CHUNK_CAPACITY = 2048 };

static int bpe_encode_chunk(const struct tokenizer *tokenizer, const uint8_t *chunk,
                            size_t chunk_length, uint32_t *output, int capacity, int count)
{
    static uint32_t ids[CHUNK_CAPACITY];
    if (chunk_length > CHUNK_CAPACITY) {
        return -1;
    }
    size_t n = chunk_length;
    for (size_t index = 0; index < n; ++index) {
        ids[index] = tokenizer->byte_tokens[chunk[index]];
    }
    while (n > 1) {
        uint32_t best_rank = UINT32_MAX;
        uint32_t best_merged = 0;
        size_t best_pos = 0;
        for (size_t index = 0; index + 1 < n; ++index) {
            uint32_t merged;
            const uint32_t rank = merge_lookup(tokenizer, ids[index], ids[index + 1], &merged);
            if (rank < best_rank) {
                best_rank = rank;
                best_merged = merged;
                best_pos = index;
            }
        }
        if (best_rank == UINT32_MAX) {
            break;
        }
        ids[best_pos] = best_merged;
        memmove(ids + best_pos + 1, ids + best_pos + 2, (n - best_pos - 2) * sizeof(uint32_t));
        n -= 1;
    }
    for (size_t index = 0; index < n; ++index) {
        if (count >= capacity) {
            return -1;
        }
        output[count++] = ids[index];
    }
    return count;
}

static int encode_segment(const struct tokenizer *tokenizer, const uint8_t *text, size_t length,
                          uint32_t *output, int capacity, int count)
{
    size_t position = 0;
    while (position < length) {
        const size_t span = pretoken_length(text, length, position);
        count = bpe_encode_chunk(tokenizer, text + position, span, output, capacity, count);
        if (count < 0) {
            return -1;
        }
        position += span;
    }
    return count;
}

static int encode_text(const struct tokenizer *tokenizer, const char *input, uint32_t *output,
                       int capacity)
{
    const uint8_t *text = (const uint8_t *)input;
    const size_t length = strlen(input);
    size_t position = 0;
    int count = 0;
    while (position < length) {
        /* Earliest special-token occurrence from here; ties favor the longer
         * literal because the table is sorted by descending length. */
        size_t special_at = length;
        uint32_t special_index = 0;
        for (uint32_t index = 0; index < tokenizer->special_count; ++index) {
            const size_t begin = tokenizer->special_offsets[index];
            const size_t size = tokenizer->special_offsets[index + 1] - begin;
            if (size == 0 || size > length - position) {
                continue;
            }
            for (size_t at = position; at + size <= length && at < special_at; ++at) {
                if (memcmp(text + at, tokenizer->special_bytes + begin, size) == 0) {
                    special_at = at;
                    special_index = index;
                    break;
                }
            }
        }
        count = encode_segment(tokenizer, text + position, special_at - position, output,
                               capacity, count);
        if (count < 0) {
            return -1;
        }
        if (special_at < length) {
            if (count >= capacity) {
                return -1;
            }
            output[count++] = tokenizer->special_ids[special_index];
            position = special_at + tokenizer->special_offsets[special_index + 1] -
                       tokenizer->special_offsets[special_index];
        } else {
            position = length;
        }
    }
    return count;
}

/* Run one prompt through the graph and print one JSON result line. */
static int run_one(const struct model *model, const uint32_t *token_ids, int token_count,
                   const uint32_t *answer_ids, int answer_count, int case_index)
{
    struct state state;
    struct workspace *ws = calloc(1, sizeof(struct workspace));
    if (ws == NULL || allocate_state(&state, (size_t)token_count + 1U) != 0) {
        free(ws);
        return -1;
    }
    ws->scores = malloc(((size_t)token_count + 1U) * sizeof(float));
    if (ws->scores == NULL) {
        free(ws);
        free_state(&state);
        return -1;
    }

    const double start = monotonic_seconds();
    for (int position = 0; position < token_count; ++position) {
        forward_token(model, &state, ws, token_ids[position], (size_t)position);
    }
    rms_norm_one_plus(ws->hidden, f32_data(&model->image, model->final_norm), ws->normed,
                      HIDDEN);

    float best_logit = -INFINITY;
    int best_index = 0;
    float logits[MAX_ANSWERS];
    float row[HIDDEN];
    for (int index = 0; index < answer_count; ++index) {
        dequantize_q4_row(&model->image, model->embed, answer_ids[index], row);
        double sum = 0.0;
        for (uint32_t i = 0; i < HIDDEN; ++i) {
            sum += (double)ws->normed[i] * row[i];
        }
        logits[index] = (float)sum;
        if (logits[index] > best_logit) {
            best_logit = logits[index];
            best_index = index;
        }
    }
    const double elapsed = monotonic_seconds() - start;

    printf("{");
    if (case_index >= 0) {
        printf("\"case\":%d,", case_index);
    }
    printf("\"tokens\":%d,\"answers\":[", token_count);
    for (int index = 0; index < answer_count; ++index) {
        printf("%s{\"id\":%u,\"logit\":%.9g}", index == 0 ? "" : ",", answer_ids[index],
               logits[index]);
    }
    printf("],\"chosen\":%u,\"chosen_index\":%d,\"prefill_seconds\":%.6f,"
           "\"prefill_tokens_per_second\":%.6f}\n",
           answer_ids[best_index], best_index, elapsed, (double)token_count / elapsed);
    fflush(stdout);

    free(ws->scores);
    free(ws);
    free_state(&state);
    return 0;
}

static int parse_id_list(const char *text, uint32_t *values, int capacity)
{
    int count = 0;
    while (*text != '\0') {
        char *end = NULL;
        unsigned long parsed = strtoul(text, &end, 10);
        if (end == text || count >= capacity) {
            return -1;
        }
        values[count++] = (uint32_t)parsed;
        text = *end == ',' ? end + 1 : end;
        if (*end != ',' && *end != '\0') {
            return -1;
        }
    }
    return count;
}

int main(int argc, char **argv)
{
    const char *image_path = NULL;
    const char *ids_text = NULL;
    const char *answers_text = NULL;
    const char *prompt_text = NULL;
    const char *answer_words = NULL;
    const char *prompts_file = NULL;
    int encode_only = 0;
    const char *usage = "usage: %s IMAGE (--prompt TEXT | --ids ID,... | --prompts-file PATH) "
                        "(--answers-text WORD,... | --answers ID,...)\n";
    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--ids") == 0 && index + 1 < argc) {
            ids_text = argv[++index];
        } else if (strcmp(argv[index], "--answers") == 0 && index + 1 < argc) {
            answers_text = argv[++index];
        } else if (strcmp(argv[index], "--prompt") == 0 && index + 1 < argc) {
            prompt_text = argv[++index];
        } else if (strcmp(argv[index], "--answers-text") == 0 && index + 1 < argc) {
            answer_words = argv[++index];
        } else if (strcmp(argv[index], "--prompts-file") == 0 && index + 1 < argc) {
            prompts_file = argv[++index];
        } else if (strcmp(argv[index], "--encode-only") == 0) {
            encode_only = 1;
        } else if (image_path == NULL) {
            image_path = argv[index];
        } else {
            fprintf(stderr, usage, argv[0]);
            return 2;
        }
    }
    if (image_path == NULL ||
        (ids_text == NULL && prompt_text == NULL && prompts_file == NULL) ||
        (answers_text == NULL && answer_words == NULL)) {
        fprintf(stderr, usage, argv[0]);
        return 2;
    }

    struct model model;
    if (resolve_model(image_path, &model) != 0) {
        fprintf(stderr, "invalid or unsupported task image: %s\n", image_path);
        return 2;
    }
    if ((prompt_text != NULL || answer_words != NULL || prompts_file != NULL) &&
        !model.tokenizer.present) {
        fprintf(stderr, "image carries no tokenizer tables; pass --ids/--answers\n");
        return 2;
    }

    static uint32_t token_ids[MAX_TOKENS];
    static uint32_t answer_ids[MAX_ANSWERS];
    int token_count = 0;
    int answer_count = 0;
    if (ids_text != NULL) {
        token_count = parse_id_list(ids_text, token_ids, MAX_TOKENS);
    } else if (prompt_text != NULL) {
        token_count = encode_text(&model.tokenizer, prompt_text, token_ids, MAX_TOKENS);
    }
    if (answers_text != NULL) {
        answer_count = parse_id_list(answers_text, answer_ids, MAX_ANSWERS);
    } else {
        char buffer[512];
        const char *cursor = answer_words;
        while (*cursor != '\0' && answer_count >= 0) {
            const char *comma = strchr(cursor, ',');
            const size_t size = comma ? (size_t)(comma - cursor) : strlen(cursor);
            if (size == 0 || size >= sizeof(buffer) || answer_count >= MAX_ANSWERS) {
                answer_count = -1;
                break;
            }
            memcpy(buffer, cursor, size);
            buffer[size] = '\0';
            uint32_t encoded[8];
            const int encoded_count = encode_text(&model.tokenizer, buffer, encoded, 8);
            if (encoded_count != 1) {
                fprintf(stderr, "answer '%s' is not a single token (%d)\n", buffer,
                        encoded_count);
                answer_count = -1;
                break;
            }
            answer_ids[answer_count++] = encoded[0];
            cursor = comma ? comma + 1 : cursor + size;
        }
    }
    if (answer_count <= 0 || (prompts_file == NULL && token_count <= 0)) {
        fprintf(stderr, "invalid prompt or answer list\n");
        return 2;
    }
    if (prompts_file != NULL) {
#ifdef _OPENMP
        omp_set_dynamic(0);
#endif
        FILE *file = fopen(prompts_file, "rb");
        if (file == NULL) {
            fprintf(stderr, "cannot open %s\n", prompts_file);
            return 2;
        }
        fseek(file, 0, SEEK_END);
        const long file_size = ftell(file);
        fseek(file, 0, SEEK_SET);
        char *data = malloc((size_t)file_size + 1U);
        if (data == NULL || fread(data, 1, (size_t)file_size, file) != (size_t)file_size) {
            fprintf(stderr, "cannot read %s\n", prompts_file);
            fclose(file);
            return 2;
        }
        fclose(file);
        data[file_size] = '\0';

        int case_index = 0;
        size_t position = 0;
        while (position < (size_t)file_size) {
            const char *record = data + position;
            const size_t record_length = strlen(record);
            if (record_length > 0) {
                const int count = encode_text(&model.tokenizer, record, token_ids, MAX_TOKENS);
                if (count <= 0 ||
                    run_one(&model, token_ids, count, answer_ids, answer_count,
                            case_index) != 0) {
                    fprintf(stderr, "case %d failed\n", case_index);
                    free(data);
                    return 2;
                }
                case_index += 1;
            }
            position += record_length + 1U;
        }
        free(data);
        munmap((void *)model.image.base, model.image.bytes);
        close(model.image.fd);
        return 0;
    }
    if (encode_only) {
        for (int index = 0; index < token_count; ++index) {
            printf("%s%u", index == 0 ? "" : ",", token_ids[index]);
        }
        printf("\n");
        return 0;
    }
    for (int index = 0; index < token_count; ++index) {
        if (token_ids[index] >= model.image.header->vocab_rows) {
            fprintf(stderr, "token id outside the vocabulary\n");
            return 2;
        }
    }
    for (int index = 0; index < answer_count; ++index) {
        if (answer_ids[index] >= model.image.header->vocab_rows) {
            fprintf(stderr, "answer id outside the vocabulary\n");
            return 2;
        }
    }

#ifdef _OPENMP
    omp_set_dynamic(0);
#endif
    if (run_one(&model, token_ids, token_count, answer_ids, answer_count, -1) != 0) {
        fprintf(stderr, "execution failed\n");
        return 2;
    }
    munmap((void *)model.image.base, model.image.bytes);
    close(model.image.fd);
    return 0;
}
