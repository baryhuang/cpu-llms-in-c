/* Packs the Qwen3.6-27B multi-token-prediction head from the assembled
 * BF16 mtp.* safetensors into two images: a standard attention image
 * (layer_index 64) carrying the MTP transformer layer, and a small MTP
 * extras image carrying the fc projection and its three norm vectors.
 * Unlike the main-layer packers, the source here is BF16, so this tool
 * quantizes to affine Q4 group-64 itself, matching the deployed layout:
 * scale = (max - min) / 15 and bias = min, both stored FP16, with byte j
 * of a group holding elements 2j (low nibble) and 2j+1 (high nibble). */

#define _POSIX_C_SOURCE 200809L

#include "qwen36_m3.h"
#include "qwen36_m3_attention_image.h"
#include "qwen36_m3_mtp_image.h"
#include "qwen36_safetensors.h"
#include "qwen36_sha256.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static uint64_t align_page(uint64_t value) {
    return (value + QWEN36_M3_ATTENTION_HEADER_BYTES - 1) &
           ~(uint64_t)(QWEN36_M3_ATTENTION_HEADER_BYTES - 1);
}

static int pread_exact(int file, void *output, size_t length,
                       uint64_t offset) {
    unsigned char *bytes = output;
    size_t done = 0;
    while (done < length) {
        ssize_t amount = pread(file, bytes + done, length - done,
                               (off_t)(offset + done));
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) return -1;
        done += (size_t)amount;
    }
    return 0;
}

static int pwrite_exact(int file, const void *input, size_t length,
                        uint64_t offset) {
    const unsigned char *bytes = input;
    size_t done = 0;
    while (done < length) {
        ssize_t amount = pwrite(file, bytes + done, length - done,
                                (off_t)(offset + done));
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) return -1;
        done += (size_t)amount;
    }
    return 0;
}

static int verify_sha256(int file, const char *expected) {
    enum { CHUNK = 8 * 1024 * 1024 };
    unsigned char *buffer = malloc(CHUNK);
    unsigned char digest[32];
    char actual[65];
    qwen36_sha256_context context;
    if (buffer == NULL) return -1;
    qwen36_sha256_init(&context);
    uint64_t offset = 0;
    for (;;) {
        ssize_t amount = pread(file, buffer, CHUNK, (off_t)offset);
        if (amount < 0 && errno == EINTR) continue;
        if (amount < 0) {
            free(buffer);
            return -1;
        }
        if (amount == 0) break;
        qwen36_sha256_update(&context, buffer, (size_t)amount);
        offset += (uint64_t)amount;
    }
    free(buffer);
    qwen36_sha256_final(&context, digest);
    for (size_t index = 0; index < 32; ++index) {
        snprintf(actual + index * 2, 3, "%02x", digest[index]);
    }
    actual[64] = '\0';
    if (strcmp(actual, expected) != 0) {
        fprintf(stderr, "source SHA-256 mismatch: expected %s, got %s\n",
                expected, actual);
        return -1;
    }
    return 0;
}

static uint16_t float_to_half_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000u;
    uint32_t exponent = (bits >> 23) & 0xffu;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        return (uint16_t)(sign | (mantissa != 0 ? 0x7e00u : 0x7c00u));
    }
    int half_exponent = (int)exponent - 127 + 15;
    if (half_exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return (uint16_t)sign;
        mantissa |= 0x800000u;
        unsigned shift = (unsigned)(14 - half_exponent);
        uint32_t rounded = mantissa + ((1u << (shift - 1)) - 1u) +
                           ((mantissa >> shift) & 1u);
        return (uint16_t)(sign | (rounded >> shift));
    }
    uint32_t rounded = mantissa + 0xfffu + ((mantissa >> 13) & 1u);
    if (rounded & 0x800000u) {
        rounded = 0;
        if (++half_exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    }
    return (uint16_t)(sign | ((uint32_t)half_exponent << 10) |
                      (rounded >> 13));
}

static float half_bits_to_float(uint16_t bits) {
    uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t mantissa = bits & 0x3ffu;
    uint32_t out;
    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3ffu;
            out = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fu) {
        out = sign | 0x7f800000u | (mantissa << 13);
    } else {
        out = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float value;
    memcpy(&value, &out, sizeof(value));
    return value;
}

static float bf16_to_float(uint16_t input) {
    uint32_t bits = (uint32_t)input << 16;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int find_tensor(const char *path, const char *name,
                       qwen36_tensor_view *view) {
    char error[512];
    int status = qwen36_safetensors_find(path, name, 1, view,
                                         error, sizeof(error));
    if (status != 0) fprintf(stderr, "%s\n", error);
    return status;
}

static int validate_bf16_matrix(const qwen36_tensor_view *view,
                                const char *name, uint64_t rows,
                                uint64_t columns) {
    if (strcmp(view->dtype, "BF16") != 0 || view->rank != 2 ||
        view->shape[0] != rows || view->shape[1] != columns ||
        view->data_length != rows * columns * 2) {
        fprintf(stderr, "%s: expected BF16 [%" PRIu64 ", %" PRIu64 "]\n",
                name, rows, columns);
        return -1;
    }
    return 0;
}

static int validate_bf16_vector(const qwen36_tensor_view *view,
                                const char *name, uint64_t values) {
    if (strcmp(view->dtype, "BF16") != 0 || view->rank != 1 ||
        view->shape[0] != values || view->data_length != values * 2) {
        fprintf(stderr, "%s: expected BF16 vector length %" PRIu64 "\n",
                name, values);
        return -1;
    }
    return 0;
}

/* Quantize one BF16 matrix to Q4 group-64 and write quants and
 * interleaved FP16 [scale, bias] metadata at the given offsets. */
static int quantize_matrix(int source, int output,
                           const qwen36_tensor_view *view,
                           uint64_t quants_offset,
                           uint64_t metadata_offset) {
    uint64_t rows = view->shape[0];
    uint64_t columns = view->shape[1];
    uint64_t groups = columns / 64;
    uint16_t *row_bf16 = malloc(columns * 2);
    float *row = malloc(columns * sizeof(float));
    unsigned char *quants = malloc(columns / 2);
    uint16_t *metadata = malloc(groups * 2 * sizeof(uint16_t));
    if (row_bf16 == NULL || row == NULL || quants == NULL ||
        metadata == NULL) {
        free(row_bf16); free(row); free(quants); free(metadata);
        return -1;
    }
    int failed = 0;
    for (uint64_t r = 0; r < rows && !failed; ++r) {
        if (pread_exact(source, row_bf16, columns * 2,
                        view->data_start + r * columns * 2) != 0) {
            failed = 1;
            break;
        }
        for (uint64_t c = 0; c < columns; ++c)
            row[c] = bf16_to_float(row_bf16[c]);
        for (uint64_t g = 0; g < groups; ++g) {
            const float *values = row + g * 64;
            float minimum = values[0];
            float maximum = values[0];
            for (int i = 1; i < 64; ++i) {
                if (values[i] < minimum) minimum = values[i];
                if (values[i] > maximum) maximum = values[i];
            }
            float scale = (maximum - minimum) / 15.0f;
            uint16_t scale_h = float_to_half_bits(scale);
            uint16_t bias_h = float_to_half_bits(minimum);
            /* Quantize against the FP16-rounded scale and bias the
             * runtime will actually dequantize with. */
            float scale_r = half_bits_to_float(scale_h);
            float bias_r = half_bits_to_float(bias_h);
            if (scale_r <= 0.0f || !isfinite(scale_r)) {
                scale_r = 1.0f;
                scale_h = float_to_half_bits(1.0f);
            }
            for (int i = 0; i < 32; ++i) {
                int q0 = (int)lrintf((values[i * 2] - bias_r) / scale_r);
                int q1 = (int)lrintf((values[i * 2 + 1] - bias_r) /
                                     scale_r);
                if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
                if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
                quants[g * 32 + i] =
                    (unsigned char)(q0 | (q1 << 4));
            }
            metadata[g * 2] = scale_h;
            metadata[g * 2 + 1] = bias_h;
        }
        if (pwrite_exact(output, quants, columns / 2,
                         quants_offset + r * (columns / 2)) != 0 ||
            pwrite_exact(output, metadata, groups * 4,
                         metadata_offset + r * groups * 4) != 0) {
            failed = 1;
        }
    }
    free(row_bf16); free(row); free(quants); free(metadata);
    return failed ? -1 : 0;
}

/* Every norm vector in the official BF16 checkpoint uses the
 * Hugging Face delta convention: the effective multiplier is 1 + w
 * (Gemma-style RMSNorm, confirmed against the reference serving
 * implementations). The deployed runtime uses direct multipliers, so the
 * conversion happens here at pack time, exactly as the mlx conversion
 * did for the main model's norms. */
static int convert_norm_vector_f32(int source, int output,
                                   const qwen36_tensor_view *tensor,
                                   uint64_t output_offset) {
    size_t values = (size_t)(tensor->data_length / 2);
    uint16_t *input = malloc(values * 2);
    float *converted = malloc(values * sizeof(float));
    if (input == NULL || converted == NULL ||
        pread_exact(source, input, values * 2, tensor->data_start) != 0) {
        free(input); free(converted);
        return -1;
    }
    for (size_t index = 0; index < values; ++index) {
        converted[index] = 1.0f + bf16_to_float(input[index]);
    }
    int status = pwrite_exact(output, converted, values * sizeof(float),
                              output_offset);
    free(input); free(converted);
    return status;
}

int main(int argc, char **argv) {
    if (argc != 5 || strlen(argv[4]) != 64) {
        fprintf(stderr, "usage: %s MTP_BF16.safetensors "
                        "OUTPUT_LAYER.q36att OUTPUT_MTP.q36mtp "
                        "SOURCE_SHA256\n", argv[0]);
        return 2;
    }
    if (strcmp(argv[4], QWEN36_M3_EXPECTED_MTP_SHA256) != 0) {
        fprintf(stderr, "SHA-256 is not the pinned MTP source\n");
        return 2;
    }

    qwen36_tensor_view fc, gate, up, down, q, k, v, o;
    qwen36_tensor_view input_norm, post_norm, q_norm, k_norm;
    qwen36_tensor_view embed_norm, hidden_norm, final_norm;
#define FIND(name, view) (find_tensor(argv[1], name, &(view)) != 0)
    if (FIND("mtp.fc.weight", fc) ||
        FIND("mtp.layers.0.mlp.gate_proj.weight", gate) ||
        FIND("mtp.layers.0.mlp.up_proj.weight", up) ||
        FIND("mtp.layers.0.mlp.down_proj.weight", down) ||
        FIND("mtp.layers.0.self_attn.q_proj.weight", q) ||
        FIND("mtp.layers.0.self_attn.k_proj.weight", k) ||
        FIND("mtp.layers.0.self_attn.v_proj.weight", v) ||
        FIND("mtp.layers.0.self_attn.o_proj.weight", o) ||
        FIND("mtp.layers.0.input_layernorm.weight", input_norm) ||
        FIND("mtp.layers.0.post_attention_layernorm.weight", post_norm) ||
        FIND("mtp.layers.0.self_attn.q_norm.weight", q_norm) ||
        FIND("mtp.layers.0.self_attn.k_norm.weight", k_norm) ||
        FIND("mtp.pre_fc_norm_embedding.weight", embed_norm) ||
        FIND("mtp.pre_fc_norm_hidden.weight", hidden_norm) ||
        FIND("mtp.norm.weight", final_norm)) {
        return 3;
    }
#undef FIND
    if (validate_bf16_matrix(&fc, "mtp.fc", QWEN36_HIDDEN_SIZE,
                             2 * QWEN36_HIDDEN_SIZE) ||
        validate_bf16_matrix(&gate, "gate", QWEN36_MLP_SIZE,
                             QWEN36_HIDDEN_SIZE) ||
        validate_bf16_matrix(&up, "up", QWEN36_MLP_SIZE,
                             QWEN36_HIDDEN_SIZE) ||
        validate_bf16_matrix(&down, "down", QWEN36_HIDDEN_SIZE,
                             QWEN36_MLP_SIZE) ||
        validate_bf16_matrix(&q, "q", QWEN36_ATTENTION_Q_ROWS,
                             QWEN36_HIDDEN_SIZE) ||
        validate_bf16_matrix(&k, "k", QWEN36_ATTENTION_K_ROWS,
                             QWEN36_HIDDEN_SIZE) ||
        validate_bf16_matrix(&v, "v", QWEN36_ATTENTION_V_ROWS,
                             QWEN36_HIDDEN_SIZE) ||
        validate_bf16_matrix(&o, "o", QWEN36_HIDDEN_SIZE,
                             QWEN36_ATTENTION_HEADS *
                             QWEN36_ATTENTION_HEAD_SIZE) ||
        validate_bf16_vector(&input_norm, "input_norm",
                             QWEN36_HIDDEN_SIZE) ||
        validate_bf16_vector(&post_norm, "post_norm",
                             QWEN36_HIDDEN_SIZE) ||
        validate_bf16_vector(&q_norm, "q_norm",
                             QWEN36_ATTENTION_HEAD_SIZE) ||
        validate_bf16_vector(&k_norm, "k_norm",
                             QWEN36_ATTENTION_HEAD_SIZE) ||
        validate_bf16_vector(&embed_norm, "embed_norm",
                             QWEN36_HIDDEN_SIZE) ||
        validate_bf16_vector(&hidden_norm, "hidden_norm",
                             QWEN36_HIDDEN_SIZE) ||
        validate_bf16_vector(&final_norm, "final_norm",
                             QWEN36_HIDDEN_SIZE)) {
        return 4;
    }

    int source = open(argv[1], O_RDONLY);
    if (source < 0 || verify_sha256(source, argv[4]) != 0) {
        if (source >= 0) close(source);
        return 5;
    }

    /* Layer image: identical layout to the main attention packer, with
     * layer_index 64 marking the MTP layer. */
    const uint64_t hidden_groups = 80;
    const uint64_t mlp_weight_bytes =
        (uint64_t)QWEN36_MLP_SIZE * QWEN36_HIDDEN_SIZE / 2;
    const uint64_t mlp_meta_plane =
        (uint64_t)QWEN36_MLP_SIZE * hidden_groups * 2;
    const uint64_t q_weight_bytes =
        (uint64_t)QWEN36_ATTENTION_Q_ROWS * QWEN36_HIDDEN_SIZE / 2;
    const uint64_t kv_weight_bytes =
        (uint64_t)QWEN36_ATTENTION_K_ROWS * QWEN36_HIDDEN_SIZE / 2;
    const uint64_t o_weight_bytes =
        (uint64_t)QWEN36_HIDDEN_SIZE *
        (QWEN36_ATTENTION_HEADS * QWEN36_ATTENTION_HEAD_SIZE) / 2;

    qwen36_m3_attention_image_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, QWEN36_M3_ATTENTION_IMAGE_MAGIC, 8);
    h.version = QWEN36_M3_ATTENTION_IMAGE_VERSION;
    h.header_bytes = QWEN36_M3_ATTENTION_HEADER_BYTES;
    h.layer_index = QWEN36_M3_MTP_LAYER_INDEX;
    h.hidden_size = QWEN36_HIDDEN_SIZE;
    h.intermediate_size = QWEN36_MLP_SIZE;
    h.group_size = QWEN36_Q4_GROUP_SIZE;
    h.q_heads = QWEN36_ATTENTION_HEADS;
    h.kv_heads = QWEN36_ATTENTION_KV_HEADS;
    h.head_size = QWEN36_ATTENTION_HEAD_SIZE;
    h.rotary_size = QWEN36_ATTENTION_ROTARY_SIZE;
    h.input_rows = QWEN36_ATTENTION_INPUT_ROWS;
    h.input_groups_per_row = hidden_groups;
    h.output_rows = QWEN36_HIDDEN_SIZE;
    h.output_groups_per_row = 96;
    uint64_t offset = QWEN36_M3_ATTENTION_HEADER_BYTES;
#define SEGMENT(field, bytes) do { \
    h.field##_offset = offset; h.field##_bytes = (bytes); \
    offset += (bytes); \
} while (0)
    SEGMENT(gate_quants, mlp_weight_bytes);
    SEGMENT(gate_metadata, mlp_meta_plane * 2);
    SEGMENT(up_quants, mlp_weight_bytes);
    SEGMENT(up_metadata, mlp_meta_plane * 2);
    SEGMENT(down_quants, mlp_weight_bytes);
    SEGMENT(down_metadata, mlp_meta_plane * 2);
    offset = align_page(offset);
    h.input_norm_constants_index = 0;
    h.post_norm_constants_index = QWEN36_HIDDEN_SIZE;
    h.q_norm_constants_index = 2 * QWEN36_HIDDEN_SIZE;
    h.k_norm_constants_index =
        h.q_norm_constants_index + QWEN36_ATTENTION_HEAD_SIZE;
    h.constants_f32_count =
        (uint32_t)(h.k_norm_constants_index + QWEN36_ATTENTION_HEAD_SIZE);
    h.constants_offset = offset;
    h.constants_bytes = align_page(
        (uint64_t)h.constants_f32_count * sizeof(float));
    offset += h.constants_bytes;
    SEGMENT(attention_input_quants, q_weight_bytes + 2 * kv_weight_bytes);
    SEGMENT(attention_input_metadata,
            (uint64_t)QWEN36_ATTENTION_INPUT_ROWS * hidden_groups * 4);
    SEGMENT(attention_output_quants, o_weight_bytes);
    SEGMENT(attention_output_metadata,
            (uint64_t)QWEN36_HIDDEN_SIZE * 96 * 4);
#undef SEGMENT
    memcpy(h.source_sha256, argv[4], 64);
    uint64_t layer_total = offset;

    int layer_out = open(argv[2], O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (layer_out < 0) {
        fprintf(stderr, "cannot create %s: %s\n", argv[2],
                strerror(errno));
        close(source);
        return 5;
    }
    uint64_t q_meta =
        h.attention_input_metadata_offset;
    int failed =
        pwrite_exact(layer_out, &h, sizeof(h), 0) != 0 ||
        quantize_matrix(source, layer_out, &gate,
                        h.gate_quants_offset, h.gate_metadata_offset) ||
        quantize_matrix(source, layer_out, &up,
                        h.up_quants_offset, h.up_metadata_offset) ||
        quantize_matrix(source, layer_out, &down,
                        h.down_quants_offset, h.down_metadata_offset) ||
        convert_norm_vector_f32(source, layer_out, &input_norm,
            h.constants_offset + h.input_norm_constants_index * 4) ||
        convert_norm_vector_f32(source, layer_out, &post_norm,
            h.constants_offset + h.post_norm_constants_index * 4) ||
        convert_norm_vector_f32(source, layer_out, &q_norm,
            h.constants_offset + h.q_norm_constants_index * 4) ||
        convert_norm_vector_f32(source, layer_out, &k_norm,
            h.constants_offset + h.k_norm_constants_index * 4) ||
        quantize_matrix(source, layer_out, &q,
                        h.attention_input_quants_offset, q_meta) ||
        quantize_matrix(source, layer_out, &k,
                        h.attention_input_quants_offset + q_weight_bytes,
                        q_meta + (uint64_t)QWEN36_ATTENTION_Q_ROWS *
                                 hidden_groups * 4) ||
        quantize_matrix(source, layer_out, &v,
                        h.attention_input_quants_offset + q_weight_bytes +
                            kv_weight_bytes,
                        q_meta + (uint64_t)(QWEN36_ATTENTION_Q_ROWS +
                                            QWEN36_ATTENTION_K_ROWS) *
                                 hidden_groups * 4) ||
        quantize_matrix(source, layer_out, &o,
                        h.attention_output_quants_offset,
                        h.attention_output_metadata_offset) ||
        ftruncate(layer_out, (off_t)layer_total) != 0 ||
        fsync(layer_out) != 0;
    close(layer_out);
    if (failed) {
        fprintf(stderr, "layer packing failed\n");
        unlink(argv[2]);
        close(source);
        return 6;
    }

    /* Extras image: fc projection and the three surrounding norms. */
    qwen36_m3_mtp_image_header m;
    memset(&m, 0, sizeof(m));
    memcpy(m.magic, QWEN36_M3_MTP_IMAGE_MAGIC, 8);
    m.version = QWEN36_M3_MTP_IMAGE_VERSION;
    m.header_bytes = QWEN36_M3_MTP_HEADER_BYTES;
    m.hidden_size = QWEN36_HIDDEN_SIZE;
    m.fc_rows = QWEN36_HIDDEN_SIZE;
    m.fc_groups_per_row = 2 * QWEN36_HIDDEN_SIZE / 64;
    m.group_size = QWEN36_Q4_GROUP_SIZE;
    m.constants_f32_count = 3 * QWEN36_HIDDEN_SIZE;
    m.embedding_norm_constants_index = 0;
    m.hidden_norm_constants_index = QWEN36_HIDDEN_SIZE;
    m.final_norm_constants_index = 2 * QWEN36_HIDDEN_SIZE;
    uint64_t cursor = QWEN36_M3_MTP_HEADER_BYTES;
    m.fc_quants_offset = cursor;
    m.fc_quants_bytes =
        (uint64_t)QWEN36_HIDDEN_SIZE * (2 * QWEN36_HIDDEN_SIZE) / 2;
    cursor += m.fc_quants_bytes;
    m.fc_metadata_offset = cursor;
    m.fc_metadata_bytes =
        (uint64_t)QWEN36_HIDDEN_SIZE * m.fc_groups_per_row * 4;
    cursor += m.fc_metadata_bytes;
    cursor = align_page(cursor);
    m.constants_offset = cursor;
    m.constants_bytes = align_page(
        (uint64_t)m.constants_f32_count * sizeof(float));
    cursor += m.constants_bytes;
    memcpy(m.source_sha256, argv[4], 64);

    int mtp_out = open(argv[3], O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (mtp_out < 0) {
        fprintf(stderr, "cannot create %s: %s\n", argv[3],
                strerror(errno));
        close(source);
        return 5;
    }
    failed =
        pwrite_exact(mtp_out, &m, sizeof(m), 0) != 0 ||
        quantize_matrix(source, mtp_out, &fc,
                        m.fc_quants_offset, m.fc_metadata_offset) ||
        convert_norm_vector_f32(source, mtp_out, &embed_norm,
            m.constants_offset + m.embedding_norm_constants_index * 4) ||
        convert_norm_vector_f32(source, mtp_out, &hidden_norm,
            m.constants_offset + m.hidden_norm_constants_index * 4) ||
        convert_norm_vector_f32(source, mtp_out, &final_norm,
            m.constants_offset + m.final_norm_constants_index * 4) ||
        ftruncate(mtp_out, (off_t)cursor) != 0 ||
        fsync(mtp_out) != 0;
    close(mtp_out);
    close(source);
    if (failed) {
        fprintf(stderr, "mtp extras packing failed\n");
        unlink(argv[3]);
        return 6;
    }
    printf("{\"layer_image\":\"%s\",\"layer_bytes\":%" PRIu64 ","
           "\"mtp_image\":\"%s\",\"mtp_bytes\":%" PRIu64 ","
           "\"source_sha256\":\"%s\"}\n",
           argv[2], layer_total, argv[3], cursor, argv[4]);
    return 0;
}
