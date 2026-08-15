#define _POSIX_C_SOURCE 200809L

#include "qwen38_m3.h"
#include "qwen38_m3_attention_image.h"
#include "qwen38_sha256.h"
#include "qwen38_safetensors.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static uint64_t align_page(uint64_t value) {
    return (value + QWEN38_M3_ATTENTION_HEADER_BYTES - 1) &
           ~(uint64_t)(QWEN38_M3_ATTENTION_HEADER_BYTES - 1);
}

static int pread_exact(int file, void *output, size_t length, uint64_t offset) {
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

static int pinned_sha(const char *sha) {
    return strcmp(sha, QWEN38_M3_EXPECTED_SOURCE_SHA256) == 0 ||
           strcmp(sha, QWEN38_M3_EXPECTED_SOURCE_SHA256_2) == 0 ||
           strcmp(sha, QWEN38_M3_EXPECTED_SOURCE_SHA256_3) == 0 ||
           strcmp(sha, QWEN38_M3_EXPECTED_MTP_SHA256) == 0;
}

static int verify_sha256(int file, const char *expected) {
    enum { CHUNK = 8 * 1024 * 1024 };
    unsigned char *buffer = malloc(CHUNK);
    unsigned char digest[32];
    char actual[65];
    qwen38_sha256_context context;
    if (buffer == NULL) return -1;
    qwen38_sha256_init(&context);
    uint64_t offset = 0;
    for (;;) {
        ssize_t amount = pread(file, buffer, CHUNK, (off_t)offset);
        if (amount < 0 && errno == EINTR) continue;
        if (amount < 0) {
            free(buffer);
            return -1;
        }
        if (amount == 0) break;
        qwen38_sha256_update(&context, buffer, (size_t)amount);
        offset += (uint64_t)amount;
    }
    free(buffer);
    qwen38_sha256_final(&context, digest);
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

static float bf16_to_float(uint16_t input) {
    uint32_t bits = (uint32_t)input << 16;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint16_t bf16_to_half(uint16_t input) {
    return float_to_half_bits(bf16_to_float(input));
}

static int find_tensor(const char *path, const char *name,
                       qwen38_tensor_view *view) {
    char error[512];
    int status = qwen38_safetensors_find(path, name, 1, view,
                                         error, sizeof(error));
    if (status != 0) fprintf(stderr, "%s\n", error);
    return status;
}

static int validate_matrix(const qwen38_tensor_view *view, const char *name,
                           const char *dtype, uint64_t rows, uint64_t columns,
                           uint64_t bytes) {
    if (strcmp(view->dtype, dtype) != 0 || view->rank != 2 ||
        view->shape[0] != rows || view->shape[1] != columns ||
        view->data_length != bytes) {
        fprintf(stderr, "%s: unexpected dtype, shape or byte length\n", name);
        return -1;
    }
    return 0;
}

static int validate_vector(const qwen38_tensor_view *view, const char *name,
                           uint64_t values) {
    if (strcmp(view->dtype, "BF16") != 0 || view->rank != 1 ||
        view->shape[0] != values || view->data_length != values * 2) {
        fprintf(stderr, "%s: expected BF16 vector length %" PRIu64 "\n",
                name, values);
        return -1;
    }
    return 0;
}

static int copy_tensor(int source, int output,
                       const qwen38_tensor_view *tensor,
                       uint64_t output_offset) {
    enum { CHUNK = 8 * 1024 * 1024 };
    unsigned char *buffer = malloc(CHUNK);
    if (buffer == NULL) return -1;
    uint64_t done = 0;
    while (done < tensor->data_length) {
        size_t amount = (size_t)(tensor->data_length - done);
        if (amount > CHUNK) amount = CHUNK;
        if (pread_exact(source, buffer, amount,
                        tensor->data_start + done) != 0 ||
            pwrite_exact(output, buffer, amount,
                         output_offset + done) != 0) {
            free(buffer);
            return -1;
        }
        done += amount;
    }
    free(buffer);
    return 0;
}

static int convert_metadata(int source, int output,
                            const qwen38_tensor_view *scale,
                            const qwen38_tensor_view *bias,
                            uint64_t output_offset) {
    size_t values = (size_t)(scale->data_length / 2);
    uint16_t *scales = malloc(values * 2);
    uint16_t *biases = malloc(values * 2);
    uint16_t *combined = malloc(values * 4);
    if (scales == NULL || biases == NULL || combined == NULL ||
        pread_exact(source, scales, values * 2, scale->data_start) != 0 ||
        pread_exact(source, biases, values * 2, bias->data_start) != 0) {
        free(scales); free(biases); free(combined);
        return -1;
    }
    for (size_t index = 0; index < values; ++index) {
        combined[index * 2] = bf16_to_half(scales[index]);
        combined[index * 2 + 1] = bf16_to_half(biases[index]);
    }
    int status = pwrite_exact(output, combined, values * 4, output_offset);
    free(scales); free(biases); free(combined);
    return status;
}

static int convert_vector_f32(int source, int output,
                              const qwen38_tensor_view *tensor,
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
        converted[index] = bf16_to_float(input[index]);
    }
    int status = pwrite_exact(output, converted, values * sizeof(float),
                              output_offset);
    free(input); free(converted);
    return status;
}

int main(int argc, char **argv) {
    if (argc != 5 || strlen(argv[3]) != 64) {
        fprintf(stderr, "usage: %s SOURCE.safetensors OUTPUT.q38att "
                        "SOURCE_SHA256 LAYER_INDEX\n", argv[0]);
        return 2;
    }
    char *end = NULL;
    errno = 0;
    unsigned long parsed = strtoul(argv[4], &end, 10);
    if (errno != 0 || end == argv[4] || *end != '\0' ||
        parsed > 64 || (parsed < 64 && parsed % 4 != 3) ||
        !pinned_sha(argv[3])) {
        fprintf(stderr, "invalid pinned source, SHA-256 or attention layer\n");
        return 2;
    }
    /* Layer 64 is the MTP draft layer packed from a standalone
     * quantized MTP checkpoint, whose tensors sit at layers.0.*. */
    unsigned layer = (unsigned)parsed;
#define MAKE_NAME(variable, suffix) \
    char variable##_storage[160]; \
    if (layer == 64) \
        snprintf(variable##_storage, sizeof(variable##_storage), \
                 "layers.0.%s", suffix); \
    else \
        snprintf(variable##_storage, sizeof(variable##_storage), \
                 "language_model.model.layers.%u.%s", layer, suffix); \
    const char *const variable = variable##_storage
    MAKE_NAME(GATE_W, "mlp.gate_proj.weight");
    MAKE_NAME(GATE_S, "mlp.gate_proj.scales");
    MAKE_NAME(GATE_B, "mlp.gate_proj.biases");
    MAKE_NAME(UP_W, "mlp.up_proj.weight");
    MAKE_NAME(UP_S, "mlp.up_proj.scales");
    MAKE_NAME(UP_B, "mlp.up_proj.biases");
    MAKE_NAME(DOWN_W, "mlp.down_proj.weight");
    MAKE_NAME(DOWN_S, "mlp.down_proj.scales");
    MAKE_NAME(DOWN_B, "mlp.down_proj.biases");
    MAKE_NAME(INPUT_NORM, "input_layernorm.weight");
    MAKE_NAME(POST_NORM, "post_attention_layernorm.weight");
    MAKE_NAME(Q_W, "self_attn.q_proj.weight");
    MAKE_NAME(Q_S, "self_attn.q_proj.scales");
    MAKE_NAME(Q_B, "self_attn.q_proj.biases");
    MAKE_NAME(K_W, "self_attn.k_proj.weight");
    MAKE_NAME(K_S, "self_attn.k_proj.scales");
    MAKE_NAME(K_B, "self_attn.k_proj.biases");
    MAKE_NAME(V_W, "self_attn.v_proj.weight");
    MAKE_NAME(V_S, "self_attn.v_proj.scales");
    MAKE_NAME(V_B, "self_attn.v_proj.biases");
    MAKE_NAME(O_W, "self_attn.o_proj.weight");
    MAKE_NAME(O_S, "self_attn.o_proj.scales");
    MAKE_NAME(O_B, "self_attn.o_proj.biases");
    MAKE_NAME(Q_NORM, "self_attn.q_norm.weight");
    MAKE_NAME(K_NORM, "self_attn.k_norm.weight");
#undef MAKE_NAME

    qwen38_tensor_view gate_w, gate_s, gate_b, up_w, up_s, up_b;
    qwen38_tensor_view down_w, down_s, down_b;
    qwen38_tensor_view input_norm, post_norm, q_norm, k_norm;
    qwen38_tensor_view q_w, q_s, q_b, k_w, k_s, k_b, v_w, v_s, v_b;
    qwen38_tensor_view o_w, o_s, o_b;
#define FIND(name, view) (find_tensor(argv[1], name, &(view)) != 0)
    if (FIND(GATE_W, gate_w) || FIND(GATE_S, gate_s) ||
        FIND(GATE_B, gate_b) || FIND(UP_W, up_w) || FIND(UP_S, up_s) ||
        FIND(UP_B, up_b) || FIND(DOWN_W, down_w) ||
        FIND(DOWN_S, down_s) || FIND(DOWN_B, down_b) ||
        FIND(INPUT_NORM, input_norm) || FIND(POST_NORM, post_norm) ||
        FIND(Q_W, q_w) || FIND(Q_S, q_s) || FIND(Q_B, q_b) ||
        FIND(K_W, k_w) || FIND(K_S, k_s) || FIND(K_B, k_b) ||
        FIND(V_W, v_w) || FIND(V_S, v_s) || FIND(V_B, v_b) ||
        FIND(O_W, o_w) || FIND(O_S, o_s) || FIND(O_B, o_b) ||
        FIND(Q_NORM, q_norm) || FIND(K_NORM, k_norm)) {
        return 3;
    }
#undef FIND
    const uint64_t hidden_groups = 80;
    const uint64_t mlp_groups = 272;
    const uint64_t mlp_weight_bytes =
        (uint64_t)QWEN38_MLP_SIZE * QWEN38_HIDDEN_SIZE / 2;
    const uint64_t mlp_meta_plane =
        (uint64_t)QWEN38_MLP_SIZE * hidden_groups * 2;
    const uint64_t q_weight_bytes =
        (uint64_t)QWEN38_ATTENTION_Q_ROWS * QWEN38_HIDDEN_SIZE / 2;
    const uint64_t kv_weight_bytes =
        (uint64_t)QWEN38_ATTENTION_K_ROWS * QWEN38_HIDDEN_SIZE / 2;
    const uint64_t o_weight_bytes =
        (uint64_t)QWEN38_HIDDEN_SIZE *
        (QWEN38_ATTENTION_HEADS * QWEN38_ATTENTION_HEAD_SIZE) / 2;
#define VM(view, name, dtype, rows, cols, bytes) \
    validate_matrix(&(view), name, dtype, rows, cols, bytes)
    if (VM(gate_w, GATE_W, "U32", QWEN38_MLP_SIZE, 640,
           mlp_weight_bytes) ||
        VM(up_w, UP_W, "U32", QWEN38_MLP_SIZE, 640, mlp_weight_bytes) ||
        VM(down_w, DOWN_W, "U32", QWEN38_HIDDEN_SIZE, 2176,
           mlp_weight_bytes) ||
        VM(gate_s, GATE_S, "BF16", QWEN38_MLP_SIZE, hidden_groups,
           mlp_meta_plane) ||
        VM(gate_b, GATE_B, "BF16", QWEN38_MLP_SIZE, hidden_groups,
           mlp_meta_plane) ||
        VM(up_s, UP_S, "BF16", QWEN38_MLP_SIZE, hidden_groups,
           mlp_meta_plane) ||
        VM(up_b, UP_B, "BF16", QWEN38_MLP_SIZE, hidden_groups,
           mlp_meta_plane) ||
        VM(down_s, DOWN_S, "BF16", QWEN38_HIDDEN_SIZE, mlp_groups,
           mlp_meta_plane) ||
        VM(down_b, DOWN_B, "BF16", QWEN38_HIDDEN_SIZE, mlp_groups,
           mlp_meta_plane) ||
        VM(q_w, Q_W, "U32", QWEN38_ATTENTION_Q_ROWS, 640,
           q_weight_bytes) ||
        VM(q_s, Q_S, "BF16", QWEN38_ATTENTION_Q_ROWS, hidden_groups,
           (uint64_t)QWEN38_ATTENTION_Q_ROWS * hidden_groups * 2) ||
        VM(q_b, Q_B, "BF16", QWEN38_ATTENTION_Q_ROWS, hidden_groups,
           (uint64_t)QWEN38_ATTENTION_Q_ROWS * hidden_groups * 2) ||
        VM(k_w, K_W, "U32", QWEN38_ATTENTION_K_ROWS, 640,
           kv_weight_bytes) ||
        VM(k_s, K_S, "BF16", QWEN38_ATTENTION_K_ROWS, hidden_groups,
           (uint64_t)QWEN38_ATTENTION_K_ROWS * hidden_groups * 2) ||
        VM(k_b, K_B, "BF16", QWEN38_ATTENTION_K_ROWS, hidden_groups,
           (uint64_t)QWEN38_ATTENTION_K_ROWS * hidden_groups * 2) ||
        VM(v_w, V_W, "U32", QWEN38_ATTENTION_V_ROWS, 640,
           kv_weight_bytes) ||
        VM(v_s, V_S, "BF16", QWEN38_ATTENTION_V_ROWS, hidden_groups,
           (uint64_t)QWEN38_ATTENTION_V_ROWS * hidden_groups * 2) ||
        VM(v_b, V_B, "BF16", QWEN38_ATTENTION_V_ROWS, hidden_groups,
           (uint64_t)QWEN38_ATTENTION_V_ROWS * hidden_groups * 2) ||
        VM(o_w, O_W, "U32", QWEN38_HIDDEN_SIZE, 768, o_weight_bytes) ||
        VM(o_s, O_S, "BF16", QWEN38_HIDDEN_SIZE, 96,
           (uint64_t)QWEN38_HIDDEN_SIZE * 96 * 2) ||
        VM(o_b, O_B, "BF16", QWEN38_HIDDEN_SIZE, 96,
           (uint64_t)QWEN38_HIDDEN_SIZE * 96 * 2) ||
        validate_vector(&input_norm, INPUT_NORM, QWEN38_HIDDEN_SIZE) ||
        validate_vector(&post_norm, POST_NORM, QWEN38_HIDDEN_SIZE) ||
        validate_vector(&q_norm, Q_NORM, QWEN38_ATTENTION_HEAD_SIZE) ||
        validate_vector(&k_norm, K_NORM, QWEN38_ATTENTION_HEAD_SIZE)) {
        return 4;
    }
#undef VM

    qwen38_m3_attention_image_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, QWEN38_M3_ATTENTION_IMAGE_MAGIC, 8);
    h.version = QWEN38_M3_ATTENTION_IMAGE_VERSION;
    h.header_bytes = QWEN38_M3_ATTENTION_HEADER_BYTES;
    h.layer_index = layer;
    h.hidden_size = QWEN38_HIDDEN_SIZE;
    h.intermediate_size = QWEN38_MLP_SIZE;
    h.group_size = QWEN38_Q4_GROUP_SIZE;
    h.q_heads = QWEN38_ATTENTION_HEADS;
    h.kv_heads = QWEN38_ATTENTION_KV_HEADS;
    h.head_size = QWEN38_ATTENTION_HEAD_SIZE;
    h.rotary_size = QWEN38_ATTENTION_ROTARY_SIZE;
    h.input_rows = QWEN38_ATTENTION_INPUT_ROWS;
    h.input_groups_per_row = hidden_groups;
    h.output_rows = QWEN38_HIDDEN_SIZE;
    h.output_groups_per_row = 96;
    uint64_t offset = QWEN38_M3_ATTENTION_HEADER_BYTES;
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
    h.post_norm_constants_index = QWEN38_HIDDEN_SIZE;
    h.q_norm_constants_index = 2 * QWEN38_HIDDEN_SIZE;
    h.k_norm_constants_index =
        h.q_norm_constants_index + QWEN38_ATTENTION_HEAD_SIZE;
    h.constants_f32_count =
        (uint32_t)(h.k_norm_constants_index + QWEN38_ATTENTION_HEAD_SIZE);
    h.constants_offset = offset;
    h.constants_bytes = align_page(
        (uint64_t)h.constants_f32_count * sizeof(float));
    offset += h.constants_bytes;
    SEGMENT(attention_input_quants,
            q_weight_bytes + 2 * kv_weight_bytes);
    SEGMENT(attention_input_metadata,
            (uint64_t)QWEN38_ATTENTION_INPUT_ROWS * hidden_groups * 4);
    SEGMENT(attention_output_quants, o_weight_bytes);
    SEGMENT(attention_output_metadata,
            (uint64_t)QWEN38_HIDDEN_SIZE * 96 * 4);
#undef SEGMENT
    memcpy(h.source_sha256, argv[3], 64);

    int source = open(argv[1], O_RDONLY);
    if (source < 0 || verify_sha256(source, argv[3]) != 0) {
        if (source >= 0) close(source);
        return 5;
    }
    int output = open(argv[2], O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (output < 0) {
        fprintf(stderr, "cannot create %s: %s\n", argv[2], strerror(errno));
        close(source);
        return 5;
    }
    uint64_t q_cursor = h.attention_input_quants_offset;
    uint64_t m_cursor = h.attention_input_metadata_offset;
#define APPEND_TENSOR(view) \
    (copy_tensor(source, output, &(view), q_cursor) != 0 ? -1 : \
        (q_cursor += (view).data_length, 0))
#define APPEND_META(scale, bias) \
    (convert_metadata(source, output, &(scale), &(bias), m_cursor) != 0 ? -1 : \
        (m_cursor += (scale).data_length * 2, 0))
    int failed =
        pwrite_exact(output, &h, sizeof(h), 0) != 0 ||
        copy_tensor(source, output, &gate_w, h.gate_quants_offset) != 0 ||
        convert_metadata(source, output, &gate_s, &gate_b,
                         h.gate_metadata_offset) != 0 ||
        copy_tensor(source, output, &up_w, h.up_quants_offset) != 0 ||
        convert_metadata(source, output, &up_s, &up_b,
                         h.up_metadata_offset) != 0 ||
        copy_tensor(source, output, &down_w, h.down_quants_offset) != 0 ||
        convert_metadata(source, output, &down_s, &down_b,
                         h.down_metadata_offset) != 0 ||
        convert_vector_f32(source, output, &input_norm,
            h.constants_offset + h.input_norm_constants_index * 4) != 0 ||
        convert_vector_f32(source, output, &post_norm,
            h.constants_offset + h.post_norm_constants_index * 4) != 0 ||
        convert_vector_f32(source, output, &q_norm,
            h.constants_offset + h.q_norm_constants_index * 4) != 0 ||
        convert_vector_f32(source, output, &k_norm,
            h.constants_offset + h.k_norm_constants_index * 4) != 0 ||
        APPEND_TENSOR(q_w) || APPEND_TENSOR(k_w) || APPEND_TENSOR(v_w) ||
        APPEND_META(q_s, q_b) || APPEND_META(k_s, k_b) ||
        APPEND_META(v_s, v_b) ||
        copy_tensor(source, output, &o_w,
                    h.attention_output_quants_offset) != 0 ||
        convert_metadata(source, output, &o_s, &o_b,
                         h.attention_output_metadata_offset) != 0 ||
        q_cursor != h.attention_input_quants_offset +
                    h.attention_input_quants_bytes ||
        m_cursor != h.attention_input_metadata_offset +
                    h.attention_input_metadata_bytes ||
        ftruncate(output, (off_t)offset) != 0 || fsync(output) != 0;
#undef APPEND_TENSOR
#undef APPEND_META
    close(source);
    close(output);
    if (failed) {
        fprintf(stderr, "packing failed: %s\n", strerror(errno));
        unlink(argv[2]);
        return 6;
    }
    printf("{\"layer\":%u,\"source\":\"%s\",\"output\":\"%s\","
           "\"source_sha256\":\"%s\",\"bytes\":%" PRIu64 "}\n",
           layer, argv[1], argv[2], argv[3], offset);
    return 0;
}
