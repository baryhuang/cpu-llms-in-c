#define _POSIX_C_SOURCE 200809L

#include "qwen38_m3.h"
#include "qwen38_m3_image.h"
#include "qwen38_sha256.h"
#include "qwen38_safetensors.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    DELTA_QKV_ROWS = 10240,
    DELTA_Z_ROWS = 6144,
    DELTA_SCALAR_ROWS = 48,
    DELTA_INPUT_ROWS = DELTA_QKV_ROWS + DELTA_Z_ROWS +
                       2 * DELTA_SCALAR_ROWS,
    DELTA_OUTPUT_INPUTS = 6144,
    DELTA_CONV_VALUES = DELTA_QKV_ROWS * 4
};

static uint64_t align_page(uint64_t value) {
    return (value + QWEN38_M3_IMAGE_HEADER_BYTES - 1) &
           ~(uint64_t)(QWEN38_M3_IMAGE_HEADER_BYTES - 1);
}

static int pread_exact(int file, void *output, size_t length, uint64_t offset) {
    unsigned char *bytes = output;
    size_t done = 0;
    while (done < length) {
        ssize_t amount = pread(file, bytes + done, length - done,
                               (off_t)(offset + done));
        if (amount < 0 && errno == EINTR) {
            continue;
        }
        if (amount <= 0) {
            return -1;
        }
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
        if (amount < 0 && errno == EINTR) {
            continue;
        }
        if (amount <= 0) {
            return -1;
        }
        done += (size_t)amount;
    }
    return 0;
}

static int verify_source_sha256(int file, const char *expected) {
    enum { CHUNK_BYTES = 8 * 1024 * 1024 };
    unsigned char *buffer = malloc(CHUNK_BYTES);
    unsigned char digest[32];
    char actual[65];
    qwen38_sha256_context context;
    if (buffer == NULL) {
        free(buffer);
        return -1;
    }
    qwen38_sha256_init(&context);
    uint64_t offset = 0;
    for (;;) {
        ssize_t amount = pread(file, buffer, CHUNK_BYTES, (off_t)offset);
        if (amount < 0 && errno == EINTR) {
            continue;
        }
        if (amount < 0) {
            free(buffer);
            return -1;
        }
        if (amount == 0) {
            break;
        }
        qwen38_sha256_update(&context, buffer, (size_t)amount);
        offset += (uint64_t)amount;
    }
    free(buffer);
    qwen38_sha256_final(&context, digest);
    for (size_t index = 0; index < sizeof(digest); ++index) {
        snprintf(actual + index * 2, 3, "%02x", digest[index]);
    }
    actual[64] = '\0';
    if (strcmp(actual, expected) != 0) {
        fprintf(stderr, "source SHA-256 mismatch: expected %s, got %s\n",
                expected, actual);
        return 1;
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
    if (half_exponent >= 31) {
        return (uint16_t)(sign | 0x7c00u);
    }
    if (half_exponent <= 0) {
        if (half_exponent < -10) {
            return (uint16_t)sign;
        }
        mantissa |= 0x800000u;
        unsigned shift = (unsigned)(14 - half_exponent);
        uint32_t rounded = mantissa + ((1u << (shift - 1)) - 1u) +
                           ((mantissa >> shift) & 1u);
        return (uint16_t)(sign | (rounded >> shift));
    }
    uint32_t rounded = mantissa + 0xfffu + ((mantissa >> 13) & 1u);
    if (rounded & 0x800000u) {
        rounded = 0;
        ++half_exponent;
        if (half_exponent >= 31) {
            return (uint16_t)(sign | 0x7c00u);
        }
    }
    return (uint16_t)(sign | ((uint32_t)half_exponent << 10) |
                      (rounded >> 13));
}

static float half_bits_to_float(uint16_t input) {
    uint32_t sign = (uint32_t)(input & 0x8000u) << 16;
    uint32_t exponent = (input >> 10) & 0x1fu;
    uint32_t mantissa = input & 0x03ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ffu;
            bits = sign | (uint32_t)(127 - 15 - shift) << 23 |
                   mantissa << 13;
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | mantissa << 13;
    } else {
        bits = sign | (exponent + (127 - 15)) << 23 | mantissa << 13;
    }
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static uint16_t bf16_to_half(uint16_t input) {
    uint32_t bits = (uint32_t)input << 16;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return float_to_half_bits(value);
}

static float bf16_to_float(uint16_t input) {
    uint32_t bits = (uint32_t)input << 16;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static int source_reference_first_8(int source,
                                    const qwen38_tensor_view *gate_weight,
                                    const qwen38_tensor_view *gate_scale,
                                    const qwen38_tensor_view *gate_bias,
                                    const qwen38_tensor_view *up_weight,
                                    const qwen38_tensor_view *up_scale,
                                    const qwen38_tensor_view *up_bias,
                                    float output[8]) {
    const size_t rows = 8;
    const size_t weight_row_bytes = QWEN38_HIDDEN_SIZE / 2;
    const size_t groups_per_row = QWEN38_HIDDEN_SIZE / QWEN38_Q4_GROUP_SIZE;
    const size_t meta_row_bytes = groups_per_row * sizeof(uint16_t);
    uint8_t gate_q[rows * QWEN38_HIDDEN_SIZE / 2];
    uint8_t up_q[rows * QWEN38_HIDDEN_SIZE / 2];
    uint16_t gate_s[rows * groups_per_row];
    uint16_t gate_b[rows * groups_per_row];
    uint16_t up_s[rows * groups_per_row];
    uint16_t up_b[rows * groups_per_row];
    if (pread_exact(source, gate_q, sizeof(gate_q), gate_weight->data_start) != 0 ||
        pread_exact(source, up_q, sizeof(up_q), up_weight->data_start) != 0 ||
        pread_exact(source, gate_s, sizeof(gate_s), gate_scale->data_start) != 0 ||
        pread_exact(source, gate_b, sizeof(gate_b), gate_bias->data_start) != 0 ||
        pread_exact(source, up_s, sizeof(up_s), up_scale->data_start) != 0 ||
        pread_exact(source, up_b, sizeof(up_b), up_bias->data_start) != 0) {
        return -1;
    }
    (void)weight_row_bytes;
    (void)meta_row_bytes;
    for (size_t row = 0; row < rows; ++row) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        for (size_t index = 0; index < QWEN38_HIDDEN_SIZE; ++index) {
            size_t group = index / QWEN38_Q4_GROUP_SIZE;
            size_t quant_byte = row * (QWEN38_HIDDEN_SIZE / 2) + index / 2;
            uint8_t gate_bits = gate_q[quant_byte];
            uint8_t up_bits = up_q[quant_byte];
            unsigned gate_quant = (index & 1u) == 0 ?
                                  gate_bits & 0x0fu : gate_bits >> 4;
            unsigned up_quant = (index & 1u) == 0 ?
                                up_bits & 0x0fu : up_bits >> 4;
            size_t meta_index = row * groups_per_row + group;
            float gate_value = bf16_to_float(gate_s[meta_index]) * gate_quant +
                               bf16_to_float(gate_b[meta_index]);
            float up_value = bf16_to_float(up_s[meta_index]) * up_quant +
                             bf16_to_float(up_b[meta_index]);
            float activation_f32 = 0.75f * sinf((float)index * 0.013f) +
                                   0.20f * cosf((float)index * 0.031f);
            float activation = half_bits_to_float(
                float_to_half_bits(activation_f32));
            gate_sum += gate_value * activation;
            up_sum += up_value * activation;
        }
        output[row] = (gate_sum / (1.0f + expf(-gate_sum))) * up_sum;
    }
    return 0;
}

static int source_mlp_reference_first_8(
    int source,
    const qwen38_tensor_view *gate_weight,
    const qwen38_tensor_view *gate_scale,
    const qwen38_tensor_view *gate_bias,
    const qwen38_tensor_view *up_weight,
    const qwen38_tensor_view *up_scale,
    const qwen38_tensor_view *up_bias,
    const qwen38_tensor_view *down_weight,
    const qwen38_tensor_view *down_scale,
    const qwen38_tensor_view *down_bias,
    float output[8]) {
    const size_t hidden = QWEN38_HIDDEN_SIZE;
    const size_t intermediate_rows = QWEN38_MLP_SIZE;
    const size_t gate_groups = hidden / QWEN38_Q4_GROUP_SIZE;
    const size_t down_groups = intermediate_rows / QWEN38_Q4_GROUP_SIZE;
    uint8_t *gate_q = malloc((size_t)gate_weight->data_length);
    uint8_t *up_q = malloc((size_t)up_weight->data_length);
    uint16_t *gate_s = malloc((size_t)gate_scale->data_length);
    uint16_t *gate_b = malloc((size_t)gate_bias->data_length);
    uint16_t *up_s = malloc((size_t)up_scale->data_length);
    uint16_t *up_b = malloc((size_t)up_bias->data_length);
    uint8_t *down_q = malloc(8 * intermediate_rows / 2);
    uint16_t *down_s = malloc(8 * down_groups * sizeof(uint16_t));
    uint16_t *down_b = malloc(8 * down_groups * sizeof(uint16_t));
    float *intermediate = malloc(intermediate_rows * sizeof(float));
    if (gate_q == NULL || up_q == NULL || gate_s == NULL || gate_b == NULL ||
        up_s == NULL || up_b == NULL || down_q == NULL || down_s == NULL ||
        down_b == NULL || intermediate == NULL ||
        pread_exact(source, gate_q, (size_t)gate_weight->data_length,
                    gate_weight->data_start) != 0 ||
        pread_exact(source, up_q, (size_t)up_weight->data_length,
                    up_weight->data_start) != 0 ||
        pread_exact(source, gate_s, (size_t)gate_scale->data_length,
                    gate_scale->data_start) != 0 ||
        pread_exact(source, gate_b, (size_t)gate_bias->data_length,
                    gate_bias->data_start) != 0 ||
        pread_exact(source, up_s, (size_t)up_scale->data_length,
                    up_scale->data_start) != 0 ||
        pread_exact(source, up_b, (size_t)up_bias->data_length,
                    up_bias->data_start) != 0 ||
        pread_exact(source, down_q, 8 * intermediate_rows / 2,
                    down_weight->data_start) != 0 ||
        pread_exact(source, down_s, 8 * down_groups * sizeof(uint16_t),
                    down_scale->data_start) != 0 ||
        pread_exact(source, down_b, 8 * down_groups * sizeof(uint16_t),
                    down_bias->data_start) != 0) {
        free(gate_q); free(up_q); free(gate_s); free(gate_b);
        free(up_s); free(up_b); free(down_q); free(down_s); free(down_b);
        free(intermediate);
        return -1;
    }
    float *activation = malloc(hidden * sizeof(float));
    if (activation == NULL) {
        free(gate_q); free(up_q); free(gate_s); free(gate_b);
        free(up_s); free(up_b); free(down_q); free(down_s); free(down_b);
        free(intermediate);
        return -1;
    }
    for (size_t index = 0; index < hidden; ++index) {
        float value = 0.75f * sinf((float)index * 0.013f) +
                      0.20f * cosf((float)index * 0.031f);
        activation[index] = half_bits_to_float(float_to_half_bits(value));
    }
    for (size_t row = 0; row < intermediate_rows; ++row) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        size_t weight_row = row * hidden / 2;
        size_t meta_row = row * gate_groups;
        for (size_t index = 0; index < hidden; ++index) {
            size_t group = index / QWEN38_Q4_GROUP_SIZE;
            uint8_t gate_bits = gate_q[weight_row + index / 2];
            uint8_t up_bits = up_q[weight_row + index / 2];
            unsigned gate_quant = (index & 1u) == 0 ?
                                  gate_bits & 0x0fu : gate_bits >> 4;
            unsigned up_quant = (index & 1u) == 0 ?
                                up_bits & 0x0fu : up_bits >> 4;
            size_t meta = meta_row + group;
            gate_sum += (bf16_to_float(gate_s[meta]) * gate_quant +
                         bf16_to_float(gate_b[meta])) * activation[index];
            up_sum += (bf16_to_float(up_s[meta]) * up_quant +
                       bf16_to_float(up_b[meta])) * activation[index];
        }
        intermediate[row] =
            (gate_sum / (1.0f + expf(-gate_sum))) * up_sum;
    }
    for (size_t row = 0; row < 8; ++row) {
        float sum = 0.0f;
        size_t weight_row = row * intermediate_rows / 2;
        size_t meta_row = row * down_groups;
        for (size_t index = 0; index < intermediate_rows; ++index) {
            uint8_t bits = down_q[weight_row + index / 2];
            unsigned quant = (index & 1u) == 0 ? bits & 0x0fu : bits >> 4;
            size_t meta = meta_row + index / QWEN38_Q4_GROUP_SIZE;
            float weight = bf16_to_float(down_s[meta]) * quant +
                           bf16_to_float(down_b[meta]);
            sum += weight * intermediate[index];
        }
        output[row] = sum + activation[row];
    }
    free(activation);
    free(gate_q); free(up_q); free(gate_s); free(gate_b);
    free(up_s); free(up_b); free(down_q); free(down_s); free(down_b);
    free(intermediate);
    return 0;
}

static int validate_tensor(const qwen38_tensor_view *view, const char *name,
                           const char *dtype, uint64_t rows, uint64_t columns,
                           uint64_t expected_bytes) {
    if (strcmp(view->dtype, dtype) != 0 || view->rank != 2 ||
        view->shape[0] != rows || view->shape[1] != columns ||
        view->data_length != expected_bytes) {
        fprintf(stderr,
                "%s: expected %s [%" PRIu64 ", %" PRIu64
                "] / %" PRIu64 " bytes, got %s rank %zu / %" PRIu64
                " bytes\n",
                name, dtype, rows, columns, expected_bytes, view->dtype,
                view->rank, view->data_length);
        return -1;
    }
    return 0;
}

static int validate_vector(const qwen38_tensor_view *view, const char *name,
                           uint64_t values) {
    if (strcmp(view->dtype, "BF16") != 0 || view->rank != 1 ||
        view->shape[0] != values ||
        view->data_length != values * sizeof(uint16_t)) {
        fprintf(stderr,
                "%s: expected BF16 [%.0f] / %.0f bytes, got %s rank %zu / "
                "%" PRIu64 " bytes\n",
                name, (double)values, (double)(values * sizeof(uint16_t)),
                view->dtype, view->rank, view->data_length);
        return -1;
    }
    return 0;
}

static int validate_conv(const qwen38_tensor_view *view, const char *name) {
    if (strcmp(view->dtype, "BF16") != 0 || view->rank != 3 ||
        view->shape[0] != DELTA_QKV_ROWS || view->shape[1] != 4 ||
        view->shape[2] != 1 ||
        view->data_length != DELTA_CONV_VALUES * sizeof(uint16_t)) {
        fprintf(stderr, "%s: expected BF16 [%u, 4, 1]\n",
                name, DELTA_QKV_ROWS);
        return -1;
    }
    return 0;
}

static int copy_weight(int source, int output,
                       const qwen38_tensor_view *weight,
                       uint64_t output_offset) {
    enum { CHUNK_BYTES = 8 * 1024 * 1024 };
    unsigned char *buffer = malloc(CHUNK_BYTES);
    if (buffer == NULL) {
        return -1;
    }
    uint64_t done = 0;
    while (done < weight->data_length) {
        size_t amount = (size_t)(weight->data_length - done);
        if (amount > CHUNK_BYTES) {
            amount = CHUNK_BYTES;
        }
        if (pread_exact(source, buffer, amount, weight->data_start + done) != 0 ||
            pwrite_exact(output, buffer, amount, output_offset + done) != 0) {
            free(buffer);
            return -1;
        }
        done += amount;
    }
    free(buffer);
    return 0;
}

static int convert_metadata(int source, int output,
                            const qwen38_tensor_view *scales,
                            const qwen38_tensor_view *biases,
                            size_t value_count, uint64_t output_offset) {
    size_t source_bytes = value_count * sizeof(uint16_t);
    uint16_t *scale_data = malloc(source_bytes);
    uint16_t *bias_data = malloc(source_bytes);
    uint16_t *interleaved = malloc(source_bytes * 2);
    if (scale_data == NULL || bias_data == NULL || interleaved == NULL ||
        pread_exact(source, scale_data, source_bytes, scales->data_start) != 0 ||
        pread_exact(source, bias_data, source_bytes, biases->data_start) != 0) {
        free(scale_data);
        free(bias_data);
        free(interleaved);
        return -1;
    }
    for (size_t index = 0; index < value_count; ++index) {
        interleaved[index * 2] = bf16_to_half(scale_data[index]);
        interleaved[index * 2 + 1] = bf16_to_half(bias_data[index]);
    }
    int status = pwrite_exact(output, interleaved, source_bytes * 2,
                              output_offset);
    free(scale_data);
    free(bias_data);
    free(interleaved);
    return status;
}

static int append_weight(int source, int output,
                         const qwen38_tensor_view *weight,
                         uint64_t *output_offset) {
    int status = copy_weight(source, output, weight, *output_offset);
    *output_offset += weight->data_length;
    return status;
}

static int append_metadata(int source, int output,
                           const qwen38_tensor_view *scales,
                           const qwen38_tensor_view *biases,
                           uint64_t *output_offset) {
    size_t values = (size_t)(scales->data_length / sizeof(uint16_t));
    int status = convert_metadata(source, output, scales, biases, values,
                                  *output_offset);
    *output_offset += values * 2 * sizeof(uint16_t);
    return status;
}

static int write_bf16_as_f32(int source, int output,
                             const qwen38_tensor_view *tensor,
                             uint64_t output_offset) {
    size_t values = (size_t)(tensor->data_length / sizeof(uint16_t));
    uint16_t *input = malloc(values * sizeof(*input));
    float *converted = malloc(values * sizeof(*converted));
    if (input == NULL || converted == NULL ||
        pread_exact(source, input, tensor->data_length,
                    tensor->data_start) != 0) {
        free(input);
        free(converted);
        return -1;
    }
    for (size_t index = 0; index < values; ++index) {
        converted[index] = bf16_to_float(input[index]);
    }
    int status = pwrite_exact(output, converted,
                              values * sizeof(*converted), output_offset);
    free(input);
    free(converted);
    return status;
}

static int find_tensor(const char *path, const char *name,
                       qwen38_tensor_view *view) {
    char error[512];
    int status = qwen38_safetensors_find(path, name, 1, view,
                                         error, sizeof(error));
    if (status != 0) {
        fprintf(stderr, "%s\n", error);
    }
    return status;
}

int main(int argc, char **argv) {
    if ((argc != 5 && argc != 7) ||
        strlen(argv[3]) != QWEN38_M3_SOURCE_SHA256_LENGTH ||
        (argc == 7 &&
         strlen(argv[6]) != QWEN38_M3_SOURCE_SHA256_LENGTH)) {
        fprintf(stderr, "usage: %s SOURCE.safetensors OUTPUT.q38m3 "
                        "SOURCE_SHA256 LAYER_INDEX "
                        "[MLP_SOURCE.safetensors MLP_SOURCE_SHA256]\n",
                argv[0]);
        return 2;
    }
    char *layer_end = NULL;
    errno = 0;
    unsigned long parsed_layer = strtoul(argv[4], &layer_end, 10);
    if (errno != 0 || layer_end == argv[4] || *layer_end != '\0' ||
        parsed_layer >= 64 || parsed_layer % 4 == 3) {
        fprintf(stderr, "layer index must select one of the 48 DeltaNet layers\n");
        return 2;
    }
    unsigned layer_index = (unsigned)parsed_layer;
    const char *mlp_path = argc == 7 ? argv[5] : argv[1];
    const char *mlp_sha = argc == 7 ? argv[6] : argv[3];
    int core_pinned =
        strcmp(argv[3], QWEN38_M3_EXPECTED_SOURCE_SHA256) == 0 ||
        strcmp(argv[3], QWEN38_M3_EXPECTED_SOURCE_SHA256_2) == 0 ||
        strcmp(argv[3], QWEN38_M3_EXPECTED_SOURCE_SHA256_3) == 0;
    int mlp_pinned =
        strcmp(mlp_sha, QWEN38_M3_EXPECTED_SOURCE_SHA256) == 0 ||
        strcmp(mlp_sha, QWEN38_M3_EXPECTED_SOURCE_SHA256_2) == 0 ||
        strcmp(mlp_sha, QWEN38_M3_EXPECTED_SOURCE_SHA256_3) == 0;
    if (!core_pinned || !mlp_pinned) {
        fprintf(stderr, "source SHA-256 is not the compiled checkpoint pin\n");
        return 2;
    }
#define MAKE_LAYER_NAME(variable, suffix) \
    char variable##_storage[160]; \
    snprintf(variable##_storage, sizeof(variable##_storage), \
             "language_model.model.layers.%u.%s", layer_index, suffix); \
    const char *const variable = variable##_storage
    MAKE_LAYER_NAME(GATE_WEIGHT, "mlp.gate_proj.weight");
    MAKE_LAYER_NAME(GATE_SCALE, "mlp.gate_proj.scales");
    MAKE_LAYER_NAME(GATE_BIAS, "mlp.gate_proj.biases");
    MAKE_LAYER_NAME(UP_WEIGHT, "mlp.up_proj.weight");
    MAKE_LAYER_NAME(UP_SCALE, "mlp.up_proj.scales");
    MAKE_LAYER_NAME(UP_BIAS, "mlp.up_proj.biases");
    MAKE_LAYER_NAME(DOWN_WEIGHT, "mlp.down_proj.weight");
    MAKE_LAYER_NAME(DOWN_SCALE, "mlp.down_proj.scales");
    MAKE_LAYER_NAME(DOWN_BIAS, "mlp.down_proj.biases");
    MAKE_LAYER_NAME(INPUT_NORM, "input_layernorm.weight");
    MAKE_LAYER_NAME(POST_NORM, "post_attention_layernorm.weight");
    MAKE_LAYER_NAME(DELTA_QKV_WEIGHT, "linear_attn.in_proj_qkv.weight");
    MAKE_LAYER_NAME(DELTA_QKV_SCALE, "linear_attn.in_proj_qkv.scales");
    MAKE_LAYER_NAME(DELTA_QKV_BIAS, "linear_attn.in_proj_qkv.biases");
    MAKE_LAYER_NAME(DELTA_Z_WEIGHT, "linear_attn.in_proj_z.weight");
    MAKE_LAYER_NAME(DELTA_Z_SCALE, "linear_attn.in_proj_z.scales");
    MAKE_LAYER_NAME(DELTA_Z_BIAS, "linear_attn.in_proj_z.biases");
    MAKE_LAYER_NAME(DELTA_A_WEIGHT, "linear_attn.in_proj_a.weight");
    MAKE_LAYER_NAME(DELTA_A_SCALE, "linear_attn.in_proj_a.scales");
    MAKE_LAYER_NAME(DELTA_A_BIAS, "linear_attn.in_proj_a.biases");
    MAKE_LAYER_NAME(DELTA_B_WEIGHT, "linear_attn.in_proj_b.weight");
    MAKE_LAYER_NAME(DELTA_B_SCALE, "linear_attn.in_proj_b.scales");
    MAKE_LAYER_NAME(DELTA_B_BIAS, "linear_attn.in_proj_b.biases");
    MAKE_LAYER_NAME(DELTA_OUT_WEIGHT, "linear_attn.out_proj.weight");
    MAKE_LAYER_NAME(DELTA_OUT_SCALE, "linear_attn.out_proj.scales");
    MAKE_LAYER_NAME(DELTA_OUT_BIAS, "linear_attn.out_proj.biases");
    MAKE_LAYER_NAME(DELTA_CONV, "linear_attn.conv1d.weight");
    MAKE_LAYER_NAME(DELTA_A_LOG, "linear_attn.A_log");
    MAKE_LAYER_NAME(DELTA_DT_BIAS, "linear_attn.dt_bias");
    MAKE_LAYER_NAME(DELTA_NORM, "linear_attn.norm.weight");
#undef MAKE_LAYER_NAME
    qwen38_tensor_view gate_weight, gate_scale, gate_bias;
    qwen38_tensor_view up_weight, up_scale, up_bias;
    qwen38_tensor_view down_weight, down_scale, down_bias;
    qwen38_tensor_view input_norm, post_norm, delta_conv;
    qwen38_tensor_view delta_a_log, delta_dt_bias, delta_norm;
    qwen38_tensor_view delta_qkv_weight, delta_qkv_scale, delta_qkv_bias;
    qwen38_tensor_view delta_z_weight, delta_z_scale, delta_z_bias;
    qwen38_tensor_view delta_a_weight, delta_a_scale, delta_a_bias;
    qwen38_tensor_view delta_b_weight, delta_b_scale, delta_b_bias;
    qwen38_tensor_view delta_out_weight, delta_out_scale, delta_out_bias;
    if (find_tensor(mlp_path, GATE_WEIGHT, &gate_weight) != 0 ||
        find_tensor(mlp_path, GATE_SCALE, &gate_scale) != 0 ||
        find_tensor(mlp_path, GATE_BIAS, &gate_bias) != 0 ||
        find_tensor(mlp_path, UP_WEIGHT, &up_weight) != 0 ||
        find_tensor(mlp_path, UP_SCALE, &up_scale) != 0 ||
        find_tensor(mlp_path, UP_BIAS, &up_bias) != 0 ||
        find_tensor(mlp_path, DOWN_WEIGHT, &down_weight) != 0 ||
        find_tensor(mlp_path, DOWN_SCALE, &down_scale) != 0 ||
        find_tensor(mlp_path, DOWN_BIAS, &down_bias) != 0 ||
        find_tensor(argv[1], INPUT_NORM, &input_norm) != 0 ||
        find_tensor(argv[1], POST_NORM, &post_norm) != 0 ||
        find_tensor(argv[1], DELTA_CONV, &delta_conv) != 0 ||
        find_tensor(argv[1], DELTA_A_LOG, &delta_a_log) != 0 ||
        find_tensor(argv[1], DELTA_DT_BIAS, &delta_dt_bias) != 0 ||
        find_tensor(argv[1], DELTA_NORM, &delta_norm) != 0 ||
        find_tensor(argv[1], DELTA_QKV_WEIGHT, &delta_qkv_weight) != 0 ||
        find_tensor(argv[1], DELTA_QKV_SCALE, &delta_qkv_scale) != 0 ||
        find_tensor(argv[1], DELTA_QKV_BIAS, &delta_qkv_bias) != 0 ||
        find_tensor(argv[1], DELTA_Z_WEIGHT, &delta_z_weight) != 0 ||
        find_tensor(argv[1], DELTA_Z_SCALE, &delta_z_scale) != 0 ||
        find_tensor(argv[1], DELTA_Z_BIAS, &delta_z_bias) != 0 ||
        find_tensor(argv[1], DELTA_A_WEIGHT, &delta_a_weight) != 0 ||
        find_tensor(argv[1], DELTA_A_SCALE, &delta_a_scale) != 0 ||
        find_tensor(argv[1], DELTA_A_BIAS, &delta_a_bias) != 0 ||
        find_tensor(argv[1], DELTA_B_WEIGHT, &delta_b_weight) != 0 ||
        find_tensor(argv[1], DELTA_B_SCALE, &delta_b_scale) != 0 ||
        find_tensor(argv[1], DELTA_B_BIAS, &delta_b_bias) != 0 ||
        find_tensor(argv[1], DELTA_OUT_WEIGHT, &delta_out_weight) != 0 ||
        find_tensor(argv[1], DELTA_OUT_SCALE, &delta_out_scale) != 0 ||
        find_tensor(argv[1], DELTA_OUT_BIAS, &delta_out_bias) != 0) {
        return 3;
    }
    uint64_t groups_per_row = QWEN38_HIDDEN_SIZE / QWEN38_Q4_GROUP_SIZE;
    uint64_t weight_bytes = (uint64_t)QWEN38_MLP_SIZE *
                            QWEN38_HIDDEN_SIZE / 2;
    uint64_t metadata_plane_bytes = (uint64_t)QWEN38_MLP_SIZE *
                                    groups_per_row * sizeof(uint16_t);
    if (validate_tensor(&gate_weight, GATE_WEIGHT, "U32", QWEN38_MLP_SIZE,
                        QWEN38_HIDDEN_SIZE / 8, weight_bytes) != 0 ||
        validate_tensor(&up_weight, UP_WEIGHT, "U32", QWEN38_MLP_SIZE,
                        QWEN38_HIDDEN_SIZE / 8, weight_bytes) != 0 ||
        validate_tensor(&gate_scale, GATE_SCALE, "BF16", QWEN38_MLP_SIZE,
                        groups_per_row, metadata_plane_bytes) != 0 ||
        validate_tensor(&gate_bias, GATE_BIAS, "BF16", QWEN38_MLP_SIZE,
                        groups_per_row, metadata_plane_bytes) != 0 ||
        validate_tensor(&up_scale, UP_SCALE, "BF16", QWEN38_MLP_SIZE,
                        groups_per_row, metadata_plane_bytes) != 0 ||
        validate_tensor(&up_bias, UP_BIAS, "BF16", QWEN38_MLP_SIZE,
                        groups_per_row, metadata_plane_bytes) != 0 ||
        validate_tensor(&down_weight, DOWN_WEIGHT, "U32", QWEN38_HIDDEN_SIZE,
                        QWEN38_MLP_SIZE / 8, weight_bytes) != 0 ||
        validate_tensor(&down_scale, DOWN_SCALE, "BF16", QWEN38_HIDDEN_SIZE,
                        QWEN38_MLP_SIZE / QWEN38_Q4_GROUP_SIZE,
                        metadata_plane_bytes) != 0 ||
        validate_tensor(&down_bias, DOWN_BIAS, "BF16", QWEN38_HIDDEN_SIZE,
                        QWEN38_MLP_SIZE / QWEN38_Q4_GROUP_SIZE,
                        metadata_plane_bytes) != 0) {
        return 4;
    }
    const uint64_t delta_input_groups =
        QWEN38_HIDDEN_SIZE / QWEN38_Q4_GROUP_SIZE;
    const uint64_t delta_qkv_weight_bytes =
        (uint64_t)DELTA_QKV_ROWS * QWEN38_HIDDEN_SIZE / 2;
    const uint64_t delta_z_weight_bytes =
        (uint64_t)DELTA_Z_ROWS * QWEN38_HIDDEN_SIZE / 2;
    const uint64_t delta_scalar_weight_bytes =
        (uint64_t)DELTA_SCALAR_ROWS * QWEN38_HIDDEN_SIZE / 2;
    const uint64_t delta_output_groups =
        DELTA_OUTPUT_INPUTS / QWEN38_Q4_GROUP_SIZE;
    const uint64_t delta_output_weight_bytes =
        (uint64_t)QWEN38_HIDDEN_SIZE * DELTA_OUTPUT_INPUTS / 2;
    if (validate_vector(&input_norm, INPUT_NORM, QWEN38_HIDDEN_SIZE) != 0 ||
        validate_vector(&post_norm, POST_NORM, QWEN38_HIDDEN_SIZE) != 0 ||
        validate_conv(&delta_conv, DELTA_CONV) != 0 ||
        validate_vector(&delta_a_log, DELTA_A_LOG, DELTA_SCALAR_ROWS) != 0 ||
        validate_vector(&delta_dt_bias, DELTA_DT_BIAS,
                        DELTA_SCALAR_ROWS) != 0 ||
        validate_vector(&delta_norm, DELTA_NORM,
                        QWEN38_DELTA_HEAD_SIZE) != 0 ||
        validate_tensor(&delta_qkv_weight, DELTA_QKV_WEIGHT, "U32",
                        DELTA_QKV_ROWS, QWEN38_HIDDEN_SIZE / 8,
                        delta_qkv_weight_bytes) != 0 ||
        validate_tensor(&delta_qkv_scale, DELTA_QKV_SCALE, "BF16",
                        DELTA_QKV_ROWS, delta_input_groups,
                        (uint64_t)DELTA_QKV_ROWS * delta_input_groups * 2) != 0 ||
        validate_tensor(&delta_qkv_bias, DELTA_QKV_BIAS, "BF16",
                        DELTA_QKV_ROWS, delta_input_groups,
                        (uint64_t)DELTA_QKV_ROWS * delta_input_groups * 2) != 0 ||
        validate_tensor(&delta_z_weight, DELTA_Z_WEIGHT, "U32",
                        DELTA_Z_ROWS, QWEN38_HIDDEN_SIZE / 8,
                        delta_z_weight_bytes) != 0 ||
        validate_tensor(&delta_z_scale, DELTA_Z_SCALE, "BF16",
                        DELTA_Z_ROWS, delta_input_groups,
                        (uint64_t)DELTA_Z_ROWS * delta_input_groups * 2) != 0 ||
        validate_tensor(&delta_z_bias, DELTA_Z_BIAS, "BF16",
                        DELTA_Z_ROWS, delta_input_groups,
                        (uint64_t)DELTA_Z_ROWS * delta_input_groups * 2) != 0 ||
        validate_tensor(&delta_a_weight, DELTA_A_WEIGHT, "U32",
                        DELTA_SCALAR_ROWS, QWEN38_HIDDEN_SIZE / 8,
                        delta_scalar_weight_bytes) != 0 ||
        validate_tensor(&delta_a_scale, DELTA_A_SCALE, "BF16",
                        DELTA_SCALAR_ROWS, delta_input_groups,
                        (uint64_t)DELTA_SCALAR_ROWS * delta_input_groups * 2) != 0 ||
        validate_tensor(&delta_a_bias, DELTA_A_BIAS, "BF16",
                        DELTA_SCALAR_ROWS, delta_input_groups,
                        (uint64_t)DELTA_SCALAR_ROWS * delta_input_groups * 2) != 0 ||
        validate_tensor(&delta_b_weight, DELTA_B_WEIGHT, "U32",
                        DELTA_SCALAR_ROWS, QWEN38_HIDDEN_SIZE / 8,
                        delta_scalar_weight_bytes) != 0 ||
        validate_tensor(&delta_b_scale, DELTA_B_SCALE, "BF16",
                        DELTA_SCALAR_ROWS, delta_input_groups,
                        (uint64_t)DELTA_SCALAR_ROWS * delta_input_groups * 2) != 0 ||
        validate_tensor(&delta_b_bias, DELTA_B_BIAS, "BF16",
                        DELTA_SCALAR_ROWS, delta_input_groups,
                        (uint64_t)DELTA_SCALAR_ROWS * delta_input_groups * 2) != 0 ||
        validate_tensor(&delta_out_weight, DELTA_OUT_WEIGHT, "U32",
                        QWEN38_HIDDEN_SIZE, DELTA_OUTPUT_INPUTS / 8,
                        delta_output_weight_bytes) != 0 ||
        validate_tensor(&delta_out_scale, DELTA_OUT_SCALE, "BF16",
                        QWEN38_HIDDEN_SIZE, delta_output_groups,
                        (uint64_t)QWEN38_HIDDEN_SIZE * delta_output_groups * 2) != 0 ||
        validate_tensor(&delta_out_bias, DELTA_OUT_BIAS, "BF16",
                        QWEN38_HIDDEN_SIZE, delta_output_groups,
                        (uint64_t)QWEN38_HIDDEN_SIZE * delta_output_groups * 2) != 0) {
        return 4;
    }

    qwen38_m3_image_header header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, QWEN38_M3_IMAGE_MAGIC, sizeof(header.magic));
    header.version = QWEN38_M3_IMAGE_VERSION;
    header.header_bytes = QWEN38_M3_IMAGE_HEADER_BYTES;
    header.hidden_size = QWEN38_HIDDEN_SIZE;
    header.rows = QWEN38_MLP_SIZE;
    header.group_size = QWEN38_Q4_GROUP_SIZE;
    header.down_rows = QWEN38_HIDDEN_SIZE;
    header.down_groups_per_row = QWEN38_MLP_SIZE / QWEN38_Q4_GROUP_SIZE;
    header.gate_quants_offset = QWEN38_M3_IMAGE_HEADER_BYTES;
    header.gate_quants_bytes = weight_bytes;
    header.gate_metadata_offset = header.gate_quants_offset + weight_bytes;
    header.gate_metadata_bytes = metadata_plane_bytes * 2;
    header.up_quants_offset = header.gate_metadata_offset +
                             header.gate_metadata_bytes;
    header.up_quants_bytes = weight_bytes;
    header.up_metadata_offset = header.up_quants_offset + weight_bytes;
    header.up_metadata_bytes = metadata_plane_bytes * 2;
    header.down_quants_offset = header.up_metadata_offset +
                                header.up_metadata_bytes;
    header.down_quants_bytes = weight_bytes;
    header.down_metadata_offset = header.down_quants_offset + weight_bytes;
    header.down_metadata_bytes = metadata_plane_bytes * 2;
    header.layer_index = layer_index;
    header.delta_input_rows = DELTA_INPUT_ROWS;
    header.delta_input_groups_per_row = delta_input_groups;
    header.delta_output_rows = QWEN38_HIDDEN_SIZE;
    header.delta_output_groups_per_row = delta_output_groups;
    header.input_norm_constants_index = 0;
    header.post_norm_constants_index = QWEN38_HIDDEN_SIZE;
    header.conv_constants_index = 2 * QWEN38_HIDDEN_SIZE;
    header.a_log_constants_index =
        header.conv_constants_index + DELTA_CONV_VALUES;
    header.dt_bias_constants_index =
        header.a_log_constants_index + DELTA_SCALAR_ROWS;
    header.recurrent_norm_constants_index =
        header.dt_bias_constants_index + DELTA_SCALAR_ROWS;
    header.constants_f32_count =
        (uint32_t)(header.recurrent_norm_constants_index +
                   QWEN38_DELTA_HEAD_SIZE);
    header.constants_offset = align_page(header.down_metadata_offset +
                                         header.down_metadata_bytes);
    header.constants_bytes =
        align_page((uint64_t)header.constants_f32_count * sizeof(float));
    header.delta_input_quants_offset =
        header.constants_offset + header.constants_bytes;
    header.delta_input_quants_bytes =
        delta_qkv_weight_bytes + delta_z_weight_bytes +
        2 * delta_scalar_weight_bytes;
    header.delta_input_metadata_offset =
        header.delta_input_quants_offset + header.delta_input_quants_bytes;
    uint64_t delta_input_metadata_payload =
        (uint64_t)DELTA_INPUT_ROWS * delta_input_groups *
        2 * sizeof(uint16_t);
    header.delta_input_metadata_bytes =
        align_page(delta_input_metadata_payload);
    header.delta_output_quants_offset =
        header.delta_input_metadata_offset +
        header.delta_input_metadata_bytes;
    header.delta_output_quants_bytes = delta_output_weight_bytes;
    header.delta_output_metadata_offset =
        header.delta_output_quants_offset +
        header.delta_output_quants_bytes;
    header.delta_output_metadata_bytes =
        (uint64_t)QWEN38_HIDDEN_SIZE * delta_output_groups *
        2 * sizeof(uint16_t);
    memcpy(header.source_sha256, argv[3], QWEN38_M3_SOURCE_SHA256_LENGTH);
    memcpy(header.mlp_source_sha256, mlp_sha,
           QWEN38_M3_SOURCE_SHA256_LENGTH);

    int source = open(argv[1], O_RDONLY);
    if (source < 0) {
        fprintf(stderr, "open: %s\n", strerror(errno));
        return 5;
    }
    if (verify_source_sha256(source, argv[3]) != 0) {
        close(source);
        return 6;
    }
    int mlp_source = source;
    if (strcmp(mlp_path, argv[1]) != 0) {
        mlp_source = open(mlp_path, O_RDONLY);
        if (mlp_source < 0 ||
            verify_source_sha256(mlp_source, mlp_sha) != 0) {
            if (mlp_source >= 0) close(mlp_source);
            close(source);
            return 6;
        }
    }
    int output = open(argv[2], O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (output < 0) {
        fprintf(stderr, "open: %s\n", strerror(errno));
        if (mlp_source != source) close(mlp_source);
        close(source);
        return 5;
    }
    header.source_reference_count = 8;
    if (source_reference_first_8(mlp_source, &gate_weight, &gate_scale, &gate_bias,
                                 &up_weight, &up_scale, &up_bias,
                                 header.source_reference_first_8) != 0) {
        fprintf(stderr, "cannot calculate BF16 source reference\n");
        if (mlp_source != source) close(mlp_source);
        close(source);
        close(output);
        unlink(argv[2]);
        return 7;
    }
    header.source_mlp_reference_count = 8;
    if (source_mlp_reference_first_8(
            mlp_source, &gate_weight, &gate_scale, &gate_bias,
            &up_weight, &up_scale, &up_bias,
            &down_weight, &down_scale, &down_bias,
            header.source_mlp_reference_first_8) != 0) {
        fprintf(stderr, "cannot calculate full MLP BF16 source reference\n");
        if (mlp_source != source) close(mlp_source);
        close(source);
        close(output);
        unlink(argv[2]);
        return 7;
    }
    uint64_t delta_quant_cursor = header.delta_input_quants_offset;
    uint64_t delta_metadata_cursor = header.delta_input_metadata_offset;
    int failed = pwrite_exact(output, &header, sizeof(header), 0) != 0 ||
                 copy_weight(mlp_source, output, &gate_weight,
                             header.gate_quants_offset) != 0 ||
                 convert_metadata(mlp_source, output, &gate_scale, &gate_bias,
                                  (size_t)(gate_scale.data_length / 2),
                                  header.gate_metadata_offset) != 0 ||
                 copy_weight(mlp_source, output, &up_weight,
                             header.up_quants_offset) != 0 ||
                 convert_metadata(mlp_source, output, &up_scale, &up_bias,
                                  (size_t)(up_scale.data_length / 2),
                                  header.up_metadata_offset) != 0 ||
                 copy_weight(mlp_source, output, &down_weight,
                             header.down_quants_offset) != 0 ||
                 convert_metadata(mlp_source, output, &down_scale, &down_bias,
                                  (size_t)(down_scale.data_length / 2),
                                  header.down_metadata_offset) != 0 ||
                 write_bf16_as_f32(source, output, &input_norm,
                    header.constants_offset +
                    header.input_norm_constants_index * sizeof(float)) != 0 ||
                 write_bf16_as_f32(source, output, &post_norm,
                    header.constants_offset +
                    header.post_norm_constants_index * sizeof(float)) != 0 ||
                 write_bf16_as_f32(source, output, &delta_conv,
                    header.constants_offset +
                    header.conv_constants_index * sizeof(float)) != 0 ||
                 write_bf16_as_f32(source, output, &delta_a_log,
                    header.constants_offset +
                    header.a_log_constants_index * sizeof(float)) != 0 ||
                 write_bf16_as_f32(source, output, &delta_dt_bias,
                    header.constants_offset +
                    header.dt_bias_constants_index * sizeof(float)) != 0 ||
                 write_bf16_as_f32(source, output, &delta_norm,
                    header.constants_offset +
                    header.recurrent_norm_constants_index * sizeof(float)) != 0 ||
                 append_weight(source, output, &delta_qkv_weight,
                               &delta_quant_cursor) != 0 ||
                 append_weight(source, output, &delta_z_weight,
                               &delta_quant_cursor) != 0 ||
                 append_weight(source, output, &delta_a_weight,
                               &delta_quant_cursor) != 0 ||
                 append_weight(source, output, &delta_b_weight,
                               &delta_quant_cursor) != 0 ||
                 append_metadata(source, output, &delta_qkv_scale,
                                 &delta_qkv_bias,
                                 &delta_metadata_cursor) != 0 ||
                 append_metadata(source, output, &delta_z_scale,
                                 &delta_z_bias,
                                 &delta_metadata_cursor) != 0 ||
                 append_metadata(source, output, &delta_a_scale,
                                 &delta_a_bias,
                                 &delta_metadata_cursor) != 0 ||
                 append_metadata(source, output, &delta_b_scale,
                                 &delta_b_bias,
                                 &delta_metadata_cursor) != 0 ||
                 copy_weight(source, output, &delta_out_weight,
                             header.delta_output_quants_offset) != 0 ||
                 convert_metadata(source, output, &delta_out_scale,
                                  &delta_out_bias,
                                  (size_t)(delta_out_scale.data_length / 2),
                                  header.delta_output_metadata_offset) != 0 ||
                 delta_quant_cursor !=
                    header.delta_input_quants_offset +
                    header.delta_input_quants_bytes ||
                 delta_metadata_cursor !=
                    header.delta_input_metadata_offset +
                    delta_input_metadata_payload ||
                 ftruncate(output,
                    (off_t)(header.delta_output_metadata_offset +
                            header.delta_output_metadata_bytes)) != 0 ||
                 fsync(output) != 0;
    if (mlp_source != source) close(mlp_source);
    close(source);
    close(output);
    if (failed) {
        fprintf(stderr, "packing failed: %s\n", strerror(errno));
        unlink(argv[2]);
        return 8;
    }
    printf("{\"source\":\"%s\",\"output\":\"%s\","
           "\"source_sha256\":\"%s\",\"bytes\":%" PRIu64 "}\n",
           argv[1], argv[2], argv[3],
           header.delta_output_metadata_offset +
           header.delta_output_metadata_bytes);
    return 0;
}
