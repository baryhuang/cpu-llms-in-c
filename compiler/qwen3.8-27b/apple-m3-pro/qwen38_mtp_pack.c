/* Pack the Qwen3.8-27B MTP extras image (mtp.q38mtp) from the
 * standalone quantized MTP checkpoint
 * (mlx-community/Qwen3.8-27B-MTP-4bit): the fc projection arrives
 * already in the affine Q4 group-64 layout the runtime consumes, and
 * the three norm vectors arrive already folded to direct multipliers —
 * unlike the official BF16 mtp.* tensors, whose Hugging Face delta
 * convention this packer has to fold at pack time. The MTP
 * transformer layer itself is packed separately by
 * qwen38-m3-attention-pack with layer index 64. */

#define _POSIX_C_SOURCE 200809L

#include "qwen38_m3.h"
#include "qwen38_m3_mtp_image.h"
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

static int find_tensor(const char *path, const char *name,
                       qwen38_tensor_view *view) {
    char error[512];
    int status = qwen38_safetensors_find(path, name, 1, view,
                                         error, sizeof(error));
    if (status != 0) fprintf(stderr, "%s\n", error);
    return status;
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
        combined[index * 2] =
            float_to_half_bits(bf16_to_float(scales[index]));
        combined[index * 2 + 1] =
            float_to_half_bits(bf16_to_float(biases[index]));
    }
    int status = pwrite_exact(output, combined, values * 4, output_offset);
    free(scales); free(biases); free(combined);
    return status;
}

/* Direct multiplier conversion: the source vectors are already folded. */
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
    if (argc != 4 || strlen(argv[3]) != 64) {
        fprintf(stderr, "usage: %s MTP.safetensors OUTPUT.q38mtp "
                        "SOURCE_SHA256\n", argv[0]);
        return 2;
    }
    if (strcmp(argv[3], QWEN38_M3_EXPECTED_MTP_SHA256) != 0) {
        fprintf(stderr, "source is not the pinned Qwen3.8 MTP file\n");
        return 2;
    }
    qwen38_tensor_view fc_w, fc_s, fc_b;
    qwen38_tensor_view embedding_norm, hidden_norm, final_norm;
    if (find_tensor(argv[1], "fc.weight", &fc_w) ||
        find_tensor(argv[1], "fc.scales", &fc_s) ||
        find_tensor(argv[1], "fc.biases", &fc_b) ||
        find_tensor(argv[1], "pre_fc_norm_embedding.weight",
                    &embedding_norm) ||
        find_tensor(argv[1], "pre_fc_norm_hidden.weight", &hidden_norm) ||
        find_tensor(argv[1], "norm.weight", &final_norm)) {
        return 3;
    }
    const uint64_t fc_rows = 5120;
    const uint64_t fc_groups = 160;
    const uint64_t fc_quant_bytes = fc_rows * fc_groups * 64 / 2;
    const uint64_t fc_meta_bytes = fc_rows * fc_groups * 4;
    if (strcmp(fc_w.dtype, "U32") != 0 || fc_w.rank != 2 ||
        fc_w.shape[0] != fc_rows || fc_w.shape[1] != 1280 ||
        fc_w.data_length != fc_quant_bytes ||
        strcmp(fc_s.dtype, "BF16") != 0 || fc_s.shape[0] != fc_rows ||
        fc_s.shape[1] != fc_groups ||
        strcmp(fc_b.dtype, "BF16") != 0 || fc_b.shape[0] != fc_rows ||
        fc_b.shape[1] != fc_groups ||
        strcmp(embedding_norm.dtype, "BF16") != 0 ||
        embedding_norm.shape[0] != 5120 ||
        strcmp(hidden_norm.dtype, "BF16") != 0 ||
        hidden_norm.shape[0] != 5120 ||
        strcmp(final_norm.dtype, "BF16") != 0 ||
        final_norm.shape[0] != 5120) {
        fprintf(stderr, "unexpected MTP tensor dtype or shape\n");
        return 4;
    }

    qwen38_m3_mtp_image_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, QWEN38_M3_MTP_IMAGE_MAGIC, 8);
    h.version = QWEN38_M3_MTP_IMAGE_VERSION;
    h.header_bytes = QWEN38_M3_MTP_HEADER_BYTES;
    h.hidden_size = 5120;
    h.fc_rows = (uint32_t)fc_rows;
    h.fc_groups_per_row = (uint32_t)fc_groups;
    h.group_size = 64;
    h.constants_f32_count = 3 * 5120;
    h.fc_quants_offset = QWEN38_M3_MTP_HEADER_BYTES;
    h.fc_quants_bytes = fc_quant_bytes;
    h.fc_metadata_offset = h.fc_quants_offset + fc_quant_bytes;
    h.fc_metadata_bytes = fc_meta_bytes;
    h.constants_offset = h.fc_metadata_offset + fc_meta_bytes;
    h.constants_bytes = (uint64_t)h.constants_f32_count * sizeof(float);
    h.embedding_norm_constants_index = 0;
    h.hidden_norm_constants_index = 5120;
    h.final_norm_constants_index = 2 * 5120;
    memcpy(h.source_sha256, argv[3], 64);

    int source = open(argv[1], O_RDONLY);
    if (source < 0 || verify_sha256(source, argv[3]) != 0) {
        if (source >= 0) close(source);
        return 5;
    }
    int output = open(argv[2], O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (output < 0) {
        fprintf(stderr, "cannot create %s: %s\n", argv[2],
                strerror(errno));
        close(source);
        return 5;
    }
    uint64_t total = h.constants_offset + h.constants_bytes;
    int failed =
        pwrite_exact(output, &h, sizeof(h), 0) != 0 ||
        copy_tensor(source, output, &fc_w, h.fc_quants_offset) != 0 ||
        convert_metadata(source, output, &fc_s, &fc_b,
                         h.fc_metadata_offset) != 0 ||
        convert_vector_f32(source, output, &embedding_norm,
            h.constants_offset +
                h.embedding_norm_constants_index * 4) != 0 ||
        convert_vector_f32(source, output, &hidden_norm,
            h.constants_offset + h.hidden_norm_constants_index * 4) != 0 ||
        convert_vector_f32(source, output, &final_norm,
            h.constants_offset + h.final_norm_constants_index * 4) != 0 ||
        ftruncate(output, (off_t)total) != 0 || fsync(output) != 0;
    close(source);
    close(output);
    if (failed) {
        fprintf(stderr, "packing failed: %s\n", strerror(errno));
        unlink(argv[2]);
        return 6;
    }
    printf("{\"source\":\"%s\",\"output\":\"%s\",\"source_sha256\":"
           "\"%s\",\"bytes\":%" PRIu64 "}\n",
           argv[1], argv[2], argv[3], total);
    return 0;
}
