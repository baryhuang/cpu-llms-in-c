#define _POSIX_C_SOURCE 200809L

#include "qwen36_m3.h"
#include "qwen36_m3_global_image.h"
#include "qwen36_sha256.h"
#include "qwen36_safetensors.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
    for (size_t index = 0; index < 32; ++index)
        snprintf(actual + index * 2, 3, "%02x", digest[index]);
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
    if (exponent == 0xffu)
        return (uint16_t)(sign | (mantissa ? 0x7e00u : 0x7c00u));
    int half_exponent = (int)exponent - 112;
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

static float bf16_float(uint16_t input) {
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

static int copy_tensor(int source, int output,
                       const qwen36_tensor_view *tensor,
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
                            const qwen36_tensor_view *scale,
                            const qwen36_tensor_view *bias,
                            uint64_t output_offset) {
    enum { VALUES_PER_CHUNK = 1024 * 1024 };
    uint16_t *scales = malloc(VALUES_PER_CHUNK * 2);
    uint16_t *biases = malloc(VALUES_PER_CHUNK * 2);
    uint16_t *combined = malloc(VALUES_PER_CHUNK * 4);
    if (scales == NULL || biases == NULL || combined == NULL) {
        free(scales); free(biases); free(combined);
        return -1;
    }
    size_t values = (size_t)(scale->data_length / 2);
    size_t done = 0;
    while (done < values) {
        size_t count = values - done;
        if (count > VALUES_PER_CHUNK) count = VALUES_PER_CHUNK;
        if (pread_exact(source, scales, count * 2,
                        scale->data_start + done * 2) != 0 ||
            pread_exact(source, biases, count * 2,
                        bias->data_start + done * 2) != 0) {
            free(scales); free(biases); free(combined);
            return -1;
        }
        for (size_t index = 0; index < count; ++index) {
            combined[index * 2] =
                float_to_half_bits(bf16_float(scales[index]));
            combined[index * 2 + 1] =
                float_to_half_bits(bf16_float(biases[index]));
        }
        if (pwrite_exact(output, combined, count * 4,
                         output_offset + done * 4) != 0) {
            free(scales); free(biases); free(combined);
            return -1;
        }
        done += count;
    }
    free(scales); free(biases); free(combined);
    return 0;
}

static int write_norm(int source, int output,
                      const qwen36_tensor_view *norm,
                      uint64_t output_offset) {
    uint16_t input[QWEN36_HIDDEN_SIZE];
    float converted[QWEN36_HIDDEN_SIZE];
    if (pread_exact(source, input, sizeof(input), norm->data_start) != 0)
        return -1;
    for (size_t index = 0; index < QWEN36_HIDDEN_SIZE; ++index)
        converted[index] = bf16_float(input[index]);
    return pwrite_exact(output, converted, sizeof(converted), output_offset);
}

static int valid_matrix(const qwen36_tensor_view *v, const char *dtype,
                        uint64_t rows, uint64_t columns, uint64_t bytes) {
    return strcmp(v->dtype, dtype) == 0 && v->rank == 2 &&
           v->shape[0] == rows && v->shape[1] == columns &&
           v->data_length == bytes;
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s SHARD1 SHARD3 OUTPUT.q36global "
                        "SHARD1_SHA256 SHARD3_SHA256\n", argv[0]);
        return 2;
    }
    if (strcmp(argv[4], QWEN36_M3_EXPECTED_SOURCE_SHA256) != 0 ||
        strcmp(argv[5], QWEN36_M3_EXPECTED_SOURCE_SHA256_3) != 0) {
        fprintf(stderr, "global tensors require pinned shard 1 and shard 3\n");
        return 2;
    }
    qwen36_tensor_view embed_w, embed_s, embed_b;
    qwen36_tensor_view head_w, head_s, head_b, norm;
    if (find_tensor(argv[1], "language_model.model.embed_tokens.weight",
                    &embed_w) ||
        find_tensor(argv[1], "language_model.model.embed_tokens.scales",
                    &embed_s) ||
        find_tensor(argv[1], "language_model.model.embed_tokens.biases",
                    &embed_b) ||
        find_tensor(argv[2], "language_model.lm_head.weight", &head_w) ||
        find_tensor(argv[2], "language_model.lm_head.scales", &head_s) ||
        find_tensor(argv[2], "language_model.lm_head.biases", &head_b) ||
        find_tensor(argv[2], "language_model.model.norm.weight", &norm)) {
        return 3;
    }
    const uint64_t quant_bytes =
        (uint64_t)QWEN36_VOCAB_SIZE * QWEN36_HIDDEN_SIZE / 2;
    const uint64_t meta_plane_bytes =
        (uint64_t)QWEN36_VOCAB_SIZE * 80 * 2;
    if (!valid_matrix(&embed_w, "U32", QWEN36_VOCAB_SIZE, 640,
                      quant_bytes) ||
        !valid_matrix(&embed_s, "BF16", QWEN36_VOCAB_SIZE, 80,
                      meta_plane_bytes) ||
        !valid_matrix(&embed_b, "BF16", QWEN36_VOCAB_SIZE, 80,
                      meta_plane_bytes) ||
        !valid_matrix(&head_w, "U32", QWEN36_VOCAB_SIZE, 640,
                      quant_bytes) ||
        !valid_matrix(&head_s, "BF16", QWEN36_VOCAB_SIZE, 80,
                      meta_plane_bytes) ||
        !valid_matrix(&head_b, "BF16", QWEN36_VOCAB_SIZE, 80,
                      meta_plane_bytes) ||
        strcmp(norm.dtype, "BF16") != 0 || norm.rank != 1 ||
        norm.shape[0] != QWEN36_HIDDEN_SIZE ||
        norm.data_length != QWEN36_HIDDEN_SIZE * 2) {
        fprintf(stderr, "global tensor dtype, shape or length mismatch\n");
        return 4;
    }
    int shard1 = open(argv[1], O_RDONLY);
    int shard3 = open(argv[2], O_RDONLY);
    if (shard1 < 0 || shard3 < 0 ||
        verify_sha256(shard1, argv[4]) != 0 ||
        verify_sha256(shard3, argv[5]) != 0) {
        if (shard1 >= 0) close(shard1);
        if (shard3 >= 0) close(shard3);
        return 5;
    }
    qwen36_m3_global_image_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, QWEN36_M3_GLOBAL_IMAGE_MAGIC, 8);
    h.version = QWEN36_M3_GLOBAL_IMAGE_VERSION;
    h.header_bytes = QWEN36_M3_GLOBAL_HEADER_BYTES;
    h.vocab_size = QWEN36_VOCAB_SIZE;
    h.hidden_size = QWEN36_HIDDEN_SIZE;
    h.group_size = QWEN36_Q4_GROUP_SIZE;
    h.constants_f32_count = QWEN36_HIDDEN_SIZE;
    uint64_t offset = QWEN36_M3_GLOBAL_HEADER_BYTES;
#define SEGMENT(field, bytes) do { \
    h.field##_offset = offset; h.field##_bytes = (bytes); \
    offset += (bytes); \
} while (0)
    SEGMENT(embedding_quants, quant_bytes);
    SEGMENT(embedding_metadata, meta_plane_bytes * 2);
    SEGMENT(lm_head_quants, quant_bytes);
    SEGMENT(lm_head_metadata, meta_plane_bytes * 2);
    SEGMENT(constants, QWEN36_HIDDEN_SIZE * sizeof(float));
#undef SEGMENT
    memcpy(h.embedding_source_sha256, argv[4], 64);
    memcpy(h.lm_head_source_sha256, argv[5], 64);
    int output = open(argv[3], O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (output < 0) {
        fprintf(stderr, "cannot create %s: %s\n", argv[3], strerror(errno));
        close(shard1); close(shard3);
        return 5;
    }
    int failed =
        pwrite_exact(output, &h, sizeof(h), 0) != 0 ||
        copy_tensor(shard1, output, &embed_w,
                    h.embedding_quants_offset) != 0 ||
        convert_metadata(shard1, output, &embed_s, &embed_b,
                         h.embedding_metadata_offset) != 0 ||
        copy_tensor(shard3, output, &head_w,
                    h.lm_head_quants_offset) != 0 ||
        convert_metadata(shard3, output, &head_s, &head_b,
                         h.lm_head_metadata_offset) != 0 ||
        write_norm(shard3, output, &norm, h.constants_offset) != 0 ||
        ftruncate(output, (off_t)offset) != 0 || fsync(output) != 0;
    close(shard1); close(shard3); close(output);
    if (failed) {
        fprintf(stderr, "global packing failed: %s\n", strerror(errno));
        unlink(argv[3]);
        return 6;
    }
    printf("{\"output\":\"%s\",\"bytes\":%" PRIu64 ","
           "\"embedding_source_sha256\":\"%s\","
           "\"lm_head_source_sha256\":\"%s\"}\n",
           argv[3], offset, argv[4], argv[5]);
    return 0;
}
