#define _POSIX_C_SOURCE 200809L

#include "qwen38_m3_attention_image.h"
#include "qwen38_m3_global_image.h"
#include "qwen38_m3_image.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

enum { SOURCE_COUNT = 65, TENSOR_CAPACITY = 2048 };

typedef struct {
    int file;
    const unsigned char *mapping;
    uint64_t bytes;
} mapped_source;

typedef enum {
    COPY_DIRECT,
    COPY_METADATA_SCALE,
    COPY_METADATA_BIAS,
    COPY_F32_TO_BF16
} copy_kind;

typedef struct {
    char name[192];
    const char *dtype;
    uint64_t shape[3];
    unsigned rank;
    unsigned source;
    uint64_t source_offset;
    uint64_t bytes;
    uint64_t output_offset;
    copy_kind kind;
} tensor_record;

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} string_buffer;

static mapped_source sources[SOURCE_COUNT];
static tensor_record tensors[TENSOR_CAPACITY];
static size_t tensor_count;

static int append_text(string_buffer *buffer, const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    va_list copied;
    va_copy(copied, arguments);
    int needed = vsnprintf(NULL, 0, format, copied);
    va_end(copied);
    if (needed < 0) {
        va_end(arguments);
        return -1;
    }
    size_t required = buffer->length + (size_t)needed + 1;
    if (required > buffer->capacity) {
        size_t capacity = buffer->capacity == 0 ? 4096 : buffer->capacity;
        while (capacity < required) {
            if (capacity > SIZE_MAX / 2) {
                va_end(arguments);
                return -1;
            }
            capacity *= 2;
        }
        char *data = realloc(buffer->data, capacity);
        if (data == NULL) {
            va_end(arguments);
            return -1;
        }
        buffer->data = data;
        buffer->capacity = capacity;
    }
    vsnprintf(buffer->data + buffer->length,
              buffer->capacity - buffer->length, format, arguments);
    va_end(arguments);
    buffer->length += (size_t)needed;
    return 0;
}

static int map_source(unsigned index, const char *path) {
    mapped_source *source = &sources[index];
    source->file = open(path, O_RDONLY);
    struct stat status;
    if (source->file < 0 || fstat(source->file, &status) != 0 ||
        status.st_size < 4096) {
        if (source->file >= 0) close(source->file);
        source->file = -1;
        fprintf(stderr, "cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    source->bytes = (uint64_t)status.st_size;
    source->mapping = mmap(NULL, (size_t)source->bytes, PROT_READ,
                           MAP_PRIVATE, source->file, 0);
    if (source->mapping == MAP_FAILED) {
        close(source->file);
        source->file = -1;
        fprintf(stderr, "cannot map %s: %s\n", path, strerror(errno));
        return -1;
    }
    return 0;
}

static void close_sources(void) {
    for (unsigned index = 0; index < SOURCE_COUNT; ++index) {
        if (sources[index].mapping != NULL &&
            sources[index].mapping != MAP_FAILED)
            munmap((void *)sources[index].mapping,
                   (size_t)sources[index].bytes);
        if (sources[index].file >= 0) close(sources[index].file);
    }
}

static uint64_t dtype_bytes(const char *dtype) {
    if (strcmp(dtype, "F16") == 0 || strcmp(dtype, "BF16") == 0) return 2;
    if (strcmp(dtype, "F32") == 0 || strcmp(dtype, "U32") == 0) return 4;
    return 0;
}

static int add_tensor(const char *name, const char *dtype,
                      unsigned rank, const uint64_t *shape,
                      unsigned source, uint64_t source_offset,
                      copy_kind kind) {
    if (tensor_count == TENSOR_CAPACITY || rank == 0 || rank > 3 ||
        source >= SOURCE_COUNT || strlen(name) >= sizeof(tensors[0].name))
        return -1;
    uint64_t bytes = dtype_bytes(dtype);
    for (unsigned index = 0; index < rank; ++index) {
        if (shape[index] != 0 && bytes > UINT64_MAX / shape[index]) return -1;
        bytes *= shape[index];
    }
    uint64_t source_bytes = kind == COPY_DIRECT ? bytes : bytes * 2;
    if (source_offset > sources[source].bytes ||
        source_bytes > sources[source].bytes - source_offset)
        return -1;
    tensor_record *record = &tensors[tensor_count++];
    memset(record, 0, sizeof(*record));
    memcpy(record->name, name, strlen(name) + 1);
    record->dtype = dtype;
    memcpy(record->shape, shape, rank * sizeof(shape[0]));
    record->rank = rank;
    record->source = source;
    record->source_offset = source_offset;
    record->bytes = bytes;
    record->kind = kind;
    return 0;
}

static int add_vector(const char *name, unsigned source, uint64_t offset,
                      uint64_t count) {
    uint64_t shape[] = {count};
    return add_tensor(name, "BF16", 1, shape, source, offset,
                      COPY_F32_TO_BF16);
}

static int add_q4(const char *prefix, unsigned source,
                  uint64_t quant_offset, uint64_t metadata_offset,
                  uint64_t rows, uint64_t packed_columns,
                  uint64_t groups) {
    char name[192];
    uint64_t weight_shape[] = {rows, packed_columns};
    uint64_t metadata_shape[] = {rows, groups};
    snprintf(name, sizeof(name), "%s.weight", prefix);
    if (add_tensor(name, "U32", 2, weight_shape, source, quant_offset,
                   COPY_DIRECT) != 0) return -1;
    snprintf(name, sizeof(name), "%s.scales", prefix);
    if (add_tensor(name, "F16", 2, metadata_shape, source, metadata_offset,
                   COPY_METADATA_SCALE) != 0) return -1;
    snprintf(name, sizeof(name), "%s.biases", prefix);
    return add_tensor(name, "F16", 2, metadata_shape, source,
                      metadata_offset, COPY_METADATA_BIAS);
}

static int add_common_layer(unsigned layer, unsigned source,
                            uint64_t gate_q, uint64_t gate_m,
                            uint64_t up_q, uint64_t up_m,
                            uint64_t down_q, uint64_t down_m,
                            uint64_t constants, uint64_t input_norm_index,
                            uint64_t post_norm_index) {
    char prefix[192];
    snprintf(prefix, sizeof(prefix),
             "language_model.model.layers.%u.mlp.gate_proj", layer);
    if (add_q4(prefix, source, gate_q, gate_m, 17408, 640, 80) != 0)
        return -1;
    snprintf(prefix, sizeof(prefix),
             "language_model.model.layers.%u.mlp.up_proj", layer);
    if (add_q4(prefix, source, up_q, up_m, 17408, 640, 80) != 0)
        return -1;
    snprintf(prefix, sizeof(prefix),
             "language_model.model.layers.%u.mlp.down_proj", layer);
    if (add_q4(prefix, source, down_q, down_m, 5120, 2176, 272) != 0)
        return -1;
    snprintf(prefix, sizeof(prefix),
             "language_model.model.layers.%u.input_layernorm.weight", layer);
    if (add_vector(prefix, source, constants + input_norm_index * 4,
                   5120) != 0) return -1;
    snprintf(prefix, sizeof(prefix),
             "language_model.model.layers.%u.post_attention_layernorm.weight",
             layer);
    return add_vector(prefix, source, constants + post_norm_index * 4, 5120);
}

static int add_delta_layer(unsigned layer, unsigned source,
                           const qwen38_m3_image_header *header) {
    if (memcmp(header->magic, QWEN38_M3_IMAGE_MAGIC, 8) != 0 ||
        header->version != QWEN38_M3_IMAGE_VERSION ||
        header->layer_index != layer || header->hidden_size != 5120 ||
        header->delta_input_rows != 16480 ||
        header->delta_output_rows != 5120 ||
        header->delta_output_metadata_offset +
            header->delta_output_metadata_bytes != sources[source].bytes)
        return -1;
    if (add_common_layer(layer, source,
                         header->gate_quants_offset,
                         header->gate_metadata_offset,
                         header->up_quants_offset,
                         header->up_metadata_offset,
                         header->down_quants_offset,
                         header->down_metadata_offset,
                         header->constants_offset,
                         header->input_norm_constants_index,
                         header->post_norm_constants_index) != 0)
        return -1;
    char prefix[192];
    uint64_t quant = header->delta_input_quants_offset;
    uint64_t meta = header->delta_input_metadata_offset;
#define ADD_DELTA_Q4(suffix, rows, packed, groups) do { \
    snprintf(prefix, sizeof(prefix), \
        "language_model.model.layers.%u.linear_attn.%s", layer, suffix); \
    if (add_q4(prefix, source, quant, meta, rows, packed, groups) != 0) \
        return -1; \
    quant += (uint64_t)(rows) * (packed) * 4; \
    meta += (uint64_t)(rows) * (groups) * 4; \
} while (0)
    ADD_DELTA_Q4("in_proj_qkv", 10240, 640, 80);
    ADD_DELTA_Q4("in_proj_z", 6144, 640, 80);
    ADD_DELTA_Q4("in_proj_a", 48, 640, 80);
    ADD_DELTA_Q4("in_proj_b", 48, 640, 80);
#undef ADD_DELTA_Q4
    snprintf(prefix, sizeof(prefix),
             "language_model.model.layers.%u.linear_attn.out_proj", layer);
    if (add_q4(prefix, source, header->delta_output_quants_offset,
               header->delta_output_metadata_offset,
               5120, 768, 96) != 0) return -1;
    snprintf(prefix, sizeof(prefix),
             "language_model.model.layers.%u.linear_attn.conv1d.weight",
             layer);
    uint64_t conv_shape[] = {10240, 4, 1};
    if (add_tensor(prefix, "BF16", 3, conv_shape, source,
                   header->constants_offset +
                       header->conv_constants_index * 4,
                   COPY_F32_TO_BF16) != 0) return -1;
#define ADD_DELTA_VECTOR(suffix, index, count) do { \
    snprintf(prefix, sizeof(prefix), \
        "language_model.model.layers.%u.linear_attn.%s", layer, suffix); \
    if (add_vector(prefix, source, header->constants_offset + \
                   (index) * 4, count) != 0) return -1; \
} while (0)
    ADD_DELTA_VECTOR("A_log", header->a_log_constants_index, 48);
    ADD_DELTA_VECTOR("dt_bias", header->dt_bias_constants_index, 48);
    ADD_DELTA_VECTOR("norm.weight",
                     header->recurrent_norm_constants_index, 128);
#undef ADD_DELTA_VECTOR
    return quant == header->delta_input_quants_offset +
                        header->delta_input_quants_bytes &&
           meta == header->delta_input_metadata_offset +
                       16480ull * 80 * 4 ? 0 : -1;
}

static int add_attention_layer(
    unsigned layer, unsigned source,
    const qwen38_m3_attention_image_header *header) {
    if (memcmp(header->magic, QWEN38_M3_ATTENTION_IMAGE_MAGIC, 8) != 0 ||
        header->version != QWEN38_M3_ATTENTION_IMAGE_VERSION ||
        header->layer_index != layer || header->hidden_size != 5120 ||
        header->attention_output_metadata_offset +
            header->attention_output_metadata_bytes != sources[source].bytes)
        return -1;
    if (add_common_layer(layer, source,
                         header->gate_quants_offset,
                         header->gate_metadata_offset,
                         header->up_quants_offset,
                         header->up_metadata_offset,
                         header->down_quants_offset,
                         header->down_metadata_offset,
                         header->constants_offset,
                         header->input_norm_constants_index,
                         header->post_norm_constants_index) != 0)
        return -1;
    char prefix[192];
    uint64_t quant = header->attention_input_quants_offset;
    uint64_t meta = header->attention_input_metadata_offset;
#define ADD_ATTENTION_Q4(suffix, rows) do { \
    snprintf(prefix, sizeof(prefix), \
        "language_model.model.layers.%u.self_attn.%s", layer, suffix); \
    if (add_q4(prefix, source, quant, meta, rows, 640, 80) != 0) \
        return -1; \
    quant += (uint64_t)(rows) * 640 * 4; \
    meta += (uint64_t)(rows) * 80 * 4; \
} while (0)
    ADD_ATTENTION_Q4("q_proj", 12288);
    ADD_ATTENTION_Q4("k_proj", 1024);
    ADD_ATTENTION_Q4("v_proj", 1024);
#undef ADD_ATTENTION_Q4
    snprintf(prefix, sizeof(prefix),
             "language_model.model.layers.%u.self_attn.o_proj", layer);
    if (add_q4(prefix, source, header->attention_output_quants_offset,
               header->attention_output_metadata_offset,
               5120, 768, 96) != 0) return -1;
    snprintf(prefix, sizeof(prefix),
             "language_model.model.layers.%u.self_attn.q_norm.weight", layer);
    if (add_vector(prefix, source,
                   header->constants_offset +
                       header->q_norm_constants_index * 4,
                   256) != 0) return -1;
    snprintf(prefix, sizeof(prefix),
             "language_model.model.layers.%u.self_attn.k_norm.weight", layer);
    if (add_vector(prefix, source,
                   header->constants_offset +
                       header->k_norm_constants_index * 4,
                   256) != 0) return -1;
    return quant == header->attention_input_quants_offset +
                        header->attention_input_quants_bytes &&
           meta == header->attention_input_metadata_offset +
                       header->attention_input_metadata_bytes ? 0 : -1;
}

static int add_global(const qwen38_m3_global_image_header *header) {
    if (memcmp(header->magic, QWEN38_M3_GLOBAL_IMAGE_MAGIC, 8) != 0 ||
        header->version != QWEN38_M3_GLOBAL_IMAGE_VERSION ||
        header->vocab_size != 248320 || header->hidden_size != 5120 ||
        header->constants_offset + header->constants_bytes !=
            sources[0].bytes)
        return -1;
    if (add_q4("language_model.model.embed_tokens", 0,
               header->embedding_quants_offset,
               header->embedding_metadata_offset,
               248320, 640, 80) != 0 ||
        add_q4("language_model.lm_head", 0,
               header->lm_head_quants_offset,
               header->lm_head_metadata_offset,
               248320, 640, 80) != 0 ||
        add_vector("language_model.model.norm.weight", 0,
                   header->constants_offset, 5120) != 0)
        return -1;
    return 0;
}

static int write_all(int file, const void *data, size_t bytes) {
    const unsigned char *cursor = data;
    while (bytes != 0) {
        ssize_t written = write(file, cursor, bytes);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return -1;
        cursor += written;
        bytes -= (size_t)written;
    }
    return 0;
}

static int write_tensor(int output, const tensor_record *record) {
    const unsigned char *source =
        sources[record->source].mapping + record->source_offset;
    enum { BUFFER_BYTES = 1024 * 1024 };
    if (record->kind == COPY_DIRECT) {
        uint64_t done = 0;
        while (done < record->bytes) {
            size_t amount = (size_t)(record->bytes - done);
            if (amount > BUFFER_BYTES) amount = BUFFER_BYTES;
            if (write_all(output, source + done, amount) != 0) return -1;
            done += amount;
        }
        return 0;
    }
    unsigned char *buffer = malloc(BUFFER_BYTES);
    if (buffer == NULL) return -1;
    uint64_t values = record->bytes / 2;
    uint64_t done = 0;
    if (record->kind == COPY_F32_TO_BF16) {
        uint64_t done = 0;
        while (done < values) {
            size_t count = (size_t)(values - done);
            if (count > BUFFER_BYTES / 2) count = BUFFER_BYTES / 2;
            for (size_t index = 0; index < count; ++index) {
                uint32_t bits;
                memcpy(&bits, source + (done + index) * 4, sizeof(bits));
                uint16_t bf16 = (uint16_t)(bits >> 16);
                memcpy(buffer + index * 2, &bf16, sizeof(bf16));
            }
            if (write_all(output, buffer, count * 2) != 0) {
                free(buffer);
                return -1;
            }
            done += count;
        }
        free(buffer);
        return 0;
    }
    unsigned lane = record->kind == COPY_METADATA_BIAS ? 2 : 0;
    while (done < values) {
        size_t count = (size_t)(values - done);
        if (count > BUFFER_BYTES / 2) count = BUFFER_BYTES / 2;
        for (size_t index = 0; index < count; ++index)
            memcpy(buffer + index * 2,
                   source + (done + index) * 4 + lane, 2);
        if (write_all(output, buffer, count * 2) != 0) {
            free(buffer);
            return -1;
        }
        done += count;
    }
    free(buffer);
    return 0;
}

static int write_safetensors(const char *path) {
    uint64_t data_offset = 0;
    for (size_t index = 0; index < tensor_count; ++index) {
        tensors[index].output_offset = data_offset;
        if (data_offset > UINT64_MAX - tensors[index].bytes) return -1;
        data_offset += tensors[index].bytes;
    }
    if (tensor_count != 1847 || data_offset != 15132802048ull) {
        fprintf(stderr,
                "compiled text inventory mismatch: %zu tensors / %" PRIu64
                " bytes\n", tensor_count, data_offset);
        return -1;
    }
    string_buffer header = {0};
    if (append_text(&header, "{") != 0) return -1;
    for (size_t index = 0; index < tensor_count; ++index) {
        tensor_record *record = &tensors[index];
        if (append_text(&header, "%s\"%s\":{\"dtype\":\"%s\","
                        "\"shape\":[", index == 0 ? "" : ",",
                        record->name, record->dtype) != 0) {
            free(header.data);
            return -1;
        }
        for (unsigned dimension = 0; dimension < record->rank; ++dimension) {
            if (append_text(&header, "%s%" PRIu64,
                            dimension == 0 ? "" : ",",
                            record->shape[dimension]) != 0) {
                free(header.data);
                return -1;
            }
        }
        if (append_text(&header, "],\"data_offsets\":[%" PRIu64
                        ",%" PRIu64 "]}", record->output_offset,
                        record->output_offset + record->bytes) != 0) {
            free(header.data);
            return -1;
        }
    }
    if (append_text(&header, "}") != 0) {
        free(header.data);
        return -1;
    }
    size_t padded = (header.length + 7) & ~(size_t)7;
    while (header.length < padded) header.data[header.length++] = ' ';
    int output = open(path, O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (output < 0) {
        fprintf(stderr, "cannot create %s: %s\n", path, strerror(errno));
        free(header.data);
        return -1;
    }
    uint64_t header_bytes = header.length;
    int failed = write_all(output, &header_bytes, sizeof(header_bytes)) != 0 ||
                 write_all(output, header.data, header.length) != 0;
    free(header.data);
    for (size_t index = 0; !failed && index < tensor_count; ++index) {
        if (write_tensor(output, &tensors[index]) != 0) failed = 1;
    }
    if (!failed && fsync(output) != 0) failed = 1;
    if (close(output) != 0) failed = 1;
    if (failed) {
        fprintf(stderr, "checkpoint export failed: %s\n", strerror(errno));
        unlink(path);
        return -1;
    }
    printf("{\"output\":\"%s\",\"tensor_count\":%zu,"
           "\"header_bytes\":%" PRIu64 ",\"data_bytes\":%" PRIu64
           ",\"total_bytes\":%" PRIu64 "}\n",
           path, tensor_count, header_bytes, data_offset,
           8 + header_bytes + data_offset);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s COMPILED_MODEL_DIR OUTPUT.safetensors\n",
                argv[0]);
        return 2;
    }
    for (unsigned index = 0; index < SOURCE_COUNT; ++index)
        sources[index].file = -1;
    char path[1024];
    snprintf(path, sizeof(path), "%s/global.q38global", argv[1]);
    if (map_source(0, path) != 0 ||
        add_global((const qwen38_m3_global_image_header *)
                       sources[0].mapping) != 0) {
        fprintf(stderr, "invalid global image\n");
        close_sources();
        return 3;
    }
    for (unsigned layer = 0; layer < 64; ++layer) {
        const char *suffix = layer % 4 == 3 ? "q38att" : "q38delta";
        snprintf(path, sizeof(path), "%s/layer-%02u.%s",
                 argv[1], layer, suffix);
        unsigned source = layer + 1;
        if (map_source(source, path) != 0) {
            close_sources();
            return 3;
        }
        int status = layer % 4 == 3 ?
            add_attention_layer(
                layer, source,
                (const qwen38_m3_attention_image_header *)
                    sources[source].mapping) :
            add_delta_layer(
                layer, source,
                (const qwen38_m3_image_header *)sources[source].mapping);
        if (status != 0) {
            fprintf(stderr, "invalid compiled layer %u\n", layer);
            close_sources();
            return 3;
        }
    }
    int status = write_safetensors(argv[2]) == 0 ? 0 : 4;
    close_sources();
    return status;
}
