#define _POSIX_C_SOURCE 200809L

#include "qwen38_sha256.h"
#include "qwen38_tokenizer.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    unsigned char *data;
    uint32_t length;
} token_string;

enum { VOCAB_HASH_SIZE = 1 << 20 };

static void skip_space(const char **cursor) {
    while (**cursor == ' ' || **cursor == '\n' ||
           **cursor == '\r' || **cursor == '\t') ++*cursor;
}

static int hex_value(unsigned char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

static size_t encode_utf8(uint32_t codepoint, unsigned char output[4]) {
    if (codepoint <= 0x7f) {
        output[0] = (unsigned char)codepoint;
        return 1;
    }
    if (codepoint <= 0x7ff) {
        output[0] = (unsigned char)(0xc0 | (codepoint >> 6));
        output[1] = (unsigned char)(0x80 | (codepoint & 0x3f));
        return 2;
    }
    if (codepoint <= 0xffff) {
        output[0] = (unsigned char)(0xe0 | (codepoint >> 12));
        output[1] = (unsigned char)(0x80 | ((codepoint >> 6) & 0x3f));
        output[2] = (unsigned char)(0x80 | (codepoint & 0x3f));
        return 3;
    }
    output[0] = (unsigned char)(0xf0 | (codepoint >> 18));
    output[1] = (unsigned char)(0x80 | ((codepoint >> 12) & 0x3f));
    output[2] = (unsigned char)(0x80 | ((codepoint >> 6) & 0x3f));
    output[3] = (unsigned char)(0x80 | (codepoint & 0x3f));
    return 4;
}

static int parse_json_string(const char **cursor, token_string *output) {
    skip_space(cursor);
    if (**cursor != '"') return -1;
    ++*cursor;
    size_t capacity = 64;
    unsigned char *data = malloc(capacity);
    if (data == NULL) return -1;
    size_t length = 0;
    while (**cursor != '\0' && **cursor != '"') {
        if (length + 4 > capacity) {
            size_t next_capacity = capacity * 2;
            unsigned char *next = realloc(data, next_capacity);
            if (next == NULL) {
                free(data);
                return -1;
            }
            data = next;
            capacity = next_capacity;
        }
        unsigned char value = (unsigned char)*(*cursor)++;
        if (value != '\\') {
            data[length++] = value;
            continue;
        }
        unsigned char escaped = (unsigned char)*(*cursor)++;
        switch (escaped) {
            case '"': case '\\': case '/': data[length++] = escaped; break;
            case 'b': data[length++] = '\b'; break;
            case 'f': data[length++] = '\f'; break;
            case 'n': data[length++] = '\n'; break;
            case 'r': data[length++] = '\r'; break;
            case 't': data[length++] = '\t'; break;
            case 'u': {
                uint32_t codepoint = 0;
                for (unsigned index = 0; index < 4; ++index) {
                    int digit = hex_value((unsigned char)(*cursor)[index]);
                    if (digit < 0) {
                        free(data);
                        return -1;
                    }
                    codepoint = (codepoint << 4) | (uint32_t)digit;
                }
                *cursor += 4;
                if (codepoint >= 0xd800 && codepoint <= 0xdbff &&
                    (*cursor)[0] == '\\' && (*cursor)[1] == 'u') {
                    *cursor += 2;
                    uint32_t low = 0;
                    for (unsigned index = 0; index < 4; ++index) {
                        int digit =
                            hex_value((unsigned char)(*cursor)[index]);
                        if (digit < 0) {
                            free(data);
                            return -1;
                        }
                        low = (low << 4) | (uint32_t)digit;
                    }
                    *cursor += 4;
                    if (low < 0xdc00 || low > 0xdfff) {
                        free(data);
                        return -1;
                    }
                    codepoint = 0x10000 +
                        ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                }
                unsigned char encoded[4];
                size_t count = encode_utf8(codepoint, encoded);
                memcpy(data + length, encoded, count);
                length += count;
                break;
            }
            default:
                free(data);
                return -1;
        }
    }
    if (**cursor != '"') {
        free(data);
        return -1;
    }
    ++*cursor;
    output->data = data;
    output->length = (uint32_t)length;
    return 0;
}

static uint64_t string_hash(const unsigned char *data, size_t length) {
    uint64_t hash = 1469598103934665603ull;
    for (size_t index = 0; index < length; ++index) {
        hash ^= data[index];
        hash *= 1099511628211ull;
    }
    return hash;
}

static int hash_insert(uint32_t *table, token_string *tokens,
                       uint32_t token_id) {
    uint64_t hash = string_hash(tokens[token_id].data,
                                tokens[token_id].length);
    size_t slot = (size_t)hash & (VOCAB_HASH_SIZE - 1);
    for (size_t probe = 0; probe < VOCAB_HASH_SIZE; ++probe) {
        if (table[slot] == 0) {
            table[slot] = token_id + 1;
            return 0;
        }
        slot = (slot + 1) & (VOCAB_HASH_SIZE - 1);
    }
    return -1;
}

static int hash_find(const uint32_t *table, const token_string *tokens,
                     const unsigned char *data, size_t length,
                     uint32_t *token_id) {
    uint64_t hash = string_hash(data, length);
    size_t slot = (size_t)hash & (VOCAB_HASH_SIZE - 1);
    for (size_t probe = 0; probe < VOCAB_HASH_SIZE; ++probe) {
        uint32_t stored = table[slot];
        if (stored == 0) return -1;
        uint32_t id = stored - 1;
        if (tokens[id].length == length &&
            memcmp(tokens[id].data, data, length) == 0) {
            *token_id = id;
            return 0;
        }
        slot = (slot + 1) & (VOCAB_HASH_SIZE - 1);
    }
    return -1;
}

static int parse_u32(const char **cursor, uint32_t *value) {
    skip_space(cursor);
    if (!isdigit((unsigned char)**cursor)) return -1;
    uint64_t result = 0;
    while (isdigit((unsigned char)**cursor)) {
        result = result * 10 + (unsigned)(*(*cursor)++ - '0');
        if (result > UINT32_MAX) return -1;
    }
    *value = (uint32_t)result;
    return 0;
}

static int parse_vocab(const char *json, token_string *tokens) {
    const char *model = strstr(json, "\"model\"");
    const char *vocab = model != NULL ? strstr(model, "\"vocab\"") : NULL;
    if (vocab == NULL || (vocab = strchr(vocab, ':')) == NULL) return -1;
    ++vocab;
    skip_space(&vocab);
    if (*vocab++ != '{') return -1;
    size_t count = 0;
    for (;;) {
        skip_space(&vocab);
        if (*vocab == '}') {
            ++vocab;
            break;
        }
        token_string text = {0};
        uint32_t id;
        if (parse_json_string(&vocab, &text) != 0) return -1;
        skip_space(&vocab);
        if (*vocab++ != ':') {
            free(text.data);
            return -1;
        }
        if (parse_u32(&vocab, &id) != 0 ||
            id >= QWEN38_TOKENIZER_BASE_VOCAB ||
            tokens[id].data != NULL) {
            free(text.data);
            return -1;
        }
        tokens[id] = text;
        ++count;
        skip_space(&vocab);
        if (*vocab == ',') ++vocab;
        else if (*vocab != '}') return -1;
    }
    return count == QWEN38_TOKENIZER_BASE_VOCAB ? 0 : -1;
}

static int parse_added(const char *json, token_string *tokens) {
    const char *cursor = strstr(json, "\"added_tokens\"");
    if (cursor == NULL || (cursor = strchr(cursor, '[')) == NULL) return -1;
    ++cursor;
    size_t count = 0;
    while (count < QWEN38_TOKENIZER_ADDED_COUNT) {
        const char *object = strchr(cursor, '{');
        if (object == NULL) return -1;
        const char *end = strchr(object, '}');
        if (end == NULL) return -1;
        const char *id_key = strstr(object, "\"id\"");
        const char *content_key = strstr(object, "\"content\"");
        if (id_key == NULL || content_key == NULL ||
            id_key >= end || content_key >= end ||
            (id_key = strchr(id_key, ':')) == NULL ||
            (content_key = strchr(content_key, ':')) == NULL ||
            id_key >= end || content_key >= end) return -1;
        ++id_key;
        ++content_key;
        uint32_t id;
        token_string content = {0};
        if (parse_u32(&id_key, &id) != 0 ||
            parse_json_string(&content_key, &content) != 0 ||
            id < QWEN38_TOKENIZER_ADDED_FIRST ||
            id >= QWEN38_TOKENIZER_VOCAB || tokens[id].data != NULL) {
            free(content.data);
            return -1;
        }
        tokens[id] = content;
        ++count;
        cursor = end + 1;
    }
    return 0;
}

static int compare_merges(const void *left, const void *right) {
    const qwen38_token_merge_entry *a = left;
    const qwen38_token_merge_entry *b = right;
    uint64_t ka = ((uint64_t)a->left << 32) | a->right;
    uint64_t kb = ((uint64_t)b->left << 32) | b->right;
    return ka < kb ? -1 : ka > kb ? 1 : 0;
}

static int parse_merges(const char *json, const uint32_t *hash_table,
                        token_string *tokens,
                        qwen38_token_merge_entry *merges) {
    const char *model = strstr(json, "\"model\"");
    const char *cursor =
        model != NULL ? strstr(model, "\"merges\"") : NULL;
    if (cursor == NULL || (cursor = strchr(cursor, '[')) == NULL) return -1;
    ++cursor;
    for (uint32_t rank = 0; rank < QWEN38_TOKENIZER_MERGES; ++rank) {
        skip_space(&cursor);
        if (*cursor == ',') {
            ++cursor;
            skip_space(&cursor);
        }
        if (*cursor++ != '[') return -1;
        token_string left = {0};
        token_string right = {0};
        if (parse_json_string(&cursor, &left) != 0) return -1;
        skip_space(&cursor);
        if (*cursor++ != ',' ||
            parse_json_string(&cursor, &right) != 0) {
            free(left.data);
            return -1;
        }
        skip_space(&cursor);
        if (*cursor++ != ']') {
            free(left.data); free(right.data);
            return -1;
        }
        uint32_t left_id, right_id, result_id;
        size_t combined_length = (size_t)left.length + right.length;
        unsigned char *combined = malloc(combined_length);
        if (combined == NULL ||
            hash_find(hash_table, tokens, left.data, left.length,
                      &left_id) != 0 ||
            hash_find(hash_table, tokens, right.data, right.length,
                      &right_id) != 0) {
            free(left.data); free(right.data); free(combined);
            return -1;
        }
        memcpy(combined, left.data, left.length);
        memcpy(combined + left.length, right.data, right.length);
        if (hash_find(hash_table, tokens, combined, combined_length,
                      &result_id) != 0) {
            free(left.data); free(right.data); free(combined);
            return -1;
        }
        merges[rank] = (qwen38_token_merge_entry){
            .left = left_id, .right = right_id,
            .result = result_id, .rank = rank
        };
        free(left.data); free(right.data); free(combined);
    }
    qsort(merges, QWEN38_TOKENIZER_MERGES, sizeof(*merges),
          compare_merges);
    for (size_t index = 1; index < QWEN38_TOKENIZER_MERGES; ++index) {
        if (merges[index - 1].left == merges[index].left &&
            merges[index - 1].right == merges[index].right) return -1;
    }
    return 0;
}

static void build_byte_unicode(uint32_t mapping[256]) {
    unsigned used[256] = {0};
    for (unsigned value = 33; value <= 126; ++value) used[value] = 1;
    for (unsigned value = 161; value <= 172; ++value) used[value] = 1;
    for (unsigned value = 174; value <= 255; ++value) used[value] = 1;
    uint32_t next = 256;
    for (unsigned value = 0; value < 256; ++value)
        mapping[value] = used[value] ? value : next++;
}

static int decode_codepoint(const unsigned char *data, size_t length,
                            size_t *position, uint32_t *codepoint) {
    if (*position >= length) return -1;
    unsigned char first = data[(*position)++];
    if (first < 0x80) {
        *codepoint = first;
        return 0;
    }
    unsigned count = first < 0xe0 ? 2 : first < 0xf0 ? 3 : 4;
    uint32_t value = first & (0x7f >> count);
    if (*position + count - 1 > length) return -1;
    for (unsigned index = 1; index < count; ++index) {
        unsigned char next = data[(*position)++];
        if ((next & 0xc0) != 0x80) return -1;
        value = (value << 6) | (next & 0x3f);
    }
    *codepoint = value;
    return 0;
}

static int bytelevel_decode(const token_string *input,
                            const uint32_t byte_unicode[256],
                            token_string *output) {
    unsigned char *data = malloc(input->length + 1);
    if (data == NULL) return -1;
    size_t in = 0;
    size_t out = 0;
    while (in < input->length) {
        uint32_t codepoint;
        if (decode_codepoint(input->data, input->length,
                             &in, &codepoint) != 0) {
            free(data);
            return -1;
        }
        unsigned byte = 256;
        for (unsigned candidate = 0; candidate < 256; ++candidate) {
            if (byte_unicode[candidate] == codepoint) {
                byte = candidate;
                break;
            }
        }
        if (byte == 256) {
            free(data);
            return -1;
        }
        data[out++] = (unsigned char)byte;
    }
    output->data = data;
    output->length = (uint32_t)out;
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

static uint64_t align_page(uint64_t value) {
    return (value + 4095) & ~(uint64_t)4095;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s tokenizer.json OUTPUT.q38tok\n", argv[0]);
        return 2;
    }
    int input_file = open(argv[1], O_RDONLY);
    struct stat status;
    if (input_file < 0 || fstat(input_file, &status) != 0 ||
        status.st_size <= 0) {
        if (input_file >= 0) close(input_file);
        fprintf(stderr, "cannot open tokenizer source\n");
        return 3;
    }
    size_t json_length = (size_t)status.st_size;
    char *json = malloc(json_length + 1);
    if (json == NULL) {
        close(input_file);
        return 3;
    }
    size_t done = 0;
    while (done < json_length) {
        ssize_t amount = read(input_file, json + done, json_length - done);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) {
            free(json); close(input_file);
            return 3;
        }
        done += (size_t)amount;
    }
    close(input_file);
    json[json_length] = '\0';
    qwen38_sha256_context sha;
    unsigned char digest[32];
    char actual[65];
    qwen38_sha256_init(&sha);
    qwen38_sha256_update(&sha, json, json_length);
    qwen38_sha256_final(&sha, digest);
    for (size_t index = 0; index < 32; ++index)
        snprintf(actual + index * 2, 3, "%02x", digest[index]);
    actual[64] = '\0';
    if (strcmp(actual, QWEN38_TOKENIZER_SOURCE_SHA256) != 0) {
        fprintf(stderr, "tokenizer SHA-256 mismatch: %s\n", actual);
        free(json);
        return 4;
    }
    token_string *tokens =
        calloc(QWEN38_TOKENIZER_VOCAB, sizeof(*tokens));
    uint32_t *hash_table = calloc(VOCAB_HASH_SIZE, sizeof(*hash_table));
    qwen38_token_merge_entry *merges =
        malloc((size_t)QWEN38_TOKENIZER_MERGES * sizeof(*merges));
    if (tokens == NULL || hash_table == NULL || merges == NULL ||
        parse_vocab(json, tokens) != 0 ||
        parse_added(json, tokens) != 0) {
        fprintf(stderr, "cannot parse pinned tokenizer vocabulary\n");
        free(tokens); free(hash_table); free(merges); free(json);
        return 5;
    }
    for (uint32_t id = 0; id < QWEN38_TOKENIZER_VOCAB; ++id) {
        if (tokens[id].data == NULL ||
            hash_insert(hash_table, tokens, id) != 0) {
            fprintf(stderr, "invalid or duplicate tokenizer token %" PRIu32
                    "\n", id);
            return 5;
        }
    }
    if (parse_merges(json, hash_table, tokens, merges) != 0) {
        fprintf(stderr, "cannot parse pinned tokenizer merges\n");
        return 5;
    }
    uint32_t byte_unicode[256];
    build_byte_unicode(byte_unicode);
    qwen38_tokenizer_image_header header;
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, QWEN38_TOKENIZER_MAGIC, 8);
    header.version = QWEN38_TOKENIZER_VERSION;
    header.header_bytes = QWEN38_TOKENIZER_HEADER_BYTES;
    header.vocab_size = QWEN38_TOKENIZER_VOCAB;
    header.base_vocab_size = QWEN38_TOKENIZER_BASE_VOCAB;
    header.added_first = QWEN38_TOKENIZER_ADDED_FIRST;
    header.added_count = QWEN38_TOKENIZER_ADDED_COUNT;
    header.merge_count = QWEN38_TOKENIZER_MERGES;
    header.token_entry_bytes = sizeof(qwen38_token_directory_entry);
    memcpy(header.source_sha256, actual, 64);
    for (unsigned byte = 0; byte < 256; ++byte) {
        unsigned char encoded[4];
        size_t encoded_length = encode_utf8(byte_unicode[byte], encoded);
        if (hash_find(hash_table, tokens, encoded, encoded_length,
                      &header.byte_token_ids[byte]) != 0) {
            fprintf(stderr, "missing byte-level base token %u\n", byte);
            return 5;
        }
    }
    qwen38_token_directory_entry *directory =
        calloc(QWEN38_TOKENIZER_VOCAB, sizeof(*directory));
    token_string *decoded =
        calloc(QWEN38_TOKENIZER_VOCAB, sizeof(*decoded));
    if (directory == NULL || decoded == NULL) return 5;
    uint64_t blob_bytes = 0;
    for (uint32_t id = 0; id < QWEN38_TOKENIZER_VOCAB; ++id) {
        int result = id < QWEN38_TOKENIZER_BASE_VOCAB ?
            bytelevel_decode(&tokens[id], byte_unicode, &decoded[id]) :
            (decoded[id].data = malloc(tokens[id].length + 1),
             decoded[id].data == NULL ? -1 :
             (memcpy(decoded[id].data, tokens[id].data, tokens[id].length),
              decoded[id].length = tokens[id].length, 0));
        if (result != 0) {
            fprintf(stderr, "cannot byte-decode token %" PRIu32 "\n", id);
            return 5;
        }
        directory[id].offset = blob_bytes;
        directory[id].length = decoded[id].length;
        directory[id].flags =
            id >= QWEN38_TOKENIZER_ADDED_FIRST ? 1u : 0u;
        blob_bytes += decoded[id].length;
    }
    header.token_directory_offset = QWEN38_TOKENIZER_HEADER_BYTES;
    header.token_directory_bytes =
        (uint64_t)QWEN38_TOKENIZER_VOCAB * sizeof(*directory);
    header.token_blob_offset = header.token_directory_offset +
                               header.token_directory_bytes;
    header.token_blob_bytes = blob_bytes;
    header.merges_offset = align_page(header.token_blob_offset + blob_bytes);
    header.merges_bytes =
        (uint64_t)QWEN38_TOKENIZER_MERGES * sizeof(*merges);
    uint64_t file_bytes = header.merges_offset + header.merges_bytes;
    int output = open(argv[2], O_WRONLY | O_CREAT | O_EXCL, 0644);
    if (output < 0) {
        fprintf(stderr, "cannot create %s: %s\n", argv[2], strerror(errno));
        return 6;
    }
    int failed =
        pwrite_exact(output, &header, sizeof(header), 0) != 0 ||
        pwrite_exact(output, directory, (size_t)header.token_directory_bytes,
                     header.token_directory_offset) != 0;
    for (uint32_t id = 0; id < QWEN38_TOKENIZER_VOCAB && !failed; ++id) {
        failed = pwrite_exact(output, decoded[id].data, decoded[id].length,
            header.token_blob_offset + directory[id].offset) != 0;
    }
    failed = failed ||
        pwrite_exact(output, merges, (size_t)header.merges_bytes,
                     header.merges_offset) != 0 ||
        ftruncate(output, (off_t)file_bytes) != 0 || fsync(output) != 0;
    close(output);
    if (failed) {
        fprintf(stderr, "tokenizer packing failed: %s\n", strerror(errno));
        unlink(argv[2]);
        return 6;
    }
    printf("{\"source\":\"%s\",\"source_sha256\":\"%s\","
           "\"output\":\"%s\",\"bytes\":%" PRIu64 ","
           "\"vocab\":%u,\"merges\":%u}\n",
           argv[1], actual, argv[2], file_bytes,
           QWEN38_TOKENIZER_VOCAB, QWEN38_TOKENIZER_MERGES);
    return 0;
}
