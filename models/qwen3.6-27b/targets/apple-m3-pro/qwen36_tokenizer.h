#ifndef QWEN36_TOKENIZER_H
#define QWEN36_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#define QWEN36_TOKENIZER_SOURCE_SHA256 \
    "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4"

enum {
    QWEN36_TOKENIZER_VERSION = 1,
    QWEN36_TOKENIZER_HEADER_BYTES = 4096,
    /* 248320 is the padded model output width; only 248077 IDs decode. */
    QWEN36_TOKENIZER_VOCAB = 248077,
    QWEN36_TOKENIZER_BASE_VOCAB = 248044,
    QWEN36_TOKENIZER_ADDED_FIRST = 248044,
    QWEN36_TOKENIZER_ADDED_COUNT = 33,
    QWEN36_TOKENIZER_MERGES = 247587
};

typedef struct {
    uint64_t offset;
    uint32_t length;
    uint32_t flags;
} qwen36_token_directory_entry;

typedef struct {
    uint32_t left;
    uint32_t right;
    uint32_t result;
    uint32_t rank;
} qwen36_token_merge_entry;

typedef struct {
    unsigned char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t vocab_size;
    uint32_t base_vocab_size;
    uint32_t added_first;
    uint32_t added_count;
    uint32_t merge_count;
    uint32_t token_entry_bytes;
    uint64_t token_directory_offset;
    uint64_t token_directory_bytes;
    uint64_t token_blob_offset;
    uint64_t token_blob_bytes;
    uint64_t merges_offset;
    uint64_t merges_bytes;
    char source_sha256[65];
    unsigned char source_alignment[3];
    uint32_t byte_token_ids[256];
    unsigned char reserved[QWEN36_TOKENIZER_HEADER_BYTES - 8 - 8 * 4 -
                           6 * 8 - 65 - 3 - 256 * 4];
} qwen36_tokenizer_image_header;

_Static_assert(sizeof(qwen36_tokenizer_image_header) ==
                   QWEN36_TOKENIZER_HEADER_BYTES,
               "Qwen3.6 tokenizer header must be one page");

static const unsigned char QWEN36_TOKENIZER_MAGIC[8] = {
    'Q', '3', '6', 'T', 'O', 'K', '1', '\0'
};

typedef struct qwen36_tokenizer qwen36_tokenizer;

qwen36_tokenizer *qwen36_tokenizer_open(
    const char *image_path, char *error_message, size_t error_capacity);
void qwen36_tokenizer_close(qwen36_tokenizer *tokenizer);

int qwen36_tokenizer_encode(
    const qwen36_tokenizer *tokenizer,
    const char *utf8,
    uint32_t *token_ids,
    size_t token_capacity,
    size_t *token_count,
    char *error_message,
    size_t error_capacity);

int qwen36_tokenizer_decode(
    const qwen36_tokenizer *tokenizer,
    const uint32_t *token_ids,
    size_t token_count,
    char *utf8,
    size_t utf8_capacity,
    size_t *utf8_length,
    char *error_message,
    size_t error_capacity);

#endif
