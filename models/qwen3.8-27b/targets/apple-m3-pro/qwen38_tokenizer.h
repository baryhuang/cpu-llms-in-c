#ifndef QWEN38_TOKENIZER_H
#define QWEN38_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

/*
 * The byte-level BPE implementation is shared.  The compiled image contract is
 * selected at build time so a MiniMax-H3 runtime does not carry a JSON parser
 * or a second tokenizer implementation.  Function names remain ABI-stable for
 * the existing Qwen3.8 target.
 */
#if defined(MINIMAX_H3_TOKENIZER_PROFILE)
#define QWEN38_TOKENIZER_SOURCE_SHA256 \
    "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"
#else
#define QWEN38_TOKENIZER_SOURCE_SHA256 \
    "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
#endif

enum {
    QWEN38_TOKENIZER_VERSION = 1,
    QWEN38_TOKENIZER_HEADER_BYTES = 4096,
#if defined(MINIMAX_H3_TOKENIZER_PROFILE)
    QWEN38_TOKENIZER_VOCAB = 151669,
    QWEN38_TOKENIZER_BASE_VOCAB = 151643,
    QWEN38_TOKENIZER_ADDED_FIRST = 151643,
    QWEN38_TOKENIZER_ADDED_COUNT = 26,
    QWEN38_TOKENIZER_MERGES = 151387
#else
    /* 248320 is the padded model output width; only 248077 IDs decode. */
    QWEN38_TOKENIZER_VOCAB = 248077,
    QWEN38_TOKENIZER_BASE_VOCAB = 248044,
    QWEN38_TOKENIZER_ADDED_FIRST = 248044,
    QWEN38_TOKENIZER_ADDED_COUNT = 33,
    QWEN38_TOKENIZER_MERGES = 247587
#endif
};

typedef struct {
    uint64_t offset;
    uint32_t length;
    uint32_t flags;
} qwen38_token_directory_entry;

typedef struct {
    uint32_t left;
    uint32_t right;
    uint32_t result;
    uint32_t rank;
} qwen38_token_merge_entry;

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
    unsigned char reserved[QWEN38_TOKENIZER_HEADER_BYTES - 8 - 8 * 4 -
                           6 * 8 - 65 - 3 - 256 * 4];
} qwen38_tokenizer_image_header;

_Static_assert(sizeof(qwen38_tokenizer_image_header) ==
                   QWEN38_TOKENIZER_HEADER_BYTES,
               "Qwen3.8 tokenizer header must be one page");

#if defined(MINIMAX_H3_TOKENIZER_PROFILE)
static const unsigned char QWEN38_TOKENIZER_MAGIC[8] = {
    'H', '3', 'T', 'O', 'K', '1', '\0', '\0'
};
#else
static const unsigned char QWEN38_TOKENIZER_MAGIC[8] = {
    'Q', '3', '8', 'T', 'O', 'K', '1', '\0'
};
#endif

typedef struct qwen38_tokenizer qwen38_tokenizer;

qwen38_tokenizer *qwen38_tokenizer_open(
    const char *image_path, char *error_message, size_t error_capacity);
void qwen38_tokenizer_close(qwen38_tokenizer *tokenizer);

int qwen38_tokenizer_encode(
    const qwen38_tokenizer *tokenizer,
    const char *utf8,
    uint32_t *token_ids,
    size_t token_capacity,
    size_t *token_count,
    char *error_message,
    size_t error_capacity);

int qwen38_tokenizer_decode(
    const qwen38_tokenizer *tokenizer,
    const uint32_t *token_ids,
    size_t token_count,
    char *utf8,
    size_t utf8_capacity,
    size_t *utf8_length,
    char *error_message,
    size_t error_capacity);

#endif
