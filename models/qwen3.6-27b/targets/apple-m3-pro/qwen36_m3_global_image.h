#ifndef QWEN36_M3_GLOBAL_IMAGE_H
#define QWEN36_M3_GLOBAL_IMAGE_H

#include <stdint.h>

#include "qwen36_m3_image.h"

enum {
    QWEN36_M3_GLOBAL_IMAGE_VERSION = 1,
    QWEN36_M3_GLOBAL_HEADER_BYTES = 4096,
    QWEN36_VOCAB_SIZE = 248320
};

typedef struct {
    unsigned char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t vocab_size;
    uint32_t hidden_size;
    uint32_t group_size;
    uint32_t constants_f32_count;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t embedding_quants_offset;
    uint64_t embedding_quants_bytes;
    uint64_t embedding_metadata_offset;
    uint64_t embedding_metadata_bytes;
    uint64_t lm_head_quants_offset;
    uint64_t lm_head_quants_bytes;
    uint64_t lm_head_metadata_offset;
    uint64_t lm_head_metadata_bytes;
    uint64_t constants_offset;
    uint64_t constants_bytes;
    char embedding_source_sha256[QWEN36_M3_SOURCE_SHA256_LENGTH + 1];
    char lm_head_source_sha256[QWEN36_M3_SOURCE_SHA256_LENGTH + 1];
    unsigned char source_alignment[6];
    unsigned char reserved[QWEN36_M3_GLOBAL_HEADER_BYTES - 8 - 8 * 4 -
                           10 * 8 -
                           2 * (QWEN36_M3_SOURCE_SHA256_LENGTH + 1) - 6];
} qwen36_m3_global_image_header;

_Static_assert(sizeof(qwen36_m3_global_image_header) ==
                   QWEN36_M3_GLOBAL_HEADER_BYTES,
               "Qwen3.6 global image header must be one page");

static const unsigned char QWEN36_M3_GLOBAL_IMAGE_MAGIC[8] = {
    'Q', '3', '6', 'M', '3', 'G', 'L', 'B'
};

#endif
