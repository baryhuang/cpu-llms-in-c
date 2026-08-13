#ifndef QWEN36_M3_IMAGE_H
#define QWEN36_M3_IMAGE_H

#include <stdint.h>

#define QWEN36_M3_EXPECTED_SOURCE_SHA256 \
    "2689680915661f040c50c35244d08b336def279e509e1ca11873f8dd1b0e7ce0"

enum {
    QWEN36_M3_IMAGE_VERSION = 2,
    QWEN36_M3_IMAGE_HEADER_BYTES = 4096,
    QWEN36_M3_SOURCE_SHA256_LENGTH = 64
};

typedef struct {
    unsigned char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t hidden_size;
    uint32_t rows;
    uint32_t group_size;
    uint32_t reserved0;
    uint32_t source_reference_count;
    uint32_t reserved1;
    uint32_t down_rows;
    uint32_t down_groups_per_row;
    uint32_t source_mlp_reference_count;
    uint32_t reserved2;
    uint64_t gate_quants_offset;
    uint64_t gate_quants_bytes;
    uint64_t gate_metadata_offset;
    uint64_t gate_metadata_bytes;
    uint64_t up_quants_offset;
    uint64_t up_quants_bytes;
    uint64_t up_metadata_offset;
    uint64_t up_metadata_bytes;
    uint64_t down_quants_offset;
    uint64_t down_quants_bytes;
    uint64_t down_metadata_offset;
    uint64_t down_metadata_bytes;
    char source_sha256[QWEN36_M3_SOURCE_SHA256_LENGTH + 1];
    unsigned char reference_alignment[3];
    float source_reference_first_8[8];
    float source_mlp_reference_first_8[8];
    unsigned char reserved[QWEN36_M3_IMAGE_HEADER_BYTES - 8 - 12 * 4 -
                           12 * 8 - (QWEN36_M3_SOURCE_SHA256_LENGTH + 1) -
                           3 - 16 * sizeof(float)];
} qwen36_m3_image_header;

_Static_assert(sizeof(qwen36_m3_image_header) == QWEN36_M3_IMAGE_HEADER_BYTES,
               "Qwen3.6 M3 image header must be exactly one page");

static const unsigned char QWEN36_M3_IMAGE_MAGIC[8] = {
    'Q', '3', '6', 'M', '3', 'Q', '4', '\0'
};

#endif
