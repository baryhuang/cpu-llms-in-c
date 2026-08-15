#ifndef QWEN38_M3_MTP_IMAGE_H
#define QWEN38_M3_MTP_IMAGE_H

#include <stdint.h>

#include "qwen38_m3_image.h"

/* Multi-token-prediction extras image. The MTP transformer layer itself is
 * carried by a standard attention image with layer_index 64; this image
 * carries the pieces around it: the fused fc projection
 * [hidden, 2 x hidden] and the three RMS norm vectors (pre-fc embedding
 * norm, pre-fc hidden norm, final MTP norm). Embedding table and language
 * model head are shared with the main model. */

enum {
    QWEN38_M3_MTP_IMAGE_VERSION = 1,
    QWEN38_M3_MTP_HEADER_BYTES = 4096,
    QWEN38_M3_MTP_LAYER_INDEX = 64
};

typedef struct {
    unsigned char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t hidden_size;
    uint32_t fc_rows;
    uint32_t fc_groups_per_row;
    uint32_t group_size;
    uint32_t constants_f32_count;
    uint32_t reserved0;
    uint64_t fc_quants_offset;
    uint64_t fc_quants_bytes;
    uint64_t fc_metadata_offset;
    uint64_t fc_metadata_bytes;
    uint64_t constants_offset;
    uint64_t constants_bytes;
    uint64_t embedding_norm_constants_index;
    uint64_t hidden_norm_constants_index;
    uint64_t final_norm_constants_index;
    char source_sha256[QWEN38_M3_SOURCE_SHA256_LENGTH + 1];
    unsigned char source_alignment[7];
    unsigned char reserved[
        QWEN38_M3_MTP_HEADER_BYTES - 8 - 8 * 4 - 9 * 8 -
        (QWEN38_M3_SOURCE_SHA256_LENGTH + 1) - 7];
} qwen38_m3_mtp_image_header;

_Static_assert(sizeof(qwen38_m3_mtp_image_header) ==
                   QWEN38_M3_MTP_HEADER_BYTES,
               "Qwen3.8 MTP image header must be one page");

static const unsigned char QWEN38_M3_MTP_IMAGE_MAGIC[8] = {
    'Q', '3', '8', 'M', '3', 'M', 'T', 'P'
};

#endif
