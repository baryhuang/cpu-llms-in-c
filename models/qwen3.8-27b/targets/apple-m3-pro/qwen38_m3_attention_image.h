#ifndef QWEN38_M3_ATTENTION_IMAGE_H
#define QWEN38_M3_ATTENTION_IMAGE_H

#include <stdint.h>

#include "qwen38_m3_image.h"

enum {
    QWEN38_M3_ATTENTION_IMAGE_VERSION = 1,
    QWEN38_M3_ATTENTION_HEADER_BYTES = 4096,
    QWEN38_ATTENTION_Q_ROWS = 12288,
    QWEN38_ATTENTION_K_ROWS = 1024,
    QWEN38_ATTENTION_V_ROWS = 1024,
    QWEN38_ATTENTION_INPUT_ROWS = 14336,
    QWEN38_ATTENTION_HEADS = 24,
    QWEN38_ATTENTION_KV_HEADS = 4,
    QWEN38_ATTENTION_HEAD_SIZE = 256,
    QWEN38_ATTENTION_ROTARY_SIZE = 64
};

typedef struct {
    unsigned char magic[8];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t layer_index;
    uint32_t hidden_size;
    uint32_t intermediate_size;
    uint32_t group_size;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t head_size;
    uint32_t rotary_size;
    uint32_t input_rows;
    uint32_t input_groups_per_row;
    uint32_t output_rows;
    uint32_t output_groups_per_row;
    uint32_t constants_f32_count;
    uint32_t reserved0;
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
    uint64_t constants_offset;
    uint64_t constants_bytes;
    uint64_t input_norm_constants_index;
    uint64_t post_norm_constants_index;
    uint64_t q_norm_constants_index;
    uint64_t k_norm_constants_index;
    uint64_t attention_input_quants_offset;
    uint64_t attention_input_quants_bytes;
    uint64_t attention_input_metadata_offset;
    uint64_t attention_input_metadata_bytes;
    uint64_t attention_output_quants_offset;
    uint64_t attention_output_quants_bytes;
    uint64_t attention_output_metadata_offset;
    uint64_t attention_output_metadata_bytes;
    char source_sha256[QWEN38_M3_SOURCE_SHA256_LENGTH + 1];
    unsigned char source_alignment[7];
    unsigned char reserved[
        QWEN38_M3_ATTENTION_HEADER_BYTES - 8 - 16 * 4 - 26 * 8 -
        (QWEN38_M3_SOURCE_SHA256_LENGTH + 1) - 7];
} qwen38_m3_attention_image_header;

_Static_assert(sizeof(qwen38_m3_attention_image_header) ==
                   QWEN38_M3_ATTENTION_HEADER_BYTES,
               "Qwen3.8 attention image header must be one page");

static const unsigned char QWEN38_M3_ATTENTION_IMAGE_MAGIC[8] = {
    'Q', '3', '8', 'M', '3', 'A', 'T', 'T'
};

#endif
