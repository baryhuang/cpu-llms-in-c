#ifndef QWEN38_M3_IMAGE_H
#define QWEN38_M3_IMAGE_H

#include <stdint.h>

/* Pinned source: mlx-community/Qwen3.8-27B-4bit revision
 * 3e6447f082e89cc7f0bc6e5441afd38dfce760ff and
 * mlx-community/Qwen3.8-27B-MTP-4bit revision
 * b643c01b6d3b094e325edb6ebd832e16c486c575. */
#define QWEN38_M3_EXPECTED_SOURCE_SHA256 \
    "6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d"
#define QWEN38_M3_EXPECTED_SOURCE_SHA256_2 \
    "83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670"
#define QWEN38_M3_EXPECTED_SOURCE_SHA256_3 \
    "31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a"
#define QWEN38_M3_EXPECTED_MTP_SHA256 \
    "76663c101e7e8ea9c0ae17bcb95183cd7f733ce424c912b8b264a7b1c48e4cc6"

enum {
    QWEN38_M3_IMAGE_VERSION = 3,
    QWEN38_M3_MLP_ONLY_IMAGE_VERSION = 2,
    QWEN38_M3_IMAGE_HEADER_BYTES = 4096,
    QWEN38_M3_SOURCE_SHA256_LENGTH = 64
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
    char source_sha256[QWEN38_M3_SOURCE_SHA256_LENGTH + 1];
    unsigned char reference_alignment[3];
    float source_reference_first_8[8];
    float source_mlp_reference_first_8[8];
    uint32_t layer_index;
    uint32_t delta_input_rows;
    uint32_t delta_input_groups_per_row;
    uint32_t delta_output_rows;
    uint32_t delta_output_groups_per_row;
    uint32_t constants_f32_count;
    uint32_t source_layer_reference_count;
    uint32_t reserved3;
    uint64_t constants_offset;
    uint64_t constants_bytes;
    uint64_t input_norm_constants_index;
    uint64_t post_norm_constants_index;
    uint64_t conv_constants_index;
    uint64_t a_log_constants_index;
    uint64_t dt_bias_constants_index;
    uint64_t recurrent_norm_constants_index;
    uint64_t delta_input_quants_offset;
    uint64_t delta_input_quants_bytes;
    uint64_t delta_input_metadata_offset;
    uint64_t delta_input_metadata_bytes;
    uint64_t delta_output_quants_offset;
    uint64_t delta_output_quants_bytes;
    uint64_t delta_output_metadata_offset;
    uint64_t delta_output_metadata_bytes;
    float source_layer_reference_first_8[8];
    char mlp_source_sha256[QWEN38_M3_SOURCE_SHA256_LENGTH + 1];
    unsigned char reserved[QWEN38_M3_IMAGE_HEADER_BYTES - 8 - 12 * 4 -
                           8 * 4 - 12 * 8 - 16 * 8 -
                           (QWEN38_M3_SOURCE_SHA256_LENGTH + 1) - 3 -
                           24 * sizeof(float) -
                           (QWEN38_M3_SOURCE_SHA256_LENGTH + 1) - 4];
} qwen38_m3_image_header;

_Static_assert(sizeof(qwen38_m3_image_header) == QWEN38_M3_IMAGE_HEADER_BYTES,
               "Qwen3.8 M3 image header must be exactly one page");

static const unsigned char QWEN38_M3_IMAGE_MAGIC[8] = {
    'Q', '3', '8', 'M', '3', 'Q', '4', '\0'
};

#endif
