#ifndef QWEN36_M3_IMAGE_H
#define QWEN36_M3_IMAGE_H

#include <stdint.h>

#define QWEN36_M3_EXPECTED_SOURCE_SHA256 \
    "2689680915661f040c50c35244d08b336def279e509e1ca11873f8dd1b0e7ce0"
#define QWEN36_M3_EXPECTED_SOURCE_SHA256_2 \
    "983ed9b84ef9469d099e5300d957e6fdecffe21b87096c0f52c9f272a97490c6"
#define QWEN36_M3_EXPECTED_SOURCE_SHA256_3 \
    "ac23bf70b1f239a040921d6f93770d74176fd435dbf44e42317053d06c68d702"

/* Assembled BF16 mtp.* tensor file, built from ranged reads of the official
 * Qwen/Qwen3.6-27B shards (model-00013 and model-00015 of 15). */
#define QWEN36_M3_EXPECTED_MTP_SHA256 \
    "713b0fafacc94c9e541925872de3bcc3507cf5af73abbce525298467b8b4b10f"

/* Qwen3.8-27B: architecturally identical to Qwen3.6-27B (same 2,180
 * tensor names, shapes and shard layout in the mlx-community affine-4bit
 * conversions), so the same image format and runtime serve both. Source:
 * mlx-community/Qwen3.8-27B-4bit revision
 * 3e6447f082e89cc7f0bc6e5441afd38dfce760ff and
 * mlx-community/Qwen3.8-27B-MTP-4bit revision
 * b643c01b6d3b094e325edb6ebd832e16c486c575. Unlike the 3.6 BF16 mtp
 * source, the 3.8 MTP norm vectors arrive already folded to direct
 * multipliers. */
#define QWEN38_M3_EXPECTED_SOURCE_SHA256 \
    "6cc1508e96fb5d0865dfd5753a79f4ec60651bf3e2a82844a7e8ae9c60528c0d"
#define QWEN38_M3_EXPECTED_SOURCE_SHA256_2 \
    "83f2a20ca8058f486a3634a27faf99587f4cd3c156a83dee34fb99e6ac178670"
#define QWEN38_M3_EXPECTED_SOURCE_SHA256_3 \
    "31b8c91ef899f79efaaa69e3d2c096f6e2ebeb2ff20e29222abbd9ebc79e560a"
#define QWEN38_M3_EXPECTED_MTP_SHA256 \
    "76663c101e7e8ea9c0ae17bcb95183cd7f733ce424c912b8b264a7b1c48e4cc6"

enum {
    QWEN36_M3_IMAGE_VERSION = 3,
    QWEN36_M3_MLP_ONLY_IMAGE_VERSION = 2,
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
    char mlp_source_sha256[QWEN36_M3_SOURCE_SHA256_LENGTH + 1];
    unsigned char reserved[QWEN36_M3_IMAGE_HEADER_BYTES - 8 - 12 * 4 -
                           8 * 4 - 12 * 8 - 16 * 8 -
                           (QWEN36_M3_SOURCE_SHA256_LENGTH + 1) - 3 -
                           24 * sizeof(float) -
                           (QWEN36_M3_SOURCE_SHA256_LENGTH + 1) - 4];
} qwen36_m3_image_header;

_Static_assert(sizeof(qwen36_m3_image_header) == QWEN36_M3_IMAGE_HEADER_BYTES,
               "Qwen3.6 M3 image header must be exactly one page");

static const unsigned char QWEN36_M3_IMAGE_MAGIC[8] = {
    'Q', '3', '6', 'M', '3', 'Q', '4', '\0'
};

#endif
