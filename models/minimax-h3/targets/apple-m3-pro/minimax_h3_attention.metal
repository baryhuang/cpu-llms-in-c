#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint kH3HeadCount = 56;
constant uint kH3HeadDim = 128;
constant uint kH3HeadHalf4 = 32;
constant uint kH3RopeFrequencySlots = 48;
constant float kH3AttentionScale = 0.08838834764831845f;
constant uint kH3SummaryBit = 0x80000000u;
constant float kH3AliasFilter[12] = {
    0.00202896645f, 0.00938946394f, -0.0255434641f, -0.0576573755f,
    0.128572609f, 0.4432098f, 0.4432098f, 0.128572609f,
    -0.0576573755f, -0.0255434641f, 0.00938946394f, 0.00202896645f,
};

struct H3TreeNodeGPU {
    uint parent;
    uint first_child;
    uint child_count;
    uint kind;
    uint first_frame;
    uint frame_count;
    uint patch_y;
    uint patch_x;
    uint patch_h;
    uint patch_w;
    uint token_count;
    uint physical_start;
};

struct H3TreeParameters {
    uint sequence_rows;
    uint exact_rows;
    uint video_start;
    uint rows_per_video_frame;
    uint patch_columns;
    uint tile_columns;
    uint leaves_per_frame;
    uint leaf_count;
    uint aggregate_start;
    uint aggregate_count;
};

struct H3QueryBlockGPU {
    uint first_row;
    uint row_count;
    uint route_index;
};

struct H3Q4Meta {
    half scale;
    half bias;
};

struct H3GemmParameters {
    uint rows;
    uint groups_per_row;
    uint batch;
};

struct H3DenseParameters {
    uint rows;
    uint columns;
    uint batch;
    uint input_stride;
};

struct H3NormParameters {
    uint rows;
    uint columns;
    uint modulation_stride;
    uint shift_offset;
    uint scale_offset;
    float epsilon;
};

struct H3QwenAttentionParameters {
    uint token_count;
    uint query_heads;
    uint key_value_heads;
    uint head_dimension;
    float scale;
};

struct H3AudioConvParameters {
    uint batch;
    uint input_length;
    uint output_length;
    uint input_channels;
    uint output_channels;
    uint kernel_size;
    uint stride;
    uint padding;
    uint dilation;
};

inline float h3_bfloat_to_float(ushort bits) {
    return as_type<float>(uint(bits) << 16u);
}

kernel void minimax_h3_initialize_half(
    device half *output [[buffer(0)]],
    constant uint &element_count [[buffer(1)]],
    constant uint &seed [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= element_count) return;
    int value = int((index * 13u + seed) % 17u) - 8;
    output[index] = half(float(value) * 0.0625f);
}

kernel void minimax_h3_initialize_half_irregular(
    device half *output [[buffer(0)]],
    constant uint &element_count [[buffer(1)]],
    constant uint &seed [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= element_count) return;
    uint bits = index * 747796405u + seed * 2891336453u + 277803737u;
    bits = ((bits >> ((bits >> 28u) + 4u)) ^ bits) * 277803737u;
    bits = (bits >> 22u) ^ bits;
    int centered = int(bits % 2001u) - 1000;
    output[index] = half(float(centered) * 0.000731f);
}

kernel void minimax_h3_initialize_bf16(
    device bfloat *output [[buffer(0)]],
    constant uint &element_count [[buffer(1)]],
    constant uint &seed [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= element_count) return;
    int value = int((index * 13u + seed) % 17u) - 8;
    output[index] = bfloat(float(value) * 0.0625f);
}

kernel void minimax_h3_initialize_bf16_irregular(
    device bfloat *output [[buffer(0)]],
    constant uint &element_count [[buffer(1)]],
    constant uint &seed [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= element_count) return;
    uint bits = index * 747796405u + seed * 2891336453u + 277803737u;
    bits = ((bits >> ((bits >> 28u) + 4u)) ^ bits) * 277803737u;
    bits = (bits >> 22u) ^ bits;
    int centered = int(bits % 2001u) - 1000;
    output[index] = bfloat(float(centered) * 0.000731f);
}

/* The H3 packed layout and MM-RoPE frequencies are fixed for an execution
 * artifact.  Build the 48 unique (cos, sin) pairs per row once instead of
 * evaluating pow/cos/sin independently for Q and K in every head and block. */
kernel void minimax_h3_build_rope_f32(
    device const float *positions [[buffer(0)]],
    device float2 *rotary [[buffer(1)]],
    constant uint &row_count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    uint count = row_count * kH3RopeFrequencySlots;
    if (index >= count) return;
    uint row = index / kH3RopeFrequencySlots;
    uint frequency_slot = index - row * kH3RopeFrequencySlots;
    uint axis = frequency_slot / 16u;
    uint frequency = frequency_slot - axis * 16u;
    float inverse_frequency = pow(10000.0f, -float(frequency) / 16.0f);
    float angle = positions[row * 3u + axis] * inverse_frequency;
    rotary[index] = float2(cos(angle), sin(angle));
}

kernel void minimax_h3_initialize_q4(
    device uchar *output [[buffer(0)]],
    constant uint &byte_count [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= byte_count) return;
    uint block = index >> 5u;
    uint low = (index * 5u + block * 3u + 3u) & 15u;
    uint high = (index * 7u + block * 11u + 11u) & 15u;
    output[index] = uchar(low | (high << 4u));
}

kernel void minimax_h3_initialize_q4_meta(
    device H3Q4Meta *output [[buffer(0)]],
    constant uint &meta_count [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= meta_count) return;
    uint phase = index % 7u;
    output[index].scale = half(0.0175f + float(phase) * 0.0017f);
    output[index].bias = half(-0.1875f + float(phase) * 0.0078125f);
}

kernel void minimax_h3_q4_gemm_f16_mma(
    device const half *input [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const H3Q4Meta *metadata [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant H3GemmParameters &parameters [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half weight_tile[64 * 32];
    threadgroup half spill[32 * 32];
    uint output_row0 = group.x * 32u;
    uint batch0 = group.y * 32u;
    uint columns = parameters.groups_per_row * 64u;
    uint batch_row0 = batch0 + simdgroup_index * 8u;
    uint spill_row0 = simdgroup_index * 8u;
    uint output_row = tid & 31u;
    uint k_base = (tid >> 5u) * 16u;
    simdgroup_half8x8 accumulators[4];
    float fp32_accumulators[8];

    for (uint output_fragment = 0u; output_fragment < 4u;
         ++output_fragment) {
        accumulators[output_fragment] =
            make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
    }
    for (uint value = 0u; value < 8u; ++value)
        fp32_accumulators[value] = 0.0f;

    for (uint input_group = 0u; input_group < parameters.groups_per_row;
         ++input_group) {
        uint matrix_row = output_row0 + output_row;
        if (matrix_row < parameters.rows) {
            uint block = matrix_row * parameters.groups_per_row + input_group;
            H3Q4Meta meta = metadata[block];
            device const uint *words = reinterpret_cast<device const uint *>(
                quants + block * 32u + (k_base >> 1u));
            for (uint word = 0u; word < 2u; ++word) {
                uint bits = words[word];
                for (uint nibble = 0u; nibble < 8u; ++nibble) {
                    uint k = k_base + word * 8u + nibble;
                    weight_tile[k * 32u + output_row] =
                        meta.scale * half((bits >> (4u * nibble)) & 15u) +
                        meta.bias;
                }
            }
        } else {
            for (uint k = k_base; k < k_base + 16u; ++k)
                weight_tile[k * 32u + output_row] = 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0u; k < 64u; k += 8u) {
            simdgroup_half8x8 activation;
            simdgroup_load(activation,
                           input + batch_row0 * columns +
                               input_group * 64u + k,
                           columns);
            for (uint output_fragment = 0u; output_fragment < 4u;
                 ++output_fragment) {
                simdgroup_half8x8 weight;
                simdgroup_load(weight,
                               weight_tile + k * 32u + output_fragment * 8u,
                               32u);
                simdgroup_multiply_accumulate(
                    accumulators[output_fragment], activation, weight,
                    accumulators[output_fragment]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint output_fragment = 0u; output_fragment < 4u;
             ++output_fragment) {
            simdgroup_store(accumulators[output_fragment],
                            spill + spill_row0 * 32u + output_fragment * 8u,
                            32u);
            accumulators[output_fragment] =
                make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint value = 0u; value < 8u; ++value)
            fp32_accumulators[value] += float(spill[tid * 8u + value]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint value = 0u; value < 8u; ++value) {
        uint linear = tid * 8u + value;
        uint batch_row = batch0 + (linear >> 5u);
        uint matrix_row = output_row0 + (linear & 31u);
        if (batch_row < parameters.batch && matrix_row < parameters.rows) {
            output[batch_row * parameters.rows + matrix_row] =
                half(fp32_accumulators[value]);
        }
    }
}

/* The H3 block projections are much wider than the 32-row tile above.
 * Covering 64 output rows per threadgroup reuses each activation matrix
 * fragment across eight output fragments and halves dispatch/barrier work.
 * The per-64-column partials are still spilled to float, so this changes the
 * tile shape without extending the half-accumulation interval. */
kernel void minimax_h3_q4_gemm_f16_mma_wide(
    device const half *input [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const H3Q4Meta *metadata [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant H3GemmParameters &parameters [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half weight_tile[64 * 64];
    threadgroup half spill[32 * 64];
    uint output_row0 = group.x * 64u;
    uint batch0 = group.y * 32u;
    uint columns = parameters.groups_per_row * 64u;
    uint batch_row0 = batch0 + simdgroup_index * 8u;
    uint spill_row0 = simdgroup_index * 8u;
    uint output_row = tid & 63u;
    uint k_base = (tid >> 6u) * 32u;
    simdgroup_half8x8 accumulators[8];
    float fp32_accumulators[16];

    for (uint output_fragment = 0u; output_fragment < 8u;
         ++output_fragment) {
        accumulators[output_fragment] =
            make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
    }
    for (uint value = 0u; value < 16u; ++value)
        fp32_accumulators[value] = 0.0f;

    for (uint input_group = 0u; input_group < parameters.groups_per_row;
         ++input_group) {
        uint matrix_row = output_row0 + output_row;
        if (matrix_row < parameters.rows) {
            uint block = matrix_row * parameters.groups_per_row + input_group;
            H3Q4Meta meta = metadata[block];
            device const uint *words = reinterpret_cast<device const uint *>(
                quants + block * 32u + (k_base >> 1u));
            for (uint word = 0u; word < 4u; ++word) {
                uint bits = words[word];
                for (uint nibble = 0u; nibble < 8u; ++nibble) {
                    uint k = k_base + word * 8u + nibble;
                    weight_tile[k * 64u + output_row] =
                        meta.scale * half((bits >> (4u * nibble)) & 15u) +
                        meta.bias;
                }
            }
        } else {
            for (uint k = k_base; k < k_base + 32u; ++k)
                weight_tile[k * 64u + output_row] = 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0u; k < 64u; k += 8u) {
            simdgroup_half8x8 activation;
            simdgroup_load(activation,
                           input + batch_row0 * columns +
                               input_group * 64u + k,
                           columns);
            for (uint output_fragment = 0u; output_fragment < 8u;
                 ++output_fragment) {
                simdgroup_half8x8 weight;
                simdgroup_load(weight,
                               weight_tile + k * 64u + output_fragment * 8u,
                               64u);
                simdgroup_multiply_accumulate(
                    accumulators[output_fragment], activation, weight,
                    accumulators[output_fragment]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (true) {
            for (uint output_fragment = 0u; output_fragment < 8u;
                 ++output_fragment) {
                simdgroup_store(
                    accumulators[output_fragment],
                    spill + spill_row0 * 64u + output_fragment * 8u, 64u);
                accumulators[output_fragment] =
                    make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint value = 0u; value < 16u; ++value)
                fp32_accumulators[value] += float(spill[tid * 16u + value]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint value = 0u; value < 16u; ++value) {
        uint linear = tid * 16u + value;
        uint batch_row = batch0 + (linear >> 6u);
        uint matrix_row = output_row0 + (linear & 63u);
        if (batch_row < parameters.batch && matrix_row < parameters.rows) {
            output[batch_row * parameters.rows + matrix_row] =
                half(fp32_accumulators[value]);
        }
    }
}

kernel void minimax_h3_copy_gemm_first_output(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index < 8u) output[index] = input[index];
}

/* Runtime projection path for converted MLX affine storage.  The source keeps
 * scales and biases in two BF16 slabs; decoding them here avoids a second
 * resident metadata image. */
kernel void minimax_h3_q4_gemm_bf16_meta(
    device const half *input [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const ushort *scales [[buffer(2)]],
    device const ushort *biases [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant H3GemmParameters &parameters [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half weight_tile[64 * 64];
    threadgroup half spill[32 * 64];
    uint output_row0 = group.x * 64u;
    uint batch0 = group.y * 32u;
    uint columns = parameters.groups_per_row * 64u;
    uint batch_row0 = batch0 + simdgroup_index * 8u;
    uint spill_row0 = simdgroup_index * 8u;
    uint output_row = tid & 63u;
    uint k_base = (tid >> 6u) * 32u;
    simdgroup_half8x8 accumulators[8];
    float fp32_accumulators[16];
    for (uint output_fragment = 0u; output_fragment < 8u; ++output_fragment)
        accumulators[output_fragment] =
            make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
    for (uint value = 0u; value < 16u; ++value)
        fp32_accumulators[value] = 0.0f;

    for (uint input_group = 0u; input_group < parameters.groups_per_row;
         ++input_group) {
        uint matrix_row = output_row0 + output_row;
        if (matrix_row < parameters.rows) {
            uint block = matrix_row * parameters.groups_per_row + input_group;
            float scale = h3_bfloat_to_float(scales[block]);
            float bias = h3_bfloat_to_float(biases[block]);
            device const uint *words = reinterpret_cast<device const uint *>(
                quants + block * 32u + (k_base >> 1u));
            for (uint word = 0u; word < 4u; ++word) {
                uint bits = words[word];
                for (uint nibble = 0u; nibble < 8u; ++nibble) {
                    uint k = k_base + word * 8u + nibble;
                    weight_tile[k * 64u + output_row] =
                        half(scale * float((bits >> (4u * nibble)) & 15u) +
                             bias);
                }
            }
        } else {
            for (uint k = k_base; k < k_base + 32u; ++k)
                weight_tile[k * 64u + output_row] = 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0u; k < 64u; k += 8u) {
            simdgroup_half8x8 activation;
            simdgroup_load(activation,
                           input + batch_row0 * columns +
                               input_group * 64u + k,
                           columns);
            for (uint output_fragment = 0u; output_fragment < 8u;
                 ++output_fragment) {
                simdgroup_half8x8 weight;
                simdgroup_load(weight,
                               weight_tile + k * 64u + output_fragment * 8u,
                               64u);
                simdgroup_multiply_accumulate(accumulators[output_fragment],
                                               activation, weight,
                                               accumulators[output_fragment]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint output_fragment = 0u; output_fragment < 8u;
             ++output_fragment) {
            simdgroup_store(accumulators[output_fragment],
                            spill + spill_row0 * 64u +
                                output_fragment * 8u,
                            64u);
            accumulators[output_fragment] =
                make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint value = 0u; value < 16u; ++value)
            fp32_accumulators[value] += float(spill[tid * 16u + value]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint value = 0u; value < 16u; ++value) {
        uint linear = tid * 16u + value;
        uint batch_row = batch0 + (linear >> 6u);
        uint matrix_row = output_row0 + (linear & 63u);
        if (batch_row < parameters.batch && matrix_row < parameters.rows)
            output[batch_row * parameters.rows + matrix_row] =
                half(fp32_accumulators[value]);
    }
}

/* H3's released runtime keeps the refiner and denoiser activation stream in
 * BF16.  FP16 is not a valid substitute here: late residual blocks can exceed
 * 65504 even though their BF16 values remain finite.  Accumulate the complete
 * dot product in FP32 and only round once when the BF16 output is stored. */
kernel void minimax_h3_q4_gemm_bf16_activation(
    device const bfloat *input [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const ushort *scales [[buffer(2)]],
    device const ushort *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant H3GemmParameters &parameters [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup bfloat weight_tile[64 * 64];
    threadgroup float spill[32 * 64];
    uint output_row0 = group.x * 64u;
    uint batch0 = group.y * 32u;
    uint columns = parameters.groups_per_row * 64u;
    uint batch_row0 = batch0 + simdgroup_index * 8u;
    uint spill_row0 = simdgroup_index * 8u;
    uint output_row = tid & 63u;
    uint k_base = (tid >> 6u) * 32u;
    simdgroup_float8x8 accumulators[8];
    for (uint output_fragment = 0u; output_fragment < 8u;
         ++output_fragment)
        accumulators[output_fragment] =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint input_group = 0u; input_group < parameters.groups_per_row;
         ++input_group) {
        uint matrix_row = output_row0 + output_row;
        if (matrix_row < parameters.rows) {
            uint block = matrix_row * parameters.groups_per_row + input_group;
            float scale = h3_bfloat_to_float(scales[block]);
            float bias = h3_bfloat_to_float(biases[block]);
            device const uint *words = reinterpret_cast<device const uint *>(
                quants + block * 32u + (k_base >> 1u));
            for (uint word = 0u; word < 4u; ++word) {
                uint bits = words[word];
                for (uint nibble = 0u; nibble < 8u; ++nibble) {
                    uint k = k_base + word * 8u + nibble;
                    weight_tile[k * 64u + output_row] = bfloat(
                        scale * float((bits >> (4u * nibble)) & 15u) + bias);
                }
            }
        } else {
            for (uint k = k_base; k < k_base + 32u; ++k)
                weight_tile[k * 64u + output_row] = bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0u; k < 64u; k += 8u) {
            simdgroup_bfloat8x8 activation;
            simdgroup_load(activation,
                           input + batch_row0 * columns +
                               input_group * 64u + k,
                           columns);
            for (uint output_fragment = 0u; output_fragment < 8u;
                 ++output_fragment) {
                simdgroup_bfloat8x8 weight;
                simdgroup_load(weight,
                               weight_tile + k * 64u + output_fragment * 8u,
                               64u);
                simdgroup_multiply_accumulate(accumulators[output_fragment],
                                               activation, weight,
                                               accumulators[output_fragment]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint output_fragment = 0u; output_fragment < 8u;
         ++output_fragment)
        simdgroup_store(accumulators[output_fragment],
                        spill + spill_row0 * 64u + output_fragment * 8u,
                        64u);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint value = 0u; value < 16u; ++value) {
        uint linear = tid * 16u + value;
        uint batch_row = batch0 + (linear >> 6u);
        uint matrix_row = output_row0 + (linear & 63u);
        if (batch_row < parameters.batch && matrix_row < parameters.rows)
            output[batch_row * parameters.rows + matrix_row] =
                bfloat(spill[linear]);
    }
}

kernel void minimax_h3_q8_gemm_bf16_meta(
    device const half *input [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const ushort *scales [[buffer(2)]],
    device const ushort *biases [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant H3GemmParameters &parameters [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half weight_tile[64 * 64];
    threadgroup half spill[32 * 64];
    uint output_row0 = group.x * 64u;
    uint batch0 = group.y * 32u;
    uint columns = parameters.groups_per_row * 64u;
    uint batch_row0 = batch0 + simdgroup_index * 8u;
    uint spill_row0 = simdgroup_index * 8u;
    uint output_row = tid & 63u;
    uint k_base = (tid >> 6u) * 32u;
    simdgroup_half8x8 accumulators[8];
    float fp32_accumulators[16];
    for (uint output_fragment = 0u; output_fragment < 8u; ++output_fragment)
        accumulators[output_fragment] =
            make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
    for (uint value = 0u; value < 16u; ++value)
        fp32_accumulators[value] = 0.0f;
    for (uint input_group = 0u; input_group < parameters.groups_per_row;
         ++input_group) {
        uint matrix_row = output_row0 + output_row;
        if (matrix_row < parameters.rows) {
            uint block = matrix_row * parameters.groups_per_row + input_group;
            float scale = h3_bfloat_to_float(scales[block]);
            float bias = h3_bfloat_to_float(biases[block]);
            device const uchar *codes = quants + block * 64u + k_base;
            for (uint k = 0u; k < 32u; ++k)
                weight_tile[(k_base + k) * 64u + output_row] =
                    half(scale * float(codes[k]) + bias);
        } else {
            for (uint k = k_base; k < k_base + 32u; ++k)
                weight_tile[k * 64u + output_row] = 0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0u; k < 64u; k += 8u) {
            simdgroup_half8x8 activation;
            simdgroup_load(activation,
                           input + batch_row0 * columns +
                               input_group * 64u + k,
                           columns);
            for (uint output_fragment = 0u; output_fragment < 8u;
                 ++output_fragment) {
                simdgroup_half8x8 weight;
                simdgroup_load(weight,
                               weight_tile + k * 64u + output_fragment * 8u,
                               64u);
                simdgroup_multiply_accumulate(accumulators[output_fragment],
                                               activation, weight,
                                               accumulators[output_fragment]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint output_fragment = 0u; output_fragment < 8u;
             ++output_fragment) {
            simdgroup_store(accumulators[output_fragment],
                            spill + spill_row0 * 64u +
                                output_fragment * 8u,
                            64u);
            accumulators[output_fragment] =
                make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint value = 0u; value < 16u; ++value)
            fp32_accumulators[value] += float(spill[tid * 16u + value]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint value = 0u; value < 16u; ++value) {
        uint linear = tid * 16u + value;
        uint batch_row = batch0 + (linear >> 6u);
        uint matrix_row = output_row0 + (linear & 63u);
        if (batch_row < parameters.batch && matrix_row < parameters.rows)
            output[batch_row * parameters.rows + matrix_row] =
                half(fp32_accumulators[value]);
    }
}

kernel void minimax_h3_dense_bf16(
    device const half *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device const ushort *bias [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant H3DenseParameters &parameters [[buffer(4)]],
    constant uint &has_bias [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint flat = group.x * simdgroups_per_group + simdgroup_index;
    uint count = parameters.batch * parameters.rows;
    if (flat >= count) return;
    uint batch = flat / parameters.rows;
    uint row = flat - batch * parameters.rows;
    float partial = 0.0f;
    for (uint column = lane; column < parameters.columns; column += 32u)
        partial += float(input[batch * parameters.input_stride + column]) *
                   h3_bfloat_to_float(weights[row * parameters.columns + column]);
    float value = simd_sum(partial);
    if (lane == 0u) {
        if (has_bias != 0u) value += h3_bfloat_to_float(bias[row]);
        output[batch * parameters.rows + row] = half(value);
    }
}

kernel void minimax_h3_dense_bf16_f16_to_bf16(
    device const half *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device const ushort *bias [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant H3DenseParameters &parameters [[buffer(4)]],
    constant uint &has_bias [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint flat = group.x * simdgroups_per_group + simdgroup_index;
    uint count = parameters.batch * parameters.rows;
    if (flat >= count) return;
    uint batch = flat / parameters.rows;
    uint row = flat - batch * parameters.rows;
    float partial = 0.0f;
    for (uint column = lane; column < parameters.columns; column += 32u)
        partial += float(input[batch * parameters.input_stride + column]) *
                   h3_bfloat_to_float(
                       weights[row * parameters.columns + column]);
    float value = simd_sum(partial);
    if (lane == 0u) {
        if (has_bias != 0u) value += h3_bfloat_to_float(bias[row]);
        output[batch * parameters.rows + row] = bfloat(value);
    }
}

kernel void minimax_h3_dense_bf16_activation(
    device const bfloat *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device const ushort *bias [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant H3DenseParameters &parameters [[buffer(4)]],
    constant uint &has_bias [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint flat = group.x * simdgroups_per_group + simdgroup_index;
    uint count = parameters.batch * parameters.rows;
    if (flat >= count) return;
    uint batch = flat / parameters.rows;
    uint row = flat - batch * parameters.rows;
    float partial = 0.0f;
    for (uint column = lane; column < parameters.columns; column += 32u)
        partial += float(input[batch * parameters.input_stride + column]) *
                   h3_bfloat_to_float(
                       weights[row * parameters.columns + column]);
    float value = simd_sum(partial);
    if (lane == 0u) {
        if (has_bias != 0u) value += h3_bfloat_to_float(bias[row]);
        output[batch * parameters.rows + row] = bfloat(value);
    }
}

/* Add a BF16 low-rank projection to an existing BF16 base projection.  The
 * adapter rows retain the wrapped Linear module's output order. */
kernel void minimax_h3_dense_bf16_activation_add(
    device const bfloat *input [[buffer(0)]],
    device const ushort *weights [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant H3DenseParameters &parameters [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint flat = group.x * simdgroups_per_group + simdgroup_index;
    uint count = parameters.batch * parameters.rows;
    if (flat >= count) return;
    uint batch = flat / parameters.rows;
    uint row = flat - batch * parameters.rows;
    float partial = 0.0f;
    for (uint column = lane; column < parameters.columns; column += 32u)
        partial += float(input[batch * parameters.input_stride + column]) *
                   h3_bfloat_to_float(
                       weights[row * parameters.columns + column]);
    float value = simd_sum(partial);
    if (lane == 0u) {
        uint destination = batch * parameters.rows + row;
        output[destination] = bfloat(float(output[destination]) + value);
    }
}

kernel void minimax_h3_dense_f32(
    device const half *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant H3DenseParameters &parameters [[buffer(4)]],
    constant uint &has_bias [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint flat = group.x * simdgroups_per_group + simdgroup_index;
    uint count = parameters.batch * parameters.rows;
    if (flat >= count) return;
    uint batch = flat / parameters.rows;
    uint row = flat - batch * parameters.rows;
    float partial = 0.0f;
    for (uint column = lane; column < parameters.columns; column += 32u)
        partial += float(input[batch * parameters.input_stride + column]) *
                   weights[row * parameters.columns + column];
    float value = simd_sum(partial);
    if (lane == 0u) {
        if (has_bias != 0u) value += bias[row];
        output[batch * parameters.rows + row] = half(value);
    }
}

kernel void minimax_h3_dense_f32_f16_to_bf16(
    device const half *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant H3DenseParameters &parameters [[buffer(4)]],
    constant uint &has_bias [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint flat = group.x * simdgroups_per_group + simdgroup_index;
    uint count = parameters.batch * parameters.rows;
    if (flat >= count) return;
    uint batch = flat / parameters.rows;
    uint row = flat - batch * parameters.rows;
    float partial = 0.0f;
    for (uint column = lane; column < parameters.columns; column += 32u)
        partial += float(input[batch * parameters.input_stride + column]) *
                   weights[row * parameters.columns + column];
    float value = simd_sum(partial);
    if (lane == 0u) {
        if (has_bias != 0u) value += bias[row];
        output[batch * parameters.rows + row] = bfloat(value);
    }
}

kernel void minimax_h3_dense_f32_f32_to_bf16(
    device const float *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant H3DenseParameters &parameters [[buffer(4)]],
    constant uint &has_bias [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint flat = group.x * simdgroups_per_group + simdgroup_index;
    uint count = parameters.batch * parameters.rows;
    if (flat >= count) return;
    uint batch = flat / parameters.rows;
    uint row = flat - batch * parameters.rows;
    float partial = 0.0f;
    for (uint column = lane; column < parameters.columns; column += 32u)
        partial += input[batch * parameters.input_stride + column] *
                   weights[row * parameters.columns + column];
    float value = simd_sum(partial);
    if (lane == 0u) {
        if (has_bias != 0u) value += bias[row];
        output[batch * parameters.rows + row] = bfloat(value);
    }
}

kernel void minimax_h3_dense_f32_bf16_to_f32(
    device const bfloat *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant H3DenseParameters &parameters [[buffer(4)]],
    constant uint &has_bias [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint flat = group.x * simdgroups_per_group + simdgroup_index;
    uint count = parameters.batch * parameters.rows;
    if (flat >= count) return;
    uint batch = flat / parameters.rows;
    uint row = flat - batch * parameters.rows;
    float partial = 0.0f;
    for (uint column = lane; column < parameters.columns; column += 32u)
        partial += float(input[batch * parameters.input_stride + column]) *
                   weights[row * parameters.columns + column];
    float value = simd_sum(partial);
    if (lane == 0u) {
        if (has_bias != 0u) value += bias[row];
        output[batch * parameters.rows + row] = value;
    }
}

kernel void minimax_h3_dense_f32_bf16_activation(
    device const bfloat *input [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant H3DenseParameters &parameters [[buffer(4)]],
    constant uint &has_bias [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint flat = group.x * simdgroups_per_group + simdgroup_index;
    uint count = parameters.batch * parameters.rows;
    if (flat >= count) return;
    uint batch = flat / parameters.rows;
    uint row = flat - batch * parameters.rows;
    float partial = 0.0f;
    for (uint column = lane; column < parameters.columns; column += 32u)
        partial += float(input[batch * parameters.input_stride + column]) *
                   weights[row * parameters.columns + column];
    float value = simd_sum(partial);
    if (lane == 0u) {
        if (has_bias != 0u) value += bias[row];
        output[batch * parameters.rows + row] = bfloat(value);
    }
}

kernel void minimax_h3_dense_f16(
    device const half *input [[buffer(0)]],
    device const half *weights [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant H3DenseParameters &parameters [[buffer(4)]],
    constant uint &has_bias [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint flat = group.x * simdgroups_per_group + simdgroup_index;
    uint count = parameters.batch * parameters.rows;
    if (flat >= count) return;
    uint batch = flat / parameters.rows;
    uint row = flat - batch * parameters.rows;
    float partial = 0.0f;
    for (uint column = lane; column < parameters.columns; column += 32u)
        partial += float(input[batch * parameters.input_stride + column]) *
                   float(weights[row * parameters.columns + column]);
    float result = simd_sum(partial);
    if (lane == 0u) {
        if (has_bias != 0u) result += float(bias[row]);
        output[batch * parameters.rows + row] = half(result);
    }
}

/* ViT3D decoder projection path.  One threadgroup evaluates a 32x64 output
 * tile, reusing every 8x8 activation fragment across eight output fragments.
 * The source weights are [output, input], so the matrix load transposes each
 * 8x8 weight fragment in place.  Accumulation remains FP32 and the public
 * activation boundary remains FP16. */
kernel void minimax_h3_dense_f16_mma(
    device const half *input [[buffer(0)]],
    device const half *weights [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant H3DenseParameters &parameters [[buffer(4)]],
    constant uint &has_bias [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float spill[32 * 64];
    uint output_row0 = group.x * 64u;
    uint batch0 = group.y * 32u;
    uint batch_row0 = batch0 + simdgroup_index * 8u;
    uint spill_row0 = simdgroup_index * 8u;
    simdgroup_float8x8 accumulators[8];
    for (uint fragment_index = 0u; fragment_index < 8u; ++fragment_index)
        accumulators[fragment_index] =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint column = 0u; column < parameters.columns; column += 8u) {
        simdgroup_half8x8 activation;
        simdgroup_load(activation,
                       input + batch_row0 * parameters.input_stride + column,
                       parameters.input_stride);
        for (uint fragment_index = 0u; fragment_index < 8u;
             ++fragment_index) {
            simdgroup_half8x8 weight;
            simdgroup_load(
                weight,
                weights + (output_row0 + fragment_index * 8u) *
                              parameters.columns + column,
                parameters.columns, 0u, true);
            simdgroup_multiply_accumulate(accumulators[fragment_index],
                                           activation, weight,
                                           accumulators[fragment_index]);
        }
    }
    for (uint fragment_index = 0u; fragment_index < 8u; ++fragment_index)
        simdgroup_store(accumulators[fragment_index],
                        spill + spill_row0 * 64u + fragment_index * 8u, 64u);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint value = 0u; value < 16u; ++value) {
        uint linear = tid * 16u + value;
        uint batch_row = batch0 + (linear >> 6u);
        uint matrix_row = output_row0 + (linear & 63u);
        if (batch_row < parameters.batch && matrix_row < parameters.rows) {
            float result = spill[linear];
            if (has_bias != 0u) result += float(bias[matrix_row]);
            output[batch_row * parameters.rows + matrix_row] = half(result);
        }
    }
}

/* The four simdgroups process different activation rows but the same 64
 * output channels.  Stage 64x32 weights once per threadgroup so the four
 * matrix streams do not issue duplicate device-memory reads. */
kernel void minimax_h3_dense_f16_mma_weight_tiled(
    device const half *input [[buffer(0)]],
    device const half *weights [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant H3DenseParameters &parameters [[buffer(4)]],
    constant uint &has_bias [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half weight_tile[64 * 32];
    threadgroup float spill[32 * 64];
    uint output_row0 = group.x * 64u;
    uint batch0 = group.y * 32u;
    uint batch_row0 = batch0 + simdgroup_index * 8u;
    uint spill_row0 = simdgroup_index * 8u;
    simdgroup_float8x8 accumulators[8];
    for (uint fragment_index = 0u; fragment_index < 8u; ++fragment_index)
        accumulators[fragment_index] =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint column0 = 0u; column0 < parameters.columns; column0 += 32u) {
        for (uint index = tid; index < 64u * 32u; index += 128u) {
            uint output_offset = index >> 5u;
            uint column_offset = index & 31u;
            weight_tile[index] =
                weights[(output_row0 + output_offset) * parameters.columns +
                        column0 + column_offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint column_offset = 0u; column_offset < 32u;
             column_offset += 8u) {
            simdgroup_half8x8 activation;
            simdgroup_load(
                activation,
                input + batch_row0 * parameters.input_stride + column0 +
                    column_offset,
                parameters.input_stride);
            for (uint fragment_index = 0u; fragment_index < 8u;
                 ++fragment_index) {
                simdgroup_half8x8 weight;
                simdgroup_load(
                    weight,
                    weight_tile + fragment_index * 8u * 32u + column_offset,
                    32u, 0u, true);
                simdgroup_multiply_accumulate(accumulators[fragment_index],
                                               activation, weight,
                                               accumulators[fragment_index]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint fragment_index = 0u; fragment_index < 8u; ++fragment_index)
        simdgroup_store(accumulators[fragment_index],
                        spill + spill_row0 * 64u + fragment_index * 8u, 64u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint value = 0u; value < 16u; ++value) {
        uint linear = tid * 16u + value;
        uint batch_row = batch0 + (linear >> 6u);
        uint matrix_row = output_row0 + (linear & 63u);
        if (batch_row < parameters.batch && matrix_row < parameters.rows) {
            float result = spill[linear];
            if (has_bias != 0u) result += float(bias[matrix_row]);
            output[batch_row * parameters.rows + matrix_row] = half(result);
        }
    }
}

kernel void minimax_h3_dense_f16_mma_weight_tiled_b64(
    device const half *input [[buffer(0)]],
    device const half *weights [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant H3DenseParameters &parameters [[buffer(4)]],
    constant uint &has_bias [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half weight_tile[64 * 32];
    threadgroup float spill[64 * 64];
    uint output_row0 = group.x * 64u;
    uint batch0 = group.y * 64u;
    uint batch_row0 = batch0 + simdgroup_index * 8u;
    uint spill_row0 = simdgroup_index * 8u;
    simdgroup_float8x8 accumulators[8];
    for (uint fragment_index = 0u; fragment_index < 8u; ++fragment_index)
        accumulators[fragment_index] =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint column0 = 0u; column0 < parameters.columns; column0 += 32u) {
        for (uint index = tid; index < 64u * 32u; index += 256u) {
            uint output_offset = index >> 5u;
            uint column_offset = index & 31u;
            weight_tile[index] =
                weights[(output_row0 + output_offset) * parameters.columns +
                        column0 + column_offset];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint column_offset = 0u; column_offset < 32u;
             column_offset += 8u) {
            simdgroup_half8x8 activation;
            simdgroup_load(
                activation,
                input + batch_row0 * parameters.input_stride + column0 +
                    column_offset,
                parameters.input_stride);
            for (uint fragment_index = 0u; fragment_index < 8u;
                 ++fragment_index) {
                simdgroup_half8x8 weight;
                simdgroup_load(
                    weight,
                    weight_tile + fragment_index * 8u * 32u + column_offset,
                    32u, 0u, true);
                simdgroup_multiply_accumulate(accumulators[fragment_index],
                                               activation, weight,
                                               accumulators[fragment_index]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint fragment_index = 0u; fragment_index < 8u; ++fragment_index)
        simdgroup_store(accumulators[fragment_index],
                        spill + spill_row0 * 64u + fragment_index * 8u, 64u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint value = 0u; value < 16u; ++value) {
        uint linear = tid * 16u + value;
        uint batch_row = batch0 + (linear >> 6u);
        uint matrix_row = output_row0 + (linear & 63u);
        if (batch_row < parameters.batch && matrix_row < parameters.rows) {
            float result = spill[linear];
            if (has_bias != 0u) result += float(bias[matrix_row]);
            output[batch_row * parameters.rows + matrix_row] = half(result);
        }
    }
}

kernel void minimax_h3_rms_f16(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant H3NormParameters &parameters [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float sums[256];
    uint row = group.x;
    float sum = 0.0f;
    for (uint column = tid; column < parameters.columns; column += 256u) {
        float x = float(input[row * parameters.columns + column]);
        sum += x * x;
    }
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(sums[0] / float(parameters.columns) +
                          parameters.epsilon);
    for (uint column = tid; column < parameters.columns; column += 256u)
        output[row * parameters.columns + column] =
            half(float(input[row * parameters.columns + column]) * inverse *
                 float(weight[column]));
}

kernel void minimax_h3_layernorm_f16(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant H3NormParameters &parameters [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float sums[256];
    threadgroup float squared[256];
    uint row = group.x;
    float sum = 0.0f;
    float square = 0.0f;
    for (uint column = tid; column < parameters.columns; column += 256u) {
        float x = float(input[row * parameters.columns + column]);
        sum += x;
        square += x * x;
    }
    sums[tid] = sum;
    squared[tid] = square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
            squared[tid] += squared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = sums[0] / float(parameters.columns);
    float inverse = rsqrt(squared[0] / float(parameters.columns) -
                          mean * mean + parameters.epsilon);
    for (uint column = tid; column < parameters.columns; column += 256u) {
        float normalized =
            (float(input[row * parameters.columns + column]) - mean) * inverse;
        output[row * parameters.columns + column] =
            half(normalized * float(weight[column]) + float(bias[column]));
    }
}

kernel void minimax_h3_scaled_residual_f16(
    device half *residual [[buffer(0)]],
    device const half *update [[buffer(1)]],
    device const half *scale [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rows * columns) return;
    uint column = index % columns;
    residual[index] = half(float(residual[index]) +
                           float(update[index]) * float(scale[column]));
}

/* The ViT3D coordinates are identical for every 256x256 spatial tile and
 * every transformer layer.  Compile the 24 complex rotary coefficients per
 * row once, rather than evaluating pow/cos/sin for 32 heads x 36 layers. */
kernel void minimax_h3_build_video_rope_f32(
    device const float *positions [[buffer(0)]],
    device float2 *rotary [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rows * 24u) return;
    uint row = index / 24u;
    uint half_dimension = index - row * 24u;
    uint axis = half_dimension / 8u;
    uint frequency = half_dimension - axis * 8u;
    float angle = positions[row * 3u + axis] *
                  pow(100.0f, -float(frequency) / 8.0f);
    rotary[index] = float2(cos(angle), sin(angle));
}

kernel void minimax_h3_video_prepare_qkv(
    device const half *projected [[buffer(0)]],
    device const float2 *rotary [[buffer(1)]],
    device half *query [[buffer(2)]],
    device half *key [[buffer(3)]],
    device half *value [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float q_values[64];
    threadgroup float k_values[64];
    threadgroup float q_squares[64];
    threadgroup float k_squares[64];
    uint row = group.x;
    uint head = group.y;
    if (row >= rows || head >= 32u || tid >= 64u) return;
    uint base = row * 6144u + head * 3u * 64u;
    float q = float(projected[base + tid]);
    float k = float(projected[base + 64u + tid]);
    q_values[tid] = q;
    k_values[tid] = k;
    q_squares[tid] = q * q;
    k_squares[tid] = k * k;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 32u; stride != 0u; stride >>= 1u) {
        if (tid < stride) {
            q_squares[tid] += q_squares[tid + stride];
            k_squares[tid] += k_squares[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float qi = rsqrt(q_squares[0] / 64.0f + 1e-5f);
    float ki = rsqrt(k_squares[0] / 64.0f + 1e-5f);
    q *= qi;
    k *= ki;
    if (tid < 48u) {
        uint half_dimension = tid < 24u ? tid : tid - 24u;
        uint peer = tid < 24u ? tid + 24u : tid - 24u;
        float2 coefficient = rotary[row * 24u + half_dimension];
        float q_peer = q_values[peer] * qi;
        float k_peer = k_values[peer] * ki;
        q = q * coefficient.x +
            (tid < 24u ? -q_peer : q_peer) * coefficient.y;
        k = k * coefficient.x +
            (tid < 24u ? -k_peer : k_peer) * coefficient.y;
    }
    uint destination = (row * 32u + head) * 64u + tid;
    query[destination] = half(q);
    key[destination] = half(k);
    value[destination] = projected[base + 128u + tid];
}

kernel void minimax_h3_video_attention(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint flat = group.x;
    uint query_row = flat / 32u;
    uint head = flat % 32u;
    if (query_row >= rows) return;
    uint qbase = (query_row * 32u + head) * 64u;
    float accum0 = 0.0f;
    float accum1 = 0.0f;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint row = 0u; row < rows; ++row) {
        uint kbase = (row * 32u + head) * 64u;
        float partial = float(query[qbase + lane]) * float(key[kbase + lane]) +
                        float(query[qbase + lane + 32u]) *
                        float(key[kbase + lane + 32u]);
        float score = simd_sum(partial) * 0.125f;
        float next = max(maximum, score);
        float old_scale = exp(maximum - next);
        float new_scale = exp(score - next);
        accum0 = accum0 * old_scale + float(value[kbase + lane]) * new_scale;
        accum1 = accum1 * old_scale +
                 float(value[kbase + lane + 32u]) * new_scale;
        denominator = denominator * old_scale + new_scale;
        maximum = next;
    }
    output[qbase + lane] = half(accum0 / denominator);
    output[qbase + lane + 32u] = half(accum1 / denominator);
}

/* Eight query rows from one head share a 32-row K/V tile.  The online
 * softmax still visits keys in exactly the original row order; only the
 * source of the FP16 K/V load changes from device to threadgroup memory. */
kernel void minimax_h3_video_attention_tiled8(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half shared_key[32u * 64u];
    threadgroup half shared_value[32u * 64u];
    uint query_tile = group.x / 32u;
    uint head = group.x - query_tile * 32u;
    uint query_row = query_tile * 8u + simdgroup_index;
    bool active = query_row < rows;
    uint qbase = (query_row * 32u + head) * 64u;
    float q0 = active ? float(query[qbase + lane]) : 0.0f;
    float q1 = active ? float(query[qbase + lane + 32u]) : 0.0f;
    float accum0 = 0.0f;
    float accum1 = 0.0f;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint row0 = 0u; row0 < rows; row0 += 32u) {
        uint tile_rows = min(32u, rows - row0);
        uint tile_values = tile_rows * 64u;
        for (uint index = tid; index < tile_values; index += 256u) {
            uint local_row = index >> 6u;
            uint column = index & 63u;
            uint source = ((row0 + local_row) * 32u + head) * 64u + column;
            shared_key[index] = key[source];
            shared_value[index] = value[source];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint local_row = 0u; local_row < tile_rows; ++local_row) {
                uint local_base = local_row * 64u;
                float partial =
                    q0 * float(shared_key[local_base + lane]) +
                    q1 * float(shared_key[local_base + lane + 32u]);
                float score = simd_sum(partial) * 0.125f;
                float next = max(maximum, score);
                float old_scale = exp(maximum - next);
                float new_scale = exp(score - next);
                accum0 = accum0 * old_scale +
                         float(shared_value[local_base + lane]) * new_scale;
                accum1 = accum1 * old_scale +
                         float(shared_value[local_base + lane + 32u]) *
                             new_scale;
                denominator = denominator * old_scale + new_scale;
                maximum = next;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        output[qbase + lane] = half(accum0 / denominator);
        output[qbase + lane + 32u] = half(accum1 / denominator);
    }
}

kernel void minimax_h3_video_attention_tiled16(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half shared_key[32u * 64u];
    threadgroup half shared_value[32u * 64u];
    uint query_tile = group.x / 32u;
    uint head = group.x - query_tile * 32u;
    uint query_row = query_tile * 16u + simdgroup_index;
    bool active = query_row < rows;
    uint qbase = (query_row * 32u + head) * 64u;
    float q0 = active ? float(query[qbase + lane]) : 0.0f;
    float q1 = active ? float(query[qbase + lane + 32u]) : 0.0f;
    float accum0 = 0.0f;
    float accum1 = 0.0f;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint row0 = 0u; row0 < rows; row0 += 32u) {
        uint tile_rows = min(32u, rows - row0);
        uint tile_values = tile_rows * 64u;
        for (uint index = tid; index < tile_values; index += 512u) {
            uint local_row = index >> 6u;
            uint column = index & 63u;
            uint source = ((row0 + local_row) * 32u + head) * 64u + column;
            shared_key[index] = key[source];
            shared_value[index] = value[source];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (active) {
            for (uint local_row = 0u; local_row < tile_rows; ++local_row) {
                uint local_base = local_row * 64u;
                float partial =
                    q0 * float(shared_key[local_base + lane]) +
                    q1 * float(shared_key[local_base + lane + 32u]);
                float score = simd_sum(partial) * 0.125f;
                float next = max(maximum, score);
                float old_scale = exp(maximum - next);
                float new_scale = exp(score - next);
                accum0 = accum0 * old_scale +
                         float(shared_value[local_base + lane]) * new_scale;
                accum1 = accum1 * old_scale +
                         float(shared_value[local_base + lane + 32u]) *
                             new_scale;
                denominator = denominator * old_scale + new_scale;
                maximum = next;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
        output[qbase + lane] = half(accum0 / denominator);
        output[qbase + lane + 32u] = half(accum1 / denominator);
    }
}

kernel void minimax_h3_audio_conv1d_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant H3AudioConvParameters &p [[buffer(4)]],
    constant uint &has_bias [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    uint count = p.batch * p.output_length * p.output_channels;
    if (index >= count) return;
    uint output_channel = index % p.output_channels;
    uint flattened_time = index / p.output_channels;
    uint output_time = flattened_time % p.output_length;
    uint batch = flattened_time / p.output_length;
    float result = has_bias != 0u ? bias[output_channel] : 0.0f;
    for (uint tap = 0u; tap < p.kernel_size; ++tap) {
        int input_time = int(output_time * p.stride + tap * p.dilation) -
                         int(p.padding);
        if (input_time < 0 || input_time >= int(p.input_length)) continue;
        uint input_base = (batch * p.input_length + uint(input_time)) *
                          p.input_channels;
        uint weight_base = (output_channel * p.input_channels) * p.kernel_size +
                           tap;
        for (uint input_channel = 0u; input_channel < p.input_channels;
             ++input_channel)
            result += input[input_base + input_channel] *
                      weight[weight_base + input_channel * p.kernel_size];
    }
    output[index] = result;
}

kernel void minimax_h3_audio_conv_transpose1d_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant H3AudioConvParameters &p [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    uint count = p.batch * p.output_length * p.output_channels;
    if (index >= count) return;
    uint output_channel = index % p.output_channels;
    uint flattened_time = index / p.output_channels;
    uint output_time = flattened_time % p.output_length;
    uint batch = flattened_time / p.output_length;
    float result = bias[output_channel];
    for (uint tap = 0u; tap < p.kernel_size; ++tap) {
        int numerator = int(output_time + p.padding) - int(tap);
        if (numerator < 0 || uint(numerator) % p.stride != 0u) continue;
        uint input_time = uint(numerator) / p.stride;
        if (input_time >= p.input_length) continue;
        uint input_base = (batch * p.input_length + input_time) *
                          p.input_channels;
        for (uint input_channel = 0u; input_channel < p.input_channels;
             ++input_channel) {
            uint weight_index = (input_channel * p.output_channels +
                                 output_channel) * p.kernel_size + tap;
            result += input[input_base + input_channel] * weight[weight_index];
        }
    }
    output[index] = result;
}

inline float h3_audio_alias_upsampled(
    device const float *input,
    uint batch,
    uint length,
    uint channels,
    uint channel,
    int output_index) {
    int raw_index = output_index + 15;
    float result = 0.0f;
    for (int tap = 0; tap < 12; ++tap) {
        int numerator = raw_index - tap;
        if (numerator < 0 || (numerator & 1) != 0) continue;
        int padded_time = numerator / 2;
        if (padded_time < 0 || padded_time >= int(length) + 10) continue;
        int time = clamp(padded_time - 5, 0, int(length) - 1);
        result += input[(batch * length + uint(time)) * channels + channel] *
                  (2.0f * kH3AliasFilter[tap]);
    }
    return result;
}

kernel void minimax_h3_audio_alias_snake_f32(
    device const float *input [[buffer(0)]],
    device const float *alpha [[buffer(1)]],
    device const float *beta [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &batch_count [[buffer(4)]],
    constant uint &length [[buffer(5)]],
    constant uint &channels [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    uint count = batch_count * length * channels;
    if (index >= count) return;
    uint channel = index % channels;
    uint flattened = index / channels;
    uint time = flattened % length;
    uint batch = flattened / length;
    float result = 0.0f;
    for (uint tap = 0u; tap < 12u; ++tap) {
        int activated_index = clamp(int(time * 2u + tap) - 5,
                                    0, int(length * 2u) - 1);
        float x = h3_audio_alias_upsampled(input, batch, length, channels,
                                           channel, activated_index);
        float frequency = exp(alpha[channel]);
        float magnitude = exp(beta[channel]);
        float periodic = sin(x * frequency);
        float activated = x + periodic * periodic / (magnitude + 1e-9f);
        result += activated * kH3AliasFilter[tap];
    }
    output[index] = result;
}

kernel void minimax_h3_audio_residual_f32(
    device float *residual [[buffer(0)]],
    device const float *update [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) residual[index] += update[index];
}

kernel void minimax_h3_audio_average3_f32(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device const float *c [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) output[index] = (a[index] + b[index] + c[index]) / 3.0f;
}

kernel void minimax_h3_rms_adaln(
    device const half *input [[buffer(0)]],
    device const ushort *norm_weight [[buffer(1)]],
    device const ushort *modulation [[buffer(2)]],
    device const uchar *row_indices [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant H3NormParameters &parameters [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float sums[256];
    uint row = group.x;
    float sum = 0.0f;
    for (uint column = tid; column < parameters.columns; column += 256u) {
        float value = float(input[row * parameters.columns + column]);
        sum += value * value;
    }
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(sums[0] / float(parameters.columns) +
                          parameters.epsilon);
    uint modulation_row = uint(row_indices[row]) * parameters.modulation_stride;
    for (uint column = tid; column < parameters.columns; column += 256u) {
        float value = float(input[row * parameters.columns + column]) * inverse *
                      h3_bfloat_to_float(norm_weight[column]);
        if (parameters.modulation_stride != 0u) {
            float shift = h3_bfloat_to_float(
                modulation[modulation_row + parameters.shift_offset + column]);
            float scale = h3_bfloat_to_float(
                modulation[modulation_row + parameters.scale_offset + column]);
            value = value * (1.0f + scale) + shift;
        }
        output[row * parameters.columns + column] = half(value);
    }
}

kernel void minimax_h3_rms_plain(
    device const half *input [[buffer(0)]],
    device const ushort *norm_weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant H3NormParameters &parameters [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float sums[256];
    uint row = group.x;
    float sum = 0.0f;
    for (uint column = tid; column < parameters.columns; column += 256u) {
        float value = float(input[row * parameters.columns + column]);
        sum += value * value;
    }
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(sums[0] / float(parameters.columns) +
                          parameters.epsilon);
    for (uint column = tid; column < parameters.columns; column += 256u)
        output[row * parameters.columns + column] = half(
            float(input[row * parameters.columns + column]) * inverse *
            h3_bfloat_to_float(norm_weight[column]));
}

kernel void minimax_h3_rms_plain_bf16(
    device const bfloat *input [[buffer(0)]],
    device const ushort *norm_weight [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant H3NormParameters &parameters [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float sums[256];
    uint row = group.x;
    float sum = 0.0f;
    for (uint column = tid; column < parameters.columns; column += 256u) {
        float value = float(input[row * parameters.columns + column]);
        sum += value * value;
    }
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(sums[0] / float(parameters.columns) +
                          parameters.epsilon);
    for (uint column = tid; column < parameters.columns; column += 256u)
        output[row * parameters.columns + column] = bfloat(
            float(input[row * parameters.columns + column]) * inverse *
            h3_bfloat_to_float(norm_weight[column]));
}

kernel void minimax_h3_rms_adaln_bf16(
    device const bfloat *input [[buffer(0)]],
    device const ushort *norm_weight [[buffer(1)]],
    device const ushort *modulation [[buffer(2)]],
    device const uchar *row_indices [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant H3NormParameters &parameters [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float sums[256];
    uint row = group.x;
    float sum = 0.0f;
    for (uint column = tid; column < parameters.columns; column += 256u) {
        float value = float(input[row * parameters.columns + column]);
        sum += value * value;
    }
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(sums[0] / float(parameters.columns) +
                          parameters.epsilon);
    uint modulation_row = uint(row_indices[row]) *
                          parameters.modulation_stride;
    for (uint column = tid; column < parameters.columns; column += 256u) {
        float value = float(input[row * parameters.columns + column]) *
                      inverse * h3_bfloat_to_float(norm_weight[column]);
        if (parameters.modulation_stride != 0u) {
            float shift = h3_bfloat_to_float(
                modulation[modulation_row + parameters.shift_offset + column]);
            float scale = h3_bfloat_to_float(
                modulation[modulation_row + parameters.scale_offset + column]);
            value = value * (1.0f + scale) + shift;
        }
        output[row * parameters.columns + column] = bfloat(value);
    }
}

kernel void minimax_h3_silu_split(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    uint count = rows * width;
    if (index >= count) return;
    uint row = index / width;
    uint column = index - row * width;
    float gate = float(input[row * (2u * width) + column]);
    float value = float(input[row * (2u * width) + width + column]);
    output[index] = half((gate / (1.0f + exp(-gate))) * value);
}

kernel void minimax_h3_silu_split_bf16(
    device const bfloat *input [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant uint &rows [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    uint count = rows * width;
    if (index >= count) return;
    uint row = index / width;
    uint column = index - row * width;
    float gate = float(input[row * (2u * width) + column]);
    float value = float(input[row * (2u * width) + width + column]);
    output[index] = bfloat((gate / (1.0f + exp(-gate))) * value);
}

kernel void minimax_h3_silu_pair(
    device const half *gate [[buffer(0)]],
    device const half *value [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &element_count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= element_count) return;
    float g = float(gate[index]);
    output[index] = half((g / (1.0f + exp(-g))) * float(value[index]));
}

kernel void minimax_h3_gated_residual(
    device half *residual [[buffer(0)]],
    device const half *update [[buffer(1)]],
    device const ushort *modulation [[buffer(2)]],
    device const uchar *row_indices [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &modulation_stride [[buffer(6)]],
    constant uint &gate_offset [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rows * columns) return;
    uint row = index / columns;
    uint column = index - row * columns;
    float gate = modulation_stride == 0u ? 1.0f : h3_bfloat_to_float(
        modulation[uint(row_indices[row]) * modulation_stride + gate_offset +
                   column]);
    residual[index] = half(float(residual[index]) + gate * float(update[index]));
}

kernel void minimax_h3_gated_residual_bf16(
    device bfloat *residual [[buffer(0)]],
    device const bfloat *update [[buffer(1)]],
    device const ushort *modulation [[buffer(2)]],
    device const uchar *row_indices [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    constant uint &columns [[buffer(5)]],
    constant uint &modulation_stride [[buffer(6)]],
    constant uint &gate_offset [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= rows * columns) return;
    uint row = index / columns;
    uint column = index - row * columns;
    float gate = modulation_stride == 0u ? 1.0f : h3_bfloat_to_float(
        modulation[uint(row_indices[row]) * modulation_stride + gate_offset +
                   column]);
    residual[index] = bfloat(float(residual[index]) +
                             gate * float(update[index]));
}

kernel void minimax_h3_qwen_prepare_qkv(
    device const half *query_projected [[buffer(0)]],
    device const half *key_projected [[buffer(1)]],
    device const half *value_projected [[buffer(2)]],
    device const ushort *query_norm [[buffer(3)]],
    device const ushort *key_norm [[buffer(4)]],
    device half *query [[buffer(5)]],
    device half *key [[buffer(6)]],
    device half *value [[buffer(7)]],
    constant H3QwenAttentionParameters &parameters [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float sums[256];
    uint token = group.x;
    uint head = group.y;
    bool is_query = head < parameters.query_heads;
    uint local_head = is_query ? head : head - parameters.query_heads;
    uint head_count = is_query ? parameters.query_heads :
                                 parameters.key_value_heads;
    if (local_head >= head_count) return;
    device const half *source = is_query ? query_projected : key_projected;
    device const ushort *norm = is_query ? query_norm : key_norm;
    uint base = token * head_count * parameters.head_dimension +
                local_head * parameters.head_dimension;
    float squared = 0.0f;
    for (uint dimension = tid; dimension < parameters.head_dimension;
         dimension += 256u) {
        float x = float(source[base + dimension]);
        squared += x * x;
    }
    sums[tid] = squared;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inverse = rsqrt(sums[0] / float(parameters.head_dimension) + 1e-6f);
    for (uint dimension = tid; dimension < parameters.head_dimension;
         dimension += 256u) {
        float x = float(source[base + dimension]) * inverse *
                  h3_bfloat_to_float(norm[dimension]);
        uint half_dimension = parameters.head_dimension / 2u;
        uint frequency = dimension % half_dimension;
        float angle = float(token) * pow(5000000.0f,
            -2.0f * float(frequency) / float(parameters.head_dimension));
        uint peer = dimension < half_dimension ? dimension + half_dimension
                                               : dimension - half_dimension;
        float peer_value = float(source[base + peer]) * inverse *
                           h3_bfloat_to_float(norm[peer]);
        float rotated = dimension < half_dimension ? -peer_value : peer_value;
        device half *destination = is_query ? query : key;
        destination[base + dimension] = half(x * cos(angle) + rotated * sin(angle));
    }
    if (!is_query) {
        uint value_base = token * parameters.key_value_heads *
                          parameters.head_dimension +
                          local_head * parameters.head_dimension;
        for (uint dimension = tid; dimension < parameters.head_dimension;
             dimension += 256u)
            value[value_base + dimension] = value_projected[value_base + dimension];
    }
}

kernel void minimax_h3_qwen_causal_attention(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device half *output [[buffer(3)]],
    constant H3QwenAttentionParameters &parameters [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint flat = group.x * simdgroups_per_group + simdgroup_index;
    uint work = parameters.token_count * parameters.query_heads;
    if (flat >= work) return;
    uint token = flat / parameters.query_heads;
    uint q_head = flat - token * parameters.query_heads;
    uint kv_head = q_head / (parameters.query_heads /
                             parameters.key_value_heads);
    float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float maximum = -INFINITY;
    float denominator = 0.0f;
    uint q_base = (token * parameters.query_heads + q_head) *
                  parameters.head_dimension;
    for (uint source_token = 0u; source_token <= token; ++source_token) {
        uint k_base = (source_token * parameters.key_value_heads + kv_head) *
                      parameters.head_dimension;
        float partial = 0.0f;
        for (uint i = 0u; i < 4u; ++i) {
            uint dimension = lane + i * 32u;
            partial += float(query[q_base + dimension]) *
                       float(key[k_base + dimension]);
        }
        float score = simd_sum(partial) * parameters.scale;
        float next_maximum = max(maximum, score);
        float old_scale = exp(maximum - next_maximum);
        float new_scale = exp(score - next_maximum);
        uint v_base = k_base;
        for (uint i = 0u; i < 4u; ++i) {
            uint dimension = lane + i * 32u;
            accumulator[i] = accumulator[i] * old_scale +
                             float(value[v_base + dimension]) * new_scale;
        }
        denominator = denominator * old_scale + new_scale;
        maximum = next_maximum;
    }
    for (uint i = 0u; i < 4u; ++i) {
        uint dimension = lane + i * 32u;
        output[q_base + dimension] = half(accumulator[i] / denominator);
    }
}

kernel void minimax_h3_prepare_qkv(
    device const half *projected [[buffer(0)]],
    device const ushort *query_norm [[buffer(1)]],
    device const ushort *key_norm [[buffer(2)]],
    device const float2 *rotary [[buffer(3)]],
    device half *query [[buffer(4)]],
    device half *key [[buffer(5)]],
    device half *value [[buffer(6)]],
    constant uint &row_count [[buffer(7)]],
    constant uint &apply_rotary [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float source_values[128];
    threadgroup float squared[256];
    uint row = group.x;
    uint encoded_head = group.y;
    bool is_query = encoded_head < kH3HeadCount;
    uint head = is_query ? encoded_head : encoded_head - kH3HeadCount;
    if (row >= row_count || head >= kH3HeadCount) return;
    uint slab = is_query ? 0u : kH3HeadCount * kH3HeadDim;
    uint source_base = row * (3u * kH3HeadCount * kH3HeadDim) + slab +
                       head * kH3HeadDim;
    float sum = 0.0f;
    if (tid < kH3HeadDim) {
        float x = float(projected[source_base + tid]);
        source_values[tid] = x;
        sum = x * x;
    }
    squared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        if (tid < stride) squared[tid] += squared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < kH3HeadDim) {
        device const ushort *norm = is_query ? query_norm : key_norm;
        float inverse = rsqrt(squared[0] / float(kH3HeadDim) + 1e-5f);
        float x = source_values[tid] * inverse *
                  h3_bfloat_to_float(norm[tid]);
        if (apply_rotary != 0u && tid < 96u) {
            uint frequency_slot = tid % 48u;
            uint peer = tid < 48u ? tid + 48u : tid - 48u;
            float peer_value = source_values[peer] * inverse *
                               h3_bfloat_to_float(norm[peer]);
            float2 angle = rotary[row * kH3RopeFrequencySlots + frequency_slot];
            float rotated = tid < 48u ? -peer_value : peer_value;
            x = x * angle.x + rotated * angle.y;
        }
        uint destination = (row * kH3HeadCount + head) * kH3HeadDim + tid;
        if (is_query) {
            query[destination] = half(x);
        } else {
            key[destination] = half(x);
            uint value_base = row * (3u * kH3HeadCount * kH3HeadDim) +
                              2u * kH3HeadCount * kH3HeadDim +
                              head * kH3HeadDim;
            value[destination] = projected[value_base + tid];
        }
    }
}

/* Differential oracle for the precomputed MM-RoPE path.  This retains the
 * former per-head transcendental evaluation and is not used by production
 * inference. */
kernel void minimax_h3_prepare_qkv_bf16_reference(
    device const bfloat *projected [[buffer(0)]],
    device const ushort *query_norm [[buffer(1)]],
    device const ushort *key_norm [[buffer(2)]],
    device const float *positions [[buffer(3)]],
    device bfloat *query [[buffer(4)]],
    device bfloat *key [[buffer(5)]],
    device bfloat *value [[buffer(6)]],
    constant uint &row_count [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float source_values[128];
    threadgroup float squared[256];
    uint row = group.x;
    uint encoded_head = group.y;
    bool is_query = encoded_head < kH3HeadCount;
    uint head = is_query ? encoded_head : encoded_head - kH3HeadCount;
    if (row >= row_count || head >= kH3HeadCount) return;
    uint slab = is_query ? 0u : kH3HeadCount * kH3HeadDim;
    uint source_base = row * (3u * kH3HeadCount * kH3HeadDim) + slab +
                       head * kH3HeadDim;
    float sum = 0.0f;
    if (tid < kH3HeadDim) {
        float x = float(projected[source_base + tid]);
        source_values[tid] = x;
        sum = x * x;
    }
    squared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        if (tid < stride) squared[tid] += squared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < kH3HeadDim) {
        device const ushort *norm = is_query ? query_norm : key_norm;
        float inverse = rsqrt(squared[0] / float(kH3HeadDim) + 1e-5f);
        float x = source_values[tid] * inverse *
                  h3_bfloat_to_float(norm[tid]);
        if (tid < 96u) {
            uint frequency_slot = tid % 48u;
            uint axis = frequency_slot / 16u;
            uint frequency = frequency_slot - axis * 16u;
            uint peer = tid < 48u ? tid + 48u : tid - 48u;
            float peer_value = source_values[peer] * inverse *
                               h3_bfloat_to_float(norm[peer]);
            float inverse_frequency = pow(10000.0f,
                -float(frequency) / 16.0f);
            float angle = positions[row * 3u + axis] * inverse_frequency;
            float rotated = tid < 48u ? -peer_value : peer_value;
            x = x * cos(angle) + rotated * sin(angle);
        }
        uint destination = (row * kH3HeadCount + head) * kH3HeadDim + tid;
        if (is_query) {
            query[destination] = bfloat(x);
        } else {
            key[destination] = bfloat(x);
            uint value_base = row * (3u * kH3HeadCount * kH3HeadDim) +
                              2u * kH3HeadCount * kH3HeadDim +
                              head * kH3HeadDim;
            value[destination] = projected[value_base + tid];
        }
    }
}

kernel void minimax_h3_prepare_qkv_bf16(
    device const bfloat *projected [[buffer(0)]],
    device const ushort *query_norm [[buffer(1)]],
    device const ushort *key_norm [[buffer(2)]],
    device const float2 *rotary [[buffer(3)]],
    device bfloat *query [[buffer(4)]],
    device bfloat *key [[buffer(5)]],
    device bfloat *value [[buffer(6)]],
    constant uint &row_count [[buffer(7)]],
    constant uint &apply_rotary [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float source_values[128];
    threadgroup float squared[256];
    uint row = group.x;
    uint encoded_head = group.y;
    bool is_query = encoded_head < kH3HeadCount;
    uint head = is_query ? encoded_head : encoded_head - kH3HeadCount;
    if (row >= row_count || head >= kH3HeadCount) return;
    uint slab = is_query ? 0u : kH3HeadCount * kH3HeadDim;
    uint source_base = row * (3u * kH3HeadCount * kH3HeadDim) + slab +
                       head * kH3HeadDim;
    float sum = 0.0f;
    if (tid < kH3HeadDim) {
        float x = float(projected[source_base + tid]);
        source_values[tid] = x;
        sum = x * x;
    }
    squared[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride != 0u; stride >>= 1u) {
        if (tid < stride) squared[tid] += squared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < kH3HeadDim) {
        device const ushort *norm = is_query ? query_norm : key_norm;
        float inverse = rsqrt(squared[0] / float(kH3HeadDim) + 1e-5f);
        float x = source_values[tid] * inverse *
                  h3_bfloat_to_float(norm[tid]);
        if (apply_rotary != 0u && tid < 96u) {
            uint frequency_slot = tid % 48u;
            uint peer = tid < 48u ? tid + 48u : tid - 48u;
            float peer_value = source_values[peer] * inverse *
                               h3_bfloat_to_float(norm[peer]);
            float2 angle = rotary[row * kH3RopeFrequencySlots + frequency_slot];
            float rotated = tid < 48u ? -peer_value : peer_value;
            x = x * angle.x + rotated * angle.y;
        }
        uint destination = (row * kH3HeadCount + head) * kH3HeadDim + tid;
        if (is_query) {
            query[destination] = bfloat(x);
        } else {
            key[destination] = bfloat(x);
            uint value_base = row * (3u * kH3HeadCount * kH3HeadDim) +
                              2u * kH3HeadCount * kH3HeadDim +
                              head * kH3HeadDim;
            value[destination] = projected[value_base + tid];
        }
    }
}

kernel void minimax_h3_dense_attention_bf16(
    device const bfloat4 *queries [[buffer(0)]],
    device const bfloat4 *keys [[buffer(1)]],
    device const bfloat4 *values [[buffer(2)]],
    device bfloat4 *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint flat = group.x * simdgroups_per_group + simdgroup_index;
    uint work_count = rows * kH3HeadCount;
    if (flat >= work_count) return;
    uint query_row = flat / kH3HeadCount;
    uint head = flat - query_row * kH3HeadCount;
    uint query_offset =
        (query_row * kH3HeadCount + head) * kH3HeadHalf4 + lane;
    float4 query = float4(queries[query_offset]);
    float4 accumulator = 0.0f;
    float maximum = -INFINITY;
    float denominator = 0.0f;
    for (uint row = 0u; row < rows; ++row) {
        uint key_offset =
            (row * kH3HeadCount + head) * kH3HeadHalf4 + lane;
        float score = simd_sum(dot(query, float4(keys[key_offset]))) *
                      kH3AttentionScale;
        float next_maximum = max(maximum, score);
        float old_scale = exp(maximum - next_maximum);
        float new_scale = exp(score - next_maximum);
        accumulator = accumulator * old_scale +
                      float4(values[key_offset]) * new_scale;
        denominator = denominator * old_scale + new_scale;
        maximum = next_maximum;
    }
    output[query_offset] = bfloat4(accumulator / denominator);
}

/* Exact dense attention for the real-weight path.  One threadgroup evaluates
 * 64 query rows for one head.  QK is recomputed in the value pass so no N x N
 * score matrix is materialized.  The output stays FP32 until a separate
 * conversion kernel rounds it to the BF16 residual stream. */
kernel void minimax_h3_dense_attention_mma64_bf16(
    device const bfloat *queries [[buffer(0)]],
    device const bfloat *keys [[buffer(1)]],
    device const bfloat *values [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup bfloat key_tile[8 * kH3HeadDim];
    threadgroup float value_tile[8 * kH3HeadDim];
    threadgroup float score_tiles[8 * 8 * 8];
    threadgroup float probability_tiles[8 * 8 * 8];
    threadgroup float row_maximum[64];
    threadgroup float row_denominator[64];
    uint query_block_start = group.x * 64u;
    uint query_block_rows = min(64u, rows - query_block_start);
    uint head = group.y;
    uint query_base = simdgroup_index * 8u;
    uint candidate_blocks = (rows + 7u) / 8u;
    simdgroup_bfloat8x8 query_fragments[16];

    for (uint frag = 0u; frag < 16u; ++frag) {
        if (query_base < query_block_rows) {
            device const bfloat *source =
                queries + ((query_block_start + query_base) * kH3HeadCount +
                           head) * kH3HeadDim + frag * 8u;
            simdgroup_load(query_fragments[frag], source,
                           kH3HeadCount * kH3HeadDim);
        } else {
            query_fragments[frag] =
                make_filled_simdgroup_matrix<bfloat, 8, 8>(bfloat(0.0f));
        }
    }
    if (tid < 64u) {
        row_maximum[tid] = -INFINITY;
        row_denominator[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint candidate_block = 0u; candidate_block < candidate_blocks;
         ++candidate_block) {
        for (uint linear = tid; linear < 8u * kH3HeadDim; linear += 256u) {
            uint key_in_block = linear / kH3HeadDim;
            uint dimension = linear - key_in_block * kH3HeadDim;
            uint candidate = candidate_block * 8u + key_in_block;
            key_tile[linear] = candidate < rows
                ? keys[(candidate * kH3HeadCount + head) * kH3HeadDim +
                       dimension]
                : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 scores =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint frag = 0u; frag < 16u; ++frag) {
            simdgroup_bfloat8x8 key_fragment;
            simdgroup_load(key_fragment, key_tile + frag * 8u,
                           kH3HeadDim, 0u, true);
            simdgroup_multiply_accumulate(scores, query_fragments[frag],
                                           key_fragment, scores);
        }
        threadgroup float *score_tile =
            score_tiles + simdgroup_index * 64u;
        simdgroup_store(scores, score_tile, 8u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint local_query = query_base + lane;
        if (lane < 8u && local_query < query_block_rows) {
            uint valid_keys = min(8u, rows - candidate_block * 8u);
            float block_maximum = -INFINITY;
            for (uint key = 0u; key < valid_keys; ++key)
                block_maximum = max(
                    block_maximum,
                    score_tile[lane * 8u + key] * kH3AttentionScale);
            float next_maximum = max(row_maximum[local_query], block_maximum);
            float denominator = row_denominator[local_query] *
                                exp(row_maximum[local_query] - next_maximum);
            for (uint key = 0u; key < valid_keys; ++key)
                denominator += exp(score_tile[lane * 8u + key] *
                                       kH3AttentionScale -
                                   next_maximum);
            row_maximum[local_query] = next_maximum;
            row_denominator[local_query] = denominator;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_float8x8 accumulators[16];
    for (uint frag = 0u; frag < 16u; ++frag)
        accumulators[frag] =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint candidate_block = 0u; candidate_block < candidate_blocks;
         ++candidate_block) {
        for (uint linear = tid; linear < 8u * kH3HeadDim; linear += 256u) {
            uint key_in_block = linear / kH3HeadDim;
            uint dimension = linear - key_in_block * kH3HeadDim;
            uint candidate = candidate_block * 8u + key_in_block;
            if (candidate < rows) {
                uint source = (candidate * kH3HeadCount + head) *
                                  kH3HeadDim +
                              dimension;
                key_tile[linear] = keys[source];
                value_tile[linear] = float(values[source]);
            } else {
                key_tile[linear] = bfloat(0.0f);
                value_tile[linear] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 scores =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint frag = 0u; frag < 16u; ++frag) {
            simdgroup_bfloat8x8 key_fragment;
            simdgroup_load(key_fragment, key_tile + frag * 8u,
                           kH3HeadDim, 0u, true);
            simdgroup_multiply_accumulate(scores, query_fragments[frag],
                                           key_fragment, scores);
        }
        threadgroup float *score_tile =
            score_tiles + simdgroup_index * 64u;
        simdgroup_store(scores, score_tile, 8u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float *probability_tile =
            probability_tiles + simdgroup_index * 64u;
        for (uint index = lane; index < 64u; index += 32u) {
            uint query_in_simdgroup = index / 8u;
            uint key_in_block = index & 7u;
            uint local_query = query_base + query_in_simdgroup;
            uint candidate = candidate_block * 8u + key_in_block;
            probability_tile[index] =
                local_query < query_block_rows && candidate < rows
                    ? exp(score_tile[index] * kH3AttentionScale -
                          row_maximum[local_query]) /
                          row_denominator[local_query]
                    : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 probability_fragment;
        simdgroup_load(probability_fragment, probability_tile, 8u);
        for (uint frag = 0u; frag < 16u; ++frag) {
            simdgroup_float8x8 value_fragment;
            simdgroup_load(value_fragment, value_tile + frag * 8u,
                           kH3HeadDim);
            simdgroup_multiply_accumulate(accumulators[frag],
                                           probability_fragment,
                                           value_fragment,
                                           accumulators[frag]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (query_base < query_block_rows) {
        device float *destination =
            output + ((query_block_start + query_base) * kH3HeadCount + head) *
                         kH3HeadDim;
        for (uint frag = 0u; frag < 16u; ++frag)
            simdgroup_store(accumulators[frag], destination + frag * 8u,
                            kH3HeadCount * kH3HeadDim);
    }
}

/* Exact dense attention with the same FP32 QK, softmax and value
 * accumulation as minimax_h3_dense_attention_mma64_bf16.  The reference
 * path writes a rows x heads x dimensions FP32 image and launches a second
 * conversion kernel.  Once the value pass is complete, score_tiles is dead;
 * reuse those 2 KiB as a per-fragment FP32 spill and round directly to the
 * BF16 residual boundary.  This removes the 448 MiB 480p scratch image and
 * its full-device write/read without changing the arithmetic order. */
kernel void minimax_h3_dense_attention_mma64_bf16_direct(
    device const bfloat *queries [[buffer(0)]],
    device const bfloat *keys [[buffer(1)]],
    device const bfloat *values [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant uint &rows [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup bfloat key_tile[8 * kH3HeadDim];
    threadgroup float value_tile[8 * kH3HeadDim];
    threadgroup float score_tiles[8 * 8 * 8];
    threadgroup float probability_tiles[8 * 8 * 8];
    threadgroup float row_maximum[64];
    threadgroup float row_denominator[64];
    uint query_block_start = group.x * 64u;
    uint query_block_rows = min(64u, rows - query_block_start);
    uint head = group.y;
    uint query_base = simdgroup_index * 8u;
    uint candidate_blocks = (rows + 7u) / 8u;
    simdgroup_bfloat8x8 query_fragments[16];

    for (uint frag = 0u; frag < 16u; ++frag) {
        if (query_base < query_block_rows) {
            device const bfloat *source =
                queries + ((query_block_start + query_base) * kH3HeadCount +
                           head) * kH3HeadDim + frag * 8u;
            simdgroup_load(query_fragments[frag], source,
                           kH3HeadCount * kH3HeadDim);
        } else {
            query_fragments[frag] =
                make_filled_simdgroup_matrix<bfloat, 8, 8>(bfloat(0.0f));
        }
    }
    if (tid < 64u) {
        row_maximum[tid] = -INFINITY;
        row_denominator[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint candidate_block = 0u; candidate_block < candidate_blocks;
         ++candidate_block) {
        for (uint linear = tid; linear < 8u * kH3HeadDim; linear += 256u) {
            uint key_in_block = linear / kH3HeadDim;
            uint dimension = linear - key_in_block * kH3HeadDim;
            uint candidate = candidate_block * 8u + key_in_block;
            key_tile[linear] = candidate < rows
                ? keys[(candidate * kH3HeadCount + head) * kH3HeadDim +
                       dimension]
                : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 scores =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint frag = 0u; frag < 16u; ++frag) {
            simdgroup_bfloat8x8 key_fragment;
            simdgroup_load(key_fragment, key_tile + frag * 8u,
                           kH3HeadDim, 0u, true);
            simdgroup_multiply_accumulate(scores, query_fragments[frag],
                                           key_fragment, scores);
        }
        threadgroup float *score_tile =
            score_tiles + simdgroup_index * 64u;
        simdgroup_store(scores, score_tile, 8u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint local_query = query_base + lane;
        if (lane < 8u && local_query < query_block_rows) {
            uint valid_keys = min(8u, rows - candidate_block * 8u);
            float block_maximum = -INFINITY;
            for (uint key = 0u; key < valid_keys; ++key)
                block_maximum = max(
                    block_maximum,
                    score_tile[lane * 8u + key] * kH3AttentionScale);
            float next_maximum = max(row_maximum[local_query], block_maximum);
            float denominator = row_denominator[local_query] *
                                exp(row_maximum[local_query] - next_maximum);
            for (uint key = 0u; key < valid_keys; ++key)
                denominator += exp(score_tile[lane * 8u + key] *
                                       kH3AttentionScale -
                                   next_maximum);
            row_maximum[local_query] = next_maximum;
            row_denominator[local_query] = denominator;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_float8x8 accumulators[16];
    for (uint frag = 0u; frag < 16u; ++frag)
        accumulators[frag] =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint candidate_block = 0u; candidate_block < candidate_blocks;
         ++candidate_block) {
        for (uint linear = tid; linear < 8u * kH3HeadDim; linear += 256u) {
            uint key_in_block = linear / kH3HeadDim;
            uint dimension = linear - key_in_block * kH3HeadDim;
            uint candidate = candidate_block * 8u + key_in_block;
            if (candidate < rows) {
                uint source = (candidate * kH3HeadCount + head) *
                                  kH3HeadDim +
                              dimension;
                key_tile[linear] = keys[source];
                value_tile[linear] = float(values[source]);
            } else {
                key_tile[linear] = bfloat(0.0f);
                value_tile[linear] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 scores =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint frag = 0u; frag < 16u; ++frag) {
            simdgroup_bfloat8x8 key_fragment;
            simdgroup_load(key_fragment, key_tile + frag * 8u,
                           kH3HeadDim, 0u, true);
            simdgroup_multiply_accumulate(scores, query_fragments[frag],
                                           key_fragment, scores);
        }
        threadgroup float *score_tile =
            score_tiles + simdgroup_index * 64u;
        simdgroup_store(scores, score_tile, 8u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float *probability_tile =
            probability_tiles + simdgroup_index * 64u;
        for (uint index = lane; index < 64u; index += 32u) {
            uint query_in_simdgroup = index / 8u;
            uint key_in_block = index & 7u;
            uint local_query = query_base + query_in_simdgroup;
            uint candidate = candidate_block * 8u + key_in_block;
            probability_tile[index] =
                local_query < query_block_rows && candidate < rows
                    ? exp(score_tile[index] * kH3AttentionScale -
                          row_maximum[local_query]) /
                          row_denominator[local_query]
                    : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 probability_fragment;
        simdgroup_load(probability_fragment, probability_tile, 8u);
        for (uint frag = 0u; frag < 16u; ++frag) {
            simdgroup_float8x8 value_fragment;
            simdgroup_load(value_fragment, value_tile + frag * 8u,
                           kH3HeadDim);
            simdgroup_multiply_accumulate(accumulators[frag],
                                           probability_fragment,
                                           value_fragment,
                                           accumulators[frag]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    /* score_tiles is no longer live.  Spill one 8-column output fragment for
     * all eight simdgroups, then have the full threadgroup perform the exact
     * float-to-BF16 boundary conversion. */
    for (uint frag = 0u; frag < 16u; ++frag) {
        threadgroup float *spill = score_tiles + simdgroup_index * 64u;
        simdgroup_store(accumulators[frag], spill, 8u);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint linear = tid; linear < 512u; linear += 256u) {
            uint output_simdgroup = linear / 64u;
            uint in_tile = linear - output_simdgroup * 64u;
            uint query_in_simdgroup = in_tile / 8u;
            uint column = in_tile - query_in_simdgroup * 8u;
            uint local_query = output_simdgroup * 8u + query_in_simdgroup;
            if (local_query < query_block_rows) {
                uint destination =
                    ((query_block_start + local_query) * kH3HeadCount + head) *
                        kH3HeadDim +
                    frag * 8u + column;
                output[destination] = bfloat(score_tiles[linear]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void minimax_h3_f32_to_bf16(
    device const float *input [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant uint &element_count [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < element_count) output[index] = bfloat(input[index]);
}

/* The tree-attention execution image is tree-major while the released H3
 * graph is [text | audio | frame-major video].  The permutation is compiled
 * once for a fixed geometry and reused by every layer/evaluation. */
kernel void minimax_h3_reorder_bf16_to_f16(
    device const bfloat *input [[buffer(0)]],
    device const uint *logical_to_physical [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &row_count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint row_width = kH3HeadCount * kH3HeadDim;
    uint element_count = row_count * row_width;
    if (index >= element_count) return;
    uint logical_row = index / row_width;
    uint column = index - logical_row * row_width;
    uint physical_row = logical_to_physical[logical_row];
    output[physical_row * row_width + column] = half(float(input[index]));
}

kernel void minimax_h3_reorder_f16_to_bf16(
    device const half *input [[buffer(0)]],
    device const uint *logical_to_physical [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant uint &row_count [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    const uint row_width = kH3HeadCount * kH3HeadDim;
    uint element_count = row_count * row_width;
    if (index >= element_count) return;
    uint logical_row = index / row_width;
    uint column = index - logical_row * row_width;
    uint physical_row = logical_to_physical[logical_row];
    output[index] = bfloat(float(input[physical_row * row_width + column]));
}

kernel void minimax_h3_build_leaf_summaries(
    device const half4 *keys [[buffer(0)]],
    device const half4 *values [[buffer(1)]],
    device const H3TreeNodeGPU *nodes [[buffer(2)]],
    constant H3TreeParameters &parameters [[buffer(3)]],
    device half4 *summary_keys [[buffer(4)]],
    device half4 *summary_values [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint flat = group.x;
    uint local_node = flat / kH3HeadCount;
    uint head = flat - local_node * kH3HeadCount;
    if (local_node >= parameters.aggregate_count) return;
    uint leaf_index = parameters.aggregate_start + local_node;

    H3TreeNodeGPU leaf = nodes[leaf_index];
    float4 key_sum = 0.0f;
    float4 value_sum = 0.0f;
    for (uint token = 0; token < leaf.token_count; ++token) {
        uint row = leaf.physical_start + token;
        uint offset = (row * kH3HeadCount + head) * kH3HeadHalf4 + lane;
        key_sum += float4(keys[offset]);
        value_sum += float4(values[offset]);
    }
    float inverse_count = 1.0f / float(leaf.token_count);
    uint output = (leaf_index * kH3HeadCount + head) * kH3HeadHalf4 + lane;
    summary_keys[output] = half4(key_sum * inverse_count);
    summary_values[output] = half4(value_sum * inverse_count);
}

kernel void minimax_h3_build_parent_summaries(
    device const half4 *input_keys [[buffer(0)]],
    device const half4 *input_values [[buffer(1)]],
    device const H3TreeNodeGPU *nodes [[buffer(2)]],
    constant H3TreeParameters &parameters [[buffer(3)]],
    device half4 *summary_keys [[buffer(4)]],
    device half4 *summary_values [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    (void)input_keys;
    (void)input_values;
    uint flat = group.x;
    uint local_node = flat / kH3HeadCount;
    uint head = flat - local_node * kH3HeadCount;
    if (local_node >= parameters.aggregate_count) return;
    uint node_index = parameters.aggregate_start + local_node;
    H3TreeNodeGPU node = nodes[node_index];
    float4 key_sum = 0.0f;
    float4 value_sum = 0.0f;
    for (uint child_offset = 0; child_offset < node.child_count;
         ++child_offset) {
        uint child_index = node.first_child + child_offset;
        float weight = float(nodes[child_index].token_count);
        uint input = (child_index * kH3HeadCount + head) * kH3HeadHalf4 + lane;
        key_sum += float4(summary_keys[input]) * weight;
        value_sum += float4(summary_values[input]) * weight;
    }
    float inverse_count = 1.0f / float(node.token_count);
    uint output = (node_index * kH3HeadCount + head) * kH3HeadHalf4 + lane;
    summary_keys[output] = half4(key_sum * inverse_count);
    summary_values[output] = half4(value_sum * inverse_count);
}

inline void h3_online_attention_update(
    float score,
    float4 value,
    thread float &maximum,
    thread float &denominator,
    thread float4 &accumulator) {
    float next_maximum = max(maximum, score);
    float old_scale = exp(maximum - next_maximum);
    float new_scale = exp(score - next_maximum);
    accumulator = accumulator * old_scale + value * new_scale;
    denominator = denominator * old_scale + new_scale;
    maximum = next_maximum;
}

kernel void minimax_h3_hierarchical_attention(
    device const half4 *queries [[buffer(0)]],
    device const half4 *keys [[buffer(1)]],
    device const half4 *values [[buffer(2)]],
    device const half4 *summary_keys [[buffer(3)]],
    device const half4 *summary_values [[buffer(4)]],
    device const float *summary_log_counts [[buffer(5)]],
    device const uint *route_offsets [[buffer(6)]],
    device const uint *route_entries [[buffer(7)]],
    constant H3TreeParameters &parameters [[buffer(8)]],
    device half4 *output [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint flat = group.x * simdgroups_per_group + simdgroup_index;
    uint work_count = parameters.sequence_rows * kH3HeadCount;
    if (flat >= work_count) return;
    uint query_row = flat / kH3HeadCount;
    uint head = flat - query_row * kH3HeadCount;
    uint query_offset =
        (query_row * kH3HeadCount + head) * kH3HeadHalf4 + lane;
    float4 query = float4(queries[query_offset]);
    float4 accumulator = 0.0f;
    float maximum = -INFINITY;
    float denominator = 0.0f;

    for (uint row = 0; row < parameters.exact_rows; ++row) {
        uint key_offset = (row * kH3HeadCount + head) * kH3HeadHalf4 + lane;
        float partial = dot(query, float4(keys[key_offset]));
        float score = simd_sum(partial) * kH3AttentionScale;
        h3_online_attention_update(score, float4(values[key_offset]), maximum,
                                   denominator, accumulator);
    }

    uint route_index = 0u;
    if (query_row >= parameters.video_start) {
        uint video_row = query_row - parameters.video_start;
        uint frame = video_row / parameters.rows_per_video_frame;
        uint frame_row = video_row - frame * parameters.rows_per_video_frame;
        uint patch_y = frame_row / parameters.patch_columns;
        uint patch_x = frame_row - patch_y * parameters.patch_columns;
        route_index = frame * parameters.leaves_per_frame +
                      (patch_y / 8u) * parameters.tile_columns + patch_x / 8u;
    }
    uint route_start = route_offsets[route_index];
    uint route_end = route_offsets[route_index + 1u];
    for (uint route = route_start; route < route_end; ++route) {
        uint encoded = route_entries[route];
        bool is_summary = (encoded & kH3SummaryBit) != 0u;
        uint index = encoded & ~kH3SummaryBit;
        uint key_offset = (index * kH3HeadCount + head) * kH3HeadHalf4 + lane;
        float4 key = is_summary ? float4(summary_keys[key_offset])
                                : float4(keys[key_offset]);
        float4 value = is_summary ? float4(summary_values[key_offset])
                                  : float4(values[key_offset]);
        float partial = dot(query, key);
        float score = simd_sum(partial) * kH3AttentionScale;
        if (is_summary) score += summary_log_counts[index];
        h3_online_attention_update(score, value, maximum, denominator,
                                   accumulator);
    }
    output[query_offset] = half4(accumulator / denominator);
}

kernel void minimax_h3_hierarchical_attention_mma(
    device const half *queries [[buffer(0)]],
    device const half *keys [[buffer(1)]],
    device const half *values [[buffer(2)]],
    device const half *summary_keys [[buffer(3)]],
    device const half *summary_values [[buffer(4)]],
    device const float *summary_log_counts [[buffer(5)]],
    device const uint *route_offsets [[buffer(6)]],
    device const uint *route_entries [[buffer(7)]],
    constant H3TreeParameters &parameters [[buffer(8)]],
    device half *output [[buffer(9)]],
    device const H3QueryBlockGPU *query_blocks [[buffer(10)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half query_tile[8 * kH3HeadDim];
    threadgroup half key_tile[8 * kH3HeadDim];
    threadgroup half value_tile[8 * kH3HeadDim];
    threadgroup half probability_tile[8 * 8];
    threadgroup float score_tile[8 * 8];
    threadgroup float output_tile[8 * kH3HeadDim];
    threadgroup float row_maximum[8];
    threadgroup float row_denominator[8];
    threadgroup float key_log_weight[8];
    H3QueryBlockGPU query_block = query_blocks[group.x];
    uint head = group.y;
    uint route_start = route_offsets[query_block.route_index];
    uint route_end = route_offsets[query_block.route_index + 1u];
    uint route_count = route_end - route_start;
    uint candidate_count = parameters.exact_rows + route_count;
    uint candidate_blocks = (candidate_count + 7u) / 8u;

    for (uint linear = lane; linear < 8u * kH3HeadDim; linear += 32u) {
        uint query_in_block = linear / kH3HeadDim;
        uint dimension = linear - query_in_block * kH3HeadDim;
        if (query_in_block < query_block.row_count) {
            uint row = query_block.first_row + query_in_block;
            query_tile[linear] =
                queries[(row * kH3HeadCount + head) * kH3HeadDim + dimension];
        } else {
            query_tile[linear] = 0.0h;
        }
    }
    if (lane < 8u) {
        row_maximum[lane] = -INFINITY;
        row_denominator[lane] = 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    /* First pass: QK tiles establish one FP32 maximum and denominator per
     * query row. Recomputing QK in the second pass is cheaper than storing an
     * 8 x candidate_count score matrix in device memory. */
    for (uint candidate_block = 0u; candidate_block < candidate_blocks;
         ++candidate_block) {
        for (uint linear = lane; linear < 8u * kH3HeadDim; linear += 32u) {
            uint key_in_block = linear / kH3HeadDim;
            uint dimension = linear - key_in_block * kH3HeadDim;
            uint candidate = candidate_block * 8u + key_in_block;
            if (candidate < candidate_count) {
                uint encoded = candidate < parameters.exact_rows
                                   ? candidate
                                   : route_entries[route_start + candidate -
                                                   parameters.exact_rows];
                bool summary = (encoded & kH3SummaryBit) != 0u;
                uint index = encoded & ~kH3SummaryBit;
                uint source =
                    (index * kH3HeadCount + head) * kH3HeadDim + dimension;
                key_tile[linear] = summary ? summary_keys[source] : keys[source];
            } else {
                key_tile[linear] = 0.0h;
            }
        }
        if (lane < 8u) {
            uint candidate = candidate_block * 8u + lane;
            if (candidate < candidate_count &&
                candidate >= parameters.exact_rows) {
                uint encoded = route_entries[route_start + candidate -
                                             parameters.exact_rows];
                bool summary = (encoded & kH3SummaryBit) != 0u;
                uint index = encoded & ~kH3SummaryBit;
                key_log_weight[lane] = summary ? summary_log_counts[index] : 0.0f;
            } else {
                key_log_weight[lane] = 0.0f;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 scores =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint dimension = 0u; dimension < kH3HeadDim; dimension += 8u) {
            simdgroup_half8x8 query_fragment;
            simdgroup_half8x8 key_fragment;
            simdgroup_load(query_fragment, query_tile + dimension, kH3HeadDim);
            simdgroup_load(key_fragment, key_tile + dimension, kH3HeadDim,
                           0u, true);
            simdgroup_multiply_accumulate(scores, query_fragment, key_fragment,
                                          scores);
        }
        simdgroup_store(scores, score_tile, 8u);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < query_block.row_count) {
            uint valid_keys = candidate_count - candidate_block * 8u;
            if (valid_keys > 8u) valid_keys = 8u;
            float block_maximum = -INFINITY;
            for (uint key = 0u; key < valid_keys; ++key) {
                float score = score_tile[lane * 8u + key] * kH3AttentionScale +
                              key_log_weight[key];
                block_maximum = max(block_maximum, score);
            }
            float next_maximum = max(row_maximum[lane], block_maximum);
            float denominator = row_denominator[lane] *
                                exp(row_maximum[lane] - next_maximum);
            for (uint key = 0u; key < valid_keys; ++key) {
                float score = score_tile[lane * 8u + key] * kH3AttentionScale +
                              key_log_weight[key];
                denominator += exp(score - next_maximum);
            }
            row_maximum[lane] = next_maximum;
            row_denominator[lane] = denominator;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_float8x8 accumulators[16];
    for (uint fragment_index = 0u; fragment_index < 16u; ++fragment_index) {
        accumulators[fragment_index] =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    for (uint candidate_block = 0u; candidate_block < candidate_blocks;
         ++candidate_block) {
        for (uint linear = lane; linear < 8u * kH3HeadDim; linear += 32u) {
            uint key_in_block = linear / kH3HeadDim;
            uint dimension = linear - key_in_block * kH3HeadDim;
            uint candidate = candidate_block * 8u + key_in_block;
            if (candidate < candidate_count) {
                uint encoded = candidate < parameters.exact_rows
                                   ? candidate
                                   : route_entries[route_start + candidate -
                                                   parameters.exact_rows];
                bool summary = (encoded & kH3SummaryBit) != 0u;
                uint index = encoded & ~kH3SummaryBit;
                uint source =
                    (index * kH3HeadCount + head) * kH3HeadDim + dimension;
                key_tile[linear] = summary ? summary_keys[source] : keys[source];
                value_tile[linear] =
                    summary ? summary_values[source] : values[source];
            } else {
                key_tile[linear] = 0.0h;
                value_tile[linear] = 0.0h;
            }
        }
        if (lane < 8u) {
            uint candidate = candidate_block * 8u + lane;
            if (candidate < candidate_count &&
                candidate >= parameters.exact_rows) {
                uint encoded = route_entries[route_start + candidate -
                                             parameters.exact_rows];
                bool summary = (encoded & kH3SummaryBit) != 0u;
                uint index = encoded & ~kH3SummaryBit;
                key_log_weight[lane] = summary ? summary_log_counts[index] : 0.0f;
            } else {
                key_log_weight[lane] = 0.0f;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 scores =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint dimension = 0u; dimension < kH3HeadDim; dimension += 8u) {
            simdgroup_half8x8 query_fragment;
            simdgroup_half8x8 key_fragment;
            simdgroup_load(query_fragment, query_tile + dimension, kH3HeadDim);
            simdgroup_load(key_fragment, key_tile + dimension, kH3HeadDim,
                           0u, true);
            simdgroup_multiply_accumulate(scores, query_fragment, key_fragment,
                                          scores);
        }
        simdgroup_store(scores, score_tile, 8u);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint probability = lane; probability < 64u; probability += 32u) {
            uint query_in_block = probability / 8u;
            uint key_in_block = probability & 7u;
            uint candidate = candidate_block * 8u + key_in_block;
            if (query_in_block < query_block.row_count &&
                candidate < candidate_count) {
                float score = score_tile[probability] * kH3AttentionScale +
                              key_log_weight[key_in_block];
                probability_tile[probability] = half(
                    exp(score - row_maximum[query_in_block]) /
                    row_denominator[query_in_block]);
            } else {
                probability_tile[probability] = 0.0h;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 probability_fragment;
        simdgroup_load(probability_fragment, probability_tile, 8u);
        for (uint fragment_index = 0u; fragment_index < 16u; ++fragment_index) {
            simdgroup_half8x8 value_fragment;
            simdgroup_load(value_fragment,
                           value_tile + fragment_index * 8u, kH3HeadDim);
            simdgroup_multiply_accumulate(accumulators[fragment_index],
                                          probability_fragment, value_fragment,
                                          accumulators[fragment_index]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint fragment_index = 0u; fragment_index < 16u; ++fragment_index) {
        simdgroup_store(accumulators[fragment_index],
                        output_tile + fragment_index * 8u, kH3HeadDim);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint linear = lane; linear < query_block.row_count * kH3HeadDim;
         linear += 32u) {
        uint query_in_block = linear / kH3HeadDim;
        uint dimension = linear - query_in_block * kH3HeadDim;
        uint row = query_block.first_row + query_in_block;
        output[(row * kH3HeadCount + head) * kH3HeadDim + dimension] =
            half(output_tile[linear]);
    }
}

kernel void minimax_h3_hierarchical_attention_mma64(
    device const half *queries [[buffer(0)]],
    device const half *keys [[buffer(1)]],
    device const half *values [[buffer(2)]],
    device const half *summary_keys [[buffer(3)]],
    device const half *summary_values [[buffer(4)]],
    device const float *summary_log_counts [[buffer(5)]],
    device const uint *route_offsets [[buffer(6)]],
    device const uint *route_entries [[buffer(7)]],
    constant H3TreeParameters &parameters [[buffer(8)]],
    device half *output [[buffer(9)]],
    device const H3QueryBlockGPU *query_blocks [[buffer(10)]],
    device float *lse_output [[buffer(11)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup half key_tile[8 * kH3HeadDim];
    threadgroup half value_tile[8 * kH3HeadDim];
    threadgroup float score_tiles[8 * 8 * 8];
    threadgroup half probability_tiles[8 * 8 * 8];
    threadgroup float row_maximum[64];
    threadgroup float row_denominator[64];
    threadgroup float key_log_weight[8];
    H3QueryBlockGPU query_block = query_blocks[group.x];
    uint head = group.y;
    uint query_base = simdgroup_index * 8u;
    uint route_start = route_offsets[query_block.route_index];
    uint route_end = route_offsets[query_block.route_index + 1u];
    uint route_count = route_end - route_start;
    uint candidate_count = parameters.exact_rows + route_count;
    uint candidate_blocks = (candidate_count + 7u) / 8u;
    simdgroup_half8x8 query_fragments[16];

    for (uint dimension_fragment = 0u; dimension_fragment < 16u;
         ++dimension_fragment) {
        device const half *query_pointer =
            queries + ((query_block.first_row + query_base) * kH3HeadCount +
                       head) * kH3HeadDim + dimension_fragment * 8u;
        if (query_base < query_block.row_count) {
            simdgroup_load(query_fragments[dimension_fragment], query_pointer,
                           kH3HeadCount * kH3HeadDim);
        } else {
            query_fragments[dimension_fragment] =
                make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
        }
    }
    if (tid < 64u) {
        row_maximum[tid] = -INFINITY;
        row_denominator[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint candidate_block = 0u; candidate_block < candidate_blocks;
         ++candidate_block) {
        for (uint linear = tid; linear < 8u * kH3HeadDim; linear += 256u) {
            uint key_in_block = linear / kH3HeadDim;
            uint dimension = linear - key_in_block * kH3HeadDim;
            uint candidate = candidate_block * 8u + key_in_block;
            if (candidate < candidate_count) {
                uint encoded = candidate < parameters.exact_rows
                                   ? candidate
                                   : route_entries[route_start + candidate -
                                                   parameters.exact_rows];
                bool summary = (encoded & kH3SummaryBit) != 0u;
                uint index = encoded & ~kH3SummaryBit;
                uint source =
                    (index * kH3HeadCount + head) * kH3HeadDim + dimension;
                key_tile[linear] = summary ? summary_keys[source] : keys[source];
            } else {
                key_tile[linear] = 0.0h;
            }
        }
        if (tid < 8u) {
            uint candidate = candidate_block * 8u + tid;
            if (candidate < candidate_count &&
                candidate >= parameters.exact_rows) {
                uint encoded = route_entries[route_start + candidate -
                                             parameters.exact_rows];
                bool summary = (encoded & kH3SummaryBit) != 0u;
                uint index = encoded & ~kH3SummaryBit;
                key_log_weight[tid] = summary ? summary_log_counts[index] : 0.0f;
            } else {
                key_log_weight[tid] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 scores =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint dimension_fragment = 0u; dimension_fragment < 16u;
             ++dimension_fragment) {
            simdgroup_half8x8 key_fragment;
            simdgroup_load(key_fragment,
                           key_tile + dimension_fragment * 8u, kH3HeadDim,
                           0u, true);
            simdgroup_multiply_accumulate(
                scores, query_fragments[dimension_fragment], key_fragment,
                scores);
        }
        threadgroup float *score_tile =
            score_tiles + simdgroup_index * 64u;
        simdgroup_store(scores, score_tile, 8u);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        uint local_query = query_base + lane;
        if (lane < 8u && local_query < query_block.row_count) {
            uint valid_keys = candidate_count - candidate_block * 8u;
            if (valid_keys > 8u) valid_keys = 8u;
            float block_maximum = -INFINITY;
            for (uint key = 0u; key < valid_keys; ++key) {
                float score = score_tile[lane * 8u + key] * kH3AttentionScale +
                              key_log_weight[key];
                block_maximum = max(block_maximum, score);
            }
            float next_maximum = max(row_maximum[local_query], block_maximum);
            float denominator = row_denominator[local_query] *
                                exp(row_maximum[local_query] - next_maximum);
            for (uint key = 0u; key < valid_keys; ++key) {
                float score = score_tile[lane * 8u + key] * kH3AttentionScale +
                              key_log_weight[key];
                denominator += exp(score - next_maximum);
            }
            row_maximum[local_query] = next_maximum;
            row_denominator[local_query] = denominator;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane < 8u) {
        uint local_query = query_base + lane;
        if (local_query < query_block.row_count) {
            uint row = query_block.first_row + local_query;
            lse_output[row * kH3HeadCount + head] =
                row_maximum[local_query] + log(row_denominator[local_query]);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_half8x8 accumulators[16];
    for (uint fragment_index = 0u; fragment_index < 16u; ++fragment_index) {
        accumulators[fragment_index] =
            make_filled_simdgroup_matrix<half, 8, 8>(0.0h);
    }
    for (uint candidate_block = 0u; candidate_block < candidate_blocks;
         ++candidate_block) {
        for (uint linear = tid; linear < 8u * kH3HeadDim; linear += 256u) {
            uint key_in_block = linear / kH3HeadDim;
            uint dimension = linear - key_in_block * kH3HeadDim;
            uint candidate = candidate_block * 8u + key_in_block;
            if (candidate < candidate_count) {
                uint encoded = candidate < parameters.exact_rows
                                   ? candidate
                                   : route_entries[route_start + candidate -
                                                   parameters.exact_rows];
                bool summary = (encoded & kH3SummaryBit) != 0u;
                uint index = encoded & ~kH3SummaryBit;
                uint source =
                    (index * kH3HeadCount + head) * kH3HeadDim + dimension;
                key_tile[linear] = summary ? summary_keys[source] : keys[source];
                value_tile[linear] =
                    summary ? summary_values[source] : values[source];
            } else {
                key_tile[linear] = 0.0h;
                value_tile[linear] = 0.0h;
            }
        }
        if (tid < 8u) {
            uint candidate = candidate_block * 8u + tid;
            if (candidate < candidate_count &&
                candidate >= parameters.exact_rows) {
                uint encoded = route_entries[route_start + candidate -
                                             parameters.exact_rows];
                bool summary = (encoded & kH3SummaryBit) != 0u;
                uint index = encoded & ~kH3SummaryBit;
                key_log_weight[tid] = summary ? summary_log_counts[index] : 0.0f;
            } else {
                key_log_weight[tid] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 scores =
            make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        for (uint dimension_fragment = 0u; dimension_fragment < 16u;
             ++dimension_fragment) {
            simdgroup_half8x8 key_fragment;
            simdgroup_load(key_fragment,
                           key_tile + dimension_fragment * 8u, kH3HeadDim,
                           0u, true);
            simdgroup_multiply_accumulate(
                scores, query_fragments[dimension_fragment], key_fragment,
                scores);
        }
        threadgroup float *score_tile =
            score_tiles + simdgroup_index * 64u;
        simdgroup_store(scores, score_tile, 8u);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup half *probability_tile =
            probability_tiles + simdgroup_index * 64u;
        for (uint probability = lane; probability < 64u; probability += 32u) {
            uint query_in_simdgroup = probability / 8u;
            uint key_in_block = probability & 7u;
            uint local_query = query_base + query_in_simdgroup;
            uint candidate = candidate_block * 8u + key_in_block;
            if (local_query < query_block.row_count &&
                candidate < candidate_count) {
                float score = score_tile[probability] * kH3AttentionScale +
                              key_log_weight[key_in_block];
                probability_tile[probability] = half(
                    exp(score - row_maximum[local_query]) /
                    row_denominator[local_query]);
            } else {
                probability_tile[probability] = 0.0h;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 probability_fragment;
        simdgroup_load(probability_fragment, probability_tile, 8u);
        for (uint fragment_index = 0u; fragment_index < 16u;
             ++fragment_index) {
            simdgroup_half8x8 value_fragment;
            simdgroup_load(value_fragment,
                           value_tile + fragment_index * 8u, kH3HeadDim);
            simdgroup_multiply_accumulate(
                accumulators[fragment_index], probability_fragment,
                value_fragment, accumulators[fragment_index]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (query_base < query_block.row_count) {
        device half *output_pointer =
            output + ((query_block.first_row + query_base) * kH3HeadCount +
                      head) * kH3HeadDim;
        for (uint fragment_index = 0u; fragment_index < 16u;
             ++fragment_index) {
            simdgroup_store(accumulators[fragment_index],
                            output_pointer + fragment_index * 8u,
                            kH3HeadCount * kH3HeadDim);
        }
    }
}

kernel void minimax_h3_copy_first_output(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index < kH3HeadDim) output[index] = input[index];
}
