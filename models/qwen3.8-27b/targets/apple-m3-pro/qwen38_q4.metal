#include <metal_stdlib>
using namespace metal;

constant uint kRows = 17408;
constant uint kGroupSize = 64;
constant uint kGroupsPerRow = 80;
constant uint kDownRows = 5120;
constant uint kDownGroupsPerRow = 272;

struct Q4Meta {
    half scale;
    half bias;
};

struct Q4InterleavedGroup {
    uchar q[32];
    half scale;
    half bias;
};

inline float q4_interleaved_row_dot(device const Q4InterleavedGroup *weights,
                                    device const half *x,
                                    uint row,
                                    uint lane) {
    float partial = 0.0f;
    for (uint group = 0; group < kGroupsPerRow; ++group) {
        device const Q4InterleavedGroup &packed =
            weights[row * kGroupsPerRow + group];
        float scale = float(packed.scale);
        float bias = float(packed.bias);
        uint group_base = group * kGroupSize;
        uchar bits = packed.q[lane];
        uint index = lane << 1;
        float weight0 = scale * float(bits & 0x0f) + bias;
        float weight1 = scale * float(bits >> 4) + bias;
        partial += weight0 * float(x[group_base + index]);
        partial += weight1 * float(x[group_base + index + 1]);
    }
    return simd_sum(partial);
}

kernel void qwen38_q4_interleaved_matvec(
    device const half *x [[buffer(0)]],
    device const Q4InterleavedGroup *weights [[buffer(1)]],
    device float *output [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id.x * simdgroups_per_group + simdgroup_index;
    if (row >= kRows) {
        return;
    }
    float value = q4_interleaved_row_dot(weights, x, row, lane);
    if (lane == 0) {
        output[row] = value;
    }
}

inline float q4_row_dot(device const uchar *quants,
                        device const Q4Meta *metadata,
                        device const half *x,
                        uint row,
                        uint lane) {
    float partial = 0.0f;
    for (uint group = 0; group < kGroupsPerRow; ++group) {
        uint block = row * kGroupsPerRow + group;
        device const uchar *packed = quants + block * 32;
        Q4Meta meta = metadata[block];
        uint group_base = group * kGroupSize;
        uchar bits = packed[lane];
        float2 quant = float2(bits & 0x0f, bits >> 4);
        device const half2 *x2 =
            reinterpret_cast<device const half2 *>(x + group_base);
        partial += dot(float(meta.scale) * quant + float(meta.bias),
                       float2(x2[lane]));
    }
    return simd_sum(partial);
}

inline float2 q4_row_dot_pair(device const uchar *gate_quants,
                              device const Q4Meta *gate_metadata,
                              device const uchar *up_quants,
                              device const Q4Meta *up_metadata,
                              device const half *x,
                              uint row,
                              uint lane) {
    float gate_partial = 0.0f;
    float up_partial = 0.0f;
    for (uint group = 0; group < kGroupsPerRow; ++group) {
        uint block = row * kGroupsPerRow + group;
        device const uchar *gate_packed = gate_quants + block * 32;
        device const uchar *up_packed = up_quants + block * 32;
        Q4Meta gate_meta = gate_metadata[block];
        Q4Meta up_meta = up_metadata[block];
        uchar gate_bits = gate_packed[lane];
        uchar up_bits = up_packed[lane];
        float2 gate_quant = float2(gate_bits & 0x0f, gate_bits >> 4);
        float2 up_quant = float2(up_bits & 0x0f, up_bits >> 4);
        device const half2 *x2 =
            reinterpret_cast<device const half2 *>(x + group * kGroupSize);
        float2 activation = float2(x2[lane]);
        gate_partial += dot(float(gate_meta.scale) * gate_quant +
                            float(gate_meta.bias), activation);
        up_partial += dot(float(up_meta.scale) * up_quant +
                          float(up_meta.bias), activation);
    }
    return float2(simd_sum(gate_partial), simd_sum(up_partial));
}

kernel void qwen38_q4_matvec(device const half *x [[buffer(0)]],
                            device const uchar *quants [[buffer(1)]],
                            device const Q4Meta *metadata [[buffer(2)]],
                            device float *output [[buffer(3)]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint simdgroup_index [[simdgroup_index_in_threadgroup]],
                            uint simdgroups_per_group [[simdgroups_per_threadgroup]],
                            uint3 group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id.x * simdgroups_per_group + simdgroup_index;
    if (row >= kRows) {
        return;
    }
    float value = q4_row_dot(quants, metadata, x, row, lane);
    if (lane == 0) {
        output[row] = value;
    }
}

kernel void qwen38_q4_gate_up_silu(device const half *x [[buffer(0)]],
                                  device const uchar *gate_quants [[buffer(1)]],
                                  device const Q4Meta *gate_metadata [[buffer(2)]],
                                  device const uchar *up_quants [[buffer(3)]],
                                  device const Q4Meta *up_metadata [[buffer(4)]],
                                  device float *output [[buffer(5)]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint simdgroup_index [[simdgroup_index_in_threadgroup]],
                                  uint simdgroups_per_group [[simdgroups_per_threadgroup]],
                                  uint3 group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id.x * simdgroups_per_group + simdgroup_index;
    if (row >= kRows) {
        return;
    }
    float2 gate_up = q4_row_dot_pair(gate_quants, gate_metadata,
                                     up_quants, up_metadata, x, row, lane);
    if (lane == 0) {
        float gate = gate_up.x;
        float up = gate_up.y;
        float silu = gate / (1.0f + exp(-gate));
        output[row] = silu * up;
    }
}

kernel void qwen38_silu_mul(device const float *gate [[buffer(0)]],
                           device const float *up [[buffer(1)]],
                           device float *output [[buffer(2)]],
                           uint index [[thread_position_in_grid]]) {
    if (index < kRows) {
        float value = gate[index];
        output[index] = (value / (1.0f + exp(-value))) * up[index];
    }
}

kernel void qwen38_q4_down_residual(device const float *intermediate [[buffer(0)]],
                                    device const uchar *quants [[buffer(1)]],
                                    device const Q4Meta *metadata [[buffer(2)]],
                                    device const half *residual [[buffer(3)]],
                                    device float *output [[buffer(4)]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
                                    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
                                    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id.x * simdgroups_per_group + simdgroup_index;
    if (row >= kDownRows) {
        return;
    }
    float partial = 0.0f;
    for (uint group = 0; group < kDownGroupsPerRow; ++group) {
        uint block = row * kDownGroupsPerRow + group;
        uchar bits = quants[block * 32 + lane];
        Q4Meta meta = metadata[block];
        float2 quant = float2(bits & 0x0f, bits >> 4);
        device const float2 *activation =
            reinterpret_cast<device const float2 *>(intermediate + group * 64);
        partial += dot(float(meta.scale) * quant + float(meta.bias),
                       activation[lane]);
    }
    float reduced = simd_sum(partial);
    if (lane == 0) {
        output[row] = reduced + float(residual[row]);
    }
}
