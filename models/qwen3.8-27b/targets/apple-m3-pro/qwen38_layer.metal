#include <metal_stdlib>
using namespace metal;

constant uint kHidden = 5120;
constant uint kGroupSizeLayer = 64;
constant uint kDeltaInputRows = 16480;
constant uint kDeltaInputGroups = 80;
constant uint kDeltaQkvRows = 10240;
constant uint kDeltaZOffset = 10240;
constant uint kDeltaAOffset = 16384;
constant uint kDeltaBOffset = 16432;
constant uint kDeltaOutputGroups = 96;
constant uint kMlpDownGroups = 272;
constant float kRmsEpsilon = 1.0e-6f;

struct Q4LayerMeta {
    half scale;
    half bias;
};

kernel void qwen38_rmsnorm_f16_to_f16(
    device const half *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partials[256];
    float sum = 0.0f;
    for (uint index = tid; index < kHidden; index += 256) {
        float value = float(input[index]);
        sum += value * value;
    }
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(partials[0] / float(kHidden) + kRmsEpsilon);
    for (uint index = tid; index < kHidden; index += 256) {
        output[index] = half(float(input[index]) * inv_rms * weight[index]);
    }
}

kernel void qwen38_rmsnorm_f32_to_f16(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partials[256];
    float sum = 0.0f;
    for (uint index = tid; index < kHidden; index += 256) {
        float value = input[index];
        sum += value * value;
    }
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) {
            partials[tid] += partials[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms = rsqrt(partials[0] / float(kHidden) + kRmsEpsilon);
    for (uint index = tid; index < kHidden; index += 256) {
        output[index] = half(input[index] * inv_rms * weight[index]);
    }
}

kernel void qwen38_q4_delta_inputs(
    device const half *input [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4LayerMeta *metadata [[buffer(2)]],
    device float *output [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id.x * simdgroups_per_group + simdgroup_index;
    if (row >= kDeltaInputRows) {
        return;
    }
    float partial = 0.0f;
    for (uint group = 0; group < kDeltaInputGroups; ++group) {
        uint block = row * kDeltaInputGroups + group;
        uchar bits = quants[block * 32 + lane];
        Q4LayerMeta meta = metadata[block];
        float2 quant = float2(bits & 0x0f, bits >> 4);
        device const half2 *activation =
            reinterpret_cast<device const half2 *>(
                input + group * kGroupSizeLayer);
        partial += dot(float(meta.scale) * quant + float(meta.bias),
                       float2(activation[lane]));
    }
    float reduced = simd_sum(partial);
    if (lane == 0) {
        output[row] = reduced;
    }
}

kernel void qwen38_delta_causal_conv_silu(
    device const float *mixed_qkv [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *state [[buffer(2)]],
    device float *output [[buffer(3)]],
    uint channel [[thread_position_in_grid]]) {
    if (channel >= kDeltaQkvRows) {
        return;
    }
    uint base = channel * 4;
    state[base] = state[base + 1];
    state[base + 1] = state[base + 2];
    state[base + 2] = state[base + 3];
    state[base + 3] = mixed_qkv[channel];
    float value = state[base] * weights[base] +
                  state[base + 1] * weights[base + 1] +
                  state[base + 2] * weights[base + 2] +
                  state[base + 3] * weights[base + 3];
    output[channel] = value / (1.0f + exp(-value));
}

kernel void qwen38_delta_prepare(
    device const float *convolved_qkv [[buffer(0)]],
    device const float *projected [[buffer(1)]],
    device const float *a_log [[buffer(2)]],
    device const float *dt_bias [[buffer(3)]],
    device float *query [[buffer(4)]],
    device float *key [[buffer(5)]],
    device float *value [[buffer(6)]],
    device float *decay [[buffer(7)]],
    device float *beta [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float q_squared[128];
    threadgroup float k_squared[128];
    uint value_head = group_id.x;
    uint key_head = value_head / 3;
    float raw_q = convolved_qkv[key_head * 128 + tid];
    float raw_k = convolved_qkv[2048 + key_head * 128 + tid];
    q_squared[tid] = raw_q * raw_q;
    k_squared[tid] = raw_k * raw_k;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64; stride != 0; stride >>= 1) {
        if (tid < stride) {
            q_squared[tid] += q_squared[tid + stride];
            k_squared[tid] += k_squared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint output_index = value_head * 128 + tid;
    query[output_index] = raw_q *
        rsqrt(q_squared[0] + 128.0e-6f) * rsqrt(128.0f);
    key[output_index] = raw_k * rsqrt(k_squared[0] + 128.0e-6f);
    value[output_index] = convolved_qkv[4096 + output_index];
    if (tid == 0) {
        float a = projected[kDeltaAOffset + value_head] +
                  dt_bias[value_head];
        float softplus = max(a, 0.0f) + log(1.0f + exp(-abs(a)));
        float g = -exp(a_log[value_head]) * softplus;
        decay[value_head] = exp(g);
        float b = projected[kDeltaBOffset + value_head];
        beta[value_head] = 1.0f / (1.0f + exp(-b));
    }
}

kernel void qwen38_delta_gated_rmsnorm(
    device const float *core [[buffer(0)]],
    device const float *projected [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device float *output [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float squared[128];
    uint head = group_id.x;
    uint index = head * 128 + tid;
    float value = core[index];
    squared[tid] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64; stride != 0; stride >>= 1) {
        if (tid < stride) {
            squared[tid] += squared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float normalized = value * rsqrt(squared[0] / 128.0f +
                                     kRmsEpsilon) * weight[tid];
    float z = projected[kDeltaZOffset + index];
    float silu_z = z / (1.0f + exp(-z));
    output[index] = normalized * silu_z;
}

kernel void qwen38_q4_delta_output_residual(
    device const float *input [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4LayerMeta *metadata [[buffer(2)]],
    device const half *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id.x * simdgroups_per_group + simdgroup_index;
    if (row >= kHidden) {
        return;
    }
    float partial = 0.0f;
    for (uint group = 0; group < kDeltaOutputGroups; ++group) {
        uint block = row * kDeltaOutputGroups + group;
        uchar bits = quants[block * 32 + lane];
        Q4LayerMeta meta = metadata[block];
        float2 quant = float2(bits & 0x0f, bits >> 4);
        device const float2 *activation =
            reinterpret_cast<device const float2 *>(
                input + group * kGroupSizeLayer);
        partial += dot(float(meta.scale) * quant + float(meta.bias),
                       activation[lane]);
    }
    float reduced = simd_sum(partial);
    if (lane == 0) {
        output[row] = reduced + float(residual[row]);
    }
}

kernel void qwen38_q4_mlp_down_residual_f32(
    device const float *input [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4LayerMeta *metadata [[buffer(2)]],
    device const float *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id.x * simdgroups_per_group + simdgroup_index;
    if (row >= kHidden) {
        return;
    }
    float partial = 0.0f;
    for (uint group = 0; group < kMlpDownGroups; ++group) {
        uint block = row * kMlpDownGroups + group;
        uchar bits = quants[block * 32 + lane];
        Q4LayerMeta meta = metadata[block];
        float2 quant = float2(bits & 0x0f, bits >> 4);
        device const float2 *activation =
            reinterpret_cast<device const float2 *>(
                input + group * kGroupSizeLayer);
        partial += dot(float(meta.scale) * quant + float(meta.bias),
                       activation[lane]);
    }
    float reduced = simd_sum(partial);
    if (lane == 0) {
        output[row] = reduced + residual[row];
    }
}
