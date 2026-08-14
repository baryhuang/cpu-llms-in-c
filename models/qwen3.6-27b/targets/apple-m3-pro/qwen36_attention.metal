#include <metal_stdlib>
using namespace metal;

constant uint kAttentionHidden = 5120;
constant uint kAttentionInputRows = 14336;
constant uint kAttentionGroups = 80;
constant uint kAttentionKOffset = 12288;
constant uint kAttentionVOffset = 13312;
constant uint kAttentionQHeads = 24;
constant uint kAttentionKVHeads = 4;
constant uint kAttentionRotarySize = 64;
constant uint kAttentionOutputGroups = 96;
constant float kAttentionRopeTheta = 10000000.0f;
constant float kAttentionRmsEpsilon = 1.0e-6f;

struct Q4AttentionMeta {
    half scale;
    half bias;
};

struct AttentionDecodeParameters {
    uint position;
    uint context_length;
    uint cache_capacity;
    uint reserved;
};

kernel void qwen36_q4_attention_inputs(
    device const half *input [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4AttentionMeta *metadata [[buffer(2)]],
    device float *output [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id.x * simdgroups_per_group + simdgroup_index;
    if (row >= kAttentionInputRows) return;
    float partial = 0.0f;
    for (uint group = 0; group < kAttentionGroups; ++group) {
        uint block = row * kAttentionGroups + group;
        uchar bits = quants[block * 32 + lane];
        Q4AttentionMeta meta = metadata[block];
        float2 quant = float2(bits & 0x0f, bits >> 4);
        device const half2 *activation =
            reinterpret_cast<device const half2 *>(input + group * 64);
        partial += dot(float(meta.scale) * quant + float(meta.bias),
                       float2(activation[lane]));
    }
    float reduced = simd_sum(partial);
    if (lane == 0) output[row] = reduced;
}

inline float apply_rope_component(threadgroup const float *values,
                                  uint dimension, uint position) {
    if (dimension >= kAttentionRotarySize) return values[dimension];
    uint frequency = dimension & 31u;
    float exponent = -2.0f * float(frequency) /
                     float(kAttentionRotarySize);
    float angle = float(position) * pow(kAttentionRopeTheta, exponent);
    float c = cos(angle);
    float s = sin(angle);
    if (dimension < 32) {
        return values[dimension] * c - values[dimension + 32] * s;
    }
    return values[dimension] * c + values[dimension - 32] * s;
}

kernel void qwen36_attention_prepare_query(
    device const float *projected [[buffer(0)]],
    device const float *norm_weight [[buffer(1)]],
    constant AttentionDecodeParameters &parameters [[buffer(2)]],
    device float *query [[buffer(3)]],
    device float *query_gate [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float values[256];
    threadgroup float squared[256];
    uint head = group_id.x;
    uint projection_base = head * 512;
    float value = projected[projection_base + tid];
    values[tid] = value;
    squared[tid] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) squared[tid] += squared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float normalized = value *
        rsqrt(squared[0] / 256.0f + kAttentionRmsEpsilon) *
        norm_weight[tid];
    values[tid] = normalized;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint index = head * 256 + tid;
    query[index] = apply_rope_component(values, tid, parameters.position);
    float gate = projected[projection_base + 256 + tid];
    query_gate[index] = 1.0f / (1.0f + exp(-gate));
}

kernel void qwen36_attention_prepare_key_value(
    device const float *projected [[buffer(0)]],
    device const float *norm_weight [[buffer(1)]],
    constant AttentionDecodeParameters &parameters [[buffer(2)]],
    device float *key_cache [[buffer(3)]],
    device float *value_cache [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float values[256];
    threadgroup float squared[256];
    uint head = group_id.x;
    float value = projected[kAttentionKOffset + head * 256 + tid];
    values[tid] = value;
    squared[tid] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) squared[tid] += squared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float normalized = value *
        rsqrt(squared[0] / 256.0f + kAttentionRmsEpsilon) *
        norm_weight[tid];
    values[tid] = normalized;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint cache_index =
        (parameters.position * kAttentionKVHeads + head) * 256 + tid;
    key_cache[cache_index] =
        apply_rope_component(values, tid, parameters.position);
    value_cache[cache_index] =
        projected[kAttentionVOffset + head * 256 + tid];
}

kernel void qwen36_attention_scores(
    device const float *query [[buffer(0)]],
    device const float *key_cache [[buffer(1)]],
    constant AttentionDecodeParameters &parameters [[buffer(2)]],
    device float *scores [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint flat = group_id.x * simdgroups_per_group + simdgroup_index;
    uint score_count = kAttentionQHeads * parameters.context_length;
    if (flat >= score_count) return;
    uint q_head = flat / parameters.context_length;
    uint position = flat - q_head * parameters.context_length;
    uint kv_head = q_head / (kAttentionQHeads / kAttentionKVHeads);
    uint query_base = q_head * 256;
    uint key_base = (position * kAttentionKVHeads + kv_head) * 256;
    float partial = 0.0f;
    for (uint index = lane; index < 256; index += 32) {
        partial += query[query_base + index] *
                   key_cache[key_base + index];
    }
    float score = simd_sum(partial) * (1.0f / 16.0f);
    if (lane == 0) {
        scores[q_head * parameters.cache_capacity + position] = score;
    }
}

kernel void qwen36_attention_softmax_value_gate(
    device const float *scores [[buffer(0)]],
    device const float *value_cache [[buffer(1)]],
    device const float *query_gate [[buffer(2)]],
    constant AttentionDecodeParameters &parameters [[buffer(3)]],
    device float *output [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float reduction[256];
    uint q_head = group_id.x;
    uint score_base = q_head * parameters.cache_capacity;
    float local_max = -INFINITY;
    for (uint position = tid; position < parameters.context_length;
         position += 256) {
        local_max = max(local_max, scores[score_base + position]);
    }
    reduction[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) reduction[tid] =
            max(reduction[tid], reduction[tid + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float maximum = reduction[0];
    float local_sum = 0.0f;
    for (uint position = tid; position < parameters.context_length;
         position += 256) {
        local_sum += exp(scores[score_base + position] - maximum);
    }
    reduction[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float denominator = reduction[0];
    uint kv_head = q_head / (kAttentionQHeads / kAttentionKVHeads);
    float value = 0.0f;
    for (uint position = 0; position < parameters.context_length;
         ++position) {
        uint cache_index =
            (position * kAttentionKVHeads + kv_head) * 256 + tid;
        float probability =
            exp(scores[score_base + position] - maximum) / denominator;
        value += probability * value_cache[cache_index];
    }
    uint output_index = q_head * 256 + tid;
    output[output_index] = value * query_gate[output_index];
}

kernel void qwen36_q4_attention_output_residual(
    device const float *input [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4AttentionMeta *metadata [[buffer(2)]],
    device const half *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id.x * simdgroups_per_group + simdgroup_index;
    if (row >= kAttentionHidden) return;
    float partial = 0.0f;
    for (uint group = 0; group < kAttentionOutputGroups; ++group) {
        uint block = row * kAttentionOutputGroups + group;
        uchar bits = quants[block * 32 + lane];
        Q4AttentionMeta meta = metadata[block];
        float2 quant = float2(bits & 0x0f, bits >> 4);
        device const float2 *activation =
            reinterpret_cast<device const float2 *>(input + group * 64);
        partial += dot(float(meta.scale) * quant + float(meta.bias),
                       activation[lane]);
    }
    float reduced = simd_sum(partial);
    if (lane == 0) output[row] = reduced + float(residual[row]);
}
