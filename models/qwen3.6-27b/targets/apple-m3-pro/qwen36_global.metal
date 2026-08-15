#include <metal_stdlib>
using namespace metal;

constant uint kGlobalHidden = 5120;
constant uint kGlobalVocab = 248320;
constant uint kGlobalGroups = 80;

struct Q4GlobalMeta {
    half scale;
    half bias;
};

kernel void qwen36_q4_embedding_lookup(
    device const uchar *quants [[buffer(0)]],
    device const Q4GlobalMeta *metadata [[buffer(1)]],
    constant uint &token_id [[buffer(2)]],
    device half *output [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= kGlobalHidden || token_id >= kGlobalVocab) return;
    uint group = index / 64;
    uint within = index - group * 64;
    uint block = token_id * kGlobalGroups + group;
    uchar bits = quants[block * 32 + (within >> 1)];
    uint quant = (within & 1u) == 0 ? bits & 0x0f : bits >> 4;
    Q4GlobalMeta meta = metadata[block];
    output[index] = half(float(meta.scale) * float(quant) +
                         float(meta.bias));
}

kernel void qwen36_f32_to_f16_hidden(
    device const float *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index < kGlobalHidden) output[index] = half(input[index]);
}

kernel void qwen36_q4_lm_head(
    device const half *input [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4GlobalMeta *metadata [[buffer(2)]],
    device float *logits [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id.x * simdgroups_per_group + simdgroup_index;
    if (row >= kGlobalVocab) return;
    float partial = 0.0f;
    for (uint group = 0; group < kGlobalGroups; ++group) {
        uint block = row * kGlobalGroups + group;
        uchar bits = quants[block * 32 + lane];
        Q4GlobalMeta meta = metadata[block];
        float2 quant = float2(bits & 0x0f, bits >> 4);
        device const half2 *activation =
            reinterpret_cast<device const half2 *>(input + group * 64);
        partial += dot(float(meta.scale) * quant + float(meta.bias),
                       float2(activation[lane]));
    }
    float reduced = simd_sum(partial);
    if (lane == 0) logits[row] = reduced;
}

/* First-occurrence argmax over the masked head logits, written into a
 * uint32 slot. Matches the CPU loop: strict greater-than keeps the
 * lowest index among equal maxima. Lets a draft token feed the next
 * chained MTP pass without a CPU round trip. */
kernel void qwen36_argmax(
    device const float *logits [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant uint &slot [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float best_value[256];
    threadgroup uint best_index[256];
    float value = -INFINITY;
    uint index = 0;
    for (uint i = tid; i < 248320; i += 256) {
        float candidate = logits[i];
        if (candidate > value) {
            value = candidate;
            index = i;
        }
    }
    best_value[tid] = value;
    best_index[tid] = index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) {
            float other = best_value[tid + stride];
            uint other_index = best_index[tid + stride];
            if (other > best_value[tid] ||
                (other == best_value[tid] &&
                 other_index < best_index[tid])) {
                best_value[tid] = other;
                best_index[tid] = other_index;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) output[slot] = best_index[0];
}
