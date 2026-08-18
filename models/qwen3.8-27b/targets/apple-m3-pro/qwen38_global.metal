#include <metal_stdlib>
using namespace metal;

constant uint kGlobalHidden = 5120;
constant uint kGlobalVocab = 248320;
constant uint kGlobalGroups = 80;

struct Q4GlobalMeta {
    half scale;
    half bias;
};

kernel void qwen38_q4_embedding_lookup(
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

kernel void qwen38_f32_to_f16_hidden(
    device const float *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index < kGlobalHidden) output[index] = half(input[index]);
}

kernel void qwen38_q4_lm_head(
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

/* Draft-vocabulary head rows outside the static low-id prefix: one
 * simdgroup per gathered row id, writing its logit after the prefix.
 * Only the draft chain uses the restricted head; the verify keeps the
 * full vocabulary, so a missing rare token costs one rejected draft,
 * never a wrong output. */
kernel void qwen38_q4_lm_head_gathered(
    device const half *input [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4GlobalMeta *metadata [[buffer(2)]],
    device const uint *ids [[buffer(3)]],
    constant uint &count [[buffer(4)]],
    constant uint &base [[buffer(5)]],
    device float *logits [[buffer(6)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint slot = group_id.x * simdgroups_per_group + simdgroup_index;
    if (slot >= count) return;
    uint row = ids[slot];
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
    if (lane == 0) logits[base + slot] = reduced;
}

/* Argmax over the restricted draft head: the static prefix rows plus
 * the gathered extras, mapping an extra slot back to its token id. */
kernel void qwen38_argmax_limited(
    device const float *logits [[buffer(0)]],
    device uint *output [[buffer(1)]],
    constant uint &slot [[buffer(2)]],
    constant uint &limit [[buffer(3)]],
    constant uint &extra_count [[buffer(4)]],
    device const uint *extra_ids [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float best_value[256];
    threadgroup uint best_index[256];
    uint bound = limit + extra_count;
    float value = -INFINITY;
    uint index = 0;
    for (uint i = tid; i < bound; i += 256) {
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
    if (tid == 0) {
        uint best = best_index[0];
        output[slot] = best < limit ? best : extra_ids[best - limit];
    }
}

/* First-occurrence argmax over the masked head logits, written into a
 * uint32 slot. Matches the CPU loop: strict greater-than keeps the
 * lowest index among equal maxima. Lets a draft token feed the next
 * chained MTP pass without a CPU round trip. */
kernel void qwen38_argmax(
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
