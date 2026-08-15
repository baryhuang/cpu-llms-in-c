#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

/* Batched prompt prefill for the pinned Qwen3.6-27B graph. The batch size is
 * a function constant so each shape bucket (S16, S32) compiles to a fully
 * unrolled pipeline. Every output element uses the same per-element loop
 * order and reduction shape as the one-token decode kernels, so a prefilled
 * prompt must produce bitwise-identical layer state and downstream tokens. */

constant uint kBatch [[function_constant(0)]];
constant uint kPrefillMaxBatch = 32;

constant uint kPrefillHidden = 5120;
constant uint kPrefillVocab = 248320;
constant uint kPrefillEmbeddingGroups = 80;
constant uint kPrefillQkvRows = 10240;
constant uint kPrefillDeltaStride = 16480;
constant uint kPrefillDeltaZOffset = 10240;
constant uint kPrefillDeltaAOffset = 16384;
constant uint kPrefillDeltaBOffset = 16432;
constant uint kPrefillDeltaHeads = 48;
constant uint kPrefillDeltaHeadSize = 128;
constant uint kPrefillMixerWidth = 6144;
constant uint kPrefillAttentionStride = 14336;
constant uint kPrefillAttentionKOffset = 12288;
constant uint kPrefillAttentionVOffset = 13312;
constant uint kPrefillQHeads = 24;
constant uint kPrefillKVHeads = 4;
constant uint kPrefillRotarySize = 64;
constant uint kPrefillMlpWidth = 17408;
constant float kPrefillRopeTheta = 10000000.0f;
constant float kPrefillRmsEpsilon = 1.0e-6f;

struct Q4PrefillMeta {
    half scale;
    half bias;
};

struct PrefillGemmParams {
    uint rows;
    uint groups_per_row;
};

struct PrefillAttentionParams {
    uint start_position;
    uint cache_capacity;
};

kernel void qwen36_prefill_embedding(
    device const uchar *quants [[buffer(0)]],
    device const Q4PrefillMeta *metadata [[buffer(1)]],
    device const uint *token_ids [[buffer(2)]],
    device half *output [[buffer(3)]],
    uint2 position [[thread_position_in_grid]]) {
    uint index = position.x;
    uint s = position.y;
    if (index >= kPrefillHidden || s >= kBatch) return;
    uint token_id = token_ids[s];
    if (token_id >= kPrefillVocab) return;
    uint group = index / 64;
    uint within = index - group * 64;
    uint block = token_id * kPrefillEmbeddingGroups + group;
    uchar bits = quants[block * 32 + (within >> 1)];
    uint quant = (within & 1u) == 0 ? bits & 0x0f : bits >> 4;
    Q4PrefillMeta meta = metadata[block];
    output[s * kPrefillHidden + index] =
        half(float(meta.scale) * float(quant) + float(meta.bias));
}

kernel void qwen36_prefill_rmsnorm_f16(
    device const half *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float partials[256];
    uint s = group_id.x;
    device const half *in = input + s * kPrefillHidden;
    device half *out = output + s * kPrefillHidden;
    float sum = 0.0f;
    for (uint index = tid; index < kPrefillHidden; index += 256) {
        float value = float(in[index]);
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
    float inv_rms = rsqrt(partials[0] / float(kPrefillHidden) +
                          kPrefillRmsEpsilon);
    for (uint index = tid; index < kPrefillHidden; index += 256) {
        out[index] = half(float(in[index]) * inv_rms * weight[index]);
    }
}

kernel void qwen36_prefill_rmsnorm_f32(
    device const float *input [[buffer(0)]],
    device const float *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float partials[256];
    uint s = group_id.x;
    device const float *in = input + s * kPrefillHidden;
    device half *out = output + s * kPrefillHidden;
    float sum = 0.0f;
    for (uint index = tid; index < kPrefillHidden; index += 256) {
        float value = in[index];
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
    float inv_rms = rsqrt(partials[0] / float(kPrefillHidden) +
                          kPrefillRmsEpsilon);
    for (uint index = tid; index < kPrefillHidden; index += 256) {
        out[index] = half(in[index] * inv_rms * weight[index]);
    }
}

kernel void qwen36_prefill_convert_hidden(
    device const float *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]) {
    uint index = position.x;
    uint s = position.y;
    if (index >= kPrefillHidden || s >= kBatch) return;
    output[s * kPrefillHidden + index] =
        half(input[s * kPrefillHidden + index]);
}

/* One simdgroup owns one output row and accumulates all batch positions, so
 * each Q4 weight group is read once per chunk instead of once per token. */
kernel void qwen36_prefill_q4_gemm_f16(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant PrefillGemmParams &p [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id.x * simdgroups_per_group + simdgroup_index;
    if (row >= p.rows) return;
    uint columns = p.groups_per_row * 64;
    float partial[kPrefillMaxBatch];
    for (uint s = 0; s < kBatch; ++s) partial[s] = 0.0f;
    for (uint group = 0; group < p.groups_per_row; ++group) {
        uint block = row * p.groups_per_row + group;
        uchar bits = quants[block * 32 + lane];
        Q4PrefillMeta meta = metadata[block];
        float2 quant = float2(bits & 0x0f, bits >> 4);
        float2 weight = float(meta.scale) * quant + float(meta.bias);
        for (uint s = 0; s < kBatch; ++s) {
            device const half2 *x2 = reinterpret_cast<device const half2 *>(
                x + s * columns + group * 64);
            partial[s] += dot(weight, float2(x2[lane]));
        }
    }
    for (uint s = 0; s < kBatch; ++s) {
        float value = simd_sum(partial[s]);
        if (lane == 0) output[s * p.rows + row] = value;
    }
}

kernel void qwen36_prefill_q4_gemm_f32_residual_f16(
    device const float *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device const half *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant PrefillGemmParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id.x * simdgroups_per_group + simdgroup_index;
    if (row >= p.rows) return;
    uint columns = p.groups_per_row * 64;
    float partial[kPrefillMaxBatch];
    for (uint s = 0; s < kBatch; ++s) partial[s] = 0.0f;
    for (uint group = 0; group < p.groups_per_row; ++group) {
        uint block = row * p.groups_per_row + group;
        uchar bits = quants[block * 32 + lane];
        Q4PrefillMeta meta = metadata[block];
        float2 quant = float2(bits & 0x0f, bits >> 4);
        float2 weight = float(meta.scale) * quant + float(meta.bias);
        for (uint s = 0; s < kBatch; ++s) {
            device const float2 *x2 = reinterpret_cast<device const float2 *>(
                x + s * columns + group * 64);
            partial[s] += dot(weight, x2[lane]);
        }
    }
    for (uint s = 0; s < kBatch; ++s) {
        float value = simd_sum(partial[s]);
        if (lane == 0) {
            output[s * p.rows + row] =
                value + float(residual[s * p.rows + row]);
        }
    }
}

kernel void qwen36_prefill_q4_gemm_f32_residual_f32(
    device const float *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device const float *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant PrefillGemmParams &p [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint row = group_id.x * simdgroups_per_group + simdgroup_index;
    if (row >= p.rows) return;
    uint columns = p.groups_per_row * 64;
    float partial[kPrefillMaxBatch];
    for (uint s = 0; s < kBatch; ++s) partial[s] = 0.0f;
    for (uint group = 0; group < p.groups_per_row; ++group) {
        uint block = row * p.groups_per_row + group;
        uchar bits = quants[block * 32 + lane];
        Q4PrefillMeta meta = metadata[block];
        float2 quant = float2(bits & 0x0f, bits >> 4);
        float2 weight = float(meta.scale) * quant + float(meta.bias);
        for (uint s = 0; s < kBatch; ++s) {
            device const float2 *x2 = reinterpret_cast<device const float2 *>(
                x + s * columns + group * 64);
            partial[s] += dot(weight, x2[lane]);
        }
    }
    for (uint s = 0; s < kBatch; ++s) {
        float value = simd_sum(partial[s]);
        if (lane == 0) {
            output[s * p.rows + row] = value + residual[s * p.rows + row];
        }
    }
}

kernel void qwen36_prefill_silu_mul(
    device const float *gate [[buffer(0)]],
    device const float *up [[buffer(1)]],
    device float *output [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]) {
    uint index = position.x;
    uint s = position.y;
    if (index >= kPrefillMlpWidth || s >= kBatch) return;
    uint flat = s * kPrefillMlpWidth + index;
    float value = gate[flat];
    output[flat] = (value / (1.0f + exp(-value))) * up[flat];
}

/* One thread per channel walks the chunk in order, so the carried 4-tap
 * window and the exiting convolution state match the one-token kernel. */
kernel void qwen36_prefill_delta_conv(
    device const float *projected [[buffer(0)]],
    device const float *weights [[buffer(1)]],
    device float *state [[buffer(2)]],
    device float *output [[buffer(3)]],
    uint channel [[thread_position_in_grid]]) {
    if (channel >= kPrefillQkvRows) return;
    uint base = channel * 4;
    float w0 = weights[base];
    float w1 = weights[base + 1];
    float w2 = weights[base + 2];
    float w3 = weights[base + 3];
    float h0 = state[base];
    float h1 = state[base + 1];
    float h2 = state[base + 2];
    float h3 = state[base + 3];
    for (uint s = 0; s < kBatch; ++s) {
        h0 = h1;
        h1 = h2;
        h2 = h3;
        h3 = projected[s * kPrefillDeltaStride + channel];
        float value = h0 * w0 + h1 * w1 + h2 * w2 + h3 * w3;
        output[s * kPrefillQkvRows + channel] =
            value / (1.0f + exp(-value));
    }
    state[base] = h0;
    state[base + 1] = h1;
    state[base + 2] = h2;
    state[base + 3] = h3;
}

kernel void qwen36_prefill_delta_prepare(
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
    uint s = group_id.y;
    uint key_head = value_head / 3;
    device const float *conv = convolved_qkv + s * kPrefillQkvRows;
    float raw_q = conv[key_head * 128 + tid];
    float raw_k = conv[2048 + key_head * 128 + tid];
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
    uint output_index = s * kPrefillMixerWidth + value_head * 128 + tid;
    query[output_index] = raw_q *
        rsqrt(q_squared[0] + 128.0e-6f) * rsqrt(128.0f);
    key[output_index] = raw_k * rsqrt(k_squared[0] + 128.0e-6f);
    value[output_index] = conv[4096 + value_head * 128 + tid];
    if (tid == 0) {
        device const float *proj = projected + s * kPrefillDeltaStride;
        float a = proj[kPrefillDeltaAOffset + value_head] +
                  dt_bias[value_head];
        float softplus = max(a, 0.0f) + log(1.0f + exp(-abs(a)));
        float g = -exp(a_log[value_head]) * softplus;
        decay[s * kPrefillDeltaHeads + value_head] = exp(g);
        float b = proj[kPrefillDeltaBOffset + value_head];
        beta[s * kPrefillDeltaHeads + value_head] =
            1.0f / (1.0f + exp(-b));
    }
}

/* The delta rule stays sequential in time inside the kernel; parallelism is
 * across the 48 x 128 state columns. Each step is the one-token kernel. */
kernel void qwen36_prefill_delta_recurrent(
    device const float *query [[buffer(0)]],
    device const float *key [[buffer(1)]],
    device const float *value [[buffer(2)]],
    device const float *decay [[buffer(3)]],
    device const float *beta [[buffer(4)]],
    device float *state [[buffer(5)]],
    device float *output [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= kPrefillDeltaHeads * kPrefillDeltaHeadSize) return;
    uint head = index / kPrefillDeltaHeadSize;
    uint value_index = index % kPrefillDeltaHeadSize;
    uint state_base =
        head * kPrefillDeltaHeadSize * kPrefillDeltaHeadSize + value_index;
    for (uint s = 0; s < kBatch; ++s) {
        uint vector_base = s * kPrefillMixerWidth +
                           head * kPrefillDeltaHeadSize;
        float head_decay = decay[s * kPrefillDeltaHeads + head];
        float kv_memory = 0.0f;
        float previous_output = 0.0f;
        float key_query = 0.0f;
        for (uint key_index = 0; key_index < kPrefillDeltaHeadSize;
             ++key_index) {
            float old_value =
                state[state_base + key_index * kPrefillDeltaHeadSize];
            float decayed = old_value * head_decay;
            kv_memory += decayed * key[vector_base + key_index];
            previous_output += decayed * query[vector_base + key_index];
            key_query += key[vector_base + key_index] *
                         query[vector_base + key_index];
        }
        float delta = (value[vector_base + value_index] - kv_memory) *
                      beta[s * kPrefillDeltaHeads + head];
        for (uint key_index = 0; key_index < kPrefillDeltaHeadSize;
             ++key_index) {
            uint state_index =
                state_base + key_index * kPrefillDeltaHeadSize;
            state[state_index] = state[state_index] * head_decay +
                                 key[vector_base + key_index] * delta;
        }
        output[s * kPrefillMixerWidth + index] =
            previous_output + key_query * delta;
    }
}

kernel void qwen36_prefill_delta_gated_norm(
    device const float *core [[buffer(0)]],
    device const float *projected [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device float *output [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float squared[128];
    uint head = group_id.x;
    uint s = group_id.y;
    uint index = s * kPrefillMixerWidth + head * 128 + tid;
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
                                     kPrefillRmsEpsilon) * weight[tid];
    float z = projected[s * kPrefillDeltaStride + kPrefillDeltaZOffset +
                        head * 128 + tid];
    float silu_z = z / (1.0f + exp(-z));
    output[index] = normalized * silu_z;
}

inline float prefill_rope_component(threadgroup const float *values,
                                    uint dimension, uint position) {
    if (dimension >= kPrefillRotarySize) return values[dimension];
    uint frequency = dimension & 31u;
    float exponent = -2.0f * float(frequency) /
                     float(kPrefillRotarySize);
    float angle = float(position) * pow(kPrefillRopeTheta, exponent);
    float c = cos(angle);
    float ss = sin(angle);
    if (dimension < 32) {
        return values[dimension] * c - values[dimension + 32] * ss;
    }
    return values[dimension] * c + values[dimension - 32] * ss;
}

kernel void qwen36_prefill_attention_query(
    device const float *projected [[buffer(0)]],
    device const float *norm_weight [[buffer(1)]],
    constant PrefillAttentionParams &parameters [[buffer(2)]],
    device float *query [[buffer(3)]],
    device float *query_gate [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float values[256];
    threadgroup float squared[256];
    uint head = group_id.x;
    uint s = group_id.y;
    device const float *proj = projected + s * kPrefillAttentionStride;
    uint projection_base = head * 512;
    float value = proj[projection_base + tid];
    values[tid] = value;
    squared[tid] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) squared[tid] += squared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float normalized = value *
        rsqrt(squared[0] / 256.0f + kPrefillRmsEpsilon) *
        norm_weight[tid];
    values[tid] = normalized;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint index = s * kPrefillMixerWidth + head * 256 + tid;
    query[index] = prefill_rope_component(
        values, tid, parameters.start_position + s);
    float gate = proj[projection_base + 256 + tid];
    query_gate[index] = 1.0f / (1.0f + exp(-gate));
}

kernel void qwen36_prefill_attention_key_value(
    device const float *projected [[buffer(0)]],
    device const float *norm_weight [[buffer(1)]],
    constant PrefillAttentionParams &parameters [[buffer(2)]],
    device float *key_cache [[buffer(3)]],
    device float *value_cache [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float values[256];
    threadgroup float squared[256];
    uint head = group_id.x;
    uint s = group_id.y;
    device const float *proj = projected + s * kPrefillAttentionStride;
    float value = proj[kPrefillAttentionKOffset + head * 256 + tid];
    values[tid] = value;
    squared[tid] = value * value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) squared[tid] += squared[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float normalized = value *
        rsqrt(squared[0] / 256.0f + kPrefillRmsEpsilon) *
        norm_weight[tid];
    values[tid] = normalized;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint position = parameters.start_position + s;
    uint cache_index = (position * kPrefillKVHeads + head) * 256 + tid;
    key_cache[cache_index] = prefill_rope_component(values, tid, position);
    value_cache[cache_index] =
        proj[kPrefillAttentionVOffset + head * 256 + tid];
}

kernel void qwen36_prefill_attention_scores(
    device const float *query [[buffer(0)]],
    device const float *key_cache [[buffer(1)]],
    constant PrefillAttentionParams &parameters [[buffer(2)]],
    device float *scores [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_group [[simdgroups_per_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    uint s = group_id.y;
    uint flat = group_id.x * simdgroups_per_group + simdgroup_index;
    uint context_length = parameters.start_position + s + 1;
    uint score_count = kPrefillQHeads * context_length;
    if (flat >= score_count) return;
    uint q_head = flat / context_length;
    uint position = flat - q_head * context_length;
    uint kv_head = q_head / (kPrefillQHeads / kPrefillKVHeads);
    uint query_base = s * kPrefillMixerWidth + q_head * 256;
    uint key_base = (position * kPrefillKVHeads + kv_head) * 256;
    float partial = 0.0f;
    for (uint index = lane; index < 256; index += 32) {
        partial += query[query_base + index] * key_cache[key_base + index];
    }
    float score = simd_sum(partial) * (1.0f / 16.0f);
    if (lane == 0) {
        scores[(s * kPrefillQHeads + q_head) * parameters.cache_capacity +
               position] = score;
    }
}

kernel void qwen36_prefill_attention_softmax_value(
    device const float *scores [[buffer(0)]],
    device const float *value_cache [[buffer(1)]],
    device const float *query_gate [[buffer(2)]],
    constant PrefillAttentionParams &parameters [[buffer(3)]],
    device float *output [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float reduction[256];
    uint q_head = group_id.x;
    uint s = group_id.y;
    uint context_length = parameters.start_position + s + 1;
    uint score_base = (s * kPrefillQHeads + q_head) *
                      parameters.cache_capacity;
    float local_max = -INFINITY;
    for (uint position = tid; position < context_length; position += 256) {
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
    /* The same threadgroup array carries the next reduction; every thread
     * must finish reading the maximum before any thread overwrites slot 0. */
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint position = tid; position < context_length; position += 256) {
        local_sum += exp(scores[score_base + position] - maximum);
    }
    reduction[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) reduction[tid] += reduction[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float denominator = reduction[0];
    uint kv_head = q_head / (kPrefillQHeads / kPrefillKVHeads);
    float value = 0.0f;
    for (uint position = 0; position < context_length; ++position) {
        uint cache_index =
            (position * kPrefillKVHeads + kv_head) * 256 + tid;
        float probability =
            exp(scores[score_base + position] - maximum) / denominator;
        value += probability * value_cache[cache_index];
    }
    uint output_index = s * kPrefillMixerWidth + q_head * 256 + tid;
    output[output_index] = value * query_gate[output_index];
}

/* Tiled simdgroup-matrix GEMM path. The first-generation batched GEMM above
 * keeps decode-identical arithmetic per element but re-reads every batch
 * activation row from device memory once per weight group per simdgroup,
 * which multiplies activation traffic by the batch size. This path stages a
 * [batch x 64] activation tile and a dequantized [64 x 32] weight tile in
 * threadgroup memory once per threadgroup and consumes them with 8x8
 * simdgroup matrix multiply-accumulates, the standard bandwidth shape for
 * Apple-GPU prefill. Accumulation order differs from the one-token kernel,
 * so this path is gated by the argmax/token-parity standard, not bitwise. */

constant uint kGemmTileRows = 32;
constant uint kGemmTileK = 64;
constant uint kGemmTileBatch = 32;

#define QWEN36_PREFILL_GEMM_MMA_BODY(X_LOAD, STORE)                       \
    threadgroup float x_tile[kGemmTileBatch * kGemmTileK];                \
    threadgroup float w_tile[kGemmTileK * kGemmTileRows];                 \
    threadgroup float c_tile[kGemmTileBatch * kGemmTileRows];             \
    uint row0 = group_id.x * kGemmTileRows;                               \
    uint columns = p.groups_per_row * 64;                                 \
    simdgroup_float8x8 accumulator[4];                                    \
    for (uint n = 0; n < 4; ++n)                                          \
        accumulator[n] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f); \
    uint b0 = simdgroup_index * 8;                                        \
    for (uint group = 0; group < p.groups_per_row; ++group) {             \
        for (uint i = 0; i < 16; ++i) {                                   \
            uint linear = tid * 16 + i;                                   \
            uint b = linear >> 6;                                         \
            uint k = linear & 63u;                                        \
            x_tile[linear] = b < kBatch ?                                 \
                X_LOAD(b * columns + group * 64 + k) : 0.0f;              \
        }                                                                 \
        uint r = tid & 31u;                                               \
        uint k_base = (tid >> 5) * 16;                                    \
        uint block = (row0 + r) * p.groups_per_row + group;               \
        Q4PrefillMeta meta = metadata[block];                             \
        float scale = float(meta.scale);                                  \
        float bias = float(meta.bias);                                    \
        for (uint i = 0; i < 16; i += 2) {                                \
            uchar bits = quants[block * 32 + ((k_base + i) >> 1)];        \
            w_tile[(k_base + i) * kGemmTileRows + r] =                    \
                scale * float(bits & 0x0f) + bias;                        \
            w_tile[(k_base + i + 1) * kGemmTileRows + r] =                \
                scale * float(bits >> 4) + bias;                          \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
        for (uint kk = 0; kk < kGemmTileK; kk += 8) {                     \
            simdgroup_float8x8 a;                                         \
            simdgroup_load(a, x_tile + b0 * kGemmTileK + kk,              \
                           kGemmTileK);                                   \
            for (uint n = 0; n < 4; ++n) {                                \
                simdgroup_float8x8 b_fragment;                            \
                simdgroup_load(b_fragment,                                \
                               w_tile + kk * kGemmTileRows + n * 8,       \
                               kGemmTileRows);                            \
                simdgroup_multiply_accumulate(accumulator[n], a,          \
                                              b_fragment,                 \
                                              accumulator[n]);            \
            }                                                             \
        }                                                                 \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
    }                                                                     \
    for (uint n = 0; n < 4; ++n)                                          \
        simdgroup_store(accumulator[n],                                   \
                        c_tile + b0 * kGemmTileRows + n * 8,              \
                        kGemmTileRows);                                   \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    for (uint i = 0; i < 8; ++i) {                                        \
        uint linear = tid * 8 + i;                                        \
        uint b = linear >> 5;                                             \
        uint r = linear & 31u;                                            \
        if (b < kBatch) {                                                 \
            uint out_index = b * p.rows + row0 + r;                       \
            STORE;                                                        \
        }                                                                 \
    }

kernel void qwen36_prefill_q4_gemm_f16_mma(
    device const half *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant PrefillGemmParams &p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define X_LOAD_HALF(index) float(x[index])
#define STORE_PLAIN output[out_index] = c_tile[linear]
    QWEN36_PREFILL_GEMM_MMA_BODY(X_LOAD_HALF, STORE_PLAIN)
#undef X_LOAD_HALF
#undef STORE_PLAIN
}

kernel void qwen36_prefill_q4_gemm_f32_residual_f16_mma(
    device const float *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device const half *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant PrefillGemmParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define X_LOAD_FLOAT(index) x[index]
#define STORE_RESIDUAL_HALF \
    output[out_index] = c_tile[linear] + float(residual[out_index])
    QWEN36_PREFILL_GEMM_MMA_BODY(X_LOAD_FLOAT, STORE_RESIDUAL_HALF)
#undef X_LOAD_FLOAT
#undef STORE_RESIDUAL_HALF
}

kernel void qwen36_prefill_q4_gemm_f32_residual_f32_mma(
    device const float *x [[buffer(0)]],
    device const uchar *quants [[buffer(1)]],
    device const Q4PrefillMeta *metadata [[buffer(2)]],
    device const float *residual [[buffer(3)]],
    device float *output [[buffer(4)]],
    constant PrefillGemmParams &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
#define X_LOAD_FLOAT(index) x[index]
#define STORE_RESIDUAL_FLOAT \
    output[out_index] = c_tile[linear] + residual[out_index]
    QWEN36_PREFILL_GEMM_MMA_BODY(X_LOAD_FLOAT, STORE_RESIDUAL_FLOAT)
#undef X_LOAD_FLOAT
#undef STORE_RESIDUAL_FLOAT
}

/* MTP input fusion: normalized token embedding concatenated with the
 * normalized main-model hidden state, producing the [batch x 10240] input
 * of the MTP fc projection. One threadgroup per batch position. The same
 * threadgroup array carries two reductions, so a barrier separates the
 * read of one result from the next reduction's writes. */
kernel void qwen36_prefill_mtp_fuse(
    device const uchar *embedding_quants [[buffer(0)]],
    device const Q4PrefillMeta *embedding_metadata [[buffer(1)]],
    device const uint *token_ids [[buffer(2)]],
    device const half *hidden [[buffer(3)]],
    device const float *embedding_norm [[buffer(4)]],
    device const float *hidden_norm [[buffer(5)]],
    device half *output [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group_id [[threadgroup_position_in_grid]]) {
    threadgroup float partials[256];
    uint s = group_id.x;
    uint token_id = token_ids[s];
    float sum = 0.0f;
    for (uint index = tid; index < kPrefillHidden; index += 256) {
        uint group = index / 64;
        uint within = index - group * 64;
        uint block = token_id * kPrefillEmbeddingGroups + group;
        uchar bits = embedding_quants[block * 32 + (within >> 1)];
        uint quant = (within & 1u) == 0 ? bits & 0x0f : bits >> 4;
        Q4PrefillMeta meta = embedding_metadata[block];
        float value = float(meta.scale) * float(quant) + float(meta.bias);
        sum += value * value;
    }
    partials[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) partials[tid] += partials[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms_embedding = rsqrt(partials[0] / float(kPrefillHidden) +
                                    kPrefillRmsEpsilon);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint index = tid; index < kPrefillHidden; index += 256) {
        uint group = index / 64;
        uint within = index - group * 64;
        uint block = token_id * kPrefillEmbeddingGroups + group;
        uchar bits = embedding_quants[block * 32 + (within >> 1)];
        uint quant = (within & 1u) == 0 ? bits & 0x0f : bits >> 4;
        Q4PrefillMeta meta = embedding_metadata[block];
        float value = float(meta.scale) * float(quant) + float(meta.bias);
        output[s * 2 * kPrefillHidden + index] =
            half(value * inv_rms_embedding * embedding_norm[index]);
    }
    float hidden_sum = 0.0f;
    for (uint index = tid; index < kPrefillHidden; index += 256) {
        float value = float(hidden[s * kPrefillHidden + index]);
        hidden_sum += value * value;
    }
    partials[tid] = hidden_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride != 0; stride >>= 1) {
        if (tid < stride) partials[tid] += partials[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_rms_hidden = rsqrt(partials[0] / float(kPrefillHidden) +
                                 kPrefillRmsEpsilon);
    for (uint index = tid; index < kPrefillHidden; index += 256) {
        output[s * 2 * kPrefillHidden + kPrefillHidden + index] =
            half(float(hidden[s * kPrefillHidden + index]) *
                 inv_rms_hidden * hidden_norm[index]);
    }
}
