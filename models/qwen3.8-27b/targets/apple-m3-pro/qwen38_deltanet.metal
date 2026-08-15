#include <metal_stdlib>
using namespace metal;

constant uint kDeltaHeads = 48;
constant uint kDeltaHeadSize = 128;

kernel void qwen38_deltanet_direct(
    device const float *query [[buffer(0)]],
    device const float *key [[buffer(1)]],
    device const float *value [[buffer(2)]],
    device const float *decay [[buffer(3)]],
    device const float *beta [[buffer(4)]],
    device float *state [[buffer(5)]],
    device float *output [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= kDeltaHeads * kDeltaHeadSize) {
        return;
    }
    uint head = index / kDeltaHeadSize;
    uint value_index = index % kDeltaHeadSize;
    uint vector_base = head * kDeltaHeadSize;
    uint state_base = head * kDeltaHeadSize * kDeltaHeadSize + value_index;
    float kv_memory = 0.0f;
    float previous_output = 0.0f;
    float key_query = 0.0f;
    float head_decay = decay[head];
    for (uint key_index = 0; key_index < kDeltaHeadSize; ++key_index) {
        float old_value = state[state_base + key_index * kDeltaHeadSize];
        float decayed = old_value * head_decay;
        kv_memory += decayed * key[vector_base + key_index];
        previous_output += decayed * query[vector_base + key_index];
        key_query += key[vector_base + key_index] *
                     query[vector_base + key_index];
    }
    float delta = (value[vector_base + value_index] - kv_memory) * beta[head];
    for (uint key_index = 0; key_index < kDeltaHeadSize; ++key_index) {
        uint state_index = state_base + key_index * kDeltaHeadSize;
        state[state_index] = state[state_index] * head_decay +
                             key[vector_base + key_index] * delta;
    }
    output[index] = previous_output + key_query * delta;
}

/* Benchmark-only alternative. It is retained as a measured control. */
kernel void qwen38_deltanet_float2_vectorized(
    device const float *query [[buffer(0)]],
    device const float *key [[buffer(1)]],
    device const float *value [[buffer(2)]],
    device const float *decay [[buffer(3)]],
    device const float *beta [[buffer(4)]],
    device float *state [[buffer(5)]],
    device float2 *output [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    uint vectors_per_head = kDeltaHeadSize >> 1;
    if (index >= kDeltaHeads * vectors_per_head) {
        return;
    }
    uint head = index / vectors_per_head;
    uint value_vector = index % vectors_per_head;
    uint vector_base = head * kDeltaHeadSize;
    uint state_head_base = head * kDeltaHeadSize * kDeltaHeadSize;
    float2 kv_memory = 0.0f;
    float2 previous_output = 0.0f;
    float key_query = 0.0f;
    float head_decay = decay[head];
    for (uint key_index = 0; key_index < kDeltaHeadSize; ++key_index) {
        device const float2 *state2 = reinterpret_cast<device const float2 *>(
            state + state_head_base + key_index * kDeltaHeadSize);
        float2 decayed = state2[value_vector] * head_decay;
        float key_value = key[vector_base + key_index];
        float query_value = query[vector_base + key_index];
        kv_memory += decayed * key_value;
        previous_output += decayed * query_value;
        key_query += key_value * query_value;
    }
    device const float2 *value2 = reinterpret_cast<device const float2 *>(
        value + vector_base);
    float2 delta = (value2[value_vector] - kv_memory) * beta[head];
    for (uint key_index = 0; key_index < kDeltaHeadSize; ++key_index) {
        device float2 *state2 = reinterpret_cast<device float2 *>(
            state + state_head_base + key_index * kDeltaHeadSize);
        state2[value_vector] = state2[value_vector] * head_decay +
                               key[vector_base + key_index] * delta;
    }
    output[index] = previous_output + key_query * delta;
}
