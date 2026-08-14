#include "qwen36_sampler.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    float logit;
    uint32_t id;
} sample_candidate;

static uint64_t random_u64(uint64_t *state) {
    uint64_t value = *state;
    if (value == 0) value = 0x9e3779b97f4a7c15ull;
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    *state = value;
    return value * 0x2545f4914f6cdd1dull;
}

static uint32_t greedy(const float *logits, size_t count) {
    uint32_t best = 0;
    float best_value = logits[0];
    for (uint32_t id = 1; id < count; ++id) {
        if (logits[id] > best_value) {
            best_value = logits[id];
            best = id;
        }
    }
    return best;
}

int qwen36_sample_logits(qwen36_sampler *sampler, const float *logits,
                         size_t logit_count, uint32_t *token_id) {
    if (sampler == NULL || logits == NULL || token_id == NULL ||
        logit_count == 0 || logit_count > UINT32_MAX) return 1;
    if (sampler->temperature <= 0.0f || sampler->top_k <= 1) {
        *token_id = greedy(logits, logit_count);
        return 0;
    }
    uint32_t count = sampler->top_k;
    if (count > logit_count) count = (uint32_t)logit_count;
    sample_candidate *candidates = malloc((size_t)count * sizeof(*candidates));
    if (candidates == NULL) return 2;
    for (uint32_t index = 0; index < count; ++index)
        candidates[index] = (sample_candidate){.logit = -FLT_MAX, .id = 0};
    for (uint32_t id = 0; id < logit_count; ++id) {
        float value = logits[id];
        if (value <= candidates[count - 1].logit) continue;
        uint32_t position = count - 1;
        while (position != 0 && value > candidates[position - 1].logit) {
            candidates[position] = candidates[position - 1];
            --position;
        }
        candidates[position] = (sample_candidate){.logit = value, .id = id};
    }
    double sum = 0.0;
    float maximum = candidates[0].logit;
    for (uint32_t index = 0; index < count; ++index)
        sum += exp((double)(candidates[index].logit - maximum) /
                   sampler->temperature);
    double uniform = (double)(random_u64(&sampler->state) >> 11) *
                     (1.0 / 9007199254740992.0);
    double threshold = uniform * sum;
    double cumulative = 0.0;
    *token_id = candidates[count - 1].id;
    for (uint32_t index = 0; index < count; ++index) {
        cumulative += exp((double)(candidates[index].logit - maximum) /
                          sampler->temperature);
        if (threshold <= cumulative) {
            *token_id = candidates[index].id;
            break;
        }
    }
    free(candidates);
    return 0;
}
