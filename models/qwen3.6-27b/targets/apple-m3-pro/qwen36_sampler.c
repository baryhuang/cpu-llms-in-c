#include "qwen36_sampler.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

static int noted(const qwen36_sampler *sampler, uint32_t id) {
    if (sampler->presence == NULL || id >= sampler->presence_bits) return 0;
    return (sampler->presence[id >> 3] >> (id & 7)) & 1;
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

void qwen36_sampler_begin(qwen36_sampler *sampler, size_t vocabulary) {
    if (sampler == NULL) return;
    if (sampler->presence_penalty == 0.0f) return;
    size_t bytes = (vocabulary + 7) / 8;
    if (sampler->presence == NULL || sampler->presence_bits < vocabulary) {
        free(sampler->presence);
        sampler->presence = malloc(bytes);
        sampler->presence_bits = sampler->presence != NULL ? vocabulary : 0;
    }
    if (sampler->presence != NULL) memset(sampler->presence, 0, bytes);
}

void qwen36_sampler_note(qwen36_sampler *sampler, uint32_t token_id) {
    if (sampler == NULL || sampler->presence == NULL ||
        token_id >= sampler->presence_bits) return;
    sampler->presence[token_id >> 3] |=
        (unsigned char)(1u << (token_id & 7));
}

void qwen36_sampler_release(qwen36_sampler *sampler) {
    if (sampler == NULL) return;
    free(sampler->presence);
    sampler->presence = NULL;
    sampler->presence_bits = 0;
}

int qwen36_sample_logits(qwen36_sampler *sampler, const float *logits,
                         size_t logit_count, uint32_t *token_id) {
    if (sampler == NULL || logits == NULL || token_id == NULL ||
        logit_count == 0 || logit_count > UINT32_MAX) return 1;
    int penalized = sampler->presence_penalty != 0.0f &&
                    sampler->presence != NULL;
    if (sampler->temperature <= 0.0f || sampler->top_k == 1) {
        /* Greedy stays penalty-free: it is the lossless speculative
         * path's anchor and the published deterministic mode. */
        *token_id = greedy(logits, logit_count);
        return 0;
    }
    uint32_t count = sampler->top_k;
    if (count == 0) {
        int filtered = (sampler->top_p > 0.0f && sampler->top_p < 1.0f) ||
                       sampler->min_p > 0.0f;
        count = filtered ? 512 : 0;
    }
    if (count == 0) count = 512; /* plain temperature sampling pool */
    if (count > logit_count) count = (uint32_t)logit_count;
    sample_candidate *candidates =
        malloc((size_t)count * sizeof(*candidates));
    if (candidates == NULL) return 2;
    for (uint32_t index = 0; index < count; ++index)
        candidates[index] = (sample_candidate){.logit = -FLT_MAX, .id = 0};
    for (uint32_t id = 0; id < logit_count; ++id) {
        float value = logits[id];
        if (penalized && noted(sampler, id))
            value -= sampler->presence_penalty;
        if (value <= candidates[count - 1].logit) continue;
        uint32_t position = count - 1;
        while (position != 0 && value > candidates[position - 1].logit) {
            candidates[position] = candidates[position - 1];
            --position;
        }
        candidates[position] = (sample_candidate){.logit = value, .id = id};
    }
    float maximum = candidates[0].logit;
    double *weights = malloc((size_t)count * sizeof(*weights));
    if (weights == NULL) {
        free(candidates);
        return 2;
    }
    double sum = 0.0;
    for (uint32_t index = 0; index < count; ++index) {
        weights[index] = exp((double)(candidates[index].logit - maximum) /
                             sampler->temperature);
        sum += weights[index];
    }
    /* min-p: drop candidates whose probability falls below min_p times
     * the maximum candidate's probability. */
    if (sampler->min_p > 0.0f && sum > 0.0) {
        double floor_weight = (double)sampler->min_p * weights[0];
        uint32_t kept = count;
        while (kept > 1 && weights[kept - 1] < floor_weight) --kept;
        for (uint32_t index = kept; index < count; ++index)
            sum -= weights[index];
        count = kept;
    }
    /* top-p: keep the smallest sorted prefix reaching the mass. */
    if (sampler->top_p > 0.0f && sampler->top_p < 1.0f && sum > 0.0) {
        double target = (double)sampler->top_p * sum;
        double cumulative = 0.0;
        uint32_t kept = count;
        for (uint32_t index = 0; index < count; ++index) {
            cumulative += weights[index];
            if (cumulative >= target) {
                kept = index + 1;
                break;
            }
        }
        double new_sum = 0.0;
        for (uint32_t index = 0; index < kept; ++index)
            new_sum += weights[index];
        count = kept;
        sum = new_sum;
    }
    double uniform = (double)(random_u64(&sampler->state) >> 11) *
                     (1.0 / 9007199254740992.0);
    double threshold = uniform * sum;
    double cumulative = 0.0;
    *token_id = candidates[count - 1].id;
    for (uint32_t index = 0; index < count; ++index) {
        cumulative += weights[index];
        if (threshold <= cumulative) {
            *token_id = candidates[index].id;
            break;
        }
    }
    free(weights);
    free(candidates);
    return 0;
}
