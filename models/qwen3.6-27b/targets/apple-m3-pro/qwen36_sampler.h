#ifndef QWEN36_SAMPLER_H
#define QWEN36_SAMPLER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    float temperature;
    uint32_t top_k;
    uint64_t state;
} qwen36_sampler;

/* temperature <= 0 or top_k <= 1 selects deterministic greedy decoding. */
int qwen36_sample_logits(qwen36_sampler *sampler, const float *logits,
                         size_t logit_count, uint32_t *token_id);

#endif
