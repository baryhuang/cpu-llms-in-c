#ifndef QWEN36_SAMPLER_H
#define QWEN36_SAMPLER_H

#include <stddef.h>
#include <stdint.h>

/* Sampling order matches the reference stacks: presence penalty over
 * the tokens this reply has already emitted, then temperature, then
 * top-k, then min-p (relative to the surviving maximum), then top-p
 * nucleus truncation. top_k bounds the candidate pool; top_k <= 1 or
 * temperature <= 0 is deterministic greedy. When top_p or min_p are
 * active with top_k == 0 the pool is capped at 512 candidates. The
 * presence set covers output tokens only (the vLLM convention), reset
 * per reply via qwen36_sampler_begin. */
typedef struct {
    float temperature;
    uint32_t top_k;
    float top_p;            /* <= 0 or >= 1 disables */
    float min_p;            /* <= 0 disables */
    float presence_penalty; /* 0 disables */
    uint64_t state;
    unsigned char *presence; /* lazily allocated bitmap, vocab bits */
    size_t presence_bits;
} qwen36_sampler;

int qwen36_sample_logits(qwen36_sampler *sampler, const float *logits,
                         size_t logit_count, uint32_t *token_id);

/* Start a reply: clears the presence set (allocating it on first use
 * when a presence penalty is active). */
void qwen36_sampler_begin(qwen36_sampler *sampler, size_t vocabulary);

/* Record an emitted output token for the presence penalty. */
void qwen36_sampler_note(qwen36_sampler *sampler, uint32_t token_id);

void qwen36_sampler_release(qwen36_sampler *sampler);

#endif
