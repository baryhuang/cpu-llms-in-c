#include "qwen36_sampler.h"

#include <stdint.h>
#include <stdio.h>

int main(void) {
    const float logits[] = {-2.0f, 0.5f, 3.0f, 1.0f, -1.0f};
    qwen36_sampler greedy = {.temperature = 0.0f, .top_k = 0, .state = 7};
    uint32_t id = 0;
    if (qwen36_sample_logits(&greedy, logits, 5, &id) != 0 || id != 2) {
        fprintf(stderr, "greedy sampler mismatch\n");
        return 1;
    }
    qwen36_sampler first = {.temperature = 0.8f, .top_k = 3, .state = 42};
    qwen36_sampler second = first;
    for (unsigned index = 0; index < 32; ++index) {
        uint32_t left = 0;
        uint32_t right = 0;
        if (qwen36_sample_logits(&first, logits, 5, &left) != 0 ||
            qwen36_sample_logits(&second, logits, 5, &right) != 0 ||
            left != right || (left != 1 && left != 2 && left != 3)) {
            fprintf(stderr, "top-k sampler mismatch at %u\n", index);
            return 1;
        }
    }
    /* top-p: with a 0.9 nucleus over {10, 9, -10, ...} only the two
     * head candidates can ever be drawn. */
    const float peaked[] = {10.0f, 9.0f, -10.0f, -10.0f, -10.0f};
    qwen36_sampler nucleus = {.temperature = 1.0f, .top_k = 5,
                              .top_p = 0.9f, .state = 11};
    for (unsigned index = 0; index < 64; ++index) {
        if (qwen36_sample_logits(&nucleus, peaked, 5, &id) != 0 ||
            id > 1) {
            fprintf(stderr, "top-p sampler leaked tail id %u\n", id);
            return 1;
        }
    }
    /* min-p 0.5: candidates below half the maximum probability drop. */
    const float sloped[] = {5.0f, 4.9f, 0.0f, -5.0f, -5.0f};
    qwen36_sampler floor_test = {.temperature = 1.0f, .top_k = 5,
                                 .min_p = 0.5f, .state = 13};
    for (unsigned index = 0; index < 64; ++index) {
        if (qwen36_sample_logits(&floor_test, sloped, 5, &id) != 0 ||
            id > 1) {
            fprintf(stderr, "min-p sampler leaked id %u\n", id);
            return 1;
        }
    }
    /* presence penalty: once id 2 is noted with a large penalty it can
     * no longer win against id 3. */
    const float flat[] = {-2.0f, -2.0f, 3.0f, 2.5f, -2.0f};
    qwen36_sampler presence = {.temperature = 0.5f, .top_k = 2,
                               .presence_penalty = 50.0f, .state = 17};
    qwen36_sampler_begin(&presence, 5);
    qwen36_sampler_note(&presence, 2);
    for (unsigned index = 0; index < 64; ++index) {
        if (qwen36_sample_logits(&presence, flat, 5, &id) != 0 ||
            id == 2) {
            fprintf(stderr, "presence penalty failed to suppress\n");
            return 1;
        }
    }
    qwen36_sampler_release(&presence);
    /* determinism: identical configuration and seed draw identically. */
    qwen36_sampler third = {.temperature = 0.7f, .top_k = 4,
                            .top_p = 0.8f, .min_p = 0.05f, .state = 99};
    qwen36_sampler fourth = third;
    for (unsigned index = 0; index < 32; ++index) {
        uint32_t left = 0;
        uint32_t right = 0;
        if (qwen36_sample_logits(&third, logits, 5, &left) != 0 ||
            qwen36_sample_logits(&fourth, logits, 5, &right) != 0 ||
            left != right) {
            fprintf(stderr, "filtered sampler nondeterministic\n");
            return 1;
        }
    }
    puts("qwen36 sampler: ok");
    return 0;
}
