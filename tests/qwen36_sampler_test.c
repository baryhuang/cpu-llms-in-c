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
    puts("qwen36 sampler: ok");
    return 0;
}
