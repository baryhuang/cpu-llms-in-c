#define _POSIX_C_SOURCE 200809L

#include "minimindo_parallel.h"
#include "minimindo_thinker.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double seconds(void)
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1.0e-9;
}

static uint32_t argmax(const float *values, uint32_t count)
{
    uint32_t best = 0;
    for (uint32_t index = 1; index < count; ++index)
        if (values[index] > values[best]) best = index;
    return best;
}

static int parse_tokens(const char *text, uint32_t **tokens, size_t *count)
{
    char *copy = strdup(text);
    if (copy == NULL) return -1;
    size_t capacity = 32;
    uint32_t *output = malloc(capacity * sizeof(*output));
    if (output == NULL) { free(copy); return -1; }
    size_t used = 0;
    char *save = NULL;
    for (char *part = strtok_r(copy, ",", &save); part != NULL;
         part = strtok_r(NULL, ",", &save)) {
        char *end = NULL;
        errno = 0;
        unsigned long value = strtoul(part, &end, 10);
        if (errno || end == part || *end != '\0' || value > UINT32_MAX) {
            free(copy); free(output); return -1;
        }
        if (used == capacity) {
            capacity *= 2;
            uint32_t *next = realloc(output, capacity * sizeof(*output));
            if (next == NULL) { free(copy); free(output); return -1; }
            output = next;
        }
        output[used++] = (uint32_t)value;
    }
    free(copy);
    *tokens = output; *count = used;
    return used == 0 ? -1 : 0;
}

int main(int argc, char **argv)
{
    const char *image = NULL, *token_text = NULL;
    uint32_t generate = 0, max_context = 256;
    if (argc >= 2) image = argv[1];
    for (int index = 2; index < argc; ++index) {
        if (strcmp(argv[index], "--tokens") == 0 && index + 1 < argc)
            token_text = argv[++index];
        else if (strcmp(argv[index], "--generate") == 0 && index + 1 < argc)
            generate = (uint32_t)strtoul(argv[++index], NULL, 10);
        else if (strcmp(argv[index], "--max-context") == 0 && index + 1 < argc)
            max_context = (uint32_t)strtoul(argv[++index], NULL, 10);
        else { image = NULL; break; }
    }
    if (image == NULL || token_text == NULL || generate == 0) {
        fprintf(stderr, "usage: %s IMAGE --tokens ID,ID,... --generate N [--max-context N]\n", argv[0]);
        return 2;
    }
    uint32_t *tokens = NULL;
    size_t token_count = 0;
    if (parse_tokens(token_text, &tokens, &token_count) != 0) return 3;
    char error[256] = {0};
    minimindo_thinker *model =
        minimindo_thinker_open(image, max_context, error, sizeof(error));
    if (model == NULL) { fprintf(stderr, "%s\n", error); return 4; }
    const uint32_t vocab = minimindo_thinker_vocab_size(model);
    float *logits = malloc((size_t)vocab * sizeof(*logits));
    if (logits == NULL) return 5;
    if (minimindo_parallel_session_begin(4U) != 0) return 5;
    const double prefill_start = seconds();
    for (size_t index = 0; index < token_count; ++index) {
        if (minimindo_thinker_forward(model, tokens[index], logits, vocab,
                                      error, sizeof(error)) != 0) {
            fprintf(stderr, "%s\n", error);
            minimindo_parallel_session_end();
            return 6;
        }
    }
    const double prefill_seconds = seconds() - prefill_start;
    const double decode_start = seconds();
    uint32_t produced = 0;
    printf("generated_ids=");
    while (produced < generate) {
        const uint32_t next = argmax(logits, vocab);
        printf("%s%" PRIu32, produced ? "," : "", next);
        produced++;
        if (next == 2 || produced == generate) break;
        if (minimindo_thinker_forward(model, next, logits, vocab,
                                      error, sizeof(error)) != 0) {
            fprintf(stderr, "\n%s\n", error);
            minimindo_parallel_session_end();
            return 7;
        }
    }
    const double decode_seconds = seconds() - decode_start;
    printf("\nprompt_tokens=%zu prefill_seconds=%.6f prefill_tps=%.4f\n",
           token_count, prefill_seconds, token_count / prefill_seconds);
    printf("generated_tokens=%" PRIu32 " decode_seconds=%.6f decode_tps=%.4f\n",
           produced, decode_seconds, produced / decode_seconds);
    minimindo_parallel_session_end();
    minimindo_thinker_close(model);
    free(logits); free(tokens);
    return 0;
}
