#define _POSIX_C_SOURCE 200809L

#include "minimindo_thinker.h"
#include "minimindo_tokenizer.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    minimindo_thinker *thinker;
    minimindo_tokenizer *tokenizer;
    float *logits;
    uint32_t vocab, max_context, max_tokens;
} chat_runtime;

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

static char *format_prompt(const char *user)
{
    static const char prefix[] = "<|im_start|>user\n";
    static const char suffix[] =
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    const size_t length = sizeof(prefix) - 1 + strlen(user) + sizeof(suffix);
    char *prompt = malloc(length);
    if (prompt == NULL) return NULL;
    snprintf(prompt, length, "%s%s%s", prefix, user, suffix);
    return prompt;
}

static int encode_alloc(const minimindo_tokenizer *tokenizer, const char *text,
                        uint32_t **ids, size_t *count,
                        char *error, size_t error_capacity)
{
    size_t required = 0;
    (void)minimindo_tokenizer_encode(tokenizer, text, NULL, 0, &required,
                                     error, error_capacity);
    uint32_t *tokens = malloc(required * sizeof(*tokens));
    if (tokens == NULL) return -1;
    if (minimindo_tokenizer_encode(tokenizer, text, tokens, required, count,
                                   error, error_capacity) != 0) {
        free(tokens);
        return -1;
    }
    *ids = tokens;
    return 0;
}

static char *decode_alloc(const minimindo_tokenizer *tokenizer,
                          const uint32_t *ids, size_t count,
                          char *error, size_t error_capacity)
{
    size_t required = 0;
    (void)minimindo_tokenizer_decode(tokenizer, ids, count, NULL, 0,
                                     &required, error, error_capacity);
    char *text = malloc(required + 1);
    if (text == NULL) return NULL;
    if (minimindo_tokenizer_decode(tokenizer, ids, count, text, required + 1,
                                   &required, error, error_capacity) != 0) {
        free(text);
        return NULL;
    }
    return text;
}

static void json_string(const char *text)
{
    putchar('"');
    for (const unsigned char *cursor = (const unsigned char *)text;
         *cursor != '\0'; ++cursor) {
        switch (*cursor) {
            case '"': fputs("\\\"", stdout); break;
            case '\\': fputs("\\\\", stdout); break;
            case '\b': fputs("\\b", stdout); break;
            case '\f': fputs("\\f", stdout); break;
            case '\n': fputs("\\n", stdout); break;
            case '\r': fputs("\\r", stdout); break;
            case '\t': fputs("\\t", stdout); break;
            default:
                if (*cursor < 0x20) printf("\\u%04x", *cursor);
                else putchar(*cursor);
        }
    }
    putchar('"');
}

static int answer(chat_runtime *runtime, const char *user)
{
    char error[256] = {0};
    char *prompt = format_prompt(user);
    uint32_t *prompt_ids = NULL, *generated = NULL;
    size_t prompt_count = 0, generated_count = 0;
    if (prompt == NULL || encode_alloc(runtime->tokenizer, prompt, &prompt_ids,
                                       &prompt_count, error, sizeof(error)) != 0) {
        fprintf(stderr, "tokenizer: %s\n", error);
        free(prompt);
        return -1;
    }
    if (prompt_count + runtime->max_tokens > runtime->max_context) {
        fprintf(stderr, "prompt requires %zu tokens; context is %u\n",
                prompt_count, runtime->max_context);
        free(prompt); free(prompt_ids);
        return -1;
    }
    generated = malloc((size_t)runtime->max_tokens * sizeof(*generated));
    if (generated == NULL) { free(prompt); free(prompt_ids); return -1; }
    minimindo_thinker_reset(runtime->thinker);
    const double start = seconds();
    for (size_t index = 0; index < prompt_count; ++index) {
        if (minimindo_thinker_forward(runtime->thinker, prompt_ids[index],
                                      runtime->logits, runtime->vocab,
                                      error, sizeof(error)) != 0) {
            fprintf(stderr, "thinker: %s\n", error);
            free(prompt); free(prompt_ids); free(generated);
            return -1;
        }
    }
    const double prefill = seconds() - start;
    const double decode_start = seconds();
    while (generated_count < runtime->max_tokens) {
        const uint32_t token = argmax(runtime->logits, runtime->vocab);
        if (token == 2) break;
        generated[generated_count++] = token;
        if (minimindo_thinker_forward(runtime->thinker, token, runtime->logits,
                                      runtime->vocab, error, sizeof(error)) != 0) {
            fprintf(stderr, "thinker: %s\n", error);
            free(prompt); free(prompt_ids); free(generated);
            return -1;
        }
    }
    const double decode = seconds() - decode_start;
    char *text = decode_alloc(runtime->tokenizer, generated, generated_count,
                              error, sizeof(error));
    if (text == NULL) {
        fprintf(stderr, "decode: %s\n", error);
        free(prompt); free(prompt_ids); free(generated);
        return -1;
    }
    fputs("{\"text\":", stdout);
    json_string(text);
    printf(",\"prompt_tokens\":%zu,\"generated_tokens\":%zu,"
           "\"prefill_ms\":%.3f,\"decode_ms\":%.3f}\n",
           prompt_count, generated_count, prefill * 1000.0, decode * 1000.0);
    fflush(stdout);
    free(text); free(prompt); free(prompt_ids); free(generated);
    return 0;
}

static void usage(const char *program)
{
    fprintf(stderr,
        "usage: %s MODEL.mmo TOKENIZER.mmotok [--prompt TEXT | --serve] "
        "[--generate N] [--max-context N]\n", program);
}

int main(int argc, char **argv)
{
    if (argc < 4) { usage(argv[0]); return 2; }
    const char *model_path = argv[1], *tokenizer_path = argv[2], *prompt = NULL;
    int serve = 0;
    chat_runtime runtime = {.max_context = 256, .max_tokens = 32};
    for (int index = 3; index < argc; ++index) {
        if (strcmp(argv[index], "--prompt") == 0 && index + 1 < argc)
            prompt = argv[++index];
        else if (strcmp(argv[index], "--serve") == 0) serve = 1;
        else if (strcmp(argv[index], "--generate") == 0 && index + 1 < argc)
            runtime.max_tokens = (uint32_t)strtoul(argv[++index], NULL, 10);
        else if (strcmp(argv[index], "--max-context") == 0 && index + 1 < argc)
            runtime.max_context = (uint32_t)strtoul(argv[++index], NULL, 10);
        else { usage(argv[0]); return 2; }
    }
    if ((prompt == NULL) == !serve || runtime.max_tokens == 0 ||
        runtime.max_context == 0) { usage(argv[0]); return 2; }
    char error[256] = {0};
    runtime.thinker = minimindo_thinker_open(model_path, runtime.max_context,
                                              error, sizeof(error));
    if (runtime.thinker == NULL) { fprintf(stderr, "%s\n", error); return 3; }
    runtime.tokenizer = minimindo_tokenizer_open(tokenizer_path, error,
                                                 sizeof(error));
    if (runtime.tokenizer == NULL) { fprintf(stderr, "%s\n", error); return 3; }
    runtime.vocab = minimindo_thinker_vocab_size(runtime.thinker);
    runtime.logits = malloc((size_t)runtime.vocab * sizeof(*runtime.logits));
    if (runtime.logits == NULL) return 3;
    int result = 0;
    if (prompt != NULL) result = answer(&runtime, prompt) != 0;
    else {
        char *line = NULL;
        size_t capacity = 0;
        while (getline(&line, &capacity, stdin) >= 0) {
            line[strcspn(line, "\r\n")] = '\0';
            if (line[0] == '\0') {
                puts("{\"text\":\"\",\"error\":\"empty input\"}");
                fflush(stdout);
                continue;
            }
            if (answer(&runtime, line) != 0) { result = 1; break; }
        }
        free(line);
    }
    free(runtime.logits);
    minimindo_tokenizer_close(runtime.tokenizer);
    minimindo_thinker_close(runtime.thinker);
    return result;
}
