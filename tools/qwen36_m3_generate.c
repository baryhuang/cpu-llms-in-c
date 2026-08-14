#define _POSIX_C_SOURCE 200809L

#include "qwen36_m3_decode.h"
#include "qwen36_sampler.h"
#include "qwen36_tokenizer.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <unicode/uchar.h>
#include <unicode/utf8.h>

enum { QWEN36_END_OF_TEXT = 248044, QWEN36_IM_END = 248046 };

static double seconds_now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static uint32_t parse_u32(const char *text, const char *name,
                          uint32_t maximum) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value > maximum) {
        fprintf(stderr, "invalid %s: %s\n", name, text);
        exit(2);
    }
    return (uint32_t)value;
}

static uint64_t parse_u64(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        fprintf(stderr, "invalid %s: %s\n", name, text);
        exit(2);
    }
    return (uint64_t)value;
}

static float parse_float(const char *text, const char *name) {
    char *end = NULL;
    errno = 0;
    float value = strtof(text, &end);
    if (errno != 0 || end == text || *end != '\0' || value < 0.0f) {
        fprintf(stderr, "invalid %s: %s\n", name, text);
        exit(2);
    }
    return value;
}

static void print_json_string(const char *text) {
    putchar('"');
    for (const unsigned char *cursor = (const unsigned char *)text;
         *cursor != '\0'; ++cursor) {
        switch (*cursor) {
            case '"': printf("\\\""); break;
            case '\\': printf("\\\\"); break;
            case '\b': printf("\\b"); break;
            case '\f': printf("\\f"); break;
            case '\n': printf("\\n"); break;
            case '\r': printf("\\r"); break;
            case '\t': printf("\\t"); break;
            default:
                if (*cursor < 0x20) printf("\\u%04x", *cursor);
                else putchar(*cursor);
        }
    }
    putchar('"');
}

static char *trim_prompt(const char *prompt) {
    const unsigned char *bytes = (const unsigned char *)prompt;
    int32_t length = (int32_t)strlen(prompt);
    int32_t start = 0;
    while (start < length) {
        int32_t next = start;
        UChar32 codepoint;
        U8_NEXT(bytes, next, length, codepoint);
        if (codepoint < 0 || !u_isUWhiteSpace(codepoint)) break;
        start = next;
    }
    int32_t end = length;
    while (end > start) {
        int32_t previous = end;
        UChar32 codepoint;
        U8_PREV(bytes, 0, previous, codepoint);
        if (codepoint < 0 || !u_isUWhiteSpace(codepoint)) break;
        end = previous;
    }
    size_t trimmed_length = (size_t)(end - start);
    char *trimmed = malloc(trimmed_length + 1);
    if (trimmed == NULL) return NULL;
    memcpy(trimmed, prompt + start, trimmed_length);
    trimmed[trimmed_length] = '\0';
    return trimmed;
}

static char *render_chat(const char *prompt) {
    static const char prefix[] = "<|im_start|>user\n";
    static const char suffix[] =
        "<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n";
    size_t length = sizeof(prefix) - 1 + strlen(prompt) + sizeof(suffix);
    char *chat = malloc(length);
    if (chat == NULL) return NULL;
    snprintf(chat, length, "%s%s%s", prefix, prompt, suffix);
    return chat;
}

int main(int argc, char **argv) {
    if (argc != 10) {
        fprintf(stderr,
            "usage: %s MODEL_DIR METALLIB TOKENIZER.q36tok "
            "CONTEXT_CAPACITY MAX_NEW_TOKENS TEMPERATURE TOP_K SEED PROMPT\n",
            argv[0]);
        return 2;
    }
    uint32_t capacity = parse_u32(argv[4], "context capacity", 262144);
    uint32_t maximum_new = parse_u32(argv[5], "new token count", 4096);
    float temperature = parse_float(argv[6], "temperature");
    uint32_t top_k = parse_u32(argv[7], "top-k", 4096);
    uint64_t seed = parse_u64(argv[8], "seed");
    if (capacity == 0 || maximum_new == 0) {
        fprintf(stderr, "context capacity and new token count must be nonzero\n");
        return 2;
    }
    char error[512];
    double start = seconds_now();
    qwen36_tokenizer *tokenizer =
        qwen36_tokenizer_open(argv[3], error, sizeof(error));
    if (tokenizer == NULL) {
        fprintf(stderr, "tokenizer open failed: %s\n", error);
        return 3;
    }
    double tokenizer_open_ms = (seconds_now() - start) * 1000.0;
    char *trimmed_prompt = trim_prompt(argv[9]);
    char *chat = trimmed_prompt != NULL ? render_chat(trimmed_prompt) : NULL;
    uint32_t *prompt_ids = malloc((size_t)capacity * sizeof(*prompt_ids));
    if (chat == NULL || prompt_ids == NULL) {
        fprintf(stderr, "cannot allocate prompt\n");
        free(trimmed_prompt); free(chat); free(prompt_ids);
        qwen36_tokenizer_close(tokenizer);
        return 4;
    }
    size_t prompt_count = 0;
    start = seconds_now();
    if (qwen36_tokenizer_encode(tokenizer, chat, prompt_ids, capacity,
                                &prompt_count, error, sizeof(error)) != 0) {
        fprintf(stderr, "prompt encode failed: %s\n", error);
        free(trimmed_prompt); free(chat); free(prompt_ids);
        qwen36_tokenizer_close(tokenizer);
        return 4;
    }
    double tokenizer_encode_ms = (seconds_now() - start) * 1000.0;
    if (prompt_count + maximum_new > capacity) {
        fprintf(stderr, "prompt (%zu) plus generation (%u) exceeds context %u\n",
                prompt_count, maximum_new, capacity);
        free(trimmed_prompt); free(chat); free(prompt_ids);
        qwen36_tokenizer_close(tokenizer);
        return 4;
    }
    start = seconds_now();
    qwen36_m3_model *model = qwen36_m3_model_open(
        argv[1], argv[2], capacity, error, sizeof(error));
    if (model == NULL) {
        fprintf(stderr, "model open failed: %s\n", error);
        free(trimmed_prompt); free(chat); free(prompt_ids);
        qwen36_tokenizer_close(tokenizer);
        return 5;
    }
    double model_open_ms = (seconds_now() - start) * 1000.0;
    uint32_t *generated = malloc((size_t)maximum_new * sizeof(*generated));
    double *token_ms = malloc((size_t)maximum_new * sizeof(*token_ms));
    if (generated == NULL || token_ms == NULL) {
        fprintf(stderr, "cannot allocate generation buffers\n");
        free(generated); free(token_ms);
        qwen36_m3_model_close(model);
        free(trimmed_prompt); free(chat); free(prompt_ids);
        qwen36_tokenizer_close(tokenizer);
        return 6;
    }
    qwen36_sampler sampler = {
        .temperature = temperature, .top_k = top_k, .state = seed
    };
    const float *logits = NULL;
    size_t logit_count = 0;
    qwen36_m3_decode_result result = {0};
    start = seconds_now();
    double prompt_first_forward_ms = 0.0;
    for (size_t position = 0; position < prompt_count; ++position) {
        double token_start = seconds_now();
        if (qwen36_m3_model_forward(
                model, prompt_ids[position], (uint32_t)position, &result,
                &logits, &logit_count, error, sizeof(error)) != 0) {
            fprintf(stderr, "prompt forward failed at %zu: %s\n",
                    position, error);
            qwen36_m3_model_close(model);
            free(generated); free(token_ms); free(trimmed_prompt);
            free(chat); free(prompt_ids);
            qwen36_tokenizer_close(tokenizer);
            return 7;
        }
        if (position == 0)
            prompt_first_forward_ms =
                (seconds_now() - token_start) * 1000.0;
    }
    double prompt_processing_ms = (seconds_now() - start) * 1000.0;
    double prompt_after_first_ms =
        prompt_processing_ms - prompt_first_forward_ms;
    uint32_t generated_count = 0;
    size_t text_token_count = 0;
    double continuation_ms = 0.0;
    for (uint32_t index = 0; index < maximum_new; ++index) {
        uint32_t token;
        size_t sample_count = logit_count;
        if (sample_count > QWEN36_TOKENIZER_VOCAB)
            sample_count = QWEN36_TOKENIZER_VOCAB;
        if (qwen36_sample_logits(&sampler, logits, sample_count, &token) != 0) {
            fprintf(stderr, "sampling failed\n");
            return 8;
        }
        generated[generated_count] = token;
        token_ms[generated_count] = result.duration_ms;
        ++generated_count;
        if (token == QWEN36_END_OF_TEXT || token == QWEN36_IM_END) break;
        text_token_count = generated_count;
        if (index + 1 == maximum_new) break;
        uint32_t position = (uint32_t)prompt_count + index;
        double continuation_start = seconds_now();
        if (qwen36_m3_model_forward(model, token, position, &result,
                                    &logits, &logit_count,
                                    error, sizeof(error)) != 0) {
            fprintf(stderr, "generation forward failed at %u: %s\n",
                    position, error);
            return 7;
        }
        continuation_ms += (seconds_now() - continuation_start) * 1000.0;
    }
    size_t output_length = 0;
    (void)qwen36_tokenizer_decode(tokenizer, generated, text_token_count,
                                  NULL, 0, &output_length,
                                  error, sizeof(error));
    char *output = malloc(output_length + 1);
    if (output == NULL || qwen36_tokenizer_decode(
            tokenizer, generated, text_token_count, output,
            output_length + 1, &output_length,
            error, sizeof(error)) != 0) {
        fprintf(stderr, "generated text decode failed: %s\n", error);
        return 9;
    }
    printf("{\n  \"schema\": 1,\n");
    printf("  \"scope\": \"Qwen3.6-27B free-text end-to-end generation\",\n");
    printf("  \"prompt\": "); print_json_string(argv[9]); printf(",\n");
    printf("  \"rendered_chat\": "); print_json_string(chat); printf(",\n");
    printf("  \"prompt_token_ids\": [");
    for (size_t index = 0; index < prompt_count; ++index)
        printf("%s%u", index == 0 ? "" : ", ", prompt_ids[index]);
    printf("],\n  \"generated\": [\n");
    for (uint32_t index = 0; index < generated_count; ++index) {
        uint32_t produced_by = (uint32_t)prompt_count - 1 + index;
        printf("    {\"token_id\": %u, \"produced_by_position\": %u, "
               "\"source_forward_duration_ms\": %.6f}%s\n",
               generated[index], produced_by, token_ms[index],
               index + 1 == generated_count ? "" : ",");
    }
    printf("  ],\n  \"output\": "); print_json_string(output); printf(",\n");
    printf("  \"stop_reason\": \"%s\",\n",
           generated_count != 0 &&
           (generated[generated_count - 1] == QWEN36_END_OF_TEXT ||
            generated[generated_count - 1] == QWEN36_IM_END) ?
               "end_token" : "max_new_tokens");
    printf("  \"sampler\": {\"temperature\": %.6g, \"top_k\": %u, "
           "\"seed\": \"%" PRIu64 "\"},\n",
           temperature, top_k, seed);
    printf("  \"durations_ms\": {\"tokenizer_open\": %.6f, "
           "\"tokenizer_encode\": %.6f, \"model_open\": %.6f, "
           "\"prompt_processing\": %.6f, \"prompt_first_forward\": %.6f, "
           "\"prompt_after_first\": %.6f, "
           "\"time_to_first_token_after_open\": %.6f, "
           "\"continuation\": %.6f},\n",
           tokenizer_open_ms, tokenizer_encode_ms, model_open_ms,
           prompt_processing_ms, prompt_first_forward_ms,
           prompt_after_first_ms,
           tokenizer_encode_ms + prompt_processing_ms, continuation_ms);
    printf("  \"prompt_tokens_per_second\": %.6f,\n",
           1000.0 * prompt_count / prompt_processing_ms);
    if (prompt_count > 1 && prompt_after_first_ms > 0.0)
        printf("  \"prompt_tokens_per_second_after_first\": %.6f,\n",
               1000.0 * (prompt_count - 1) / prompt_after_first_ms);
    else
        printf("  \"prompt_tokens_per_second_after_first\": null,\n");
    if (generated_count > 1 && continuation_ms > 0.0)
        printf("  \"continuation_tokens_per_second\": %.6f,\n",
               1000.0 * (generated_count - 1) / continuation_ms);
    else
        printf("  \"continuation_tokens_per_second\": null,\n");
    printf("  \"memory\": {\"mapped_weights_bytes\": %zu, "
           "\"recurrent_state_bytes\": %zu, \"kv_cache_bytes\": %zu, "
           "\"physical_footprint_bytes\": %zu}\n}\n",
           result.mapped_weight_bytes, result.state_bytes,
           result.kv_cache_bytes, result.physical_footprint_bytes);
    free(output); free(generated); free(token_ms);
    qwen36_m3_model_close(model);
    free(trimmed_prompt); free(chat); free(prompt_ids);
    qwen36_tokenizer_close(tokenizer);
    return 0;
}
