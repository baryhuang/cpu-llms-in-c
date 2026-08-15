#include "qwen38_m3_decode.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s MODEL_DIR METALLIB CONTEXT_CAPACITY "
                        "GENERATED_TOKENS PROMPT_TOKEN_ID...\n", argv[0]);
        return 2;
    }
    uint32_t capacity = parse_u32(argv[3], "context capacity", 262144);
    uint32_t generated = parse_u32(argv[4], "generated token count", 4096);
    size_t prompt_count = (size_t)argc - 5;
    if (capacity == 0 || prompt_count + generated > capacity) {
        fprintf(stderr, "prompt and generation exceed context capacity\n");
        return 2;
    }
    char error[512];
    qwen38_m3_model *model = qwen38_m3_model_open(
        argv[1], argv[2], capacity, error, sizeof(error));
    if (model == NULL) {
        fprintf(stderr, "model open failed: %s\n", error);
        return 3;
    }
    printf("{\n  \"schema\": 1,\n");
    printf("  \"scope\": \"Qwen3.8-27B 64-layer greedy token-ID decode\",\n");
    printf("  \"prompt_token_ids\": [");
    for (size_t index = 0; index < prompt_count; ++index) {
        uint32_t token = parse_u32(argv[index + 5], "token ID", 248319);
        printf("%s%" PRIu32, index == 0 ? "" : ", ", token);
    }
    printf("],\n");
    uint32_t next = 0;
    double prompt_ms = 0.0;
    double last_duration_ms = 0.0;
    for (size_t position = 0; position < prompt_count; ++position) {
        uint32_t token = parse_u32(argv[position + 5],
                                   "token ID", 248319);
        qwen38_m3_decode_result result;
        int status = qwen38_m3_model_decode(
            model, token, (uint32_t)position, &result,
            error, sizeof(error));
        if (status != 0) {
            fprintf(stderr, "decode failed at position %zu: %s\n",
                    position, error);
            qwen38_m3_model_close(model);
            return 4;
        }
        next = result.next_token;
        prompt_ms += result.duration_ms;
        last_duration_ms = result.duration_ms;
    }
    uint32_t *generated_ids =
        generated != 0 ? malloc(generated * sizeof(*generated_ids)) : NULL;
    double *generated_ms =
        generated != 0 ? malloc(generated * sizeof(*generated_ms)) : NULL;
    if (generated != 0 && (generated_ids == NULL || generated_ms == NULL)) {
        fprintf(stderr, "cannot allocate generation result\n");
        free(generated_ids);
        free(generated_ms);
        qwen38_m3_model_close(model);
        return 5;
    }
    double continuation_ms = 0.0;
    if (generated != 0) {
        generated_ids[0] = next;
        generated_ms[0] = last_duration_ms;
    }
    for (uint32_t index = 1; index < generated; ++index) {
        uint32_t position = (uint32_t)prompt_count + index - 1;
        uint32_t input = generated_ids[index - 1];
        qwen38_m3_decode_result result;
        int status = qwen38_m3_model_decode(
            model, input, position, &result, error, sizeof(error));
        if (status != 0) {
            fprintf(stderr, "decode failed at position %" PRIu32 ": %s\n",
                    position, error);
            qwen38_m3_model_close(model);
            return 4;
        }
        next = result.next_token;
        generated_ids[index] = next;
        generated_ms[index] = result.duration_ms;
        continuation_ms += result.duration_ms;
    }
    printf("  \"generated\": [\n");
    for (uint32_t index = 0; index < generated; ++index) {
        uint32_t produced_at =
            (uint32_t)prompt_count - 1 + index;
        printf("    {\"token_id\": %" PRIu32
               ", \"produced_by_position\": %" PRIu32
               ", \"source_decode_duration_ms\": %.6f}%s\n",
               generated_ids[index], produced_at, generated_ms[index],
               index + 1 == generated ? "" : ",");
    }
    printf("  ],\n");
    printf("  \"prompt_processing_duration_ms\": %.6f,\n", prompt_ms);
    printf("  \"time_to_first_generated_token_ms\": %.6f,\n", prompt_ms);
    if (generated > 1) {
        printf("  \"continuation_decode_tokens_per_second\": %.6f\n",
               1000.0 * (generated - 1) / continuation_ms);
    } else {
        printf("  \"continuation_decode_tokens_per_second\": null\n");
    }
    printf("}\n");
    free(generated_ids);
    free(generated_ms);
    qwen38_m3_model_close(model);
    return 0;
}
