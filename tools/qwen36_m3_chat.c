/* Resident interactive chat for Qwen3.6-27B on Apple M3 Pro.
 *
 * Opens the tokenizer and model once - the one-time weight wiring happens
 * at startup - then serves prompts in a loop from standard input, so every
 * prompt after the first runs at the ready-state time to first token
 * instead of paying a full cold start. Layer state is reset between
 * prompts; each prompt is an independent single-turn request, the same
 * contract as the one-shot generator. */

#include "qwen36_m3_decode.h"
#include "qwen36_sampler.h"
#include "qwen36_tokenizer.h"

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
    return (double)value.tv_sec + (double)value.tv_nsec * 1.0e-9;
}

static uint32_t parse_u32(const char *text, uint32_t fallback) {
    if (text == NULL) return fallback;
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    return end != text && *end == '\0' ? (uint32_t)value : fallback;
}

static char *trim_prompt(const char *prompt) {
    const uint8_t *bytes = (const uint8_t *)prompt;
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

static size_t complete_utf8_prefix(const unsigned char *bytes,
                                   size_t length) {
    size_t cursor = 0;
    while (cursor < length) {
        unsigned char first = bytes[cursor];
        size_t width;
        if (first < 0x80) width = 1;
        else if ((first & 0xe0) == 0xc0) width = 2;
        else if ((first & 0xf0) == 0xe0) width = 3;
        else if ((first & 0xf8) == 0xf0) width = 4;
        else {
            ++cursor;
            continue;
        }
        if (length - cursor < width) break;
        size_t continuation = 1;
        while (continuation < width &&
               (bytes[cursor + continuation] & 0xc0) == 0x80)
            ++continuation;
        if (continuation != width) {
            ++cursor;
            continue;
        }
        cursor += width;
    }
    return cursor;
}

/* Write one JSON string (with quotes) to stdout. */
static void put_json_string(const char *bytes, size_t length) {
    putchar('"');
    for (size_t index = 0; index < length; ++index) {
        unsigned char byte = (unsigned char)bytes[index];
        switch (byte) {
        case '"': fputs("\\\"", stdout); break;
        case '\\': fputs("\\\\", stdout); break;
        case '\n': fputs("\\n", stdout); break;
        case '\r': fputs("\\r", stdout); break;
        case '\t': fputs("\\t", stdout); break;
        default:
            if (byte < 0x20) printf("\\u%04x", byte);
            else putchar(byte);
        }
    }
    putchar('"');
}

static void append_utf8(char **cursor, uint32_t codepoint) {
    char *out = *cursor;
    if (codepoint < 0x80) {
        *out++ = (char)codepoint;
    } else if (codepoint < 0x800) {
        *out++ = (char)(0xc0 | (codepoint >> 6));
        *out++ = (char)(0x80 | (codepoint & 0x3f));
    } else if (codepoint < 0x10000) {
        *out++ = (char)(0xe0 | (codepoint >> 12));
        *out++ = (char)(0x80 | ((codepoint >> 6) & 0x3f));
        *out++ = (char)(0x80 | (codepoint & 0x3f));
    } else {
        *out++ = (char)(0xf0 | (codepoint >> 18));
        *out++ = (char)(0x80 | ((codepoint >> 12) & 0x3f));
        *out++ = (char)(0x80 | ((codepoint >> 6) & 0x3f));
        *out++ = (char)(0x80 | (codepoint & 0x3f));
    }
    *cursor = out;
}

static int hex_value(char character) {
    if (character >= '0' && character <= '9') return character - '0';
    if (character >= 'a' && character <= 'f') return character - 'a' + 10;
    if (character >= 'A' && character <= 'F') return character - 'A' + 10;
    return -1;
}

/* Decode one JSON string literal (machine-mode request line) to UTF-8. */
static char *decode_json_string(const char *line) {
    while (*line == ' ' || *line == '\t') ++line;
    if (*line != '"') return NULL;
    ++line;
    char *output = malloc(strlen(line) * 4 + 1);
    if (output == NULL) return NULL;
    char *cursor = output;
    while (*line != '\0' && *line != '"') {
        if (*line != '\\') {
            *cursor++ = *line++;
            continue;
        }
        ++line;
        switch (*line) {
        case '"': *cursor++ = '"'; ++line; break;
        case '\\': *cursor++ = '\\'; ++line; break;
        case '/': *cursor++ = '/'; ++line; break;
        case 'b': *cursor++ = '\b'; ++line; break;
        case 'f': *cursor++ = '\f'; ++line; break;
        case 'n': *cursor++ = '\n'; ++line; break;
        case 'r': *cursor++ = '\r'; ++line; break;
        case 't': *cursor++ = '\t'; ++line; break;
        case 'u': {
            uint32_t codepoint = 0;
            ++line;
            for (int digit = 0; digit < 4; ++digit) {
                int value = hex_value(*line);
                if (value < 0) { free(output); return NULL; }
                codepoint = codepoint << 4 | (uint32_t)value;
                ++line;
            }
            if (codepoint >= 0xd800 && codepoint <= 0xdbff &&
                line[0] == '\\' && line[1] == 'u') {
                uint32_t low = 0;
                line += 2;
                for (int digit = 0; digit < 4; ++digit) {
                    int value = hex_value(*line);
                    if (value < 0) { free(output); return NULL; }
                    low = low << 4 | (uint32_t)value;
                    ++line;
                }
                codepoint = 0x10000 +
                    ((codepoint - 0xd800) << 10) + (low - 0xdc00);
            }
            append_utf8(&cursor, codepoint);
            break;
        }
        default:
            free(output);
            return NULL;
        }
    }
    if (*line != '"') { free(output); return NULL; }
    *cursor = '\0';
    return output;
}

/* Decode the visible token prefix and print only the new complete UTF-8
 * suffix, so text appears as soon as each token completes. In machine
 * mode each delta goes out as a 'D "..."' protocol line. */
static int emit_new_text(const qwen36_tokenizer *tokenizer,
                         const uint32_t *tokens, size_t token_count,
                         size_t *emitted_bytes, int machine, char *error,
                         size_t error_capacity) {
    size_t output_length = 0;
    (void)qwen36_tokenizer_decode(tokenizer, tokens, token_count, NULL, 0,
                                  &output_length, error, error_capacity);
    char *output = malloc(output_length + 1);
    if (output == NULL || qwen36_tokenizer_decode(
            tokenizer, tokens, token_count, output, output_length + 1,
            &output_length, error, error_capacity) != 0) {
        free(output);
        return -1;
    }
    size_t complete = complete_utf8_prefix(
        (const unsigned char *)output, output_length);
    if (complete > *emitted_bytes) {
        if (machine) {
            fputs("D ", stdout);
            put_json_string(output + *emitted_bytes,
                            complete - *emitted_bytes);
            putchar('\n');
        } else {
            fwrite(output + *emitted_bytes, 1, complete - *emitted_bytes,
                   stdout);
        }
        fflush(stdout);
        *emitted_bytes = complete;
    }
    free(output);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 9) {
        fprintf(stderr,
            "usage: %s MODEL_DIR METALLIB TOKENIZER.q36tok "
            "[CONTEXT] [MAX_NEW] [TEMPERATURE] [TOP_K] [SEED]\n",
            argv[0]);
        return 2;
    }
    uint32_t capacity = parse_u32(argc > 4 ? argv[4] : NULL, 4096);
    uint32_t maximum_new = parse_u32(argc > 5 ? argv[5] : NULL, 3072);
    float temperature = argc > 6 ? strtof(argv[6], NULL) : 0.0f;
    uint32_t top_k = parse_u32(argc > 7 ? argv[7] : NULL, 1);
    uint64_t seed = argc > 8 ? strtoull(argv[8], NULL, 10) : 42;

    const char *machine_env = getenv("QWEN36_MACHINE");
    int machine = machine_env != NULL && strcmp(machine_env, "0") != 0;

    char error[512];
    qwen36_tokenizer *tokenizer =
        qwen36_tokenizer_open(argv[3], error, sizeof(error));
    if (tokenizer == NULL) {
        fprintf(stderr, "tokenizer open failed: %s\n", error);
        return 3;
    }
    fprintf(stderr, "Loading Qwen3.6-27B "
                    "(one-time weight wiring at startup)...\n");
    double start = seconds_now();
    qwen36_m3_model *model = qwen36_m3_model_open(
        argv[1], argv[2], capacity, error, sizeof(error));
    if (model == NULL) {
        fprintf(stderr, "model open failed: %s\n", error);
        qwen36_tokenizer_close(tokenizer);
        return 5;
    }
    fprintf(stderr, "Model resident in %.1f s. Enter /quit to exit.\n",
            seconds_now() - start);
    if (machine) {
        printf("R {\"ready\": true, \"context\": %u, "
               "\"max_new\": %u}\n", capacity, maximum_new);
        fflush(stdout);
    }

    uint32_t *prompt_ids = malloc((size_t)capacity * sizeof(*prompt_ids));
    uint32_t *generated = malloc((size_t)maximum_new * sizeof(*generated));
    char *line = NULL;
    size_t line_capacity = 0;
    if (prompt_ids == NULL || generated == NULL) {
        fprintf(stderr, "cannot allocate buffers\n");
        return 6;
    }

    for (;;) {
        if (!machine) {
            printf("\nYou> ");
            fflush(stdout);
        }
        ssize_t read_length = getline(&line, &line_capacity, stdin);
        if (read_length < 0) {
            if (!machine) printf("\n");
            break;
        }
        char *chat = NULL;
        if (machine) {
            /* One request per line: a JSON string holding the complete
             * rendered chat template, produced by the serving shim. */
            chat = decode_json_string(line);
            if (chat == NULL) {
                if (line[0] != '\n') {
                    printf("X \"request line is not a JSON string\"\n");
                    fflush(stdout);
                }
                continue;
            }
        } else {
            char *trimmed = trim_prompt(line);
            if (trimmed == NULL) continue;
            if (trimmed[0] == '\0') {
                free(trimmed);
                continue;
            }
            if (strcmp(trimmed, "/quit") == 0 ||
                strcmp(trimmed, "/exit") == 0) {
                free(trimmed);
                break;
            }
            chat = render_chat(trimmed);
            free(trimmed);
        }
        size_t prompt_count = 0;
        if (chat == NULL || qwen36_tokenizer_encode(
                tokenizer, chat, prompt_ids, capacity, &prompt_count,
                error, sizeof(error)) != 0) {
            const char *reason = chat == NULL ? "allocation" : error;
            if (machine) {
                fputs("X ", stdout);
                put_json_string(reason, strlen(reason));
                putchar('\n');
                fflush(stdout);
            } else {
                fprintf(stderr, "prompt encode failed: %s\n", reason);
            }
            free(chat);
            continue;
        }
        free(chat);
        if (prompt_count + 1 > capacity) {
            if (machine) {
                printf("X \"prompt (%zu tokens) exceeds context %u\"\n",
                       prompt_count, capacity);
                fflush(stdout);
            } else {
                fprintf(stderr, "prompt (%zu tokens) exceeds context "
                        "%u\n", prompt_count, capacity);
            }
            continue;
        }
        /* A long prompt shrinks this reply's budget instead of failing. */
        uint32_t budget = maximum_new;
        if (prompt_count + budget > capacity)
            budget = capacity - (uint32_t)prompt_count;

        if (!machine) {
            printf("\nModel>\n");
            fflush(stdout);
        }
        double prompt_start = seconds_now();
        const float *logits = NULL;
        size_t logit_count = 0;
        qwen36_m3_decode_result result = {0};
        int failed = 0;
        if (prompt_count > 1) {
            qwen36_m3_prefill_result prefill;
            if (qwen36_m3_model_prefill(
                    model, prompt_ids, (uint32_t)(prompt_count - 1), 0,
                    &prefill, error, sizeof(error)) != 0) {
                fprintf(stderr, "prefill failed: %s\n", error);
                failed = 1;
            }
        }
        if (!failed && qwen36_m3_model_forward(
                model, prompt_ids[prompt_count - 1],
                (uint32_t)(prompt_count - 1), &result, &logits,
                &logit_count, error, sizeof(error)) != 0) {
            fprintf(stderr, "prompt forward failed: %s\n", error);
            failed = 1;
        }
        if (failed) {
            qwen36_m3_model_reset(model);
            continue;
        }

        qwen36_sampler sampler = {
            .temperature = temperature, .top_k = top_k, .state = seed
        };
        uint32_t generated_count = 0;
        size_t visible_count = 0;
        size_t emitted_bytes = 0;
        double first_token_seconds = -1.0;
        for (uint32_t index = 0; index < budget; ++index) {
            uint32_t token;
            size_t sample_count = logit_count;
            if (sample_count > QWEN36_TOKENIZER_VOCAB)
                sample_count = QWEN36_TOKENIZER_VOCAB;
            if (qwen36_sample_logits(&sampler, logits, sample_count,
                                     &token) != 0) {
                fprintf(stderr, "sampling failed\n");
                failed = 1;
                break;
            }
            generated[generated_count++] = token;
            if (token == QWEN36_END_OF_TEXT || token == QWEN36_IM_END)
                break;
            visible_count = generated_count;
            if (first_token_seconds < 0.0)
                first_token_seconds = seconds_now() - prompt_start;
            if (index + 1 == budget) {
                emit_new_text(tokenizer, generated, visible_count,
                              &emitted_bytes, machine, error,
                              sizeof(error));
                break;
            }
            uint32_t position = (uint32_t)prompt_count + index;
            if (qwen36_m3_model_forward_submit(
                    model, token, position, error, sizeof(error)) != 0) {
                fprintf(stderr, "generation submit failed: %s\n", error);
                failed = 1;
                break;
            }
            emit_new_text(tokenizer, generated, visible_count,
                          &emitted_bytes, machine, error, sizeof(error));
            if (qwen36_m3_model_forward_wait(
                    model, &result, &logits, &logit_count,
                    error, sizeof(error)) != 0) {
                fprintf(stderr, "generation wait failed: %s\n", error);
                failed = 1;
                break;
            }
        }
        double total_seconds = seconds_now() - prompt_start;
        if (machine) {
            if (failed) {
                fputs("X ", stdout);
                put_json_string(error, strlen(error));
                putchar('\n');
            } else {
                int stopped = generated_count != 0 &&
                    (generated[generated_count - 1] ==
                         QWEN36_END_OF_TEXT ||
                     generated[generated_count - 1] == QWEN36_IM_END);
                printf("E {\"tokens\": %zu, \"prompt_tokens\": %zu, "
                       "\"first_token_s\": %.3f, \"total_s\": %.3f, "
                       "\"stop\": \"%s\"}\n",
                       visible_count, prompt_count,
                       first_token_seconds >= 0.0 ?
                           first_token_seconds : 0.0,
                       total_seconds,
                       stopped ? "stop" : "length");
            }
            fflush(stdout);
        } else {
            printf("\n");
            fflush(stdout);
            if (!failed && visible_count > 1 &&
                total_seconds > first_token_seconds) {
                fprintf(stderr, "[first token %.2f s, %zu tokens, "
                        "%.1f tok/s]\n", first_token_seconds,
                        visible_count,
                        (double)(visible_count - 1) /
                        (total_seconds - first_token_seconds));
            } else if (!failed && first_token_seconds >= 0.0) {
                fprintf(stderr, "[first token %.2f s]\n",
                        first_token_seconds);
            }
        }
        qwen36_m3_model_reset(model);
    }

    free(line);
    free(prompt_ids);
    free(generated);
    qwen36_m3_model_close(model);
    qwen36_tokenizer_close(tokenizer);
    return 0;
}
