#include "qwen38_tokenizer.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s TOKENIZER.q38tok UTF8_TEXT\n", argv[0]);
        return 2;
    }
    char error[256];
    qwen38_tokenizer *tokenizer =
        qwen38_tokenizer_open(argv[1], error, sizeof(error));
    if (tokenizer == NULL) {
        fprintf(stderr, "tokenizer open failed: %s\n", error);
        return 3;
    }
    size_t capacity = strlen(argv[2]) * 2 + 32;
    uint32_t *ids = malloc(capacity * sizeof(*ids));
    size_t count = 0;
    if (ids == NULL || qwen38_tokenizer_encode(
            tokenizer, argv[2], ids, capacity, &count,
            error, sizeof(error)) != 0) {
        fprintf(stderr, "encode failed: %s\n", error);
        free(ids);
        qwen38_tokenizer_close(tokenizer);
        return 4;
    }
    size_t decoded_capacity = strlen(argv[2]) * 4 + 256;
    char *decoded = malloc(decoded_capacity);
    size_t decoded_length = 0;
    if (decoded == NULL || qwen38_tokenizer_decode(
            tokenizer, ids, count, decoded, decoded_capacity,
            &decoded_length, error, sizeof(error)) != 0) {
        fprintf(stderr, "decode failed: %s\n", error);
        free(decoded); free(ids);
        qwen38_tokenizer_close(tokenizer);
        return 5;
    }
    printf("{\n  \"input\": \"");
    for (const unsigned char *cursor = (const unsigned char *)argv[2];
         *cursor != '\0'; ++cursor) {
        if (*cursor == '"' || *cursor == '\\') putchar('\\');
        if (*cursor == '\n') printf("\\n");
        else if (*cursor == '\r') printf("\\r");
        else if (*cursor == '\t') printf("\\t");
        else putchar(*cursor);
    }
    printf("\",\n  \"token_ids\": [");
    for (size_t index = 0; index < count; ++index)
        printf("%s%" PRIu32, index == 0 ? "" : ", ", ids[index]);
    printf("],\n  \"token_count\": %zu,\n  \"decoded_utf8\": \"", count);
    for (size_t index = 0; index < decoded_length; ++index) {
        unsigned char value = (unsigned char)decoded[index];
        if (value == '"' || value == '\\') putchar('\\');
        if (value == '\n') printf("\\n");
        else if (value == '\r') printf("\\r");
        else if (value == '\t') printf("\\t");
        else putchar(value);
    }
    printf("\"\n}\n");
    free(decoded);
    free(ids);
    qwen38_tokenizer_close(tokenizer);
    return 0;
}
