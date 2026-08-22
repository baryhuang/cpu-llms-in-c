#ifndef LLM_IN_C_MINIMINDO_TOKENIZER_H
#define LLM_IN_C_MINIMINDO_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

typedef struct minimindo_tokenizer minimindo_tokenizer;

minimindo_tokenizer *minimindo_tokenizer_open(
    const char *image_path, char *error, size_t error_capacity);
void minimindo_tokenizer_close(minimindo_tokenizer *tokenizer);

int minimindo_tokenizer_encode(
    const minimindo_tokenizer *tokenizer, const char *utf8,
    uint32_t *token_ids, size_t token_capacity, size_t *token_count,
    char *error, size_t error_capacity);

int minimindo_tokenizer_decode(
    const minimindo_tokenizer *tokenizer, const uint32_t *token_ids,
    size_t token_count, char *utf8, size_t utf8_capacity,
    size_t *utf8_length, char *error, size_t error_capacity);

#endif
