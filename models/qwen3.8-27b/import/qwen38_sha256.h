#ifndef QWEN38_SHA256_H
#define QWEN38_SHA256_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t state[8];
    uint64_t total_bytes;
    unsigned char block[64];
    size_t block_bytes;
} qwen38_sha256_context;

void qwen38_sha256_init(qwen38_sha256_context *context);
void qwen38_sha256_update(qwen38_sha256_context *context,
                          const void *data, size_t length);
void qwen38_sha256_final(qwen38_sha256_context *context,
                         unsigned char digest[32]);

#endif
