#include "qwen36_sha256.h"

#include <string.h>

static const uint32_t round_constants[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static uint32_t rotate_right(uint32_t value, unsigned amount) {
    return value >> amount | value << (32 - amount);
}

static uint32_t load_be32(const unsigned char *input) {
    return (uint32_t)input[0] << 24 | (uint32_t)input[1] << 16 |
           (uint32_t)input[2] << 8 | input[3];
}

static void store_be32(unsigned char *output, uint32_t value) {
    output[0] = (unsigned char)(value >> 24);
    output[1] = (unsigned char)(value >> 16);
    output[2] = (unsigned char)(value >> 8);
    output[3] = (unsigned char)value;
}

static void transform(qwen36_sha256_context *context,
                      const unsigned char block[64]) {
    uint32_t words[64];
    for (unsigned index = 0; index < 16; ++index) {
        words[index] = load_be32(block + index * 4);
    }
    for (unsigned index = 16; index < 64; ++index) {
        uint32_t a = words[index - 15];
        uint32_t b = words[index - 2];
        uint32_t sigma0 = rotate_right(a, 7) ^ rotate_right(a, 18) ^ (a >> 3);
        uint32_t sigma1 = rotate_right(b, 17) ^ rotate_right(b, 19) ^ (b >> 10);
        words[index] = words[index - 16] + sigma0 + words[index - 7] + sigma1;
    }
    uint32_t a = context->state[0];
    uint32_t b = context->state[1];
    uint32_t c = context->state[2];
    uint32_t d = context->state[3];
    uint32_t e = context->state[4];
    uint32_t f = context->state[5];
    uint32_t g = context->state[6];
    uint32_t h = context->state[7];
    for (unsigned index = 0; index < 64; ++index) {
        uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                        rotate_right(e, 25);
        uint32_t choose = (e & f) ^ (~e & g);
        uint32_t temporary1 = h + sum1 + choose + round_constants[index] +
                              words[index];
        uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                        rotate_right(a, 22);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temporary2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temporary1;
        d = c;
        c = b;
        b = a;
        a = temporary1 + temporary2;
    }
    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

void qwen36_sha256_init(qwen36_sha256_context *context) {
    static const uint32_t initial_state[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };
    memcpy(context->state, initial_state, sizeof(initial_state));
    context->total_bytes = 0;
    context->block_bytes = 0;
}

void qwen36_sha256_update(qwen36_sha256_context *context,
                          const void *data, size_t length) {
    const unsigned char *input = data;
    context->total_bytes += length;
    while (length > 0) {
        size_t available = sizeof(context->block) - context->block_bytes;
        size_t amount = length < available ? length : available;
        memcpy(context->block + context->block_bytes, input, amount);
        context->block_bytes += amount;
        input += amount;
        length -= amount;
        if (context->block_bytes == sizeof(context->block)) {
            transform(context, context->block);
            context->block_bytes = 0;
        }
    }
}

void qwen36_sha256_final(qwen36_sha256_context *context,
                         unsigned char digest[32]) {
    uint64_t total_bits = context->total_bytes * 8;
    context->block[context->block_bytes++] = 0x80;
    if (context->block_bytes > 56) {
        memset(context->block + context->block_bytes, 0,
               sizeof(context->block) - context->block_bytes);
        transform(context, context->block);
        context->block_bytes = 0;
    }
    memset(context->block + context->block_bytes, 0, 56 - context->block_bytes);
    for (unsigned index = 0; index < 8; ++index) {
        context->block[63 - index] = (unsigned char)(total_bits >> (index * 8));
    }
    transform(context, context->block);
    for (unsigned index = 0; index < 8; ++index) {
        store_be32(digest + index * 4, context->state[index]);
    }
    memset(context, 0, sizeof(*context));
}
