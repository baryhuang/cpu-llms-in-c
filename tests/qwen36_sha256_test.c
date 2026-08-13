#include "qwen36_sha256.h"

#include <stdio.h>
#include <string.h>

static int check(const char *input, const char *expected) {
    qwen36_sha256_context context;
    unsigned char digest[32];
    char actual[65];
    qwen36_sha256_init(&context);
    size_t length = strlen(input);
    size_t split = length / 2;
    qwen36_sha256_update(&context, input, split);
    qwen36_sha256_update(&context, input + split, length - split);
    qwen36_sha256_final(&context, digest);
    for (size_t index = 0; index < sizeof(digest); ++index) {
        snprintf(actual + index * 2, 3, "%02x", digest[index]);
    }
    actual[64] = '\0';
    if (strcmp(actual, expected) != 0) {
        fprintf(stderr, "SHA-256 mismatch: %s != %s\n", actual, expected);
        return 1;
    }
    return 0;
}

int main(void) {
    if (check("", "e3b0c44298fc1c149afbf4c8996fb924"
                  "27ae41e4649b934ca495991b7852b855") != 0 ||
        check("abc", "ba7816bf8f01cfea414140de5dae2223"
                     "b00361a396177a9cb410ff61f20015ad") != 0) {
        return 1;
    }
    puts("qwen36 SHA-256: ok");
    return 0;
}
