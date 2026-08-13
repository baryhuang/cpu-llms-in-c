#define _POSIX_C_SOURCE 200809L

#include "qwen36_safetensors.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int write_all(int file, const void *data, size_t length) {
    const unsigned char *bytes = data;
    size_t done = 0;
    while (done < length) {
        ssize_t amount = write(file, bytes + done, length - done);
        if (amount <= 0) {
            return -1;
        }
        done += (size_t)amount;
    }
    return 0;
}

int main(void) {
    static const char header[] =
        "{\"tensor\":{\"dtype\":\"U32\",\"shape\":[2,2],"
        "\"data_offsets\":[0,16]},\"tensor.extra\":{\"dtype\":\"U8\","
        "\"shape\":[1],\"data_offsets\":[16,17]}}";
    static const unsigned char payload[17] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    };
    char path[] = "/tmp/qwen36-safetensors-XXXXXX";
    int file = mkstemp(path);
    if (file < 0) {
        perror("mkstemp");
        return 1;
    }
    uint64_t header_length = sizeof(header) - 1;
    unsigned char prefix[8];
    for (unsigned index = 0; index < 8; ++index) {
        prefix[index] = (unsigned char)(header_length >> (index * 8));
    }
    int failure = write_all(file, prefix, sizeof(prefix)) != 0 ||
                  write_all(file, header, (size_t)header_length) != 0 ||
                  write_all(file, payload, sizeof(payload)) != 0;
    close(file);
    if (failure) {
        fprintf(stderr, "cannot write safetensors fixture\n");
        unlink(path);
        return 1;
    }

    qwen36_tensor_view view;
    char error[256];
    int status = qwen36_safetensors_find(path, "tensor", 1, &view,
                                         error, sizeof(error));
    if (status != 0 || strcmp(view.dtype, "U32") != 0 || view.rank != 2 ||
        view.shape[0] != 2 || view.shape[1] != 2 || view.data_length != 16 ||
        view.data_start != 8 + header_length) {
        fprintf(stderr, "valid tensor lookup failed: %s\n", error);
        unlink(path);
        return 1;
    }
    status = qwen36_safetensors_find(path, "tenso", 1, &view,
                                     error, sizeof(error));
    if (status == 0) {
        fprintf(stderr, "non-exact tensor name was accepted\n");
        unlink(path);
        return 1;
    }

    file = open(path, O_WRONLY);
    if (file < 0 || ftruncate(file, (off_t)(8 + header_length + 15)) != 0) {
        perror("truncate fixture");
        if (file >= 0) close(file);
        unlink(path);
        return 1;
    }
    close(file);
    status = qwen36_safetensors_find(path, "tensor", 1, &view,
                                     error, sizeof(error));
    if (status == 0) {
        fprintf(stderr, "truncated payload was accepted\n");
        unlink(path);
        return 1;
    }
    status = qwen36_safetensors_find(path, "tensor", 0, &view,
                                     error, sizeof(error));
    unlink(path);
    if (status != 0) {
        fprintf(stderr, "header-only lookup failed: %s\n", error);
        return 1;
    }
    puts("qwen36 safetensors reader: ok");
    return 0;
}
