#include "minimax_h3_remote_safetensors.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    minimax_h3_remote_safetensors file;
    int fetch = 0;
    int first_tensor = 2;
    char error[512];
    int status;

    if (argc == 3 && strcmp(argv[1], "--header") == 0) {
        status = minimax_h3_remote_safetensors_open(argv[2], &file, error,
                                                     sizeof(error));
        if (status != 0) {
            fprintf(stderr, "%s\n", error);
            return status;
        }
        fwrite(file.header, 1u, file.header_length, stdout);
        fputc('\n', stdout);
        minimax_h3_remote_safetensors_close(&file);
        return 0;
    }

    if (argc > 1 && strcmp(argv[1], "--fetch") == 0) {
        fetch = 1;
        first_tensor = 3;
    }
    if (argc <= first_tensor) {
        fprintf(stderr, "usage: %s --header URL | [--fetch] URL TENSOR [TENSOR ...]\n",
                argv[0]);
        return 2;
    }
    status = minimax_h3_remote_safetensors_open(argv[first_tensor - 1], &file,
                                                 error, sizeof(error));
    if (status != 0) {
        fprintf(stderr, "%s\n", error);
        return status;
    }
    printf("{\n  \"url\": \"%s\",\n  \"header_bytes\": %zu,\n"
           "  \"file_bytes\": %" PRIu64 ",\n  \"tensors\": [\n",
           file.url, file.header_length, file.file_length);
    for (int argument = first_tensor; argument < argc; ++argument) {
        minimax_h3_remote_tensor tensor;
        void *bytes = NULL;
        status = minimax_h3_remote_safetensors_find(
            &file, argv[argument], &tensor, error, sizeof(error));
        if (status != 0) {
            fprintf(stderr, "%s\n", error);
            minimax_h3_remote_safetensors_close(&file);
            return status;
        }
        if (fetch) {
            bytes = malloc((size_t)tensor.data_length);
            if (bytes == NULL || minimax_h3_remote_safetensors_fetch(
                                     &file, &tensor, bytes,
                                     (size_t)tensor.data_length, error,
                                     sizeof(error)) != 0) {
                fprintf(stderr, "%s\n",
                        bytes == NULL ? "allocation failed" : error);
                free(bytes);
                minimax_h3_remote_safetensors_close(&file);
                return 1;
            }
        }
        printf("    {\"name\": \"%s\", \"dtype\": \"%s\", \"shape\": [",
               argv[argument], tensor.dtype);
        for (size_t dimension = 0u; dimension < tensor.rank; ++dimension)
            printf("%s%" PRIu64, dimension == 0u ? "" : ", ",
                   tensor.shape[dimension]);
        printf("], \"data_start\": %" PRIu64
               ", \"data_length\": %" PRIu64
               ", \"fetched\": %s}%s\n",
               tensor.data_start, tensor.data_length, fetch ? "true" : "false",
               argument + 1 == argc ? "" : ",");
        free(bytes);
    }
    puts("  ]\n}");
    minimax_h3_remote_safetensors_close(&file);
    return 0;
}
