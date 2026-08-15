#include "qwen38_safetensors.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    int header_only = 0;
    int first_tensor = 2;
    if (argc >= 2 && strcmp(argv[1], "--header-only") == 0) {
        header_only = 1;
        first_tensor = 3;
    }
    if (argc <= first_tensor) {
        fprintf(stderr, "usage: %s [--header-only] FILE TENSOR [TENSOR ...]\n",
                argv[0]);
        return 2;
    }
    const char *path = argv[first_tensor - 1];
    printf("{\n  \"file\": \"%s\",\n  \"payload_required\": %s,\n  \"tensors\": [\n",
           path, header_only ? "false" : "true");
    for (int argument = first_tensor; argument < argc; ++argument) {
        qwen38_tensor_view view;
        char error[512];
        int status = qwen38_safetensors_find(path, argv[argument], !header_only,
                                             &view, error, sizeof(error));
        if (status != 0) {
            fprintf(stderr, "%s\n", error);
            return status;
        }
        printf("    {\"name\": \"%s\", \"dtype\": \"%s\", \"shape\": [",
               argv[argument], view.dtype);
        for (size_t dimension = 0; dimension < view.rank; ++dimension) {
            printf("%s%" PRIu64, dimension == 0 ? "" : ", ",
                   view.shape[dimension]);
        }
        printf("], \"data_start\": %" PRIu64 ", \"data_length\": %" PRIu64 "}%s\n",
               view.data_start, view.data_length,
               argument + 1 == argc ? "" : ",");
    }
    printf("  ]\n}\n");
    return 0;
}
