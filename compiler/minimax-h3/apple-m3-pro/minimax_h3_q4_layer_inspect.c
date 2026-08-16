#include "minimax_h3_q4_layer.h"

#include <inttypes.h>
#include <mach/mach.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static double seconds_now(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC_RAW, &value);
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static size_t footprint(void) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    return task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info,
                     &count) == KERN_SUCCESS
               ? (size_t)info.phys_footprint
               : 0u;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s TRANSFORMER_URL LAYER_INDEX\n", argv[0]);
        return 2;
    }
    unsigned layer_index = 0u;
    if (sscanf(argv[2], "%u", &layer_index) != 1 || layer_index >= 50u)
        return 2;
    minimax_h3_remote_safetensors file;
    minimax_h3_q4_layer layer;
    char error[512];
    double started = seconds_now();
    if (minimax_h3_remote_safetensors_open(argv[1], &file, error,
                                            sizeof(error)) != 0) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
    size_t before = footprint();
    if (minimax_h3_q4_layer_load(&file, layer_index, &layer, error,
                                 sizeof(error)) != 0) {
        fprintf(stderr, "%s\n", error);
        minimax_h3_remote_safetensors_close(&file);
        return 1;
    }
    double elapsed = seconds_now() - started;
    size_t peak = footprint();
    printf("{\n  \"layer\": %u,\n  \"real_weights\": true,\n"
           "  \"allocated_bytes\": %zu,\n  \"duration_seconds\": %.6f,\n"
           "  \"footprint_before_bytes\": %zu,\n"
           "  \"footprint_after_bytes\": %zu,\n"
           "  \"fingerprint_fnv1a64\": \"%016" PRIx64 "\",\n"
           "  \"projections\": [\n",
           layer.layer_index, layer.allocated_bytes, elapsed, before, peak,
           layer.fingerprint);
    for (size_t index = 0u; index < MINIMAX_H3_Q4_PROJECTION_COUNT; ++index) {
        const minimax_h3_q4_projection *projection = &layer.projections[index];
        printf("    {\"name\": \"%s\", \"input_columns\": %u, "
               "\"output_rows\": %u, \"weight_bytes\": %zu}%s\n",
               projection->suffix, projection->input_columns,
               projection->output_rows, projection->weight_bytes,
               index + 1u == MINIMAX_H3_Q4_PROJECTION_COUNT ? "" : ",");
    }
    puts("  ]\n}");
    minimax_h3_q4_layer_free(&layer);
    minimax_h3_remote_safetensors_close(&file);
    return 0;
}
