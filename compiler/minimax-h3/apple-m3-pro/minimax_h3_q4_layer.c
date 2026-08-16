#include "minimax_h3_q4_layer.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *suffix;
    uint32_t rows;
    uint32_t columns;
} projection_contract;

static const projection_contract projection_contracts[] = {
    {"attn.qkv_proj", 21504u, 5376u},
    {"attn.out_proj", 5376u, 7168u},
    {"mlp.fc1", 28672u, 5376u},
    {"mlp.fc2", 5376u, 14336u},
};

static const char *norm_suffixes[] = {
    "norm1.weight", "norm2.weight", "attn.q_norm.weight",
    "attn.k_norm.weight",
};

static void layer_error(char *error, size_t capacity, const char *format, ...) {
    va_list arguments;
    if (error == NULL || capacity == 0u) return;
    va_start(arguments, format);
    vsnprintf(error, capacity, format, arguments);
    va_end(arguments);
}

static uint64_t fingerprint_update(uint64_t value,
                                   const unsigned char *bytes,
                                   size_t count) {
    for (size_t index = 0u; index < count; ++index) {
        value ^= bytes[index];
        value *= UINT64_C(1099511628211);
    }
    return value;
}

static int tensor_find(const minimax_h3_remote_safetensors *file,
                       unsigned layer_index,
                       const char *suffix,
                       minimax_h3_remote_tensor *tensor,
                       char *error,
                       size_t error_capacity) {
    char name[160];
    if (snprintf(name, sizeof(name), "blocks.%u.%s", layer_index, suffix) >=
        (int)sizeof(name)) {
        layer_error(error, error_capacity, "layer tensor name overflow");
        return 2;
    }
    return minimax_h3_remote_safetensors_find(file, name, tensor, error,
                                               error_capacity);
}

static int fetch_allocated(const minimax_h3_remote_safetensors *file,
                           const minimax_h3_remote_tensor *tensor,
                           void **bytes,
                           char *error,
                           size_t error_capacity) {
    if (tensor->data_length == 0u || tensor->data_length > SIZE_MAX) {
        layer_error(error, error_capacity, "invalid tensor allocation size");
        return 1;
    }
    *bytes = malloc((size_t)tensor->data_length);
    if (*bytes == NULL) {
        layer_error(error, error_capacity,
                    "cannot allocate %" PRIu64 " tensor bytes",
                    tensor->data_length);
        return 1;
    }
    if (minimax_h3_remote_safetensors_fetch(
            file, tensor, *bytes, (size_t)tensor->data_length, error,
            error_capacity) != 0) {
        free(*bytes);
        *bytes = NULL;
        return 1;
    }
    return 0;
}

void minimax_h3_q4_layer_free(minimax_h3_q4_layer *layer) {
    if (layer == NULL) return;
    for (size_t index = 0u; index < MINIMAX_H3_Q4_PROJECTION_COUNT; ++index) {
        free(layer->projections[index].weight);
        free(layer->projections[index].scales);
        free(layer->projections[index].biases);
    }
    for (size_t index = 0u; index < MINIMAX_H3_Q4_NORM_COUNT; ++index)
        free(layer->norms[index]);
    memset(layer, 0, sizeof(*layer));
}

int minimax_h3_q4_layer_load(const minimax_h3_remote_safetensors *file,
                             unsigned layer_index,
                             minimax_h3_q4_layer *layer,
                             char *error,
                             size_t error_capacity) {
    minimax_h3_q4_layer result;
    if (file == NULL || layer == NULL || layer_index >= 50u) {
        layer_error(error, error_capacity, "invalid H3 layer request");
        return 2;
    }
    memset(&result, 0, sizeof(result));
    result.layer_index = layer_index;
    result.fingerprint = UINT64_C(1469598103934665603);
    for (size_t index = 0u; index < MINIMAX_H3_Q4_PROJECTION_COUNT; ++index) {
        const projection_contract *contract = &projection_contracts[index];
        minimax_h3_q4_projection *projection = &result.projections[index];
        minimax_h3_remote_tensor weight;
        minimax_h3_remote_tensor scales;
        minimax_h3_remote_tensor biases;
        char suffix[96];
        projection->suffix = contract->suffix;
        projection->output_rows = contract->rows;
        projection->input_columns = contract->columns;
        projection->groups_per_row = contract->columns / 64u;
        if (snprintf(suffix, sizeof(suffix), "%s.weight", contract->suffix) >=
                (int)sizeof(suffix) ||
            tensor_find(file, layer_index, suffix, &weight, error,
                        error_capacity) != 0)
            goto failed;
        if (snprintf(suffix, sizeof(suffix), "%s.scales", contract->suffix) >=
                (int)sizeof(suffix) ||
            tensor_find(file, layer_index, suffix, &scales, error,
                        error_capacity) != 0)
            goto failed;
        if (snprintf(suffix, sizeof(suffix), "%s.biases", contract->suffix) >=
                (int)sizeof(suffix) ||
            tensor_find(file, layer_index, suffix, &biases, error,
                        error_capacity) != 0)
            goto failed;
        uint64_t blocks = (uint64_t)contract->rows *
                          projection->groups_per_row;
        uint64_t weight_bytes = blocks * 32u;
        uint64_t metadata_bytes = blocks * 2u;
        if (strcmp(weight.dtype, "U32") != 0 || weight.rank != 2u ||
            weight.shape[0] != contract->rows ||
            weight.shape[1] != contract->columns / 8u ||
            weight.data_length != weight_bytes ||
            strcmp(scales.dtype, "BF16") != 0 || scales.rank != 2u ||
            scales.shape[0] != contract->rows ||
            scales.shape[1] != projection->groups_per_row ||
            scales.data_length != metadata_bytes ||
            strcmp(biases.dtype, "BF16") != 0 || biases.rank != 2u ||
            biases.shape[0] != contract->rows ||
            biases.shape[1] != projection->groups_per_row ||
            biases.data_length != metadata_bytes) {
            layer_error(error, error_capacity,
                        "unexpected compact contract for blocks.%u.%s",
                        layer_index, contract->suffix);
            goto failed;
        }
        if (fetch_allocated(file, &weight, (void **)&projection->weight, error,
                            error_capacity) != 0 ||
            fetch_allocated(file, &scales, (void **)&projection->scales, error,
                            error_capacity) != 0 ||
            fetch_allocated(file, &biases, (void **)&projection->biases, error,
                            error_capacity) != 0)
            goto failed;
        projection->weight_bytes = (size_t)weight_bytes;
        projection->metadata_elements = (size_t)blocks;
        result.allocated_bytes += (size_t)(weight_bytes + 2u * metadata_bytes);
        result.fingerprint = fingerprint_update(
            result.fingerprint, projection->weight,
            projection->weight_bytes);
        result.fingerprint = fingerprint_update(
            result.fingerprint, (const unsigned char *)projection->scales,
            (size_t)metadata_bytes);
        result.fingerprint = fingerprint_update(
            result.fingerprint, (const unsigned char *)projection->biases,
            (size_t)metadata_bytes);
    }
    for (size_t index = 0u; index < MINIMAX_H3_Q4_NORM_COUNT; ++index) {
        minimax_h3_remote_tensor tensor;
        size_t elements = index < 2u ? 5376u : 128u;
        if (tensor_find(file, layer_index, norm_suffixes[index], &tensor,
                        error, error_capacity) != 0)
            goto failed;
        if (strcmp(tensor.dtype, "BF16") != 0 || tensor.rank != 1u ||
            tensor.shape[0] != elements || tensor.data_length != elements * 2u) {
            layer_error(error, error_capacity,
                        "unexpected norm contract for blocks.%u.%s",
                        layer_index, norm_suffixes[index]);
            goto failed;
        }
        if (fetch_allocated(file, &tensor, (void **)&result.norms[index], error,
                            error_capacity) != 0)
            goto failed;
        result.norm_elements[index] = elements;
        result.allocated_bytes += elements * 2u;
        result.fingerprint = fingerprint_update(
            result.fingerprint, (const unsigned char *)result.norms[index],
            elements * 2u);
    }
    *layer = result;
    return 0;

failed:
    minimax_h3_q4_layer_free(&result);
    return 1;
}
