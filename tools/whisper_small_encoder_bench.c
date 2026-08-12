#define _POSIX_C_SOURCE 200809L

#include "whisper_small_image.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double monotonic_seconds(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0.0;
    return (double)now.tv_sec + (double)now.tv_nsec * 1.0e-9;
}

static size_t parse_size(const char *text, const char *name, size_t minimum, size_t maximum)
{
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (text[0] == '\0' || end == NULL || *end != '\0' || value < minimum || value > maximum) {
        fprintf(stderr, "error: %s must be in %zu..%zu\n", name, minimum, maximum);
        exit(2);
    }
    return (size_t)value;
}

int main(int argc, char **argv)
{
    cllm_whisper_small_model model;
    cllm_whisper_small_encoder_metrics metrics;
    float *mel;
    float *output;
    size_t input_frames;
    size_t output_frames;
    size_t layer_count;
    size_t repetitions;
    double opened;
    double load_seconds;
    double total = 0.0;
    double checksum = 0.0;

    if (argc != 5) {
        fprintf(stderr, "usage: %s IMAGE.whenc INPUT_FRAMES LAYERS REPETITIONS\n", argv[0]);
        return 2;
    }
    input_frames = parse_size(argv[2], "INPUT_FRAMES", 1U, 3000U);
    layer_count = parse_size(argv[3], "LAYERS", 0U, 12U);
    repetitions = parse_size(argv[4], "REPETITIONS", 1U, 1000U);
    output_frames = (input_frames + 1U) / 2U;
    mel = malloc(80U * input_frames * sizeof(float));
    output = malloc(output_frames * 768U * sizeof(float));
    if (mel == NULL || output == NULL) {
        fprintf(stderr, "error: allocation failed\n");
        free(mel);
        free(output);
        return 2;
    }
    for (size_t index = 0U; index < 80U * input_frames; ++index)
        mel[index] = 0.21f * sinf((float)index * 0.013f + 0.2f) +
                     0.07f * cosf((float)index * 0.031f - 0.4f);

    opened = monotonic_seconds();
    if (cllm_whisper_small_model_open(argv[1], &model) != 0) {
        fprintf(stderr, "error: cannot open exact encoder image\n");
        free(mel);
        free(output);
        return 2;
    }
    load_seconds = monotonic_seconds() - opened;
    for (size_t repetition = 0U; repetition < repetitions; ++repetition) {
        double started = monotonic_seconds();
        if (cllm_whisper_small_encode_mel(&model, mel, input_frames, layer_count,
                                          output, &metrics) != 0) {
            fprintf(stderr, "error: encoder execution failed\n");
            cllm_whisper_small_model_close(&model);
            free(mel);
            free(output);
            return 2;
        }
        total += monotonic_seconds() - started;
    }
    for (size_t index = 0U; index < output_frames * 768U; ++index)
        checksum += (double)output[index] * (double)((index % 17U) + 1U);

    printf("section=benchmark model=whisper-small.en image_precision=f32 ");
    printf("input_frames=%zu output_frames=%zu layers=%zu repetitions=%zu ",
           input_frames, output_frames, layer_count, repetitions);
    printf("image_map_seconds=%.6f duration_seconds=%.6f mean_seconds=%.6f ",
           load_seconds, total, total / (double)repetitions);
    printf("stem_seconds=%.6f final_norm_seconds=%.6f workspace_bytes=%zu ",
           metrics.stem_seconds, metrics.final_norm_seconds, metrics.workspace_bytes);
    printf("checksum=%.9g\n", checksum);
    for (size_t layer = 0U; layer < layer_count; ++layer)
        printf("section=benchmark layer=%zu duration_seconds=%.6f\n",
               layer, metrics.layer_seconds[layer]);

    cllm_whisper_small_model_close(&model);
    free(mel);
    free(output);
    return 0;
}
