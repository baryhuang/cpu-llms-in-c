#include "whisper_small_frontend.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HEADER_BYTES 36U

static uint32_t read_u32(const unsigned char *bytes)
{
    return (uint32_t)bytes[0] |
           (uint32_t)bytes[1] << 8U |
           (uint32_t)bytes[2] << 16U |
           (uint32_t)bytes[3] << 24U;
}

static unsigned char *read_file(const char *path, size_t *size)
{
    FILE *file = fopen(path, "rb");
    unsigned char *data;
    long length;

    if (file == NULL) {
        fprintf(stderr, "error: cannot open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    data = malloc((size_t)length);
    if (data == NULL || fread(data, 1U, (size_t)length, file) != (size_t)length) {
        free(data);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *size = (size_t)length;
    return data;
}

int main(int argc, char **argv)
{
    unsigned char *data;
    size_t size;
    size_t n_mels;
    size_t n_state;
    size_t input_frames;
    size_t output_frames;
    size_t offset = HEADER_BYTES;
    size_t audio_count;
    size_t conv1_count;
    size_t conv2_count;
    size_t output_count;
    const float *audio;
    const float *conv1_weight;
    const float *conv1_bias;
    const float *conv2_weight;
    const float *conv2_bias;
    const float *positions;
    const float *expected;
    float *actual;
    float *scratch;
    double maximum_absolute = 0.0;
    double maximum_relative = 0.0;
    int passed = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s FIXTURE\n", argv[0]);
        return 2;
    }
    data = read_file(argv[1], &size);
    if (data == NULL || size < HEADER_BYTES || memcmp(data, "WHSTEM01", 8U) != 0 ||
        read_u32(data + 8U) != 1U || read_u32(data + 32U) != 2U) {
        fprintf(stderr, "error: invalid Whisper encoder-stem fixture\n");
        free(data);
        return 2;
    }
    n_mels = read_u32(data + 12U);
    n_state = read_u32(data + 16U);
    input_frames = read_u32(data + 20U);
    output_frames = read_u32(data + 24U);
    if (read_u32(data + 28U) != 3U ||
        output_frames != cllm_whisper_encoder_stem_output_frames(input_frames)) {
        fprintf(stderr, "error: incompatible Whisper encoder-stem dimensions\n");
        free(data);
        return 2;
    }

    audio_count = n_mels * input_frames;
    conv1_count = n_state * n_mels * 3U;
    conv2_count = n_state * n_state * 3U;
    output_count = output_frames * n_state;
    if (size != HEADER_BYTES +
                (audio_count + conv1_count + n_state + conv2_count + n_state +
                 output_count + output_count) * sizeof(float)) {
        fprintf(stderr, "error: Whisper encoder-stem fixture size mismatch\n");
        free(data);
        return 2;
    }

#define TAKE_FLOATS(name, count) \
    name = (const float *)(const void *)(data + offset); \
    offset += (count) * sizeof(float)
    TAKE_FLOATS(audio, audio_count);
    TAKE_FLOATS(conv1_weight, conv1_count);
    TAKE_FLOATS(conv1_bias, n_state);
    TAKE_FLOATS(conv2_weight, conv2_count);
    TAKE_FLOATS(conv2_bias, n_state);
    TAKE_FLOATS(positions, output_count);
    TAKE_FLOATS(expected, output_count);
#undef TAKE_FLOATS

    actual = malloc(output_count * sizeof(float));
    scratch = malloc(cllm_whisper_encoder_stem_scratch_floats(input_frames, n_state) * sizeof(float));
    if (actual == NULL || scratch == NULL ||
        cllm_whisper_encoder_stem(audio, n_mels, input_frames, n_state,
                                  conv1_weight, conv1_bias, conv2_weight, conv2_bias,
                                  positions, actual, scratch,
                                  cllm_whisper_encoder_stem_scratch_floats(input_frames, n_state)) != 0) {
        fprintf(stderr, "error: Whisper encoder-stem execution failed\n");
        free(actual);
        free(scratch);
        free(data);
        return 2;
    }

    for (size_t index = 0U; index < output_count; ++index) {
        const double absolute = fabs((double)actual[index] - (double)expected[index]);
        const double relative = absolute / fmax(fabs((double)expected[index]), 1.0e-12);
        const double tolerance = 3.0e-6 + 3.0e-5 * fabs((double)expected[index]);
        if (absolute > maximum_absolute) maximum_absolute = absolute;
        if (relative > maximum_relative) maximum_relative = relative;
        if (!isfinite(actual[index]) || absolute > tolerance) passed = 0;
    }
    printf("boundary=whisper_encoder_stem count=%zu max_abs=%.9g max_rel=%.9g status=%s\n",
           output_count, maximum_absolute, maximum_relative, passed ? "PASS" : "FAIL");
    printf("VERDICT: %s\n", passed ? "PASS" : "FAIL");

    free(actual);
    free(scratch);
    free(data);
    return passed ? 0 : 1;
}
