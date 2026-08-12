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
    uint32_t samples;
    uint32_t frames;
    size_t audio_bytes;
    size_t filter_bytes;
    size_t output_count;
    size_t output_bytes;
    const float *audio;
    const float *filters;
    const float *expected;
    float *actual;
    float *workspace;
    double maximum_absolute = 0.0;
    double maximum_relative = 0.0;
    int passed = 1;

    if (argc != 2) {
        fprintf(stderr, "usage: %s FIXTURE\n", argv[0]);
        return 2;
    }
    data = read_file(argv[1], &size);
    if (data == NULL || size < HEADER_BYTES || memcmp(data, "WHMEL001", 8U) != 0 ||
        read_u32(data + 8U) != 1U) {
        fprintf(stderr, "error: invalid Whisper small.en log-Mel fixture\n");
        free(data);
        return 2;
    }
    samples = read_u32(data + 12U);
    frames = read_u32(data + 16U);
    if (read_u32(data + 20U) != CLLM_WHISPER_N_FFT ||
        read_u32(data + 24U) != CLLM_WHISPER_HOP_LENGTH ||
        read_u32(data + 28U) != CLLM_WHISPER_SMALL_N_MELS ||
        read_u32(data + 32U) != CLLM_WHISPER_N_FREQUENCIES ||
        frames != cllm_whisper_small_log_mel_frames(samples)) {
        fprintf(stderr, "error: incompatible Whisper small.en log-Mel dimensions\n");
        free(data);
        return 2;
    }

    audio_bytes = (size_t)samples * sizeof(float);
    filter_bytes = CLLM_WHISPER_SMALL_N_MELS * CLLM_WHISPER_N_FREQUENCIES * sizeof(float);
    output_count = (size_t)frames * CLLM_WHISPER_SMALL_N_MELS;
    output_bytes = output_count * sizeof(float);
    if (size != HEADER_BYTES + audio_bytes + filter_bytes + output_bytes) {
        fprintf(stderr, "error: Whisper small.en log-Mel fixture size mismatch\n");
        free(data);
        return 2;
    }

    audio = (const float *)(const void *)(data + HEADER_BYTES);
    filters = (const float *)(const void *)(data + HEADER_BYTES + audio_bytes);
    expected = (const float *)(const void *)(data + HEADER_BYTES + audio_bytes + filter_bytes);
    actual = malloc(output_bytes);
    workspace = malloc(cllm_whisper_small_log_mel_workspace_floats(samples) * sizeof(float));
    if (actual == NULL || workspace == NULL ||
        cllm_whisper_small_log_mel(audio, samples, filters, actual, workspace,
                                   cllm_whisper_small_log_mel_workspace_floats(samples)) != 0) {
        fprintf(stderr, "error: Whisper small.en log-Mel execution failed\n");
        free(actual);
        free(workspace);
        free(data);
        return 2;
    }

    for (size_t index = 0U; index < output_count; ++index) {
        const double absolute = fabs((double)actual[index] - (double)expected[index]);
        const double relative = absolute / fmax(fabs((double)expected[index]), 1.0e-12);
        const double tolerance = 5.0e-5 + 3.0e-5 * fabs((double)expected[index]);
        if (absolute > maximum_absolute) maximum_absolute = absolute;
        if (relative > maximum_relative) maximum_relative = relative;
        if (!isfinite(actual[index]) || absolute > tolerance) passed = 0;
    }
    printf("boundary=small_en_log_mel_80 count=%zu max_abs=%.9g max_rel=%.9g status=%s\n",
           output_count, maximum_absolute, maximum_relative, passed ? "PASS" : "FAIL");
    printf("VERDICT: %s\n", passed ? "PASS" : "FAIL");

    free(actual);
    free(workspace);
    free(data);
    return passed ? 0 : 1;
}
