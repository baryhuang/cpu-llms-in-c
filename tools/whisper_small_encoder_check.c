#include "whisper_small_image.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FIXTURE_HEADER_BYTES 32U

static uint32_t read_u32(const unsigned char *bytes)
{
    return (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8U |
           (uint32_t)bytes[2] << 16U | (uint32_t)bytes[3] << 24U;
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

static int compare(const char *name, const float *actual, const float *expected,
                   size_t count, double duration, size_t workspace, int exact)
{
    double maximum_absolute = 0.0;
    double root_square = 0.0;
    int passed = 1;
    for (size_t index = 0U; index < count; ++index) {
        const double delta = (double)actual[index] - expected[index];
        const double absolute = fabs(delta);
        const double tolerance = 2.0e-4 + 2.0e-4 * fabs((double)expected[index]);
        if (absolute > maximum_absolute) maximum_absolute = absolute;
        root_square += delta * delta;
        if (!isfinite(actual[index]) || (exact && absolute > tolerance)) passed = 0;
    }
    printf("section=verification boundary=%s count=%zu max_abs=%.9g rmse=%.9g status=%s\n",
           name, count, maximum_absolute, sqrt(root_square / (double)count),
           passed ? (exact ? "PASS" : "FINITE_MEASURED") : "FAIL");
    printf("section=benchmark boundary=%s duration_seconds=%.6f workspace_bytes=%zu\n",
           name, duration, workspace);
    return passed;
}

int main(int argc, char **argv)
{
    cllm_whisper_small_model model;
    cllm_whisper_small_encoder_metrics metrics;
    unsigned char *fixture = NULL;
    size_t fixture_bytes;
    size_t input_frames, output_frames, state_count, mel_count, expected_bytes;
    const float *mel;
    const float *expected[3];
    const size_t layers[3] = {0U, 1U, 12U};
    const char *names[3] = {"real_stem", "real_layer_0", "real_layer_11_final_ln"};
    float *actual = NULL;
    int passed = 1;

    if (argc != 3) {
        fprintf(stderr, "usage: %s IMAGE.whenc FIXTURE.bin\n", argv[0]);
        return 2;
    }
    fixture = read_file(argv[2], &fixture_bytes);
    if (fixture == NULL || fixture_bytes < FIXTURE_HEADER_BYTES ||
        memcmp(fixture, "WHREAL01", 8U) != 0 || read_u32(fixture + 8U) != 1U ||
        read_u32(fixture + 24U) != 3U) {
        fprintf(stderr, "error: invalid WHREAL01 fixture\n");
        free(fixture);
        return 2;
    }
    input_frames = read_u32(fixture + 12U);
    output_frames = read_u32(fixture + 16U);
    if (read_u32(fixture + 20U) != 768U || output_frames != (input_frames + 1U) / 2U) {
        fprintf(stderr, "error: incompatible fixture dimensions\n");
        free(fixture);
        return 2;
    }
    mel_count = input_frames * 80U;
    state_count = output_frames * 768U;
    expected_bytes = FIXTURE_HEADER_BYTES + (mel_count + state_count * 3U) * sizeof(float);
    if (fixture_bytes != expected_bytes) {
        fprintf(stderr, "error: fixture size mismatch\n");
        free(fixture);
        return 2;
    }
    mel = (const float *)(const void *)(fixture + FIXTURE_HEADER_BYTES);
    expected[0] = mel + mel_count;
    expected[1] = expected[0] + state_count;
    expected[2] = expected[1] + state_count;
    actual = malloc(state_count * sizeof(float));
    if (actual == NULL || cllm_whisper_small_model_open(argv[1], &model) != 0) {
        fprintf(stderr, "error: cannot open exact encoder image\n");
        free(actual);
        free(fixture);
        return 2;
    }
    for (size_t boundary = 0U; boundary < 3U; ++boundary) {
        double duration = 0.0;
        if (cllm_whisper_small_encode_mel(&model, mel, input_frames, layers[boundary],
                actual, &metrics) != 0) {
            fprintf(stderr, "error: encoder execution failed at %s\n", names[boundary]);
            passed = 0;
            break;
        }
        duration = metrics.stem_seconds + metrics.final_norm_seconds;
        for (size_t layer = 0U; layer < metrics.executed_layers; ++layer)
            duration += metrics.layer_seconds[layer];
        if (!compare(names[boundary], actual, expected[boundary], state_count,
                     duration, metrics.workspace_bytes, model.image_version == 1U))
            passed = 0;
    }
    if (!passed)
        printf("VERDICT: FAIL\n");
    else if (model.image_version == 1U)
        printf("VERDICT: PASS\n");
    else
        printf("VERDICT: FINITE_MEASURED_NOT_QUALITY_GATE\n");
    cllm_whisper_small_model_close(&model);
    free(actual);
    free(fixture);
    return passed ? 0 : 1;
}
