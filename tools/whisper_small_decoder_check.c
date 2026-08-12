#include "whisper_small_image.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HEADER_BYTES 32U

static uint32_t read_u32(const unsigned char *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8U |
           (uint32_t)p[2] << 16U | (uint32_t)p[3] << 24U;
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
        fseek(file, 0, SEEK_SET) != 0) { fclose(file); return NULL; }
    data = malloc((size_t)length);
    if (data == NULL || fread(data, 1U, (size_t)length, file) != (size_t)length) {
        free(data); fclose(file); return NULL;
    }
    fclose(file); *size = (size_t)length; return data;
}

int main(int argc, char **argv)
{
    cllm_whisper_small_model model;
    cllm_whisper_decoder_state state;
    cllm_whisper_decoder_metrics metrics;
    unsigned char *data;
    size_t size, audio_frames, token_count, expected_size;
    const float *encoder;
    const uint32_t *tokens;
    const float *expected_hidden;
    const uint32_t *expected_tokens;
    const float *expected_logits;
    float hidden[768];
    int exact, passed = 1;
    if (argc != 3) {
        fprintf(stderr, "usage: %s FULL_IMAGE.whenc DECODER_FIXTURE.bin\n", argv[0]);
        return 2;
    }
    data = read_file(argv[2], &size);
    if (data == NULL || size < HEADER_BYTES || memcmp(data, "WHDEC001", 8U) != 0 ||
        read_u32(data + 8U) != 1U) {
        fprintf(stderr, "error: invalid decoder fixture\n"); free(data); return 2;
    }
    audio_frames = read_u32(data + 12U);
    token_count = read_u32(data + 16U);
    /* Header: magic, version, audio_frames, token_count, width, vocab, boundary_count. */
    if (read_u32(data + 20U) != 768U || read_u32(data + 24U) != 51864U ||
        read_u32(data + 28U) != 3U || token_count == 0U) {
        fprintf(stderr, "error: incompatible decoder fixture\n"); free(data); return 2;
    }
    expected_size = HEADER_BYTES + audio_frames * 768U * sizeof(float) +
        token_count * sizeof(uint32_t) + token_count * 768U * sizeof(float) +
        token_count * sizeof(uint32_t) + token_count * sizeof(float);
    if (size != expected_size) {
        fprintf(stderr, "error: decoder fixture size mismatch\n"); free(data); return 2;
    }
    encoder = (const float *)(const void *)(data + HEADER_BYTES);
    tokens = (const uint32_t *)(const void *)(encoder + audio_frames * 768U);
    expected_hidden = (const float *)(const void *)(tokens + token_count);
    expected_tokens = (const uint32_t *)(const void *)(expected_hidden + token_count * 768U);
    expected_logits = (const float *)(const void *)(expected_tokens + token_count);
    if (cllm_whisper_small_model_open(argv[1], &model) != 0 || !model.has_decoder ||
        cllm_whisper_decoder_state_init(&model.decoder, encoder, audio_frames,
            token_count, &state, &metrics) != 0) {
        fprintf(stderr, "error: full image/decoder initialization failed\n"); free(data); return 2;
    }
    exact = model.image_version == 3U;
    printf("section=benchmark boundary=decoder_cross_cache duration_seconds=%.6f allocated_bytes=%zu\n",
           metrics.cross_cache_seconds, state.allocated_bytes);
    for (size_t step = 0U; step < token_count; ++step) {
        uint32_t actual_token;
        float actual_logit;
        double maximum = 0.0, square = 0.0;
        if (cllm_whisper_decoder_step(&model.decoder, &state, tokens[step],
                &actual_token, &actual_logit, hidden, &metrics) != 0) {
            passed = 0; break;
        }
        for (size_t index = 0U; index < 768U; ++index) {
            const double delta = (double)hidden[index] - expected_hidden[step * 768U + index];
            if (fabs(delta) > maximum) maximum = fabs(delta);
            square += delta * delta;
        }
        if (exact && (maximum > 5.0e-4 || actual_token != expected_tokens[step] ||
            fabs((double)actual_logit - expected_logits[step]) > 2.0e-3)) passed = 0;
        if (!isfinite(actual_logit)) passed = 0;
        printf("section=verification boundary=decoder_step step=%zu input_token=%u "
               "expected_next_token=%u actual_next_token=%u hidden_max_abs=%.9g "
               "hidden_rmse=%.9g expected_top_logit=%.9g actual_top_logit=%.9g status=%s\n",
               step, tokens[step], expected_tokens[step], actual_token, maximum,
               sqrt(square / 768.0), expected_logits[step], actual_logit,
               passed ? (exact ? "PASS" : "FINITE_MEASURED") : "FAIL");
        printf("section=benchmark boundary=decoder_step step=%zu duration_seconds=%.6f "
               "output_head_seconds=%.6f\n", step, metrics.step_seconds,
               metrics.output_head_seconds);
    }
    printf("VERDICT: %s\n", passed ? (exact ? "PASS" : "FINITE_MEASURED_NOT_QUALITY_GATE") : "FAIL");
    cllm_whisper_decoder_state_free(&state);
    cllm_whisper_small_model_close(&model);
    free(data);
    return passed ? 0 : 1;
}
