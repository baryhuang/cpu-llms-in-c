#define _POSIX_C_SOURCE 200809L

#include "whisper_small_image.h"
#include "whisper_small_frontend.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { SAMPLE_RATE = 16000, MAX_SAMPLES = 480000, EOT = 50256,
       SOT = 50257, NO_TIMESTAMPS = 50362, TIMESTAMP_BEGIN = 50363 };

static const uint32_t suppress_tokens[] = {
    1,2,7,8,9,10,14,25,26,27,28,29,31,58,59,60,61,62,63,90,91,92,93,
    357,366,438,532,685,705,796,930,1058,1220,1267,1279,1303,1343,1377,
    1391,1635,1782,1875,2162,2361,2488,3467,4008,4211,4600,4808,5299,
    5855,6329,7203,9609,9959,10563,10786,11420,11709,11907,13163,13697,
    13700,14808,15306,16410,16791,17992,19203,19510,20724,22305,22935,
    27007,30109,30420,33409,34949,40283,40493,40549,47282,49146,
    50257,50357,50358,50359,50360,50361
};

static double monotonic_seconds(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) return 0.0;
    return (double)now.tv_sec + (double)now.tv_nsec * 1.0e-9;
}

static uint16_t u16le(const unsigned char *p)
{
    return (uint16_t)p[0] | (uint16_t)p[1] << 8U;
}

static uint32_t u32le(const unsigned char *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8U |
           (uint32_t)p[2] << 16U | (uint32_t)p[3] << 24U;
}

static int read_wav_pcm16(const char *path, int compact_window,
                          float **audio, size_t *sample_count,
                          double *original_seconds)
{
    FILE *file = fopen(path, "rb");
    unsigned char header[12];
    uint16_t format = 0U, channels = 0U, bits = 0U;
    uint32_t sample_rate = 0U;
    unsigned char *pcm = NULL;
    size_t pcm_bytes = 0U;
    if (file == NULL) {
        fprintf(stderr, "error: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    if (fread(header, 1U, sizeof(header), file) != sizeof(header) ||
        memcmp(header, "RIFF", 4U) != 0 || memcmp(header + 8U, "WAVE", 4U) != 0)
        goto fail;
    for (;;) {
        unsigned char chunk[8];
        uint32_t bytes;
        if (fread(chunk, 1U, sizeof(chunk), file) != sizeof(chunk)) break;
        bytes = u32le(chunk + 4U);
        if (memcmp(chunk, "fmt ", 4U) == 0) {
            unsigned char fmt[40];
            if (bytes < 16U || bytes > sizeof(fmt) || fread(fmt, 1U, bytes, file) != bytes)
                goto fail;
            format = u16le(fmt);
            channels = u16le(fmt + 2U);
            sample_rate = u32le(fmt + 4U);
            bits = u16le(fmt + 14U);
        } else if (memcmp(chunk, "data", 4U) == 0) {
            pcm = malloc(bytes);
            if (pcm == NULL || fread(pcm, 1U, bytes, file) != bytes) goto fail;
            pcm_bytes = bytes;
        } else if (fseek(file, bytes, SEEK_CUR) != 0) {
            goto fail;
        }
        if (bytes & 1U) fseek(file, 1L, SEEK_CUR);
        if (pcm != NULL && format != 0U) break;
    }
    fclose(file);
    if (format != 1U || channels != 1U || sample_rate != SAMPLE_RATE || bits != 16U ||
        pcm == NULL || pcm_bytes < 2U) {
        fprintf(stderr, "error: WAV must be mono 16-bit PCM at 16000 Hz\n");
        free(pcm);
        return -1;
    }
    const size_t source_samples = pcm_bytes / 2U;
    const size_t used_samples = source_samples < MAX_SAMPLES ? source_samples : MAX_SAMPLES;
    if (used_samples <= CLLM_WHISPER_N_FFT / 2U) {
        fprintf(stderr, "error: WAV is too short for the 400-sample analysis window\n");
        free(pcm);
        return -1;
    }
    *original_seconds = (double)used_samples / SAMPLE_RATE;
    *sample_count = compact_window ? used_samples : MAX_SAMPLES;
    *audio = calloc(*sample_count, sizeof(float));
    if (*audio == NULL) { free(pcm); return -1; }
    for (size_t index = 0U; index < source_samples && index < MAX_SAMPLES; ++index) {
        const int16_t value = (int16_t)u16le(pcm + index * 2U);
        (*audio)[index] = (float)value / 32768.0f;
    }
    free(pcm);
    return 0;
fail:
    fprintf(stderr, "error: invalid or unsupported WAV container\n");
    free(pcm); fclose(file); return -1;
}

static void build_suppression(unsigned char *suppressed, int first_text_token)
{
    memset(suppressed, 0, CLLM_WHISPER_SMALL_VOCABULARY);
    for (size_t index = 0U; index < sizeof(suppress_tokens) / sizeof(suppress_tokens[0]); ++index)
        suppressed[suppress_tokens[index]] = 1U;
    suppressed[NO_TIMESTAMPS] = 1U;
    for (size_t token = TIMESTAMP_BEGIN; token < CLLM_WHISPER_SMALL_VOCABULARY; ++token)
        suppressed[token] = 1U;
    if (first_text_token) {
        suppressed[220U] = 1U;
        suppressed[EOT] = 1U;
    }
}

static int append_piece(const cllm_whisper_decoder_weights *decoder,
                        uint32_t token, unsigned char **text, size_t *length, size_t *capacity)
{
    size_t start, end, piece;
    unsigned char *grown;
    if (token >= CLLM_WHISPER_SMALL_VOCABULARY || decoder->token_offsets == NULL)
        return -1;
    if (decoder->token_special[token] || token >= TIMESTAMP_BEGIN) return 0;
    start = decoder->token_offsets[token];
    end = decoder->token_offsets[token + 1U];
    if (end < start) return -1;
    piece = end - start;
    if (*length + piece + 1U > *capacity) {
        size_t next = *capacity == 0U ? 256U : *capacity;
        while (next < *length + piece + 1U) next *= 2U;
        grown = realloc(*text, next);
        if (grown == NULL) return -1;
        *text = grown; *capacity = next;
    }
    memcpy(*text + *length, decoder->token_bytes + start, piece);
    *length += piece;
    (*text)[*length] = '\0';
    return 0;
}

int main(int argc, char **argv)
{
    cllm_whisper_small_model model;
    cllm_whisper_small_encoder_metrics encoder_metrics;
    cllm_whisper_decoder_state decoder_state;
    cllm_whisper_decoder_metrics decoder_metrics;
    float *audio = NULL, *mel = NULL, *mel_workspace = NULL, *encoder = NULL;
    unsigned char *suppressed = NULL, *text = NULL;
    size_t samples, input_frames, output_frames, maximum_tokens = 64U;
    size_t text_length = 0U, text_capacity = 0U;
    double audio_seconds, frontend_started, frontend_seconds, encoder_seconds = 0.0;
    double decoder_seconds = 0.0, output_head_seconds = 0.0, total_seconds;
    double run_started;
    uint32_t token = SOT, next_token;
    float next_logit;
    int compact_window = 0;
    int result = 2;
    memset(&model, 0, sizeof(model));
    model.image.fd = -1;
    memset(&decoder_state, 0, sizeof(decoder_state));
    if (argc < 3 || argc > 5) {
        fprintf(stderr, "usage: %s FULL_IMAGE.whenc AUDIO.wav [MAX_TOKENS] "
                        "[fixed30|compact]\n", argv[0]);
        return 2;
    }
    if (argc >= 4) {
        maximum_tokens = strtoul(argv[3], NULL, 10);
        if (maximum_tokens < 3U || maximum_tokens > 448U) return 2;
    }
    if (argc == 5) {
        if (strcmp(argv[4], "compact") == 0) compact_window = 1;
        else if (strcmp(argv[4], "fixed30") != 0) return 2;
    }
    if (read_wav_pcm16(argv[2], compact_window, &audio, &samples, &audio_seconds) != 0 ||
        cllm_whisper_small_model_open(argv[1], &model) != 0 || !model.has_decoder)
        goto cleanup;
    run_started = monotonic_seconds();
    input_frames = cllm_whisper_small_log_mel_frames(samples);
    output_frames = cllm_whisper_encoder_stem_output_frames(input_frames);
    mel = malloc(80U * input_frames * sizeof(float));
    mel_workspace = malloc(cllm_whisper_small_log_mel_workspace_floats(samples) * sizeof(float));
    encoder = malloc(output_frames * 768U * sizeof(float));
    suppressed = malloc(CLLM_WHISPER_SMALL_VOCABULARY);
    if (mel == NULL || mel_workspace == NULL || encoder == NULL || suppressed == NULL)
        goto cleanup;

    frontend_started = monotonic_seconds();
    if (cllm_whisper_small_log_mel(audio, samples, model.mel_filters, mel, mel_workspace,
            cllm_whisper_small_log_mel_workspace_floats(samples)) != 0)
        goto cleanup;
    frontend_seconds = monotonic_seconds() - frontend_started;
    if (cllm_whisper_small_encode_mel(&model, mel, input_frames, 12U,
                                      encoder, &encoder_metrics) != 0)
        goto cleanup;
    encoder_seconds = encoder_metrics.stem_seconds + encoder_metrics.final_norm_seconds;
    for (size_t layer = 0U; layer < 12U; ++layer) encoder_seconds += encoder_metrics.layer_seconds[layer];
    if (cllm_whisper_decoder_state_init(&model.decoder, encoder, output_frames,
            maximum_tokens, &decoder_state, &decoder_metrics) != 0)
        goto cleanup;
    decoder_seconds += decoder_metrics.cross_cache_seconds;

    /* Consume forced prefix. Ignore the SOT prediction; force no-timestamps at position 1. */
    if (cllm_whisper_decoder_consume(&model.decoder, &decoder_state, SOT,
                                     NULL, &decoder_metrics) != 0)
        goto cleanup_state;
    decoder_seconds += decoder_metrics.step_seconds;
    token = NO_TIMESTAMPS;
    for (size_t generated = 0U; generated + 2U < maximum_tokens; ++generated) {
        build_suppression(suppressed, generated == 0U);
        if (cllm_whisper_decoder_step_filtered(&model.decoder, &decoder_state, token,
                suppressed, &next_token, &next_logit, NULL, &decoder_metrics) != 0)
            goto cleanup_state;
        decoder_seconds += decoder_metrics.step_seconds;
        output_head_seconds += decoder_metrics.output_head_seconds;
        printf("section=decode position=%zu input_token=%u next_token=%u top_logit=%.9g "
               "duration_seconds=%.6f output_head_seconds=%.6f\n",
               decoder_state.token_count - 1U, token, next_token, next_logit,
               decoder_metrics.step_seconds, decoder_metrics.output_head_seconds);
        if (next_token == EOT) break;
        if (append_piece(&model.decoder, next_token, &text, &text_length, &text_capacity) != 0)
            goto cleanup_state;
        token = next_token;
    }
    if (text == NULL) {
        text = calloc(1U, 1U);
        if (text == NULL) goto cleanup_state;
    }
    total_seconds = monotonic_seconds() - run_started;
    printf("section=result window=%s audio_seconds=%.6f input_frames=%zu output_frames=%zu "
           "frontend_seconds=%.6f encoder_seconds=%.6f decoder_core_seconds=%.6f "
           "output_head_seconds=%.6f total_seconds=%.6f rtf=%.6f "
           "decoder_cache_bytes=%zu generated_tokens=%zu\n",
           compact_window ? "compact" : "fixed30", audio_seconds,
           input_frames, output_frames, frontend_seconds,
           encoder_seconds, decoder_seconds, output_head_seconds,
           total_seconds, total_seconds / audio_seconds,
           decoder_state.allocated_bytes, decoder_state.token_count - 2U);
    printf("transcript_utf8_bytes=%zu\n", text_length);
    printf("TRANSCRIPT: ");
    fwrite(text, 1U, text_length, stdout);
    putchar('\n');
    result = 0;

cleanup_state:
    cllm_whisper_decoder_state_free(&decoder_state);
cleanup:
    if (model.image.base != NULL) cllm_whisper_small_model_close(&model);
    free(audio); free(mel); free(mel_workspace); free(encoder); free(suppressed); free(text);
    return result;
}
