#ifndef LLM_IN_C_MINIMINDO_AUDIO_ENCODER_H
#define LLM_IN_C_MINIMINDO_AUDIO_ENCODER_H

#include <stddef.h>
#include <stdint.h>

typedef struct minimindo_audio_encoder minimindo_audio_encoder;
typedef struct minimindo_audio_encoder_stream minimindo_audio_encoder_stream;

typedef struct {
    double frontend_ms;
    double encoder_ms;
    double projector_ms;
} minimindo_audio_encoder_profile;

minimindo_audio_encoder *minimindo_audio_encoder_open(const char *image_path,
                                                       char *error,
                                                       size_t error_capacity);
void minimindo_audio_encoder_close(minimindo_audio_encoder *model);

size_t minimindo_audio_encoder_frames(size_t sample_count);

/* Output is frame-major [frames][768], ready for Thinker embedding injection. */
int minimindo_audio_encoder_encode_pcm16(minimindo_audio_encoder *model,
                                         const int16_t *samples,
                                         size_t sample_count,
                                         float *output,
                                         size_t output_count,
                                         size_t *output_frames,
                                         char *error,
                                         size_t error_capacity);

/*
 * Native online SenseVoice path.  Audio is committed in 480 ms center chunks
 * with 240 ms of right lookahead.  Only embeddings whose right context is
 * complete are returned; end_of_stream flushes the final tail.  The encoder
 * keeps truncated per-layer K/V history, so callers can immediately prefill
 * every returned [frame][768] embedding into the language models.
 */
minimindo_audio_encoder_stream *minimindo_audio_encoder_stream_open(
    minimindo_audio_encoder *model, char *error, size_t error_capacity);
void minimindo_audio_encoder_stream_close(
    minimindo_audio_encoder_stream *stream);
int minimindo_audio_encoder_stream_push_pcm16(
    minimindo_audio_encoder_stream *stream,
    const int16_t *samples,
    size_t sample_count,
    int end_of_stream,
    float *output,
    size_t output_count,
    size_t *output_frames,
    char *error,
    size_t error_capacity);
size_t minimindo_audio_encoder_stream_total_frames(
    const minimindo_audio_encoder_stream *stream);
void minimindo_audio_encoder_stream_profile(
    const minimindo_audio_encoder_stream *stream,
    minimindo_audio_encoder_profile *profile);

void minimindo_audio_encoder_last_profile(
    const minimindo_audio_encoder *model,
    minimindo_audio_encoder_profile *profile);

#endif
