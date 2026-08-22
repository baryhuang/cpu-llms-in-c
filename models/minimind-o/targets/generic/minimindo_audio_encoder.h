#ifndef LLM_IN_C_MINIMINDO_AUDIO_ENCODER_H
#define LLM_IN_C_MINIMINDO_AUDIO_ENCODER_H

#include <stddef.h>
#include <stdint.h>

typedef struct minimindo_audio_encoder minimindo_audio_encoder;

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

void minimindo_audio_encoder_last_profile(
    const minimindo_audio_encoder *model,
    minimindo_audio_encoder_profile *profile);

#endif
