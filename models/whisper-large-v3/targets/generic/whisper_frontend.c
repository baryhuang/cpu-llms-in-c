#include "whisper_frontend.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define CLLM_PI 3.14159265358979323846264338327950288

static size_t reflect_index(long index, size_t count)
{
    const long last = (long)count - 1L;

    while (index < 0L || index > last) {
        if (index < 0L) {
            index = -index;
        } else {
            index = 2L * last - index;
        }
    }
    return (size_t)index;
}

size_t cllm_whisper_log_mel_frames(size_t sample_count)
{
    /* torch.stft(center=true) emits floor(N / hop) + 1 frames. OpenAI
       Whisper drops the last frame with stft[..., :-1]. */
    return sample_count / CLLM_WHISPER_HOP_LENGTH;
}

size_t cllm_whisper_log_mel_workspace_floats(size_t sample_count)
{
    const size_t frames = cllm_whisper_log_mel_frames(sample_count);
    if (frames > SIZE_MAX / CLLM_WHISPER_N_FREQUENCIES) {
        return 0U;
    }
    return frames * CLLM_WHISPER_N_FREQUENCIES;
}

int cllm_whisper_log_mel_128(const float *audio,
                             size_t sample_count,
                             const float *mel_filters,
                             float *output,
                             float *workspace,
                             size_t workspace_floats)
{
    const size_t frames = cllm_whisper_log_mel_frames(sample_count);
    const size_t required = cllm_whisper_log_mel_workspace_floats(sample_count);
    float maximum = -INFINITY;

    if (audio == NULL || mel_filters == NULL || output == NULL || workspace == NULL ||
        sample_count <= CLLM_WHISPER_N_FFT / 2U || frames == 0U || required == 0U ||
        workspace_floats < required) {
        return -1;
    }

    for (size_t frame = 0U; frame < frames; ++frame) {
        const long frame_start = (long)(frame * CLLM_WHISPER_HOP_LENGTH) -
                                 (long)(CLLM_WHISPER_N_FFT / 2U);

        for (size_t frequency = 0U; frequency < CLLM_WHISPER_N_FREQUENCIES;
             ++frequency) {
            double real = 0.0;
            double imaginary = 0.0;

            for (size_t index = 0U; index < CLLM_WHISPER_N_FFT; ++index) {
                const float window = 0.5f - 0.5f *
                    cosf((float)(2.0 * CLLM_PI * (double)index /
                                 (double)CLLM_WHISPER_N_FFT));
                const size_t source = reflect_index(frame_start + (long)index,
                                                    sample_count);
                const float value = audio[source] * window;
                const double angle = -2.0 * CLLM_PI * (double)frequency *
                                     (double)index / (double)CLLM_WHISPER_N_FFT;
                real += (double)value * cos(angle);
                imaginary += (double)value * sin(angle);
            }
            workspace[frame * CLLM_WHISPER_N_FREQUENCIES + frequency] =
                (float)(real * real + imaginary * imaginary);
        }
    }

    for (size_t mel = 0U; mel < CLLM_WHISPER_LARGE_V3_N_MELS; ++mel) {
        const float *filter = mel_filters + mel * CLLM_WHISPER_N_FREQUENCIES;
        for (size_t frame = 0U; frame < frames; ++frame) {
            double sum = 0.0;
            for (size_t frequency = 0U; frequency < CLLM_WHISPER_N_FREQUENCIES;
                 ++frequency) {
                sum += (double)filter[frequency] *
                       (double)workspace[frame * CLLM_WHISPER_N_FREQUENCIES + frequency];
            }
            if (sum < 1.0e-10) {
                sum = 1.0e-10;
            }
            output[mel * frames + frame] = log10f((float)sum);
            if (output[mel * frames + frame] > maximum) {
                maximum = output[mel * frames + frame];
            }
        }
    }

    for (size_t index = 0U; index < CLLM_WHISPER_LARGE_V3_N_MELS * frames; ++index) {
        if (output[index] < maximum - 8.0f) {
            output[index] = maximum - 8.0f;
        }
        output[index] = (output[index] + 4.0f) / 4.0f;
    }
    return 0;
}
