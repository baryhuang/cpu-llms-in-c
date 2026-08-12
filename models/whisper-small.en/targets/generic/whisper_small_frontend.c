#include "whisper_small_frontend.h"

#include <math.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define CLLM_PI 3.14159265358979323846264338327950288
#define CLLM_WHISPER_CONV_KERNEL 3U

typedef struct {
    double real;
    double imaginary;
} cllm_complex;

static void fft_400_recursive(const cllm_complex *input,
                              size_t stride,
                              cllm_complex *output,
                              size_t count,
                              const cllm_complex *twiddle)
{
    cllm_complex sub[CLLM_WHISPER_N_FFT];
    size_t radix, inner;

    if (count == 1U) {
        output[0] = input[0];
        return;
    }
    radix = count % 5U == 0U ? 5U : 2U;
    inner = count / radix;
    for (size_t branch = 0U; branch < radix; ++branch)
        fft_400_recursive(input + branch * stride, stride * radix,
                          sub + branch * inner, inner, twiddle);

    for (size_t frequency = 0U; frequency < count; ++frequency) {
        cllm_complex sum = {0.0, 0.0};
        for (size_t branch = 0U; branch < radix; ++branch) {
            const size_t index = (branch * frequency *
                (CLLM_WHISPER_N_FFT / count)) % CLLM_WHISPER_N_FFT;
            const cllm_complex value = sub[branch * inner + frequency % inner];
            sum.real += value.real * twiddle[index].real -
                        value.imaginary * twiddle[index].imaginary;
            sum.imaginary += value.real * twiddle[index].imaginary +
                             value.imaginary * twiddle[index].real;
        }
        output[frequency] = sum;
    }
}

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

static float exact_gelu(float value)
{
    return 0.5f * value * (1.0f + erff(value * 0.70710678118654752440f));
}

size_t cllm_whisper_small_log_mel_frames(size_t sample_count)
{
    return sample_count / CLLM_WHISPER_HOP_LENGTH;
}

size_t cllm_whisper_small_log_mel_workspace_floats(size_t sample_count)
{
    const size_t frames = cllm_whisper_small_log_mel_frames(sample_count);
    if (frames > SIZE_MAX / CLLM_WHISPER_N_FREQUENCIES) {
        return 0U;
    }
    return frames * CLLM_WHISPER_N_FREQUENCIES;
}

int cllm_whisper_small_log_mel(const float *audio,
                               size_t sample_count,
                               const float *mel_filters,
                               float *output,
                               float *workspace,
                               size_t workspace_floats)
{
    const size_t frames = cllm_whisper_small_log_mel_frames(sample_count);
    const size_t required = cllm_whisper_small_log_mel_workspace_floats(sample_count);
    float window[CLLM_WHISPER_N_FFT];
    cllm_complex twiddle[CLLM_WHISPER_N_FFT];
    float maximum = -INFINITY;

    if (audio == NULL || mel_filters == NULL || output == NULL || workspace == NULL ||
        sample_count <= CLLM_WHISPER_N_FFT / 2U || frames == 0U || required == 0U ||
        workspace_floats < required) {
        return -1;
    }

    for (size_t index = 0U; index < CLLM_WHISPER_N_FFT; ++index) {
        const double angle = -2.0 * CLLM_PI * (double)index /
                             (double)CLLM_WHISPER_N_FFT;
        window[index] = 0.5f - 0.5f * cosf((float)-angle);
        twiddle[index].real = cos(angle);
        twiddle[index].imaginary = sin(angle);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t frame = 0U; frame < frames; ++frame) {
        const long frame_start = (long)(frame * CLLM_WHISPER_HOP_LENGTH) -
                                 (long)(CLLM_WHISPER_N_FFT / 2U);
        cllm_complex fft_input[CLLM_WHISPER_N_FFT];
        cllm_complex fft_output[CLLM_WHISPER_N_FFT];

        for (size_t index = 0U; index < CLLM_WHISPER_N_FFT; ++index) {
            const size_t source = reflect_index(frame_start + (long)index,
                                                sample_count);
            fft_input[index].real = (double)audio[source] * window[index];
            fft_input[index].imaginary = 0.0;
        }
        fft_400_recursive(fft_input, 1U, fft_output, CLLM_WHISPER_N_FFT, twiddle);
        for (size_t frequency = 0U; frequency < CLLM_WHISPER_N_FREQUENCIES;
             ++frequency) {
            const double real = fft_output[frequency].real;
            const double imaginary = fft_output[frequency].imaginary;
            workspace[frame * CLLM_WHISPER_N_FREQUENCIES + frequency] =
                (float)(real * real + imaginary * imaginary);
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(max:maximum)
#endif
    for (size_t mel = 0U; mel < CLLM_WHISPER_SMALL_N_MELS; ++mel) {
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

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t index = 0U; index < CLLM_WHISPER_SMALL_N_MELS * frames; ++index) {
        if (output[index] < maximum - 8.0f) {
            output[index] = maximum - 8.0f;
        }
        output[index] = (output[index] + 4.0f) / 4.0f;
    }
    return 0;
}

size_t cllm_whisper_encoder_stem_output_frames(size_t input_frames)
{
    return input_frames / 2U + input_frames % 2U;
}

size_t cllm_whisper_encoder_stem_scratch_floats(size_t input_frames,
                                                 size_t n_state)
{
    if (n_state != 0U && input_frames > SIZE_MAX / n_state) {
        return 0U;
    }
    return input_frames * n_state;
}

int cllm_whisper_encoder_stem(const float *mel,
                              size_t n_mels,
                              size_t input_frames,
                              size_t n_state,
                              const float *conv1_weight,
                              const float *conv1_bias,
                              const float *conv2_weight,
                              const float *conv2_bias,
                              const float *positions,
                              float *output,
                              float *scratch,
                              size_t scratch_floats)
{
    const size_t output_frames = cllm_whisper_encoder_stem_output_frames(input_frames);
    const size_t required = cllm_whisper_encoder_stem_scratch_floats(input_frames, n_state);

    if (mel == NULL || conv1_weight == NULL || conv1_bias == NULL ||
        conv2_weight == NULL || conv2_bias == NULL || positions == NULL ||
        output == NULL || scratch == NULL || n_mels == 0U || input_frames == 0U ||
        n_state == 0U || required == 0U || scratch_floats < required) {
        return -1;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t output_channel = 0U; output_channel < n_state; ++output_channel) {
        for (size_t frame = 0U; frame < input_frames; ++frame) {
            double sum = conv1_bias[output_channel];
            for (size_t input_channel = 0U; input_channel < n_mels; ++input_channel) {
                const float *kernel = conv1_weight +
                    (output_channel * n_mels + input_channel) * CLLM_WHISPER_CONV_KERNEL;
                for (size_t tap = 0U; tap < CLLM_WHISPER_CONV_KERNEL; ++tap) {
                    const long source = (long)frame + (long)tap - 1L;
                    if (source >= 0L && (size_t)source < input_frames) {
                        sum += (double)kernel[tap] *
                               (double)mel[input_channel * input_frames + (size_t)source];
                    }
                }
            }
            scratch[output_channel * input_frames + frame] = exact_gelu((float)sum);
        }
    }

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (size_t output_frame = 0U; output_frame < output_frames; ++output_frame) {
        for (size_t output_channel = 0U; output_channel < n_state; ++output_channel) {
            double sum = conv2_bias[output_channel];
            for (size_t input_channel = 0U; input_channel < n_state; ++input_channel) {
                const float *kernel = conv2_weight +
                    (output_channel * n_state + input_channel) * CLLM_WHISPER_CONV_KERNEL;
                for (size_t tap = 0U; tap < CLLM_WHISPER_CONV_KERNEL; ++tap) {
                    const long source = (long)(output_frame * 2U) + (long)tap - 1L;
                    if (source >= 0L && (size_t)source < input_frames) {
                        sum += (double)kernel[tap] *
                               (double)scratch[input_channel * input_frames + (size_t)source];
                    }
                }
            }
            output[output_frame * n_state + output_channel] =
                exact_gelu((float)sum) + positions[output_frame * n_state + output_channel];
        }
    }
    return 0;
}
