/* Continuous PCM16 capture segmenter for the ThreeHub A113X voice assistant.
 *
 * A single arecord process owns ALSA for the lifetime of the service and pipes
 * raw 16 kHz mono PCM here. This process always drains that stream, including
 * while inference and speaker playback are active. Amplitude gating creates
 * atomic WAV files for the production Whisper/Silero VAD stage.
 */

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <dirent.h>
#include <fcntl.h>
#include <math.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum {
    SAMPLE_RATE = 16000,
    FRAME_SAMPLES = 160,
    DEFAULT_START_MS = 100,
    DEFAULT_STOP_MS = 1200,
    DEFAULT_PRE_ROLL_MS = 300,
    DEFAULT_MAX_SECONDS = 35
};

struct options {
    const char *spool_dir;
    const char *status_path;
    const char *mute_path;
    double start_percent;
    double stop_percent;
    int start_ms;
    int stop_ms;
    int pre_roll_ms;
    int max_seconds;
    int max_queued;
};

static volatile sig_atomic_t stop_requested;

static void handle_signal(int signal_number)
{
    (void)signal_number;
    stop_requested = 1;
}

static int write_all(int fd, const void *buffer, size_t bytes)
{
    const uint8_t *at = buffer;
    while (bytes > 0U) {
        const ssize_t written = write(fd, at, bytes);
        if (written < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        at += (size_t)written;
        bytes -= (size_t)written;
    }
    return 0;
}

static void put_u16le(uint8_t *output, uint16_t value)
{
    output[0] = (uint8_t)value;
    output[1] = (uint8_t)(value >> 8U);
}

static void put_u32le(uint8_t *output, uint32_t value)
{
    output[0] = (uint8_t)value;
    output[1] = (uint8_t)(value >> 8U);
    output[2] = (uint8_t)(value >> 16U);
    output[3] = (uint8_t)(value >> 24U);
}

static int write_wav(const struct options *options, const int16_t *samples,
                     size_t sample_count, unsigned long long sequence)
{
    struct timespec now;
    if (clock_gettime(CLOCK_REALTIME, &now) != 0) {
        return -1;
    }

    char final_path[1024];
    char temporary_path[1024];
    const int final_size = snprintf(final_path, sizeof(final_path),
                                    "%s/utterance-%lld-%09ld-%06llu.wav",
                                    options->spool_dir, (long long)now.tv_sec, now.tv_nsec,
                                    sequence);
    if (final_size < 0 || (size_t)final_size >= sizeof(final_path)) {
        errno = ENAMETOOLONG;
        return -1;
    }
    const int temporary_size = snprintf(temporary_path, sizeof(temporary_path), "%s.part",
                                        final_path);
    if (temporary_size < 0 || (size_t)temporary_size >= sizeof(temporary_path)) {
        errno = ENAMETOOLONG;
        return -1;
    }

    const uint64_t pcm_bytes64 = sample_count * sizeof(int16_t);
    if (pcm_bytes64 > UINT32_MAX - 36U) {
        errno = EFBIG;
        return -1;
    }
    const uint32_t pcm_bytes = (uint32_t)pcm_bytes64;
    uint8_t header[44] = {0};
    memcpy(header, "RIFF", 4);
    put_u32le(header + 4, 36U + pcm_bytes);
    memcpy(header + 8, "WAVEfmt ", 8);
    put_u32le(header + 16, 16U);
    put_u16le(header + 20, 1U);
    put_u16le(header + 22, 1U);
    put_u32le(header + 24, SAMPLE_RATE);
    put_u32le(header + 28, SAMPLE_RATE * 2U);
    put_u16le(header + 32, 2U);
    put_u16le(header + 34, 16U);
    memcpy(header + 36, "data", 4);
    put_u32le(header + 40, pcm_bytes);

    const int fd = open(temporary_path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {
        return -1;
    }
    int status = write_all(fd, header, sizeof(header));
    if (status == 0) {
        status = write_all(fd, samples, (size_t)pcm_bytes);
    }
    if (close(fd) != 0) {
        status = -1;
    }
    if (status == 0 && rename(temporary_path, final_path) == 0) {
        return 0;
    }
    unlink(temporary_path);
    return -1;
}

static int queued_utterances(const char *spool_dir)
{
    DIR *directory = opendir(spool_dir);
    if (directory == NULL) {
        return -1;
    }
    int count = 0;
    struct dirent *entry;
    while ((entry = readdir(directory)) != NULL) {
        const size_t length = strlen(entry->d_name);
        if (length > 4U && strcmp(entry->d_name + length - 4U, ".wav") == 0) {
            count += 1;
        }
    }
    closedir(directory);
    return count;
}

static void publish_status(const struct options *options, const char *state,
                           double peak_percent, double rms_percent,
                           unsigned long long sequence, unsigned long long dropped)
{
    char temporary_path[1024];
    const int path_size = snprintf(temporary_path, sizeof(temporary_path), "%s.tmp.%ld",
                                   options->status_path, (long)getpid());
    if (path_size < 0 || (size_t)path_size >= sizeof(temporary_path)) {
        return;
    }
    FILE *file = fopen(temporary_path, "wb");
    if (file == NULL) {
        return;
    }
    fprintf(file,
            "{\"state\":\"%s\",\"peak_percent\":%.3f,\"rms_percent\":%.3f,"
            "\"start_threshold_percent\":%.3f,\"stop_threshold_percent\":%.3f,"
            "\"utterances\":%llu,\"dropped\":%llu}\n",
            state, peak_percent, rms_percent, options->start_percent, options->stop_percent,
            sequence, dropped);
    if (fclose(file) == 0) {
        if (rename(temporary_path, options->status_path) != 0) {
            unlink(temporary_path);
        }
    } else {
        unlink(temporary_path);
    }
}

static int parse_positive_int(const char *text, int *output)
{
    char *end = NULL;
    const long value = strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 1 || value > 3600000) {
        return -1;
    }
    *output = (int)value;
    return 0;
}

static int parse_percent(const char *text, double *output)
{
    char *end = NULL;
    const double value = strtod(text, &end);
    if (end == text || *end != '\0' || !isfinite(value) || value <= 0.0 || value > 100.0) {
        return -1;
    }
    *output = value;
    return 0;
}

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s --spool DIR --status FILE --mute-file FILE "
            "[--start-percent N] [--stop-percent N] [--start-ms N] "
            "[--stop-ms N] [--pre-roll-ms N] [--max-seconds N] [--max-queued N]\n",
            program);
}

static int parse_options(int argc, char **argv, struct options *options)
{
    memset(options, 0, sizeof(*options));
    options->start_percent = 2.0;
    options->stop_percent = 2.0;
    options->start_ms = DEFAULT_START_MS;
    options->stop_ms = DEFAULT_STOP_MS;
    options->pre_roll_ms = DEFAULT_PRE_ROLL_MS;
    options->max_seconds = DEFAULT_MAX_SECONDS;
    options->max_queued = 8;

    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--spool") == 0 && index + 1 < argc) {
            options->spool_dir = argv[++index];
        } else if (strcmp(argv[index], "--status") == 0 && index + 1 < argc) {
            options->status_path = argv[++index];
        } else if (strcmp(argv[index], "--mute-file") == 0 && index + 1 < argc) {
            options->mute_path = argv[++index];
        } else if (strcmp(argv[index], "--start-percent") == 0 && index + 1 < argc) {
            if (parse_percent(argv[++index], &options->start_percent) != 0) return -1;
        } else if (strcmp(argv[index], "--stop-percent") == 0 && index + 1 < argc) {
            if (parse_percent(argv[++index], &options->stop_percent) != 0) return -1;
        } else if (strcmp(argv[index], "--start-ms") == 0 && index + 1 < argc) {
            if (parse_positive_int(argv[++index], &options->start_ms) != 0) return -1;
        } else if (strcmp(argv[index], "--stop-ms") == 0 && index + 1 < argc) {
            if (parse_positive_int(argv[++index], &options->stop_ms) != 0) return -1;
        } else if (strcmp(argv[index], "--pre-roll-ms") == 0 && index + 1 < argc) {
            if (parse_positive_int(argv[++index], &options->pre_roll_ms) != 0) return -1;
        } else if (strcmp(argv[index], "--max-seconds") == 0 && index + 1 < argc) {
            if (parse_positive_int(argv[++index], &options->max_seconds) != 0) return -1;
        } else if (strcmp(argv[index], "--max-queued") == 0 && index + 1 < argc) {
            if (parse_positive_int(argv[++index], &options->max_queued) != 0) return -1;
        } else {
            return -1;
        }
    }
    return options->spool_dir != NULL && options->status_path != NULL &&
                   options->mute_path != NULL
               ? 0
               : -1;
}

static void ring_append(int16_t *ring, size_t capacity, size_t *at, size_t *count,
                        const int16_t *samples, size_t sample_count)
{
    for (size_t index = 0; index < sample_count; ++index) {
        ring[*at] = samples[index];
        *at = (*at + 1U) % capacity;
        if (*count < capacity) {
            *count += 1U;
        }
    }
}

static size_t ring_copy(const int16_t *ring, size_t capacity, size_t at, size_t count,
                        int16_t *output)
{
    const size_t begin = count == capacity ? at : 0U;
    for (size_t index = 0; index < count; ++index) {
        output[index] = ring[(begin + index) % capacity];
    }
    return count;
}

int main(int argc, char **argv)
{
    struct options options;
    if (parse_options(argc, argv, &options) != 0) {
        usage(argv[0]);
        return 2;
    }
    if (mkdir(options.spool_dir, 0700) != 0 && errno != EEXIST) {
        perror("mkdir spool");
        return 2;
    }

    const size_t pre_roll_samples = (size_t)options.pre_roll_ms * SAMPLE_RATE / 1000U;
    const size_t maximum_samples = (size_t)options.max_seconds * SAMPLE_RATE;
    int16_t *pre_roll = calloc(pre_roll_samples, sizeof(int16_t));
    int16_t *utterance = malloc(maximum_samples * sizeof(int16_t));
    if (pre_roll == NULL || utterance == NULL) {
        fprintf(stderr, "capture allocation failed\n");
        free(pre_roll);
        free(utterance);
        return 2;
    }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    const int start_frames = (options.start_ms + 9) / 10;
    const int stop_frames = (options.stop_ms + 9) / 10;
    const double start_amplitude = options.start_percent * 32767.0 / 100.0;
    const double stop_amplitude = options.stop_percent * 32767.0 / 100.0;
    size_t pre_roll_at = 0;
    size_t pre_roll_count = 0;
    size_t utterance_count = 0;
    int active_frames = 0;
    int silent_frames = 0;
    int speaking = 0;
    int status_divider = 0;
    unsigned long long sequence = 0;
    unsigned long long dropped = 0;
    int16_t frame[FRAME_SAMPLES];
    size_t frame_count = 0;

    publish_status(&options, "listening", 0.0, 0.0, sequence, dropped);
    while (!stop_requested) {
        const ssize_t bytes = read(STDIN_FILENO, frame + frame_count,
                                   (FRAME_SAMPLES - frame_count) * sizeof(int16_t));
        if (bytes == 0) {
            break;
        }
        if (bytes < 0) {
            if (errno == EINTR) continue;
            perror("capture read");
            break;
        }
        if ((bytes & 1) != 0) {
            fprintf(stderr, "capture received an incomplete PCM16 sample\n");
            break;
        }
        frame_count += (size_t)bytes / sizeof(int16_t);
        if (frame_count < FRAME_SAMPLES) {
            continue;
        }

        int peak = 0;
        double squares = 0.0;
        for (size_t index = 0; index < FRAME_SAMPLES; ++index) {
            const int value = frame[index] == INT16_MIN ? 32768 : abs(frame[index]);
            if (value > peak) peak = value;
            squares += (double)frame[index] * frame[index];
        }
        const double peak_percent = (double)peak * 100.0 / 32767.0;
        const double rms_percent = sqrt(squares / FRAME_SAMPLES) * 100.0 / 32767.0;
        const int muted = access(options.mute_path, F_OK) == 0;

        if (muted) {
            speaking = 0;
            active_frames = 0;
            silent_frames = 0;
            utterance_count = 0;
            pre_roll_at = 0;
            pre_roll_count = 0;
        } else if (!speaking) {
            ring_append(pre_roll, pre_roll_samples, &pre_roll_at, &pre_roll_count, frame,
                        FRAME_SAMPLES);
            const double rms_amplitude = sqrt(squares / FRAME_SAMPLES);
            active_frames = rms_amplitude >= start_amplitude ? active_frames + 1 : 0;
            if (active_frames >= start_frames) {
                utterance_count = ring_copy(pre_roll, pre_roll_samples, pre_roll_at,
                                            pre_roll_count, utterance);
                speaking = 1;
                silent_frames = 0;
            }
        } else {
            size_t available = maximum_samples - utterance_count;
            size_t append = FRAME_SAMPLES < available ? FRAME_SAMPLES : available;
            memcpy(utterance + utterance_count, frame, append * sizeof(int16_t));
            utterance_count += append;
            const double rms_amplitude = sqrt(squares / FRAME_SAMPLES);
            silent_frames = rms_amplitude < stop_amplitude ? silent_frames + 1 : 0;
            if (silent_frames >= stop_frames || utterance_count == maximum_samples) {
                sequence += 1U;
                const int queued = queued_utterances(options.spool_dir);
                if (queued < 0 || queued >= options.max_queued) {
                    dropped += 1U;
                } else if (write_wav(&options, utterance, utterance_count, sequence) != 0) {
                    perror("write utterance");
                    dropped += 1U;
                }
                speaking = 0;
                active_frames = 0;
                silent_frames = 0;
                utterance_count = 0;
                pre_roll_at = 0;
                pre_roll_count = 0;
            }
        }

        status_divider += 1;
        if (status_divider >= 10) {
            publish_status(&options, muted ? "playback-muted" : speaking ? "speech" : "listening",
                           peak_percent, rms_percent, sequence, dropped);
            status_divider = 0;
        }
        frame_count = 0;
    }

    publish_status(&options, "stopped", 0.0, 0.0, sequence, dropped);
    free(pre_roll);
    free(utterance);
    return stop_requested ? 0 : 1;
}
