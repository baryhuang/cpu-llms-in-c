#define _GNU_SOURCE

#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/sysinfo.h>
#include <time.h>
#include <unistd.h>

#if defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#define DEFAULT_MIB 512U
#define DEFAULT_SECONDS 2.0
#define MAX_BENCH_THREADS 4
#define CACHE_LINE_BYTES 64U

static volatile uint64_t global_sink;

struct worker_args {
    const uint64_t *data;
    size_t words;
    double duration_seconds;
    int cpu_index;
    pthread_barrier_t *barrier;
    uint64_t bytes_read;
    uint64_t checksum;
    double elapsed_seconds;
};

static double monotonic_seconds(void)
{
    struct timespec timestamp;

    if (clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp) != 0) {
        perror("clock_gettime");
        exit(2);
    }

    return (double)timestamp.tv_sec + (double)timestamp.tv_nsec * 1.0e-9;
}

static void pin_current_thread(int cpu_index)
{
    cpu_set_t set;

    CPU_ZERO(&set);
    CPU_SET(cpu_index, &set);
    if (pthread_setaffinity_np(pthread_self(), sizeof(set), &set) != 0) {
        fprintf(stderr, "warning: failed to pin worker to CPU %d\n", cpu_index);
    }
}

static void *read_worker(void *opaque)
{
    struct worker_args *args = opaque;
    const uint64_t *data = args->data;
    const size_t words = args->words;
    uint64_t sum0 = 0;
    uint64_t sum1 = 0;
    uint64_t sum2 = 0;
    uint64_t sum3 = 0;
    uint64_t bytes_read = 0;
    double start;
    double now;

    pin_current_thread(args->cpu_index);
    pthread_barrier_wait(args->barrier);
    start = monotonic_seconds();
    now = start;

    do {
        size_t index = 0;

        for (; index + 16 <= words; index += 16) {
            sum0 += data[index + 0] + data[index + 4] + data[index + 8] + data[index + 12];
            sum1 += data[index + 1] + data[index + 5] + data[index + 9] + data[index + 13];
            sum2 += data[index + 2] + data[index + 6] + data[index + 10] + data[index + 14];
            sum3 += data[index + 3] + data[index + 7] + data[index + 11] + data[index + 15];
        }
        for (; index < words; ++index) {
            sum0 += data[index];
        }

        bytes_read += (uint64_t)words * sizeof(*data);
        now = monotonic_seconds();
    } while (now - start < args->duration_seconds);

    args->bytes_read = bytes_read;
    args->checksum = sum0 ^ sum1 ^ sum2 ^ sum3;
    args->elapsed_seconds = now - start;
    return NULL;
}

static int run_read_benchmark(const uint64_t *buffer, size_t bytes, int threads,
                              double duration_seconds)
{
    pthread_t *thread_ids = NULL;
    struct worker_args *args = NULL;
    pthread_barrier_t barrier;
    const size_t total_words = bytes / sizeof(*buffer);
    const size_t words_per_thread = total_words / (size_t)threads;
    uint64_t total_bytes = 0;
    uint64_t checksum = 0;
    double longest_elapsed = 0.0;
    int result = 1;

    thread_ids = calloc((size_t)threads, sizeof(*thread_ids));
    args = calloc((size_t)threads, sizeof(*args));
    if (thread_ids == NULL || args == NULL) {
        fprintf(stderr, "error: worker allocation failed\n");
        goto cleanup;
    }

    if (pthread_barrier_init(&barrier, NULL, (unsigned)threads + 1U) != 0) {
        fprintf(stderr, "error: barrier initialization failed\n");
        goto cleanup;
    }

    for (int index = 0; index < threads; ++index) {
        args[index].data = buffer + (size_t)index * words_per_thread;
        args[index].words = index == threads - 1
                                ? total_words - (size_t)index * words_per_thread
                                : words_per_thread;
        args[index].duration_seconds = duration_seconds;
        args[index].cpu_index = index;
        args[index].barrier = &barrier;

        if (pthread_create(&thread_ids[index], NULL, read_worker, &args[index]) != 0) {
            fprintf(stderr, "error: failed to create worker %d\n", index);
            exit(2);
        }
    }

    pthread_barrier_wait(&barrier);

    for (int index = 0; index < threads; ++index) {
        pthread_join(thread_ids[index], NULL);
    }

    for (int index = 0; index < threads; ++index) {
        total_bytes += args[index].bytes_read;
        checksum ^= args[index].checksum;
        if (args[index].elapsed_seconds > longest_elapsed) {
            longest_elapsed = args[index].elapsed_seconds;
        }
    }

    global_sink ^= checksum;
    printf("read_sum threads=%d bytes=%" PRIu64 " seconds=%.6f gib_per_second=%.3f checksum=%" PRIu64 "\n",
           threads, total_bytes, longest_elapsed,
           (double)total_bytes / longest_elapsed / 1073741824.0, checksum);
    pthread_barrier_destroy(&barrier);
    result = 0;

cleanup:
    free(args);
    free(thread_ids);
    return result;
}

static unsigned long long parse_mib(const char *text)
{
    char *end = NULL;
    unsigned long long value;

    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || value < 16 || value > 2048) {
        fprintf(stderr, "error: --mib must be an integer from 16 through 2048\n");
        exit(2);
    }
    return value;
}

static double parse_seconds(const char *text)
{
    char *end = NULL;
    double value;

    errno = 0;
    value = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || value < 0.1 || value > 60.0) {
        fprintf(stderr, "error: --seconds must be from 0.1 through 60.0\n");
        exit(2);
    }
    return value;
}

static void print_usage(const char *program)
{
    fprintf(stderr, "usage: %s [--mib N] [--seconds N]\n", program);
}

static void print_isa(void)
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_cpu_init();
    printf("isa arch=x86 sse4_2=%d avx=%d avx2=%d fma=%d\n",
           __builtin_cpu_supports("sse4.2") != 0,
           __builtin_cpu_supports("avx") != 0,
           __builtin_cpu_supports("avx2") != 0,
           __builtin_cpu_supports("fma") != 0);
#elif defined(__aarch64__)
    const unsigned long hwcap = getauxval(AT_HWCAP);
    int asimd = 0;
    int asimddp = 0;
    int sve = 0;
#ifdef HWCAP_ASIMD
    asimd = (hwcap & HWCAP_ASIMD) != 0;
#endif
#ifdef HWCAP_ASIMDDP
    asimddp = (hwcap & HWCAP_ASIMDDP) != 0;
#endif
#ifdef HWCAP_SVE
    sve = (hwcap & HWCAP_SVE) != 0;
#endif
    printf("isa arch=aarch64 asimd=%d asimddp=%d sve=%d\n", asimd, asimddp, sve);
#else
    printf("isa arch=unknown\n");
#endif
}

int main(int argc, char **argv)
{
    unsigned long long requested_mib = DEFAULT_MIB;
    double duration_seconds = DEFAULT_SECONDS;
    struct sysinfo system_info;
    long logical_cpus;
    long page_size;
    size_t bytes;
    uint64_t *buffer = NULL;
    int max_threads;

    for (int index = 1; index < argc; ++index) {
        if (strcmp(argv[index], "--mib") == 0 && index + 1 < argc) {
            requested_mib = parse_mib(argv[++index]);
        } else if (strcmp(argv[index], "--seconds") == 0 && index + 1 < argc) {
            duration_seconds = parse_seconds(argv[++index]);
        } else {
            print_usage(argv[0]);
            return 2;
        }
    }

    logical_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    page_size = sysconf(_SC_PAGESIZE);
    if (logical_cpus < 1 || page_size < 1 || sysinfo(&system_info) != 0) {
        perror("system query");
        return 2;
    }

    if (requested_mib > SIZE_MAX / (1024U * 1024U)) {
        fprintf(stderr, "error: requested buffer is too large\n");
        return 2;
    }
    bytes = (size_t)requested_mib * 1024U * 1024U;

    if (posix_memalign((void **)&buffer,
                       (size_t)page_size > CACHE_LINE_BYTES ? (size_t)page_size : CACHE_LINE_BYTES,
                       bytes) != 0) {
        fprintf(stderr, "error: failed to allocate %zu bytes\n", bytes);
        return 2;
    }

    for (size_t index = 0; index < bytes / sizeof(*buffer); ++index) {
        buffer[index] = (uint64_t)index * UINT64_C(0x9e3779b97f4a7c15) + UINT64_C(0xd1b54a32d192ed03);
    }
    (void)madvise(buffer, bytes, MADV_HUGEPAGE);

    printf("target_probe_version=1\n");
    printf("logical_cpus=%ld\n", logical_cpus);
    printf("page_size_bytes=%ld\n", page_size);
    printf("physical_memory_bytes=%" PRIu64 "\n",
           (uint64_t)system_info.totalram * (uint64_t)system_info.mem_unit);
    printf("buffer_bytes=%zu\n", bytes);
    printf("duration_seconds=%.3f\n", duration_seconds);
    print_isa();

    max_threads = logical_cpus < MAX_BENCH_THREADS ? (int)logical_cpus : MAX_BENCH_THREADS;
    if (run_read_benchmark(buffer, bytes, 1, duration_seconds) != 0) {
        free(buffer);
        return 2;
    }
    if (max_threads > 1 && run_read_benchmark(buffer, bytes, max_threads, duration_seconds) != 0) {
        free(buffer);
        return 2;
    }

    printf("sink=%" PRIu64 "\n", global_sink);
    free(buffer);
    return 0;
}
