#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "minimindo_parallel.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(__linux__)
#include <linux/futex.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

enum { MAX_THREADS = 4, BACKGROUND_WORKERS = 3 };

typedef struct {
    pthread_t thread;
    atomic_uint active;
    atomic_uint stop;
    atomic_uint epoch;
    atomic_uint complete;
    minimindo_parallel_task task;
    void *context;
    size_t begin;
    size_t end;
    unsigned cpu;
} parallel_worker;

static parallel_worker workers[BACKGROUND_WORKERS];
static pthread_once_t pool_once = PTHREAD_ONCE_INIT;
static atomic_int pool_ready;
static _Thread_local unsigned caller_threads = 1U;
static _Thread_local unsigned session_threads = 0U;

static inline void spin_pause(void)
{
#if defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#elif defined(__x86_64__)
    __asm__ volatile("pause" ::: "memory");
#else
    atomic_signal_fence(memory_order_seq_cst);
#endif
}

static void mailbox_wait(atomic_uint *active)
{
#if defined(__linux__)
    _Static_assert(sizeof(atomic_uint) == sizeof(uint32_t),
                   "Linux futex word must be 32 bits");
    (void)syscall(SYS_futex, (uint32_t *)(void *)active,
                  FUTEX_WAIT_PRIVATE, 0U, NULL, NULL, 0U);
#else
    /* Non-Linux is a development target.  Production A113X uses futex_wait,
     * while this fallback keeps the worker persistent without a mutex. */
    (void)active;
    spin_pause();
#endif
}

static void mailbox_wake(atomic_uint *active)
{
#if defined(__linux__)
    (void)syscall(SYS_futex, (uint32_t *)(void *)active,
                  FUTEX_WAKE_PRIVATE, 1U, NULL, NULL, 0U);
#else
    (void)active;
#endif
}

int minimindo_parallel_pin_current(unsigned cpu)
{
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
    (void)cpu;
    return 0;
#endif
}

static void *worker_main(void *opaque)
{
    parallel_worker *worker = opaque;
    (void)minimindo_parallel_pin_current(worker->cpu);
    unsigned observed = 0U;
    for (;;) {
        while (atomic_load_explicit(&worker->active, memory_order_acquire) == 0U &&
               atomic_load_explicit(&worker->stop, memory_order_relaxed) == 0U)
            mailbox_wait(&worker->active);
        if (atomic_load_explicit(&worker->stop, memory_order_relaxed) != 0U)
            break;
        while (atomic_load_explicit(&worker->active, memory_order_acquire) != 0U) {
            const unsigned epoch =
                atomic_load_explicit(&worker->epoch, memory_order_acquire);
            if (epoch == observed) {
                spin_pause();
                continue;
            }
            observed = epoch;
            minimindo_parallel_task task = worker->task;
            task(worker->context, worker->begin, worker->end);
            atomic_store_explicit(&worker->complete, epoch,
                                  memory_order_release);
        }
    }
    return NULL;
}

static void pool_shutdown(void)
{
    if (atomic_load_explicit(&pool_ready, memory_order_acquire) == 0) return;
    for (unsigned index = 0; index < BACKGROUND_WORKERS; ++index) {
        atomic_store_explicit(&workers[index].stop, 1U, memory_order_relaxed);
        atomic_store_explicit(&workers[index].active, 1U,
                              memory_order_release);
        mailbox_wake(&workers[index].active);
    }
    for (unsigned index = 0; index < BACKGROUND_WORKERS; ++index)
        pthread_join(workers[index].thread, NULL);
}

static void pool_initialize(void)
{
    unsigned initialized = 0U;
    for (unsigned index = 0; index < BACKGROUND_WORKERS; ++index) {
        parallel_worker *worker = &workers[index];
        worker->cpu = index + 1U;
        if (pthread_create(&worker->thread, NULL, worker_main, worker) != 0)
            break;
        ++initialized;
    }
    if (initialized != BACKGROUND_WORKERS) {
        for (unsigned index = 0; index < initialized; ++index) {
            atomic_store_explicit(&workers[index].stop, 1U,
                                  memory_order_relaxed);
            atomic_store_explicit(&workers[index].active, 1U,
                                  memory_order_release);
            mailbox_wake(&workers[index].active);
            pthread_join(workers[index].thread, NULL);
        }
        return;
    }
    atomic_store_explicit(&pool_ready, 1, memory_order_release);
    atexit(pool_shutdown);
}

static int ensure_pool(void)
{
    pthread_once(&pool_once, pool_initialize);
    return atomic_load_explicit(&pool_ready, memory_order_acquire) != 0;
}

void minimindo_parallel_set_threads(unsigned threads)
{
    if (threads == 0U) threads = 1U;
    if (threads > MAX_THREADS) threads = MAX_THREADS;
    caller_threads = threads;
}

unsigned minimindo_parallel_threads(void)
{
    return caller_threads;
}

int minimindo_parallel_session_begin(unsigned threads)
{
    minimindo_parallel_set_threads(threads);
    session_threads = 1U;
    if (caller_threads == 1U) return 0;
    if (!ensure_pool()) {
        caller_threads = 1U;
        return -1;
    }
    /* Speech assigns the pool to exactly one dispatcher: main during
     * Thinker/Talker, then Mimi after the producer handoff.  There is no
     * multi-producer lock because the architecture never creates one. */
    session_threads = caller_threads;
    for (unsigned lane = 1U; lane < caller_threads; ++lane) {
        parallel_worker *worker = &workers[lane - 1U];
        atomic_store_explicit(&worker->active, 1U, memory_order_release);
        mailbox_wake(&worker->active);
    }
    return 0;
}

void minimindo_parallel_session_end(void)
{
    for (unsigned lane = 1U; lane < session_threads; ++lane)
        atomic_store_explicit(&workers[lane - 1U].active, 0U,
                              memory_order_release);
    session_threads = 0U;
    caller_threads = 1U;
}

void minimindo_parallel_for(size_t count, minimindo_parallel_task task,
                            void *context)
{
    unsigned threads = caller_threads;
    if (count == 0U) return;
    if (threads > count) threads = (unsigned)count;
    if (threads <= 1U || !ensure_pool()) {
        task(context, 0U, count);
        return;
    }
    unsigned epochs[BACKGROUND_WORKERS] = {0};
    for (unsigned lane = 1U; lane < threads; ++lane) {
        parallel_worker *worker = &workers[lane - 1U];
        worker->task = task;
        worker->context = context;
        worker->begin = count * lane / threads;
        worker->end = count * (lane + 1U) / threads;
        epochs[lane - 1U] =
            atomic_fetch_add_explicit(&worker->epoch, 1U,
                                      memory_order_release) + 1U;
    }
    task(context, 0U, count / threads);
    for (unsigned lane = 1U; lane < threads; ++lane) {
        parallel_worker *worker = &workers[lane - 1U];
        while (atomic_load_explicit(&worker->complete,
                                    memory_order_acquire) !=
               epochs[lane - 1U])
            spin_pause();
    }
}
