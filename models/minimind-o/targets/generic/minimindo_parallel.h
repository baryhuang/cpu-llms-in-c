#ifndef LLM_IN_C_MINIMINDO_PARALLEL_H
#define LLM_IN_C_MINIMINDO_PARALLEL_H

#include <stddef.h>

typedef void (*minimindo_parallel_task)(void *context, size_t begin,
                                        size_t end);

/*
 * A process-wide, persistent three-worker pool. The calling thread is lane 0;
 * workers 1..3 are pinned to CPUs 1..3 on Linux. A session wakes its workers
 * once, and they spin on acquire/release epochs until session_end. Every
 * worker mailbox is SPSC; the speech stage handoff guarantees there is only
 * one dispatcher. This avoids an OpenMP team entry, futex sleep and scheduler
 * wakeup for every matrix.
 */
int minimindo_parallel_session_begin(unsigned threads);
void minimindo_parallel_session_end(void);
void minimindo_parallel_set_threads(unsigned threads);
unsigned minimindo_parallel_threads(void);
void minimindo_parallel_for(size_t count, minimindo_parallel_task task,
                            void *context);

/* Pin the calling thread for explicit A113X stage ownership. */
int minimindo_parallel_pin_current(unsigned cpu);

#endif
