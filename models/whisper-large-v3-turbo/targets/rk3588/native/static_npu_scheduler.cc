#include "static_npu_scheduler.h"

#if defined(__linux__)
#include <pthread.h>
#endif

namespace llmc {

StaticNpuScheduler::StaticNpuScheduler() {
  for (size_t index = 0; index < kWorkerCount; ++index) {
    workers_[index] = std::thread([this, index] { worker_loop(index); });
  }
}

StaticNpuScheduler::~StaticNpuScheduler() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
    ++generation_;
  }
  start_condition_.notify_all();
  for (auto &worker : workers_) {
    if (worker.joinable())
      worker.join();
  }
}

void StaticNpuScheduler::dispatch(
    const std::array<Task, kWorkerCount> &tasks) noexcept {
  std::unique_lock<std::mutex> lock(mutex_);
  tasks_ = tasks;
  completed_ = 0;
  ++generation_;
  ++dispatches_;
  start_condition_.notify_all();
  done_condition_.wait(lock,
                       [this] { return completed_ == kWorkerCount; });
}

void StaticNpuScheduler::worker_loop(size_t worker_index) noexcept {
#if defined(__linux__)
  char name[16]{};
  name[0] = 'l';
  name[1] = 'l';
  name[2] = 'm';
  name[3] = 'c';
  name[4] = '-';
  name[5] = 'n';
  name[6] = 'p';
  name[7] = 'u';
  name[8] = static_cast<char>('0' + worker_index);
  pthread_setname_np(pthread_self(), name);
#endif
  uint64_t observed_generation = 0;
  for (;;) {
    Task task;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      start_condition_.wait(lock, [this, observed_generation] {
        return stopping_ || generation_ != observed_generation;
      });
      if (stopping_)
        return;
      observed_generation = generation_;
      task = tasks_[worker_index];
    }
    if (task.function != nullptr)
      task.function(task.argument);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ++completed_;
      if (completed_ == kWorkerCount)
        done_condition_.notify_one();
    }
  }
}

StaticNpuScheduler &static_npu_scheduler() {
  static StaticNpuScheduler scheduler;
  return scheduler;
}

} // namespace llmc
