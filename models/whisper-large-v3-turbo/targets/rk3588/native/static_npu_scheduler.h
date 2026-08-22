#pragma once

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>

namespace llmc {

// One long-lived host worker owns each RK3588 NPU core. Dispatch uses fixed
// task slots: no threads, queues, promises or task containers are allocated in
// the inference hot path.
class StaticNpuScheduler {
public:
  static constexpr size_t kWorkerCount = 3;

  struct Task {
    void (*function)(void *) noexcept = nullptr;
    void *argument = nullptr;
  };

  StaticNpuScheduler();
  StaticNpuScheduler(const StaticNpuScheduler &) = delete;
  StaticNpuScheduler &operator=(const StaticNpuScheduler &) = delete;
  ~StaticNpuScheduler();

  // The caller must not invoke dispatch concurrently or from a worker task.
  // Null task slots are valid and complete without calling a function.
  void dispatch(const std::array<Task, kWorkerCount> &tasks) noexcept;
  uint64_t dispatches() const { return dispatches_; }

private:
  void worker_loop(size_t worker_index) noexcept;

  std::array<std::thread, kWorkerCount> workers_;
  std::array<Task, kWorkerCount> tasks_{};
  std::mutex mutex_;
  std::condition_variable start_condition_;
  std::condition_variable done_condition_;
  uint64_t generation_ = 0;
  uint64_t dispatches_ = 0;
  size_t completed_ = 0;
  bool stopping_ = false;
};

StaticNpuScheduler &static_npu_scheduler();

} // namespace llmc
