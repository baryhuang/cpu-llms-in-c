#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "float16_compat.h"
#include "rknn_api.h"
#include "rknn_matmul_api.h"
#include "static_npu_scheduler.h"
#include "w4a16_model.h"

namespace llmc {

enum class W4RunMode {
  kBaseline,
  kLlmc,
};

const char *run_mode_name(W4RunMode mode);
W4RunMode parse_run_mode(const char *text);
rknn_core_mask parse_core_mask(const char *text);
int configured_core_count(const char *text);
std::vector<rknn_core_mask> parallel_npu_core_masks();

struct W4MatmulMetrics {
  uint64_t runs = 0;
  uint64_t npu_jobs = 0;
  uint64_t weight_dequantizations = 0;
  uint64_t weight_uploads = 0;
  double weight_dequant_ms = 0.0;
  double weight_upload_ms = 0.0;
  double input_copy_ms = 0.0;
  double output_copy_ms = 0.0;
  double npu_run_ms = 0.0;
};

struct F16MatmulMetrics {
  uint64_t runs = 0;
  uint64_t npu_jobs = 0;
  double input_copy_ms = 0.0;
  double npu_run_ms = 0.0;
  double output_copy_ms = 0.0;
};

// LLMC's primary mixed-precision linear primitive. The C++ runtime owns the
// packed INT4 [N, K] weight, expands it to FP16 B=[K, N] on the CPU, and asks
// RKNPU2 to execute only FP16 x FP16 -> FP16. No fused Rockchip W4A16 dtype is
// requested. Baseline expands and uploads on every invocation; LLMC does it
// once and keeps the FP16 NPU weight buffer resident.
class W4Linear {
public:
  W4Linear(const W4Tensor &weight, int rows, W4RunMode mode,
           rknn_core_mask core_mask);
  W4Linear(const W4Tensor &weight, int rows, W4RunMode mode,
           std::vector<rknn_core_mask> core_masks);
  W4Linear(const W4Linear &) = delete;
  W4Linear &operator=(const W4Linear &) = delete;
  ~W4Linear();

  int rows() const { return rows_; }
  int input_columns() const { return input_columns_; }
  int output_columns() const { return output_columns_; }
  size_t shard_count() const { return shard_count_; }

  // Convenience interface for CPU stages. The copies are reported separately
  // so they cannot be hidden inside the NPU timing.
  void run(const llmc_float16 *input, llmc_float16 *output);

  const W4MatmulMetrics &metrics() const { return metrics_; }

private:
  struct Shard;
  struct ShardTask;
  static void create_shard_task(void *argument) noexcept;
  static void destroy_shard_task(void *argument) noexcept;
  static void dequantize_shard_task(void *argument) noexcept;
  static void bind_resident_weight_task(void *argument) noexcept;
  static void run_shard_task(void *argument) noexcept;
  void prepare_weights();
  void destroy_shards() noexcept;
  void check(int rc, const char *operation) const;

  const W4Tensor &weight_;
  int rows_ = 0;
  int input_columns_ = 0;
  int output_columns_ = 0;
  W4RunMode mode_ = W4RunMode::kBaseline;
  StaticNpuScheduler &scheduler_;
  std::array<std::unique_ptr<Shard>, StaticNpuScheduler::kWorkerCount> shards_;
  size_t shard_count_ = 0;
  W4MatmulMetrics metrics_{};
};

// Ahead-of-time linear plan used by the full Whisper runtime. A small number
// of shape plans own A/C workspaces, while every model tensor owns only its
// resident B shards. All weights are expanded during model initialization;
// inference performs no dequantization, context creation or DMA allocation.
class AotW4LinearPlan {
public:
  class Weight {
  public:
    Weight(const Weight &) = delete;
    Weight &operator=(const Weight &) = delete;
    ~Weight();

    uint64_t bytes() const { return bytes_; }

  private:
    friend class AotW4LinearPlan;
    explicit Weight(AotW4LinearPlan *owner) : owner_(owner) {}

    AotW4LinearPlan *owner_ = nullptr;
    std::array<rknn_tensor_mem *, StaticNpuScheduler::kWorkerCount> buffers_{};
    uint64_t bytes_ = 0;
  };

  AotW4LinearPlan(int rows, int input_columns, int output_columns,
                  std::vector<rknn_core_mask> core_masks);
  AotW4LinearPlan(const AotW4LinearPlan &) = delete;
  AotW4LinearPlan &operator=(const AotW4LinearPlan &) = delete;
  ~AotW4LinearPlan();

  std::unique_ptr<Weight> prepare(const W4Tensor &weight);
  void run(const Weight &weight, const llmc_float16 *input,
           llmc_float16 *output);

  int rows() const { return rows_; }
  int input_columns() const { return input_columns_; }
  int output_columns() const { return output_columns_; }
  size_t shard_count() const { return shard_count_; }
  const W4MatmulMetrics &metrics() const { return metrics_; }

private:
  struct Shard;
  struct ShardTask;
  static void create_shard_task(void *argument) noexcept;
  static void destroy_shard_task(void *argument) noexcept;
  static void prepare_weight_task(void *argument) noexcept;
  static void destroy_weight_task(void *argument) noexcept;
  static void run_shard_task(void *argument) noexcept;
  void destroy_shards() noexcept;
  void destroy_weight(Weight &weight) noexcept;
  void check(int result, const char *operation) const;

  int rows_ = 0;
  int input_columns_ = 0;
  int output_columns_ = 0;
  StaticNpuScheduler &scheduler_;
  std::array<std::unique_ptr<Shard>, StaticNpuScheduler::kWorkerCount> shards_;
  size_t shard_count_ = 0;
  W4MatmulMetrics metrics_{};
};

// Persistent FP16 MatMul used for QK^T and softmax(QK^T)V. Its dimensions are
// fixed, while both inputs may change on every invocation.
class F16Matmul {
public:
  F16Matmul(int rows, int inner, int columns, rknn_core_mask core_mask);
  F16Matmul(int rows, int inner, int columns,
            std::vector<rknn_core_mask> core_masks);
  F16Matmul(const F16Matmul &) = delete;
  F16Matmul &operator=(const F16Matmul &) = delete;
  ~F16Matmul();

  void run(const llmc_float16 *a, const llmc_float16 *b, llmc_float16 *c);
  size_t shard_count() const { return shard_count_; }
  const F16MatmulMetrics &metrics() const { return metrics_; }

private:
  struct Shard;
  struct ShardTask;
  static void create_shard_task(void *argument) noexcept;
  static void destroy_shard_task(void *argument) noexcept;
  static void run_shard_task(void *argument) noexcept;
  void destroy_shards() noexcept;
  void check(int rc, const char *operation) const;

  int rows_ = 0;
  int inner_ = 0;
  int columns_ = 0;
  StaticNpuScheduler &scheduler_;
  std::array<std::unique_ptr<Shard>, StaticNpuScheduler::kWorkerCount> shards_;
  size_t shard_count_ = 0;
  F16MatmulMetrics metrics_{};
};

// Full-N FP16 MatMul contexts for attention-head scheduling. Context i is
// created, run and destroyed only by persistent worker i. The caller invokes
// run_on_worker from an existing scheduler task, allowing each worker to run a
// static sequence of complete heads without nested dispatch or N-sharding.
class WorkerF16Matmul {
public:
  WorkerF16Matmul(int rows, int inner, int columns,
                  std::vector<rknn_core_mask> core_masks);
  WorkerF16Matmul(const WorkerF16Matmul &) = delete;
  WorkerF16Matmul &operator=(const WorkerF16Matmul &) = delete;
  ~WorkerF16Matmul();

  int run_on_worker(size_t worker_index, const llmc_float16 *a,
                    const llmc_float16 *b, llmc_float16 *c) noexcept;

  int rows() const { return rows_; }
  int inner() const { return inner_; }
  int columns() const { return columns_; }

private:
  struct Core;
  struct CoreTask;
  static void create_core_task(void *argument) noexcept;
  static void destroy_core_task(void *argument) noexcept;
  void destroy_cores() noexcept;

  int rows_ = 0;
  int inner_ = 0;
  int columns_ = 0;
  StaticNpuScheduler &scheduler_;
  std::array<std::unique_ptr<Core>, StaticNpuScheduler::kWorkerCount> cores_;
};

} // namespace llmc
