#include "w4a16_matmul.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace llmc {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

struct ColumnShard {
  int offset = 0;
  int columns = 0;
};

std::vector<ColumnShard> split_columns(int columns, size_t shard_count,
                                       int alignment) {
  const int kAlignment = alignment;
  if (columns <= 0 || columns % kAlignment != 0 || shard_count == 0 ||
      static_cast<size_t>(columns / kAlignment) < shard_count) {
    throw std::runtime_error("cannot split FP16 MatMul output columns");
  }
  const int units = columns / kAlignment;
  const int base = units / static_cast<int>(shard_count);
  const int remainder = units % static_cast<int>(shard_count);
  std::vector<ColumnShard> result;
  result.reserve(shard_count);
  int offset = 0;
  for (size_t index = 0; index < shard_count; ++index) {
    const int shard_units =
        base + (static_cast<int>(index) < remainder ? 1 : 0);
    const int shard_columns = shard_units * kAlignment;
    result.push_back({offset, shard_columns});
    offset += shard_columns;
  }
  if (offset != columns)
    throw std::runtime_error("internal FP16 column-shard error");
  return result;
}

} // namespace

const char *run_mode_name(W4RunMode mode) {
  return mode == W4RunMode::kLlmc ? "llmc" : "baseline";
}

W4RunMode parse_run_mode(const char *text) {
  if (std::strcmp(text, "baseline") == 0)
    return W4RunMode::kBaseline;
  if (std::strcmp(text, "llmc") == 0)
    return W4RunMode::kLlmc;
  throw std::runtime_error(std::string("invalid W4A16 mode: ") + text);
}

rknn_core_mask parse_core_mask(const char *text) {
  if (std::strcmp(text, "auto") == 0)
    return RKNN_NPU_CORE_AUTO;
  if (std::strcmp(text, "0") == 0)
    return RKNN_NPU_CORE_0;
  if (std::strcmp(text, "1") == 0)
    return RKNN_NPU_CORE_1;
  if (std::strcmp(text, "2") == 0)
    return RKNN_NPU_CORE_2;
  if (std::strcmp(text, "0,1") == 0)
    return RKNN_NPU_CORE_0_1;
  if (std::strcmp(text, "0,1,2") == 0)
    return RKNN_NPU_CORE_0_1_2;
  if (std::strcmp(text, "all") == 0)
    return RKNN_NPU_CORE_ALL;
  throw std::runtime_error(std::string("invalid core mask: ") + text);
}

int configured_core_count(const char *text) {
  if (std::strcmp(text, "0") == 0 || std::strcmp(text, "1") == 0 ||
      std::strcmp(text, "2") == 0) {
    return 1;
  }
  if (std::strcmp(text, "0,1") == 0)
    return 2;
  if (std::strcmp(text, "0,1,2") == 0 || std::strcmp(text, "all") == 0) {
    return 3;
  }
  return 0;
}

std::vector<rknn_core_mask> parallel_npu_core_masks() {
  return {RKNN_NPU_CORE_0, RKNN_NPU_CORE_1, RKNN_NPU_CORE_2};
}

struct W4Linear::Shard {
  int column_offset = 0;
  int columns = 0;
  rknn_core_mask core_mask = RKNN_NPU_CORE_AUTO;
  rknn_matmul_ctx context = 0;
  rknn_matmul_info info{};
  rknn_matmul_io_attr io{};
  rknn_tensor_mem *a = nullptr;
  rknn_tensor_mem *b = nullptr;
  rknn_tensor_mem *c = nullptr;
};

struct W4Linear::ShardTask {
  W4Linear *owner = nullptr;
  Shard *shard = nullptr;
  int result = RKNN_SUCC;
  const char *operation = nullptr;
  double elapsed = 0.0;
  std::exception_ptr error;
};

void W4Linear::create_shard_task(void *argument) noexcept {
  auto &task = *static_cast<ShardTask *>(argument);
  Shard &shard = *task.shard;
  task.operation = "rknn_matmul_create";
  task.result = rknn_matmul_create(&shard.context, &shard.info, &shard.io);
  if (task.result != RKNN_SUCC)
    return;
  if (shard.core_mask != RKNN_NPU_CORE_AUTO) {
    task.operation = "rknn_matmul_set_core_mask";
    task.result = rknn_matmul_set_core_mask(shard.context, shard.core_mask);
    if (task.result != RKNN_SUCC)
      return;
  }
  shard.a = rknn_create_mem(shard.context, shard.io.A.size);
  shard.b = rknn_create_mem(shard.context, shard.io.B.size);
  shard.c = rknn_create_mem(shard.context, shard.io.C.size);
  if (shard.a == nullptr || shard.b == nullptr || shard.c == nullptr) {
    task.operation = "rknn_create_mem(W4 shard)";
    task.result = -1;
    return;
  }
  const uint64_t expected_a = static_cast<uint64_t>(task.owner->rows_) *
                              task.owner->input_columns_ *
                              sizeof(llmc_float16);
  const uint64_t expected_b =
      static_cast<uint64_t>(task.owner->input_columns_) * shard.columns *
      sizeof(llmc_float16);
  const uint64_t expected_c = static_cast<uint64_t>(task.owner->rows_) *
                              shard.columns * sizeof(llmc_float16);
  if (shard.io.A.size < expected_a || shard.io.B.size < expected_b ||
      shard.io.C.size < expected_c) {
    task.operation = "RKNPU2 MatMul buffer size validation";
    task.result = -1;
    return;
  }
  task.operation = "rknn_matmul_set_io_mem(C shard)";
  task.result =
      rknn_matmul_set_io_mem(shard.context, shard.c, &shard.io.C);
}

void W4Linear::destroy_shard_task(void *argument) noexcept {
  auto &task = *static_cast<ShardTask *>(argument);
  Shard &shard = *task.shard;
  if (shard.a != nullptr) {
    rknn_destroy_mem(shard.context, shard.a);
    shard.a = nullptr;
  }
  if (shard.b != nullptr) {
    rknn_destroy_mem(shard.context, shard.b);
    shard.b = nullptr;
  }
  if (shard.c != nullptr) {
    rknn_destroy_mem(shard.context, shard.c);
    shard.c = nullptr;
  }
  if (shard.context != 0) {
    rknn_matmul_destroy(shard.context);
    shard.context = 0;
  }
}

void W4Linear::dequantize_shard_task(void *argument) noexcept {
  auto &task = *static_cast<ShardTask *>(argument);
  try {
    Shard &shard = *task.shard;
    const auto start = Clock::now();
    dequantize_w4a16_columns_to_fp16_kn(
        task.owner->weight_, shard.column_offset, shard.columns,
        static_cast<llmc_float16 *>(shard.b->virt_addr),
        static_cast<size_t>(task.owner->input_columns_) * shard.columns);
    task.elapsed = elapsed_ms(start, Clock::now());
  } catch (...) {
    task.error = std::current_exception();
  }
}

void W4Linear::bind_resident_weight_task(void *argument) noexcept {
  auto &task = *static_cast<ShardTask *>(argument);
  Shard &shard = *task.shard;
  task.operation = "rknn_matmul_set_io_mem(B shard)";
  task.result =
      rknn_matmul_set_io_mem(shard.context, shard.b, &shard.io.B);
}

void W4Linear::run_shard_task(void *argument) noexcept {
  auto &task = *static_cast<ShardTask *>(argument);
  Shard &shard = *task.shard;
  task.operation = "rknn_matmul_set_io_mem(A shard)";
  task.result =
      rknn_matmul_set_io_mem(shard.context, shard.a, &shard.io.A);
  if (task.result != RKNN_SUCC)
    return;
  task.operation = "rknn_matmul_run(W4 shard)";
  task.result = rknn_matmul_run(shard.context);
}

W4Linear::W4Linear(const W4Tensor &weight, int rows, W4RunMode mode,
                   rknn_core_mask core_mask)
    : W4Linear(weight, rows, mode,
               std::vector<rknn_core_mask>{core_mask}) {}

W4Linear::W4Linear(const W4Tensor &weight, int rows, W4RunMode mode,
                   std::vector<rknn_core_mask> core_masks)
    : weight_(weight), rows_(rows),
      input_columns_(static_cast<int>(weight.shape[1])),
      output_columns_(static_cast<int>(weight.shape[0])), mode_(mode),
      scheduler_(static_npu_scheduler()) {
  if (!is_w4_encoding(weight.encoding) || weight.dimensions != 2 ||
      weight.group_size != 32 || rows <= 0) {
    throw std::runtime_error("W4Linear requires a 2-D group-32 W4A16 tensor");
  }
  if (input_columns_ % 32 != 0 || output_columns_ % 16 != 0) {
    throw std::runtime_error("W4Linear shape violates RK3588 FP16 alignment");
  }
  if (core_masks.empty() ||
      core_masks.size() > StaticNpuScheduler::kWorkerCount)
    throw std::runtime_error("W4Linear requires at least one NPU core");

  const auto layout = split_columns(output_columns_, core_masks.size(), 32);
  shard_count_ = layout.size();
  for (size_t index = 0; index < layout.size(); ++index) {
    shards_[index] = std::make_unique<Shard>();
    auto &shard = shards_[index];
    shard->column_offset = layout[index].offset;
    shard->columns = layout[index].columns;
    shard->core_mask = core_masks[index];
    shard->info.M = rows_;
    shard->info.K = input_columns_;
    shard->info.N = shard->columns;
    shard->info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
    shard->info.B_layout = RKNN_MM_LAYOUT_NORM;
    shard->info.AC_layout = RKNN_MM_LAYOUT_NORM;
  }
  std::array<ShardTask, StaticNpuScheduler::kWorkerCount> create_jobs{};
  std::array<StaticNpuScheduler::Task, StaticNpuScheduler::kWorkerCount>
      create_tasks{};
  for (size_t index = 0; index < shard_count_; ++index) {
    create_jobs[index] = {this, shards_[index].get()};
    create_tasks[index] = {create_shard_task, &create_jobs[index]};
  }
  scheduler_.dispatch(create_tasks);
  for (size_t index = 0; index < shard_count_; ++index) {
    if (create_jobs[index].result != RKNN_SUCC) {
      const std::string operation = create_jobs[index].operation != nullptr
                                        ? create_jobs[index].operation
                                        : "W4 shard creation";
      const int result = create_jobs[index].result;
      destroy_shards();
      throw std::runtime_error(operation + " failed: " +
                               std::to_string(result));
    }
  }
  if (mode_ == W4RunMode::kLlmc) {
    prepare_weights();
    const auto upload_start = Clock::now();
    std::array<ShardTask, StaticNpuScheduler::kWorkerCount> bind_jobs{};
    std::array<StaticNpuScheduler::Task, StaticNpuScheduler::kWorkerCount>
        bind_tasks{};
    for (size_t index = 0; index < shard_count_; ++index) {
      bind_jobs[index] = {this, shards_[index].get()};
      bind_tasks[index] = {bind_resident_weight_task, &bind_jobs[index]};
    }
    scheduler_.dispatch(bind_tasks);
    for (size_t index = 0; index < shard_count_; ++index)
      check(bind_jobs[index].result, bind_jobs[index].operation);
    metrics_.weight_upload_ms += elapsed_ms(upload_start, Clock::now());
  }
}

W4Linear::~W4Linear() { destroy_shards(); }

void W4Linear::destroy_shards() noexcept {
  std::array<ShardTask, StaticNpuScheduler::kWorkerCount> jobs{};
  std::array<StaticNpuScheduler::Task, StaticNpuScheduler::kWorkerCount> tasks{};
  for (size_t index = 0; index < shard_count_; ++index) {
    jobs[index] = {this, shards_[index].get()};
    tasks[index] = {destroy_shard_task, &jobs[index]};
  }
  if (shard_count_ != 0)
    scheduler_.dispatch(tasks);
  shard_count_ = 0;
}

void W4Linear::check(int rc, const char *operation) const {
  if (rc != RKNN_SUCC) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + std::to_string(rc));
  }
}

void W4Linear::prepare_weights() {
  std::array<ShardTask, StaticNpuScheduler::kWorkerCount> jobs{};
  std::array<StaticNpuScheduler::Task, StaticNpuScheduler::kWorkerCount> tasks{};
  for (size_t index = 0; index < shard_count_; ++index) {
    jobs[index] = {this, shards_[index].get()};
    tasks[index] = {dequantize_shard_task, &jobs[index]};
  }
  scheduler_.dispatch(tasks);
  double maximum_elapsed = 0.0;
  for (size_t index = 0; index < shard_count_; ++index) {
    if (jobs[index].error)
      std::rethrow_exception(jobs[index].error);
    maximum_elapsed = std::max(maximum_elapsed, jobs[index].elapsed);
  }
  metrics_.weight_dequant_ms += maximum_elapsed;
  metrics_.weight_dequantizations += shard_count_;
  metrics_.weight_uploads += shard_count_;
}

void W4Linear::run(const llmc_float16 *input_data, llmc_float16 *output_data) {
  if (mode_ == W4RunMode::kBaseline) {
    prepare_weights();
    const auto upload_start = Clock::now();
    std::array<ShardTask, StaticNpuScheduler::kWorkerCount> bind_jobs{};
    std::array<StaticNpuScheduler::Task, StaticNpuScheduler::kWorkerCount>
        bind_tasks{};
    for (size_t index = 0; index < shard_count_; ++index) {
      bind_jobs[index] = {this, shards_[index].get()};
      bind_tasks[index] = {bind_resident_weight_task, &bind_jobs[index]};
    }
    scheduler_.dispatch(bind_tasks);
    for (size_t index = 0; index < shard_count_; ++index)
      check(bind_jobs[index].result, bind_jobs[index].operation);
    metrics_.weight_upload_ms += elapsed_ms(upload_start, Clock::now());
  }
  auto start = Clock::now();
  const size_t a_bytes = static_cast<size_t>(rows_) * input_columns_ *
                         sizeof(llmc_float16);
  for (size_t index = 0; index < shard_count_; ++index) {
    auto &shard = shards_[index];
    std::memcpy(shard->a->virt_addr, input_data, a_bytes);
  }
  metrics_.input_copy_ms += elapsed_ms(start, Clock::now());

  start = Clock::now();
  std::array<ShardTask, StaticNpuScheduler::kWorkerCount> run_jobs{};
  std::array<StaticNpuScheduler::Task, StaticNpuScheduler::kWorkerCount>
      run_tasks{};
  for (size_t index = 0; index < shard_count_; ++index) {
    run_jobs[index] = {this, shards_[index].get()};
    run_tasks[index] = {run_shard_task, &run_jobs[index]};
  }
  scheduler_.dispatch(run_tasks);
  metrics_.npu_run_ms += elapsed_ms(start, Clock::now());
  for (size_t index = 0; index < shard_count_; ++index)
    check(run_jobs[index].result, run_jobs[index].operation);
  metrics_.npu_jobs += shard_count_;
  ++metrics_.runs;

  start = Clock::now();
  for (int row = 0; row < rows_; ++row) {
    for (size_t index = 0; index < shard_count_; ++index) {
      const auto &shard = shards_[index];
      const auto *source = static_cast<const llmc_float16 *>(
                               shard->c->virt_addr) +
                           static_cast<size_t>(row) * shard->columns;
      llmc_float16 *target = output_data +
                             static_cast<size_t>(row) * output_columns_ +
                             shard->column_offset;
      std::memcpy(target, source,
                  static_cast<size_t>(shard->columns) *
                      sizeof(llmc_float16));
    }
  }
  metrics_.output_copy_ms += elapsed_ms(start, Clock::now());
}

struct AotW4LinearPlan::Shard {
  int column_offset = 0;
  int columns = 0;
  rknn_core_mask core_mask = RKNN_NPU_CORE_AUTO;
  rknn_matmul_ctx context = 0;
  rknn_matmul_info info{};
  rknn_matmul_io_attr io{};
  rknn_tensor_mem *a = nullptr;
  rknn_tensor_mem *c = nullptr;
};

struct AotW4LinearPlan::ShardTask {
  AotW4LinearPlan *owner = nullptr;
  Shard *shard = nullptr;
  Weight *weight = nullptr;
  const W4Tensor *tensor = nullptr;
  size_t worker_index = 0;
  int result = RKNN_SUCC;
  const char *operation = nullptr;
  double elapsed = 0.0;
  std::exception_ptr error;
};

void AotW4LinearPlan::create_shard_task(void *argument) noexcept {
  auto &task = *static_cast<ShardTask *>(argument);
  Shard &shard = *task.shard;
  task.operation = "rknn_matmul_create(AOT W4 plan)";
  task.result = rknn_matmul_create(&shard.context, &shard.info, &shard.io);
  if (task.result != RKNN_SUCC)
    return;
  task.operation = "rknn_matmul_set_core_mask(AOT W4 plan)";
  task.result = rknn_matmul_set_core_mask(shard.context, shard.core_mask);
  if (task.result != RKNN_SUCC)
    return;
  shard.a = rknn_create_mem(shard.context, shard.io.A.size);
  shard.c = rknn_create_mem(shard.context, shard.io.C.size);
  if (shard.a == nullptr || shard.c == nullptr) {
    task.operation = "rknn_create_mem(AOT W4 A/C)";
    task.result = -1;
    return;
  }
  task.operation = "rknn_matmul_set_io_mem(AOT W4 C)";
  task.result =
      rknn_matmul_set_io_mem(shard.context, shard.c, &shard.io.C);
}

void AotW4LinearPlan::destroy_shard_task(void *argument) noexcept {
  auto &task = *static_cast<ShardTask *>(argument);
  Shard &shard = *task.shard;
  if (shard.a != nullptr) {
    rknn_destroy_mem(shard.context, shard.a);
    shard.a = nullptr;
  }
  if (shard.c != nullptr) {
    rknn_destroy_mem(shard.context, shard.c);
    shard.c = nullptr;
  }
  if (shard.context != 0) {
    rknn_matmul_destroy(shard.context);
    shard.context = 0;
  }
}

void AotW4LinearPlan::prepare_weight_task(void *argument) noexcept {
  auto &task = *static_cast<ShardTask *>(argument);
  try {
    Shard &shard = *task.shard;
    const auto start = Clock::now();
    rknn_tensor_mem *&buffer = task.weight->buffers_[task.worker_index];
    buffer = rknn_create_mem(shard.context, shard.io.B.size);
    if (buffer == nullptr) {
      task.operation = "rknn_create_mem(AOT W4 B)";
      task.result = -1;
      return;
    }
    dequantize_w4a16_columns_to_fp16_kn(
        *task.tensor, shard.column_offset, shard.columns,
        static_cast<llmc_float16 *>(buffer->virt_addr),
        static_cast<size_t>(task.owner->input_columns_) * shard.columns);
    task.elapsed = elapsed_ms(start, Clock::now());
  } catch (...) {
    task.error = std::current_exception();
  }
}

void AotW4LinearPlan::destroy_weight_task(void *argument) noexcept {
  auto &task = *static_cast<ShardTask *>(argument);
  rknn_tensor_mem *&buffer = task.weight->buffers_[task.worker_index];
  if (buffer != nullptr) {
    rknn_destroy_mem(task.shard->context, buffer);
    buffer = nullptr;
  }
}

void AotW4LinearPlan::run_shard_task(void *argument) noexcept {
  auto &task = *static_cast<ShardTask *>(argument);
  Shard &shard = *task.shard;
  task.operation = "rknn_matmul_set_io_mem(AOT W4 A)";
  task.result =
      rknn_matmul_set_io_mem(shard.context, shard.a, &shard.io.A);
  if (task.result != RKNN_SUCC)
    return;
  task.operation = "rknn_matmul_set_io_mem(AOT W4 B)";
  task.result = rknn_matmul_set_io_mem(
      shard.context, task.weight->buffers_[task.worker_index], &shard.io.B);
  if (task.result != RKNN_SUCC)
    return;
  task.operation = "rknn_matmul_run(AOT W4 shard)";
  task.result = rknn_matmul_run(shard.context);
}

AotW4LinearPlan::AotW4LinearPlan(
    int rows, int input_columns, int output_columns,
    std::vector<rknn_core_mask> core_masks)
    : rows_(rows), input_columns_(input_columns),
      output_columns_(output_columns), scheduler_(static_npu_scheduler()) {
  if (rows_ <= 0 || input_columns_ <= 0 || input_columns_ % 32 != 0 ||
      output_columns_ <= 0 || output_columns_ % 32 != 0 ||
      core_masks.size() != StaticNpuScheduler::kWorkerCount) {
    throw std::runtime_error("invalid AOT W4 linear plan shape or core set");
  }
  const auto layout = split_columns(output_columns_, core_masks.size(), 32);
  shard_count_ = layout.size();
  for (size_t index = 0; index < shard_count_; ++index) {
    shards_[index] = std::make_unique<Shard>();
    Shard &shard = *shards_[index];
    shard.column_offset = layout[index].offset;
    shard.columns = layout[index].columns;
    shard.core_mask = core_masks[index];
    shard.info.M = rows_;
    shard.info.K = input_columns_;
    shard.info.N = shard.columns;
    shard.info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
    shard.info.B_layout = RKNN_MM_LAYOUT_NORM;
    shard.info.AC_layout = RKNN_MM_LAYOUT_NORM;
  }
  std::array<ShardTask, StaticNpuScheduler::kWorkerCount> jobs{};
  std::array<StaticNpuScheduler::Task, StaticNpuScheduler::kWorkerCount> tasks{};
  for (size_t index = 0; index < shard_count_; ++index) {
    jobs[index] = {this, shards_[index].get(), nullptr, nullptr, index};
    tasks[index] = {create_shard_task, &jobs[index]};
  }
  scheduler_.dispatch(tasks);
  for (size_t index = 0; index < shard_count_; ++index) {
    if (jobs[index].result != RKNN_SUCC) {
      const std::string operation = jobs[index].operation != nullptr
                                        ? jobs[index].operation
                                        : "AOT W4 plan creation";
      const int result = jobs[index].result;
      destroy_shards();
      throw std::runtime_error(operation + " failed: " +
                               std::to_string(result));
    }
  }
}

AotW4LinearPlan::~AotW4LinearPlan() { destroy_shards(); }

AotW4LinearPlan::Weight::~Weight() {
  if (owner_ != nullptr)
    owner_->destroy_weight(*this);
}

void AotW4LinearPlan::check(int result, const char *operation) const {
  if (result != RKNN_SUCC) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             std::to_string(result));
  }
}

std::unique_ptr<AotW4LinearPlan::Weight>
AotW4LinearPlan::prepare(const W4Tensor &tensor) {
  if (!is_w4_encoding(tensor.encoding) || tensor.dimensions != 2 ||
      static_cast<int>(tensor.shape[0]) != output_columns_ ||
      static_cast<int>(tensor.shape[1]) != input_columns_) {
    throw std::runtime_error("W4 tensor does not match AOT linear plan");
  }
  auto weight = std::unique_ptr<Weight>(new Weight(this));
  std::array<ShardTask, StaticNpuScheduler::kWorkerCount> jobs{};
  std::array<StaticNpuScheduler::Task, StaticNpuScheduler::kWorkerCount> tasks{};
  for (size_t index = 0; index < shard_count_; ++index) {
    jobs[index] = {this, shards_[index].get(), weight.get(), &tensor, index};
    tasks[index] = {prepare_weight_task, &jobs[index]};
  }
  const auto start = Clock::now();
  scheduler_.dispatch(tasks);
  for (size_t index = 0; index < shard_count_; ++index) {
    if (jobs[index].error)
      std::rethrow_exception(jobs[index].error);
    check(jobs[index].result,
          jobs[index].operation != nullptr ? jobs[index].operation
                                           : "AOT W4 prepare");
    weight->bytes_ += shards_[index]->io.B.size;
  }
  metrics_.weight_dequant_ms += elapsed_ms(start, Clock::now());
  metrics_.weight_dequantizations += shard_count_;
  metrics_.weight_uploads += shard_count_;
  metrics_.weight_upload_ms += elapsed_ms(start, Clock::now());
  return weight;
}

void AotW4LinearPlan::destroy_weight(Weight &weight) noexcept {
  std::array<ShardTask, StaticNpuScheduler::kWorkerCount> jobs{};
  std::array<StaticNpuScheduler::Task, StaticNpuScheduler::kWorkerCount> tasks{};
  for (size_t index = 0; index < shard_count_; ++index) {
    jobs[index] = {this, shards_[index].get(), &weight, nullptr, index};
    tasks[index] = {destroy_weight_task, &jobs[index]};
  }
  scheduler_.dispatch(tasks);
  weight.owner_ = nullptr;
  weight.bytes_ = 0;
}

void AotW4LinearPlan::destroy_shards() noexcept {
  std::array<ShardTask, StaticNpuScheduler::kWorkerCount> jobs{};
  std::array<StaticNpuScheduler::Task, StaticNpuScheduler::kWorkerCount> tasks{};
  for (size_t index = 0; index < shard_count_; ++index) {
    jobs[index] = {this, shards_[index].get(), nullptr, nullptr, index};
    tasks[index] = {destroy_shard_task, &jobs[index]};
  }
  if (shard_count_ != 0)
    scheduler_.dispatch(tasks);
  shard_count_ = 0;
}

void AotW4LinearPlan::run(const Weight &weight,
                          const llmc_float16 *input_data,
                          llmc_float16 *output_data) {
  if (weight.owner_ != this)
    throw std::runtime_error("AOT W4 weight belongs to a different plan");
  const auto copy_start = Clock::now();
  const size_t a_bytes = static_cast<size_t>(rows_) * input_columns_ *
                         sizeof(llmc_float16);
  for (size_t index = 0; index < shard_count_; ++index)
    std::memcpy(shards_[index]->a->virt_addr, input_data, a_bytes);
  metrics_.input_copy_ms += elapsed_ms(copy_start, Clock::now());

  std::array<ShardTask, StaticNpuScheduler::kWorkerCount> jobs{};
  std::array<StaticNpuScheduler::Task, StaticNpuScheduler::kWorkerCount> tasks{};
  for (size_t index = 0; index < shard_count_; ++index) {
    jobs[index] = {this, shards_[index].get(),
                   const_cast<Weight *>(&weight), nullptr, index};
    tasks[index] = {run_shard_task, &jobs[index]};
  }
  const auto run_start = Clock::now();
  scheduler_.dispatch(tasks);
  metrics_.npu_run_ms += elapsed_ms(run_start, Clock::now());
  for (size_t index = 0; index < shard_count_; ++index)
    check(jobs[index].result, jobs[index].operation);
  metrics_.npu_jobs += shard_count_;
  ++metrics_.runs;

  const auto output_start = Clock::now();
  for (int row = 0; row < rows_; ++row) {
    for (size_t index = 0; index < shard_count_; ++index) {
      const Shard &shard = *shards_[index];
      const auto *source = static_cast<const llmc_float16 *>(
                               shard.c->virt_addr) +
                           static_cast<size_t>(row) * shard.columns;
      llmc_float16 *target = output_data +
                             static_cast<size_t>(row) * output_columns_ +
                             shard.column_offset;
      std::memcpy(target, source,
                  static_cast<size_t>(shard.columns) * sizeof(llmc_float16));
    }
  }
  metrics_.output_copy_ms += elapsed_ms(output_start, Clock::now());
}

struct F16Matmul::Shard {
  int column_offset = 0;
  int columns = 0;
  rknn_core_mask core_mask = RKNN_NPU_CORE_AUTO;
  rknn_matmul_ctx context = 0;
  rknn_matmul_io_attr io{};
  rknn_tensor_mem *a = nullptr;
  rknn_tensor_mem *b = nullptr;
  rknn_tensor_mem *c = nullptr;
};

struct F16Matmul::ShardTask {
  F16Matmul *owner = nullptr;
  Shard *shard = nullptr;
  int result = RKNN_SUCC;
  const char *operation = nullptr;
};

void F16Matmul::create_shard_task(void *argument) noexcept {
  auto &task = *static_cast<ShardTask *>(argument);
  Shard &shard = *task.shard;
  rknn_matmul_info info{};
  info.M = task.owner->rows_;
  info.K = task.owner->inner_;
  info.N = shard.columns;
  info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
  info.B_layout = RKNN_MM_LAYOUT_NORM;
  info.AC_layout = RKNN_MM_LAYOUT_NORM;
  task.operation = "rknn_matmul_create(FP16)";
  task.result = rknn_matmul_create(&shard.context, &info, &shard.io);
  if (task.result != RKNN_SUCC)
    return;
  if (shard.core_mask != RKNN_NPU_CORE_AUTO) {
    task.operation = "rknn_matmul_set_core_mask(FP16)";
    task.result = rknn_matmul_set_core_mask(shard.context, shard.core_mask);
    if (task.result != RKNN_SUCC)
      return;
  }
  shard.a = rknn_create_mem(shard.context, shard.io.A.size);
  shard.b = rknn_create_mem(shard.context, shard.io.B.size);
  shard.c = rknn_create_mem(shard.context, shard.io.C.size);
  if (shard.a == nullptr || shard.b == nullptr || shard.c == nullptr) {
    task.operation = "rknn_create_mem(FP16 shard)";
    task.result = -1;
    return;
  }
  task.operation = "rknn_matmul_set_io_mem(FP16 C shard)";
  task.result =
      rknn_matmul_set_io_mem(shard.context, shard.c, &shard.io.C);
}

void F16Matmul::destroy_shard_task(void *argument) noexcept {
  auto &task = *static_cast<ShardTask *>(argument);
  Shard &shard = *task.shard;
  if (shard.a != nullptr) {
    rknn_destroy_mem(shard.context, shard.a);
    shard.a = nullptr;
  }
  if (shard.b != nullptr) {
    rknn_destroy_mem(shard.context, shard.b);
    shard.b = nullptr;
  }
  if (shard.c != nullptr) {
    rknn_destroy_mem(shard.context, shard.c);
    shard.c = nullptr;
  }
  if (shard.context != 0) {
    rknn_matmul_destroy(shard.context);
    shard.context = 0;
  }
}

void F16Matmul::run_shard_task(void *argument) noexcept {
  auto &task = *static_cast<ShardTask *>(argument);
  Shard &shard = *task.shard;
  task.operation = "rknn_matmul_set_io_mem(FP16 A shard)";
  task.result =
      rknn_matmul_set_io_mem(shard.context, shard.a, &shard.io.A);
  if (task.result != RKNN_SUCC)
    return;
  task.operation = "rknn_matmul_set_io_mem(FP16 B shard)";
  task.result =
      rknn_matmul_set_io_mem(shard.context, shard.b, &shard.io.B);
  if (task.result != RKNN_SUCC)
    return;
  task.operation = "rknn_matmul_run(FP16 shard)";
  task.result = rknn_matmul_run(shard.context);
}

F16Matmul::F16Matmul(int rows, int inner, int columns, rknn_core_mask core_mask)
    : F16Matmul(rows, inner, columns,
                std::vector<rknn_core_mask>{core_mask}) {}

F16Matmul::F16Matmul(int rows, int inner, int columns,
                     std::vector<rknn_core_mask> core_masks)
    : rows_(rows), inner_(inner), columns_(columns),
      scheduler_(static_npu_scheduler()) {
  if (rows <= 0 || inner <= 0 || columns <= 0 || inner % 32 != 0 ||
      columns % 16 != 0) {
    throw std::runtime_error("invalid RK3588 FP16 MatMul shape");
  }
  if (core_masks.empty() ||
      core_masks.size() > StaticNpuScheduler::kWorkerCount)
    throw std::runtime_error("FP16 MatMul requires at least one NPU core");
  const auto layout = split_columns(columns_, core_masks.size(), 16);
  shard_count_ = layout.size();
  for (size_t index = 0; index < layout.size(); ++index) {
    shards_[index] = std::make_unique<Shard>();
    auto &shard = shards_[index];
    shard->column_offset = layout[index].offset;
    shard->columns = layout[index].columns;
    shard->core_mask = core_masks[index];
  }
  std::array<ShardTask, StaticNpuScheduler::kWorkerCount> jobs{};
  std::array<StaticNpuScheduler::Task, StaticNpuScheduler::kWorkerCount> tasks{};
  for (size_t index = 0; index < shard_count_; ++index) {
    jobs[index] = {this, shards_[index].get()};
    tasks[index] = {create_shard_task, &jobs[index]};
  }
  scheduler_.dispatch(tasks);
  for (size_t index = 0; index < shard_count_; ++index) {
    if (jobs[index].result != RKNN_SUCC) {
      const std::string operation = jobs[index].operation != nullptr
                                        ? jobs[index].operation
                                        : "FP16 shard creation";
      const int result = jobs[index].result;
      destroy_shards();
      throw std::runtime_error(operation + " failed: " +
                               std::to_string(result));
    }
  }
}

F16Matmul::~F16Matmul() { destroy_shards(); }

void F16Matmul::destroy_shards() noexcept {
  std::array<ShardTask, StaticNpuScheduler::kWorkerCount> jobs{};
  std::array<StaticNpuScheduler::Task, StaticNpuScheduler::kWorkerCount> tasks{};
  for (size_t index = 0; index < shard_count_; ++index) {
    jobs[index] = {this, shards_[index].get()};
    tasks[index] = {destroy_shard_task, &jobs[index]};
  }
  if (shard_count_ != 0)
    scheduler_.dispatch(tasks);
  shard_count_ = 0;
}

void F16Matmul::check(int rc, const char *operation) const {
  if (rc != RKNN_SUCC) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + std::to_string(rc));
  }
}

void F16Matmul::run(const llmc_float16 *a_data, const llmc_float16 *b_data,
                    llmc_float16 *c_data) {
  const size_t a_bytes =
      static_cast<size_t>(rows_) * inner_ * sizeof(llmc_float16);
  auto start = Clock::now();
  for (size_t index = 0; index < shard_count_; ++index) {
    auto &shard = shards_[index];
    std::memcpy(shard->a->virt_addr, a_data, a_bytes);
    auto *shard_b = static_cast<llmc_float16 *>(shard->b->virt_addr);
    for (int inner = 0; inner < inner_; ++inner) {
      std::memcpy(shard_b + static_cast<size_t>(inner) * shard->columns,
                  b_data + static_cast<size_t>(inner) * columns_ +
                      shard->column_offset,
                  static_cast<size_t>(shard->columns) *
                      sizeof(llmc_float16));
    }
  }
  metrics_.input_copy_ms += elapsed_ms(start, Clock::now());

  start = Clock::now();
  std::array<ShardTask, StaticNpuScheduler::kWorkerCount> run_jobs{};
  std::array<StaticNpuScheduler::Task, StaticNpuScheduler::kWorkerCount>
      run_tasks{};
  for (size_t index = 0; index < shard_count_; ++index) {
    run_jobs[index] = {this, shards_[index].get()};
    run_tasks[index] = {run_shard_task, &run_jobs[index]};
  }
  scheduler_.dispatch(run_tasks);
  metrics_.npu_run_ms += elapsed_ms(start, Clock::now());
  for (size_t index = 0; index < shard_count_; ++index)
    check(run_jobs[index].result, run_jobs[index].operation);

  start = Clock::now();
  for (int row = 0; row < rows_; ++row) {
    for (size_t index = 0; index < shard_count_; ++index) {
      const auto &shard = shards_[index];
      const auto *source = static_cast<const llmc_float16 *>(
                               shard->c->virt_addr) +
                           static_cast<size_t>(row) * shard->columns;
      llmc_float16 *target = c_data +
                             static_cast<size_t>(row) * columns_ +
                             shard->column_offset;
      std::memcpy(target, source,
                  static_cast<size_t>(shard->columns) *
                      sizeof(llmc_float16));
    }
  }
  metrics_.output_copy_ms += elapsed_ms(start, Clock::now());
  metrics_.npu_jobs += shard_count_;
  ++metrics_.runs;
}

struct WorkerF16Matmul::Core {
  rknn_core_mask core_mask = RKNN_NPU_CORE_AUTO;
  rknn_matmul_ctx context = 0;
  rknn_matmul_io_attr io{};
  rknn_tensor_mem *a = nullptr;
  rknn_tensor_mem *b = nullptr;
  rknn_tensor_mem *c = nullptr;
};

struct WorkerF16Matmul::CoreTask {
  WorkerF16Matmul *owner = nullptr;
  Core *core = nullptr;
  int result = RKNN_SUCC;
  const char *operation = nullptr;
};

void WorkerF16Matmul::create_core_task(void *argument) noexcept {
  auto &task = *static_cast<CoreTask *>(argument);
  Core &core = *task.core;
  rknn_matmul_info info{};
  info.M = task.owner->rows_;
  info.K = task.owner->inner_;
  info.N = task.owner->columns_;
  info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
  info.B_layout = RKNN_MM_LAYOUT_NORM;
  info.AC_layout = RKNN_MM_LAYOUT_NORM;
  task.operation = "rknn_matmul_create(attention worker)";
  task.result = rknn_matmul_create(&core.context, &info, &core.io);
  if (task.result != RKNN_SUCC)
    return;
  task.operation = "rknn_matmul_set_core_mask(attention worker)";
  task.result = rknn_matmul_set_core_mask(core.context, core.core_mask);
  if (task.result != RKNN_SUCC)
    return;
  core.a = rknn_create_mem(core.context, core.io.A.size);
  core.b = rknn_create_mem(core.context, core.io.B.size);
  core.c = rknn_create_mem(core.context, core.io.C.size);
  if (core.a == nullptr || core.b == nullptr || core.c == nullptr) {
    task.operation = "rknn_create_mem(attention worker)";
    task.result = -1;
    return;
  }
  const uint64_t expected_a = static_cast<uint64_t>(task.owner->rows_) *
                              task.owner->inner_ * sizeof(llmc_float16);
  const uint64_t expected_b = static_cast<uint64_t>(task.owner->inner_) *
                              task.owner->columns_ * sizeof(llmc_float16);
  const uint64_t expected_c = static_cast<uint64_t>(task.owner->rows_) *
                              task.owner->columns_ * sizeof(llmc_float16);
  if (core.io.A.size < expected_a || core.io.B.size < expected_b ||
      core.io.C.size < expected_c) {
    task.operation = "attention worker buffer size validation";
    task.result = -1;
    return;
  }
  task.operation = "rknn_matmul_set_io_mem(attention worker C)";
  task.result = rknn_matmul_set_io_mem(core.context, core.c, &core.io.C);
}

void WorkerF16Matmul::destroy_core_task(void *argument) noexcept {
  auto &task = *static_cast<CoreTask *>(argument);
  Core &core = *task.core;
  if (core.a != nullptr) {
    rknn_destroy_mem(core.context, core.a);
    core.a = nullptr;
  }
  if (core.b != nullptr) {
    rknn_destroy_mem(core.context, core.b);
    core.b = nullptr;
  }
  if (core.c != nullptr) {
    rknn_destroy_mem(core.context, core.c);
    core.c = nullptr;
  }
  if (core.context != 0) {
    rknn_matmul_destroy(core.context);
    core.context = 0;
  }
}

WorkerF16Matmul::WorkerF16Matmul(
    int rows, int inner, int columns,
    std::vector<rknn_core_mask> core_masks)
    : rows_(rows), inner_(inner), columns_(columns),
      scheduler_(static_npu_scheduler()) {
  if (rows_ <= 0 || inner_ <= 0 || inner_ % 32 != 0 || columns_ <= 0 ||
      columns_ % 16 != 0 ||
      core_masks.size() != StaticNpuScheduler::kWorkerCount) {
    throw std::runtime_error("invalid attention-worker FP16 MatMul shape");
  }
  std::array<CoreTask, StaticNpuScheduler::kWorkerCount> jobs{};
  std::array<StaticNpuScheduler::Task, StaticNpuScheduler::kWorkerCount> tasks{};
  for (size_t index = 0; index < StaticNpuScheduler::kWorkerCount; ++index) {
    cores_[index] = std::make_unique<Core>();
    cores_[index]->core_mask = core_masks[index];
    jobs[index] = {this, cores_[index].get()};
    tasks[index] = {create_core_task, &jobs[index]};
  }
  scheduler_.dispatch(tasks);
  for (size_t index = 0; index < StaticNpuScheduler::kWorkerCount; ++index) {
    if (jobs[index].result != RKNN_SUCC) {
      const std::string operation = jobs[index].operation != nullptr
                                        ? jobs[index].operation
                                        : "attention worker creation";
      const int result = jobs[index].result;
      destroy_cores();
      throw std::runtime_error(operation + " failed: " +
                               std::to_string(result));
    }
  }
}

WorkerF16Matmul::~WorkerF16Matmul() { destroy_cores(); }

void WorkerF16Matmul::destroy_cores() noexcept {
  std::array<CoreTask, StaticNpuScheduler::kWorkerCount> jobs{};
  std::array<StaticNpuScheduler::Task, StaticNpuScheduler::kWorkerCount> tasks{};
  for (size_t index = 0; index < StaticNpuScheduler::kWorkerCount; ++index) {
    if (!cores_[index])
      continue;
    jobs[index] = {this, cores_[index].get()};
    tasks[index] = {destroy_core_task, &jobs[index]};
  }
  scheduler_.dispatch(tasks);
}

int WorkerF16Matmul::run_on_worker(size_t worker_index,
                                   const llmc_float16 *a_data,
                                   const llmc_float16 *b_data,
                                   llmc_float16 *c_data) noexcept {
  if (worker_index >= StaticNpuScheduler::kWorkerCount || a_data == nullptr ||
      b_data == nullptr || c_data == nullptr) {
    return -1;
  }
  Core &core = *cores_[worker_index];
  std::memcpy(core.a->virt_addr, a_data,
              static_cast<size_t>(rows_) * inner_ * sizeof(llmc_float16));
  std::memcpy(core.b->virt_addr, b_data,
              static_cast<size_t>(inner_) * columns_ * sizeof(llmc_float16));
  int result = rknn_matmul_set_io_mem(core.context, core.a, &core.io.A);
  if (result != RKNN_SUCC)
    return result;
  result = rknn_matmul_set_io_mem(core.context, core.b, &core.io.B);
  if (result != RKNN_SUCC)
    return result;
  result = rknn_matmul_run(core.context);
  if (result != RKNN_SUCC)
    return result;
  std::memcpy(c_data, core.c->virt_addr,
              static_cast<size_t>(rows_) * columns_ * sizeof(llmc_float16));
  return RKNN_SUCC;
}

} // namespace llmc
