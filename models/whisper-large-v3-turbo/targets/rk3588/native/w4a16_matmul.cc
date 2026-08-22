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

std::vector<ColumnShard> split_columns(int columns, size_t shard_count) {
  constexpr int kAlignment = 16;
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

  ~Shard() {
    if (a != nullptr)
      rknn_destroy_mem(context, a);
    if (b != nullptr)
      rknn_destroy_mem(context, b);
    if (c != nullptr)
      rknn_destroy_mem(context, c);
    if (context != 0)
      rknn_matmul_destroy(context);
  }
};

W4Linear::W4Linear(const W4Tensor &weight, int rows, W4RunMode mode,
                   rknn_core_mask core_mask)
    : W4Linear(weight, rows, mode,
               std::vector<rknn_core_mask>{core_mask}) {}

W4Linear::W4Linear(const W4Tensor &weight, int rows, W4RunMode mode,
                   std::vector<rknn_core_mask> core_masks)
    : weight_(weight), rows_(rows),
      input_columns_(static_cast<int>(weight.shape[1])),
      output_columns_(static_cast<int>(weight.shape[0])), mode_(mode) {
  if (weight.encoding != W4Encoding::kW4A16Group || weight.dimensions != 2 ||
      weight.group_size != 32 || rows <= 0) {
    throw std::runtime_error("W4Linear requires a 2-D group-32 W4A16 tensor");
  }
  if (input_columns_ % 32 != 0 || output_columns_ % 16 != 0) {
    throw std::runtime_error("W4Linear shape violates RK3588 FP16 alignment");
  }
  if (core_masks.empty())
    throw std::runtime_error("W4Linear requires at least one NPU core");

  const auto layout = split_columns(output_columns_, core_masks.size());
  shards_.reserve(layout.size());
  for (size_t index = 0; index < layout.size(); ++index) {
    auto shard = std::make_unique<Shard>();
    shard->column_offset = layout[index].offset;
    shard->columns = layout[index].columns;
    shard->core_mask = core_masks[index];
    shard->info.M = rows_;
    shard->info.K = input_columns_;
    shard->info.N = shard->columns;
    shard->info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
    shard->info.B_layout = RKNN_MM_LAYOUT_NORM;
    shard->info.AC_layout = RKNN_MM_LAYOUT_NORM;
    check(rknn_matmul_create(&shard->context, &shard->info, &shard->io),
          "rknn_matmul_create");
    if (shard->core_mask != RKNN_NPU_CORE_AUTO) {
      check(rknn_matmul_set_core_mask(shard->context, shard->core_mask),
            "rknn_matmul_set_core_mask");
    }
    shard->a = rknn_create_mem(shard->context, shard->io.A.size);
    shard->b = rknn_create_mem(shard->context, shard->io.B.size);
    shard->c = rknn_create_mem(shard->context, shard->io.C.size);
    if (shard->a == nullptr || shard->b == nullptr || shard->c == nullptr) {
      throw std::runtime_error("rknn_create_mem failed for W4Linear");
    }
    const uint64_t expected_a =
        static_cast<uint64_t>(rows_) * input_columns_ * sizeof(llmc_float16);
    const uint64_t expected_c =
        static_cast<uint64_t>(rows_) * shard->columns * sizeof(llmc_float16);
    const uint64_t expected_b = static_cast<uint64_t>(input_columns_) *
                                shard->columns * sizeof(llmc_float16);
    if (shard->io.A.size < expected_a || shard->io.B.size < expected_b ||
        shard->io.C.size < expected_c) {
      throw std::runtime_error("RKNPU2 returned undersized MatMul buffers");
    }
    check(rknn_matmul_set_io_mem(shard->context, shard->c, &shard->io.C),
          "rknn_matmul_set_io_mem(C shard)");
    shards_.push_back(std::move(shard));
  }
  if (mode_ == W4RunMode::kLlmc) {
    prepare_weights();
    const auto upload_start = Clock::now();
    for (auto &shard : shards_) {
      check(rknn_matmul_set_io_mem(shard->context, shard->b, &shard->io.B),
            "rknn_matmul_set_io_mem(B resident shard)");
    }
    metrics_.weight_upload_ms += elapsed_ms(upload_start, Clock::now());
  }
}

W4Linear::~W4Linear() = default;

void W4Linear::check(int rc, const char *operation) const {
  if (rc != RKNN_SUCC) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + std::to_string(rc));
  }
}

void W4Linear::prepare_weights() {
  std::vector<double> elapsed(shards_.size(), 0.0);
  std::vector<std::exception_ptr> errors(shards_.size());
  std::vector<std::thread> workers;
  workers.reserve(shards_.size());
  for (size_t index = 0; index < shards_.size(); ++index) {
    workers.emplace_back([&, index] {
      try {
        Shard &shard = *shards_[index];
        const auto start = Clock::now();
        dequantize_w4a16_columns_to_fp16_kn(
            weight_, shard.column_offset, shard.columns,
            static_cast<llmc_float16 *>(shard.b->virt_addr),
            static_cast<size_t>(input_columns_) * shard.columns);
        elapsed[index] = elapsed_ms(start, Clock::now());
      } catch (...) {
        errors[index] = std::current_exception();
      }
    });
  }
  for (auto &worker : workers)
    worker.join();
  for (const auto &error : errors) {
    if (error)
      std::rethrow_exception(error);
  }
  metrics_.weight_dequant_ms +=
      *std::max_element(elapsed.begin(), elapsed.end());
  metrics_.weight_dequantizations += shards_.size();
  metrics_.weight_uploads += shards_.size();
}

void W4Linear::run(const llmc_float16 *input_data, llmc_float16 *output_data) {
  if (mode_ == W4RunMode::kBaseline) {
    prepare_weights();
    const auto upload_start = Clock::now();
    for (auto &shard : shards_) {
      check(rknn_matmul_set_io_mem(shard->context, shard->b, &shard->io.B),
            "rknn_matmul_set_io_mem(B baseline shard)");
    }
    metrics_.weight_upload_ms += elapsed_ms(upload_start, Clock::now());
  }
  auto start = Clock::now();
  const size_t a_bytes = static_cast<size_t>(rows_) * input_columns_ *
                         sizeof(llmc_float16);
  for (auto &shard : shards_) {
    std::memcpy(shard->a->virt_addr, input_data, a_bytes);
    check(rknn_matmul_set_io_mem(shard->context, shard->a, &shard->io.A),
          "rknn_matmul_set_io_mem(A shard)");
  }
  metrics_.input_copy_ms += elapsed_ms(start, Clock::now());

  start = Clock::now();
  std::vector<int> results(shards_.size(), RKNN_SUCC);
  std::vector<std::thread> workers;
  workers.reserve(shards_.size());
  for (size_t index = 0; index < shards_.size(); ++index) {
    workers.emplace_back([&, index] {
      results[index] = rknn_matmul_run(shards_[index]->context);
    });
  }
  for (auto &worker : workers)
    worker.join();
  metrics_.npu_run_ms += elapsed_ms(start, Clock::now());
  for (int result : results)
    check(result, "rknn_matmul_run(W4 shard)");
  metrics_.npu_jobs += shards_.size();
  ++metrics_.runs;

  start = Clock::now();
  for (int row = 0; row < rows_; ++row) {
    for (const auto &shard : shards_) {
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

struct F16Matmul::Shard {
  int column_offset = 0;
  int columns = 0;
  rknn_core_mask core_mask = RKNN_NPU_CORE_AUTO;
  rknn_matmul_ctx context = 0;
  rknn_matmul_io_attr io{};
  rknn_tensor_mem *a = nullptr;
  rknn_tensor_mem *b = nullptr;
  rknn_tensor_mem *c = nullptr;

  ~Shard() {
    if (a != nullptr)
      rknn_destroy_mem(context, a);
    if (b != nullptr)
      rknn_destroy_mem(context, b);
    if (c != nullptr)
      rknn_destroy_mem(context, c);
    if (context != 0)
      rknn_matmul_destroy(context);
  }
};

F16Matmul::F16Matmul(int rows, int inner, int columns, rknn_core_mask core_mask)
    : F16Matmul(rows, inner, columns,
                std::vector<rknn_core_mask>{core_mask}) {}

F16Matmul::F16Matmul(int rows, int inner, int columns,
                     std::vector<rknn_core_mask> core_masks)
    : rows_(rows), inner_(inner), columns_(columns) {
  if (rows <= 0 || inner <= 0 || columns <= 0 || inner % 32 != 0 ||
      columns % 16 != 0) {
    throw std::runtime_error("invalid RK3588 FP16 MatMul shape");
  }
  if (core_masks.empty())
    throw std::runtime_error("FP16 MatMul requires at least one NPU core");
  const auto layout = split_columns(columns_, core_masks.size());
  shards_.reserve(layout.size());
  for (size_t index = 0; index < layout.size(); ++index) {
    auto shard = std::make_unique<Shard>();
    shard->column_offset = layout[index].offset;
    shard->columns = layout[index].columns;
    shard->core_mask = core_masks[index];
    rknn_matmul_info info{};
    info.M = rows_;
    info.K = inner_;
    info.N = shard->columns;
    info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
    info.B_layout = RKNN_MM_LAYOUT_NORM;
    info.AC_layout = RKNN_MM_LAYOUT_NORM;
    check(rknn_matmul_create(&shard->context, &info, &shard->io),
          "rknn_matmul_create(FP16)");
    if (shard->core_mask != RKNN_NPU_CORE_AUTO) {
      check(rknn_matmul_set_core_mask(shard->context, shard->core_mask),
            "rknn_matmul_set_core_mask(FP16)");
    }
    shard->a = rknn_create_mem(shard->context, shard->io.A.size);
    shard->b = rknn_create_mem(shard->context, shard->io.B.size);
    shard->c = rknn_create_mem(shard->context, shard->io.C.size);
    if (shard->a == nullptr || shard->b == nullptr || shard->c == nullptr) {
      throw std::runtime_error("rknn_create_mem failed for FP16 MatMul");
    }
    check(rknn_matmul_set_io_mem(shard->context, shard->c, &shard->io.C),
          "rknn_matmul_set_io_mem(FP16 C shard)");
    shards_.push_back(std::move(shard));
  }
}

F16Matmul::~F16Matmul() = default;

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
  for (auto &shard : shards_) {
    std::memcpy(shard->a->virt_addr, a_data, a_bytes);
    auto *shard_b = static_cast<llmc_float16 *>(shard->b->virt_addr);
    for (int inner = 0; inner < inner_; ++inner) {
      std::memcpy(shard_b + static_cast<size_t>(inner) * shard->columns,
                  b_data + static_cast<size_t>(inner) * columns_ +
                      shard->column_offset,
                  static_cast<size_t>(shard->columns) *
                      sizeof(llmc_float16));
    }
    check(rknn_matmul_set_io_mem(shard->context, shard->a, &shard->io.A),
          "rknn_matmul_set_io_mem(FP16 A shard)");
    check(rknn_matmul_set_io_mem(shard->context, shard->b, &shard->io.B),
          "rknn_matmul_set_io_mem(FP16 B shard)");
  }
  metrics_.input_copy_ms += elapsed_ms(start, Clock::now());

  start = Clock::now();
  std::vector<int> results(shards_.size(), RKNN_SUCC);
  std::vector<std::thread> workers;
  workers.reserve(shards_.size());
  for (size_t index = 0; index < shards_.size(); ++index) {
    workers.emplace_back([&, index] {
      results[index] = rknn_matmul_run(shards_[index]->context);
    });
  }
  for (auto &worker : workers)
    worker.join();
  metrics_.npu_run_ms += elapsed_ms(start, Clock::now());
  for (int result : results)
    check(result, "rknn_matmul_run(FP16 shard)");

  start = Clock::now();
  for (int row = 0; row < rows_; ++row) {
    for (const auto &shard : shards_) {
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
  metrics_.npu_jobs += shards_.size();
  ++metrics_.runs;
}

} // namespace llmc
