#include "w4a16_matmul.h"

#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

namespace llmc {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
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

W4Linear::W4Linear(const W4Tensor &weight, int rows, W4RunMode mode,
                   rknn_core_mask core_mask)
    : weight_(weight), rows_(rows),
      input_columns_(static_cast<int>(weight.shape[1])),
      output_columns_(static_cast<int>(weight.shape[0])), mode_(mode) {
  if (weight.encoding != W4Encoding::kW4A16Group || weight.dimensions != 2 ||
      weight.group_size != 32 || rows <= 0) {
    throw std::runtime_error("W4Linear requires a 2-D group-32 W4A16 tensor");
  }
  if (input_columns_ % 32 != 0 || output_columns_ % 64 != 0) {
    throw std::runtime_error("W4Linear shape violates RK3588 INT4 alignment");
  }

  info_.M = rows_;
  info_.K = input_columns_;
  info_.N = output_columns_;
  info_.type = RKNN_FLOAT16_MM_INT4_TO_FLOAT16;
  info_.B_layout =
      mode_ == W4RunMode::kLlmc ? RKNN_MM_LAYOUT_NATIVE : RKNN_MM_LAYOUT_NORM;
  info_.B_quant_type = RKNN_QUANT_TYPE_PER_GROUP_SYM;
  info_.AC_layout = RKNN_MM_LAYOUT_NORM;
  info_.AC_quant_type = RKNN_QUANT_TYPE_PER_LAYER_SYM;
  info_.group_size = 32;

  try {
    check(rknn_matmul_create(&context_, &info_, &io_), "rknn_matmul_create");
    if (core_mask != RKNN_NPU_CORE_AUTO) {
      check(rknn_matmul_set_core_mask(context_, core_mask),
            "rknn_matmul_set_core_mask");
    }

    rknn_quant_params scales{};
    std::memcpy(scales.name, io_.B.name, RKNN_MAX_NAME_LEN);
    scales.scale = const_cast<float *>(weight_.scales);
    scales.scale_len = static_cast<int32_t>(weight_.scale_size / sizeof(float));
    check(rknn_matmul_set_quant_params(context_, &scales),
          "rknn_matmul_set_quant_params(B)");

    a_ = rknn_create_mem(context_, io_.A.size);
    b_ = rknn_create_mem(context_, io_.B.size);
    c_ = rknn_create_mem(context_, io_.C.size);
    if (a_ == nullptr || b_ == nullptr || c_ == nullptr) {
      throw std::runtime_error("rknn_create_mem failed for W4Linear");
    }
    check(rknn_matmul_set_io_mem(context_, a_, &io_.A),
          "rknn_matmul_set_io_mem(A)");
    check(rknn_matmul_set_io_mem(context_, b_, &io_.B),
          "rknn_matmul_set_io_mem(B)");
    check(rknn_matmul_set_io_mem(context_, c_, &io_.C),
          "rknn_matmul_set_io_mem(C)");

    const uint64_t expected_a =
        static_cast<uint64_t>(rows_) * input_columns_ * sizeof(llmc_float16);
    const uint64_t expected_c =
        static_cast<uint64_t>(rows_) * output_columns_ * sizeof(llmc_float16);
    if (io_.A.size < expected_a || io_.C.size < expected_c ||
        io_.B.size < weight_.data_size) {
      throw std::runtime_error("RKNPU2 returned undersized MatMul buffers");
    }
    if (mode_ == W4RunMode::kLlmc)
      upload_weight();
  } catch (...) {
    if (a_ != nullptr)
      rknn_destroy_mem(context_, a_);
    if (b_ != nullptr)
      rknn_destroy_mem(context_, b_);
    if (c_ != nullptr)
      rknn_destroy_mem(context_, c_);
    if (context_ != 0)
      rknn_matmul_destroy(context_);
    a_ = b_ = c_ = nullptr;
    context_ = 0;
    throw;
  }
}

W4Linear::~W4Linear() {
  if (a_ != nullptr)
    rknn_destroy_mem(context_, a_);
  if (b_ != nullptr)
    rknn_destroy_mem(context_, b_);
  if (c_ != nullptr)
    rknn_destroy_mem(context_, c_);
  if (context_ != 0)
    rknn_matmul_destroy(context_);
}

llmc_float16 *W4Linear::input() {
  return static_cast<llmc_float16 *>(a_->virt_addr);
}

const llmc_float16 *W4Linear::output() const {
  return static_cast<const llmc_float16 *>(c_->virt_addr);
}

void W4Linear::check(int rc, const char *operation) const {
  if (rc != RKNN_SUCC) {
    throw std::runtime_error(std::string(operation) +
                             " failed: " + std::to_string(rc));
  }
}

void W4Linear::upload_weight() {
  const auto start = Clock::now();
  if (mode_ == W4RunMode::kLlmc) {
    check(rknn_B_normal_layout_to_native_layout(
              const_cast<void *>(weight_.data), b_->virt_addr, input_columns_,
              output_columns_, &info_),
          "rknn_B_normal_layout_to_native_layout");
  } else {
    std::memcpy(b_->virt_addr, weight_.data, weight_.data_size);
  }
  check(rknn_mem_sync(context_, b_, RKNN_MEMORY_SYNC_TO_DEVICE),
        "rknn_mem_sync(B to device)");
  metrics_.weight_upload_ms += elapsed_ms(start, Clock::now());
  ++metrics_.weight_uploads;
}

void W4Linear::run_bound() {
  if (mode_ == W4RunMode::kBaseline)
    upload_weight();
  check(rknn_mem_sync(context_, a_, RKNN_MEMORY_SYNC_TO_DEVICE),
        "rknn_mem_sync(A to device)");
  const auto start = Clock::now();
  check(rknn_matmul_run(context_), "rknn_matmul_run");
  metrics_.npu_run_ms += elapsed_ms(start, Clock::now());
  check(rknn_mem_sync(context_, c_, RKNN_MEMORY_SYNC_FROM_DEVICE),
        "rknn_mem_sync(C from device)");
  ++metrics_.runs;
}

void W4Linear::run(const llmc_float16 *input_data, llmc_float16 *output_data) {
  const size_t input_bytes =
      static_cast<size_t>(rows_) * input_columns_ * sizeof(llmc_float16);
  const size_t output_bytes =
      static_cast<size_t>(rows_) * output_columns_ * sizeof(llmc_float16);
  auto start = Clock::now();
  std::memcpy(input(), input_data, input_bytes);
  metrics_.input_copy_ms += elapsed_ms(start, Clock::now());
  run_bound();
  start = Clock::now();
  std::memcpy(output_data, output(), output_bytes);
  metrics_.output_copy_ms += elapsed_ms(start, Clock::now());
}

F16Matmul::F16Matmul(int rows, int inner, int columns, rknn_core_mask core_mask)
    : rows_(rows), inner_(inner), columns_(columns) {
  if (rows <= 0 || inner <= 0 || columns <= 0 || inner % 32 != 0 ||
      columns % 16 != 0) {
    throw std::runtime_error("invalid RK3588 FP16 MatMul shape");
  }
  rknn_matmul_info info{};
  info.M = rows;
  info.K = inner;
  info.N = columns;
  info.type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16;
  info.B_layout = RKNN_MM_LAYOUT_NORM;
  info.AC_layout = RKNN_MM_LAYOUT_NORM;
  try {
    check(rknn_matmul_create(&context_, &info, &io_),
          "rknn_matmul_create(FP16)");
    if (core_mask != RKNN_NPU_CORE_AUTO) {
      check(rknn_matmul_set_core_mask(context_, core_mask),
            "rknn_matmul_set_core_mask(FP16)");
    }
    a_ = rknn_create_mem(context_, io_.A.size);
    b_ = rknn_create_mem(context_, io_.B.size);
    c_ = rknn_create_mem(context_, io_.C.size);
    if (a_ == nullptr || b_ == nullptr || c_ == nullptr) {
      throw std::runtime_error("rknn_create_mem failed for FP16 MatMul");
    }
    check(rknn_matmul_set_io_mem(context_, a_, &io_.A),
          "rknn_matmul_set_io_mem(FP16 A)");
    check(rknn_matmul_set_io_mem(context_, b_, &io_.B),
          "rknn_matmul_set_io_mem(FP16 B)");
    check(rknn_matmul_set_io_mem(context_, c_, &io_.C),
          "rknn_matmul_set_io_mem(FP16 C)");
  } catch (...) {
    if (a_ != nullptr)
      rknn_destroy_mem(context_, a_);
    if (b_ != nullptr)
      rknn_destroy_mem(context_, b_);
    if (c_ != nullptr)
      rknn_destroy_mem(context_, c_);
    if (context_ != 0)
      rknn_matmul_destroy(context_);
    a_ = b_ = c_ = nullptr;
    context_ = 0;
    throw;
  }
}

F16Matmul::~F16Matmul() {
  if (a_ != nullptr)
    rknn_destroy_mem(context_, a_);
  if (b_ != nullptr)
    rknn_destroy_mem(context_, b_);
  if (c_ != nullptr)
    rknn_destroy_mem(context_, c_);
  if (context_ != 0)
    rknn_matmul_destroy(context_);
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
  const size_t b_bytes =
      static_cast<size_t>(inner_) * columns_ * sizeof(llmc_float16);
  const size_t c_bytes =
      static_cast<size_t>(rows_) * columns_ * sizeof(llmc_float16);
  auto start = Clock::now();
  std::memcpy(a_->virt_addr, a_data, a_bytes);
  std::memcpy(b_->virt_addr, b_data, b_bytes);
  check(rknn_mem_sync(context_, a_, RKNN_MEMORY_SYNC_TO_DEVICE),
        "rknn_mem_sync(FP16 A)");
  check(rknn_mem_sync(context_, b_, RKNN_MEMORY_SYNC_TO_DEVICE),
        "rknn_mem_sync(FP16 B)");
  metrics_.input_copy_ms += elapsed_ms(start, Clock::now());
  start = Clock::now();
  check(rknn_matmul_run(context_), "rknn_matmul_run(FP16)");
  metrics_.npu_run_ms += elapsed_ms(start, Clock::now());
  check(rknn_mem_sync(context_, c_, RKNN_MEMORY_SYNC_FROM_DEVICE),
        "rknn_mem_sync(FP16 C)");
  start = Clock::now();
  std::memcpy(c_data, c_->virt_addr, c_bytes);
  metrics_.output_copy_ms += elapsed_ms(start, Clock::now());
  ++metrics_.runs;
}

} // namespace llmc
