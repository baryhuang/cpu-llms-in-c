#pragma once

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "float16_compat.h"
#include "w4a16_model.h"

namespace llmc {

struct OpenClW4Metrics {
  uint64_t calls = 0;
  double kernel_ms = 0.0;
  double transfer_ms = 0.0;
  double wall_ms = 0.0;
};

// One process-wide OpenCL context and one in-order, profiling-enabled command
// queue. The target runtime is native C++; the OpenCL C kernel is compiled once
// when this object is constructed and reused for every inference.
class OpenClW4Context {
public:
  OpenClW4Context();
  OpenClW4Context(const OpenClW4Context &) = delete;
  OpenClW4Context &operator=(const OpenClW4Context &) = delete;
  ~OpenClW4Context();

  cl_context context() const { return context_; }
  cl_command_queue queue() const { return queue_; }
  cl_device_id device() const { return device_; }
  cl_kernel split_kernel() const { return split_kernel_; }
  cl_kernel pair_kernel() const { return pair_kernel_; }
  cl_kernel quad_kernel() const { return quad_kernel_; }
  cl_kernel octet_kernel() const { return octet_kernel_; }
  cl_kernel batched_kernel() const { return batched_kernel_; }
  cl_kernel batched_octet4_kernel() const { return batched_octet4_kernel_; }
  const std::string &device_name() const { return device_name_; }
  cl_uint compute_units() const { return compute_units_; }

private:
  cl_device_id device_ = nullptr;
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_program program_ = nullptr;
  cl_kernel split_kernel_ = nullptr;
  cl_kernel pair_kernel_ = nullptr;
  cl_kernel quad_kernel_ = nullptr;
  cl_kernel octet_kernel_ = nullptr;
  cl_kernel batched_kernel_ = nullptr;
  cl_kernel batched_octet4_kernel_ = nullptr;
  std::string device_name_;
  cl_uint compute_units_ = 0;
};

// Fixed-shape, allocation-free W4A16 linear operator. Packed INT4 weights stay
// resident on the GPU in the package's N32/K32 layout. A and C buffers are
// allocated once for max_rows; run() only enqueues write, kernel and read.
class OpenClW4Linear {
public:
  OpenClW4Linear(OpenClW4Context &context, const W4Tensor &weight,
                 size_t max_rows);
  OpenClW4Linear(const OpenClW4Linear &) = delete;
  OpenClW4Linear &operator=(const OpenClW4Linear &) = delete;
  ~OpenClW4Linear();

  void run(const llmc_float16 *input, llmc_float16 *output, size_t rows);

  size_t input_columns() const { return input_columns_; }
  size_t output_columns() const { return output_columns_; }
  size_t max_rows() const { return max_rows_; }
  const OpenClW4Metrics &metrics() const { return metrics_; }

private:
  OpenClW4Context &context_;
  size_t input_columns_ = 0;
  size_t output_columns_ = 0;
  size_t max_rows_ = 0;
  cl_mem weight_ = nullptr;
  cl_mem input_ = nullptr;
  cl_mem output_ = nullptr;
  OpenClW4Metrics metrics_;
};

// Shape-shared A/C workspace used by the full Whisper runtime. Many resident
// packed weights can share one plan, avoiding per-weight activation buffers.
class OpenClW4LinearPlan {
public:
  class Weight {
  public:
    Weight(const Weight &) = delete;
    Weight &operator=(const Weight &) = delete;
    ~Weight();

    uint64_t bytes() const { return bytes_; }

  private:
    friend class OpenClW4LinearPlan;
    Weight() = default;
    cl_mem buffer_ = nullptr;
    uint64_t bytes_ = 0;
  };

  OpenClW4LinearPlan(OpenClW4Context &context, size_t rows,
                     size_t input_columns, size_t output_columns);
  OpenClW4LinearPlan(const OpenClW4LinearPlan &) = delete;
  OpenClW4LinearPlan &operator=(const OpenClW4LinearPlan &) = delete;
  ~OpenClW4LinearPlan();

  std::unique_ptr<Weight> prepare(const W4Tensor &weight);
  void run(const Weight &weight, const llmc_float16 *input,
           llmc_float16 *output);

  size_t rows() const { return rows_; }
  size_t input_columns() const { return input_columns_; }
  size_t output_columns() const { return output_columns_; }
  const OpenClW4Metrics &metrics() const { return metrics_; }

private:
  OpenClW4Context &context_;
  size_t rows_ = 0;
  size_t input_columns_ = 0;
  size_t output_columns_ = 0;
  cl_mem input_ = nullptr;
  cl_mem output_ = nullptr;
  OpenClW4Metrics metrics_;
};

} // namespace llmc
