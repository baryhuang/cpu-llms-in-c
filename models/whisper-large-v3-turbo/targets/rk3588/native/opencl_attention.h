#pragma once

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>

#include <cstddef>
#include <cstdint>

#include "float16_compat.h"
#include "opencl_w4a16.h"

namespace llmc {

struct OpenClAttentionMetrics {
  uint64_t blocks = 0;
  double qk_ms = 0.0;
  double softmax_ms = 0.0;
  double pv_ms = 0.0;
  double transfer_ms = 0.0;
  double wall_ms = 0.0;
};

// Fixed Whisper-large-v3-turbo multi-head attention plan. Q/K/V and output
// buffers cover the maximum 1500-token encoder shape; the 20-head score arena
// remains resident on the GPU and is reused by every block.
class OpenClAttention {
public:
  explicit OpenClAttention(OpenClW4Context &context);
  OpenClAttention(const OpenClAttention &) = delete;
  OpenClAttention &operator=(const OpenClAttention &) = delete;
  ~OpenClAttention();

  // Row-major Q is [query_rows, 1280]. For normal encoder/self attention,
  // K/V are row-major [key_rows, 1280]. For cross attention, packed K is
  // [20,64,key_stride] and packed V is [20,key_stride,64].
  void run(const llmc_float16 *query, const llmc_float16 *key,
           const llmc_float16 *value, llmc_float16 *output,
           size_t query_rows, size_t key_rows, size_t key_stride,
           bool packed_key_value);

  const OpenClAttentionMetrics &metrics() const { return metrics_; }

private:
  OpenClW4Context &context_;
  cl_program program_ = nullptr;
  cl_kernel qk_kernel_ = nullptr;
  cl_kernel softmax_kernel_ = nullptr;
  cl_kernel pv_kernel_ = nullptr;
  cl_mem query_ = nullptr;
  cl_mem key_ = nullptr;
  cl_mem value_ = nullptr;
  cl_mem scores_ = nullptr;
  cl_mem output_ = nullptr;
  OpenClAttentionMetrics metrics_;
};

} // namespace llmc
