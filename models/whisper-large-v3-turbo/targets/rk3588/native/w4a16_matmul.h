#pragma once

#include <cstdint>

#include "float16_compat.h"
#include "rknn_api.h"
#include "rknn_matmul_api.h"
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

struct W4MatmulMetrics {
  uint64_t runs = 0;
  uint64_t weight_uploads = 0;
  double weight_upload_ms = 0.0;
  double input_copy_ms = 0.0;
  double output_copy_ms = 0.0;
  double npu_run_ms = 0.0;
};

struct F16MatmulMetrics {
  uint64_t runs = 0;
  double input_copy_ms = 0.0;
  double npu_run_ms = 0.0;
  double output_copy_ms = 0.0;
};

// One persistent RKNPU2 FP16 x INT4 -> FP16 context.  PyTorch Linear weights
// are stored as [N, K] in W4Tensor and packed as RKNPU2's normal B=[K, N].
//
// RKNPU2 requires native A/C and B layouts for group-quantized INT4. Baseline
// converts and uploads the normal-layout packed weight before every
// invocation. LLMC converts B once, keeps it resident, and reuses all A/B/C
// allocations. Normal-layout CPU inputs/outputs are converted at the API
// boundary in both modes.
class W4Linear {
public:
  W4Linear(const W4Tensor &weight, int rows, W4RunMode mode,
           rknn_core_mask core_mask);
  W4Linear(const W4Linear &) = delete;
  W4Linear &operator=(const W4Linear &) = delete;
  ~W4Linear();

  int rows() const { return rows_; }
  int input_columns() const { return input_columns_; }
  int output_columns() const { return output_columns_; }
  llmc_float16 *input();
  const llmc_float16 *output() const;

  // Execute using already-bound native-layout input/output buffers. This is
  // the LLMC zero-copy primitive used when adjacent stages can share DMA.
  void run_bound();

  // Convenience interface for CPU stages. The copies are reported separately
  // so they cannot be hidden inside the NPU timing.
  void run(const llmc_float16 *input, llmc_float16 *output);

  const W4MatmulMetrics &metrics() const { return metrics_; }

private:
  void upload_weight();
  void check(int rc, const char *operation) const;

  const W4Tensor &weight_;
  int rows_ = 0;
  int input_columns_ = 0;
  int output_columns_ = 0;
  W4RunMode mode_ = W4RunMode::kBaseline;
  rknn_matmul_ctx context_ = 0;
  rknn_matmul_info info_{};
  rknn_matmul_io_attr io_{};
  rknn_tensor_mem *a_ = nullptr;
  rknn_tensor_mem *b_ = nullptr;
  rknn_tensor_mem *c_ = nullptr;
  W4MatmulMetrics metrics_{};
};

// Persistent FP16 MatMul used for QK^T and softmax(QK^T)V. Its dimensions are
// fixed, while both inputs may change on every invocation.
class F16Matmul {
public:
  F16Matmul(int rows, int inner, int columns, rknn_core_mask core_mask);
  F16Matmul(const F16Matmul &) = delete;
  F16Matmul &operator=(const F16Matmul &) = delete;
  ~F16Matmul();

  void run(const llmc_float16 *a, const llmc_float16 *b, llmc_float16 *c);
  const F16MatmulMetrics &metrics() const { return metrics_; }

private:
  void check(int rc, const char *operation) const;

  int rows_ = 0;
  int inner_ = 0;
  int columns_ = 0;
  rknn_matmul_ctx context_ = 0;
  rknn_matmul_io_attr io_{};
  rknn_tensor_mem *a_ = nullptr;
  rknn_tensor_mem *b_ = nullptr;
  rknn_tensor_mem *c_ = nullptr;
  F16MatmulMetrics metrics_{};
};

} // namespace llmc
