#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

#include "float16_compat.h"

namespace llmc {

enum class W4Encoding : uint8_t {
  kFloat16 = 0,
  kW4A16Group = 1,
  // CPU-ready N32 x K32 tiles. Each 64-byte-aligned record contains 32 FP32
  // scales followed by the corresponding 32 x 32 signed INT4 values.
  kW4A16GroupTiled = 2,
};

inline bool is_w4_encoding(W4Encoding encoding) {
  return encoding == W4Encoding::kW4A16Group ||
         encoding == W4Encoding::kW4A16GroupTiled;
}

struct W4Tensor {
  std::string name;
  W4Encoding encoding = W4Encoding::kFloat16;
  uint8_t dimensions = 0;
  std::array<uint32_t, 4> shape{};
  uint32_t group_size = 0;
  const void *data = nullptr;
  uint64_t data_size = 0;
  const float *scales = nullptr;
  uint64_t scale_size = 0;
};

// Expand the package's signed group-32 INT4 [N, K] weight into the normal
// FP16 B=[K, N] layout consumed by RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16.
// Quantization and scheduling belong to the LLMC runtime; RKNPU2 only sees
// the resulting FP16 matrix.
void dequantize_w4a16_to_fp16_kn(const W4Tensor &tensor,
                                 llmc_float16 *output,
                                 size_t output_elements);
void dequantize_w4a16_columns_to_fp16_kn(const W4Tensor &tensor,
                                         size_t column_offset,
                                         size_t column_count,
                                         llmc_float16 *output,
                                         size_t output_elements);

class W4Model {
public:
  explicit W4Model(const char *path);
  W4Model(const W4Model &) = delete;
  W4Model &operator=(const W4Model &) = delete;
  ~W4Model();

  const W4Tensor &require(std::string_view name) const;
  const W4Tensor &require(const std::string &name) const;
  const W4Tensor &require(const char *name) const;
  bool contains(std::string_view name) const;
  size_t tensor_count() const { return tensors_.size(); }
  uint64_t file_size() const { return file_size_; }
  const std::unordered_map<std::string, W4Tensor> &tensors() const {
    return tensors_;
  }

private:
  int file_descriptor_ = -1;
  void *mapping_ = nullptr;
  uint64_t file_size_ = 0;
  std::unordered_map<std::string, W4Tensor> tensors_;
};

} // namespace llmc
