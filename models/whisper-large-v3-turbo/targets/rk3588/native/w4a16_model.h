#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

namespace llmc {

enum class W4Encoding : uint8_t {
  kFloat16 = 0,
  kW4A16Group = 1,
};

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

class W4Model {
public:
  explicit W4Model(const char *path);
  W4Model(const W4Model &) = delete;
  W4Model &operator=(const W4Model &) = delete;
  ~W4Model();

  const W4Tensor &require(std::string_view name) const;
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
