#include "w4a16_model.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace llmc {
namespace {

constexpr std::array<char, 8> kMagic = {'L', 'L', 'M', 'C',
                                        'W', '4', 'A', '\0'};
constexpr uint32_t kVersion = 2;
constexpr uint32_t kTiledVersion = 3;
constexpr uint64_t kAlignment = 64;
constexpr uint64_t kTileColumns = 32;
constexpr uint64_t kTileInner = 32;
constexpr uint64_t kTileScaleBytes = kTileColumns * sizeof(float);
constexpr uint64_t kTilePackedBytes = kTileColumns * kTileInner / 2;
constexpr uint64_t kTileRecordBytes = kTileScaleBytes + kTilePackedBytes;
static_assert(kTileRecordBytes % kAlignment == 0);

#pragma pack(push, 1)
struct FileHeader {
  char magic[8];
  uint32_t version;
  uint32_t tensor_count;
  uint64_t payload_start;
};

struct FileEntry {
  uint32_t name_length;
  uint8_t encoding;
  uint8_t dimensions;
  uint16_t reserved0;
  uint32_t shape[4];
  uint32_t group_size;
  uint32_t reserved1;
  uint64_t data_offset;
  uint64_t data_size;
  uint64_t scale_offset;
  uint64_t scale_size;
};
#pragma pack(pop)

static_assert(sizeof(FileHeader) == 24);
static_assert(sizeof(FileEntry) == 64);

void require_range(uint64_t offset, uint64_t size, uint64_t file_size,
                   const char *description) {
  if (offset > file_size || size > file_size - offset) {
    throw std::runtime_error(std::string(description) +
                             " is outside model file");
  }
}

uint64_t element_count(const FileEntry &entry) {
  uint64_t elements = 1;
  for (uint8_t index = 0; index < entry.dimensions; ++index) {
    if (entry.shape[index] == 0 ||
        elements > std::numeric_limits<uint64_t>::max() / entry.shape[index]) {
      throw std::runtime_error("invalid tensor dimensions");
    }
    elements *= entry.shape[index];
  }
  return elements;
}

} // namespace

W4Model::W4Model(const char *path) {
  try {
    file_descriptor_ = open(path, O_RDONLY | O_CLOEXEC);
    if (file_descriptor_ < 0) {
      throw std::runtime_error(std::string("open failed: ") + path);
    }
    struct stat status{};
    if (fstat(file_descriptor_, &status) != 0 || status.st_size < 0) {
      throw std::runtime_error("fstat failed for W4A16 model");
    }
    file_size_ = static_cast<uint64_t>(status.st_size);
    require_range(0, sizeof(FileHeader), file_size_, "model header");
    mapping_ =
        mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, file_descriptor_, 0);
    if (mapping_ == MAP_FAILED) {
      mapping_ = nullptr;
      throw std::runtime_error("mmap failed for W4A16 model");
    }

    const auto *bytes = static_cast<const uint8_t *>(mapping_);
    const auto *header = reinterpret_cast<const FileHeader *>(bytes);
    if (!std::equal(kMagic.begin(), kMagic.end(), header->magic) ||
        (header->version != kVersion && header->version != kTiledVersion)) {
      throw std::runtime_error(
          "invalid W4A16 model header (expected format version 2 or 3)");
    }
    if (header->payload_start % kAlignment != 0 ||
        header->payload_start > file_size_) {
      throw std::runtime_error("invalid W4A16 payload offset");
    }
    if (header->tensor_count >
        (header->payload_start - sizeof(FileHeader)) / sizeof(FileEntry)) {
      throw std::runtime_error("invalid W4A16 tensor count");
    }

    uint64_t cursor = sizeof(FileHeader);
    tensors_.reserve(header->tensor_count);
    for (uint32_t index = 0; index < header->tensor_count; ++index) {
      require_range(cursor, sizeof(FileEntry), header->payload_start,
                    "tensor table entry");
      const auto *entry = reinterpret_cast<const FileEntry *>(bytes + cursor);
      cursor += sizeof(FileEntry);
      if (entry->name_length == 0 || entry->reserved0 != 0 ||
          entry->reserved1 != 0) {
        throw std::runtime_error("invalid W4A16 tensor table entry");
      }
      require_range(cursor, entry->name_length, header->payload_start,
                    "tensor name");
      std::string name(reinterpret_cast<const char *>(bytes + cursor),
                       entry->name_length);
      cursor += entry->name_length;

      if (entry->dimensions == 0 || entry->dimensions > 4) {
        throw std::runtime_error("invalid dimensions for tensor " + name);
      }
      if (entry->data_offset < header->payload_start ||
          entry->data_offset % kAlignment != 0) {
        throw std::runtime_error("invalid tensor data offset: " + name);
      }
      require_range(entry->data_offset, entry->data_size, file_size_,
                    "tensor data");
      const uint64_t elements = element_count(*entry);
      W4Tensor tensor;
      tensor.name = name;
      tensor.encoding = static_cast<W4Encoding>(entry->encoding);
      tensor.dimensions = entry->dimensions;
      std::copy(entry->shape, entry->shape + 4, tensor.shape.begin());
      tensor.group_size = entry->group_size;
      tensor.data = bytes + entry->data_offset;
      tensor.data_size = entry->data_size;

      if (tensor.encoding == W4Encoding::kFloat16) {
        if (entry->data_size != elements * 2 || entry->scale_size != 0) {
          throw std::runtime_error("invalid FP16 tensor payload: " + name);
        }
      } else if (tensor.encoding == W4Encoding::kW4A16Group) {
        if (entry->dimensions != 2 || entry->group_size != 32 ||
            entry->shape[1] % entry->group_size != 0 ||
            entry->data_size != (elements + 1) / 2) {
          throw std::runtime_error("invalid W4A16 tensor payload: " + name);
        }
        const uint64_t expected_scale_size =
            static_cast<uint64_t>(entry->shape[0]) *
            (entry->shape[1] / entry->group_size) * sizeof(float);
        if (entry->scale_size != expected_scale_size) {
          throw std::runtime_error("invalid W4A16 scales: " + name);
        }
        if (entry->scale_offset < header->payload_start ||
            entry->scale_offset % kAlignment != 0) {
          throw std::runtime_error("invalid tensor scale offset: " + name);
        }
        require_range(entry->scale_offset, entry->scale_size, file_size_,
                      "tensor scales");
        tensor.scales =
            reinterpret_cast<const float *>(bytes + entry->scale_offset);
        tensor.scale_size = entry->scale_size;
      } else if (tensor.encoding == W4Encoding::kW4A16GroupTiled) {
        if (header->version != kTiledVersion || entry->dimensions != 2 ||
            entry->group_size != kTileInner ||
            entry->shape[0] % kTileColumns != 0 ||
            entry->shape[1] % kTileInner != 0 || entry->scale_size != 0) {
          throw std::runtime_error("invalid tiled W4 tensor: " + name);
        }
        const uint64_t expected_data_size =
            (entry->shape[0] / kTileColumns) *
            (entry->shape[1] / kTileInner) * kTileRecordBytes;
        if (entry->data_size != expected_data_size) {
          throw std::runtime_error("invalid tiled W4 payload: " + name);
        }
      } else {
        throw std::runtime_error("unknown tensor encoding: " + name);
      }
      if (!tensors_.emplace(name, std::move(tensor)).second) {
        throw std::runtime_error("duplicate tensor name: " + name);
      }
    }
    if (cursor > header->payload_start) {
      throw std::runtime_error("W4A16 tensor table overlaps payload");
    }
  } catch (...) {
    if (mapping_ != nullptr) {
      munmap(mapping_, file_size_);
      mapping_ = nullptr;
    }
    if (file_descriptor_ >= 0) {
      close(file_descriptor_);
      file_descriptor_ = -1;
    }
    throw;
  }
}

W4Model::~W4Model() {
  if (mapping_ != nullptr)
    munmap(mapping_, file_size_);
  if (file_descriptor_ >= 0)
    close(file_descriptor_);
}

const W4Tensor &W4Model::require(std::string_view name) const {
  const auto found = tensors_.find(std::string(name));
  if (found == tensors_.end()) {
    throw std::runtime_error("missing tensor: " + std::string(name));
  }
  return found->second;
}

const W4Tensor &W4Model::require(const std::string &name) const {
  const auto found = tensors_.find(name);
  if (found == tensors_.end())
    throw std::runtime_error("missing tensor: " + name);
  return found->second;
}

const W4Tensor &W4Model::require(const char *name) const {
  return require(std::string_view(name));
}

bool W4Model::contains(std::string_view name) const {
  return tensors_.find(std::string(name)) != tensors_.end();
}

void dequantize_w4a16_to_fp16_kn(const W4Tensor &tensor,
                                 llmc_float16 *output,
                                 size_t output_elements) {
  dequantize_w4a16_columns_to_fp16_kn(
      tensor, 0, tensor.shape[0], output, output_elements);
}

void dequantize_w4a16_columns_to_fp16_kn(const W4Tensor &tensor,
                                         size_t column_offset,
                                         size_t column_count,
                                         llmc_float16 *output,
                                         size_t output_elements) {
  if (!is_w4_encoding(tensor.encoding) ||
      tensor.dimensions != 2 || tensor.group_size == 0 ||
      tensor.shape[1] % tensor.group_size != 0 || output == nullptr) {
    throw std::runtime_error("invalid tensor for W4A16 CPU dequantization");
  }
  const size_t columns = tensor.shape[0];
  const size_t inner = tensor.shape[1];
  if (column_count == 0 || column_offset > columns ||
      column_count > columns - column_offset) {
    throw std::runtime_error("invalid W4A16 dequantization column shard");
  }
  if (column_count != 0 &&
      inner > std::numeric_limits<size_t>::max() / column_count) {
    throw std::runtime_error("W4A16 dequantization size overflow");
  }
  const size_t source_elements = inner * columns;
  const size_t output_required = inner * column_count;
  if (output_elements < output_required) {
    throw std::runtime_error("undersized W4A16 dequantization buffer");
  }

  if (tensor.encoding == W4Encoding::kW4A16GroupTiled) {
    if (column_offset % kTileColumns != 0 ||
        column_count % kTileColumns != 0) {
      throw std::runtime_error("tiled W4 shard is not N32 aligned");
    }
    const size_t inner_tiles = inner / kTileInner;
    const size_t first_column_tile = column_offset / kTileColumns;
    const size_t column_tiles = column_count / kTileColumns;
    const size_t required_bytes =
        (columns / kTileColumns) * inner_tiles * kTileRecordBytes;
    if (tensor.data_size < required_bytes) {
      throw std::runtime_error("undersized tiled W4 payload");
    }
    const auto *payload = static_cast<const uint8_t *>(tensor.data);
    for (size_t local_column_tile = 0; local_column_tile < column_tiles;
         ++local_column_tile) {
      const size_t source_column_tile =
          first_column_tile + local_column_tile;
      for (size_t inner_tile = 0; inner_tile < inner_tiles; ++inner_tile) {
        const size_t record_index = source_column_tile * inner_tiles +
                                    inner_tile;
        const uint8_t *record = payload + record_index * kTileRecordBytes;
        const auto *scales = reinterpret_cast<const float *>(record);
        const uint8_t *packed = record + kTileScaleBytes;
        for (size_t inner_lane = 0; inner_lane < kTileInner; ++inner_lane) {
          llmc_float16 *row =
              output + (inner_tile * kTileInner + inner_lane) * column_count +
              local_column_tile * kTileColumns;
          for (size_t column_lane = 0; column_lane < kTileColumns;
               ++column_lane) {
            const size_t index = inner_lane * kTileColumns + column_lane;
            const uint8_t byte = packed[index / 2];
            const uint8_t nibble =
                index & 1 ? static_cast<uint8_t>(byte >> 4)
                          : static_cast<uint8_t>(byte & 0x0f);
            const int8_t quantized =
                nibble < 8
                    ? static_cast<int8_t>(nibble)
                    : static_cast<int8_t>(static_cast<int>(nibble) - 16);
            row[column_lane] = static_cast<llmc_float16>(
                static_cast<float>(quantized) * scales[column_lane]);
          }
        }
      }
    }
    return;
  }

  if (tensor.data_size < (source_elements + 1) / 2 ||
      tensor.scale_size <
          (inner / tensor.group_size) * columns * sizeof(float)) {
    throw std::runtime_error("undersized W4A16 v2 dequantization buffer");
  }

  const auto *packed = static_cast<const uint8_t *>(tensor.data);
  for (size_t k = 0; k < inner; ++k) {
    const float *group_scales =
        tensor.scales + (k / tensor.group_size) * columns;
    llmc_float16 *row = output + k * column_count;
    for (size_t n = 0; n < column_count; ++n) {
      const size_t source_column = column_offset + n;
      const size_t index = k * columns + source_column;
      const uint8_t byte = packed[index / 2];
      const uint8_t nibble =
          index & 1 ? static_cast<uint8_t>(byte >> 4)
                    : static_cast<uint8_t>(byte & 0x0f);
      const int8_t quantized =
          nibble < 8 ? static_cast<int8_t>(nibble)
                     : static_cast<int8_t>(static_cast<int>(nibble) - 16);
      row[n] = static_cast<llmc_float16>(
          static_cast<float>(quantized) * group_scales[source_column]);
    }
  }
}

} // namespace llmc
