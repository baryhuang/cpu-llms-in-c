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
constexpr uint64_t kAlignment = 64;

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
        header->version != kVersion) {
      throw std::runtime_error(
          "invalid W4A16 model header (expected format version 2)");
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

bool W4Model::contains(std::string_view name) const {
  return tensors_.find(std::string(name)) != tensors_.end();
}

} // namespace llmc
