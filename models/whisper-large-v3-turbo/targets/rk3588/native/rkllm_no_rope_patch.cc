#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kPositionCopyCallOffset = 0x1c96a0;
constexpr std::array<std::uint8_t, 4> kExpectedCall = {
    0x6d, 0x4b, 0x04, 0x94};
constexpr std::array<std::uint8_t, 4> kAarch64Nop = {
    0x1f, 0x20, 0x03, 0xd5};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s INPUT_librkllmrt.so OUTPUT_librkllmrt.so\n",
                 argv[0]);
    return 2;
  }
  if (std::string(argv[1]) == argv[2]) {
    std::fprintf(stderr, "input and output must be different files\n");
    return 2;
  }
  try {
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) throw std::runtime_error("cannot open input runtime");
    std::vector<std::uint8_t> image(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (image.size() < kPositionCopyCallOffset + kExpectedCall.size()) {
      throw std::runtime_error("runtime is smaller than the pinned patch offset");
    }
    for (std::size_t index = 0; index < kExpectedCall.size(); ++index) {
      if (image[kPositionCopyCallOffset + index] != kExpectedCall[index]) {
        throw std::runtime_error(
            "runtime bytes do not match RKLLM 1.3.0; refusing to patch");
      }
      image[kPositionCopyCallOffset + index] = kAarch64Nop[index];
    }
    std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open output runtime");
    output.write(reinterpret_cast<const char*>(image.data()), image.size());
    if (!output) throw std::runtime_error("failed to write patched runtime");
    std::printf("patched unused no-RoPE position copy at file offset 0x%zx\n",
                kPositionCopyCallOffset);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
  return 0;
}
