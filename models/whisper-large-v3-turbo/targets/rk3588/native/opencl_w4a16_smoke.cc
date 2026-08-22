#include "opencl_w4a16.h"
#include "w4a16_model.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

float reference_output(const llmc::W4Tensor &tensor,
                       const llmc_float16 *input, size_t column) {
  constexpr size_t kColumns = 32;
  constexpr size_t kInner = 32;
  constexpr size_t kRecordBytes = 640;
  const size_t inner = tensor.shape[1];
  const size_t inner_tiles = inner / kInner;
  const size_t column_tile = column / kColumns;
  const size_t column_lane = column % kColumns;
  const auto *payload = static_cast<const uint8_t *>(tensor.data);
  float sum = 0.0f;
  for (size_t inner_tile = 0; inner_tile < inner_tiles; ++inner_tile) {
    const auto *record =
        payload + (column_tile * inner_tiles + inner_tile) * kRecordBytes;
    const auto *scales = reinterpret_cast<const float *>(record);
    const auto *packed = record + kColumns * sizeof(float);
    for (size_t inner_lane = 0; inner_lane < kInner; ++inner_lane) {
      const size_t packed_index = inner_lane * kColumns + column_lane;
      const uint8_t byte = packed[packed_index / 2];
      const uint8_t nibble =
          packed_index & 1 ? static_cast<uint8_t>(byte >> 4)
                           : static_cast<uint8_t>(byte & 15);
      const int quantized = nibble < 8 ? nibble : static_cast<int>(nibble) - 16;
      const size_t k = inner_tile * kInner + inner_lane;
      sum += static_cast<float>(input[k]) *
             (static_cast<float>(quantized) * scales[column_lane]);
    }
  }
  return sum;
}

long long read_integer(const char *path) {
  std::ifstream stream(path);
  long long value = -1;
  stream >> value;
  return value;
}

struct GpuSample {
  long long load = -1;
  long long frequency = -1;
};

GpuSample sample_gpu() {
  return {
      read_integer("/sys/devices/platform/fb000000.gpu/devfreq/fb000000.gpu/load"),
      read_integer("/sys/devices/platform/fb000000.gpu/devfreq/fb000000.gpu/cur_freq")};
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2 || argc > 5) {
    std::fprintf(stderr,
                 "usage: %s DECODER.llmc [TENSOR] [LOOPS] [ROWS]\n",
                 argv[0]);
    return 2;
  }
  try {
    const char *tensor_name =
        argc >= 3 ? argv[2] : "model.proj_out.weight";
    const int loops = argc >= 4 ? std::atoi(argv[3]) : 20;
    const size_t rows = argc >= 5 ? std::strtoul(argv[4], nullptr, 10) : 1;
    if (loops <= 0 || rows == 0)
      throw std::runtime_error("loops and rows must be positive");

    llmc::W4Model model(argv[1]);
    const llmc::W4Tensor &weight = model.require(tensor_name);
    std::vector<llmc_float16> input(rows * weight.shape[1]);
    std::vector<llmc_float16> output(rows * weight.shape[0]);
    for (size_t index = 0; index < input.size(); ++index) {
      input[index] = static_cast<llmc_float16>(
          0.125f * std::sin(static_cast<float>(index % weight.shape[1]) *
                            0.017f) +
          0.0625f * std::cos(static_cast<float>(index) * 0.011f));
    }

    llmc::OpenClW4Context context;
    llmc::OpenClW4Linear linear(context, weight, rows);
    linear.run(input.data(), output.data(), rows);

    const size_t samples[] = {0, 1, 31, 32, 127,
                              static_cast<size_t>(weight.shape[0] - 1)};
    float maximum_error = 0.0f;
    float maximum_relative_error = 0.0f;
    for (size_t column : samples) {
      const float expected = reference_output(weight, input.data(), column);
      const float actual = static_cast<float>(output[column]);
      const float error = std::abs(actual - expected);
      maximum_error = std::max(maximum_error, error);
      maximum_relative_error =
          std::max(maximum_relative_error,
                   error / std::max(1.0e-6f, std::abs(expected)));
      std::printf("sample[%zu]: gpu=%.8f cpu=%.8f abs_error=%.8f\n",
                  column, actual, expected, error);
    }
    if (!std::isfinite(maximum_error) || maximum_relative_error > 0.02f)
      throw std::runtime_error("OpenCL W4A16 correctness check failed");

    std::vector<double> kernel_samples;
    std::vector<double> wall_samples;
    kernel_samples.reserve(loops);
    wall_samples.reserve(loops);
    std::atomic<long long> max_load{-1};
    std::atomic<long long> max_frequency{-1};
    std::atomic<bool> stop_sampler{false};
    std::thread sampler([&]() {
      while (!stop_sampler.load(std::memory_order_relaxed)) {
        const GpuSample sample = sample_gpu();
        long long previous = max_load.load(std::memory_order_relaxed);
        while (sample.load > previous &&
               !max_load.compare_exchange_weak(previous, sample.load,
                                               std::memory_order_relaxed)) {
        }
        previous = max_frequency.load(std::memory_order_relaxed);
        while (sample.frequency > previous &&
               !max_frequency.compare_exchange_weak(
                   previous, sample.frequency, std::memory_order_relaxed)) {
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
    for (int loop = 0; loop < loops; ++loop) {
      const llmc::OpenClW4Metrics before = linear.metrics();
      linear.run(input.data(), output.data(), rows);
      const llmc::OpenClW4Metrics after = linear.metrics();
      kernel_samples.push_back(after.kernel_ms - before.kernel_ms);
      wall_samples.push_back(after.wall_ms - before.wall_ms);
    }
    stop_sampler.store(true, std::memory_order_relaxed);
    sampler.join();

    const auto summarize = [](std::vector<double> values) {
      std::sort(values.begin(), values.end());
      const double total = std::accumulate(values.begin(), values.end(), 0.0);
      return std::array<double, 3>{total / values.size(),
                                   values[values.size() / 2], values.back()};
    };
    const auto kernel = summarize(kernel_samples);
    const auto wall = summarize(wall_samples);
    const llmc::OpenClW4Metrics metrics = linear.metrics();

    std::printf("runtime: llmc-native-cpp-opencl-w4a16\n");
    std::printf("device: %s\n", context.device_name().c_str());
    std::printf("gpu_compute_units: %u\n", context.compute_units());
    std::printf("tensor: %s\n", tensor_name);
    std::printf("shape: M=%zu K=%u N=%u\n", rows, weight.shape[1],
                weight.shape[0]);
    std::printf("weight_storage: packed-int4-n32-k32 (%llu bytes)\n",
                static_cast<unsigned long long>(weight.data_size));
    std::printf("loops: %d\n", loops);
    std::printf("kernel_ms: mean=%.6f median=%.6f max=%.6f\n", kernel[0],
                kernel[1], kernel[2]);
    std::printf("wall_ms: mean=%.6f median=%.6f max=%.6f\n", wall[0],
                wall[1], wall[2]);
    std::printf("transfer_ms_total: %.6f\n", metrics.transfer_ms);
    std::printf("gpu_load_max_raw: %lld\n",
                max_load.load(std::memory_order_relaxed));
    std::printf("gpu_frequency_max_hz: %lld\n",
                max_frequency.load(std::memory_order_relaxed));
    std::printf("max_abs_error: %.8f\n", maximum_error);
    std::printf("max_relative_error: %.8f\n", maximum_relative_error);
    std::printf("correctness: passed\n");
  } catch (const std::exception &error) {
    std::fprintf(stderr, "opencl_w4a16_smoke: failed: %s\n", error.what());
    return 1;
  }
  return 0;
}
