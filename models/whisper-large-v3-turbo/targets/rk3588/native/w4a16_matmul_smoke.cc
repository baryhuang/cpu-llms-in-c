#include "w4a16_matmul.h"
#include "w4a16_model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

std::vector<int> read_npu_load() {
  std::ifstream input("/sys/kernel/debug/rknpu/load");
  const std::string text((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  std::vector<int> result(3, -1);
  for (int core = 0; core < 3; ++core) {
    const std::string label = "Core" + std::to_string(core);
    size_t position = text.find(label);
    if (position == std::string::npos)
      continue;
    position = text.find(':', position + label.size());
    if (position == std::string::npos)
      continue;
    const char *start = text.c_str() + position + 1;
    char *end = nullptr;
    long load = std::strtol(start, &end, 10);
    if (end != start && load >= 0 && load <= 100)
      result[core] = load;
  }
  return result;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 4 || argc > 8) {
    std::fprintf(stderr,
                 "usage: %s MODEL.llmc TENSOR baseline|llmc "
                 "[ROWS] [LOOPS] [CORE_MASK] [--print-output]\n",
                 argv[0]);
    return 2;
  }
  try {
    const int rows = argc >= 5 ? std::stoi(argv[4]) : 1;
    const int loops = argc >= 6 ? std::stoi(argv[5]) : 5;
    const char *core_text = argc >= 7 ? argv[6] : "auto";
    const bool print_output =
        argc == 8 && std::strcmp(argv[7], "--print-output") == 0;
    if (rows <= 0 || loops <= 0 || (argc == 8 && !print_output)) {
      throw std::runtime_error("ROWS and LOOPS must be positive");
    }

    llmc::W4Model model(argv[1]);
    const auto mode = llmc::parse_run_mode(argv[3]);
    const auto &weight = model.require(argv[2]);
    llmc::W4Linear linear(weight, rows, mode, llmc::parse_core_mask(core_text));

    std::vector<llmc_float16> input(static_cast<size_t>(rows) *
                                    linear.input_columns());
    std::vector<llmc_float16> output(static_cast<size_t>(rows) *
                                     linear.output_columns());
    for (size_t index = 0; index < input.size(); ++index) {
      input[index] = static_cast<llmc_float16>(
          std::sin(static_cast<double>(index % 251) * 0.03125) * 0.25);
    }
    for (int loop = 0; loop < loops; ++loop) {
      linear.run(input.data(), output.data());
    }

    double checksum = 0.0;
    float maximum = 0.0f;
    for (size_t index = 0; index < output.size(); ++index) {
      const float value = static_cast<float>(output[index]);
      if (!std::isfinite(value)) {
        throw std::runtime_error("MatMul produced a non-finite output");
      }
      checksum += value * static_cast<double>((index % 127) + 1);
      maximum = std::max(maximum, std::abs(value));
      if (print_output && index < 16) {
        std::printf("output_%zu: %.8f\n", index, value);
      }
    }
    const auto &metrics = linear.metrics();
    std::printf("runtime: proprietary-rknpu2-matmul\n");
    std::printf("mode: %s\n", llmc::run_mode_name(mode));
    std::printf("tensor: %s\n", argv[2]);
    std::printf("shape_mkn: %d,%d,%d\n", rows, linear.input_columns(),
                linear.output_columns());
    std::printf("core_mask: %s\n", core_text);
    if (std::strcmp(core_text, "auto") == 0) {
      std::printf("npu_configured_cores: auto\n");
    } else {
      std::printf("npu_configured_cores: %d\n",
                  llmc::configured_core_count(core_text));
    }
    std::printf("runs: %llu\n", static_cast<unsigned long long>(metrics.runs));
    std::printf("weight_uploads: %llu\n",
                static_cast<unsigned long long>(metrics.weight_uploads));
    std::printf("weight_upload_ms: %.3f\n", metrics.weight_upload_ms);
    std::printf("input_copy_ms: %.3f\n", metrics.input_copy_ms);
    std::printf("npu_run_ms: %.3f\n", metrics.npu_run_ms);
    std::printf("avg_npu_run_ms: %.3f\n",
                metrics.runs == 0 ? 0.0 : metrics.npu_run_ms / metrics.runs);
    std::printf("output_copy_ms: %.3f\n", metrics.output_copy_ms);
    std::printf("output_max_abs: %.8f\n", maximum);
    std::printf("output_checksum: %.12f\n", checksum);
    const auto loads = read_npu_load();
    int observed_cores = 0;
    for (int core = 0; core < 3; ++core) {
      std::printf("npu_core%d_load_pct_after_run: %d\n", core, loads[core]);
      if (loads[core] > 0)
        ++observed_cores;
    }
    std::printf("npu_observed_active_cores_after_run: %d\n", observed_cores);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
  return 0;
}
