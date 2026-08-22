#include "w4a16_matmul.h"
#include "w4a16_model.h"

#include <algorithm>
#include <array>
#include <atomic>
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
                 "[ROWS] [LOOPS] [CORE_MASK|parallel3] [--print-output]\n",
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
    const auto core_masks =
        std::strcmp(core_text, "parallel3") == 0
            ? llmc::parallel_npu_core_masks()
            : std::vector<rknn_core_mask>{llmc::parse_core_mask(core_text)};
    llmc::W4Linear linear(weight, rows, mode, core_masks);

    std::vector<llmc_float16> input(static_cast<size_t>(rows) *
                                    linear.input_columns());
    std::vector<llmc_float16> output(static_cast<size_t>(rows) *
                                     linear.output_columns());
    for (size_t index = 0; index < input.size(); ++index) {
      input[index] = static_cast<llmc_float16>(
          std::sin(static_cast<double>(index % 251) * 0.03125) * 0.25);
    }
    std::atomic<bool> stop_monitor{false};
    std::array<int, 3> maximum_load = {-1, -1, -1};
    std::thread monitor([&] {
      while (!stop_monitor.load(std::memory_order_relaxed)) {
        const auto loads = read_npu_load();
        for (int core = 0; core < 3; ++core)
          maximum_load[core] = std::max(maximum_load[core], loads[core]);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    });
    try {
      for (int loop = 0; loop < loops; ++loop) {
        linear.run(input.data(), output.data());
      }
    } catch (...) {
      stop_monitor.store(true, std::memory_order_relaxed);
      monitor.join();
      throw;
    }
    stop_monitor.store(true, std::memory_order_relaxed);
    monitor.join();

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

    // Validate the complete first output row against the same dequantized
    // FP16 B=[K,N] matrix on the CPU. This catches nibble sign, scale order,
    // transpose and stale DMA-buffer errors before a Whisper benchmark runs.
    const size_t weight_elements =
        static_cast<size_t>(linear.input_columns()) * linear.output_columns();
    std::vector<llmc_float16> fp16_weight(weight_elements);
    llmc::dequantize_w4a16_to_fp16_kn(weight, fp16_weight.data(),
                                      fp16_weight.size());
    std::vector<float> cpu_reference(linear.output_columns(), 0.0f);
    for (int inner = 0; inner < linear.input_columns(); ++inner) {
      const float activation = static_cast<float>(input[inner]);
      const llmc_float16 *weight_row =
          fp16_weight.data() +
          static_cast<size_t>(inner) * linear.output_columns();
      for (int column = 0; column < linear.output_columns(); ++column) {
        cpu_reference[column] +=
            activation * static_cast<float>(weight_row[column]);
      }
    }
    double dot = 0.0;
    double cpu_norm = 0.0;
    double npu_norm = 0.0;
    double maximum_error = 0.0;
    for (int column = 0; column < linear.output_columns(); ++column) {
      const double expected = cpu_reference[column];
      const double actual = static_cast<float>(output[column]);
      dot += expected * actual;
      cpu_norm += expected * expected;
      npu_norm += actual * actual;
      maximum_error = std::max(maximum_error, std::abs(expected - actual));
    }
    const double cosine =
        dot / (std::sqrt(cpu_norm) * std::sqrt(npu_norm));
    if (!std::isfinite(cosine) || cosine < 0.9999) {
      throw std::runtime_error("NPU FP16 MatMul disagrees with CPU reference");
    }

    // Exercise two different inputs through one persistent FP16 context so
    // attention MatMuls cannot silently reuse stale DMA contents.
    const int dynamic_columns =
        16 * static_cast<int>(core_masks.size());
    llmc::F16Matmul dynamic_fp16(1, 32, dynamic_columns, core_masks);
    std::vector<llmc_float16> dynamic_a(32);
    std::vector<llmc_float16> dynamic_b(32 * dynamic_columns);
    std::vector<llmc_float16> dynamic_c(dynamic_columns);
    double dynamic_cosine = 0.0;
    for (int pass = 0; pass < 2; ++pass) {
      for (size_t index = 0; index < dynamic_a.size(); ++index) {
        dynamic_a[index] = static_cast<llmc_float16>(
            std::sin((static_cast<double>(index) + pass * 7.0) * 0.13));
      }
      for (size_t index = 0; index < dynamic_b.size(); ++index) {
        dynamic_b[index] = static_cast<llmc_float16>(
            std::cos((static_cast<double>(index) + pass * 11.0) * 0.017));
      }
      dynamic_fp16.run(dynamic_a.data(), dynamic_b.data(), dynamic_c.data());
      double dynamic_dot = 0.0;
      double dynamic_cpu_norm = 0.0;
      double dynamic_npu_norm = 0.0;
      for (int column = 0; column < dynamic_columns; ++column) {
        float expected = 0.0f;
        for (int inner = 0; inner < 32; ++inner) {
          expected += static_cast<float>(dynamic_a[inner]) *
                      static_cast<float>(
                          dynamic_b[inner * dynamic_columns + column]);
        }
        const double actual = static_cast<float>(dynamic_c[column]);
        dynamic_dot += expected * actual;
        dynamic_cpu_norm += expected * expected;
        dynamic_npu_norm += actual * actual;
      }
      dynamic_cosine = dynamic_dot /
                       (std::sqrt(dynamic_cpu_norm) *
                        std::sqrt(dynamic_npu_norm));
      if (!std::isfinite(dynamic_cosine) || dynamic_cosine < 0.9999) {
        throw std::runtime_error(
            "persistent NPU FP16 MatMul reused stale input data");
      }
    }
    const auto &metrics = linear.metrics();
    std::printf("runtime: llmc-cpp-w4-cpu-dequant-rknpu2-fp16-matmul\n");
    std::printf("mode: %s\n", llmc::run_mode_name(mode));
    std::printf("tensor: %s\n", argv[2]);
    std::printf("shape_mkn: %d,%d,%d\n", rows, linear.input_columns(),
                linear.output_columns());
    std::printf("core_mask: %s\n", core_text);
    if (std::strcmp(core_text, "parallel3") == 0) {
      std::printf("npu_scheduler: n-shard-parallel3\n");
      std::printf("npu_configured_cores: 3\n");
    } else if (std::strcmp(core_text, "auto") == 0) {
      std::printf("npu_configured_cores: auto\n");
    } else {
      std::printf("npu_configured_cores: %d\n",
                  llmc::configured_core_count(core_text));
    }
    std::printf("runs: %llu\n", static_cast<unsigned long long>(metrics.runs));
    std::printf("npu_shards_per_run: %zu\n", linear.shard_count());
    std::printf("npu_jobs: %llu\n",
                static_cast<unsigned long long>(metrics.npu_jobs));
    std::printf("weight_dequantizations: %llu\n",
                static_cast<unsigned long long>(
                    metrics.weight_dequantizations));
    std::printf("weight_dequant_ms: %.3f\n", metrics.weight_dequant_ms);
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
    std::printf("cpu_reference_cosine: %.9f\n", cosine);
    std::printf("cpu_reference_max_abs_error: %.9f\n", maximum_error);
    std::printf("dynamic_fp16_rebind_cosine: %.9f\n", dynamic_cosine);
    const auto loads = read_npu_load();
    int observed_cores = 0;
    for (int core = 0; core < 3; ++core) {
      maximum_load[core] = std::max(maximum_load[core], loads[core]);
      std::printf("npu_core%d_load_pct_max: %d\n", core,
                  maximum_load[core]);
      std::printf("npu_core%d_load_pct_after_run: %d\n", core, loads[core]);
      if (maximum_load[core] > 0)
        ++observed_cores;
    }
    std::printf("npu_observed_active_cores_after_run: %d\n", observed_cores);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
  return 0;
}
