#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fftw3.h>

#include "pcm16_wav.h"
#include "w4a16_matmul.h"
#include "w4a16_model.h"

namespace {

constexpr int kSampleRate = 16000;
constexpr int kFftSize = 400;
constexpr int kHopLength = 160;
constexpr int kAudioSamples = 30 * kSampleRate;
constexpr int kFrames = 3000;
constexpr int kBins = kFftSize / 2 + 1;
constexpr int kMels = 128;
constexpr int kModel = 1280;
constexpr int kFfn = 5120;
constexpr int kHeads = 20;
constexpr int kHead = 64;
constexpr int kEncoderFrames = 1500;
constexpr int kEncoderPaddedFrames = 1504;
constexpr int kEncoderLayers = 32;
constexpr int kDecoderLayers = 4;
constexpr int kDecoderPositions = 448;
constexpr int kVocab = 51866;
constexpr int kEos = 50257;
constexpr int kSot = 50258;
constexpr int kEnglish = 50259;
constexpr int kTranscribe = 50360;
constexpr int kNoTimestamps = 50364;
constexpr int kMaxNewTokens = 128;
constexpr float kLayerNormEpsilon = 1.0e-5f;
constexpr float kAttentionScale = 0.125f; // 1 / sqrt(64)

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

const llmc_float16 *fp16(const llmc::W4Tensor &tensor) {
  if (tensor.encoding != llmc::W4Encoding::kFloat16) {
    throw std::runtime_error("expected FP16 tensor: " + tensor.name);
  }
  return static_cast<const llmc_float16 *>(tensor.data);
}

void add_bias(llmc_float16 *values, int rows, int columns,
              const llmc_float16 *bias) {
  if (bias == nullptr)
    return;
  for (int row = 0; row < rows; ++row) {
    llmc_float16 *destination = values + static_cast<size_t>(row) * columns;
    for (int column = 0; column < columns; ++column) {
      destination[column] =
          static_cast<llmc_float16>(static_cast<float>(destination[column]) +
                                    static_cast<float>(bias[column]));
    }
  }
}

void add_in_place(llmc_float16 *destination, const llmc_float16 *source,
                  size_t count) {
  for (size_t index = 0; index < count; ++index) {
    destination[index] =
        static_cast<llmc_float16>(static_cast<float>(destination[index]) +
                                  static_cast<float>(source[index]));
  }
}

void gelu_in_place(llmc_float16 *values, size_t count) {
  constexpr float kInverseSqrtTwo = 0.7071067811865475244f;
  for (size_t index = 0; index < count; ++index) {
    const float value = static_cast<float>(values[index]);
    values[index] = static_cast<llmc_float16>(
        0.5f * value * (1.0f + std::erf(value * kInverseSqrtTwo)));
  }
}

void layer_norm(const llmc_float16 *input, llmc_float16 *output, int rows,
                int columns, const llmc_float16 *weight,
                const llmc_float16 *bias) {
  for (int row = 0; row < rows; ++row) {
    const llmc_float16 *source = input + static_cast<size_t>(row) * columns;
    llmc_float16 *destination = output + static_cast<size_t>(row) * columns;
    double sum = 0.0;
    double squared_sum = 0.0;
    for (int column = 0; column < columns; ++column) {
      const float value = static_cast<float>(source[column]);
      sum += value;
      squared_sum += static_cast<double>(value) * value;
    }
    const float mean = static_cast<float>(sum / columns);
    const float variance =
        std::max(0.0f, static_cast<float>(squared_sum / columns) - mean * mean);
    const float inverse = 1.0f / std::sqrt(variance + kLayerNormEpsilon);
    for (int column = 0; column < columns; ++column) {
      const float normalized =
          (static_cast<float>(source[column]) - mean) * inverse;
      destination[column] = static_cast<llmc_float16>(
          normalized * static_cast<float>(weight[column]) +
          static_cast<float>(bias[column]));
    }
  }
}

class NpuLoadMonitor {
public:
  enum class Stage : int { kIdle, kEncoder, kDecoder };

  ~NpuLoadMonitor() { stop(); }

  void start() {
    if (running_.exchange(true))
      return;
    worker_ = std::thread([this] { loop(); });
  }

  void stage(Stage stage) {
    stage_.store(static_cast<int>(stage), std::memory_order_release);
  }

  void stop() {
    if (!running_.exchange(false))
      return;
    if (worker_.joinable())
      worker_.join();
  }

  void print() const {
    std::printf("npu_load_source: /sys/kernel/debug/rknpu/load\n");
    std::printf("npu_load_sample_interval_ms: 10\n");
    if (!available_) {
      std::printf("npu_load_status: unavailable\n");
      return;
    }
    std::printf("npu_load_status: sampled\n");
    print_stage("encoder", stats_[0]);
    print_stage("decoder", stats_[1]);
    int active = 0;
    for (int core = 0; core < 3; ++core) {
      if (stats_[0].maximum[core] > 0 || stats_[1].maximum[core] > 0)
        ++active;
    }
    std::printf("npu_observed_active_cores: %d\n", active);
  }

private:
  struct Stats {
    uint64_t samples = 0;
    std::array<uint64_t, 3> sum{};
    std::array<int, 3> maximum{};
  };

  static void print_stage(const char *name, const Stats &stats) {
    std::printf("npu_%s_load_samples: %llu\n", name,
                static_cast<unsigned long long>(stats.samples));
    for (int core = 0; core < 3; ++core) {
      const double average =
          stats.samples == 0
              ? 0.0
              : static_cast<double>(stats.sum[core]) / stats.samples;
      std::printf("npu_%s_core%d_avg_load_pct: %.2f\n", name, core, average);
      std::printf("npu_%s_core%d_max_load_pct: %d\n", name, core,
                  stats.maximum[core]);
    }
  }

  static bool parse(const std::string &text, std::array<int, 3> &loads) {
    bool found = false;
    for (int core = 0; core < 3; ++core) {
      const std::string label = "Core" + std::to_string(core);
      size_t position = text.find(label);
      if (position == std::string::npos)
        continue;
      position = text.find(':', position + label.size());
      if (position == std::string::npos)
        continue;
      const char *begin = text.c_str() + position + 1;
      char *end = nullptr;
      const long value = std::strtol(begin, &end, 10);
      if (end != begin && value >= 0 && value <= 100) {
        loads[core] = static_cast<int>(value);
        found = true;
      }
    }
    return found;
  }

  void loop() {
    while (running_.load(std::memory_order_acquire)) {
      std::ifstream input("/sys/kernel/debug/rknpu/load");
      const std::string text((std::istreambuf_iterator<char>(input)),
                             std::istreambuf_iterator<char>());
      std::array<int, 3> loads{};
      if (parse(text, loads)) {
        available_ = true;
        const int stage = stage_.load(std::memory_order_acquire);
        if (stage == static_cast<int>(Stage::kEncoder) ||
            stage == static_cast<int>(Stage::kDecoder)) {
          Stats &stats =
              stats_[stage == static_cast<int>(Stage::kEncoder) ? 0 : 1];
          ++stats.samples;
          for (int core = 0; core < 3; ++core) {
            stats.sum[core] += loads[core];
            stats.maximum[core] = std::max(stats.maximum[core], loads[core]);
          }
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  std::atomic<bool> running_{false};
  std::atomic<int> stage_{static_cast<int>(Stage::kIdle)};
  std::thread worker_;
  bool available_ = false;
  std::array<Stats, 2> stats_{};
};

double hz_to_mel(double hz) {
  constexpr double kSpacing = 200.0 / 3.0;
  constexpr double kMinLogHz = 1000.0;
  constexpr double kMinLogMel = kMinLogHz / kSpacing;
  constexpr double kLogStep = 0.06875177742094912;
  return hz >= kMinLogHz ? kMinLogMel + std::log(hz / kMinLogHz) / kLogStep
                         : hz / kSpacing;
}

double mel_to_hz(double mel) {
  constexpr double kSpacing = 200.0 / 3.0;
  constexpr double kMinLogHz = 1000.0;
  constexpr double kMinLogMel = kMinLogHz / kSpacing;
  constexpr double kLogStep = 0.06875177742094912;
  return mel >= kMinLogMel ? kMinLogHz * std::exp(kLogStep * (mel - kMinLogMel))
                           : kSpacing * mel;
}

std::vector<float> mel_filters() {
  std::array<double, kMels + 2> frequencies{};
  const double minimum = hz_to_mel(0.0);
  const double maximum = hz_to_mel(kSampleRate / 2.0);
  for (int index = 0; index < kMels + 2; ++index) {
    frequencies[index] =
        mel_to_hz(minimum + (maximum - minimum) * index / (kMels + 1));
  }
  std::vector<float> result(kMels * kBins, 0.0f);
  for (int mel = 0; mel < kMels; ++mel) {
    const double lower = frequencies[mel];
    const double center = frequencies[mel + 1];
    const double upper = frequencies[mel + 2];
    const double normalization = 2.0 / (upper - lower);
    for (int bin = 0; bin < kBins; ++bin) {
      const double frequency = (kSampleRate / 2.0) * bin / (kBins - 1);
      const double up = (frequency - lower) / (center - lower);
      const double down = (upper - frequency) / (upper - center);
      result[mel * kBins + bin] =
          static_cast<float>(normalization * std::max(0.0, std::min(up, down)));
    }
  }
  return result;
}

std::vector<llmc_float16> extract_features(const llmc::PcmAudio &audio) {
  std::vector<float> samples(kAudioSamples, 0.0f);
  std::copy(
      audio.samples.begin(),
      audio.samples.begin() +
          std::min(audio.samples.size(), static_cast<size_t>(kAudioSamples)),
      samples.begin());
  constexpr int kPad = kFftSize / 2;
  std::vector<float> padded(kAudioSamples + 2 * kPad, 0.0f);
  std::copy(samples.begin(), samples.end(), padded.begin() + kPad);
  for (int index = 0; index < kPad; ++index) {
    padded[index] = samples[kPad - index];
    padded[kPad + kAudioSamples + index] = samples[kAudioSamples - 2 - index];
  }
  std::vector<float> window(kFftSize);
  for (int index = 0; index < kFftSize; ++index) {
    window[index] =
        static_cast<float>(0.5 - 0.5 * std::cos(2.0 * M_PI * index / kFftSize));
  }
  std::vector<float> fft_input(kFftSize);
  fftwf_complex *fft_output =
      static_cast<fftwf_complex *>(fftwf_malloc(sizeof(fftwf_complex) * kBins));
  if (fft_output == nullptr)
    throw std::bad_alloc();
  fftwf_plan plan = fftwf_plan_dft_r2c_1d(kFftSize, fft_input.data(),
                                          fft_output, FFTW_ESTIMATE);
  if (plan == nullptr) {
    fftwf_free(fft_output);
    throw std::runtime_error("fftwf_plan_dft_r2c_1d failed");
  }
  const auto filters = mel_filters();
  std::vector<float> features(kMels * kFrames, 0.0f);
  std::array<float, kBins> power{};
  for (int frame = 0; frame < kFrames; ++frame) {
    const int offset = frame * kHopLength;
    for (int index = 0; index < kFftSize; ++index) {
      fft_input[index] = padded[offset + index] * window[index];
    }
    fftwf_execute(plan);
    for (int bin = 0; bin < kBins; ++bin) {
      power[bin] = fft_output[bin][0] * fft_output[bin][0] +
                   fft_output[bin][1] * fft_output[bin][1];
    }
    for (int mel = 0; mel < kMels; ++mel) {
      float sum = 0.0f;
      for (int bin = 0; bin < kBins; ++bin) {
        sum += filters[mel * kBins + bin] * power[bin];
      }
      features[mel * kFrames + frame] = sum;
    }
  }
  fftwf_destroy_plan(plan);
  fftwf_free(fft_output);
  float maximum = -std::numeric_limits<float>::infinity();
  for (float &value : features) {
    value = std::log10(std::max(value, 1.0e-10f));
    maximum = std::max(maximum, value);
  }
  std::vector<llmc_float16> result(features.size());
  for (size_t index = 0; index < features.size(); ++index) {
    result[index] = static_cast<llmc_float16>(
        (std::max(features[index], maximum - 8.0f) + 4.0f) / 4.0f);
  }
  return result;
}

class LinearExecutor {
public:
  LinearExecutor(const llmc::W4Model &encoder, const llmc::W4Model &decoder,
                 llmc::W4RunMode mode,
                 std::vector<rknn_core_mask> core_masks)
      : encoder_(encoder), decoder_(decoder), mode_(mode),
        core_masks_(std::move(core_masks)) {}

  void encoder(const std::string &weight_name, const llmc_float16 *input,
               llmc_float16 *output, int rows,
               const std::string &bias_name = std::string()) {
    run(encoder_, nullptr, weight_name, input, output, rows, bias_name);
  }

  void decoder(const std::string &weight_name, const llmc_float16 *input,
               llmc_float16 *output,
               const std::string &bias_name = std::string()) {
    run(decoder_, nullptr, weight_name, input, output, 1, bias_name);
  }

  uint64_t runs() const { return runs_; }
  uint64_t baseline_weight_uploads() const { return baseline_weight_uploads_; }
  uint64_t aot_weight_bytes() const { return aot_weight_bytes_; }
  size_t aot_plan_count() const { return aot_plans_.size(); }
  size_t aot_weight_count() const { return aot_weights_.size(); }
  double wall_ms() const { return wall_ms_; }

  void prepare() {
    if (mode_ != llmc::W4RunMode::kLlmc)
      return;
    prepare_model(encoder_, true);
    prepare_model(decoder_, false);
  }

private:
  struct PlanShape {
    int rows = 0;
    int input_columns = 0;
    int output_columns = 0;

    bool operator==(const PlanShape &other) const {
      return rows == other.rows && input_columns == other.input_columns &&
             output_columns == other.output_columns;
    }
  };

  struct PlanShapeHash {
    size_t operator()(const PlanShape &shape) const {
      size_t result = static_cast<size_t>(shape.rows);
      result = result * 1315423911u + static_cast<size_t>(shape.input_columns);
      result = result * 1315423911u + static_cast<size_t>(shape.output_columns);
      return result;
    }
  };

  struct PreparedLinear {
    llmc::AotW4LinearPlan *plan = nullptr;
    std::unique_ptr<llmc::AotW4LinearPlan::Weight> weight;
  };

  void prepare_model(const llmc::W4Model &model, bool encoder_scope) {
    for (const auto &pair : model.tensors()) {
      const auto &tensor = pair.second;
      if (!llmc::is_w4_encoding(tensor.encoding) ||
          aot_weights_.find(pair.first) != aot_weights_.end()) {
        continue;
      }
      int rows = 1;
      if (encoder_scope) {
        rows = pair.first == "model.encoder.conv1.weight" ? kFrames
                                                           : kEncoderFrames;
      }
      const PlanShape shape{rows, static_cast<int>(tensor.shape[1]),
                            static_cast<int>(tensor.shape[0])};
      auto plan = aot_plans_.find(shape);
      if (plan == aot_plans_.end()) {
        plan = aot_plans_
                   .emplace(shape,
                            std::make_unique<llmc::AotW4LinearPlan>(
                                shape.rows, shape.input_columns,
                                shape.output_columns, core_masks_))
                   .first;
      }
      auto weight = plan->second->prepare(tensor);
      aot_weight_bytes_ += weight->bytes();
      aot_weights_.emplace(
          pair.first, PreparedLinear{plan->second.get(), std::move(weight)});
    }
  }

  void run(const llmc::W4Model &model, void *,
           const std::string &weight_name, const llmc_float16 *input,
           llmc_float16 *output, int rows, const std::string &bias_name) {
    const auto begin = Clock::now();
    int output_columns = 0;
    if (mode_ == llmc::W4RunMode::kLlmc) {
      const auto found = aot_weights_.find(weight_name);
      if (found == aot_weights_.end())
        throw std::runtime_error("missing AOT W4 linear: " + weight_name);
      if (found->second.plan->rows() != rows)
        throw std::runtime_error("AOT W4 row mismatch: " + weight_name);
      found->second.plan->run(*found->second.weight, input, output);
      output_columns = found->second.plan->output_columns();
    } else {
      llmc::W4Linear linear(model.require(weight_name), rows, mode_,
                            core_masks_);
      linear.run(input, output);
      output_columns = linear.output_columns();
      ++baseline_weight_uploads_;
    }
    if (!bias_name.empty()) {
      add_bias(output, rows, output_columns,
               fp16(model.require(bias_name)));
    }
    ++runs_;
    wall_ms_ += elapsed_ms(begin, Clock::now());
  }

  const llmc::W4Model &encoder_;
  const llmc::W4Model &decoder_;
  llmc::W4RunMode mode_;
  std::vector<rknn_core_mask> core_masks_;
  // Member destruction is reverse declaration order: weights release their
  // B buffers through the owning plans before the plans destroy A/C contexts.
  std::unordered_map<PlanShape, std::unique_ptr<llmc::AotW4LinearPlan>,
                     PlanShapeHash>
      aot_plans_;
  std::unordered_map<std::string, PreparedLinear> aot_weights_;
  uint64_t aot_weight_bytes_ = 0;
  uint64_t runs_ = 0;
  uint64_t baseline_weight_uploads_ = 0;
  double wall_ms_ = 0.0;
};

class AttentionExecutor {
public:
  explicit AttentionExecutor(std::vector<rknn_core_mask> core_masks)
      : scheduler_(llmc::static_npu_scheduler()),
        encoder_qk_(kEncoderFrames, kHead, kEncoderPaddedFrames, core_masks),
        encoder_pv_(kEncoderFrames, kEncoderPaddedFrames, kHead, core_masks),
        decoder_self_qk_(1, kHead, kDecoderPositions, core_masks),
        decoder_self_pv_(1, kDecoderPositions, kHead, core_masks),
        decoder_cross_qk_(1, kHead, kEncoderPaddedFrames, core_masks),
        decoder_cross_pv_(1, kEncoderPaddedFrames, kHead, core_masks) {}

  void encoder(const llmc_float16 *query, const llmc_float16 *key,
               const llmc_float16 *value, llmc_float16 *output) {
    run(query, key, value, output, kEncoderFrames, kEncoderFrames,
        kEncoderPaddedFrames, false, encoder_qk_, encoder_pv_);
  }

  void decoder_self(const llmc_float16 *query, const llmc_float16 *key,
                    const llmc_float16 *value, int valid_positions,
                    llmc_float16 *output) {
    run(query, key, value, output, 1, valid_positions, kDecoderPositions,
        false, decoder_self_qk_, decoder_self_pv_);
  }

  void decoder_cross(const llmc_float16 *query, const llmc_float16 *key,
                     const llmc_float16 *value, llmc_float16 *output) {
    run(query, key, value, output, 1, kEncoderFrames, kEncoderPaddedFrames,
        true, decoder_cross_qk_, decoder_cross_pv_);
  }

  uint64_t runs() const { return runs_; }
  uint64_t dispatches() const { return dispatches_; }
  double wall_ms() const { return wall_ms_; }

private:
  struct Scratch {
    Scratch()
        : q(static_cast<size_t>(kEncoderFrames) * kHead),
          kt(static_cast<size_t>(kHead) * kEncoderPaddedFrames),
          scores(static_cast<size_t>(kEncoderFrames) * kEncoderPaddedFrames),
          probabilities(scores.size()),
          v(static_cast<size_t>(kEncoderPaddedFrames) * kHead),
          head_output(static_cast<size_t>(kEncoderFrames) * kHead) {}

    std::vector<llmc_float16> q;
    std::vector<llmc_float16> kt;
    std::vector<llmc_float16> scores;
    std::vector<llmc_float16> probabilities;
    std::vector<llmc_float16> v;
    std::vector<llmc_float16> head_output;
  };

  struct HeadTask {
    AttentionExecutor *owner = nullptr;
    const llmc_float16 *query = nullptr;
    const llmc_float16 *key = nullptr;
    const llmc_float16 *value = nullptr;
    llmc_float16 *output = nullptr;
    int query_rows = 0;
    int key_rows = 0;
    int padded_key_rows = 0;
    bool prepacked_key_value = false;
    llmc::WorkerF16Matmul *qk = nullptr;
    llmc::WorkerF16Matmul *pv = nullptr;
    size_t worker_index = 0;
    int result = RKNN_SUCC;
  };

  static void run_heads_task(void *argument) noexcept {
    auto &task = *static_cast<HeadTask *>(argument);
    task.result = task.owner->run_heads(task);
  }

  int run_heads(HeadTask &task) noexcept {
    Scratch &scratch = scratch_[task.worker_index];
    for (int head = static_cast<int>(task.worker_index); head < kHeads;
         head += static_cast<int>(llmc::StaticNpuScheduler::kWorkerCount)) {
      const llmc_float16 *packed_key = nullptr;
      const llmc_float16 *packed_value = nullptr;
      if (task.prepacked_key_value) {
        packed_key = task.key + static_cast<size_t>(head) * kHead *
                                    task.padded_key_rows;
        packed_value = task.value + static_cast<size_t>(head) *
                                        task.padded_key_rows * kHead;
      } else {
        std::fill_n(scratch.kt.data(),
                    static_cast<size_t>(kHead) * task.padded_key_rows,
                    static_cast<llmc_float16>(0.0f));
        std::fill_n(scratch.v.data(),
                    static_cast<size_t>(task.padded_key_rows) * kHead,
                    static_cast<llmc_float16>(0.0f));
      }
      for (int row = 0; row < task.query_rows; ++row) {
        for (int column = 0; column < kHead; ++column) {
          scratch.q[static_cast<size_t>(row) * kHead + column] =
              static_cast<llmc_float16>(
                  static_cast<float>(
                      task.query[static_cast<size_t>(row) * kModel +
                                 head * kHead + column]) *
                  kAttentionScale);
        }
      }
      if (!task.prepacked_key_value) {
        for (int row = 0; row < task.key_rows; ++row) {
          for (int column = 0; column < kHead; ++column) {
            scratch.kt[static_cast<size_t>(column) * task.padded_key_rows +
                       row] =
                task.key[static_cast<size_t>(row) * kModel + head * kHead +
                         column];
            scratch.v[static_cast<size_t>(row) * kHead + column] =
                task.value[static_cast<size_t>(row) * kModel + head * kHead +
                           column];
          }
        }
        packed_key = scratch.kt.data();
        packed_value = scratch.v.data();
      }
      int result = task.qk->run_on_worker(
          task.worker_index, scratch.q.data(), packed_key,
          scratch.scores.data());
      if (result != RKNN_SUCC)
        return result;
      for (int row = 0; row < task.query_rows; ++row) {
        const llmc_float16 *score_row =
            scratch.scores.data() +
            static_cast<size_t>(row) * task.padded_key_rows;
        llmc_float16 *probability_row =
            scratch.probabilities.data() +
            static_cast<size_t>(row) * task.padded_key_rows;
        float maximum = -std::numeric_limits<float>::infinity();
        for (int column = 0; column < task.key_rows; ++column)
          maximum = std::max(maximum, static_cast<float>(score_row[column]));
        double total = 0.0;
        for (int column = 0; column < task.key_rows; ++column)
          total += std::exp(static_cast<float>(score_row[column]) - maximum);
        const float inverse = static_cast<float>(1.0 / total);
        for (int column = 0; column < task.key_rows; ++column) {
          probability_row[column] = static_cast<llmc_float16>(
              std::exp(static_cast<float>(score_row[column]) - maximum) *
              inverse);
        }
        std::fill(probability_row + task.key_rows,
                  probability_row + task.padded_key_rows,
                  static_cast<llmc_float16>(0.0f));
      }
      result = task.pv->run_on_worker(
          task.worker_index, scratch.probabilities.data(), packed_value,
          scratch.head_output.data());
      if (result != RKNN_SUCC)
        return result;
      for (int row = 0; row < task.query_rows; ++row) {
        for (int column = 0; column < kHead; ++column) {
          task.output[static_cast<size_t>(row) * kModel + head * kHead +
                      column] =
              scratch.head_output[static_cast<size_t>(row) * kHead + column];
        }
      }
    }
    return RKNN_SUCC;
  }

  void run(const llmc_float16 *query, const llmc_float16 *key,
           const llmc_float16 *value, llmc_float16 *output, int query_rows,
           int key_rows, int padded_key_rows, bool prepacked_key_value,
           llmc::WorkerF16Matmul &qk, llmc::WorkerF16Matmul &pv) {
    const auto begin = Clock::now();
    std::array<HeadTask, llmc::StaticNpuScheduler::kWorkerCount> jobs{};
    std::array<llmc::StaticNpuScheduler::Task,
               llmc::StaticNpuScheduler::kWorkerCount>
        tasks{};
    for (size_t index = 0; index < jobs.size(); ++index) {
      jobs[index] = {this, query, key, value, output, query_rows, key_rows,
                     padded_key_rows, prepacked_key_value, &qk, &pv, index};
      tasks[index] = {run_heads_task, &jobs[index]};
    }
    scheduler_.dispatch(tasks);
    for (const auto &job : jobs) {
      if (job.result != RKNN_SUCC) {
        throw std::runtime_error("head-parallel attention failed: " +
                                 std::to_string(job.result));
      }
    }
    runs_ += kHeads * 2;
    ++dispatches_;
    wall_ms_ += elapsed_ms(begin, Clock::now());
  }

  llmc::StaticNpuScheduler &scheduler_;
  llmc::WorkerF16Matmul encoder_qk_;
  llmc::WorkerF16Matmul encoder_pv_;
  llmc::WorkerF16Matmul decoder_self_qk_;
  llmc::WorkerF16Matmul decoder_self_pv_;
  llmc::WorkerF16Matmul decoder_cross_qk_;
  llmc::WorkerF16Matmul decoder_cross_pv_;
  std::array<Scratch, llmc::StaticNpuScheduler::kWorkerCount> scratch_;
  uint64_t runs_ = 0;
  uint64_t dispatches_ = 0;
  double wall_ms_ = 0.0;
};

struct CrossCache {
  CrossCache() {
    const size_t key_elements =
        static_cast<size_t>(kHeads) * kHead * kEncoderPaddedFrames;
    const size_t value_elements =
        static_cast<size_t>(kHeads) * kEncoderPaddedFrames * kHead;
    for (int layer = 0; layer < kDecoderLayers; ++layer) {
      key[layer].assign(key_elements, static_cast<llmc_float16>(0.0f));
      value[layer].assign(value_elements, static_cast<llmc_float16>(0.0f));
    }
  }

  void pack(int layer, const llmc_float16 *row_major_key,
            const llmc_float16 *row_major_value) {
    for (int head = 0; head < kHeads; ++head) {
      llmc_float16 *head_key =
          key[layer].data() +
          static_cast<size_t>(head) * kHead * kEncoderPaddedFrames;
      llmc_float16 *head_value =
          value[layer].data() +
          static_cast<size_t>(head) * kEncoderPaddedFrames * kHead;
      for (int row = 0; row < kEncoderFrames; ++row) {
        for (int column = 0; column < kHead; ++column) {
          const size_t source = static_cast<size_t>(row) * kModel +
                                head * kHead + column;
          head_key[static_cast<size_t>(column) * kEncoderPaddedFrames + row] =
              row_major_key[source];
          head_value[static_cast<size_t>(row) * kHead + column] =
              row_major_value[source];
        }
      }
    }
  }

  std::array<std::vector<llmc_float16>, kDecoderLayers> key;
  std::array<std::vector<llmc_float16>, kDecoderLayers> value;
};

struct EncoderWorkspace {
  EncoderWorkspace()
      : conv1_input(static_cast<size_t>(kFrames) * kMels * 3),
        conv1(static_cast<size_t>(kFrames) * kModel),
        conv2_input(static_cast<size_t>(kEncoderFrames) * kModel * 3),
        hidden(static_cast<size_t>(kEncoderFrames) * kModel),
        normalized(hidden.size()), query(hidden.size()), key(hidden.size()),
        value(hidden.size()), attended(hidden.size()), projected(hidden.size()),
        expanded(static_cast<size_t>(kEncoderFrames) * kFfn) {}

  std::vector<llmc_float16> conv1_input;
  std::vector<llmc_float16> conv1;
  std::vector<llmc_float16> conv2_input;
  std::vector<llmc_float16> hidden;
  std::vector<llmc_float16> normalized;
  std::vector<llmc_float16> query;
  std::vector<llmc_float16> key;
  std::vector<llmc_float16> value;
  std::vector<llmc_float16> attended;
  std::vector<llmc_float16> projected;
  std::vector<llmc_float16> expanded;
};

void run_encoder(const std::vector<llmc_float16> &features,
                 const llmc::W4Model &model, LinearExecutor &linear,
                 AttentionExecutor &attention, CrossCache &cross,
                 EncoderWorkspace &workspace) {
  if (features.size() != static_cast<size_t>(kMels) * kFrames) {
    throw std::runtime_error("feature shape must be [128, 3000]");
  }
  auto &conv1_input = workspace.conv1_input;
  for (int frame = 0; frame < kFrames; ++frame) {
    for (int mel = 0; mel < kMels; ++mel) {
      for (int kernel = 0; kernel < 3; ++kernel) {
        const int source_frame = frame + kernel - 1;
        if (source_frame >= 0 && source_frame < kFrames) {
          conv1_input[(static_cast<size_t>(frame) * kMels + mel) * 3 + kernel] =
              features[static_cast<size_t>(mel) * kFrames + source_frame];
        }
      }
    }
  }
  auto &conv1 = workspace.conv1;
  linear.encoder("model.encoder.conv1.weight", conv1_input.data(), conv1.data(),
                 kFrames, "model.encoder.conv1.bias");
  gelu_in_place(conv1.data(), conv1.size());

  auto &conv2_input = workspace.conv2_input;
  for (int frame = 0; frame < kEncoderFrames; ++frame) {
    for (int channel = 0; channel < kModel; ++channel) {
      for (int kernel = 0; kernel < 3; ++kernel) {
        const int source_frame = frame * 2 + kernel - 1;
        if (source_frame >= 0 && source_frame < kFrames) {
          conv2_input[(static_cast<size_t>(frame) * kModel + channel) * 3 +
                      kernel] =
              conv1[static_cast<size_t>(source_frame) * kModel + channel];
        }
      }
    }
  }
  auto &hidden = workspace.hidden;
  linear.encoder("model.encoder.conv2.weight", conv2_input.data(),
                 hidden.data(), kEncoderFrames, "model.encoder.conv2.bias");
  gelu_in_place(hidden.data(), hidden.size());
  const llmc_float16 *positions =
      fp16(model.require("model.encoder.embed_positions.weight"));
  add_in_place(hidden.data(), positions, hidden.size());

  auto &normalized = workspace.normalized;
  auto &query = workspace.query;
  auto &key = workspace.key;
  auto &value = workspace.value;
  auto &attended = workspace.attended;
  auto &projected = workspace.projected;
  auto &expanded = workspace.expanded;

  for (int layer = 0; layer < kEncoderLayers; ++layer) {
    const std::string prefix =
        "model.encoder.layers." + std::to_string(layer) + ".";
    layer_norm(hidden.data(), normalized.data(), kEncoderFrames, kModel,
               fp16(model.require(prefix + "self_attn_layer_norm.weight")),
               fp16(model.require(prefix + "self_attn_layer_norm.bias")));
    linear.encoder(prefix + "self_attn.q_proj.weight", normalized.data(),
                   query.data(), kEncoderFrames,
                   prefix + "self_attn.q_proj.bias");
    linear.encoder(prefix + "self_attn.k_proj.weight", normalized.data(),
                   key.data(), kEncoderFrames);
    linear.encoder(prefix + "self_attn.v_proj.weight", normalized.data(),
                   value.data(), kEncoderFrames,
                   prefix + "self_attn.v_proj.bias");
    attention.encoder(query.data(), key.data(), value.data(), attended.data());
    linear.encoder(prefix + "self_attn.out_proj.weight", attended.data(),
                   projected.data(), kEncoderFrames,
                   prefix + "self_attn.out_proj.bias");
    add_in_place(hidden.data(), projected.data(), hidden.size());

    layer_norm(hidden.data(), normalized.data(), kEncoderFrames, kModel,
               fp16(model.require(prefix + "final_layer_norm.weight")),
               fp16(model.require(prefix + "final_layer_norm.bias")));
    linear.encoder(prefix + "fc1.weight", normalized.data(), expanded.data(),
                   kEncoderFrames, prefix + "fc1.bias");
    gelu_in_place(expanded.data(), expanded.size());
    linear.encoder(prefix + "fc2.weight", expanded.data(), projected.data(),
                   kEncoderFrames, prefix + "fc2.bias");
    add_in_place(hidden.data(), projected.data(), hidden.size());
  }
  layer_norm(hidden.data(), normalized.data(), kEncoderFrames, kModel,
             fp16(model.require("model.encoder.layer_norm.weight")),
             fp16(model.require("model.encoder.layer_norm.bias")));
  hidden.swap(normalized);

  for (int layer = 0; layer < kDecoderLayers; ++layer) {
    const std::string prefix =
        "model.decoder.layers." + std::to_string(layer) + ".encoder_attn.";
    linear.encoder(prefix + "k_proj.weight", hidden.data(), key.data(),
                   kEncoderFrames);
    linear.encoder(prefix + "v_proj.weight", hidden.data(), value.data(),
                   kEncoderFrames, prefix + "v_proj.bias");
    cross.pack(layer, key.data(), value.data());
  }
}

struct DecoderLayerNames {
  void initialize(int layer) {
    const std::string prefix =
        "model.decoder.layers." + std::to_string(layer) + ".";
    self_norm_weight = prefix + "self_attn_layer_norm.weight";
    self_norm_bias = prefix + "self_attn_layer_norm.bias";
    self_q_weight = prefix + "self_attn.q_proj.weight";
    self_q_bias = prefix + "self_attn.q_proj.bias";
    self_k_weight = prefix + "self_attn.k_proj.weight";
    self_v_weight = prefix + "self_attn.v_proj.weight";
    self_v_bias = prefix + "self_attn.v_proj.bias";
    self_out_weight = prefix + "self_attn.out_proj.weight";
    self_out_bias = prefix + "self_attn.out_proj.bias";
    cross_norm_weight = prefix + "encoder_attn_layer_norm.weight";
    cross_norm_bias = prefix + "encoder_attn_layer_norm.bias";
    cross_q_weight = prefix + "encoder_attn.q_proj.weight";
    cross_q_bias = prefix + "encoder_attn.q_proj.bias";
    cross_out_weight = prefix + "encoder_attn.out_proj.weight";
    cross_out_bias = prefix + "encoder_attn.out_proj.bias";
    final_norm_weight = prefix + "final_layer_norm.weight";
    final_norm_bias = prefix + "final_layer_norm.bias";
    fc1_weight = prefix + "fc1.weight";
    fc1_bias = prefix + "fc1.bias";
    fc2_weight = prefix + "fc2.weight";
    fc2_bias = prefix + "fc2.bias";
  }

  std::string self_norm_weight;
  std::string self_norm_bias;
  std::string self_q_weight;
  std::string self_q_bias;
  std::string self_k_weight;
  std::string self_v_weight;
  std::string self_v_bias;
  std::string self_out_weight;
  std::string self_out_bias;
  std::string cross_norm_weight;
  std::string cross_norm_bias;
  std::string cross_q_weight;
  std::string cross_q_bias;
  std::string cross_out_weight;
  std::string cross_out_bias;
  std::string final_norm_weight;
  std::string final_norm_bias;
  std::string fc1_weight;
  std::string fc1_bias;
  std::string fc2_weight;
  std::string fc2_bias;
};

class Decoder {
public:
  Decoder(const llmc::W4Model &model, LinearExecutor &linear,
          AttentionExecutor &attention, const CrossCache &cross)
      : model_(model), linear_(linear), attention_(attention), cross_(cross),
        embeddings_(fp16(model.require("model.decoder.embed_tokens.weight"))),
        positions_(fp16(model.require("model.decoder.embed_positions.weight"))),
        final_norm_weight_(
            fp16(model.require("model.decoder.layer_norm.weight"))),
        final_norm_bias_(fp16(model.require("model.decoder.layer_norm.bias"))),
        hidden_(kModel), normalized_(kModel), query_(kModel), key_(kModel),
        value_(kModel), attended_(kModel), projected_(kModel), expanded_(kFfn),
        logits_(51904) {
    for (int layer = 0; layer < kDecoderLayers; ++layer) {
      names_[layer].initialize(layer);
      self_key_[layer].assign(static_cast<size_t>(kDecoderPositions) * kModel,
                              static_cast<llmc_float16>(0.0f));
      self_value_[layer].assign(static_cast<size_t>(kDecoderPositions) * kModel,
                                static_cast<llmc_float16>(0.0f));
    }
  }

  const std::vector<llmc_float16> &step(int token, int position) {
    if (token < 0 || token >= kVocab || position < 0 ||
        position >= kDecoderPositions) {
      throw std::runtime_error("decoder token or position out of range");
    }
    for (int column = 0; column < kModel; ++column) {
      hidden_[column] = static_cast<llmc_float16>(
          static_cast<float>(
              embeddings_[static_cast<size_t>(token) * kModel + column]) +
          static_cast<float>(
              positions_[static_cast<size_t>(position) * kModel + column]));
    }

    for (int layer = 0; layer < kDecoderLayers; ++layer) {
      const DecoderLayerNames &names = names_[layer];
      layer_norm(hidden_.data(), normalized_.data(), 1, kModel,
                 fp16(model_.require(names.self_norm_weight)),
                 fp16(model_.require(names.self_norm_bias)));
      linear_.decoder(names.self_q_weight, normalized_.data(), query_.data(),
                      names.self_q_bias);
      linear_.decoder(names.self_k_weight, normalized_.data(), key_.data());
      linear_.decoder(names.self_v_weight, normalized_.data(), value_.data(),
                      names.self_v_bias);
      std::copy(key_.begin(), key_.end(),
                self_key_[layer].begin() +
                    static_cast<size_t>(position) * kModel);
      std::copy(value_.begin(), value_.end(),
                self_value_[layer].begin() +
                    static_cast<size_t>(position) * kModel);
      attention_.decoder_self(query_.data(), self_key_[layer].data(),
                              self_value_[layer].data(), position + 1,
                              attended_.data());
      linear_.decoder(names.self_out_weight, attended_.data(),
                      projected_.data(), names.self_out_bias);
      add_in_place(hidden_.data(), projected_.data(), hidden_.size());

      layer_norm(
          hidden_.data(), normalized_.data(), 1, kModel,
          fp16(model_.require(names.cross_norm_weight)),
          fp16(model_.require(names.cross_norm_bias)));
      linear_.decoder(names.cross_q_weight, normalized_.data(), query_.data(),
                      names.cross_q_bias);
      attention_.decoder_cross(query_.data(), cross_.key[layer].data(),
                               cross_.value[layer].data(), attended_.data());
      linear_.decoder(names.cross_out_weight, attended_.data(),
                      projected_.data(), names.cross_out_bias);
      add_in_place(hidden_.data(), projected_.data(), hidden_.size());

      layer_norm(hidden_.data(), normalized_.data(), 1, kModel,
                 fp16(model_.require(names.final_norm_weight)),
                 fp16(model_.require(names.final_norm_bias)));
      linear_.decoder(names.fc1_weight, normalized_.data(), expanded_.data(),
                      names.fc1_bias);
      gelu_in_place(expanded_.data(), expanded_.size());
      linear_.decoder(names.fc2_weight, expanded_.data(), projected_.data(),
                      names.fc2_bias);
      add_in_place(hidden_.data(), projected_.data(), hidden_.size());
    }
    layer_norm(hidden_.data(), normalized_.data(), 1, kModel,
               final_norm_weight_, final_norm_bias_);
    linear_.decoder("model.proj_out.weight", normalized_.data(), logits_.data());
    return logits_;
  }

private:
  const llmc::W4Model &model_;
  LinearExecutor &linear_;
  AttentionExecutor &attention_;
  const CrossCache &cross_;
  const llmc_float16 *embeddings_ = nullptr;
  const llmc_float16 *positions_ = nullptr;
  const llmc_float16 *final_norm_weight_ = nullptr;
  const llmc_float16 *final_norm_bias_ = nullptr;
  std::array<DecoderLayerNames, kDecoderLayers> names_;
  std::array<std::vector<llmc_float16>, kDecoderLayers> self_key_;
  std::array<std::vector<llmc_float16>, kDecoderLayers> self_value_;
  std::vector<llmc_float16> hidden_;
  std::vector<llmc_float16> normalized_;
  std::vector<llmc_float16> query_;
  std::vector<llmc_float16> key_;
  std::vector<llmc_float16> value_;
  std::vector<llmc_float16> attended_;
  std::vector<llmc_float16> projected_;
  std::vector<llmc_float16> expanded_;
  std::vector<llmc_float16> logits_;
};

constexpr std::array<int, 88> kSuppressTokens = {
    1,     2,     7,     8,     9,     10,    14,    25,    26,    27,
    28,    29,    31,    58,    59,    60,    61,    62,    63,    90,
    91,    92,    93,    359,   503,   522,   542,   873,   893,   902,
    918,   922,   931,   1350,  1853,  1982,  2460,  2627,  3246,  3253,
    3268,  3536,  3846,  3961,  4183,  4667,  6585,  6647,  7273,  9061,
    9383,  10428, 10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635,
    15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130,
    26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863, 47425,
    49870, 50254, 50258, 50359, 50360, 50361, 50362, 50363};

bool suppressed(int token, int decode_index) {
  return std::find(kSuppressTokens.begin(), kSuppressTokens.end(), token) !=
             kSuppressTokens.end() ||
         (decode_index == 0 && (token == 220 || token == kEos));
}

int argmax(const std::vector<llmc_float16> &logits, int decode_index) {
  int best = -1;
  float best_value = -std::numeric_limits<float>::infinity();
  for (int token = 0; token < kVocab; ++token) {
    if (suppressed(token, decode_index))
      continue;
    const float value = static_cast<float>(logits[token]);
    if (value > best_value) {
      best_value = value;
      best = token;
    }
  }
  return best;
}

void append_utf8(uint32_t codepoint, std::string &output) {
  if (codepoint <= 0x7f) {
    output.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7ff) {
    output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  } else if (codepoint <= 0xffff) {
    output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  } else {
    output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
  }
}

uint32_t hex4(const std::string &text, size_t &position) {
  uint32_t value = 0;
  for (int index = 0; index < 4; ++index) {
    if (position >= text.size())
      throw std::runtime_error("bad JSON escape");
    const char c = text[position++];
    value <<= 4;
    if (c >= '0' && c <= '9')
      value |= c - '0';
    else if (c >= 'a' && c <= 'f')
      value |= c - 'a' + 10;
    else if (c >= 'A' && c <= 'F')
      value |= c - 'A' + 10;
    else
      throw std::runtime_error("bad JSON hex escape");
  }
  return value;
}

std::string json_string(const std::string &text, size_t &position) {
  if (position >= text.size() || text[position++] != '"') {
    throw std::runtime_error("expected JSON string");
  }
  std::string output;
  while (position < text.size()) {
    const char c = text[position++];
    if (c == '"')
      return output;
    if (c != '\\') {
      output.push_back(c);
      continue;
    }
    if (position >= text.size())
      throw std::runtime_error("bad JSON escape");
    const char escaped = text[position++];
    if (escaped == '"' || escaped == '\\' || escaped == '/') {
      output.push_back(escaped);
    } else if (escaped == 'b')
      output.push_back('\b');
    else if (escaped == 'f')
      output.push_back('\f');
    else if (escaped == 'n')
      output.push_back('\n');
    else if (escaped == 'r')
      output.push_back('\r');
    else if (escaped == 't')
      output.push_back('\t');
    else if (escaped == 'u') {
      uint32_t codepoint = hex4(text, position);
      if (codepoint >= 0xd800 && codepoint <= 0xdbff &&
          position + 2 <= text.size() && text[position] == '\\' &&
          text[position + 1] == 'u') {
        position += 2;
        const uint32_t low = hex4(text, position);
        if (low < 0xdc00 || low > 0xdfff) {
          throw std::runtime_error("bad JSON surrogate pair");
        }
        codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
      }
      append_utf8(codepoint, output);
    } else {
      throw std::runtime_error("unsupported JSON escape");
    }
  }
  throw std::runtime_error("unterminated JSON string");
}

uint32_t utf8_codepoint(const std::string &text, size_t &position) {
  const uint8_t first = static_cast<uint8_t>(text[position++]);
  if ((first & 0x80) == 0)
    return first;
  int remaining = 0;
  uint32_t codepoint = 0;
  if ((first & 0xe0) == 0xc0) {
    remaining = 1;
    codepoint = first & 0x1f;
  } else if ((first & 0xf0) == 0xe0) {
    remaining = 2;
    codepoint = first & 0x0f;
  } else if ((first & 0xf8) == 0xf0) {
    remaining = 3;
    codepoint = first & 0x07;
  } else {
    throw std::runtime_error("invalid UTF-8");
  }
  while (remaining-- > 0) {
    if (position >= text.size())
      throw std::runtime_error("truncated UTF-8");
    const uint8_t next = static_cast<uint8_t>(text[position++]);
    if ((next & 0xc0) != 0x80)
      throw std::runtime_error("invalid UTF-8");
    codepoint = (codepoint << 6) | (next & 0x3f);
  }
  return codepoint;
}

const std::unordered_map<uint32_t, uint8_t> &byte_decoder() {
  static const auto decoder = [] {
    std::array<bool, 256> direct{};
    for (int value = 33; value <= 126; ++value)
      direct[value] = true;
    for (int value = 161; value <= 172; ++value)
      direct[value] = true;
    for (int value = 174; value <= 255; ++value)
      direct[value] = true;
    std::unordered_map<uint32_t, uint8_t> result;
    for (int value = 0; value < 256; ++value) {
      if (direct[value])
        result[value] = static_cast<uint8_t>(value);
    }
    uint32_t synthetic = 256;
    for (int value = 0; value < 256; ++value) {
      if (!direct[value])
        result[synthetic++] = static_cast<uint8_t>(value);
    }
    return result;
  }();
  return decoder;
}

std::string token_bytes(const std::string &token) {
  std::string output;
  size_t position = 0;
  while (position < token.size()) {
    const uint32_t codepoint = utf8_codepoint(token, position);
    const auto found = byte_decoder().find(codepoint);
    if (found == byte_decoder().end()) {
      throw std::runtime_error("unmapped vocabulary codepoint");
    }
    output.push_back(static_cast<char>(found->second));
  }
  return output;
}

void whitespace(const std::string &text, size_t &position) {
  while (position < text.size() &&
         (text[position] == ' ' || text[position] == '\n' ||
          text[position] == '\r' || text[position] == '\t')) {
    ++position;
  }
}

std::vector<std::string> load_vocabulary(const char *path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error(std::string("cannot open ") + path);
  const std::string text((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  std::vector<std::string> vocabulary(50257);
  size_t position = 0;
  whitespace(text, position);
  if (position >= text.size() || text[position++] != '{') {
    throw std::runtime_error("vocab.json must contain an object");
  }
  while (true) {
    whitespace(text, position);
    if (position < text.size() && text[position] == '}')
      break;
    const std::string token = json_string(text, position);
    whitespace(text, position);
    if (position >= text.size() || text[position++] != ':') {
      throw std::runtime_error("expected ':' in vocab.json");
    }
    whitespace(text, position);
    int id = 0;
    if (position >= text.size() || text[position] < '0' ||
        text[position] > '9') {
      throw std::runtime_error("expected token id in vocab.json");
    }
    while (position < text.size() && text[position] >= '0' &&
           text[position] <= '9') {
      id = id * 10 + text[position++] - '0';
    }
    if (id >= 0 && id < static_cast<int>(vocabulary.size())) {
      vocabulary[id] = token_bytes(token);
    }
    whitespace(text, position);
    if (position < text.size() && text[position] == ',') {
      ++position;
    } else if (position >= text.size() || text[position] != '}') {
      throw std::runtime_error("expected ',' or '}' in vocab.json");
    }
  }
  return vocabulary;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 6 || argc > 8) {
    std::fprintf(stderr,
                 "usage: %s ENCODER.llmc DECODER.llmc vocab.json AUDIO.wav "
                 "baseline|llmc [--npu-scheduler parallel3]\n",
                 argv[0]);
    return 2;
  }
  try {
    const char *scheduler_text = "parallel3";
    if (argc == 8) {
      if (std::strcmp(argv[6], "--npu-scheduler") != 0) {
        throw std::runtime_error("expected --npu-scheduler");
      }
      scheduler_text = argv[7];
    } else if (argc == 7) {
      throw std::runtime_error("--npu-scheduler requires a value");
    }
    if (std::strcmp(scheduler_text, "parallel3") != 0)
      throw std::runtime_error("Whisper runtime requires parallel3 scheduler");
    const auto mode = llmc::parse_run_mode(argv[5]);
    const auto core_masks = llmc::parallel_npu_core_masks();
    const auto init_begin = Clock::now();
    llmc::W4Model encoder_model(argv[1]);
    llmc::W4Model decoder_model(argv[2]);
    const auto vocabulary = load_vocabulary(argv[3]);
    LinearExecutor linear(encoder_model, decoder_model, mode, core_masks);
    AttentionExecutor attention(core_masks);
    CrossCache cross;
    EncoderWorkspace encoder_workspace;
    Decoder decoder(decoder_model, linear, attention, cross);
    linear.prepare();
    const auto init_end = Clock::now();

    const auto preprocess_begin = Clock::now();
    const llmc::PcmAudio audio =
        llmc::read_mono_pcm16_wav(argv[4], kSampleRate);
    const double audio_seconds =
        audio.samples.size() / static_cast<double>(audio.sample_rate);
    const auto features = extract_features(audio);
    const auto preprocess_end = Clock::now();

    NpuLoadMonitor monitor;
    monitor.start();
    monitor.stage(NpuLoadMonitor::Stage::kEncoder);
    const auto encoder_begin = Clock::now();
    run_encoder(features, encoder_model, linear, attention, cross,
                encoder_workspace);
    const auto encoder_end = Clock::now();
    const double encoder_linear_wall_ms = linear.wall_ms();
    const double encoder_attention_wall_ms = attention.wall_ms();
    monitor.stage(NpuLoadMonitor::Stage::kDecoder);
    std::vector<int> tokens = {kSot, kEnglish, kTranscribe, kNoTimestamps};
    tokens.reserve(kMaxNewTokens + 4);
    std::vector<double> decoder_steps;
    decoder_steps.reserve(kMaxNewTokens + 4);
    const std::vector<llmc_float16> *logits = nullptr;
    for (size_t position = 0; position < tokens.size(); ++position) {
      const auto begin = Clock::now();
      logits = &decoder.step(tokens[position], static_cast<int>(position));
      decoder_steps.push_back(elapsed_ms(begin, Clock::now()));
    }
    for (int decode_index = 0; decode_index < kMaxNewTokens; ++decode_index) {
      const int next = argmax(*logits, decode_index);
      if (next < 0)
        throw std::runtime_error("argmax produced no token");
      tokens.push_back(next);
      if (next == kEos || tokens.size() >= kDecoderPositions)
        break;
      const auto begin = Clock::now();
      logits = &decoder.step(next, static_cast<int>(tokens.size() - 1));
      decoder_steps.push_back(elapsed_ms(begin, Clock::now()));
    }
    const auto inference_end = Clock::now();
    monitor.stage(NpuLoadMonitor::Stage::kIdle);
    monitor.stop();

    std::string text;
    for (int token : tokens) {
      if (token >= 0 && token < static_cast<int>(vocabulary.size())) {
        text += vocabulary[token];
      }
    }
    while (!text.empty() && (text.front() == ' ' || text.front() == '\n')) {
      text.erase(text.begin());
    }
    while (!text.empty() && (text.back() == ' ' || text.back() == '\n')) {
      text.pop_back();
    }
    double decoder_total = 0.0;
    for (double value : decoder_steps)
      decoder_total += value;
    std::printf("runtime: llmc-native-cpp-aot-w4-rknpu2-fp16-matmul\n");
    std::printf("quantization: cpu-w4-group32-to-npu-fp16\n");
    std::printf("mode: %s\n", llmc::run_mode_name(mode));
    std::printf("audio_seconds: %.3f\n", audio_seconds);
    std::printf("model_init_ms: %.3f\n", elapsed_ms(init_begin, init_end));
    std::printf("preprocess_ms: %.3f\n",
                elapsed_ms(preprocess_begin, preprocess_end));
    std::printf("encoder_ms: %.3f\n", elapsed_ms(encoder_begin, encoder_end));
    std::printf("decoder_steps: %zu\n", decoder_steps.size());
    std::printf("decoder_total_ms: %.3f\n", decoder_total);
    std::printf("avg_decoder_step_ms: %.3f\n",
                decoder_steps.empty() ? 0.0
                                      : decoder_total / decoder_steps.size());
    std::printf("inference_ms: %.3f\n",
                elapsed_ms(encoder_begin, inference_end));
    std::printf("generated_tokens: %zu\n", tokens.size() - 4);
    std::printf("w4_linear_runs: %llu\n",
                static_cast<unsigned long long>(linear.runs()));
    std::printf("fp16_attention_matmul_runs: %llu\n",
                static_cast<unsigned long long>(attention.runs()));
    std::printf("attention_host_dispatches: %llu\n",
                static_cast<unsigned long long>(attention.dispatches()));
    std::printf(
        "baseline_weight_uploads: %llu\n",
        static_cast<unsigned long long>(linear.baseline_weight_uploads()));
    std::printf("aot_linear_plans: %zu\n", linear.aot_plan_count());
    std::printf("aot_prepared_weights: %zu\n", linear.aot_weight_count());
    std::printf("aot_resident_weight_bytes: %llu\n",
                static_cast<unsigned long long>(linear.aot_weight_bytes()));
    std::printf("w4_linear_wall_ms: %.3f\n", linear.wall_ms());
    std::printf("attention_wall_ms: %.3f\n", attention.wall_ms());
    std::printf("encoder_w4_linear_wall_ms: %.3f\n",
                encoder_linear_wall_ms);
    std::printf("decoder_w4_linear_wall_ms: %.3f\n",
                linear.wall_ms() - encoder_linear_wall_ms);
    std::printf("encoder_attention_wall_ms: %.3f\n",
                encoder_attention_wall_ms);
    std::printf("decoder_attention_wall_ms: %.3f\n",
                attention.wall_ms() - encoder_attention_wall_ms);
    std::printf("npu_scheduler: %s\n", scheduler_text);
    std::printf("npu_sharding: output-columns-N-aligned32\n");
    std::printf("attention_scheduler: static-head-round-robin-Core0-Core1-Core2\n");
    std::printf("npu_worker_model: one-persistent-thread-per-core\n");
    std::printf("npu_configured_cores: 3\n");
    monitor.print();
    std::printf("token_ids:");
    for (int token : tokens)
      std::printf(" %d", token);
    std::printf("\ntext:\n%s\n", text.c_str());
  } catch (const std::exception &error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
  return 0;
}
