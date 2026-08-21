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
                 llmc::W4RunMode mode, rknn_core_mask core_mask)
      : encoder_(encoder), decoder_(decoder), mode_(mode),
        core_mask_(core_mask) {}

  void encoder(const std::string &weight_name, const llmc_float16 *input,
               llmc_float16 *output, int rows,
               const std::string &bias_name = std::string()) {
    run(encoder_, &encoder_linears_, weight_name, input, output, rows,
        bias_name);
  }

  void decoder(const std::string &weight_name, const llmc_float16 *input,
               llmc_float16 *output,
               const std::string &bias_name = std::string()) {
    run(decoder_, &decoder_linears_, weight_name, input, output, 1, bias_name);
  }

  uint64_t runs() const { return runs_; }
  uint64_t baseline_weight_uploads() const { return baseline_weight_uploads_; }
  double wall_ms() const { return wall_ms_; }

  void prepare() {
    for (const auto &pair : encoder_.tensors()) {
      if (pair.second.encoding != llmc::W4Encoding::kW4A16Group)
        continue;
      const int rows =
          pair.first == "model.encoder.conv1.weight" ? kFrames : kEncoderFrames;
      encoder_linears_.emplace(
          pair.first, std::make_unique<llmc::W4Linear>(pair.second, rows, mode_,
                                                       core_mask_));
    }
    for (const auto &pair : decoder_.tensors()) {
      if (pair.second.encoding != llmc::W4Encoding::kW4A16Group)
        continue;
      decoder_linears_.emplace(
          pair.first,
          std::make_unique<llmc::W4Linear>(pair.second, 1, mode_, core_mask_));
    }
  }

private:
  using Cache =
      std::unordered_map<std::string, std::unique_ptr<llmc::W4Linear>>;

  void run(const llmc::W4Model &model, Cache *cache,
           const std::string &weight_name, const llmc_float16 *input,
           llmc_float16 *output, int rows, const std::string &bias_name) {
    const auto begin = Clock::now();
    llmc::W4Linear *linear = nullptr;
    std::unique_ptr<llmc::W4Linear> temporary;
    if (cache == nullptr) {
      temporary = std::make_unique<llmc::W4Linear>(model.require(weight_name),
                                                   rows, mode_, core_mask_);
      linear = temporary.get();
    } else {
      auto found = cache->find(weight_name);
      if (found == cache->end()) {
        auto inserted = cache->emplace(
            weight_name,
            std::make_unique<llmc::W4Linear>(model.require(weight_name), rows,
                                             mode_, core_mask_));
        found = inserted.first;
      }
      linear = found->second.get();
    }
    linear->run(input, output);
    if (!bias_name.empty()) {
      add_bias(output, rows, linear->output_columns(),
               fp16(model.require(bias_name)));
    }
    if (mode_ == llmc::W4RunMode::kBaseline)
      ++baseline_weight_uploads_;
    ++runs_;
    wall_ms_ += elapsed_ms(begin, Clock::now());
  }

  const llmc::W4Model &encoder_;
  const llmc::W4Model &decoder_;
  llmc::W4RunMode mode_;
  rknn_core_mask core_mask_;
  Cache encoder_linears_;
  Cache decoder_linears_;
  uint64_t runs_ = 0;
  uint64_t baseline_weight_uploads_ = 0;
  double wall_ms_ = 0.0;
};

class AttentionExecutor {
public:
  explicit AttentionExecutor(rknn_core_mask core_mask)
      : encoder_qk_(kEncoderFrames, kHead, kEncoderPaddedFrames, core_mask),
        encoder_pv_(kEncoderFrames, kEncoderPaddedFrames, kHead, core_mask),
        decoder_self_qk_(1, kHead, kDecoderPositions, core_mask),
        decoder_self_pv_(1, kDecoderPositions, kHead, core_mask),
        decoder_cross_qk_(1, kHead, kEncoderPaddedFrames, core_mask),
        decoder_cross_pv_(1, kEncoderPaddedFrames, kHead, core_mask) {}

  void encoder(const llmc_float16 *query, const llmc_float16 *key,
               const llmc_float16 *value, llmc_float16 *output) {
    run(query, key, value, output, kEncoderFrames, kEncoderFrames,
        kEncoderPaddedFrames, encoder_qk_, encoder_pv_);
  }

  void decoder_self(const llmc_float16 *query, const llmc_float16 *key,
                    const llmc_float16 *value, int valid_positions,
                    llmc_float16 *output) {
    run(query, key, value, output, 1, valid_positions, kDecoderPositions,
        decoder_self_qk_, decoder_self_pv_);
  }

  void decoder_cross(const llmc_float16 *query, const llmc_float16 *key,
                     const llmc_float16 *value, llmc_float16 *output) {
    run(query, key, value, output, 1, kEncoderFrames, kEncoderPaddedFrames,
        decoder_cross_qk_, decoder_cross_pv_);
  }

  uint64_t runs() const { return runs_; }
  double wall_ms() const { return wall_ms_; }

private:
  void run(const llmc_float16 *query, const llmc_float16 *key,
           const llmc_float16 *value, llmc_float16 *output, int query_rows,
           int key_rows, int padded_key_rows, llmc::F16Matmul &qk,
           llmc::F16Matmul &pv) {
    const auto begin = Clock::now();
    std::vector<llmc_float16> q(static_cast<size_t>(query_rows) * kHead);
    std::vector<llmc_float16> kt(static_cast<size_t>(kHead) * padded_key_rows);
    std::vector<llmc_float16> scores(static_cast<size_t>(query_rows) *
                                     padded_key_rows);
    std::vector<llmc_float16> probabilities(scores.size(),
                                            static_cast<llmc_float16>(0.0f));
    std::vector<llmc_float16> v(static_cast<size_t>(padded_key_rows) * kHead,
                                static_cast<llmc_float16>(0.0f));
    std::vector<llmc_float16> head_output(static_cast<size_t>(query_rows) *
                                          kHead);

    for (int head = 0; head < kHeads; ++head) {
      std::fill(kt.begin(), kt.end(), static_cast<llmc_float16>(0.0f));
      std::fill(v.begin(), v.end(), static_cast<llmc_float16>(0.0f));
      for (int row = 0; row < query_rows; ++row) {
        for (int column = 0; column < kHead; ++column) {
          q[static_cast<size_t>(row) * kHead + column] =
              static_cast<llmc_float16>(
                  static_cast<float>(query[static_cast<size_t>(row) * kModel +
                                           head * kHead + column]) *
                  kAttentionScale);
        }
      }
      for (int row = 0; row < key_rows; ++row) {
        for (int column = 0; column < kHead; ++column) {
          kt[static_cast<size_t>(column) * padded_key_rows + row] =
              key[static_cast<size_t>(row) * kModel + head * kHead + column];
          v[static_cast<size_t>(row) * kHead + column] =
              value[static_cast<size_t>(row) * kModel + head * kHead + column];
        }
      }
      qk.run(q.data(), kt.data(), scores.data());
      for (int row = 0; row < query_rows; ++row) {
        const llmc_float16 *score_row =
            scores.data() + static_cast<size_t>(row) * padded_key_rows;
        llmc_float16 *probability_row =
            probabilities.data() + static_cast<size_t>(row) * padded_key_rows;
        float maximum = -std::numeric_limits<float>::infinity();
        for (int column = 0; column < key_rows; ++column) {
          maximum = std::max(maximum, static_cast<float>(score_row[column]));
        }
        double total = 0.0;
        for (int column = 0; column < key_rows; ++column) {
          total += std::exp(static_cast<float>(score_row[column]) - maximum);
        }
        const float inverse = static_cast<float>(1.0 / total);
        for (int column = 0; column < key_rows; ++column) {
          probability_row[column] = static_cast<llmc_float16>(
              std::exp(static_cast<float>(score_row[column]) - maximum) *
              inverse);
        }
        std::fill(probability_row + key_rows, probability_row + padded_key_rows,
                  static_cast<llmc_float16>(0.0f));
      }
      pv.run(probabilities.data(), v.data(), head_output.data());
      for (int row = 0; row < query_rows; ++row) {
        for (int column = 0; column < kHead; ++column) {
          output[static_cast<size_t>(row) * kModel + head * kHead + column] =
              head_output[static_cast<size_t>(row) * kHead + column];
        }
      }
      runs_ += 2;
    }
    wall_ms_ += elapsed_ms(begin, Clock::now());
  }

  llmc::F16Matmul encoder_qk_;
  llmc::F16Matmul encoder_pv_;
  llmc::F16Matmul decoder_self_qk_;
  llmc::F16Matmul decoder_self_pv_;
  llmc::F16Matmul decoder_cross_qk_;
  llmc::F16Matmul decoder_cross_pv_;
  uint64_t runs_ = 0;
  double wall_ms_ = 0.0;
};

struct CrossCache {
  std::array<std::vector<llmc_float16>, kDecoderLayers> key;
  std::array<std::vector<llmc_float16>, kDecoderLayers> value;
};

std::vector<llmc_float16> run_encoder(const std::vector<llmc_float16> &features,
                                      const llmc::W4Model &model,
                                      LinearExecutor &linear,
                                      AttentionExecutor &attention,
                                      CrossCache &cross) {
  if (features.size() != static_cast<size_t>(kMels) * kFrames) {
    throw std::runtime_error("feature shape must be [128, 3000]");
  }
  std::vector<llmc_float16> conv1_input(static_cast<size_t>(kFrames) * kMels *
                                            3,
                                        static_cast<llmc_float16>(0.0f));
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
  std::vector<llmc_float16> conv1(static_cast<size_t>(kFrames) * kModel);
  linear.encoder("model.encoder.conv1.weight", conv1_input.data(), conv1.data(),
                 kFrames, "model.encoder.conv1.bias");
  gelu_in_place(conv1.data(), conv1.size());

  std::vector<llmc_float16> conv2_input(static_cast<size_t>(kEncoderFrames) *
                                            kModel * 3,
                                        static_cast<llmc_float16>(0.0f));
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
  std::vector<llmc_float16> hidden(static_cast<size_t>(kEncoderFrames) *
                                   kModel);
  linear.encoder("model.encoder.conv2.weight", conv2_input.data(),
                 hidden.data(), kEncoderFrames, "model.encoder.conv2.bias");
  gelu_in_place(hidden.data(), hidden.size());
  const llmc_float16 *positions =
      fp16(model.require("model.encoder.embed_positions.weight"));
  add_in_place(hidden.data(), positions, hidden.size());

  std::vector<llmc_float16> normalized(hidden.size());
  std::vector<llmc_float16> query(hidden.size());
  std::vector<llmc_float16> key(hidden.size());
  std::vector<llmc_float16> value(hidden.size());
  std::vector<llmc_float16> attended(hidden.size());
  std::vector<llmc_float16> projected(hidden.size());
  std::vector<llmc_float16> expanded(static_cast<size_t>(kEncoderFrames) *
                                     kFfn);

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
    cross.key[layer].resize(hidden.size());
    cross.value[layer].resize(hidden.size());
    linear.encoder(prefix + "k_proj.weight", hidden.data(),
                   cross.key[layer].data(), kEncoderFrames);
    linear.encoder(prefix + "v_proj.weight", hidden.data(),
                   cross.value[layer].data(), kEncoderFrames,
                   prefix + "v_proj.bias");
  }
  return hidden;
}

class Decoder {
public:
  Decoder(const llmc::W4Model &model, LinearExecutor &linear,
          AttentionExecutor &attention, const CrossCache &cross)
      : model_(model), linear_(linear), attention_(attention), cross_(cross) {
    for (int layer = 0; layer < kDecoderLayers; ++layer) {
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
    const llmc_float16 *embeddings =
        fp16(model_.require("model.decoder.embed_tokens.weight"));
    const llmc_float16 *positions =
        fp16(model_.require("model.decoder.embed_positions.weight"));
    std::vector<llmc_float16> hidden(kModel);
    for (int column = 0; column < kModel; ++column) {
      hidden[column] = static_cast<llmc_float16>(
          static_cast<float>(
              embeddings[static_cast<size_t>(token) * kModel + column]) +
          static_cast<float>(
              positions[static_cast<size_t>(position) * kModel + column]));
    }

    std::vector<llmc_float16> normalized(kModel);
    std::vector<llmc_float16> query(kModel);
    std::vector<llmc_float16> key(kModel);
    std::vector<llmc_float16> value(kModel);
    std::vector<llmc_float16> attended(kModel);
    std::vector<llmc_float16> projected(kModel);
    std::vector<llmc_float16> expanded(kFfn);

    for (int layer = 0; layer < kDecoderLayers; ++layer) {
      const std::string prefix =
          "model.decoder.layers." + std::to_string(layer) + ".";
      layer_norm(hidden.data(), normalized.data(), 1, kModel,
                 fp16(model_.require(prefix + "self_attn_layer_norm.weight")),
                 fp16(model_.require(prefix + "self_attn_layer_norm.bias")));
      linear_.decoder(prefix + "self_attn.q_proj.weight", normalized.data(),
                      query.data(), prefix + "self_attn.q_proj.bias");
      linear_.decoder(prefix + "self_attn.k_proj.weight", normalized.data(),
                      key.data());
      linear_.decoder(prefix + "self_attn.v_proj.weight", normalized.data(),
                      value.data(), prefix + "self_attn.v_proj.bias");
      std::copy(key.begin(), key.end(),
                self_key_[layer].begin() +
                    static_cast<size_t>(position) * kModel);
      std::copy(value.begin(), value.end(),
                self_value_[layer].begin() +
                    static_cast<size_t>(position) * kModel);
      attention_.decoder_self(query.data(), self_key_[layer].data(),
                              self_value_[layer].data(), position + 1,
                              attended.data());
      linear_.decoder(prefix + "self_attn.out_proj.weight", attended.data(),
                      projected.data(), prefix + "self_attn.out_proj.bias");
      add_in_place(hidden.data(), projected.data(), hidden.size());

      layer_norm(
          hidden.data(), normalized.data(), 1, kModel,
          fp16(model_.require(prefix + "encoder_attn_layer_norm.weight")),
          fp16(model_.require(prefix + "encoder_attn_layer_norm.bias")));
      linear_.decoder(prefix + "encoder_attn.q_proj.weight", normalized.data(),
                      query.data(), prefix + "encoder_attn.q_proj.bias");
      attention_.decoder_cross(query.data(), cross_.key[layer].data(),
                               cross_.value[layer].data(), attended.data());
      linear_.decoder(prefix + "encoder_attn.out_proj.weight", attended.data(),
                      projected.data(), prefix + "encoder_attn.out_proj.bias");
      add_in_place(hidden.data(), projected.data(), hidden.size());

      layer_norm(hidden.data(), normalized.data(), 1, kModel,
                 fp16(model_.require(prefix + "final_layer_norm.weight")),
                 fp16(model_.require(prefix + "final_layer_norm.bias")));
      linear_.decoder(prefix + "fc1.weight", normalized.data(), expanded.data(),
                      prefix + "fc1.bias");
      gelu_in_place(expanded.data(), expanded.size());
      linear_.decoder(prefix + "fc2.weight", expanded.data(), projected.data(),
                      prefix + "fc2.bias");
      add_in_place(hidden.data(), projected.data(), hidden.size());
    }
    layer_norm(hidden.data(), normalized.data(), 1, kModel,
               fp16(model_.require("model.decoder.layer_norm.weight")),
               fp16(model_.require("model.decoder.layer_norm.bias")));
    logits_.resize(51904);
    linear_.decoder("model.proj_out.weight", normalized.data(), logits_.data());
    return logits_;
  }

private:
  const llmc::W4Model &model_;
  LinearExecutor &linear_;
  AttentionExecutor &attention_;
  const CrossCache &cross_;
  std::array<std::vector<llmc_float16>, kDecoderLayers> self_key_;
  std::array<std::vector<llmc_float16>, kDecoderLayers> self_value_;
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
                 "baseline|llmc [--core-mask auto|0|1|2|0,1|0,1,2|all]\n",
                 argv[0]);
    return 2;
  }
  try {
    const char *core_text = "auto";
    if (argc == 8) {
      if (std::strcmp(argv[6], "--core-mask") != 0) {
        throw std::runtime_error("expected --core-mask");
      }
      core_text = argv[7];
    } else if (argc == 7) {
      throw std::runtime_error("--core-mask requires a value");
    }
    const auto mode = llmc::parse_run_mode(argv[5]);
    const auto core_mask = llmc::parse_core_mask(core_text);
    const auto init_begin = Clock::now();
    llmc::W4Model encoder_model(argv[1]);
    llmc::W4Model decoder_model(argv[2]);
    const auto vocabulary = load_vocabulary(argv[3]);
    LinearExecutor linear(encoder_model, decoder_model, mode, core_mask);
    AttentionExecutor attention(core_mask);
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
    CrossCache cross;
    monitor.stage(NpuLoadMonitor::Stage::kEncoder);
    const auto encoder_begin = Clock::now();
    const auto encoder_hidden =
        run_encoder(features, encoder_model, linear, attention, cross);
    const auto encoder_end = Clock::now();
    monitor.stage(NpuLoadMonitor::Stage::kDecoder);
    Decoder decoder(decoder_model, linear, attention, cross);
    std::vector<int> tokens = {kSot, kEnglish, kTranscribe, kNoTimestamps};
    std::vector<double> decoder_steps;
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
    std::printf("runtime: proprietary-rknpu2-matmul-native-cpp\n");
    std::printf("quantization: w4a16-group32\n");
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
    std::printf(
        "baseline_weight_uploads: %llu\n",
        static_cast<unsigned long long>(linear.baseline_weight_uploads()));
    std::printf("w4_linear_wall_ms: %.3f\n", linear.wall_ms());
    std::printf("attention_wall_ms: %.3f\n", attention.wall_ms());
    std::printf("core_mask: %s\n", core_text);
    if (std::strcmp(core_text, "auto") == 0) {
      std::printf("npu_configured_cores: auto\n");
    } else {
      std::printf("npu_configured_cores: %d\n",
                  llmc::configured_core_count(core_text));
    }
    monitor.print();
    std::printf("token_ids:");
    for (int token : tokens)
      std::printf(" %d", token);
    std::printf("\ntext:\n%s\n", text.c_str());
    (void)encoder_hidden;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
  return 0;
}
