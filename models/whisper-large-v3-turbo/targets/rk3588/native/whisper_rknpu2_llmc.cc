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
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fftw3.h>

#include "audio_utils.h"
#include "float16_compat.h"
#include "rknn_api.h"

namespace {

constexpr int kSampleRate = 16000;
constexpr int kFftSize = 400;
constexpr int kHopLength = 160;
constexpr int kSamples = 30 * kSampleRate;
constexpr int kFrames = 3000;
constexpr int kBins = kFftSize / 2 + 1;
constexpr int kMels = 128;
constexpr int kVocabSize = 51866;
constexpr int kEos = 50257;
constexpr int kSot = 50258;
constexpr int kEnglish = 50259;
constexpr int kTranscribe = 50360;
constexpr int kNoTimestamps = 50364;
constexpr int kMaxNewTokens = 128;

using Clock = std::chrono::steady_clock;

constexpr int kNpuCoreCount = 3;
constexpr int kNpuLoadSampleIntervalMs = 10;
constexpr const char* kNpuLoadPath = "/sys/kernel/debug/rknpu/load";

double milliseconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

void check_rknn(int rc, const char* operation) {
  if (rc != RKNN_SUCC) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             std::to_string(rc));
  }
}

enum class NpuStage : int {
  kIdle = 0,
  kEncoder = 1,
  kDecoder = 2,
};

struct NpuLoadStats {
  uint64_t samples = 0;
  std::array<uint64_t, kNpuCoreCount> sums{};
  std::array<int, kNpuCoreCount> maxima{};
};

class NpuLoadMonitor {
 public:
  explicit NpuLoadMonitor(std::string path) : path_(std::move(path)) {}
  NpuLoadMonitor(const NpuLoadMonitor&) = delete;
  NpuLoadMonitor& operator=(const NpuLoadMonitor&) = delete;

  ~NpuLoadMonitor() { stop(); }

  void start() {
    if (running_.exchange(true)) return;
    worker_ = std::thread([this] { sample_loop(); });
  }

  void set_stage(NpuStage stage) {
    stage_.store(static_cast<int>(stage), std::memory_order_release);
  }

  void stop() {
    if (!running_.exchange(false)) return;
    if (worker_.joinable()) worker_.join();
  }

  void print() const {
    std::printf("npu_load_source: %s\n", path_.c_str());
    std::printf("npu_load_sample_interval_ms: %d\n",
                kNpuLoadSampleIntervalMs);
    if (!source_available_) {
      std::printf("npu_load_status: unavailable\n");
      std::printf("npu_observed_active_cores: unavailable\n");
      return;
    }
    std::printf("npu_load_status: sampled\n");
    print_stage("encoder", stats_[0]);
    print_stage("decoder", stats_[1]);

    int active_cores = 0;
    for (int core = 0; core < kNpuCoreCount; ++core) {
      if (stats_[0].maxima[core] > 0 || stats_[1].maxima[core] > 0) {
        ++active_cores;
      }
    }
    std::printf("npu_observed_active_cores: %d\n", active_cores);
  }

 private:
  static bool parse_load(const std::string& text,
                         std::array<int, kNpuCoreCount>& loads) {
    bool found_any = false;
    for (int core = 0; core < kNpuCoreCount; ++core) {
      const std::string label = "Core" + std::to_string(core);
      size_t position = text.find(label);
      if (position == std::string::npos) continue;
      position = text.find(':', position + label.size());
      if (position == std::string::npos) continue;
      const char* begin = text.c_str() + position + 1;
      char* end = nullptr;
      const long value = std::strtol(begin, &end, 10);
      if (end == begin || value < 0 || value > 100) continue;
      loads[core] = static_cast<int>(value);
      found_any = true;
    }
    return found_any;
  }

  static void print_stage(const char* name, const NpuLoadStats& stats) {
    std::printf("npu_%s_load_samples: %llu\n", name,
                static_cast<unsigned long long>(stats.samples));
    for (int core = 0; core < kNpuCoreCount; ++core) {
      const double average = stats.samples == 0
          ? 0.0
          : static_cast<double>(stats.sums[core]) / stats.samples;
      std::printf("npu_%s_core%d_avg_load_pct: %.2f\n", name, core,
                  average);
      std::printf("npu_%s_core%d_max_load_pct: %d\n", name, core,
                  stats.maxima[core]);
    }
  }

  void sample_loop() {
    while (running_.load(std::memory_order_acquire)) {
      std::ifstream input(path_);
      std::string text((std::istreambuf_iterator<char>(input)),
                       std::istreambuf_iterator<char>());
      std::array<int, kNpuCoreCount> loads{};
      if (input.good() || input.eof()) {
        if (parse_load(text, loads)) {
          source_available_ = true;
          const int stage = stage_.load(std::memory_order_acquire);
          if (stage == static_cast<int>(NpuStage::kEncoder) ||
              stage == static_cast<int>(NpuStage::kDecoder)) {
            NpuLoadStats& stats =
                stats_[stage == static_cast<int>(NpuStage::kEncoder) ? 0 : 1];
            ++stats.samples;
            for (int core = 0; core < kNpuCoreCount; ++core) {
              stats.sums[core] += loads[core];
              stats.maxima[core] = std::max(stats.maxima[core], loads[core]);
            }
          }
        }
      }
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kNpuLoadSampleIntervalMs));
    }
  }

  std::string path_;
  std::atomic<bool> running_{false};
  std::atomic<int> stage_{static_cast<int>(NpuStage::kIdle)};
  std::thread worker_;
  bool source_available_ = false;
  std::array<NpuLoadStats, 2> stats_{};
};

struct Model {
  rknn_context context = 0;
  std::vector<rknn_tensor_attr> inputs;
  std::vector<rknn_tensor_attr> outputs;

  Model() = default;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  ~Model() {
    if (context != 0) {
      rknn_destroy(context);
    }
  }

  void init(const char* path, uint32_t flags, rknn_core_mask core_mask) {
    check_rknn(rknn_init(&context, const_cast<char*>(path), 0, flags, nullptr),
               "rknn_init");
    if (core_mask != RKNN_NPU_CORE_AUTO) {
      check_rknn(rknn_set_core_mask(context, core_mask), "rknn_set_core_mask");
    }

    rknn_input_output_num count{};
    check_rknn(rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &count, sizeof(count)),
               "RKNN_QUERY_IN_OUT_NUM");
    inputs.resize(count.n_input);
    outputs.resize(count.n_output);
    for (uint32_t i = 0; i < count.n_input; ++i) {
      inputs[i].index = i;
      check_rknn(rknn_query(context, RKNN_QUERY_INPUT_ATTR, &inputs[i],
                            sizeof(inputs[i])),
                 "RKNN_QUERY_INPUT_ATTR");
    }
    for (uint32_t i = 0; i < count.n_output; ++i) {
      outputs[i].index = i;
      check_rknn(rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &outputs[i],
                            sizeof(outputs[i])),
                 "RKNN_QUERY_OUTPUT_ATTR");
    }
  }
};

struct OwnedMemory {
  rknn_context context = 0;
  rknn_tensor_mem* memory = nullptr;

  OwnedMemory() = default;
  OwnedMemory(const OwnedMemory&) = delete;
  OwnedMemory& operator=(const OwnedMemory&) = delete;

  OwnedMemory(OwnedMemory&& other) noexcept
      : context(other.context), memory(other.memory) {
    other.context = 0;
    other.memory = nullptr;
  }

  OwnedMemory& operator=(OwnedMemory&& other) noexcept {
    if (this != &other) {
      reset();
      context = other.context;
      memory = other.memory;
      other.context = 0;
      other.memory = nullptr;
    }
    return *this;
  }

  ~OwnedMemory() { reset(); }

  void reset() {
    if (memory != nullptr) {
      rknn_destroy_mem(context, memory);
      memory = nullptr;
      context = 0;
    }
  }
};

OwnedMemory create_memory(rknn_context context, uint32_t size) {
  OwnedMemory result;
  result.context = context;
  result.memory = rknn_create_mem(context, size);
  if (result.memory == nullptr) {
    throw std::runtime_error("rknn_create_mem failed for " +
                             std::to_string(size) + " bytes");
  }
  return result;
}

OwnedMemory import_memory(rknn_context context, rknn_tensor_mem* source) {
  OwnedMemory result;
  result.context = context;
  result.memory = rknn_create_mem_from_fd(context, source->fd, source->virt_addr,
                                          source->size, source->offset);
  if (result.memory == nullptr) {
    throw std::runtime_error("rknn_create_mem_from_fd failed");
  }
  return result;
}

void bind_memory(Model& model, OwnedMemory& memory, rknn_tensor_attr& attr) {
  check_rknn(rknn_set_io_mem(model.context, memory.memory, &attr),
             "rknn_set_io_mem");
}

double hz_to_mel(double hz) {
  constexpr double kFrequencySpacing = 200.0 / 3.0;
  constexpr double kMinLogHz = 1000.0;
  constexpr double kMinLogMel = kMinLogHz / kFrequencySpacing;
  constexpr double kLogStep = 0.06875177742094912;  // log(6.4) / 27
  if (hz >= kMinLogHz) {
    return kMinLogMel + std::log(hz / kMinLogHz) / kLogStep;
  }
  return hz / kFrequencySpacing;
}

double mel_to_hz(double mel) {
  constexpr double kFrequencySpacing = 200.0 / 3.0;
  constexpr double kMinLogHz = 1000.0;
  constexpr double kMinLogMel = kMinLogHz / kFrequencySpacing;
  constexpr double kLogStep = 0.06875177742094912;  // log(6.4) / 27
  if (mel >= kMinLogMel) {
    return kMinLogHz * std::exp(kLogStep * (mel - kMinLogMel));
  }
  return kFrequencySpacing * mel;
}

std::vector<float> make_slaney_mel_filters() {
  std::array<double, kMels + 2> mel_frequencies{};
  const double mel_min = hz_to_mel(0.0);
  const double mel_max = hz_to_mel(kSampleRate / 2.0);
  for (int i = 0; i < kMels + 2; ++i) {
    const double mel = mel_min + (mel_max - mel_min) * i / (kMels + 1);
    mel_frequencies[i] = mel_to_hz(mel);
  }

  std::vector<float> filters(kMels * kBins, 0.0f);
  for (int mel = 0; mel < kMels; ++mel) {
    const double lower = mel_frequencies[mel];
    const double center = mel_frequencies[mel + 1];
    const double upper = mel_frequencies[mel + 2];
    const double normalization = 2.0 / (upper - lower);
    for (int bin = 0; bin < kBins; ++bin) {
      const double frequency = (kSampleRate / 2.0) * bin / (kBins - 1);
      const double lower_slope = (frequency - lower) / (center - lower);
      const double upper_slope = (upper - frequency) / (upper - center);
      filters[mel * kBins + bin] = static_cast<float>(
          normalization * std::max(0.0, std::min(lower_slope, upper_slope)));
    }
  }
  return filters;
}

std::vector<float> extract_features(const audio_buffer_t& audio) {
  if (audio.num_channels != 1 || audio.sample_rate != kSampleRate) {
    throw std::runtime_error("audio must be mono 16 kHz after conversion");
  }

  std::vector<float> samples(kSamples, 0.0f);
  const int copy_count = std::min(audio.num_frames, kSamples);
  std::copy(audio.data, audio.data + copy_count, samples.begin());

  constexpr int kPad = kFftSize / 2;
  std::vector<float> padded(kSamples + 2 * kPad, 0.0f);
  std::copy(samples.begin(), samples.end(), padded.begin() + kPad);
  for (int i = 0; i < kPad; ++i) {
    padded[i] = samples[kPad - i];
    padded[kPad + kSamples + i] = samples[kSamples - 2 - i];
  }

  std::vector<float> window(kFftSize);
  for (int i = 0; i < kFftSize; ++i) {
    window[i] = static_cast<float>(0.5 -
        0.5 * std::cos(2.0 * M_PI * i / static_cast<double>(kFftSize)));
  }

  std::vector<float> fft_input(kFftSize);
  fftwf_complex* fft_output = static_cast<fftwf_complex*>(
      fftwf_malloc(sizeof(fftwf_complex) * kBins));
  if (fft_output == nullptr) {
    throw std::bad_alloc();
  }
  fftwf_plan plan = fftwf_plan_dft_r2c_1d(kFftSize, fft_input.data(),
                                           fft_output, FFTW_ESTIMATE);
  if (plan == nullptr) {
    fftwf_free(fft_output);
    throw std::runtime_error("fftwf_plan_dft_r2c_1d failed");
  }

  const std::vector<float> filters = make_slaney_mel_filters();
  std::vector<float> features(kMels * kFrames, 0.0f);
  std::array<float, kBins> power{};
  for (int frame = 0; frame < kFrames; ++frame) {
    const int offset = frame * kHopLength;
    for (int i = 0; i < kFftSize; ++i) {
      fft_input[i] = padded[offset + i] * window[i];
    }
    fftwf_execute(plan);
    for (int bin = 0; bin < kBins; ++bin) {
      power[bin] = fft_output[bin][0] * fft_output[bin][0] +
                   fft_output[bin][1] * fft_output[bin][1];
    }
    for (int mel = 0; mel < kMels; ++mel) {
      float sum = 0.0f;
      const float* filter = filters.data() + mel * kBins;
      for (int bin = 0; bin < kBins; ++bin) {
        sum += filter[bin] * power[bin];
      }
      features[mel * kFrames + frame] = sum;
    }
  }
  fftwf_destroy_plan(plan);
  fftwf_free(fft_output);

  float maximum = -std::numeric_limits<float>::infinity();
  for (float& value : features) {
    value = std::log10(std::max(value, 1.0e-10f));
    maximum = std::max(maximum, value);
  }
  const float floor = maximum - 8.0f;
  for (float& value : features) {
    value = (std::max(value, floor) + 4.0f) / 4.0f;
  }
  return features;
}

void append_utf8(uint32_t codepoint, std::string& output) {
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

uint32_t parse_hex4(const std::string& text, size_t& position) {
  uint32_t value = 0;
  for (int i = 0; i < 4; ++i) {
    if (position >= text.size()) {
      throw std::runtime_error("truncated JSON unicode escape");
    }
    const char c = text[position++];
    value <<= 4;
    if (c >= '0' && c <= '9') value |= c - '0';
    else if (c >= 'a' && c <= 'f') value |= c - 'a' + 10;
    else if (c >= 'A' && c <= 'F') value |= c - 'A' + 10;
    else throw std::runtime_error("invalid JSON unicode escape");
  }
  return value;
}

std::string parse_json_string(const std::string& text, size_t& position) {
  if (position >= text.size() || text[position++] != '"') {
    throw std::runtime_error("expected JSON string");
  }
  std::string result;
  while (position < text.size()) {
    const unsigned char c = text[position++];
    if (c == '"') return result;
    if (c != '\\') {
      result.push_back(static_cast<char>(c));
      continue;
    }
    if (position >= text.size()) throw std::runtime_error("truncated JSON escape");
    const char escaped = text[position++];
    switch (escaped) {
      case '"': result.push_back('"'); break;
      case '\\': result.push_back('\\'); break;
      case '/': result.push_back('/'); break;
      case 'b': result.push_back('\b'); break;
      case 'f': result.push_back('\f'); break;
      case 'n': result.push_back('\n'); break;
      case 'r': result.push_back('\r'); break;
      case 't': result.push_back('\t'); break;
      case 'u': {
        uint32_t codepoint = parse_hex4(text, position);
        if (codepoint >= 0xd800 && codepoint <= 0xdbff &&
            position + 2 <= text.size() && text[position] == '\\' &&
            text[position + 1] == 'u') {
          position += 2;
          const uint32_t low = parse_hex4(text, position);
          if (low < 0xdc00 || low > 0xdfff) {
            throw std::runtime_error("invalid JSON surrogate pair");
          }
          codepoint = 0x10000 + ((codepoint - 0xd800) << 10) +
                      (low - 0xdc00);
        }
        append_utf8(codepoint, result);
        break;
      }
      default: throw std::runtime_error("unsupported JSON escape");
    }
  }
  throw std::runtime_error("unterminated JSON string");
}

uint32_t next_utf8_codepoint(const std::string& text, size_t& position) {
  const uint8_t first = static_cast<uint8_t>(text[position++]);
  if ((first & 0x80) == 0) return first;
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
    throw std::runtime_error("invalid UTF-8 in vocabulary");
  }
  while (remaining-- > 0) {
    if (position >= text.size()) throw std::runtime_error("truncated UTF-8");
    const uint8_t next = static_cast<uint8_t>(text[position++]);
    if ((next & 0xc0) != 0x80) throw std::runtime_error("invalid UTF-8 continuation");
    codepoint = (codepoint << 6) | (next & 0x3f);
  }
  return codepoint;
}

std::unordered_map<uint32_t, uint8_t> byte_decoder() {
  std::array<bool, 256> direct{};
  for (int b = 33; b <= 126; ++b) direct[b] = true;
  for (int b = 161; b <= 172; ++b) direct[b] = true;
  for (int b = 174; b <= 255; ++b) direct[b] = true;

  std::unordered_map<uint32_t, uint8_t> result;
  for (int b = 0; b < 256; ++b) {
    if (direct[b]) result[static_cast<uint32_t>(b)] = static_cast<uint8_t>(b);
  }
  uint32_t synthetic = 256;
  for (int b = 0; b < 256; ++b) {
    if (!direct[b]) result[synthetic++] = static_cast<uint8_t>(b);
  }
  return result;
}

std::string token_unicode_to_bytes(const std::string& token) {
  static const auto decoder = byte_decoder();
  std::string result;
  size_t position = 0;
  while (position < token.size()) {
    const uint32_t codepoint = next_utf8_codepoint(token, position);
    const auto found = decoder.find(codepoint);
    if (found == decoder.end()) {
      throw std::runtime_error("vocabulary contains an unmapped codepoint");
    }
    result.push_back(static_cast<char>(found->second));
  }
  return result;
}

void skip_whitespace(const std::string& text, size_t& position) {
  while (position < text.size() &&
         (text[position] == ' ' || text[position] == '\n' ||
          text[position] == '\r' || text[position] == '\t')) {
    ++position;
  }
}

std::vector<std::string> load_vocabulary(const char* path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error(std::string("cannot open ") + path);
  const std::string text((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
  std::vector<std::string> vocabulary(50257);
  size_t position = 0;
  skip_whitespace(text, position);
  if (position >= text.size() || text[position++] != '{') {
    throw std::runtime_error("vocab.json must contain an object");
  }
  while (true) {
    skip_whitespace(text, position);
    if (position < text.size() && text[position] == '}') break;
    const std::string token = parse_json_string(text, position);
    skip_whitespace(text, position);
    if (position >= text.size() || text[position++] != ':') {
      throw std::runtime_error("expected ':' in vocab.json");
    }
    skip_whitespace(text, position);
    int id = 0;
    if (position >= text.size() || text[position] < '0' || text[position] > '9') {
      throw std::runtime_error("expected token id in vocab.json");
    }
    while (position < text.size() && text[position] >= '0' &&
           text[position] <= '9') {
      id = id * 10 + (text[position++] - '0');
    }
    if (id >= 0 && id < static_cast<int>(vocabulary.size())) {
      vocabulary[id] = token_unicode_to_bytes(token);
    }
    skip_whitespace(text, position);
    if (position < text.size() && text[position] == ',') {
      ++position;
      continue;
    }
    if (position < text.size() && text[position] == '}') break;
    throw std::runtime_error("expected ',' or '}' in vocab.json");
  }
  return vocabulary;
}

constexpr std::array<int, 88> kSuppressTokens = {
    1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61,
    62, 63, 90, 91, 92, 93, 359, 503, 522, 542, 873, 893, 902, 918,
    922, 931, 1350, 1853, 1982, 2460, 2627, 3246, 3253, 3268, 3536,
    3846, 3961, 4183, 4667, 6585, 6647, 7273, 9061, 9383, 10428,
    10929, 11938, 12033, 12331, 12562, 13793, 14157, 14635, 15265,
    15618, 16553, 16604, 18362, 18956, 20075, 21675, 22520, 26130,
    26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865, 42863,
    47425, 49870, 50254, 50258, 50359, 50360, 50361, 50362, 50363};

bool suppressed(int token, int decode_index) {
  if (std::find(kSuppressTokens.begin(), kSuppressTokens.end(), token) !=
      kSuppressTokens.end()) {
    return true;
  }
  return decode_index == 0 && (token == 220 || token == kEos);
}

int argmax_logits(const llmc_float16* logits, int decode_index) {
  int best_token = -1;
  float best_value = -std::numeric_limits<float>::infinity();
  for (int token = 0; token < kVocabSize; ++token) {
    if (suppressed(token, decode_index)) continue;
    const float value = static_cast<float>(logits[token]);
    if (value > best_value) {
      best_value = value;
      best_token = token;
    }
  }
  return best_token;
}

rknn_core_mask parse_core_mask(const std::string& text) {
  if (text == "auto") return RKNN_NPU_CORE_AUTO;
  if (text == "0") return RKNN_NPU_CORE_0;
  if (text == "1") return RKNN_NPU_CORE_1;
  if (text == "2") return RKNN_NPU_CORE_2;
  if (text == "0,1") return RKNN_NPU_CORE_0_1;
  if (text == "0,1,2") return RKNN_NPU_CORE_0_1_2;
  if (text == "all") return RKNN_NPU_CORE_ALL;
  throw std::runtime_error("invalid core mask: " + text);
}

int configured_core_count(const std::string& text) {
  if (text == "0" || text == "1" || text == "2") return 1;
  if (text == "0,1") return 2;
  if (text == "0,1,2" || text == "all") return 3;
  return 0;
}

void validate_models(const Model& encoder, const Model& decoder) {
  if (encoder.inputs.size() != 1 || encoder.outputs.size() != 3 ||
      decoder.inputs.size() != 6 || decoder.outputs.size() != 3) {
    throw std::runtime_error("unexpected encoder/decoder I/O count");
  }
  if (encoder.inputs[0].size != kMels * kFrames * sizeof(llmc_float16) ||
      decoder.outputs[0].size != kVocabSize * sizeof(llmc_float16)) {
    throw std::runtime_error("model tensor shapes do not match Whisper large-v3-turbo");
  }
  if (encoder.outputs[1].size != decoder.inputs[1].size ||
      encoder.outputs[2].size != decoder.inputs[2].size ||
      decoder.inputs[3].size != decoder.outputs[1].size ||
      decoder.inputs[4].size != decoder.outputs[2].size) {
    throw std::runtime_error("KV cache tensor sizes are incompatible");
  }
}

struct AudioOwner {
  audio_buffer_t buffer{};
  ~AudioOwner() { std::free(buffer.data); }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5 || argc > 7) {
    std::fprintf(stderr,
                 "usage: %s ENCODER.rknn DECODER.rknn vocab.json AUDIO.wav "
                 "[--core-mask auto|0|1|2|0,1|0,1,2|all]\n",
                 argv[0]);
    return 2;
  }

  try {
    std::string core_text = "auto";
    if (argc == 7) {
      if (std::string(argv[5]) != "--core-mask") {
        throw std::runtime_error("expected --core-mask");
      }
      core_text = argv[6];
    }
    const rknn_core_mask core_mask = parse_core_mask(core_text);

    const auto vocabulary = load_vocabulary(argv[3]);

    // LLMC keeps all KV tensors in DMA memory. CPU cache maintenance is only
    // performed for the feature input, token/position scalars, and logits.
    constexpr uint32_t kEncoderFlags =
        RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE |
        RKNN_FLAG_DISABLE_FLUSH_OUTPUT_MEM_CACHE;
    constexpr uint32_t kDecoderFlags =
        RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE |
        RKNN_FLAG_DISABLE_FLUSH_OUTPUT_MEM_CACHE;

    const auto model_init_start = Clock::now();
    Model encoder;
    encoder.init(argv[1], kEncoderFlags, core_mask);
    Model decoder;
    decoder.init(argv[2], kDecoderFlags, core_mask);
    validate_models(encoder, decoder);

    auto encoder_input = create_memory(encoder.context, encoder.inputs[0].size);
    auto encoder_hidden = create_memory(encoder.context, encoder.outputs[0].size);
    auto encoder_k = create_memory(encoder.context, encoder.outputs[1].size);
    auto encoder_v = create_memory(encoder.context, encoder.outputs[2].size);
    bind_memory(encoder, encoder_input, encoder.inputs[0]);
    bind_memory(encoder, encoder_hidden, encoder.outputs[0]);
    bind_memory(encoder, encoder_k, encoder.outputs[1]);
    bind_memory(encoder, encoder_v, encoder.outputs[2]);

    auto input_ids = create_memory(decoder.context, decoder.inputs[0].size);
    auto decoder_encoder_k = import_memory(decoder.context, encoder_k.memory);
    auto decoder_encoder_v = import_memory(decoder.context, encoder_v.memory);
    auto self_k = create_memory(decoder.context, decoder.inputs[3].size);
    auto self_v = create_memory(decoder.context, decoder.inputs[4].size);
    auto cache_position = create_memory(decoder.context, decoder.inputs[5].size);
    auto logits = create_memory(decoder.context, decoder.outputs[0].size);
    bind_memory(decoder, input_ids, decoder.inputs[0]);
    bind_memory(decoder, decoder_encoder_k, decoder.inputs[1]);
    bind_memory(decoder, decoder_encoder_v, decoder.inputs[2]);
    bind_memory(decoder, self_k, decoder.inputs[3]);
    bind_memory(decoder, self_v, decoder.inputs[4]);
    bind_memory(decoder, cache_position, decoder.inputs[5]);
    bind_memory(decoder, logits, decoder.outputs[0]);
    bind_memory(decoder, self_k, decoder.outputs[1]);
    bind_memory(decoder, self_v, decoder.outputs[2]);

    std::memset(self_k.memory->virt_addr, 0, self_k.memory->size);
    std::memset(self_v.memory->virt_addr, 0, self_v.memory->size);
    check_rknn(rknn_mem_sync(decoder.context, self_k.memory,
                             RKNN_MEMORY_SYNC_TO_DEVICE),
               "sync self_k");
    check_rknn(rknn_mem_sync(decoder.context, self_v.memory,
                             RKNN_MEMORY_SYNC_TO_DEVICE),
               "sync self_v");
    const auto model_init_end = Clock::now();

    const auto inference_start = Clock::now();
    AudioOwner audio;
    if (read_audio(argv[4], &audio.buffer) != 0) {
      throw std::runtime_error("read_audio failed");
    }
    const double audio_seconds =
        audio.buffer.num_frames / static_cast<double>(audio.buffer.sample_rate);
    if (audio.buffer.num_channels == 2 && convert_channels(&audio.buffer) != 0) {
      throw std::runtime_error("convert_channels failed");
    }
    if (audio.buffer.sample_rate != kSampleRate &&
        resample_audio(&audio.buffer, audio.buffer.sample_rate, kSampleRate) != 0) {
      throw std::runtime_error("resample_audio failed");
    }
    const std::vector<float> features = extract_features(audio.buffer);
    auto* feature_half = static_cast<llmc_float16*>(encoder_input.memory->virt_addr);
    for (size_t i = 0; i < features.size(); ++i) {
      feature_half[i] = static_cast<llmc_float16>(features[i]);
    }
    check_rknn(rknn_mem_sync(encoder.context, encoder_input.memory,
                             RKNN_MEMORY_SYNC_TO_DEVICE),
               "sync encoder input");
    const auto preprocess_end = Clock::now();

    NpuLoadMonitor npu_load_monitor(kNpuLoadPath);
    npu_load_monitor.start();
    const auto encoder_start = Clock::now();
    npu_load_monitor.set_stage(NpuStage::kEncoder);
    check_rknn(rknn_run(encoder.context, nullptr), "encoder rknn_run");
    npu_load_monitor.set_stage(NpuStage::kIdle);
    const auto encoder_end = Clock::now();

    // The encoder K/V buffers stay device-resident and are imported into the
    // decoder by DMA-BUF fd. No 30.72 MB host copy occurs here.
    std::vector<int> tokens = {kSot, kEnglish, kTranscribe, kNoTimestamps};
    std::vector<double> decoder_latencies;
    auto decoder_step = [&](int token, int64_t position) {
      *static_cast<int64_t*>(input_ids.memory->virt_addr) = token;
      *static_cast<int64_t*>(cache_position.memory->virt_addr) = position;
      check_rknn(rknn_mem_sync(decoder.context, input_ids.memory,
                               RKNN_MEMORY_SYNC_TO_DEVICE),
                 "sync input_ids");
      check_rknn(rknn_mem_sync(decoder.context, cache_position.memory,
                               RKNN_MEMORY_SYNC_TO_DEVICE),
                 "sync cache_position");
      const auto start = Clock::now();
      npu_load_monitor.set_stage(NpuStage::kDecoder);
      check_rknn(rknn_run(decoder.context, nullptr), "decoder rknn_run");
      npu_load_monitor.set_stage(NpuStage::kIdle);
      check_rknn(rknn_mem_sync(decoder.context, logits.memory,
                               RKNN_MEMORY_SYNC_FROM_DEVICE),
                 "sync logits");
      const auto end = Clock::now();
      decoder_latencies.push_back(milliseconds(start, end));
    };

    for (size_t step = 0; step < tokens.size(); ++step) {
      decoder_step(tokens[step], static_cast<int64_t>(step));
    }

    for (int decode_index = 0; decode_index < kMaxNewTokens; ++decode_index) {
      const int next = argmax_logits(
          static_cast<const llmc_float16*>(logits.memory->virt_addr), decode_index);
      if (next < 0) throw std::runtime_error("argmax produced no token");
      tokens.push_back(next);
      if (next == kEos) break;
      const int64_t position = static_cast<int64_t>(tokens.size() - 1);
      if (position >= 448) break;
      decoder_step(next, position);
    }
    const auto inference_end = Clock::now();
    npu_load_monitor.stop();

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

    double decoder_total_ms = 0.0;
    for (double value : decoder_latencies) decoder_total_ms += value;
    const int generated_tokens = static_cast<int>(tokens.size()) - 4;
    std::printf("audio_seconds: %.2f\n", audio_seconds);
    std::printf("generated_tokens: %d\n", generated_tokens);
    std::printf("model_init_ms: %.2f\n",
                milliseconds(model_init_start, model_init_end));
    std::printf("preprocess_ms: %.2f\n",
                milliseconds(inference_start, preprocess_end));
    std::printf("encoder_ms: %.2f\n",
                milliseconds(encoder_start, encoder_end));
    std::printf("decoder_steps: %zu\n", decoder_latencies.size());
    std::printf("avg_decoder_step_ms: %.2f\n",
                decoder_latencies.empty()
                    ? 0.0
                    : decoder_total_ms / decoder_latencies.size());
    std::printf("inference_ms: %.2f\n",
                milliseconds(inference_start, inference_end));
    std::printf("zero_copy_decoder: true\n");
    std::printf("runtime: native-cpp\n");
    std::printf("core_mask: %s\n", core_text.c_str());
    if (core_text == "auto") {
      std::printf("npu_configured_cores: auto\n");
    } else {
      std::printf("npu_configured_cores: %d\n",
                  configured_core_count(core_text));
    }
    npu_load_monitor.print();
    std::printf("text:\n%s\n", text.c_str());
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
  return 0;
}
