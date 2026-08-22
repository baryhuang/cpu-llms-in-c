#include <sched.h>

#define main whisper_rknpu2_legacy_main
#include "whisper_rknpu2_llmc.cc"
#undef main

#include "rkllm.h"

namespace {

#ifndef LLMC_RKLLM_OPTIMIZED
#define LLMC_RKLLM_OPTIMIZED 0
#endif

constexpr bool kOptimized = LLMC_RKLLM_OPTIMIZED != 0;
constexpr int kDecoderLayers = 4;
constexpr int kEncoderTokens = 1500;
constexpr int kAttentionHeads = 20;
constexpr int kHeadDim = 64;
constexpr size_t kCrossElements =
    static_cast<size_t>(kDecoderLayers) * kEncoderTokens *
    kAttentionHeads * kHeadDim;
constexpr char kAuxMagic[8] = {'W', 'R', 'K', 'L', 'A', 'U', 'X', '1'};

struct AuxHeader {
  char magic[8];
  uint32_t version;
  uint32_t vocabulary;
  uint32_t positions;
  uint32_t hidden;
  uint32_t logit_bias;
};
static_assert(sizeof(AuxHeader) == 28, "unexpected auxiliary header size");

class AuxData {
 public:
  explicit AuxData(const char* path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error(std::string("cannot open ") + path);
    bytes_.assign(std::istreambuf_iterator<char>(input),
                  std::istreambuf_iterator<char>());
    if (bytes_.size() < sizeof(AuxHeader)) {
      throw std::runtime_error("truncated RKLLM auxiliary data");
    }
    std::memcpy(&header_, bytes_.data(), sizeof(header_));
    if (std::memcmp(header_.magic, kAuxMagic, sizeof(kAuxMagic)) != 0 ||
        header_.version != 1 || header_.vocabulary != kVocabSize ||
        header_.positions != 448 || header_.hidden != 1280 ||
        header_.logit_bias != kVocabSize) {
      throw std::runtime_error("invalid RKLLM auxiliary header");
    }
    const size_t token_count =
        static_cast<size_t>(header_.vocabulary) * header_.hidden;
    const size_t position_count =
        static_cast<size_t>(header_.positions) * header_.hidden;
    const size_t expected = sizeof(AuxHeader) +
        (token_count + position_count) * sizeof(llmc_float16) +
        static_cast<size_t>(header_.logit_bias) * sizeof(float);
    if (bytes_.size() != expected) {
      throw std::runtime_error("RKLLM auxiliary size mismatch");
    }
    const auto* base = reinterpret_cast<const unsigned char*>(bytes_.data()) +
                       sizeof(AuxHeader);
    tokens_ = reinterpret_cast<const llmc_float16*>(base);
    positions_ = tokens_ + token_count;
    logit_bias_ = reinterpret_cast<const float*>(positions_ + position_count);
  }

  void embedding(int token, int position, float* output) const {
    if (token < 0 || token >= static_cast<int>(header_.vocabulary) ||
        position < 0 || position >= static_cast<int>(header_.positions)) {
      throw std::runtime_error("embedding index is out of range");
    }
    const llmc_float16* token_row =
        tokens_ + static_cast<size_t>(token) * header_.hidden;
    const llmc_float16* position_row =
        positions_ + static_cast<size_t>(position) * header_.hidden;
    for (uint32_t column = 0; column < header_.hidden; ++column) {
      output[column] = static_cast<float>(token_row[column]) +
                       static_cast<float>(position_row[column]);
    }
  }

  const float* logit_bias() const { return logit_bias_; }
  int hidden() const { return static_cast<int>(header_.hidden); }

 private:
  std::vector<char> bytes_;
  AuxHeader header_{};
  const llmc_float16* tokens_ = nullptr;
  const llmc_float16* positions_ = nullptr;
  const float* logit_bias_ = nullptr;
};

float read_tensor_element(const rknn_tensor_attr& attr, const void* data,
                          size_t index) {
  switch (attr.type) {
    case RKNN_TENSOR_FLOAT32:
      return static_cast<const float*>(data)[index];
    case RKNN_TENSOR_FLOAT16:
      return static_cast<float>(static_cast<const llmc_float16*>(data)[index]);
    case RKNN_TENSOR_INT8:
      return (static_cast<int>(static_cast<const int8_t*>(data)[index]) -
              attr.zp) * attr.scale;
    case RKNN_TENSOR_UINT8:
      return (static_cast<int>(static_cast<const uint8_t*>(data)[index]) -
              attr.zp) * attr.scale;
    default:
      throw std::runtime_error("unsupported encoder tensor type");
  }
}

void write_tensor_elements(const rknn_tensor_attr& attr, void* data,
                           const std::vector<float>& values) {
  if (attr.n_elems != values.size()) {
    throw std::runtime_error("encoder input element count mismatch");
  }
  switch (attr.type) {
    case RKNN_TENSOR_FLOAT32:
      std::copy(values.begin(), values.end(), static_cast<float*>(data));
      return;
    case RKNN_TENSOR_FLOAT16: {
      auto* output = static_cast<llmc_float16*>(data);
      for (size_t index = 0; index < values.size(); ++index) {
        output[index] = static_cast<llmc_float16>(values[index]);
      }
      return;
    }
    case RKNN_TENSOR_INT8: {
      auto* output = static_cast<int8_t*>(data);
      for (size_t index = 0; index < values.size(); ++index) {
        const int quantized = static_cast<int>(
            std::lrint(values[index] / attr.scale + attr.zp));
        output[index] = static_cast<int8_t>(
            std::max(-128, std::min(127, quantized)));
      }
      return;
    }
    case RKNN_TENSOR_UINT8: {
      auto* output = static_cast<uint8_t*>(data);
      for (size_t index = 0; index < values.size(); ++index) {
        const int quantized = static_cast<int>(
            std::lrint(values[index] / attr.scale + attr.zp));
        output[index] = static_cast<uint8_t>(
            std::max(0, std::min(255, quantized)));
      }
      return;
    }
    default:
      throw std::runtime_error("unsupported encoder input type");
  }
}

void pack_cross_cache(const rknn_tensor_attr& key_attr, const void* key_data,
                      const rknn_tensor_attr& value_attr,
                      const void* value_data, std::vector<float>& key,
                      std::vector<float>& value) {
  if (key_attr.n_elems != kCrossElements ||
      value_attr.n_elems != kCrossElements) {
    throw std::runtime_error("unexpected encoder cross-cache shape");
  }
  key.resize(kCrossElements);
  value.resize(kCrossElements);
  for (int layer = 0; layer < kDecoderLayers; ++layer) {
    for (int head = 0; head < kAttentionHeads; ++head) {
      for (int token = 0; token < kEncoderTokens; ++token) {
        for (int dimension = 0; dimension < kHeadDim; ++dimension) {
          const size_t source =
              (((static_cast<size_t>(layer) * kAttentionHeads + head) *
                 kEncoderTokens + token) * kHeadDim + dimension);
          const size_t key_target =
              (((static_cast<size_t>(layer) * kEncoderTokens + token) *
                 kAttentionHeads + head) * kHeadDim + dimension);
          const size_t value_target =
              (((static_cast<size_t>(layer) * kAttentionHeads + head) *
                 kHeadDim + dimension) * kEncoderTokens + token);
          key[key_target] = read_tensor_element(key_attr, key_data, source);
          value[value_target] =
              read_tensor_element(value_attr, value_data, source);
        }
      }
    }
  }
}

struct CrossCache {
  std::vector<float> key;
  std::vector<float> value;
};

CrossCache run_encoder_baseline(Model& encoder,
                                const std::vector<float>& features) {
  rknn_input input{};
  input.index = 0;
  input.buf = const_cast<float*>(features.data());
  input.size = features.size() * sizeof(float);
  input.pass_through = 0;
  input.type = RKNN_TENSOR_FLOAT32;
  input.fmt = RKNN_TENSOR_NCHW;
  check_rknn(rknn_inputs_set(encoder.context, 1, &input), "rknn_inputs_set");
  check_rknn(rknn_run(encoder.context, nullptr), "encoder rknn_run");
  std::vector<rknn_output> outputs(encoder.outputs.size());
  for (size_t index = 0; index < outputs.size(); ++index) {
    outputs[index].index = static_cast<uint32_t>(index);
    outputs[index].want_float = 1;
  }
  check_rknn(rknn_outputs_get(encoder.context, outputs.size(), outputs.data(),
                              nullptr),
             "rknn_outputs_get");
  CrossCache cache;
  rknn_tensor_attr float_attr = encoder.outputs[1];
  float_attr.type = RKNN_TENSOR_FLOAT32;
  float_attr.scale = 1.0f;
  float_attr.zp = 0;
  pack_cross_cache(float_attr, outputs[1].buf, float_attr, outputs[2].buf,
                   cache.key, cache.value);
  check_rknn(rknn_outputs_release(encoder.context, outputs.size(),
                                  outputs.data()),
             "rknn_outputs_release");
  return cache;
}

CrossCache run_encoder_llmc(Model& encoder,
                            const std::vector<float>& features) {
  auto input = create_memory(encoder.context, encoder.inputs[0].size_with_stride);
  std::vector<OwnedMemory> outputs;
  outputs.reserve(encoder.outputs.size());
  write_tensor_elements(encoder.inputs[0], input.memory->virt_addr, features);
  bind_memory(encoder, input, encoder.inputs[0]);
  for (auto& attr : encoder.outputs) {
    outputs.push_back(create_memory(encoder.context, attr.size_with_stride));
    bind_memory(encoder, outputs.back(), attr);
  }
  check_rknn(rknn_mem_sync(encoder.context, input.memory,
                           RKNN_MEMORY_SYNC_TO_DEVICE),
             "sync encoder input");
  check_rknn(rknn_run(encoder.context, nullptr), "encoder rknn_run");
  for (size_t index : {size_t{1}, size_t{2}}) {
    check_rknn(rknn_mem_sync(encoder.context, outputs[index].memory,
                             RKNN_MEMORY_SYNC_FROM_DEVICE),
               "sync encoder cross cache");
  }
  CrossCache cache;
  pack_cross_cache(encoder.outputs[1], outputs[1].memory->virt_addr,
                   encoder.outputs[2], outputs[2].memory->virt_addr,
                   cache.key, cache.value);
  return cache;
}

struct DecoderState {
  const AuxData* aux = nullptr;
  int embedding_position = 0;
  std::vector<float> logits;
  bool result_ready = false;
  double reported_generate_ms = 0.0;
};

int embedding_callback(void* userdata, int32_t* tokens, uint64_t num_tokens,
                       void* embed, uint64_t length) {
  auto* state = static_cast<DecoderState*>(userdata);
  const uint64_t expected = num_tokens * state->aux->hidden() * sizeof(float);
  if (length != expected) return -1;
  auto* output = static_cast<float*>(embed);
  try {
    for (uint64_t token = 0; token < num_tokens; ++token) {
      state->aux->embedding(tokens[token],
                            state->embedding_position + static_cast<int>(token),
                            output + token * state->aux->hidden());
    }
  } catch (...) {
    return -1;
  }
  return 0;
}

int tokenizer_callback(void*, const char*, int32_t, int32_t*, int32_t) {
  // Whisper always enters RKLLM with explicit token IDs or embeddings.  The
  // exported custom model deliberately has no text tokenizer, but RKLLM 1.3.0
  // still requires a callback to be registered before it will initialize.
  // Returning an empty token list is therefore the correct no-op contract for
  // this runner; the callback is never used by either inference path below.
  return 0;
}

int result_callback(RKLLMResult* result, void* userdata, LLMCallState state) {
  auto* decoder = static_cast<DecoderState*>(userdata);
  if (state == RKLLM_RUN_ERROR) return 0;
  if (result != nullptr && result->logits.logits != nullptr &&
      result->logits.vocab_size == kVocabSize &&
      result->logits.num_tokens > 0) {
    const float* source = result->logits.logits +
        static_cast<size_t>(result->logits.num_tokens - 1) * kVocabSize;
    if (kOptimized) {
      if (decoder->logits.size() != kVocabSize) {
        decoder->logits.resize(kVocabSize);
      }
    } else {
      decoder->logits = std::vector<float>(kVocabSize);
    }
    std::copy(source, source + kVocabSize, decoder->logits.begin());
    decoder->reported_generate_ms = result->perf.generate_time_ms;
    decoder->result_ready = true;
  }
  if (state == RKLLM_RUN_FINISH && kOptimized) return 2;
  return 0;
}

class RKLLMDecoder {
 public:
  RKLLMDecoder(const char* model_path, const AuxData& aux) {
    state_.aux = &aux;
    if (kOptimized) state_.logits.resize(kVocabSize);
    RKLLMParam params = rkllm_createDefaultParam();
    params.model_path = model_path;
    // RKLLM validates the 1500-token cross-attention cache against this same
    // limit. The decoding loop below still enforces Whisper's 448 positions.
    params.max_context_len = 2048;
    params.max_new_tokens = 1;
    params.top_k = 1;
    params.top_p = 1.0f;
    params.temperature = 0.0f;
    params.skip_special_token = false;
    params.ignore_eos_token = true;
    params.is_async = false;
    params.extend_param.use_cross_attn = 1;
    if (kOptimized) {
      params.extend_param.enabled_cpus_num = 4;
      params.extend_param.enabled_cpus_mask = CPU4 | CPU5 | CPU6 | CPU7;
    }
    RKLLMCallback callback{};
    callback.result_callback = result_callback;
    callback.result_userdata = &state_;
    callback.embed_callback = embedding_callback;
    callback.embed_userdata = &state_;
    callback.tokenizer_callback = tokenizer_callback;
    const int rc = rkllm_init(&handle_, &params, &callback);
    if (rc != 0 || handle_ == nullptr) {
      throw std::runtime_error("rkllm_init failed: " + std::to_string(rc));
    }
  }

  RKLLMDecoder(const RKLLMDecoder&) = delete;
  RKLLMDecoder& operator=(const RKLLMDecoder&) = delete;

  ~RKLLMDecoder() {
    if (handle_ != nullptr) rkllm_destroy(handle_);
  }

  void set_cross_cache(CrossCache& cache) {
    mask_.assign(kEncoderTokens, 1.0f);
    // All-zero positions are intentional: Whisper cross attention has no RoPE.
    positions_.assign(kEncoderTokens, 0);
    cross_params_ = {};
    cross_params_.encoder_k_cache = cache.key.data();
    cross_params_.encoder_v_cache = cache.value.data();
    cross_params_.encoder_mask = mask_.data();
    cross_params_.encoder_pos = positions_.data();
    cross_params_.num_tokens = kEncoderTokens;
    const int rc = rkllm_set_cross_attn_params(handle_, &cross_params_);
    if (rc != 0) {
      throw std::runtime_error("rkllm_set_cross_attn_params failed: " +
                               std::to_string(rc));
    }
  }

  const std::vector<float>& run_tokens(const int32_t* tokens, size_t count,
                                       int position) {
    state_.embedding_position = position;
    state_.result_ready = false;
    RKLLMInput input{};
    input.role = "user";
    input.enable_thinking = false;
    std::vector<float> baseline_embedding;
    std::vector<float>& embedding = kOptimized ? input_embedding_
                                                : baseline_embedding;
    embedding.resize(count * state_.aux->hidden());
    for (size_t index = 0; index < count; ++index) {
      state_.aux->embedding(tokens[index], position + static_cast<int>(index),
                            embedding.data() + index * state_.aux->hidden());
    }
    input.input_type = RKLLM_INPUT_EMBED;
    input.embed_input.embed = embedding.data();
    input.embed_input.n_tokens = count;
    RKLLMInferParam infer{};
    infer.mode = RKLLM_INFER_GET_LOGITS;
    infer.keep_history = 1;
    infer.max_new_tokens = 1;
    const int rc = rkllm_run(handle_, &input, &infer, &state_);
    if (rc != 0 || !state_.result_ready) {
      throw std::runtime_error("rkllm_run/get_logits failed: " +
                               std::to_string(rc));
    }
    return state_.logits;
  }

  double reported_generate_ms() const { return state_.reported_generate_ms; }

 private:
  LLMHandle handle_ = nullptr;
  DecoderState state_;
  std::vector<float> input_embedding_;
  std::vector<float> mask_;
  std::vector<int32_t> positions_;
  // Runtime 1.3.0 retains this object until rkllm_run; its pointer must not
  // refer to a temporary stack object.
  RKLLMCrossAttnParam cross_params_{};
};

int argmax_rkllm(const std::vector<float>& logits, const float* bias,
                 int decode_index) {
  int best_token = -1;
  float best_value = -std::numeric_limits<float>::infinity();
  for (int token = 0; token < kVocabSize; ++token) {
    if (suppressed(token, decode_index)) continue;
    const float value = logits[token] + bias[token];
    if (value > best_value) {
      best_value = value;
      best_token = token;
    }
  }
  return best_token;
}

void pin_optimized_process() {
  if (!kOptimized) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(4, &set);
  CPU_SET(5, &set);
  CPU_SET(6, &set);
  CPU_SET(7, &set);
  if (sched_setaffinity(0, sizeof(set), &set) != 0) {
    throw std::runtime_error("sched_setaffinity failed");
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::fprintf(stderr,
                 "usage: %s ENCODER.rknn DECODER.rkllm runtime_aux.bin "
                 "vocab.json AUDIO.wav\n",
                 argv[0]);
    return 2;
  }
  try {
    pin_optimized_process();
    const auto vocabulary = load_vocabulary(argv[4]);
    const AuxData aux(argv[3]);
    const auto init_start = Clock::now();
    Model encoder;
    const uint32_t encoder_flags = kOptimized
        ? RKNN_FLAG_DISABLE_FLUSH_INPUT_MEM_CACHE |
              RKNN_FLAG_DISABLE_FLUSH_OUTPUT_MEM_CACHE
        : 0;
    encoder.init(argv[1], encoder_flags, RKNN_NPU_CORE_ALL);
    if (encoder.inputs.size() != 1 || encoder.outputs.size() < 3) {
      throw std::runtime_error("unexpected encoder I/O count");
    }
    RKLLMDecoder decoder(argv[2], aux);
    const auto init_end = Clock::now();

    const auto inference_start = Clock::now();
    AudioOwner audio;
    if (read_audio(argv[5], &audio.buffer) != 0) {
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
    const auto preprocess_end = Clock::now();

    NpuLoadMonitor monitor(kNpuLoadPath);
    monitor.start();
    monitor.set_stage(NpuStage::kEncoder);
    const auto encoder_start = Clock::now();
    CrossCache cache = kOptimized
        ? run_encoder_llmc(encoder, features)
        : run_encoder_baseline(encoder, features);
    const auto encoder_end = Clock::now();
    monitor.set_stage(NpuStage::kIdle);
    decoder.set_cross_cache(cache);

    std::vector<int32_t> tokens;
    tokens.reserve(448);
    tokens.insert(tokens.end(), {kSot, kEnglish, kTranscribe, kNoTimestamps});
    std::vector<double> decoder_latencies;
    decoder_latencies.reserve(kMaxNewTokens + 4);
    monitor.set_stage(NpuStage::kDecoder);
    const auto prefix_start = Clock::now();
    const std::vector<float>* logits = nullptr;
    if (kOptimized) {
      logits = &decoder.run_tokens(tokens.data(), tokens.size(), 0);
      decoder_latencies.push_back(milliseconds(prefix_start, Clock::now()));
    } else {
      for (size_t index = 0; index < tokens.size(); ++index) {
        const auto step_start = Clock::now();
        logits = &decoder.run_tokens(&tokens[index], 1, static_cast<int>(index));
        decoder_latencies.push_back(milliseconds(step_start, Clock::now()));
      }
    }
    for (int decode_index = 0; decode_index < kMaxNewTokens; ++decode_index) {
      const int next = argmax_rkllm(*logits, aux.logit_bias(), decode_index);
      if (next < 0) throw std::runtime_error("argmax produced no token");
      tokens.push_back(next);
      if (next == kEos || tokens.size() >= 448) break;
      const auto step_start = Clock::now();
      logits = &decoder.run_tokens(&tokens.back(), 1,
                                   static_cast<int>(tokens.size() - 1));
      decoder_latencies.push_back(milliseconds(step_start, Clock::now()));
    }
    const auto decoder_end = Clock::now();
    monitor.set_stage(NpuStage::kIdle);
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
    for (double latency : decoder_latencies) decoder_total += latency;
    const int generated_tokens = static_cast<int>(tokens.size()) - 4;
    std::vector<unsigned char> seen(kVocabSize, 0);
    int unique_generated_tokens = 0;
    for (size_t index = 4; index < tokens.size(); ++index) {
      const int token = tokens[index];
      if (token >= 0 && token < kVocabSize && seen[token] == 0) {
        seen[token] = 1;
        ++unique_generated_tokens;
      }
    }
    const bool degenerate_output =
        generated_tokens == kMaxNewTokens && unique_generated_tokens <= 1;
    std::printf("implementation: %s\n",
                kOptimized ? "llmc-native-cpp" : "rkllm-native-baseline");
    std::printf("quantization: w8a8\n");
    std::printf("rkllm_npu_cores: 3\n");
    std::printf("rkllm_runtime_compatibility: "
                "skip-unused-no-rope-position-copy\n");
    std::printf("audio_seconds: %.3f\n", audio_seconds);
    std::printf("generated_tokens: %d\n", generated_tokens);
    std::printf("unique_generated_tokens: %d\n", unique_generated_tokens);
    std::printf("recognition_status: %s\n",
                degenerate_output ? "failed-degenerate-output" : "completed");
    std::printf("model_init_ms: %.2f\n", milliseconds(init_start, init_end));
    std::printf("preprocess_ms: %.2f\n",
                milliseconds(inference_start, preprocess_end));
    std::printf("encoder_ms: %.2f\n",
                milliseconds(encoder_start, encoder_end));
    std::printf("decoder_ms: %.2f\n", decoder_total);
    std::printf("decoder_calls: %zu\n", decoder_latencies.size());
    std::printf("avg_decoder_call_ms: %.2f\n",
                decoder_latencies.empty()
                    ? 0.0
                    : decoder_total / decoder_latencies.size());
    std::printf("end_to_end_ms: %.2f\n",
                milliseconds(inference_start, decoder_end));
    std::printf("cross_cache_elements: %zu\n", kCrossElements * 2);
    std::printf("position_embedding: native-callback\n");
    std::printf("decode_mode: get-logits-greedy\n");
    monitor.print();
    std::printf("text:\n%s\n", text.c_str());
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
  return 0;
}
