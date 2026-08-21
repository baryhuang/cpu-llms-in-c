#include "pcm16_wav.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace llmc {
namespace {

uint16_t little_u16(const uint8_t *bytes) {
  return static_cast<uint16_t>(bytes[0]) |
         (static_cast<uint16_t>(bytes[1]) << 8);
}

uint32_t little_u32(const uint8_t *bytes) {
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8) |
         (static_cast<uint32_t>(bytes[2]) << 16) |
         (static_cast<uint32_t>(bytes[3]) << 24);
}

} // namespace

PcmAudio read_mono_pcm16_wav(const char *path, int required_sample_rate) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error(std::string("cannot open ") + path);
  const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(input)),
                                   std::istreambuf_iterator<char>());
  if (bytes.size() < 12 || std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
    throw std::runtime_error("audio is not a RIFF/WAVE file");
  }
  uint16_t format = 0;
  uint16_t channels = 0;
  uint16_t bits_per_sample = 0;
  uint32_t sample_rate = 0;
  const uint8_t *pcm = nullptr;
  uint32_t pcm_bytes = 0;
  size_t cursor = 12;
  while (cursor + 8 <= bytes.size()) {
    const uint8_t *header = bytes.data() + cursor;
    const uint32_t chunk_size = little_u32(header + 4);
    cursor += 8;
    if (chunk_size > bytes.size() - cursor) {
      throw std::runtime_error("truncated WAV chunk");
    }
    if (std::memcmp(header, "fmt ", 4) == 0) {
      if (chunk_size < 16)
        throw std::runtime_error("short WAV fmt chunk");
      format = little_u16(bytes.data() + cursor);
      channels = little_u16(bytes.data() + cursor + 2);
      sample_rate = little_u32(bytes.data() + cursor + 4);
      bits_per_sample = little_u16(bytes.data() + cursor + 14);
    } else if (std::memcmp(header, "data", 4) == 0) {
      pcm = bytes.data() + cursor;
      pcm_bytes = chunk_size;
    }
    cursor += chunk_size + (chunk_size & 1u);
  }
  if (format != 1 || channels != 1 ||
      sample_rate != static_cast<uint32_t>(required_sample_rate) ||
      bits_per_sample != 16 || pcm == nullptr || pcm_bytes % 2 != 0) {
    throw std::runtime_error(
        "runner requires mono PCM16 WAV at the configured sample rate");
  }
  PcmAudio audio;
  audio.sample_rate = static_cast<int>(sample_rate);
  audio.samples.resize(pcm_bytes / 2);
  for (size_t index = 0; index < audio.samples.size(); ++index) {
    const int16_t sample = static_cast<int16_t>(little_u16(pcm + index * 2));
    audio.samples[index] = sample / 32768.0f;
  }
  return audio;
}

} // namespace llmc
