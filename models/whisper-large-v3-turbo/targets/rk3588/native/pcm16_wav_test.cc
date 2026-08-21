#include "pcm16_wav.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s std30.wav\n", argv[0]);
    return 2;
  }
  try {
    const auto audio = llmc::read_mono_pcm16_wav(argv[1], 16000);
    if (audio.samples.size() != 436320) {
      throw std::runtime_error("std30.wav must contain 436320 samples");
    }
    double checksum = 0.0;
    for (size_t index = 0; index < audio.samples.size(); ++index) {
      checksum += audio.samples[index] * static_cast<double>((index % 251) + 1);
    }
    if (std::abs(checksum - 1373.705871582031) > 1.0e-9) {
      throw std::runtime_error("std30.wav PCM checksum mismatch");
    }
    std::printf("sample_rate: %d\n", audio.sample_rate);
    std::printf("samples: %zu\n", audio.samples.size());
    std::printf("pcm_checksum: %.12f\n", checksum);
    std::printf("wav_validation: passed\n");
  } catch (const std::exception &error) {
    std::fprintf(stderr, "wav_validation: failed: %s\n", error.what());
    return 1;
  }
  return 0;
}
