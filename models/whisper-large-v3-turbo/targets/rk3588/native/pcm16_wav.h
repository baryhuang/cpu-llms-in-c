#pragma once

#include <vector>

namespace llmc {

struct PcmAudio {
  std::vector<float> samples;
  int sample_rate = 0;
};

PcmAudio read_mono_pcm16_wav(const char *path, int required_sample_rate);

} // namespace llmc
