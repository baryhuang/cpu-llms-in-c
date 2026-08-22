# MiniMind-O

Status: native-C speech-to-speech prototype running as a resident service on
the Amlogic A113D/A113X target. The runtime target is `minimind-3o`, with the
vision path removed. Python/PyTorch is allowed only on the build host to
produce pinned fixtures and packed model images; the target executable and hot
path do not embed or launch Python.

Implemented components include the Q8 Thinker and Talker, tokenizer,
SenseVoice audio encoder/projector, stateful Mimi decoder, continuous ALSA
capture/VAD, Cortex-A53 NEON kernels and four-core OpenMP scheduling. Target
architecture, correctness gates and measurements are in the
[A113X target record](targets/a113x/README.md).

Upstream source: <https://github.com/jingyaogong/minimind-o>.
