# Tests

```sh
make test
```

This command runs the Python unit tests and the miniature C layer fixture. It does not download a checkpoint.

## Test surfaces

| Surface | Input | Checks |
|---|---|---|
| Python reference | synthetic arrays and `models/gemma-4-e2b/profile.json` | RoPE, normalization, attention, PLE, profile encoding |
| C layer fixture | `fixtures/gemma4_layer_v1.bin` | ten tensor boundaries for one miniature local layer |
| Official layer-0 fixture | generated, ignored | two real token rows and official layer-0 weights |
| Complete task graph | generated, ignored | all 35 text layers and two output labels |
| Whisper large-v3 front end | `fixtures/whisper_log_mel_128_v1.bin` | pinned 128-bin filterbank, centered STFT, log clamp and normalization against the scalar C boundary |
| Whisper small.en front end | `fixtures/whisper_log_mel_80_v1.bin` | pinned 80-bin filterbank, centered STFT, log clamp and normalization against the scalar C boundary |
| Whisper encoder stem | `fixtures/whisper_encoder_stem_v1.bin` | Conv1D, exact GELU, stride-2 Conv1D, layout transpose and position addition |
| Whisper encoder block | `fixtures/whisper_encoder_block_v1.bin` | pre-norm multi-head self-attention, residual, pre-norm GELU FFN and residual |

The official layer-0 result is recorded in [`../models/gemma-4-e2b/layer0-validation.json`](../models/gemma-4-e2b/layer0-validation.json). The complete task result and every per-case value are in [`../models/gemma-4-e2b/results.json`](../models/gemma-4-e2b/results.json) and [`../REVIEW.html`](../REVIEW.html).

## Complete task reproduction

```sh
python3 compiler/evaluate_gemma4_task_reference.py \
  --checkpoint /path/to/model.safetensors \
  --config /path/to/config.json \
  --tokenizer /path/to/tokenizer.json \
  --profile models/gemma-4-e2b/profile.json \
  --output reference.json

python3 compiler/compile_gemma4_task_image.py \
  --checkpoint /path/to/model.safetensors \
  --config /path/to/config.json \
  --tokenizer /path/to/tokenizer.json \
  --profile models/gemma-4-e2b/profile.json \
  --output hazard-v1.g4task

make OMPFLAGS=-fopenmp build/gemma4-task
OMP_NUM_THREADS=2 build/gemma4-task hazard-v1.g4task all
```

The generated checkpoint slices, reference output, model image, and binaries are excluded from Git.
