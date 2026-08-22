# MiniMind-O native streaming: engineering learnings

Date: 2026-08-22
Target: ThirdReality TRHub-V3, Amlogic A113D/A113X, four Cortex-A53 cores,
2 GiB RAM
Runtime constraint: native C11 on the box; no Python process and no OpenMP

## Final solution and measured results

The final P1 prototype is one always-resident native-C service. It implements
both input-token streaming and output-audio streaming; there is no production
batch-mode switch:

```text
ALSA capture -> VAD -> PCM SPSC queue -> online SenseVoice/W8A8
                                          -> Thinker/Talker incremental prefill
                                          -> ownership handoff at EOS
                                          -> Talker code SPSC queue
                                          -> streaming Mimi decode
                                          -> PCM SPSC queue -> ALSA playback
```

The implementation makes the following production choices:

- VAD starts the encoder/prefill thread at `speech_start`, not at
  `speech_end`. Stable audio embeddings enter Thinker and Talker while capture
  is still running.
- SenseVoice commits 8 LFR center frames with 4 frames of right context. Each
  of its 70 layers retains a 32-frame K/V cache; the uncertain VAD tail is not
  committed until it becomes speech.
- Cortex-A53 NEON W8A8 kernels and a three-worker persistent pthread pool use
  all four CPU cores without OpenMP team creation.
- Thinker/Talker state is single-owner. The input worker is joined at EOS and
  ownership is handed to the generation thread; model state has no mutex.
- The four data paths are SPSC queues. Release/acquire atomics publish payloads;
  a mutex is used only around an empty-queue condition-variable sleep and its
  enqueue notification.
- Talker frames are decoded as they are produced, and decoded PCM is written
  directly to ALSA. Capture continues to drain while inference and playback
  run.
- The executable and model releases are SHA256-pinned. The systemd runner
  verifies them, downloads missing artifacts through a `.part` file, and then
  starts the already-warm resident service.

The measured result on the A113X box is:

| Gate | Final measured result |
|---|---|
| Online SenseVoice, 1.18 s fixture | 1.11 s wall, 365% CPU, 228,580 KiB peak RSS |
| Encoder improvement | 3.04 s to 1.11 s, 2.74x wall-clock speedup |
| W8A8 numerical gate | cosine 0.998620222, RMSE 0.029285131 over 15,360 values |
| Input prefill correctness | positions `4 -> 12 -> 24 -> 37`; whole-prompt tokenizer parity |
| Persistent compute pool | 2,000 job passes and 100 ownership handoffs passed |
| Four-core text/audio workload | 8.23 s wall at 358% CPU; 40 Mimi frames |
| Streaming Mimi, 8 frames/0.64 s PCM | 1.26 s wall; 6.19x faster than the original decoder |
| Output overlap trace | first Talker frame 2.087 s, first Mimi frame 2.844 s, 9 frames decoded before producer EOS |
| Physical playback fixture | first audio 8.044 s, playback end 9.469 s, zero ALSA underruns |
| Empty-cache boot recovery | 78 s download/restore; warm restart verification 6 s; warm-up 539 ms |

These results prove the queueing, ownership, incremental-input, multi-core, and
audio-output paths. They do **not** yet meet the 1--3 second full conversational
latency target. The 1.11-second encoder number is a file-fed component test,
not an end-to-end live result, and the physical playback fixture predates the
final W8A8 input kernel. The remaining dominant costs are autoregressive
Thinker/Talker generation and Mimi throughput: even after optimization, Mimi
needs 1.26 seconds to synthesize 0.64 seconds of audio. A fresh combined live
mic-to-speaker trace is therefore a required next acceptance test; the project
must not infer it by adding unrelated microbenchmarks.

This record captures the mistakes, measurements, and design rules learned
while turning the first MiniMind-O speech-to-speech prototype into an
always-resident, bidirectionally streaming pipeline. It is intentionally more
general than a changelog: the goal is to prevent the same architectural errors
in later AI-hub models.

## 1. Output streaming is only half of streaming

The first implementation streamed Talker codebooks into Mimi and streamed
decoded PCM into ALSA, but it still waited for VAD `speech_end` before starting
SenseVoice and Thinker/Talker prefill. It was correct to call the output path
streaming, but incorrect to call the complete interaction end-to-end
streaming.

The decisive rule is:

> As soon as speech starts, stable audio representations must enter the model
> state while capture continues.

The production input path now starts at `speech_start`:

```text
ALSA capture
    -> VAD/pre-roll
    -> online SenseVoice chunk
    -> committed 768-wide audio embedding
    -> <|audio_pad|> replacement
    -> Thinker bridge prefill
    -> Talker prefill
```

At `speech_end`, the runtime processes only the encoder tail and the fixed
assistant suffix. It does not cold-encode or cold-prefill the whole utterance.

The observable acceptance criterion is stronger than a final JSON boolean. On
a normal multi-second utterance, at least one `input_audio_commit` and
`input_prefill` event must occur before `speech_end`. A diagram or an output
worker alone cannot establish this.

## 2. A frozen bidirectional encoder cannot commit an arbitrary prefix

SenseVoice in this checkpoint uses full-sequence self-attention and an
11-position bidirectional FSMN. Re-running the offline graph on a longer audio
prefix can change embeddings previously produced for the shorter prefix.
Naively injecting those earlier embeddings into a causal language-model KV
cache would make irreversible decisions from non-final values.

The practical native-C solution is chunk-aware truncated attention:

- 8 LFR frames form the committed center, approximately 480 ms;
- 4 LFR frames are right lookahead, approximately 240 ms;
- each of the 70 encoder layers keeps 32 committed K/V frames, approximately
  1.92 seconds;
- right-lookahead frames are recomputed in the next chunk;
- only the center enters the cache and language models;
- EOS flushes all remaining frames.

This follows the streaming shape of SenseVoice SANM, but it is an online
approximation of a frozen bidirectional checkpoint. It is not bit-equivalent
to whole-utterance inference and must not be described as such.

The frontend has its own future dependency: one LFR position stacks seven mel
frames around a six-frame stride. A non-final LFR frame is exposed only after
the required right mel context exists. Encoder streaming must respect both
frontend stability and transformer/FSMN lookahead.

## 3. Committed tokens cannot be retracted, so VAD owns the uncertain tail

The original VAD appended low-energy trailing chunks and trimmed them only
after deciding that speech had ended. That works for a batch encoder but not
for a model whose KV cache has already consumed audio tokens.

The streaming design keeps possible trailing silence in a small pending ring:

- if speech resumes, the pending chunks are enqueued in order;
- if VAD closes the turn, they are discarded without entering the model;
- pre-roll remains bounded and is submitted when speech is first confirmed.

This is a general rule for incremental systems: ambiguity must stay outside an
irreversible state machine until it is resolved.

## 4. Model-state ownership is better than model-state locking

Thinker and Talker do not need a read/write lock. During capture-side prefill,
the input-inference thread is their only owner. At EOS, that thread finishes
and is joined. Generation then becomes the only owner. No two threads access
the same model state concurrently.

The ownership sequence is:

```text
input thread: reset -> prefix -> audio chunks -> suffix -> stop
                                                        |
                                                   join/handoff
                                                        |
main thread:                                      generation -> EOS
```

This eliminates locks around KV cache updates, tensor buffers, matrix kernels,
and model position. A lock would hide an invalid ownership model rather than
make it correct.

## 5. Locks belong only at queue enqueue/dequeue sleep boundaries

There are four data pipelines with SPSC ownership:

| Queue | Producer | Consumer |
|---|---|---|
| capture PCM | ALSA capture pthread | VAD/main thread |
| confirmed input PCM | VAD/main thread | input encoder/prefill pthread |
| Mimi code frames | Talker/main thread | Mimi decoder pthread |
| decoded PCM frames | Mimi decoder pthread | ALSA playback pthread |

Payload and sequence publication use release/acquire atomics. Mutexes exist
only to close the condition-variable race when a consumer sleeps on an empty
queue and a producer announces new data. No mutex covers:

- model inference or KV/history state;
- frontend, encoder, projector, Thinker, Talker, or Mimi computation;
- memcpy or queue payload ownership;
- file or ALSA I/O;
- worker-pool matrix dispatch.

The persistent compute pool uses one SPSC mailbox per worker, atomic epochs,
and futex sleep only between inference sessions. This replaced OpenMP team
creation and its large context-switch/control overhead. The A113X acceptance
test passed 2,000 parallel job passes and 100 session handoffs.

The lesson is not that every mutex is slow. The lesson is that a mutex should
never compensate for unclear data ownership, and its protected interval must
not include work that can be performed after dequeue or before enqueue.

## 6. Overlap and acceleration solve different latency terms

Input streaming hides work under the user's speaking time; it does not make
that work cheaper. The first online Q8 x f32 SenseVoice implementation still
needed 3.04 seconds for a pinned 1.18-second fixture, despite using about 337%
aggregate CPU. It was structurally streaming but slower than real time.

On Cortex-A53, Armv8.0 NEON W8A8 dense products were the successful local
optimization:

- activations are symmetrically quantized per input position;
- int8 weights and activations use widening `vmull_s8`/`vpadalq_s16` dots;
- four input positions reuse one weight-row load;
- normalized Q/K/V input is quantized once and shared across all three
  projections;
- the audio projector remains Q8 x f32 because it was not a measured
  bottleneck.

Measured on the box:

| Online encoder | Wall | Aggregate CPU | Peak RSS |
|---|---:|---:|---:|
| Q8 weights x f32 activation | 3.04 s | 337% | 228,568 KiB |
| W8A8, final shared-QKV build | 1.11 s | 365% | 228,580 KiB |

The final speedup is 2.74x. Sharing Q/K/V activation quantization improved the
last W8A8 version from 1.15 to 1.11 seconds and preserved its output byte for
byte.

The general latency equation is therefore:

```text
post-speech wait = max(0, input work - speaking-time overlap)
                 + final right-context flush
                 + fixed suffix prefill
                 + first response token/code/PCM buffer
```

Measure and optimize each term separately. A smaller final number without
stage events cannot prove which work was actually overlapped.

## 7. Numerical similarity is necessary but not a quality claim

The W8A8 online embedding gate compared 15,360 float values against the Q8 x
f32 online graph:

- cosine similarity: `0.998620222`;
- RMSE: `0.029285131`;
- maximum absolute error: `0.138373837`.

These numbers show that the kernel did not catastrophically corrupt encoder
output. They do not prove equal multilingual recognition or response quality.
Autoregressive sampling can amplify small logit differences, especially in a
small model. Physical Chinese/English tests and a fixed speech evaluation set
remain separate acceptance gates.

Likewise, a file-fed test queues audio faster than real time. Its
`speech_start_to_ready_ms` validates state transitions and total work, not the
amount of work hidden during a live utterance. Live overlap must be established
from timestamps around actual `speech_start`, `input_prefill`, and
`speech_end` events.

## 8. Logging must describe handoffs, not just stage completion

The useful production events are:

```text
EVENT  speech_start
EVENT  input_stream_start
STREAM input_audio_commit
STREAM input_prefill
EVENT  speech_end
EVENT  input_stream_eos
EVENT  input_caught_up
STREAM talker_produce
STREAM mimi_decode
STREAM alsa_write
EVENT  model_end
EVENT  playback_end
EVENT  inference_end
```

Each queue boundary needs a timestamp and monotonic count. Thinker and Talker
positions are printed after input prefill, allowing direct verification that
the caches advanced. For the pinned file test they advanced `4 -> 12 -> 24 ->
37`, and the incremental token sequence matched whole-prompt tokenization.

Final metrics distinguish input streaming, decoder/generation overlap, ALSA
streaming, first-audio latency, producer completion, and codec drain. One
generic `streaming=true` field is insufficient for debugging.

## 9. Operational correctness is part of latency correctness

Early versions suffered broken pipes, model warm-up failures, lost capture
during inference, silent playback, and truncated audio. Those are pipeline
failures, not secondary deployment issues.

The production rules are:

- all model images stay mapped in one resident service;
- ALSA capture is continuously drained, including during inference/playback;
- the service never falls back to a batch mode;
- model and executable artifacts are SHA256-pinned;
- the runner downloads missing/corrupt files to a `.part` path and atomically
  renames only after validation;
- systemd restarts the service and initializes microphone/speaker controls;
- release executables are built and tested before upload;
- target binaries link only `libc`/`libm` and do not depend on `libgomp`.

This prevents a fast benchmark binary from becoming a slow or nonfunctional
product after reboot.

## 10. What remains true after this increment

- A sub-lookahead utterance may have no input commit before EOS; that is a
  consequence of the 240 ms right-context quality policy, not a batch fallback.
- The frozen bidirectional encoder still trades some offline context for
  bounded streaming context.
- MiniMind-O response quality and Mimi real-time factor are independent of
  input-token streaming and must be evaluated independently.
- The fast safety-dialog path and the accurate notification/reasoning path
  should remain separate product paths; optimizing one model cannot satisfy
  both latency/accuracy objectives automatically.

## Code and evidence

- Online encoder and W8A8 kernels:
  [`targets/generic/minimindo_audio_encoder.c`](targets/generic/minimindo_audio_encoder.c)
- PCM queue, incremental prefill, model ownership handoff, and stream logs:
  [`targets/generic/minimindo_speech.c`](targets/generic/minimindo_speech.c)
- A113X design and measurements:
  [`targets/a113x/README.md`](targets/a113x/README.md)
- Machine-readable results:
  [`targets/a113x/results.json`](targets/a113x/results.json)
- Auto-download deployment runner:
  [`../../tools/threehub-voice/run-minimindo-native-a113x.sh`](../../tools/threehub-voice/run-minimindo-native-a113x.sh)
