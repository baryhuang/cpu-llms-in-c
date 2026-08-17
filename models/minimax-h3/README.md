# MiniMax-H3

Status: the portable C graph rules and Apple M3 Pro runtime are implemented
through tokenizer, streamed 50-layer Qwen conditioner, 50-block H3 denoiser,
Video VAE, Audio VAE and media mux. The optimized real-weight
864×480×124 Turbo-4 run completed in 2,418.708 seconds with a 4.136 GiB
runtime peak physical footprint and zero swaps. All 124 frames decode and four
prompt-aligned shots are visible. Functional and visual smoke gates pass;
official full-precision parity and the speed target remain open.

| N-to-N milestone | 480p seconds | Evidence |
|---|---:|---|
| Correct seven-chunk baseline | **9,294.869743** | measured complete run |
| Exact buffer/binding/RoPE/command cleanup | 9,112.577745 | projected from 68.619048 s per VAE task |
| + 32-row simdgroup-matrix GEMM | 2,624.139600 | projected from 6.824399 s/task |
| + eight-query exact tiled attention | 2,567.102130 | projected from 6.281185 s/task |
| + 32-row shared-weight GEMM | 2,544.817665 | projected from 6.068952 s/task |
| + 64-row shared-weight GEMM | 2,435.747655 | projected from 5.030190 s/task |
| Current full validation | **2,418.708237** | measured complete run; **3.843×** over baseline |

## Latest measured run breakdown

| Stage | Seconds | Runtime share |
|---|---:|---:|
| Tokenizer | 0.007698 | <0.001% |
| Metal setup | 0.043712 | 0.002% |
| Text image access | 4.693886 | 0.194% |
| 50-layer text conditioner | 4.567422 | 0.189% |
| Turbo AdaLN compile | 0.868117 | 0.036% |
| H3 RoPE precompute | 0.000996 | <0.001% |
| **H3 denoise** | **1,898.120497** | **78.477%** |
| Video VAE precompute | 0.007911 | <0.001% |
| **Video VAE decode** | **487.273811** | **20.146%** |
| Audio VAE decode | 13.760077 | 0.569% |
| H.264/AAC mux | 0.671221 | 0.028% |
| Other measured runtime | 8.692889 | 0.359% |
| **Model runtime** | **2,418.708237** | **100%** |
| Preflight hashes and shell overhead | 65.671763 | outside runtime |
| **Command wall** | **2,484.380000** | runtime + preflight |

The model-runtime rows add exactly to 2,418.708237 seconds. Command wall adds
preflight SHA-256 checks of the local 28.22 GB conditioner and 780 MB Turbo
adapter. Post-run media verification is excluded from both measurements.

Projected rows use the baseline's measured non-VAE time plus 105 real-weight
VAE tasks; they are not presented as full runs. Reasons, deltas and quality
gates are in the [target optimization ledger](targets/apple-m3-pro/README.md#optimization-ledger).

## Pinned sources

| Item | Revision |
|---|---|
| [MiniMax-H3 weights](https://huggingface.co/MiniMaxAI/MiniMax-H3) | `42ed227ee7df40d41602854ae760620d6eb651fe` |
| [MiniMax-H3 release repository](https://github.com/MiniMax-AI/MiniMax-H3) | `d21241f0a4b3acbb34c97dae47fa417b7065e438` |
| [Diffusers H3 reference](https://github.com/huggingface/diffusers/blob/d5baa4fb548294f47dbca49890abd4b291204c60/src/diffusers/models/transformers/transformer_minimax_h3.py) | `d5baa4fb548294f47dbca49890abd4b291204c60` |
| [Turbo v4-600 EMA LoRA](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora) | `43a74557ac3f6539db8e0f2a959d03feb7a81480` |
| [Third-party compact execution artifact](https://huggingface.co/Sawfwair/MiniMax-H3-FL2VA-MLX-4bit) | `e1244ad93d60c737c7e0f065a1c9372f3de7caf8` |

File sizes, parameter counts and checksums are pinned in [`source.json`](source.json). The repository does not contain weights.

## Model graph

H3 Base is a joint video and stereo-audio diffusion model, not an autoregressive language model. T2VA packs text, audio and video into one non-causal self-attention sequence.

| Component | Released graph |
|---|---|
| Conditioner | Qwen3-VL-32B; H3 consumes the hidden state after language layer 50 |
| Text refinement | projection from 5120 to 5376, followed by 2 transformer blocks |
| H3 transformer | 50 blocks, hidden 5376, 56 heads, head dimension 128 |
| Feed-forward | SwiGLU, inner dimension 14336 |
| Video input | 24-channel VAE latent, patch `(1,2,2)` |
| Audio input | 32-channel latent, two channel-major streams at 40 rows/s each |
| Conditioning | per-row AdaLN selected by `(timestep, modality)` |
| Position | three-axis `(time,height,width)` RoPE; 16 frequencies per axis |
| Sampling | one guidance-distilled forward per step; separate video shift 12 and audio shift 3 |
| Output | video velocity and audio velocity, followed by separate VAEs |

H3 has no cross-attention inside the 50-block stack. Every block performs full self-attention over the packed document in the exact graph. The initial open release does not include MiniMax's trained sparse-attention implementation.

## Released weight cost

One FL2VA/T2VA workflow is 144,016,187,900 tensor/checkpoint bytes before tokenizer and small configuration files.

| Component | Parameters | Released bytes | GiB |
|---|---:|---:|---:|
| Qwen3-VL-32B conditioner | 33,357,390,064 | 66,714,780,128 | 62.13 |
| H3 transformer | 33,122,992,896 | 66,280,430,144 | 61.73 |
| Video VAE | 2,603,868,984 | 10,415,548,320 | 9.70 |
| Audio VAE | 151,326,585 | 605,429,308 | 0.56 |
| **Total** | **69,235,578,529** | **144,016,187,900** | **134.13** |

The Hugging Face repository repeats components under multiple workflow layouts. Downloading the entire repository is not required and is not the compiler input contract.

## Exact compile-time reductions

| Rewrite | Reason it is valid | Effect |
|---|---|---|
| Stop Qwen after language layer 49 | H3 consumes `hidden_states[50]`; layers 50-63, final norm and LM head do not contribute | removes unused conditioner execution and packed output weights |
| Run conditioner once | prompt embeddings do not change during denoising | conditioner can be unloaded before the H3 stack |
| Precompute fixed prompt embeddings | a deployment profile may pin the complete prompt | removes Qwen from that artifact's target-time path |
| Precompute fixed-schedule AdaLN outputs | all sampling timesteps and modality IDs are known at compile time | removes the timestep MLP, curve interpolation and block AdaLN projections from target execution |
| Static packed layout and RoPE | canvas, frame count, text-row count and workflow determine every row coordinate | no graph construction, row scatter or position-grid allocation at target time |
| Static Video VAE bindings and RoPE | the 36-layer decoder reuses one weight graph and one position grid for every same-shaped spatial tile | bind tensor offsets and generate rotary coefficients once before tile execution |
| No CFG branch | H3 is guidance-distilled | one forward per denoise evaluation |
| Online exact attention | dense score matrices are not model outputs | preserves dense attention without allocating `S x S` scores |

The public pruned H3 checkpoints already replace the original 13.01B-parameter AdaLN projections with a small interpolated curve basis. Direct fixed-step tables are still useful for removing interpolation and the remaining per-block projection, but they are not a new 26 GB reduction relative to that pruned baseline.

## Approximate reductions requiring quality measurement

| Rewrite | Target form | Required comparison |
|---|---|---|
| Mixed W4/W8 transformer | per-layer formats selected by calibration error | block output error and fixed-seed final media |
| Q/K/V activation compression | FP8 or INT8 storage between projection and attention | attention output and trajectory drift |
| Summary-before-K/V projection | average post-norm/AdaLN hidden rows by tree node, then project summaries once | algebraic K/V identity plus summary-attention trajectory drift |
| Hierarchical sparse attention | exact selected leaves plus multilevel summaries for rejected regions | route density, attention error and prompt/motion coverage |
| Cross-step route reuse | refresh only tree nodes whose centroids changed past a bound | per-step route recall and final media |
| Residual block reuse | skip a suffix only after a first-block change test | cached-step list and trajectory drift |
| Reduced denoise schedule | Turbo v4-600 at four evaluations | output review against the pinned Turbo peer |
| VAE W8/mixed precision | sensitive boundary layers retained at FP16/FP32 | decoded frame/audio error and perceptual review |

These approximations are independent axes. A result must state exactly which axes were active.

## Token and branch reuse

The next compute target is QKV/MLP, not another attention-only shortcut. Two
primary studies constrain the compiler search space:

| Evidence | Useful observation | H3 adaptation |
|---|---|---|
| [TAPE, arXiv:2605.17837](https://arxiv.org/abs/2605.17837) | temporally smooth token importance, reselect across layers, prune more at noisy early steps and relax late | smooth matching spatial leaves across H3 latent frames; search five 10-layer bands and four exact Turbo evaluations |
| [ASTRAEA, arXiv:2506.05096](https://arxiv.org/abs/2506.05096) | select queries and per-token MLP work while preserving the K/V domain; previous LSE and input delta are low-overhead selection signals | the Metal primitive emits all row/head LSE values; compiler search chooses modality-specific budgets |

H3's proposed cache unit is the branch output before the AOT AdaLN gate.
Reusing a previous block output would retain the previous timestep's
modulation. Reusing the ungated attention or MLP branch and applying the
current exact compiled gate separates feature reuse from the known timestep
change. This remains a design hypothesis until a real-weight four-evaluation
trajectory test passes.

The fixed four-step workload makes offline search small enough to include
`(evaluation, 10-layer band, modality, route budget)`. Compilation cost is not
part of the target budget. Candidate schedules are ranked against dense
boundary tensors and final audio/video metrics; a fast schedule is rejected if
it only preserves a synthetic attention fixture.

The sparse path does not first project K/V for all 15,485 rows. H3's K/V
linear layers have no bias, so the tree may average normalized and modulated
hidden rows before projection: `W·mean(x) = mean(W·x)`. Selected leaves retain
exact rows; rejected branches contribute one projected summary. This exchange
is algebraically exact for each summary vector. Replacing many attention keys
by that vector remains approximate and must pass the trajectory gate.

## Implemented portable C boundary

[`targets/generic/minimax_h3.c`](targets/generic/minimax_h3.c) currently owns:

| Boundary | Implemented behavior |
|---|---|
| Frame alignment | request rounded up to `17n+5` |
| Video temporal compression | `17n+5` pixel frames become `5n+2` latent frames |
| Audio temporal geometry | `round(frames / 24 * 40)` latents per channel |
| T2VA row order | `[text | left audio | right audio | video]` |
| Video row order | latent-frame major, then patch row, then patch column |
| RoPE coordinates | official aspect-normalized spatial grid and `(1,4,4,4,4) * 5/3` temporal spacing |
| Scheduler | shifted float32 sigma grid and data-ward Euler update |
| AdaLN addressing | `timestep_index * 3 + modality_tag` |

The code allocates no memory. The caller supplies layout and scheduler buffers. FL2VA condition rows, Ref2VA, tensor kernels and image formats remain open.

## Validation state

| Gate | State |
|---|---|
| Default 1344×768×124 geometry | pass: 37,296 video rows, 414 audio rows |
| Speedrun 864×480×124 geometry | pass: 15,485 total rows with 86 text rows |
| Shift 12 / shift 3 schedule | pass against pinned float32 bit patterns |
| Apple M3 Pro attention primitive | 49.893 ms at 15,485 rows including per-row/head LSE; synthetic Q/K/V, approximate route, quality gate open |
| Apple M3 Pro W4 projection primitive | 618.205 ms for one 5,376→14,336 projection at 15,485 rows; 194.602 ms at 4,646 rows; synthetic arithmetic only |
| Sparse work planner | central band at first reusable evaluation: Q 3,054, K/V 3,420 and MLP 2,583 rows; 18.370% of dense projection MACs |
| Small-layout RoPE order and tags | pass |
| Official tensor-boundary parity | tokenizer IDs and compact-artifact tensor schemas pinned; independent dense trajectory parity remains open |
| Packed checkpoint import | compiler/import tooling supports remote or local safetensors; deployed inference requires local packed files and performs no network reads |
| Real-weight downstream path | pass at 32×32×22: H3 30 intervals, video VAE, audio VAE and MP4 mux |
| Free-prompt N-to-N generation | pass at 128×128×22: 15 prompt tokens → 50-layer conditioner → 30 H3 intervals → both VAEs → verified 22-frame H.264 + stereo AAC MP4 |
| Corrected 480p N-to-N generation | pass at 864×480×124 Turbo-4: 240 prompt tokens → four H3 evaluations → seven-chunk Video VAE → synchronized MP4 in 2,418.708 s; 3.843× over the pre-optimization baseline |
| Corrected 480p media quality | four prompt-aligned shots, 0 flat frames, no sampled tile seam or temporal break; same-prompt official parity remains open |

Exact input, output media, stage timings, CPU time, process footprint and post-run verification are in the [Apple M3 Pro review](targets/apple-m3-pro/REVIEW.html) and [raw results](targets/apple-m3-pro/results.json).

Build the C gate with `make build/minimax-h3-test`, then run `build/minimax-h3-test`.
