#!/usr/bin/env python3
"""Run the 12-case evaluation for the Qwen3.5-0.8B artifact.

Benchmark and verification stay separate, following the Gemma record:
the C runtime executes all cases in one process (warm measurement,
wrapped in /usr/bin/time -v for peak RSS), and the independent NumPy
BF16 reference provides the verification decisions. Output follows the
results.json schema of the repository.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gemma4_task_reference import decode_float32  # noqa: E402
from q4_image import sha256_file  # noqa: E402
from qwen35_reference import Qwen35TextConfig, score_answers  # noqa: E402
from safetensors_file import SafetensorsFile  # noqa: E402

TEXT_PREFIX = "model.language_model."


def canonical_prompt(system_prompt: str, observation: str) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{observation}<|im_end|>\n<|im_start|>assistant\n"
    )


class CheckpointWeights(dict):
    def __init__(self, source: SafetensorsFile):
        super().__init__()
        self.source = source

    def __missing__(self, name):
        info = self.source.tensors[TEXT_PREFIX + name]
        array = decode_float32(
            self.source.read_bytes(TEXT_PREFIX + name), info.dtype, info.shape
        )
        value = np.asarray(array, dtype=np.float32)
        self[name] = value
        return value


def run_c_runtime(binary: Path, image: Path, prompts: list[str], answers: list[str],
                  threads: int) -> tuple[list[dict], dict]:
    with tempfile.NamedTemporaryFile("w", suffix=".prompts", delete=False) as handle:
        handle.write("\0".join(prompts))
        prompts_path = handle.name
    command = [
        "/usr/bin/time", "-v",
        str(binary), str(image),
        "--prompts-file", prompts_path,
        "--answers-text", ",".join(answers),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env={"OMP_NUM_THREADS": str(threads), "PATH": "/usr/bin:/bin"},
    )
    if result.returncode != 0:
        raise RuntimeError(f"runtime failed: {result.stderr[-2000:]}")
    cases = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    rss_match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", result.stderr)
    wall_match = re.search(r"Elapsed \(wall clock\) time.*: (.+)", result.stderr)
    process = {
        "maximum_resident_kib": int(rss_match.group(1)) if rss_match else None,
        "wall_time": wall_match.group(1).strip() if wall_match else None,
    }
    return cases, process


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    pins = json.loads((repository / "models/qwen3.5-0.8b/pins.json").read_text())
    profile = json.loads((repository / "models/qwen3.5-0.8b/profile.json").read_text())
    manifest = json.loads(
        args.image.with_suffix(args.image.suffix + ".json").read_text()
    )
    if sha256_file(args.tokenizer) != pins["model"]["files"]["tokenizer.json"]:
        raise ValueError("tokenizer does not match the pinned revision")

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    answers = profile["answers"]
    answer_ids = []
    for answer in answers:
        ids = tokenizer.encode(answer, add_special_tokens=False).ids
        if len(ids) != 1:
            raise ValueError(f"answer {answer} is not a single token")
        answer_ids.append(ids[0])

    prompts = [
        canonical_prompt(profile["system_prompt"], case["observation"])
        for case in profile["cases"]
    ]

    print("running the C runtime (warm, single process)...", flush=True)
    c_cases, c_process = run_c_runtime(
        args.binary, args.image, prompts, answers, args.threads
    )
    if len(c_cases) != len(profile["cases"]):
        raise RuntimeError("case count mismatch from the C runtime")

    print("running the BF16 NumPy reference...", flush=True)
    config = Qwen35TextConfig()
    reference_cases = []
    reference_started = time.monotonic()
    with SafetensorsFile(args.checkpoint) as source:
        weights = CheckpointWeights(source)
        for case, prompt in zip(profile["cases"], prompts):
            ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            scores = score_answers(config, weights, ids, answer_ids)
            reference_cases.append(
                {
                    "id": case["id"],
                    "logits": {a: float(s) for a, s in zip(answers, scores)},
                    "predicted": answers[int(np.argmax(scores))],
                }
            )
            print(f"  reference {case['id']}: {reference_cases[-1]['predicted']}", flush=True)
    reference_seconds = time.monotonic() - reference_started

    verification_cases = []
    image_correct = reference_correct = agreement = 0
    for case, c_result, ref in zip(profile["cases"], c_cases, reference_cases):
        image_predicted = answers[c_result["chosen_index"]]
        expected = case["expected"]
        image_correct += image_predicted == expected
        reference_correct += ref["predicted"] == expected
        agreement += image_predicted == ref["predicted"]
        verification_cases.append(
            {
                "id": case["id"],
                "expected": expected,
                "reference_predicted": ref["predicted"],
                "image_predicted": image_predicted,
                "reference_logits": ref["logits"],
                "image_logits": {
                    answer: entry["logit"]
                    for answer, entry in zip(answers, c_result["answers"])
                },
            }
        )

    total_tokens = sum(entry["tokens"] for entry in c_cases)
    total_seconds = sum(entry["prefill_seconds"] for entry in c_cases)
    output = {
        "schema_version": 1,
        "date": time.strftime("%Y-%m-%d"),
        "classification": "prompt-defined smoke evaluation; not a product safety evaluation",
        "profile_id": profile["profile_id"],
        "source": {
            "checkpoint_revision": pins["model"]["revision"],
            "checkpoint_sha256": pins["model"]["files"][
                "model.safetensors-00001-of-00001.safetensors"
            ],
            "tokenizer_sha256": pins["model"]["files"]["tokenizer.json"],
        },
        "image": manifest,
        "runtime": {
            "language": "C11",
            "threads": args.threads,
            "binary_sha256": sha256_file(args.binary),
            "input": "prompt text via the in-image tokenizer; per-call answer set",
            "framework_runtime_dependencies": [],
        },
        "verification": {
            "scope": "output checks; excluded from the C runtime benchmark",
            "summary": {
                "cases": len(profile["cases"]),
                "image_human_label_accuracy": image_correct / len(profile["cases"]),
                "reference_human_label_accuracy": reference_correct / len(profile["cases"]),
                "image_reference_decision_agreement": agreement / len(profile["cases"]),
            },
            "reference_execution": {
                "kind": "independent NumPy BF16 reference, not a Transformers run",
                "compute_duration_seconds": reference_seconds,
            },
            "cases": verification_cases,
        },
        "benchmark": {
            "scope": "C runtime warm measurements on the machine below; verification excluded",
            "machine": "unpinned x86-64 development machine; not a target benchmark",
            "warm": {
                "total_prompt_tokens": total_tokens,
                "total_classification_duration_seconds": round(total_seconds, 6),
                "aggregate_classification_tokens_per_second": round(
                    total_tokens / total_seconds, 6
                ),
                "maximum_resident_kib": c_process["maximum_resident_kib"],
                "wall_time": c_process["wall_time"],
                "cases": [
                    {
                        "id": case["id"],
                        "prompt_tokens": entry["tokens"],
                        "classification_duration_seconds": entry["prefill_seconds"],
                        "classification_tokens_per_second": entry[
                            "prefill_tokens_per_second"
                        ],
                    }
                    for case, entry in zip(profile["cases"], c_cases)
                ],
            },
        },
        "limitations": [
            "The profile contains twelve obvious English examples and is not a safety benchmark.",
            "The reference is an independent NumPy execution of pinned BF16 weights, not a Transformers run.",
            "Benchmark numbers are from an unpinned development machine, not the A113X target.",
            "Prefill is sequential per token; batched prefill is roadmap step 2.",
            "No distribution-shift, adversarial, calibration, or long-context evaluation has been run.",
        ],
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["verification"]["summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
