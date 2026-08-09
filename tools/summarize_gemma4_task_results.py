#!/usr/bin/env python3
"""Combine bounded-task reference, image, and C runtime measurements."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


def parse_runtime_log(path: Path):
    cases = []
    text = path.read_text()
    for line in text.splitlines():
        if line.startswith('{"case"'):
            cases.append(json.loads(line))
    metrics = {}
    patterns = {
        "wall_time": r"Elapsed \(wall clock\) time.*: ([0-9:.]+)",
        "maximum_resident_kib": r"Maximum resident set size \(kbytes\): (\d+)",
        "file_system_inputs": r"File system inputs: (\d+)",
        "file_system_outputs": r"File system outputs: (\d+)",
        "major_page_faults": r"Major \(requiring I/O\) page faults: (\d+)",
        "minor_page_faults": r"Minor \(reclaiming a frame\) page faults: (\d+)",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            metrics[name] = int(match.group(1)) if match.group(1).isdigit() else match.group(1)
    return cases, metrics


def distribution(values: list[float]) -> dict:
    return {
        "minimum": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--reference-log", required=True, type=Path)
    parser.add_argument("--runtime-log", required=True, type=Path)
    parser.add_argument("--cold-log", required=True, type=Path)
    parser.add_argument("--image-manifest", required=True, type=Path)
    parser.add_argument("--warm-binary-sha256", required=True)
    parser.add_argument("--cold-binary-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text())
    _, reference_metrics = parse_runtime_log(args.reference_log)
    image = json.loads(args.image_manifest.read_text())
    runtime_cases, runtime_metrics = parse_runtime_log(args.runtime_log)
    cold_cases, cold_metrics = parse_runtime_log(args.cold_log)
    if len(runtime_cases) != len(reference["results"]):
        raise ValueError("runtime and reference case counts differ")

    teacher_agreement = 0
    human_correct = 0
    false_negatives = 0
    false_positives = 0
    comparisons = []
    label_names = ["safe", "danger"]
    for runtime, teacher in zip(runtime_cases, reference["results"]):
        expected = label_names[runtime["expected"]]
        predicted = label_names[runtime["predicted"]]
        teacher_predicted = teacher["predicted"]
        human_correct += predicted == expected
        teacher_agreement += predicted == teacher_predicted
        false_negatives += expected == "danger" and predicted == "safe"
        false_positives += expected == "safe" and predicted == "danger"
        comparisons.append(
            {
                "id": teacher["id"],
                "expected": expected,
                "reference_predicted": teacher_predicted,
                "q4_predicted": predicted,
                "reference_logits": teacher["logits"],
                "q4_logits": {
                    "safe": runtime["safe_logit"],
                    "danger": runtime["danger_logit"],
                },
                "prompt_tokens": runtime["tokens"],
                "prompt_seconds": runtime["prompt_seconds"],
                "prefill_tokens_per_second": runtime["prefill_tokens_per_second"],
                "decode_seconds": runtime["decode_seconds"],
                "decode_tokens_per_second": runtime["decode_tokens_per_second"],
            }
        )

    total_tokens = sum(case["tokens"] for case in runtime_cases)
    total_prompt_seconds = sum(case["prompt_seconds"] for case in runtime_cases)
    total_decode_seconds = sum(case["decode_seconds"] for case in runtime_cases)
    output = {
        "schema_version": 1,
        "date": "2026-08-10",
        "classification": "bounded-profile smoke measurement; not a product safety evaluation",
        "profile_id": reference["profile_id"],
        "source": {
            "checkpoint_revision": reference["checkpoint_revision"],
            "checkpoint_sha256": reference["checkpoint_sha256"],
            "tokenizer_sha256": reference["tokenizer_sha256"],
        },
        "image": image,
        "runtime": {
            "language": "C11",
            "threads": 2,
            "warm_binary_sha256": args.warm_binary_sha256,
            "cold_binary_sha256": args.cold_binary_sha256,
            "binary_difference": "post-run input validation and error-path cleanup; arithmetic hot path unchanged",
            "framework_runtime_dependencies": [],
        },
        "quality": {
            "cases": len(runtime_cases),
            "safe_cases": sum(case["expected"] == 0 for case in runtime_cases),
            "danger_cases": sum(case["expected"] == 1 for case in runtime_cases),
            "reference_human_label_accuracy": reference["accuracy"],
            "q4_human_label_accuracy": human_correct / len(runtime_cases),
            "q4_reference_decision_agreement": teacher_agreement / len(runtime_cases),
            "q4_false_negatives": false_negatives,
            "q4_false_positives": false_positives,
        },
        "reference_measurement": {
            "elapsed_seconds": reference["elapsed_seconds"],
            **reference_metrics,
        },
        "warm_measurement": {
            "total_prompt_tokens": total_tokens,
            "total_prompt_seconds": total_prompt_seconds,
            "aggregate_prefill_tokens_per_second": total_tokens / total_prompt_seconds,
            "per_case_prefill_tokens_per_second": distribution(
                [case["prefill_tokens_per_second"] for case in runtime_cases]
            ),
            "total_decode_steps": len(runtime_cases),
            "total_decode_seconds": total_decode_seconds,
            "aggregate_decode_tokens_per_second": len(runtime_cases) / total_decode_seconds,
            "per_case_decode_tokens_per_second": distribution(
                [case["decode_tokens_per_second"] for case in runtime_cases]
            ),
            **runtime_metrics,
        },
        "cold_measurement": {
            "case": cold_cases[0] if len(cold_cases) == 1 else cold_cases,
            **cold_metrics,
        },
        "comparisons": comparisons,
        "limitations": [
            "The profile contains twelve obvious English examples and is not a safety benchmark.",
            "The reference is an independent NumPy execution of pinned BF16 weights and equations, not a Transformers BF16 run.",
            "The runtime accepts precompiled token sequences from this profile; it does not implement a general tokenizer.",
            "The scalar Q4 kernel is not an SSE4.2-optimized target kernel.",
            "No distribution-shift, adversarial, calibration, or long-context evaluation has been run.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
