#!/usr/bin/env python3
"""Evaluate a bounded Gemma 4 task profile from the pinned BF16 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

try:
    from .gemma4_task_reference import (
        MODEL_PREFIX,
        build_compiled_token_rows,
        forward_candidates,
        gather_profile_inputs,
        load_rows,
        validate_text_config,
    )
    from .safetensors_file import SafetensorsFile
except ImportError:
    from gemma4_task_reference import (
        MODEL_PREFIX,
        build_compiled_token_rows,
        forward_candidates,
        gather_profile_inputs,
        load_rows,
        validate_text_config,
    )
    from safetensors_file import SafetensorsFile


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_prompt(system_prompt: str, observation: str) -> str:
    return (
        "<bos><|turn>system\n"
        + system_prompt.strip()
        + "<turn|>\n<|turn>user\n"
        + observation.strip()
        + "<turn|>\n<|turn>model\n"
    )


def encode_profile(profile: dict, tokenizer: Tokenizer, maximum_cases: int | None = None):
    cases = profile["cases"][:maximum_cases]
    sequences = []
    for case in cases:
        encoded = tokenizer.encode(
            canonical_prompt(profile["system_prompt"], case["observation"]),
            add_special_tokens=False,
        )
        sequences.append(encoded.ids)
    label_ids = []
    for label in profile["labels"]:
        encoded = tokenizer.encode(label["text"], add_special_tokens=False)
        if len(encoded.ids) != 1:
            raise ValueError(f"label {label['name']} does not encode to one token: {encoded.ids}")
        label_ids.append(encoded.ids[0])
    return cases, sequences, label_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-cases", type=int)
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    pins = json.loads((repository / "models/gemma-4-e2b/pins.json").read_text())
    if args.checkpoint.stat().st_size != pins["model"]["model_bytes"]:
        raise ValueError("checkpoint size does not match the pinned revision")
    if sha256_file(args.config) != pins["model"]["files"]["config.json"]:
        raise ValueError("config does not match the pinned revision")
    if sha256_file(args.tokenizer) != pins["model"]["files"]["tokenizer.json"]:
        raise ValueError("tokenizer does not match the pinned revision")

    config = json.loads(args.config.read_text())
    text = validate_text_config(config)
    profile = json.loads(args.profile.read_text())
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    cases, sequences, label_ids = encode_profile(profile, tokenizer, args.maximum_cases)
    unique_token_ids = sorted(set(label_ids).union(*(set(sequence) for sequence in sequences)))

    started = time.monotonic()
    with SafetensorsFile(args.checkpoint) as source:
        embeddings, folded_ple = build_compiled_token_rows(source, text, unique_token_ids)
        inputs, ple, lengths = gather_profile_inputs(
            sequences, unique_token_ids, embeddings, folded_ple
        )
        candidate_rows = load_rows(source, f"{MODEL_PREFIX}.embed_tokens.weight", label_ids)

        def progress(layer: int, total: int) -> None:
            elapsed = time.monotonic() - started
            print(f"reference layer={layer}/{total} elapsed_seconds={elapsed:.3f}", flush=True)

        logits, final_states = forward_candidates(
            source, text, inputs, ple, lengths, candidate_rows, progress=progress
        )

    label_names = [label["name"] for label in profile["labels"]]
    results = []
    correct = 0
    for index, case in enumerate(cases):
        predicted_index = int(np.argmax(logits[index]))
        predicted = label_names[predicted_index]
        expected = case["expected"]
        correct += int(predicted == expected)
        results.append(
            {
                "id": case["id"],
                "expected": expected,
                "predicted": predicted,
                "token_count": lengths[index],
                "logits": {
                    label_names[label_index]: float(logits[index, label_index])
                    for label_index in range(len(label_names))
                },
                "margin": float(logits[index, predicted_index] - logits[index, 1 - predicted_index]),
                "final_hidden_l2": float(np.linalg.norm(final_states[index])),
            }
        )
    elapsed = time.monotonic() - started
    output = {
        "schema_version": 1,
        "classification": "bounded-profile BF16-weight NumPy reference; not a product safety evaluation",
        "profile_id": profile["profile_id"],
        "checkpoint_revision": pins["model"]["revision"],
        "checkpoint_sha256": pins["model"]["files"]["model.safetensors"],
        "tokenizer_sha256": pins["model"]["files"]["tokenizer.json"],
        "case_count": len(cases),
        "unique_compiled_tokens": len(unique_token_ids),
        "label_token_ids": dict(zip(label_names, label_ids)),
        "accuracy": correct / len(cases),
        "elapsed_seconds": elapsed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
