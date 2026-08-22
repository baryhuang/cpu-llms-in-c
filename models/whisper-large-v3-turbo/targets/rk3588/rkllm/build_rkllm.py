#!/usr/bin/env python3
"""Build only the RK3588 three-core W8A8 RKLLM decoder."""

import argparse
from pathlib import Path

from rkllm.api import RKLLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--custom-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()

    llm = RKLLM()
    ret = llm.load_huggingface(
        model=str(args.model),
        device="cpu",
        dtype="float16",
        custom_config=str(args.custom_config),
        load_weight=True,
    )
    if ret != 0:
        raise SystemExit(f"load_huggingface failed: {ret}")
    ret = llm.build(
        do_quantization=True,
        # Match the official W8A8 export example: level 1 prioritizes
        # quantized-model accuracy over the level-0 speed/memory tradeoff.
        optimization_level=1,
        quantized_dtype="w8a8",
        quantized_algorithm="normal",
        target_platform="rk3588",
        num_npu_core=3,
        dataset=None if args.dataset is None else str(args.dataset),
        # RKLLM applies this limit to cross-attention input as well as the
        # decoder history. Whisper uses 1500 encoder frames, while its decoder
        # position table remains limited to 448 tokens.
        max_context=2048,
    )
    if ret != 0:
        raise SystemExit(f"build failed: {ret}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ret = llm.export_rkllm(
        str(args.output),
        export_tokenizer=False,
        # Cross-attention setup in runtime 1.3.0 accesses the embedding tensor
        # before the first callback-driven run. Keep the centered table in the
        # model even though inference supplies explicit token+position embeds.
        export_embedding=True,
    )
    if ret != 0:
        raise SystemExit(f"export_rkllm failed: {ret}")
    print(args.output)


if __name__ == "__main__":
    main()
