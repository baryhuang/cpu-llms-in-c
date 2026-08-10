#!/usr/bin/env python3
"""Export a real-weight Gemma 4 layer-0 fixture from the pinned checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path

import numpy as np

try:
    from .safetensors_file import SafetensorsFile
except ImportError:
    from safetensors_file import SafetensorsFile


MAGIC = b"G4LYR001"
VERSION = 1
TENSOR_COUNT = 28
MODEL_PREFIX = "model.language_model"
LAYER_PREFIX = f"{MODEL_PREFIX}.layers.0"

WEIGHT_NAMES = {
    "input_norm": f"{LAYER_PREFIX}.input_layernorm.weight",
    "q_proj": f"{LAYER_PREFIX}.self_attn.q_proj.weight",
    "k_proj": f"{LAYER_PREFIX}.self_attn.k_proj.weight",
    "v_proj": f"{LAYER_PREFIX}.self_attn.v_proj.weight",
    "q_norm": f"{LAYER_PREFIX}.self_attn.q_norm.weight",
    "k_norm": f"{LAYER_PREFIX}.self_attn.k_norm.weight",
    "o_proj": f"{LAYER_PREFIX}.self_attn.o_proj.weight",
    "post_attention_norm": f"{LAYER_PREFIX}.post_attention_layernorm.weight",
    "pre_feedforward_norm": f"{LAYER_PREFIX}.pre_feedforward_layernorm.weight",
    "gate_proj": f"{LAYER_PREFIX}.mlp.gate_proj.weight",
    "up_proj": f"{LAYER_PREFIX}.mlp.up_proj.weight",
    "down_proj": f"{LAYER_PREFIX}.mlp.down_proj.weight",
    "post_feedforward_norm": f"{LAYER_PREFIX}.post_feedforward_layernorm.weight",
    "ple_gate": f"{LAYER_PREFIX}.per_layer_input_gate.weight",
    "ple_projection": f"{LAYER_PREFIX}.per_layer_projection.weight",
    "post_ple_norm": f"{LAYER_PREFIX}.post_per_layer_input_norm.weight",
}

EXPECTED_SHAPES = {
    "input_norm": (1536,),
    "q_proj": (2048, 1536),
    "k_proj": (256, 1536),
    "v_proj": (256, 1536),
    "q_norm": (256,),
    "k_norm": (256,),
    "o_proj": (1536, 2048),
    "post_attention_norm": (1536,),
    "pre_feedforward_norm": (1536,),
    "gate_proj": (6144, 1536),
    "up_proj": (6144, 1536),
    "down_proj": (1536, 6144),
    "post_feedforward_norm": (1536,),
    "ple_gate": (256, 1536),
    "ple_projection": (1536, 256),
    "post_ple_norm": (1536,),
}

FIXTURE_ORDER = [
    "input",
    "per_layer_input",
    "input_norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "q_norm",
    "k_norm",
    "o_proj",
    "post_attention_norm",
    "pre_feedforward_norm",
    "gate_proj",
    "up_proj",
    "down_proj",
    "post_feedforward_norm",
    "ple_gate",
    "ple_projection",
    "post_ple_norm",
    "expected_normalized_input",
    "expected_query",
    "expected_key",
    "expected_value",
    "expected_attention",
    "expected_after_attention",
    "expected_mlp",
    "expected_after_mlp",
    "expected_ple",
    "expected_output",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_float32(payload: bytes, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    if dtype == "BF16":
        bits = np.frombuffer(payload, dtype="<u2").astype("<u4") << np.uint32(16)
        values = bits.view("<f4")
    elif dtype == "F32":
        values = np.frombuffer(payload, dtype="<f4").copy()
    else:
        raise ValueError(f"expected BF16 or F32, found {dtype}")
    return values.reshape(shape)


def load_tensor(source: SafetensorsFile, name: str) -> np.ndarray:
    info = source.tensors[name]
    return decode_float32(source.read_bytes(name), info.dtype, info.shape)


def load_range(source: SafetensorsFile, name: str, offset: int, count: int) -> np.ndarray:
    info = source.tensors[name]
    return decode_float32(source.read_element_bytes(name, offset, count), info.dtype, (count,))


def bf16_scalar(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]


def linear(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (inputs.astype(np.float64) @ weights.astype(np.float64).T).astype(np.float32)


def rms_norm(inputs: np.ndarray, scale: np.ndarray | None, epsilon: float) -> np.ndarray:
    values = inputs.astype(np.float64)
    inverse_root = np.power(np.mean(values * values, axis=-1, keepdims=True) + epsilon, -0.5)
    output = values * inverse_root
    if scale is not None:
        output *= scale.astype(np.float64)
    return output.astype(np.float32)


def gelu_pytorch_tanh(values: np.ndarray) -> np.ndarray:
    x = values.astype(np.float64)
    inner = 0.79788456080286535588 * (x + 0.044715 * x * x * x)
    return (0.5 * x * (1.0 + np.tanh(inner))).astype(np.float32)


def apply_rope(states: np.ndarray, theta: float) -> np.ndarray:
    output = states.copy()
    head_dim = states.shape[-1]
    half = head_dim // 2
    inverse_frequency = 1.0 / np.power(theta, np.arange(0, head_dim, 2) / head_dim)
    for position in range(states.shape[0]):
        cosine = np.cos(position * inverse_frequency)
        sine = np.sin(position * inverse_frequency)
        first = states[position, :, :half].astype(np.float64)
        second = states[position, :, half:].astype(np.float64)
        output[position, :, :half] = (first * cosine - second * sine).astype(np.float32)
        output[position, :, half:] = (second * cosine + first * sine).astype(np.float32)
    return output


def attention_mqa(query: np.ndarray, key: np.ndarray, value: np.ndarray, window: int) -> np.ndarray:
    sequence_length, query_heads, head_dim = query.shape
    kv_heads = key.shape[1]
    groups = query_heads // kv_heads
    output = np.zeros_like(query)
    for query_position in range(sequence_length):
        first_key = max(0, query_position + 1 - window) if window else 0
        for query_head in range(query_heads):
            kv_head = query_head // groups
            scores = []
            raw_maximum = -math.inf
            for key_position in range(first_key, query_position + 1):
                score = float(
                    np.sum(
                        query[query_position, query_head].astype(np.float64)
                        * key[key_position, kv_head].astype(np.float64),
                        dtype=np.float64,
                    )
                )
                scores.append(np.float32(score))
                raw_maximum = max(raw_maximum, score)
            exponentials = np.array(
                [np.float32(math.exp(float(score) - raw_maximum)) for score in scores],
                dtype=np.float32,
            )
            denominator = sum(math.exp(float(score) - raw_maximum) for score in scores)
            for index in range(head_dim):
                total = 0.0
                for score_index, key_position in enumerate(range(first_key, query_position + 1)):
                    total += (
                        float(exponentials[score_index])
                        / denominator
                        * float(value[key_position, kv_head, index])
                    )
                output[query_position, query_head, index] = np.float32(total)
    return output


def validate_config(config: dict) -> dict:
    text = config["text_config"]
    required = {
        "hidden_size": 1536,
        "num_hidden_layers": 35,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "intermediate_size": 6144,
        "hidden_size_per_layer_input": 256,
        "sliding_window": 512,
        "enable_moe_block": False,
        "hidden_activation": "gelu_pytorch_tanh",
    }
    for key, expected in required.items():
        if text.get(key) != expected:
            raise ValueError(f"unexpected text_config.{key}: {text.get(key)!r}")
    if text["layer_types"][0] != "sliding_attention":
        raise ValueError("layer 0 is not sliding attention")
    return text


def build_inputs(source: SafetensorsFile, text: dict, token_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    hidden = text["hidden_size"]
    layers = text["num_hidden_layers"]
    ple_size = text["hidden_size_per_layer_input"]
    vocabulary = text["vocab_size"]
    if not token_ids or any(token < 0 or token >= vocabulary for token in token_ids):
        raise ValueError("token IDs are outside the configured vocabulary")

    embedding_name = f"{MODEL_PREFIX}.embed_tokens.weight"
    ple_embedding_name = f"{MODEL_PREFIX}.embed_tokens_per_layer.weight"
    model_projection_name = f"{MODEL_PREFIX}.per_layer_model_projection.weight"
    projection_norm_name = f"{MODEL_PREFIX}.per_layer_projection_norm.weight"

    embedding_scale = bf16_scalar(math.sqrt(hidden))
    token_ple_scale = bf16_scalar(math.sqrt(ple_size))
    model_projection_scale = 1.0 / math.sqrt(hidden)
    input_scale = 1.0 / math.sqrt(2.0)

    inputs = []
    token_ple = []
    packed_ple_size = layers * ple_size
    for token_id in token_ids:
        inputs.append(load_range(source, embedding_name, token_id * hidden, hidden) * embedding_scale)
        token_ple.append(
            load_range(source, ple_embedding_name, token_id * packed_ple_size, ple_size)
            * token_ple_scale
        )
    inputs_array = np.stack(inputs).astype(np.float32)
    token_ple_array = np.stack(token_ple).astype(np.float32)

    projection = load_range(source, model_projection_name, 0, ple_size * hidden).reshape(
        ple_size, hidden
    )
    projection_norm = load_tensor(source, projection_norm_name)
    contextual = linear(inputs_array, projection) * np.float32(model_projection_scale)
    contextual = rms_norm(contextual, projection_norm, text["rms_norm_eps"])
    per_layer_input = ((contextual + token_ple_array) * np.float32(input_scale)).astype(np.float32)
    return inputs_array, per_layer_input


def run_layer(text: dict, tensors: dict[str, np.ndarray]) -> None:
    sequence_length = tensors["input"].shape[0]
    query_heads = text["num_attention_heads"]
    kv_heads = text["num_key_value_heads"]
    head_dim = text["head_dim"]
    epsilon = text["rms_norm_eps"]

    normalized = rms_norm(tensors["input"], tensors["input_norm"], epsilon)
    query = linear(normalized, tensors["q_proj"]).reshape(sequence_length, query_heads, head_dim)
    key = linear(normalized, tensors["k_proj"]).reshape(sequence_length, kv_heads, head_dim)
    value = linear(normalized, tensors["v_proj"]).reshape(sequence_length, kv_heads, head_dim)
    query = rms_norm(query, tensors["q_norm"], epsilon)
    key = rms_norm(key, tensors["k_norm"], epsilon)
    value = rms_norm(value, None, epsilon)
    rope_theta = text["rope_parameters"]["sliding_attention"]["rope_theta"]
    query = apply_rope(query, rope_theta)
    key = apply_rope(key, rope_theta)

    attention_heads = attention_mqa(query, key, value, text["sliding_window"])
    attention = linear(attention_heads.reshape(sequence_length, -1), tensors["o_proj"])
    after_attention = (
        tensors["input"] + rms_norm(attention, tensors["post_attention_norm"], epsilon)
    ).astype(np.float32)

    normalized_mlp = rms_norm(after_attention, tensors["pre_feedforward_norm"], epsilon)
    gate = linear(normalized_mlp, tensors["gate_proj"])
    up = linear(normalized_mlp, tensors["up_proj"])
    mlp = linear(gelu_pytorch_tanh(gate) * up, tensors["down_proj"])
    after_mlp = (
        after_attention + rms_norm(mlp, tensors["post_feedforward_norm"], epsilon)
    ).astype(np.float32)

    ple_gate = gelu_pytorch_tanh(linear(after_mlp, tensors["ple_gate"]))
    ple = linear(ple_gate * tensors["per_layer_input"], tensors["ple_projection"])
    output = (
        (after_mlp + rms_norm(ple, tensors["post_ple_norm"], epsilon))
        * tensors["layer_scalar"]
    ).astype(np.float32)

    tensors.update(
        {
            "expected_normalized_input": normalized,
            "expected_query": query.reshape(sequence_length, -1),
            "expected_key": key.reshape(sequence_length, -1),
            "expected_value": value.reshape(sequence_length, -1),
            "expected_attention": attention,
            "expected_after_attention": after_attention,
            "expected_mlp": mlp,
            "expected_after_mlp": after_mlp,
            "expected_ple": ple,
            "expected_output": output,
        }
    )


def write_fixture(path: Path, text: dict, tensors: dict[str, np.ndarray]) -> None:
    sequence_length = tensors["input"].shape[0]
    layer_scalar = float(tensors["layer_scalar"])
    header = struct.pack(
        "<8s10I3f",
        MAGIC,
        VERSION,
        sequence_length,
        text["hidden_size"],
        text["num_attention_heads"],
        text["num_key_value_heads"],
        text["head_dim"],
        text["intermediate_size"],
        text["hidden_size_per_layer_input"],
        text["sliding_window"],
        TENSOR_COUNT,
        text["rms_norm_eps"],
        text["rope_parameters"]["sliding_attention"]["rope_theta"],
        layer_scalar,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as output:
        output.write(header)
        for name in FIXTURE_ORDER:
            output.write(np.asarray(tensors[name], dtype="<f4").tobytes(order="C"))
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--token-ids", default="2,1")
    parser.add_argument("--skip-checkpoint-sha256", action="store_true")
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    pins = json.loads((repository / "models/gemma-4-e2b/pins.json").read_text())
    expected_config_hash = pins["model"]["files"]["config.json"]
    expected_checkpoint_hash = pins["model"]["files"]["model.safetensors"]
    if sha256_file(args.config) != expected_config_hash:
        raise ValueError("config.json does not match the pinned revision")
    if args.checkpoint.stat().st_size != pins["model"]["model_bytes"]:
        raise ValueError("model.safetensors size does not match the pinned revision")
    if not args.skip_checkpoint_sha256 and sha256_file(args.checkpoint) != expected_checkpoint_hash:
        raise ValueError("model.safetensors SHA-256 does not match the pinned revision")

    config = json.loads(args.config.read_text())
    text = validate_config(config)
    token_ids = [int(value) for value in args.token_ids.split(",")]

    with SafetensorsFile(args.checkpoint) as source:
        tensors = {}
        for logical_name, checkpoint_name in WEIGHT_NAMES.items():
            info = source.tensors.get(checkpoint_name)
            if info is None:
                raise KeyError(f"checkpoint is missing {checkpoint_name}")
            if info.dtype != "BF16" or info.shape != EXPECTED_SHAPES[logical_name]:
                raise ValueError(
                    f"unexpected {checkpoint_name}: dtype={info.dtype}, shape={info.shape}"
                )
            tensors[logical_name] = load_tensor(source, checkpoint_name)
        layer_scalar_name = f"{LAYER_PREFIX}.layer_scalar"
        tensors["layer_scalar"] = np.float32(load_tensor(source, layer_scalar_name).reshape(-1)[0])
        tensors["input"], tensors["per_layer_input"] = build_inputs(source, text, token_ids)

    run_layer(text, tensors)
    write_fixture(args.output, text, tensors)
    print(
        json.dumps(
            {
                "fixture": str(args.output),
                "bytes": args.output.stat().st_size,
                "sha256": sha256_file(args.output),
                "checkpoint_sha256": expected_checkpoint_hash,
                "token_ids": token_ids,
                "layer": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
