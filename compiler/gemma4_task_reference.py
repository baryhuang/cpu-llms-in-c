"""Streaming-weight NumPy reference for the pinned Gemma 4 E2B text graph."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable

import numpy as np

try:
    from .safetensors_file import SafetensorsFile
except ImportError:
    from safetensors_file import SafetensorsFile


MODEL_PREFIX = "model.language_model"


def bf16_scalar(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]


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


def load_rows(source: SafetensorsFile, name: str, row_ids: Iterable[int]) -> np.ndarray:
    info = source.tensors[name]
    if len(info.shape) != 2:
        raise ValueError(f"{name} is not a matrix")
    width = info.shape[1]
    rows = []
    for row_id in row_ids:
        if row_id < 0 or row_id >= info.shape[0]:
            raise ValueError(f"row {row_id} is outside {name}")
        payload = source.read_element_bytes(name, row_id * width, width)
        rows.append(decode_float32(payload, info.dtype, (width,)))
    return np.stack(rows).astype(np.float32)


def rms_norm(values: np.ndarray, scale: np.ndarray | None, epsilon: float) -> np.ndarray:
    squares = np.mean(values.astype(np.float32) ** 2, axis=-1, keepdims=True)
    output = values.astype(np.float32) * np.power(squares + epsilon, -0.5)
    if scale is not None:
        output *= scale.astype(np.float32)
    return output.astype(np.float32)


def gelu_pytorch_tanh(values: np.ndarray) -> np.ndarray:
    x = values.astype(np.float32)
    inner = np.float32(0.7978845608028654) * (x + np.float32(0.044715) * x * x * x)
    return (np.float32(0.5) * x * (np.float32(1.0) + np.tanh(inner))).astype(np.float32)


def linear(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.matmul(values.astype(np.float32), weights.astype(np.float32).T).astype(np.float32)


def apply_rope(states: np.ndarray, positions: np.ndarray, theta: float, rotary_fraction: float) -> np.ndarray:
    head_dim = states.shape[-1]
    half = head_dim // 2
    rotated = int(rotary_fraction * half)
    inverse_frequency = np.zeros(half, dtype=np.float32)
    if rotated:
        indices = np.arange(rotated, dtype=np.float32) * np.float32(2.0)
        inverse_frequency[:rotated] = np.float32(1.0) / np.power(
            np.float32(theta), indices / np.float32(head_dim)
        )
    angles = positions.astype(np.float32)[:, None] * inverse_frequency[None, :]
    cosine = np.cos(angles).astype(np.float32)
    sine = np.sin(angles).astype(np.float32)
    first = states[..., :half].astype(np.float32)
    second = states[..., half:].astype(np.float32)
    output = np.empty_like(states, dtype=np.float32)
    output[..., :half] = first * cosine[None, :, None, :] - second * sine[None, :, None, :]
    output[..., half:] = second * cosine[None, :, None, :] + first * sine[None, :, None, :]
    return output


def causal_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    lengths: list[int],
    window: int | None,
) -> np.ndarray:
    batch, sequence, heads, head_dim = query.shape
    output = np.zeros_like(query, dtype=np.float32)
    key_values = key[:, :, 0, :]
    value_values = value[:, :, 0, :]
    for batch_index, length in enumerate(lengths):
        for head in range(heads):
            scores = np.matmul(
                query[batch_index, :length, head], key_values[batch_index, :length].T
            ).astype(np.float32)
            row = np.arange(length)[:, None]
            column = np.arange(length)[None, :]
            allowed = column <= row
            if window is not None:
                allowed &= column > row - window
            scores[~allowed] = -np.inf
            maximum = np.max(scores, axis=-1, keepdims=True)
            probabilities = np.exp(scores - maximum).astype(np.float32)
            probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
            output[batch_index, :length, head] = np.matmul(
                probabilities, value_values[batch_index, :length]
            ).astype(np.float32)
    return output


def validate_text_config(config: dict) -> dict:
    text = config["text_config"]
    expected = {
        "hidden_size": 1536,
        "num_hidden_layers": 35,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "global_head_dim": 512,
        "intermediate_size": 6144,
        "hidden_size_per_layer_input": 256,
        "num_kv_shared_layers": 20,
        "sliding_window": 512,
        "enable_moe_block": False,
        "hidden_activation": "gelu_pytorch_tanh",
    }
    for key, value in expected.items():
        if text.get(key) != value:
            raise ValueError(f"unexpected text_config.{key}: {text.get(key)!r}")
    if len(text["layer_types"]) != text["num_hidden_layers"]:
        raise ValueError("layer_types length does not match num_hidden_layers")
    return text


def build_compiled_token_rows(
    source: SafetensorsFile, text: dict, token_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return scaled main embeddings and fully folded PLE rows for token_ids."""
    hidden = text["hidden_size"]
    layers = text["num_hidden_layers"]
    ple_size = text["hidden_size_per_layer_input"]
    main = load_rows(source, f"{MODEL_PREFIX}.embed_tokens.weight", token_ids)
    main *= np.float32(bf16_scalar(math.sqrt(hidden)))

    token_ple = load_rows(source, f"{MODEL_PREFIX}.embed_tokens_per_layer.weight", token_ids)
    token_ple = token_ple.reshape(len(token_ids), layers, ple_size)
    token_ple *= np.float32(bf16_scalar(math.sqrt(ple_size)))

    projection = load_tensor(source, f"{MODEL_PREFIX}.per_layer_model_projection.weight")
    contextual = linear(main, projection) * np.float32(hidden**-0.5)
    contextual = contextual.reshape(len(token_ids), layers, ple_size)
    projection_norm = load_tensor(source, f"{MODEL_PREFIX}.per_layer_projection_norm.weight")
    contextual = rms_norm(contextual, projection_norm, text["rms_norm_eps"])
    folded = (contextual + token_ple) * np.float32(2.0**-0.5)
    return main.astype(np.float32), folded.astype(np.float32)


def gather_profile_inputs(
    sequences: list[list[int]],
    token_ids: list[int],
    embeddings: np.ndarray,
    folded_ple: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    token_to_row = {token: index for index, token in enumerate(token_ids)}
    lengths = [len(sequence) for sequence in sequences]
    maximum = max(lengths)
    hidden = embeddings.shape[1]
    layers = folded_ple.shape[1]
    ple_size = folded_ple.shape[2]
    inputs = np.zeros((len(sequences), maximum, hidden), dtype=np.float32)
    ple = np.zeros((len(sequences), maximum, layers, ple_size), dtype=np.float32)
    for batch_index, sequence in enumerate(sequences):
        rows = [token_to_row[token] for token in sequence]
        inputs[batch_index, : len(rows)] = embeddings[rows]
        ple[batch_index, : len(rows)] = folded_ple[rows]
    return inputs, ple, lengths


def forward_candidates(
    source: SafetensorsFile,
    text: dict,
    hidden_states: np.ndarray,
    per_layer_inputs: np.ndarray,
    lengths: list[int],
    candidate_rows: np.ndarray,
    progress=None,
) -> tuple[np.ndarray, np.ndarray]:
    epsilon = text["rms_norm_eps"]
    batch, sequence, hidden = hidden_states.shape
    positions = np.arange(sequence, dtype=np.float32)
    shared: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for layer in range(text["num_hidden_layers"]):
        prefix = f"{MODEL_PREFIX}.layers.{layer}"
        layer_type = text["layer_types"][layer]
        is_sliding = layer_type == "sliding_attention"
        head_dim = text["head_dim"] if is_sliding else text["global_head_dim"]
        intermediate = text["intermediate_size"] * (2 if layer >= 15 else 1)

        residual = hidden_states
        normalized = rms_norm(
            hidden_states, load_tensor(source, f"{prefix}.input_layernorm.weight"), epsilon
        )
        flat = normalized.reshape(batch * sequence, hidden)
        query = linear(flat, load_tensor(source, f"{prefix}.self_attn.q_proj.weight"))
        query = query.reshape(batch, sequence, text["num_attention_heads"], head_dim)
        query = rms_norm(
            query, load_tensor(source, f"{prefix}.self_attn.q_norm.weight"), epsilon
        )
        rope = text["rope_parameters"][layer_type]
        query = apply_rope(
            query,
            positions,
            rope["rope_theta"],
            rope.get("partial_rotary_factor", 1.0),
        )

        if layer >= 15:
            key, value = shared[layer_type]
        else:
            key = linear(flat, load_tensor(source, f"{prefix}.self_attn.k_proj.weight"))
            value = linear(flat, load_tensor(source, f"{prefix}.self_attn.v_proj.weight"))
            key = key.reshape(batch, sequence, 1, head_dim)
            value = value.reshape(batch, sequence, 1, head_dim)
            key = rms_norm(
                key, load_tensor(source, f"{prefix}.self_attn.k_norm.weight"), epsilon
            )
            value = rms_norm(value, None, epsilon)
            key = apply_rope(
                key,
                positions,
                rope["rope_theta"],
                rope.get("partial_rotary_factor", 1.0),
            )
            if layer in (13, 14):
                shared[layer_type] = (key, value)

        attended = causal_attention(
            query, key, value, lengths, text["sliding_window"] if is_sliding else None
        )
        attention = linear(
            attended.reshape(batch * sequence, text["num_attention_heads"] * head_dim),
            load_tensor(source, f"{prefix}.self_attn.o_proj.weight"),
        ).reshape(batch, sequence, hidden)
        attention = rms_norm(
            attention,
            load_tensor(source, f"{prefix}.post_attention_layernorm.weight"),
            epsilon,
        )
        hidden_states = (residual + attention).astype(np.float32)

        residual = hidden_states
        normalized = rms_norm(
            hidden_states,
            load_tensor(source, f"{prefix}.pre_feedforward_layernorm.weight"),
            epsilon,
        )
        flat = normalized.reshape(batch * sequence, hidden)
        gate = linear(flat, load_tensor(source, f"{prefix}.mlp.gate_proj.weight"))
        up = linear(flat, load_tensor(source, f"{prefix}.mlp.up_proj.weight"))
        if gate.shape[1] != intermediate or up.shape[1] != intermediate:
            raise ValueError(f"unexpected MLP width in layer {layer}")
        mlp = linear(
            gelu_pytorch_tanh(gate) * up,
            load_tensor(source, f"{prefix}.mlp.down_proj.weight"),
        ).reshape(batch, sequence, hidden)
        mlp = rms_norm(
            mlp, load_tensor(source, f"{prefix}.post_feedforward_layernorm.weight"), epsilon
        )
        hidden_states = (residual + mlp).astype(np.float32)

        residual = hidden_states
        ple_gate = linear(
            hidden_states.reshape(batch * sequence, hidden),
            load_tensor(source, f"{prefix}.per_layer_input_gate.weight"),
        )
        ple_gate = gelu_pytorch_tanh(ple_gate).reshape(batch, sequence, -1)
        ple = linear(
            (ple_gate * per_layer_inputs[:, :, layer, :]).reshape(batch * sequence, -1),
            load_tensor(source, f"{prefix}.per_layer_projection.weight"),
        ).reshape(batch, sequence, hidden)
        ple = rms_norm(
            ple, load_tensor(source, f"{prefix}.post_per_layer_input_norm.weight"), epsilon
        )
        scalar = float(load_tensor(source, f"{prefix}.layer_scalar").reshape(-1)[0])
        hidden_states = ((residual + ple) * np.float32(scalar)).astype(np.float32)
        if progress is not None:
            progress(layer + 1, text["num_hidden_layers"])

    hidden_states = rms_norm(
        hidden_states, load_tensor(source, f"{MODEL_PREFIX}.norm.weight"), epsilon
    )
    final_states = np.stack(
        [hidden_states[index, length - 1] for index, length in enumerate(lengths)]
    )
    logits = linear(final_states, candidate_rows)
    cap = np.float32(text["final_logit_softcapping"])
    logits = np.tanh(logits / cap).astype(np.float32) * cap
    return logits, final_states
