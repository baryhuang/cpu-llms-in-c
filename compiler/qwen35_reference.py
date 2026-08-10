#!/usr/bin/env python3
"""Independent NumPy reference for the Qwen3.5-0.8B text graph.

Implements the pinned transformers qwen3_5 oracle (see
models/qwen3.5-0.8b/pins.json) with plain NumPy: Gated-DeltaNet linear
attention, full attention with an output gate, SwiGLU MLP, and the
(1 + weight) RMSNorm variant. Text-only inputs make every MRoPE section
share one position index, so rotary embedding is computed as standard
RoPE over the rotary dimensions; this is the exact rewrite recorded in
models/qwen3.5-0.8b/README.md.

Weights are a flat dict of NumPy arrays keyed by the checkpoint tensor
names relative to the text model, e.g. "layers.0.linear_attn.A_log".
All arithmetic is float32, matching mamba_ssm_dtype and the fp32 gate
math in the oracle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Qwen35TextConfig:
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    full_attention_interval: int = 4
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    intermediate_size: int = 3584
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1e7
    partial_rotary_factor: float = 0.25

    def layer_type(self, layer: int) -> str:
        if (layer + 1) % self.full_attention_interval == 0:
            return "full_attention"
        return "linear_attention"


@dataclass
class Boundaries:
    """Optional per-tensor capture used by fixtures and differential tests."""

    capture: bool = False
    values: dict = field(default_factory=dict)

    def record(self, name: str, value: np.ndarray) -> None:
        if self.capture:
            self.values[name] = np.array(value, dtype=np.float32, copy=True)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x = x.astype(np.float32)
    variance = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(variance + eps) * (1.0 + weight.astype(np.float32))


def rms_norm_gated(x: np.ndarray, weight: np.ndarray, gate: np.ndarray, eps: float) -> np.ndarray:
    x = x.astype(np.float32)
    variance = np.mean(x * x, axis=-1, keepdims=True)
    normalized = x / np.sqrt(variance + eps) * weight.astype(np.float32)
    return normalized * silu(gate.astype(np.float32))


def l2_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.sqrt(np.sum(x * x, axis=-1, keepdims=True) + eps)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def causal_conv1d(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Depthwise causal convolution over the sequence axis.

    x: [seq, channels]; weight: [channels, kernel]. Matches the oracle's
    grouped Conv1d with left padding kernel-1 and no bias, followed by SiLU.
    """
    seq_len, channels = x.shape
    kernel = weight.shape[1]
    padded = np.concatenate([np.zeros((kernel - 1, channels), dtype=np.float32), x], axis=0)
    out = np.zeros_like(x, dtype=np.float32)
    for tap in range(kernel):
        out += padded[tap : tap + seq_len] * weight[:, tap]
    return silu(out)


def rope_tables(config: Qwen35TextConfig, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin over the rotary dimensions for text-only (single-index) input."""
    rotary_dim = int(config.head_dim * config.partial_rotary_factor)
    inv_freq = 1.0 / (
        config.rope_theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim)
    )
    freqs = np.outer(positions.astype(np.float32), inv_freq)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb), np.sin(emb)


def apply_rope(states: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """states: [seq, heads, head_dim]; cos/sin: [seq, rotary_dim]."""
    rotary_dim = cos.shape[-1]
    rot = states[..., :rotary_dim]
    rest = states[..., rotary_dim:]
    half = rotary_dim // 2
    rotated = np.concatenate([-rot[..., half:], rot[..., :half]], axis=-1)
    rot = rot * cos[:, None, :] + rotated * sin[:, None, :]
    return np.concatenate([rot, rest], axis=-1)


def gated_delta_net(
    config: Qwen35TextConfig,
    weights: dict,
    prefix: str,
    hidden: np.ndarray,
    boundaries: Boundaries,
) -> np.ndarray:
    """Recurrent gated delta rule over [seq, hidden] float32 input."""
    seq_len = hidden.shape[0]
    num_heads = config.linear_num_value_heads
    head_k = config.linear_key_head_dim
    head_v = config.linear_value_head_dim
    key_dim = config.linear_num_key_heads * head_k
    value_dim = num_heads * head_v

    mixed_qkv = hidden @ weights[f"{prefix}.in_proj_qkv.weight"].T
    z = (hidden @ weights[f"{prefix}.in_proj_z.weight"].T).reshape(seq_len, num_heads, head_v)
    b = hidden @ weights[f"{prefix}.in_proj_b.weight"].T
    a = hidden @ weights[f"{prefix}.in_proj_a.weight"].T

    conv_weight = weights[f"{prefix}.conv1d.weight"].reshape(-1, config.linear_conv_kernel_dim)
    mixed_qkv = causal_conv1d(mixed_qkv, conv_weight)
    boundaries.record(f"{prefix}.post_conv", mixed_qkv)

    query = mixed_qkv[:, :key_dim].reshape(seq_len, -1, head_k)
    key = mixed_qkv[:, key_dim : 2 * key_dim].reshape(seq_len, -1, head_k)
    value = mixed_qkv[:, 2 * key_dim :].reshape(seq_len, num_heads, head_v)

    beta = 1.0 / (1.0 + np.exp(-b))
    gate = -np.exp(weights[f"{prefix}.A_log"].astype(np.float32)) * softplus(
        a + weights[f"{prefix}.dt_bias"].astype(np.float32)
    )
    boundaries.record(f"{prefix}.gate", gate)

    repeat = num_heads // config.linear_num_key_heads
    if repeat > 1:
        query = np.repeat(query, repeat, axis=1)
        key = np.repeat(key, repeat, axis=1)

    query = l2_norm(query) / np.sqrt(np.float32(head_k))
    key = l2_norm(key)

    state = np.zeros((num_heads, head_k, head_v), dtype=np.float32)
    core_out = np.zeros((seq_len, num_heads, head_v), dtype=np.float32)
    for t in range(seq_len):
        state *= np.exp(gate[t])[:, None, None]
        kv_mem = np.einsum("hkv,hk->hv", state, key[t])
        delta = (value[t] - kv_mem) * beta[t][:, None]
        state += key[t][:, :, None] * delta[:, None, :]
        core_out[t] = np.einsum("hkv,hk->hv", state, query[t])
    boundaries.record(f"{prefix}.core_out", core_out)
    boundaries.record(f"{prefix}.state", state)

    norm_weight = weights[f"{prefix}.norm.weight"]
    gated = rms_norm_gated(core_out, norm_weight, z, config.rms_norm_eps)
    return gated.reshape(seq_len, value_dim) @ weights[f"{prefix}.out_proj.weight"].T


def full_attention(
    config: Qwen35TextConfig,
    weights: dict,
    prefix: str,
    hidden: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    boundaries: Boundaries,
) -> np.ndarray:
    seq_len = hidden.shape[0]
    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    eps = config.rms_norm_eps

    q_and_gate = (hidden @ weights[f"{prefix}.q_proj.weight"].T).reshape(
        seq_len, heads, 2 * head_dim
    )
    query = q_and_gate[..., :head_dim]
    gate = q_and_gate[..., head_dim:]

    key = (hidden @ weights[f"{prefix}.k_proj.weight"].T).reshape(seq_len, kv_heads, head_dim)
    value = (hidden @ weights[f"{prefix}.v_proj.weight"].T).reshape(seq_len, kv_heads, head_dim)

    query = rms_norm(query, weights[f"{prefix}.q_norm.weight"], eps)
    key = rms_norm(key, weights[f"{prefix}.k_norm.weight"], eps)
    query = apply_rope(query, cos, sin)
    key = apply_rope(key, cos, sin)
    boundaries.record(f"{prefix}.query", query)
    boundaries.record(f"{prefix}.key", key)

    group = heads // kv_heads
    scale = 1.0 / np.sqrt(np.float32(head_dim))
    output = np.zeros((seq_len, heads, head_dim), dtype=np.float32)
    for head in range(heads):
        kv_head = head // group
        scores = (query[:, head] @ key[:, kv_head].T) * scale
        scores += np.triu(np.full((seq_len, seq_len), -np.inf, dtype=np.float32), k=1)
        scores -= scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs /= probs.sum(axis=-1, keepdims=True)
        output[:, head] = probs @ value[:, kv_head]
    boundaries.record(f"{prefix}.attention", output)

    output = output.reshape(seq_len, heads * head_dim)
    output *= 1.0 / (1.0 + np.exp(-gate.reshape(seq_len, heads * head_dim)))
    return output @ weights[f"{prefix}.o_proj.weight"].T


def mlp(config: Qwen35TextConfig, weights: dict, prefix: str, hidden: np.ndarray) -> np.ndarray:
    gate = silu(hidden @ weights[f"{prefix}.gate_proj.weight"].T)
    up = hidden @ weights[f"{prefix}.up_proj.weight"].T
    return (gate * up) @ weights[f"{prefix}.down_proj.weight"].T


def forward_text(
    config: Qwen35TextConfig,
    weights: dict,
    token_ids: list[int],
    boundaries: Boundaries | None = None,
) -> np.ndarray:
    """Run the full text graph; returns final hidden states [seq, hidden]."""
    boundaries = boundaries or Boundaries()
    positions = np.arange(len(token_ids))
    cos, sin = rope_tables(config, positions)

    hidden = weights["embed_tokens.weight"][token_ids].astype(np.float32)
    boundaries.record("embedding", hidden)
    for layer in range(config.num_hidden_layers):
        prefix = f"layers.{layer}"
        normed = rms_norm(hidden, weights[f"{prefix}.input_layernorm.weight"], config.rms_norm_eps)
        if config.layer_type(layer) == "linear_attention":
            mixed = gated_delta_net(config, weights, f"{prefix}.linear_attn", normed, boundaries)
        else:
            mixed = full_attention(
                config, weights, f"{prefix}.self_attn", normed, cos, sin, boundaries
            )
        hidden = hidden + mixed
        boundaries.record(f"{prefix}.after_mixer", hidden)

        normed = rms_norm(
            hidden, weights[f"{prefix}.post_attention_layernorm.weight"], config.rms_norm_eps
        )
        hidden = hidden + mlp(config, weights, f"{prefix}.mlp", normed)
        boundaries.record(f"{prefix}.after_mlp", hidden)

    hidden = rms_norm(hidden, weights["norm.weight"], config.rms_norm_eps)
    boundaries.record("final_norm", hidden)
    return hidden


def score_answers(
    config: Qwen35TextConfig,
    weights: dict,
    token_ids: list[int],
    answer_token_ids: list[int],
    boundaries: Boundaries | None = None,
) -> np.ndarray:
    """Prompt-defined decision: logits of the answer rows only (tied head)."""
    hidden = forward_text(config, weights, token_ids, boundaries)
    rows = weights["embed_tokens.weight"][answer_token_ids].astype(np.float32)
    return rows @ hidden[-1]
