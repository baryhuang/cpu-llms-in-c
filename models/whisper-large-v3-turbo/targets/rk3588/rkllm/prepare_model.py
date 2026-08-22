#!/usr/bin/env python3
"""Rewrite Whisper Turbo's decoder into RKLLM's exact RMSNorm schema.

Python is used only for offline model conversion.  Target inference remains a
native C/C++ process.
"""

import argparse
import json
import shutil
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


MAGIC = b"WRKLAUX1"
VERSION = 1
TOKENIZER_FILES = (
    "added_tokens.json",
    "generation_config.json",
    "merges.txt",
    "normalizer.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def centered_rows(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.float()
    return (value - value.mean(dim=-1, keepdim=True)).half().contiguous()


def centered_output(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.float()
    return (value - value.mean(dim=0, keepdim=True)).half().contiguous()


def centered_bias(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.float()
    return (value - value.mean()).half().contiguous()


def get_tensor(reader, name: str) -> torch.Tensor:
    return reader.get_tensor(name).cpu()


def projection_after_norm(reader, prefix: str, norm_bias: torch.Tensor):
    weight = get_tensor(reader, prefix + ".weight")
    if prefix + ".bias" in reader.keys():
        bias = get_tensor(reader, prefix + ".bias").float()
    else:
        bias = torch.zeros(weight.shape[0], dtype=torch.float32)
    bias = bias + weight.float().matmul(norm_bias.float())
    return weight.half().contiguous(), bias.half().contiguous()


def residual_projection(reader, prefix: str):
    weight = centered_output(get_tensor(reader, prefix + ".weight"))
    if prefix + ".bias" in reader.keys():
        bias = centered_bias(get_tensor(reader, prefix + ".bias"))
    else:
        bias = torch.zeros(weight.shape[0], dtype=torch.float16)
    return weight, bias


def write_auxiliary(path: Path, tokens: torch.Tensor, positions: torch.Tensor,
                    final_logit_bias: torch.Tensor) -> None:
    if tokens.dtype != torch.float16 or positions.dtype != torch.float16:
        raise ValueError("embedding tables must use float16 storage")
    with path.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack(
            "<IIIII",
            VERSION,
            tokens.shape[0],
            positions.shape[0],
            tokens.shape[1],
            final_logit_bias.numel(),
        ))
        stream.write(tokens.numpy().tobytes(order="C"))
        stream.write(positions.numpy().tobytes(order="C"))
        stream.write(final_logit_bias.float().numpy().tobytes(order="C"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source_weights = args.source / "model.safetensors"
    if not source_weights.is_file():
        raise FileNotFoundError(source_weights)
    args.output.mkdir(parents=True, exist_ok=True)

    state = {}
    with safe_open(source_weights, framework="pt", device="cpu") as reader:
        token_source = get_tensor(reader, "model.decoder.embed_tokens.weight")
        position_source = get_tensor(reader, "model.decoder.embed_positions.weight")
        token_centered = centered_rows(token_source)
        position_centered = centered_rows(position_source)
        state["model.embed_tokens.weight"] = token_centered

        for layer in range(4):
            source = f"model.decoder.layers.{layer}"
            target = f"model.layers.{layer}"

            self_norm_weight = get_tensor(reader, source + ".self_attn_layer_norm.weight")
            self_norm_bias = get_tensor(reader, source + ".self_attn_layer_norm.bias")
            cross_norm_weight = get_tensor(reader, source + ".encoder_attn_layer_norm.weight")
            cross_norm_bias = get_tensor(reader, source + ".encoder_attn_layer_norm.bias")
            ffn_norm_weight = get_tensor(reader, source + ".final_layer_norm.weight")
            ffn_norm_bias = get_tensor(reader, source + ".final_layer_norm.bias")
            state[target + ".input_layernorm.weight"] = self_norm_weight.half().contiguous()
            state[target + ".cross_layernorm.weight"] = cross_norm_weight.half().contiguous()
            state[target + ".post_attention_layernorm.weight"] = ffn_norm_weight.half().contiguous()

            qkv_weights = []
            qkv_biases = []
            for projection in ("q_proj", "k_proj", "v_proj"):
                weight, bias = projection_after_norm(
                    reader, source + ".self_attn." + projection, self_norm_bias)
                qkv_weights.append(weight)
                qkv_biases.append(bias)
            state[target + ".self_attn.qkv_proj.weight"] = torch.cat(
                qkv_weights, dim=0).contiguous()
            state[target + ".self_attn.qkv_proj.bias"] = torch.cat(
                qkv_biases, dim=0).contiguous()
            weight, bias = residual_projection(reader, source + ".self_attn.out_proj")
            state[target + ".self_attn.o_proj.weight"] = weight
            state[target + ".self_attn.o_proj.bias"] = bias

            weight, bias = projection_after_norm(
                reader, source + ".encoder_attn.q_proj", cross_norm_bias)
            state[target + ".cross_attn.cross_q_proj.weight"] = weight
            state[target + ".cross_attn.cross_q_proj.bias"] = bias
            weight, bias = residual_projection(reader, source + ".encoder_attn.out_proj")
            state[target + ".cross_attn.cross_o_proj.weight"] = weight
            state[target + ".cross_attn.cross_o_proj.bias"] = bias

            weight, bias = projection_after_norm(reader, source + ".fc1", ffn_norm_bias)
            state[target + ".mlp.up_proj.weight"] = weight
            state[target + ".mlp.up_proj.bias"] = bias
            weight, bias = residual_projection(reader, source + ".fc2")
            state[target + ".mlp.down_proj.weight"] = weight
            state[target + ".mlp.down_proj.bias"] = bias

        final_norm_weight = get_tensor(reader, "model.decoder.layer_norm.weight")
        final_norm_bias = get_tensor(reader, "model.decoder.layer_norm.bias")
        state["model.norm.weight"] = final_norm_weight.half().contiguous()
        # Whisper ties proj_out to the uncentered token embedding table.
        state["lm_head.weight"] = token_source.half().contiguous()
        final_logit_bias = token_source.float().matmul(final_norm_bias.float())

    save_file(state, args.output / "model.safetensors")
    write_auxiliary(
        args.output / "runtime_aux.bin",
        token_centered,
        position_centered,
        final_logit_bias,
    )

    script_dir = Path(__file__).resolve().parent
    for name in (
        "configuration_whisper_rkllm.py",
        "modeling_whisper_rkllm.py",
    ):
        shutil.copy2(script_dir / name, args.output / name)
    for name in TOKENIZER_FILES:
        source = args.source / name
        if source.is_file():
            shutil.copy2(source, args.output / name)

    config = {
        "architectures": ["WhisperRKLLMForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_whisper_rkllm.WhisperRKLLMConfig",
            "AutoModel": "modeling_whisper_rkllm.WhisperRKLLMModel",
            "AutoModelForCausalLM": "modeling_whisper_rkllm.WhisperRKLLMForCausalLM",
        },
        "model_type": "whisper_rkllm",
        "vocab_size": 51866,
        "hidden_size": 1280,
        "intermediate_size": 5120,
        "num_hidden_layers": 4,
        "num_attention_heads": 20,
        "num_key_value_heads": 20,
        "max_position_embeddings": 448,
        "hidden_act": "gelu",
        "rms_norm_eps": 1.0e-5,
        "attention_bias": True,
        "no_rope_layer_interval": 1,
        "rope_theta": 10000.0,
        "bos_token_id": 50257,
        "eos_token_id": 50257,
        "pad_token_id": 50257,
        "decoder_start_token_id": 50258,
        "tie_word_embeddings": False,
        "torch_dtype": "float16",
        "transformers_version": "5.8.0",
        "use_cache": True,
    }
    (args.output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output / 'model.safetensors'}")
    print(f"wrote {args.output / 'runtime_aux.bin'}")


if __name__ == "__main__":
    main()
