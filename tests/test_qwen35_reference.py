"""Tests for the Qwen3.5 NumPy reference.

The invariant tests always run. The differential test compares the
reference against the pinned transformers qwen3_5 implementation on a
tiny random model and is skipped when torch/transformers are absent, so
`make test` stays green in minimal environments.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compiler"))

from qwen35_reference import (  # noqa: E402
    Boundaries,
    Qwen35TextConfig,
    causal_conv1d,
    forward_text,
    gated_delta_net,
    rms_norm,
    rope_tables,
    apply_rope,
    score_answers,
    silu,
)

TINY = Qwen35TextConfig(
    hidden_size=64,
    num_hidden_layers=4,
    full_attention_interval=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    intermediate_size=96,
    linear_num_key_heads=4,
    linear_num_value_heads=4,
    linear_key_head_dim=8,
    linear_value_head_dim=8,
    linear_conv_kernel_dim=4,
    vocab_size=128,
)


def tiny_weights(config: Qwen35TextConfig, rng: np.random.Generator) -> dict:
    def normal(*shape):
        return rng.normal(0.0, 0.05, size=shape).astype(np.float32)

    key_dim = config.linear_num_key_heads * config.linear_key_head_dim
    value_dim = config.linear_num_value_heads * config.linear_value_head_dim
    conv_dim = 2 * key_dim + value_dim
    weights = {
        "embed_tokens.weight": normal(config.vocab_size, config.hidden_size),
        "norm.weight": normal(config.hidden_size),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"layers.{layer}"
        weights[f"{prefix}.input_layernorm.weight"] = normal(config.hidden_size)
        weights[f"{prefix}.post_attention_layernorm.weight"] = normal(config.hidden_size)
        weights[f"{prefix}.mlp.gate_proj.weight"] = normal(
            config.intermediate_size, config.hidden_size
        )
        weights[f"{prefix}.mlp.up_proj.weight"] = normal(
            config.intermediate_size, config.hidden_size
        )
        weights[f"{prefix}.mlp.down_proj.weight"] = normal(
            config.hidden_size, config.intermediate_size
        )
        if config.layer_type(layer) == "linear_attention":
            linear = f"{prefix}.linear_attn"
            weights[f"{linear}.in_proj_qkv.weight"] = normal(conv_dim, config.hidden_size)
            weights[f"{linear}.in_proj_z.weight"] = normal(value_dim, config.hidden_size)
            weights[f"{linear}.in_proj_b.weight"] = normal(
                config.linear_num_value_heads, config.hidden_size
            )
            weights[f"{linear}.in_proj_a.weight"] = normal(
                config.linear_num_value_heads, config.hidden_size
            )
            weights[f"{linear}.conv1d.weight"] = normal(
                conv_dim, 1, config.linear_conv_kernel_dim
            )
            weights[f"{linear}.dt_bias"] = np.ones(
                config.linear_num_value_heads, dtype=np.float32
            )
            weights[f"{linear}.A_log"] = np.log(
                rng.uniform(0.5, 4.0, config.linear_num_value_heads).astype(np.float32)
            )
            weights[f"{linear}.norm.weight"] = np.ones(
                config.linear_value_head_dim, dtype=np.float32
            )
            weights[f"{linear}.out_proj.weight"] = normal(config.hidden_size, value_dim)
        else:
            attn = f"{prefix}.self_attn"
            weights[f"{attn}.q_proj.weight"] = normal(
                config.num_attention_heads * config.head_dim * 2, config.hidden_size
            )
            weights[f"{attn}.k_proj.weight"] = normal(
                config.num_key_value_heads * config.head_dim, config.hidden_size
            )
            weights[f"{attn}.v_proj.weight"] = normal(
                config.num_key_value_heads * config.head_dim, config.hidden_size
            )
            weights[f"{attn}.o_proj.weight"] = normal(
                config.hidden_size, config.num_attention_heads * config.head_dim
            )
            weights[f"{attn}.q_norm.weight"] = normal(config.head_dim)
            weights[f"{attn}.k_norm.weight"] = normal(config.head_dim)
    return weights


class InvariantTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.weights = tiny_weights(TINY, self.rng)

    def test_rms_norm_matches_formula(self):
        x = self.rng.normal(size=(3, 8)).astype(np.float32)
        w = self.rng.normal(size=8).astype(np.float32)
        expected = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6) * (1.0 + w)
        np.testing.assert_allclose(rms_norm(x, w, 1e-6), expected, rtol=1e-6)

    def test_causal_conv1d_matches_bruteforce(self):
        x = self.rng.normal(size=(6, 5)).astype(np.float32)
        w = self.rng.normal(size=(5, 4)).astype(np.float32)
        got = causal_conv1d(x, w)
        padded = np.concatenate([np.zeros((3, 5), dtype=np.float32), x])
        for t in range(6):
            for c in range(5):
                acc = sum(padded[t + tap, c] * w[c, tap] for tap in range(4))
                self.assertAlmostEqual(got[t, c], silu(np.float32(acc)), places=5)

    def test_rope_position_zero_is_identity(self):
        cos, sin = rope_tables(TINY, np.array([0]))
        states = self.rng.normal(size=(1, 2, TINY.head_dim)).astype(np.float32)
        np.testing.assert_allclose(apply_rope(states, cos, sin), states, rtol=1e-6)

    def test_deltanet_zero_input_gives_zero_output(self):
        # Zero hidden input zeroes every projection, the conv (silu(0)=0),
        # q/k/v, the recurrent state, and the silu(z) output gate exactly.
        hidden = np.zeros((5, TINY.hidden_size), dtype=np.float32)
        out = gated_delta_net(TINY, self.weights, "layers.0.linear_attn", hidden, Boundaries())
        np.testing.assert_array_equal(out, np.zeros_like(out))

    def test_forward_shapes_and_determinism(self):
        tokens = [1, 5, 9, 2]
        first = forward_text(TINY, self.weights, tokens)
        second = forward_text(TINY, self.weights, tokens)
        self.assertEqual(first.shape, (4, TINY.hidden_size))
        np.testing.assert_array_equal(first, second)

    def test_score_answers_matches_tied_head(self):
        tokens = [3, 4, 5]
        hidden = forward_text(TINY, self.weights, tokens)
        scores = score_answers(TINY, self.weights, tokens, [7, 11])
        expected = self.weights["embed_tokens.weight"][[7, 11]] @ hidden[-1]
        np.testing.assert_allclose(scores, expected, rtol=1e-6)

    def test_boundary_capture_names(self):
        boundaries = Boundaries(capture=True)
        forward_text(TINY, self.weights, [1, 2, 3], boundaries)
        for name in (
            "embedding",
            "layers.0.linear_attn.post_conv",
            "layers.0.linear_attn.gate",
            "layers.0.linear_attn.core_out",
            "layers.0.linear_attn.state",
            "layers.3.self_attn.query",
            "layers.3.self_attn.attention",
            "final_norm",
        ):
            self.assertIn(name, boundaries.values)


def _load_torch_model():
    import torch  # noqa: F401
    from transformers import Qwen3_5TextConfig as HFConfig
    from transformers import Qwen3_5TextModel

    hf_config = HFConfig(
        vocab_size=TINY.vocab_size,
        hidden_size=TINY.hidden_size,
        intermediate_size=TINY.intermediate_size,
        num_hidden_layers=TINY.num_hidden_layers,
        num_attention_heads=TINY.num_attention_heads,
        num_key_value_heads=TINY.num_key_value_heads,
        head_dim=TINY.head_dim,
        linear_num_key_heads=TINY.linear_num_key_heads,
        linear_num_value_heads=TINY.linear_num_value_heads,
        linear_key_head_dim=TINY.linear_key_head_dim,
        linear_value_head_dim=TINY.linear_value_head_dim,
        linear_conv_kernel_dim=TINY.linear_conv_kernel_dim,
        layer_types=[TINY.layer_type(i) for i in range(TINY.num_hidden_layers)],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": TINY.rope_theta,
            "partial_rotary_factor": TINY.partial_rotary_factor,
            "mrope_section": [3, 3, 2],
            "mrope_interleaved": True,
        },
        rms_norm_eps=TINY.rms_norm_eps,
        attn_implementation="eager",
    )
    model = Qwen3_5TextModel(hf_config)
    model.eval()
    return model


class DifferentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.model = _load_torch_model()
        except Exception as error:  # torch or pinned transformers unavailable
            raise unittest.SkipTest(f"differential oracle unavailable: {error}")

    def test_matches_transformers_forward(self):
        import torch

        state = {
            name: parameter.detach().numpy().astype(np.float32)
            for name, parameter in self.model.state_dict().items()
        }
        tokens = [2, 17, 40, 3, 89, 21, 55]
        with torch.no_grad():
            expected = self.model(
                input_ids=torch.tensor([tokens]), use_cache=False
            ).last_hidden_state[0].numpy()
        got = forward_text(TINY, state, tokens)
        np.testing.assert_allclose(got, expected, rtol=2e-4, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
