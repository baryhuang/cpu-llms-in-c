"""Cross-check the committed Qwen3.5 layer fixture against the reference.

The fixture generator computes in double precision with explicit float32
casts; the NumPy reference computes in float32. This test ties them
together within float tolerance, closing the chain: the C layer matches
the fixture exactly, the fixture matches the reference here, and the
reference matches the pinned transformers oracle differentially.
"""

import struct
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "compiler"))

from qwen35_reference import (  # noqa: E402
    Boundaries,
    Qwen35TextConfig,
    full_attention,
    gated_delta_net,
    rope_tables,
)

FIXTURE = REPOSITORY / "tests/fixtures/qwen35_layer_v1.bin"
HEADER = struct.Struct("<8s13I2f")


def load_fixture():
    if not FIXTURE.exists():
        subprocess.run(
            [
                sys.executable,
                str(REPOSITORY / "compiler/generate_qwen35_layer_fixture.py"),
                "--output",
                str(FIXTURE),
            ],
            check=True,
        )
    raw = FIXTURE.read_bytes()
    fields = HEADER.unpack_from(raw)
    (magic, version, seq, hidden, lk_heads, lv_heads, lk_dim, lv_dim, kernel,
     q_heads, kv_heads, head_dim, rotary, inter, epsilon, theta) = fields
    assert magic == b"QW35LYR1" and version == 1
    key_dim = lk_heads * lk_dim
    value_dim = lv_heads * lv_dim
    conv_dim = 2 * key_dim + value_dim
    query_size = q_heads * head_dim
    kv_size = kv_heads * head_dim
    sizes = [
        ("input", seq * hidden),
        ("l0_input_norm", hidden),
        ("l0_in_proj_qkv", conv_dim * hidden),
        ("l0_in_proj_z", value_dim * hidden),
        ("l0_in_proj_b", lv_heads * hidden),
        ("l0_in_proj_a", lv_heads * hidden),
        ("l0_conv", conv_dim * kernel),
        ("l0_a_log", lv_heads),
        ("l0_dt_bias", lv_heads),
        ("l0_gated_norm", lv_dim),
        ("l0_out_proj", hidden * value_dim),
        ("l0_post_norm", hidden),
        ("l0_gate_proj", inter * hidden),
        ("l0_up_proj", inter * hidden),
        ("l0_down_proj", hidden * inter),
        ("l1_input_norm", hidden),
        ("l1_q_proj", 2 * query_size * hidden),
        ("l1_k_proj", kv_size * hidden),
        ("l1_v_proj", kv_size * hidden),
        ("l1_q_norm", head_dim),
        ("l1_k_norm", head_dim),
        ("l1_o_proj", hidden * query_size),
        ("l1_post_norm", hidden),
        ("l1_gate_proj", inter * hidden),
        ("l1_up_proj", inter * hidden),
        ("l1_down_proj", hidden * inter),
        ("expected_l0_normed", seq * hidden),
        ("expected_l0_post_conv", seq * conv_dim),
        ("expected_l0_gate", seq * lv_heads),
        ("expected_l0_core_out", seq * value_dim),
        ("expected_l0_state", lv_heads * lk_dim * lv_dim),
        ("expected_l0_gated", seq * value_dim),
        ("expected_l0_mixer", seq * hidden),
        ("expected_l0_after_mixer", seq * hidden),
        ("expected_l0_after_mlp", seq * hidden),
        ("expected_l1_normed", seq * hidden),
        ("expected_l1_query", seq * query_size),
        ("expected_l1_key", seq * kv_size),
        ("expected_l1_attention", seq * query_size),
        ("expected_l1_gated_attention", seq * query_size),
        ("expected_l1_mixer", seq * hidden),
        ("expected_l1_after_mixer", seq * hidden),
        ("expected_output", seq * hidden),
    ]
    tensors = {}
    offset = HEADER.size
    for name, count in sizes:
        tensors[name] = np.frombuffer(raw, dtype="<f4", count=count, offset=offset).copy()
        offset += count * 4
    assert offset == len(raw)
    config = Qwen35TextConfig(
        hidden_size=hidden,
        num_attention_heads=q_heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        intermediate_size=inter,
        linear_num_key_heads=lk_heads,
        linear_num_value_heads=lv_heads,
        linear_key_head_dim=lk_dim,
        linear_value_head_dim=lv_dim,
        linear_conv_kernel_dim=kernel,
        rms_norm_eps=epsilon,
        rope_theta=theta,
        partial_rotary_factor=rotary / head_dim,
    )
    return config, tensors, (seq, kernel, conv_dim)


class FixtureAgreesWithReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.tensors, (cls.seq, kernel, cls.conv_dim) = load_fixture()

    def test_deltanet_boundaries(self):
        t = self.tensors
        weights = {
            "l0.in_proj_qkv.weight": t["l0_in_proj_qkv"].reshape(self.conv_dim, -1),
            "l0.in_proj_z.weight": t["l0_in_proj_z"].reshape(-1, self.config.hidden_size),
            "l0.in_proj_b.weight": t["l0_in_proj_b"].reshape(-1, self.config.hidden_size),
            "l0.in_proj_a.weight": t["l0_in_proj_a"].reshape(-1, self.config.hidden_size),
            "l0.conv1d.weight": t["l0_conv"].reshape(
                self.conv_dim, 1, self.config.linear_conv_kernel_dim
            ),
            "l0.A_log": t["l0_a_log"],
            "l0.dt_bias": t["l0_dt_bias"],
            "l0.norm.weight": t["l0_gated_norm"],
            "l0.out_proj.weight": t["l0_out_proj"].reshape(self.config.hidden_size, -1),
        }
        normed = t["expected_l0_normed"].reshape(self.seq, self.config.hidden_size)
        boundaries = Boundaries(capture=True)
        mixer = gated_delta_net(self.config, weights, "l0", normed, boundaries)
        np.testing.assert_allclose(
            boundaries.values["l0.post_conv"].ravel(), t["expected_l0_post_conv"], atol=1e-5
        )
        np.testing.assert_allclose(
            boundaries.values["l0.gate"].ravel(), t["expected_l0_gate"], atol=1e-5
        )
        np.testing.assert_allclose(
            boundaries.values["l0.core_out"].ravel(), t["expected_l0_core_out"], atol=1e-5
        )
        np.testing.assert_allclose(
            boundaries.values["l0.state"].ravel(), t["expected_l0_state"], atol=1e-5
        )
        np.testing.assert_allclose(mixer.ravel(), t["expected_l0_mixer"], atol=1e-5)

    def test_attention_boundaries(self):
        t = self.tensors
        weights = {
            "l1.q_proj.weight": t["l1_q_proj"].reshape(-1, self.config.hidden_size),
            "l1.k_proj.weight": t["l1_k_proj"].reshape(-1, self.config.hidden_size),
            "l1.v_proj.weight": t["l1_v_proj"].reshape(-1, self.config.hidden_size),
            "l1.q_norm.weight": t["l1_q_norm"],
            "l1.k_norm.weight": t["l1_k_norm"],
            "l1.o_proj.weight": t["l1_o_proj"].reshape(self.config.hidden_size, -1),
        }
        normed = t["expected_l1_normed"].reshape(self.seq, self.config.hidden_size)
        cos, sin = rope_tables(self.config, np.arange(self.seq))
        boundaries = Boundaries(capture=True)
        mixer = full_attention(self.config, weights, "l1", normed, cos, sin, boundaries)
        np.testing.assert_allclose(
            boundaries.values["l1.query"].ravel(), t["expected_l1_query"], atol=1e-5
        )
        np.testing.assert_allclose(
            boundaries.values["l1.key"].ravel(), t["expected_l1_key"], atol=1e-5
        )
        np.testing.assert_allclose(
            boundaries.values["l1.attention"].ravel(), t["expected_l1_attention"], atol=1e-5
        )
        np.testing.assert_allclose(mixer.ravel(), t["expected_l1_mixer"], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
