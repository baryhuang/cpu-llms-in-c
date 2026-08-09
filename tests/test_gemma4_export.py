import unittest

import numpy as np

from tools.export_gemma4_layer_fixture import (
    apply_rope,
    bf16_scalar,
    gelu_pytorch_tanh,
    linear,
    rms_norm,
)


class Gemma4ExportMathTest(unittest.TestCase):
    def test_bfloat16_round_to_nearest_even(self):
        self.assertEqual(bf16_scalar(1.0), 1.0)
        self.assertEqual(bf16_scalar(39.191835884530846), 39.25)

    def test_linear_and_rms_norm(self):
        inputs = np.array([[1.0, -2.0]], dtype=np.float32)
        weights = np.array([[0.5, 0.25], [-1.0, 2.0]], dtype=np.float32)
        np.testing.assert_array_equal(linear(inputs, weights), [[0.0, -5.0]])
        result = rms_norm(inputs, np.array([1.0, 0.5], dtype=np.float32), 1.0e-6)
        self.assertEqual(result.dtype, np.float32)
        self.assertAlmostEqual(float(result[0, 0]), 0.6324554, places=6)
        self.assertAlmostEqual(float(result[0, 1]), -0.6324554, places=6)

    def test_rope_position_zero_is_identity(self):
        states = np.arange(16, dtype=np.float32).reshape(1, 2, 8)
        np.testing.assert_array_equal(apply_rope(states, 10000.0), states)

    def test_gelu_zero(self):
        self.assertEqual(float(gelu_pytorch_tanh(np.array([0.0], dtype=np.float32))[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
