import sys
import unittest
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "compiler"))

from q4_image import (  # noqa: E402
    quantize_q4_grouped,
    quantize_q4_grouped_output_blocked,
    quantize_q8_grouped,
    quantize_q8_grouped_output_blocked,
)


class OutputBlockedQuantization(unittest.TestCase):
    def setUp(self):
        values = np.arange(8 * 256, dtype=np.float32).reshape(8, 256)
        self.weights = np.sin(values * np.float32(0.017))

    def check_layout(self, row_major, blocked, record_bytes):
        rows = 8
        groups = 2
        expected = np.frombuffer(row_major, dtype=np.uint8).reshape(
            rows, groups, record_bytes
        )
        expected = expected.reshape(2, 4, groups, record_bytes).transpose(0, 2, 1, 3)
        actual = np.frombuffer(blocked, dtype=np.uint8).reshape(2, groups, 4, record_bytes)
        np.testing.assert_array_equal(actual, expected)

    def test_q4_records_follow_four_output_kernel_order(self):
        self.check_layout(
            quantize_q4_grouped(self.weights),
            quantize_q4_grouped_output_blocked(self.weights, 4),
            66,
        )

    def test_q8_records_follow_four_output_kernel_order(self):
        self.check_layout(
            quantize_q8_grouped(self.weights),
            quantize_q8_grouped_output_blocked(self.weights, 4),
            130,
        )

    def test_output_block_requires_divisible_rows(self):
        with self.assertRaises(ValueError):
            quantize_q4_grouped_output_blocked(self.weights[:6], 4)


if __name__ == "__main__":
    unittest.main()
