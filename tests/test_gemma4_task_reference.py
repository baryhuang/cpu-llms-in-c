import json
import unittest
from pathlib import Path

import numpy as np

from compiler.gemma4_task_reference import apply_rope


class Gemma4TaskReferenceTest(unittest.TestCase):
    def test_position_zero_rope_is_identity(self):
        states = np.arange(16, dtype=np.float32).reshape(1, 2, 1, 8)
        result = apply_rope(states, np.array([0, 1], dtype=np.float32), 10000.0, 1.0)
        np.testing.assert_array_equal(result[:, 0], states[:, 0])

    def test_proportional_rope_leaves_nope_dimensions_unchanged(self):
        states = np.arange(8, dtype=np.float32).reshape(1, 1, 1, 8)
        result = apply_rope(states, np.array([1], dtype=np.float32), 1000000.0, 0.25)
        np.testing.assert_array_equal(result[..., 1:4], states[..., 1:4])
        np.testing.assert_array_equal(result[..., 5:8], states[..., 5:8])
        self.assertNotEqual(float(result[..., 0].item()), float(states[..., 0].item()))

    def test_hazard_profile_is_balanced_and_bounded(self):
        repository = Path(__file__).resolve().parents[1]
        profile = json.loads(
            (repository / "models/gemma-4-e2b/profile.json").read_text()
        )
        labels = [item["name"] for item in profile["labels"]]
        self.assertEqual(labels, ["safe", "danger"])
        expected = [case["expected"] for case in profile["cases"]]
        self.assertEqual(expected.count("safe"), 6)
        self.assertEqual(expected.count("danger"), 6)
        self.assertEqual(len({case["id"] for case in profile["cases"]}), 12)


if __name__ == "__main__":
    unittest.main()
