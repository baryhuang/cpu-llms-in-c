import json
import unittest
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from compiler.summarize_gemma4_task_results import (
    benchmark_case,
    wall_duration_seconds,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def four_places(value):
    return str(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def six_places(value):
    return str(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


class ReviewReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (REPOSITORY / "REVIEW.html").read_text()
        cls.profile = json.loads(
            (REPOSITORY / "models/gemma-4-e2b/profile.json").read_text()
        )
        cls.results = json.loads(
            (REPOSITORY / "models/gemma-4-e2b/results.json").read_text()
        )

    def test_all_inputs_are_visible(self):
        self.assertEqual(self.html.count("data-expected="), len(self.profile["cases"]))
        self.assertIn(self.profile["system_prompt"], self.html)
        for case in self.profile["cases"]:
            self.assertIn(case["id"], self.html)
            self.assertIn(case["observation"], self.html)

    def test_all_recorded_case_values_are_visible(self):
        comparisons = {
            case["id"]: case for case in self.results["verification"]["cases"]
        }
        timings = {
            case["id"]: case for case in self.results["benchmark"]["warm"]["cases"]
        }
        for profile_case in self.profile["cases"]:
            result = comparisons[profile_case["id"]]
            timing = timings[profile_case["id"]]
            verification_values = (
                result["reference_logits"]["safe"],
                result["reference_logits"]["danger"],
                result["q4_logits"]["safe"],
                result["q4_logits"]["danger"],
            )
            benchmark_values = (
                timing["classification_duration_seconds"],
                timing["classification_tokens_per_second"],
                timing["extra_label_decode_duration_seconds"],
                timing["extra_label_decode_tokens_per_second"],
            )
            for value in verification_values:
                self.assertIn(four_places(value), self.html, profile_case["id"])
            for value in benchmark_values:
                self.assertIn(six_places(value), self.html, profile_case["id"])

    def test_verification_and_benchmark_are_separate(self):
        for case in self.results["verification"]["cases"]:
            self.assertFalse(any("duration" in key for key in case))
            self.assertNotIn("prompt_tokens", case)
        for case in self.results["benchmark"]["warm"]["cases"]:
            self.assertTrue(any("duration" in key for key in case))
            self.assertNotIn("expected", case)
            self.assertNotIn("q4_logits", case)

    def test_benchmark_duration_helpers(self):
        self.assertEqual(wall_duration_seconds("14:03.24"), 843.24)
        case = benchmark_case(
            {
                "tokens": 43,
                "prompt_seconds": 62.786325,
                "prefill_tokens_per_second": 0.684863,
                "decode_seconds": 1.436223,
                "decode_tokens_per_second": 0.696271,
            },
            "safe_power_isolated",
        )
        self.assertEqual(case["classification_duration_seconds"], 62.786325)
        self.assertEqual(case["extra_label_decode_duration_seconds"], 1.436223)

    def test_runtime_boundary_is_explicit(self):
        self.assertIn("IMAGE [CASE_INDEX|all]", self.html)
        self.assertIn("does not accept arbitrary text", self.html)
        self.assertEqual(
            self.results["output_contract"]["encoding"], {"safe": 0, "danger": 1}
        )
        self.assertEqual(
            self.results["output_contract"]["semantic_output"],
            "one binary decision bit",
        )
        self.assertIn("one bit", self.html)
        runtime_source = (REPOSITORY / "models/gemma-4-e2b/targets/generic/gemma4_task.c").read_text()
        self.assertIn('strcmp(argv[2], "all")', runtime_source)
        self.assertIn("strtoul(argv[2]", runtime_source)


if __name__ == "__main__":
    unittest.main()
