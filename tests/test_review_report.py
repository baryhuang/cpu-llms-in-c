import json
import unittest
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]


def four_places(value):
    return str(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


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
        comparisons = {case["id"]: case for case in self.results["comparisons"]}
        for profile_case in self.profile["cases"]:
            result = comparisons[profile_case["id"]]
            values = (
                result["reference_logits"]["safe"],
                result["reference_logits"]["danger"],
                result["q4_logits"]["safe"],
                result["q4_logits"]["danger"],
                result["prompt_seconds"],
                result["prefill_tokens_per_second"],
                result["decode_seconds"],
                result["decode_tokens_per_second"],
            )
            for value in values:
                self.assertIn(four_places(value), self.html, profile_case["id"])

    def test_runtime_boundary_is_explicit(self):
        self.assertIn("IMAGE [CASE_INDEX|all]", self.html)
        self.assertIn("does not accept arbitrary text", self.html)
        runtime_source = (REPOSITORY / "src/gemma4_task.c").read_text()
        self.assertIn('strcmp(argv[2], "all")', runtime_source)
        self.assertIn("strtoul(argv[2]", runtime_source)


if __name__ == "__main__":
    unittest.main()
