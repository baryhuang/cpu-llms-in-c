import html
import hashlib
import json
import re
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
BENCHMARK = REPOSITORY / "models/qwen3.5-0.8b/benchmarks/arc-easy-5"


class ArcEasyFiveReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads((BENCHMARK / "profile.json").read_text())
        cls.local_results = json.loads(
            (BENCHMARK / "results-macos-m3-pro.json").read_text()
        )
        cls.results = json.loads((BENCHMARK / "results-a113x.json").read_text())
        cls.review = (BENCHMARK / "REVIEW.html").read_text()

    def test_profile_is_five_source_order_cases(self):
        self.assertEqual([case["ordinal"] for case in self.profile["cases"]], [1, 2, 3, 4, 5])
        self.assertEqual(len(self.profile["cases"]), 5)
        self.assertIn("source order", self.profile["selection_rule"])
        self.assertIn("Do not inspect model outputs", self.profile["selection_rule"])

    def test_result_pins_exact_profile(self):
        digest = hashlib.sha256((BENCHMARK / "profile.json").read_bytes()).hexdigest()
        self.assertEqual(digest, self.results["profile_sha256"])

    def test_predeclared_parser_reproduces_scores(self):
        expected = {case["id"]: case["answerKey"] for case in self.profile["cases"]}
        for case in self.results["cases"]:
            matches = re.findall(r"(?<![A-Za-z])Answer: ([A-D])(?![A-Za-z])", case["generated_text"])
            parsed = matches[-1] if matches else None
            self.assertEqual(parsed, case["parsed"], case["id"])
            self.assertEqual(parsed == expected[case["id"]], case["correct"], case["id"])
        self.assertEqual(sum(case["correct"] for case in self.results["cases"]), 4)

    def test_review_contains_exact_inputs_outputs_and_duration(self):
        results = {case["id"]: case for case in self.results["cases"]}
        for case in self.profile["cases"]:
            measured = results[case["id"]]
            self.assertIn(html.escape(case["prompt"]), self.review, case["id"])
            self.assertIn(html.escape(measured["generated_text"]), self.review, case["id"])
            self.assertIn(f'{measured["prefill_seconds"]:.6f} s', self.review, case["id"])
            self.assertIn(f'{measured["time_to_first_token_seconds"]:.6f} s', self.review, case["id"])
            self.assertIn(f'{measured["wall_duration_seconds"]:.2f} s', self.review, case["id"])

    def test_review_does_not_claim_official_score_or_a113x_timing(self):
        self.assertIn("not the official ARC-Easy", self.review)
        self.assertIn("Primary timing and resource measurements on this page are from the A113X target", self.review)
        self.assertFalse(self.results["scoring"]["official_arc_score"])

    def test_cpu_memory_and_swap_are_visible(self):
        for case in self.results["cases"]:
            self.assertIn(f'{case["cpu_percent"]}%', self.review, case["id"])
            self.assertIn(f'{case["maximum_resident_kib"]:,} KiB', self.review, case["id"])
        aggregate = self.results["aggregate_timing_and_resources"]
        self.assertIn(f'{aggregate["weighted_cpu_percent"]:.2f}%', self.review)
        self.assertIn(f'{aggregate["maximum_resident_kib"]:,} KiB', self.review)
        self.assertIn("device swap used was 0 before and after", self.review)

    def test_target_and_local_outputs_match(self):
        local = {case["id"]: case for case in self.local_results["cases"]}
        for case in self.results["cases"]:
            self.assertEqual(case["token_ids"], local[case["id"]]["token_ids"])


if __name__ == "__main__":
    unittest.main()
