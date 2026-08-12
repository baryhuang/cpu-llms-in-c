import html
import json
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]


class Qwen35GenerationReviewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.review = (
            REPOSITORY
            / "models/qwen3.5-0.8b/targets/a113x/GENERATION_REVIEW.html"
        ).read_text()
        cls.results = json.loads(
            (
                REPOSITORY
                / "models/qwen3.5-0.8b/targets/a113x/results.json"
            ).read_text()
        )["generation_benchmark"]

    def test_exact_human_readable_input_and_output_are_visible(self):
        self.assertIn(html.escape(self.results["prompt"]), self.review)
        self.assertIn(html.escape(self.results["generated_text"]), self.review)
        self.assertIn("Exact human-readable model output", self.review)

    def test_recorded_measurements_are_visible(self):
        fields = (
            "prefill_seconds",
            "prefill_tokens_per_second",
            "first_token_head_seconds",
            "time_to_first_token_seconds",
            "steady_decode_seconds",
            "steady_decode_tokens_per_second",
            "generation_after_prefill_seconds",
            "generation_after_prefill_tokens_per_second",
        )
        for field in fields:
            self.assertIn(f'{self.results[field]:.6f}', self.review, field)
        self.assertIn(f'{self.results["maximum_resident_kib"]:,}', self.review)

    def test_every_token_duration_is_visible(self):
        for duration in self.results["per_generated_token_seconds"]:
            self.assertIn(f"{duration:.6f}", self.review)

    def test_tokens_are_a_collapsed_audit_detail(self):
        self.assertIn("<details>", self.review)
        self.assertIn("Audit data: token IDs and binary hashes", self.review)
        self.assertIn("Generated token IDs", self.review)


if __name__ == "__main__":
    unittest.main()
