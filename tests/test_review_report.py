import json
import unittest
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

from compiler.summarize_gemma4_task_results import (
    benchmark_case,
    wall_duration_seconds,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def six_places(value):
    return str(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


class ReviewReportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = (REPOSITORY / "REVIEW.html").read_text()
        cls.results = json.loads(
            (
                REPOSITORY
                / "models/whisper-large-v3-turbo/targets/a113x/results.json"
            ).read_text()
        )

    def test_review_is_repository_level(self):
        self.assertIn("model × target engineering review", self.html)
        self.assertIn("Implemented model × target matrix", self.html)
        self.assertIn("Whisper large-v3-turbo on A113X", self.html)
        self.assertNotIn("Gemma 4 E2B bounded-runtime test review", self.html)

    def test_all_implemented_pairs_are_visible(self):
        expected_rows = (
            ("Qwen3.8-27B", "Apple M3 Pro"),
            ("MiniMax-H3", "Apple M3 Pro"),
            ("Qwen3.5-0.8B", "Amlogic A113X"),
            ("Whisper large-v3", "Jetson Orin Nano"),
            ("Whisper large-v3-turbo", "Jetson Orin Nano"),
            ("Whisper large-v3-turbo", "Amlogic A113X"),
            ("Whisper small.en", "Amlogic A113X"),
            ("Gemma 4 E2B", "generic two-vCPU x86-64"),
        )
        self.assertEqual(self.html.count("<tr data-target="), len(expected_rows))
        for model, target in expected_rows:
            self.assertIn(model, self.html)
            self.assertIn(target, self.html)

    def test_a113x_increment_values_match_results_json(self):
        increment = self.results["optimization_increment"]
        for precision in ("compact_jfk_q4", "compact_jfk_q5"):
            result = increment[precision]
            values = (
                result["baseline"]["total_seconds"],
                result["optimized"]["total_seconds"],
                result["end_to_end_speedup"],
            )
            for value in values:
                self.assertIn(six_places(value), self.html)

        micro = increment["stem_microbenchmark_1100_frames"]
        for key in ("baseline_mean_seconds", "optimized_mean_seconds", "speedup"):
            self.assertIn(six_places(micro[key]), self.html)
        self.assertIn(str(micro["checksum_all_runs"]), self.html)

    def test_rejected_full_path_result_is_visible(self):
        rejected = self.results["optimization_increment"]["rejected_four_row_gemm"]
        self.assertIn(six_places(rejected["compact_q5_four_row_seconds"]), self.html)
        self.assertIn("REJECT · four-row × four-output GEMM", self.html)

    def test_a113x_is_added_to_fixed_window_cross_device_table(self):
        expected = (
            "Cross-device comparison — A113X added",
            "1,272,397 ms",
            "1,300.079433 s",
            "47.674× real time",
            "229.763245 s encoder",
            "236.933801 s end to end",
        )
        for value in expected:
            self.assertIn(value, self.html)
        self.assertIn("before the new stem increment", self.html)
        self.assertIn("always pad the encoder to 30 seconds", self.html)

    def test_primary_a113x_records_are_linked(self):
        required_links = (
            "models/whisper-large-v3-turbo/targets/a113x/results.json",
            "models/whisper-large-v3-turbo/targets/a113x/README.md",
            "models/whisper-large-v3-turbo/targets/a113x/benchmarks/device/compact-jfk-neon-stem-q4.log",
            "models/whisper-large-v3-turbo/targets/a113x/benchmarks/device/compact-jfk-neon-stem-q4.time",
            "models/whisper-large-v3-turbo/targets/a113x/benchmarks/device/std30-q4.log",
            "models/whisper-large-v3-turbo/targets/jetson-orin/results.json",
            "models/whisper-large-v3-turbo/targets/a113x/whisper_turbo_frontend.c",
        )
        for link in required_links:
            self.assertIn(f'href="{link}"', self.html)
            self.assertTrue((REPOSITORY / link).is_file(), link)

    def test_legacy_benchmark_duration_helpers(self):
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


if __name__ == "__main__":
    unittest.main()
