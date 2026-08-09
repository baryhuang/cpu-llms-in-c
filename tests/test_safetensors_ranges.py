import unittest

from tools.fetch_safetensors_ranges import coalesce, split_ranges, tensor_slice_range


class SafetensorsRangesTest(unittest.TestCase):
    def test_coalesces_nearby_ranges(self):
        self.assertEqual(coalesce([(100, 120), (0, 10), (15, 30)], 5), [(0, 30), (100, 120)])

    def test_splits_large_ranges(self):
        self.assertEqual(split_ranges([(10, 21)], 4), [(10, 14), (14, 18), (18, 21)])

    def test_tensor_element_slice(self):
        header = {"weight": {"dtype": "BF16", "shape": [4, 8], "data_offsets": [10, 74]}}
        self.assertEqual(tensor_slice_range(header, 1000, "weight", 8, 8), (1026, 1042))

    def test_rejects_out_of_range_slice(self):
        header = {"weight": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
        with self.assertRaisesRegex(ValueError, "outside"):
            tensor_slice_range(header, 100, "weight", 3, 2)


if __name__ == "__main__":
    unittest.main()
