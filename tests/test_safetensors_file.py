import json
import struct
import tempfile
import unittest
from pathlib import Path

from compiler.safetensors_file import SafetensorsFile


def write_file(path, header, payload):
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class SafetensorsFileTest(unittest.TestCase):
    def test_selective_f32_and_bf16_reads(self):
        f32_payload = struct.pack("<2f", 1.25, -2.5)
        bf16_payload = struct.pack("<2H", 0x3F80, 0xC020)
        header = {
            "__metadata__": {"format": "test"},
            "root.first.weight": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, len(f32_payload)],
            },
            "root.second.weight": {
                "dtype": "BF16",
                "shape": [1, 2],
                "data_offsets": [len(f32_payload), len(f32_payload) + len(bf16_payload)],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.safetensors"
            write_file(path, header, f32_payload + bf16_payload)
            with SafetensorsFile(path) as source:
                self.assertEqual(source.metadata, {"format": "test"})
                self.assertEqual(source.read_float32("root.first.weight"), [1.25, -2.5])
                self.assertEqual(source.read_float32("root.second.weight"), [1.0, -2.5])
                self.assertEqual(source.read_float32_range("root.second.weight", 1, 1), [-2.5])
                self.assertEqual(source.find_unique_suffix("second.weight"), "root.second.weight")
                with self.assertRaisesRegex(ValueError, "outside"):
                    source.read_element_bytes("root.first.weight", 1, 2)

    def test_rejects_payload_size_mismatch(self):
        header = {"bad": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.safetensors"
            write_file(path, header, b"\0" * 4)
            with self.assertRaisesRegex(ValueError, "payload size mismatch"):
                SafetensorsFile(path)

    def test_rejects_overlapping_payloads(self):
        header = {
            "one": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
            "two": {"dtype": "U8", "shape": [2], "data_offsets": [1, 3]},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlap.safetensors"
            write_file(path, header, b"\0" * 3)
            with self.assertRaisesRegex(ValueError, "overlapping tensor payloads"):
                SafetensorsFile(path)


if __name__ == "__main__":
    unittest.main()
