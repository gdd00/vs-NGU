#!/usr/bin/env python3
"""Verify real FP16 archive payloads, source integrity, and converter safety."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from convert_ngu_fp16 import convert_archive, encode_parameter


class FP16ArchiveTests(unittest.TestCase):
    def test_all_files_are_real_half_size_binary16(self):
        metadata = json.loads((ARGS.archive / "manifest.json").read_text())
        compiled_metadata = json.loads((ROOT / "models/madvr09217-fp16-index.json").read_text())
        self.assertEqual(metadata, compiled_metadata)
        self.assertEqual(metadata["dtype"], "ieee754-binary16-le")
        self.assertEqual(len(metadata["models"]), 212)
        self.assertEqual(list(ARGS.archive.rglob("*.f32le")), [])
        half_files = set()
        total = original_total = 0
        for model in metadata["models"]:
            for stage in model["stages"]:
                for kind in ("weights", "bias"):
                    spec = stage[kind]
                    path = ARGS.archive / spec["path"]
                    source = ARGS.source / spec["path"].replace(".f16le", ".f32le")
                    original, actual = source.read_bytes(), path.read_bytes()
                    self.assertEqual(hashlib.sha256(original).hexdigest(), spec["source_sha256"])
                    self.assertEqual(hashlib.sha256(actual).hexdigest(), spec["sha256"])
                    self.assertEqual(len(actual), spec["size"])
                    self.assertEqual(len(actual) * 2, len(original))
                    self.assertEqual(actual, encode_parameter(original, kind))
                    self.assertTrue(all(math.isfinite(v[0]) for v in struct.iter_unpack("<e", actual)))
                    total += len(actual)
                    original_total += len(original)
                    half_files.add(path)
        self.assertEqual(half_files, set(ARGS.archive.rglob("*.f16le")))
        self.assertEqual(len(half_files), 3284)
        self.assertEqual((original_total, total), (7581008, 3790504))
        self.assertEqual(total, metadata["summary"]["parameter_bytes"])

    def test_rounding_bias_scaling_and_invalid_values(self):
        values = [0.0, -0.0, 1.0 + 2**-11, 1.0 + 3 * 2**-11, 2**-24, 2**-25]
        actual = encode_parameter(struct.pack("<6f", *values), "weights")
        self.assertEqual(struct.unpack("<6H", actual), (0, 0x8000, 0x3c00, 0x3c02, 1, 0))
        # The raw bias exceeds the half finite limit, but the scaled bias does not.
        bias = encode_parameter(struct.pack("<f", 65520), "bias")
        self.assertTrue(math.isfinite(struct.unpack("<e", bias)[0]))
        for value in (65520.0, math.inf, -math.inf, math.nan):
            with self.assertRaises(ValueError):
                encode_parameter(struct.pack("<f", value), "weights")
        for data in (b"", b"\0", b"\0\0"):
            with self.assertRaises(ValueError):
                encode_parameter(data, "weights")

    def test_existing_or_source_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            marker = output / "marker"
            marker.write_text("preserve")
            with self.assertRaises(FileExistsError):
                convert_archive(ARGS.source, output)
            self.assertEqual(marker.read_text(), "preserve")
        with self.assertRaises(ValueError):
            convert_archive(ARGS.source, ARGS.source)
        with self.assertRaises(ValueError):
            convert_archive(ARGS.source, ARGS.source / "nested-output")

    def test_bad_source_is_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "source", Path(temp) / "output"
            path = source / "aa_2x_low/stage00.weights.f32le"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "size/SHA-256"):
                convert_archive(source, output)
            self.assertFalse(output.exists())

    def test_reconversion_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temp:
            output, index = Path(temp) / "converted", Path(temp) / "metadata.json"
            convert_archive(ARGS.source, output, metadata_output=index)
            expected = (ARGS.archive / "manifest.json").read_bytes()
            self.assertEqual((output / "manifest.json").read_bytes(), expected)
            self.assertEqual(index.read_bytes(), expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    ARGS = parser.parse_args()
    unittest.main(argv=[__file__, "-v"])
