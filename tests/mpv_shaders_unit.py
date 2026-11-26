#!/usr/bin/env python3
"""Weight-free checks for the mpv exporter's catalog and integrity contract."""

import hashlib
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import export_mpv_shaders as export


class ExportTests(unittest.TestCase):
    def test_catalog(self):
        models = export.selected_models(export.INDEX)
        paths = {export.model_path(m) for m in models}
        expected = {Path(f"{family}/{quality}.glsl")
                    for family in export.FAMILIES.values() for quality in export.QUALITIES}
        expected.update(Path(f"sharp/options/{quality}_option{i:02d}.glsl")
                        for quality in export.QUALITIES for i in range(1, 15))
        self.assertEqual(len(models), 72)
        self.assertEqual(paths, expected)

    def test_parameter_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic.weights"
            for fp16 in (False, True):
                with self.subTest(fp16=fp16):
                    fmt = "<e" if fp16 else "<f"
                    data = struct.pack(fmt, -0.125)
                    metadata = dict(size=len(data), sha256=hashlib.sha256(data).hexdigest())
                    path.write_bytes(data)
                    self.assertEqual(export.parameter(path, metadata, 1, fp16), (-0.125,))
                    path.write_bytes(data + b"\x00")
                    with self.assertRaisesRegex(ValueError, "size mismatch"):
                        export.parameter(path, metadata, 1, fp16)
                    path.write_bytes(struct.pack(fmt, 0.125))
                    with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                        export.parameter(path, metadata, 1, fp16)
                    data = struct.pack(fmt, float("inf"))
                    path.write_bytes(data)
                    metadata["sha256"] = hashlib.sha256(data).hexdigest()
                    with self.assertRaisesRegex(ValueError, "non-finite parameter"):
                        export.parameter(path, metadata, 1, fp16)

    def test_missing_weights_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "NGU"
            proc = subprocess.run(
                [sys.executable, str(Path(export.__file__)), "--weights-root", str(root),
                 "--output", str(target), "--model", "aa_2x_low"],
                capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("stage00.weights.f32le", proc.stderr)
            self.assertFalse(target.exists())

    def test_fused_schedules_and_portable_shared_memory(self):
        shapes = [
            ("sharp_2x_low", False, [(1, 8, 3), (8, 8, 3), (8, 8, 3), (8, 1, 4)], 1),
            ("aa_2x_low", True, [(1, 8, 3), (8, 8, 3), (8, 8, 3)] + [(8, 1, 3)] * 3, 1),
            ("sharp_2x_medium", False, [(1, 12, 3), (12, 8, 3), (8, 8, 3), (8, 8, 3), (8, 8, 3), (8, 1, 4)], 2),
            ("sharp_2x_high", False, [(1, 12, 3)] + [(12, 12, 3)] * 5 + [(12, 1, 4)], 4),
            ("sharp_2x_very_high", False, [(1, 16, 3)] + [(16, 16, 3)] * 7 + [(16, 1, 4)], 5),
            ("aa_2x_very_high", True, [(1, 16, 3)] + [(16, 16, 3)] * 7 + [(16, 1, 3)] * 3, 5),
        ]
        for model_id, aa, spec, expected_passes in shapes:
            layers = [(ci, co, (0.015625,) * (ci * co * k * k), (0.00390625,) * co)
                      for ci, co, k in spec]
            for native_fp16 in (False, True):
                glsl = export.render(model_id, layers, aa, native_fp16)
                self.assertEqual(glsl.count("//!HOOK LUMA"), expected_passes)
                self.assertEqual(glsl.count("//!COMPUTE "), expected_passes)
                self.assertEqual("#define NGU_NATIVE_FP16 1" in glsl, native_fp16)
                self.assertEqual("//!OFFSET -0.5 -0.5" in glsl, aa)


if __name__ == "__main__":
    unittest.main()
