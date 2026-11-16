#!/usr/bin/env python3
"""Check shared-root script defaults and non-destructive archive creation.

Supply --madvr for a full extraction/conversion round trip from an authorized
binary. All test output and archive-misidentification fixtures live in temporary directories.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import convert_ngu_fp16 as converter
import extract_ngu_models as extractor


def run_script(name, *args, project=ROOT):
    return subprocess.run([sys.executable, str(project / "tools" / name), *map(str, args)],
                          text=True, capture_output=True, check=False)


class WeightsLayoutTests(unittest.TestCase):
    def test_project_defaults_and_explicit_archive_overrides(self):
        manifest = {"models": [], "auxiliary_models": [], "summary": {"stage_count": 0}}
        with patch.object(extractor, "extract", return_value=manifest) as extract, redirect_stdout(io.StringIO()):
            self.assertEqual(extractor.main(["source.ax"]), 0)
            extract.assert_called_once_with(Path("source.ax"), ROOT / "weights/weights", None, None)
            extract.reset_mock()
            self.assertEqual(extractor.main(["source.ax", "--weights-root", "common root"]), 0)
            extract.assert_called_once_with(Path("source.ax"), Path("common root/weights"), None, None)
            extract.reset_mock()
            self.assertEqual(extractor.main(["source.ax", "--output", "legacy archive"]), 0)
            extract.assert_called_once_with(Path("source.ax"), Path("legacy archive"), None, None)

        with (patch.object(converter, "convert_archive", return_value={"summary": {}}) as convert,
              redirect_stdout(io.StringIO())):
            self.assertEqual(converter.main([]), 0)
            convert.assert_called_once_with(ROOT / "weights/weights", ROOT / "weights/weights-fp16",
                                            ROOT / "models/madvr09217-index.json", None)
            convert.reset_mock()
            self.assertEqual(converter.main(["--source", "old FP32", "--output", "new FP16"]), 0)
            convert.assert_called_once_with(Path("old FP32"), Path("new FP16"),
                                            ROOT / "models/madvr09217-index.json", None)

    def test_conversion_in_temporary_common_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "common weights"
            root.mkdir()
            marker = root / "preserve.txt"
            marker.write_text("user data")
            # An explicitly named FP32 child may be a symlink; the common
            # root itself must not be a symlink to a parameter archive.
            (root / "weights").symlink_to(ARGS.source.resolve(), target_is_directory=True)
            source_manifest = (ARGS.source / "manifest.json").read_bytes()
            result = run_script("convert_ngu_fp16.py", "--weights-root", root)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = root / "weights-fp16"
            expected = (ROOT / "models/madvr09217-fp16-index.json").read_bytes()
            self.assertEqual((output / "manifest.json").read_bytes(), expected)
            self.assertEqual(len(list(output.rglob("*.f16le"))), 3284)
            self.assertEqual(list(output.rglob("*.f32le")), [])
            self.assertEqual((ARGS.source / "manifest.json").read_bytes(), source_manifest)
            self.assertEqual(marker.read_text(), "user data")
            # Existing converted data must survive a repeated invocation.
            repeated = run_script("convert_ngu_fp16.py", "--weights-root", root)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("refusing to overwrite", repeated.stderr)
            self.assertEqual((output / "manifest.json").read_bytes(), expected)
            self.assertEqual(marker.read_text(), "user data")

    def test_missing_source_and_bad_binary_do_not_create_archives(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "missing root"
            result = run_script("convert_ngu_fp16.py", "--weights-root", root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(f"missing FP32 archive: {root / 'weights'}", result.stderr)
            self.assertFalse(root.exists())
            source = Path(temp) / "not-madvr.ax"
            source.write_bytes(b"invalid source")
            result = run_script("extract_ngu_models.py", source, "--weights-root", root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("unsupported source", result.stderr)
            self.assertFalse(root.exists())

    def test_extraction_refuses_nonempty_output(self):
        # Test the output guard without redistributing a PE or private weights.
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "weights"
            output.mkdir()
            marker = output / "user-data"
            marker.write_text("preserve")
            with self.assertRaisesRegex(extractor.FormatError, "new or empty directory"):
                extractor.extract(Path("source.ax"), output)
            self.assertEqual(marker.read_text(), "preserve")

    def test_archive_cannot_be_used_as_root_or_output_ancestor(self):
        for extension in ("f32le", "f16le"):
            for manifest in (False, True):
                with self.subTest(extension=extension, manifest=manifest), tempfile.TemporaryDirectory() as temp:
                    base = Path(temp)
                    archive = base / "existing archive"
                    parameter = archive / "aa_2x_low" / ("stage00.weights." + extension)
                    parameter.parent.mkdir(parents=True)
                    parameter.write_bytes(b"preserve private data")
                    if manifest:
                        (archive / "manifest.json").write_text("{}")
                    alias = base / "archive symlink"
                    alias.symlink_to(archive, target_is_directory=True)
                    before = sorted(str(p.relative_to(archive)) for p in archive.rglob("*"))
                    for root in (archive, alias):
                        for script, args in (("extract_ngu_models.py", ["missing-source.ax"]),
                                             ("convert_ngu_fp16.py", [])):
                            result = run_script(script, *args, "--weights-root", root)
                            self.assertNotEqual(result.returncode, 0)
                            self.assertIn("parameter archive", result.stderr)
                        # Explicit paths and the Python API cannot bypass the guard.
                        result = run_script("extract_ngu_models.py", "missing-source.ax",
                                            "--output", root / "nested")
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("inside parameter archive", result.stderr)
                        with self.assertRaisesRegex(ValueError, "inside parameter archive"):
                            extractor.extract(Path("missing-source.ax"), root / "nested")
                        with self.assertRaisesRegex(ValueError, "inside parameter archive"):
                            converter.convert_archive(ARGS.source, root / "nested")
                        with self.assertRaisesRegex(ValueError, "inside parameter archive"):
                            converter.convert_archive(ARGS.source, base / "converted",
                                                      metadata_output=root / "extra-metadata.json")
                    self.assertEqual(sorted(str(p.relative_to(archive)) for p in archive.rglob("*")), before)
                    self.assertEqual(parameter.read_bytes(), b"preserve private data")
                    self.assertTrue(alias.is_symlink())
                    self.assertFalse((base / "converted").exists())

    def test_symlink_destinations_are_rejected_before_writing(self):
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                target = base / "external target"
                if existing:
                    target.mkdir()
                alias = base / "output symlink"
                alias.symlink_to(target, target_is_directory=True)
                converted = base / "converted"
                for destination in (alias, alias / "nested"):
                    with self.assertRaisesRegex(ValueError, "symlink"):
                        extractor.extract(Path("missing-source.ax"), destination)
                    with self.assertRaisesRegex(ValueError, "symlink"):
                        converter.convert_archive(ARGS.source, destination)
                    with self.assertRaisesRegex(ValueError, "symlink"):
                        converter.convert_archive(ARGS.source, converted, metadata_output=destination)
                    for script, args in (("extract_ngu_models.py", ["missing-source.ax"]),
                                         ("convert_ngu_fp16.py", ["--source", ARGS.source])):
                        result = run_script(script, *args, "--output", destination)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("symlink", result.stderr)
                        result = run_script(script, *args, "--weights-root", alias)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("symlink", result.stderr)
                    result = run_script("convert_ngu_fp16.py", "--source", ARGS.source,
                                        "--output", converted, "--metadata-output", destination)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("symlink", result.stderr)
                self.assertTrue(alias.is_symlink())
                self.assertEqual(target.exists(), existing)
                self.assertEqual(list(target.rglob("*")), [])
                self.assertFalse(converted.exists())

    def test_existing_and_symlinked_metadata_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            output = base / "converted"
            metadata = base / "metadata.json"
            metadata.write_bytes(b"preserve metadata")
            with self.assertRaises(FileExistsError):
                converter.convert_archive(ARGS.source, output, metadata_output=metadata)
            alias = base / "metadata symlink"
            alias.symlink_to(metadata)
            with self.assertRaisesRegex(ValueError, "symlink"):
                converter.convert_archive(ARGS.source, output, metadata_output=alias)
            self.assertEqual(metadata.read_bytes(), b"preserve metadata")
            metadata.unlink()  # A dangling metadata link must fail before creating the archive too.
            with self.assertRaisesRegex(ValueError, "symlink"):
                converter.convert_archive(ARGS.source, output, metadata_output=alias)
            self.assertTrue(alias.is_symlink())
            self.assertFalse(metadata.exists())
            self.assertFalse(output.exists())

    def test_parent_traversal_destinations_are_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            archive = base / "existing archive"
            parameter = archive / "aa_2x_low/stage00.weights.f32le"
            parameter.parent.mkdir(parents=True)
            parameter.write_bytes(b"preserve private data")
            alias = base / "archive symlink"
            alias.symlink_to(archive, target_is_directory=True)
            before = sorted(str(p.relative_to(archive)) for p in archive.rglob("*"))
            converted = base / "converted"
            for destination in (archive / "missing/../../outside",
                                archive / "missing/../nested",
                                base / "missing/../archive symlink/../outside"):
                with self.subTest(destination=destination):
                    with self.assertRaisesRegex(ValueError, "parent traversal"):
                        extractor.extract(Path("missing-source.ax"), destination)
                    with self.assertRaisesRegex(ValueError, "parent traversal"):
                        converter.convert_archive(ARGS.source, destination)
                    with self.assertRaisesRegex(ValueError, "parent traversal"):
                        converter.convert_archive(ARGS.source, converted, metadata_output=destination)
                    for script, args in (("extract_ngu_models.py", ["missing-source.ax"]),
                                         ("convert_ngu_fp16.py", ["--source", ARGS.source])):
                        for option in ("--output", "--weights-root"):
                            result = run_script(script, *args, option, destination)
                            self.assertNotEqual(result.returncode, 0)
                            self.assertIn("parent traversal", result.stderr)
                    result = run_script("convert_ngu_fp16.py", "--source", ARGS.source,
                                        "--output", converted, "--metadata-output", destination)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("parent traversal", result.stderr)
            self.assertEqual(sorted(str(p.relative_to(archive)) for p in archive.rglob("*")), before)
            self.assertEqual(parameter.read_bytes(), b"preserve private data")
            self.assertTrue(alias.is_symlink())
            self.assertFalse((base / "missing").exists())
            self.assertFalse((base / "outside").exists())
            self.assertFalse(converted.exists())

    def test_metadata_and_archive_destinations_must_be_disjoint(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            output = base / "new parent/converted"
            for metadata in (output, output / "manifest.json", output / "aa_2x_low",
                             output / "aa_2x_low/stage00.weights.f16le",
                             output / "extra.json", output.parent):
                with self.subTest(metadata=metadata):
                    with self.assertRaisesRegex(ValueError, "metadata output must be separate"):
                        converter.convert_archive(ARGS.source, output, metadata_output=metadata)
                    result = run_script("convert_ngu_fp16.py", "--source", ARGS.source,
                                        "--output", output, "--metadata-output", metadata)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("metadata output must be separate", result.stderr)
                    self.assertEqual(list(base.iterdir()), [])

    def test_default_scripts_reject_a_previous_archive_or_symlink(self):
        for symlink in (False, True):
            with self.subTest(symlink=symlink), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                project = base / "project"
                project.mkdir()
                archive = base / "old FP32" if symlink else project / "weights"
                archive.mkdir()
                marker = archive / "manifest.json"
                marker.write_text("preserve")
                if symlink:
                    (project / "weights").symlink_to(archive, target_is_directory=True)
                error = io.StringIO()
                with patch.object(extractor, "ROOT", project), redirect_stderr(error):
                    self.assertEqual(extractor.main(["missing-source.ax"]), 1)
                self.assertIn("inside parameter archive", error.getvalue())
                error = io.StringIO()
                with patch.object(converter, "ROOT", project), redirect_stderr(error):
                    self.assertEqual(converter.main([]), 1)
                self.assertIn("inside parameter archive", error.getvalue())
                self.assertEqual(list(archive.iterdir()), [marker])
                self.assertEqual(marker.read_text(), "preserve")

    def test_actual_common_root_layout(self):
        self.assertEqual(ARGS.source.name, "weights")
        half = ARGS.source.parent / "weights-fp16"
        self.assertEqual(len(list(ARGS.source.rglob("*.f32le"))), 3284)
        self.assertEqual(len(list(half.rglob("*.f16le"))), 3284)
        self.assertEqual(list(ARGS.source.rglob("*.f16le")), [])
        self.assertEqual(list(half.rglob("*.f32le")), [])
        self.assertEqual((half / "manifest.json").read_bytes(),
                         (ROOT / "models/madvr09217-fp16-index.json").read_bytes())

    def test_fresh_cmake_defaults(self):
        if ARGS.cmake is None:
            self.skipTest("CMake supplies --cmake to test a fresh, uncached configuration")
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            build = base / "build"
            command = [str(ARGS.cmake), "-S", str(ROOT), "-B", str(build), "-DBUILD_TESTING=ON"]
            if ARGS.generator:
                command += ["-G", ARGS.generator]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            cache = (build / "CMakeCache.txt").read_text()
            default = next(line.split("=", 1)[1] for line in cache.splitlines()
                           if line.startswith("NGU_WEIGHTS_ROOT:PATH="))
            self.assertEqual(Path(default), ROOT / "weights")
            self.assertIn(str(ROOT / "weights"), (build / "CTestTestfile.cmake").read_text())

    def test_authorized_default_extraction_and_conversion_round_trip(self):
        if ARGS.madvr is None:
            self.skipTest("supply --madvr (CMake NGU_MADVR_SOURCE) for authorized PE extraction")
        with tempfile.TemporaryDirectory() as temp:
            # Relocate the tools and public index so the real CLI defaults can
            # generate both archives without touching the user's project data.
            project = Path(temp) / "project"
            (project / "tools").mkdir(parents=True)
            (project / "models").mkdir()
            for name in ("extract_ngu_models.py", "extract_madvr.py",
                         "convert_ngu_fp16.py", "generate_ngu_catalog.py"):
                shutil.copyfile(ROOT / "tools" / name, project / "tools" / name)
            shutil.copyfile(ROOT / "models/madvr09217-index.json",
                            project / "models/madvr09217-index.json")
            root = project / "weights"
            root.mkdir()
            marker = root / "user-data"
            marker.write_text("preserve")
            full = root / "weights"
            full.mkdir()  # An existing empty archive destination remains supported.
            extracted = run_script("extract_ngu_models.py", ARGS.madvr.resolve(), project=project)
            self.assertEqual(extracted.returncode, 0, extracted.stderr)
            manifest = json.loads((full / "manifest.json").read_text())
            self.assertEqual((len(manifest["models"]), len(manifest["auxiliary_models"])), (140, 72))
            self.assertEqual(len(list(full.rglob("*.f32le"))), 3284)
            converted = run_script("convert_ngu_fp16.py", project=project)
            self.assertEqual(converted.returncode, 0, converted.stderr)
            self.assertEqual((root / "weights-fp16/manifest.json").read_bytes(),
                             (ROOT / "models/madvr09217-fp16-index.json").read_bytes())
            repeated = run_script("extract_ngu_models.py", ARGS.madvr.resolve(), project=project)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("new or empty directory", repeated.stderr)
            self.assertEqual(marker.read_text(), "preserve")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="<common-root>/weights FP32 archive (read-only)")
    parser.add_argument("--madvr", type=Path, help="authorized madVR64.ax 0.92.17")
    parser.add_argument("--cmake", type=Path, help="CMake executable for fresh default-configuration tests")
    parser.add_argument("--generator", help="parent CMake generator")
    ARGS = parser.parse_args()
    unittest.main(argv=[__file__, "-v"])
