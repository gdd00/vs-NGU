#!/usr/bin/env python3
"""Exercise the weight installer, including a real offline wheel build.

The build uses the current interpreter's pip, scikit-build-core and
VapourSynth, plus the same native build prerequisites as CMake. Optional
--madvr exercises full extraction without shipping a third-party binary.
Set NGU_REQUIRE_WHEEL_TEST=1 to make missing wheel test tools a failure.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import sysconfig
import tempfile
from threading import Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from vsngu import cli

PAYLOAD = b"synthetic binary for validation tests"
EXPECTED_FP16 = ROOT / "models/madvr09217-fp16-index.json"
BUNDLE = {
    "vsngu/_bundled/tools/extract_madvr.py",
    "vsngu/_bundled/tools/extract_ngu_models.py",
    "vsngu/_bundled/tools/convert_ngu_fp16.py",
    "vsngu/_bundled/tools/generate_ngu_catalog.py",
    "vsngu/_bundled/models/madvr09217-index.json",
}


def archive(path: Path, payload: bytes = PAYLOAD, *, exact=True) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("madVR64 [debug].ax", b"wrong debug contents")
        z.writestr("nested/madVR64.ax", b"wrong nested contents")
        z.writestr("../madVR64.ax", b"do not extract")
        z.writestr("madVR.ax", b"wrong x86 contents")
        if exact:
            z.writestr("madVR64.ax", payload)
    return path


@contextmanager
def http_server(routes):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.path, self.headers))
            status, headers, body = routes[self.path]
            self.send_response(status)
            for name, value in headers.items():
                self.send_header(name, str(value))
            self.end_headers()
            try:
                for offset in range(0, len(body), 1024):
                    self.wfile.write(body[offset:offset + 1024])
            except (BrokenPipeError, ConnectionResetError):
                pass  # The client may reject a response before reading it all.

        def log_message(self, *_):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", seen
        finally:
            server.shutdown()
            thread.join()


class WeightsCliTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.root = self.base / "target"
        self.enterContext(patch.dict(os.environ, {"no_proxy": "127.0.0.1", "NO_PROXY": "127.0.0.1"}))

    def invoke(self, *args, root=None, status=0):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            result = cli.main(["--weights-root", str(root or self.root), *map(str, args)])
        self.assertEqual(result, status, out.getvalue() + err.getvalue())
        self.assertEqual(list(self.base.rglob(".vsngu-*")), [])
        return out.getvalue(), err.getvalue()

    def check_rejected(self, root, message):
        with patch.object(cli, "download") as download:
            _, error = self.invoke(root=root, status=1)
        self.assertIn(message, error)
        download.assert_not_called()
        return error

    def run_process(self, *command, env=None):
        result = subprocess.run(list(map(str, command)), cwd=self.base, env=env,
                                capture_output=True, text=True, timeout=240)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_default_path_prefers_wheel_only_with_native_plugin(self):
        sibling = self.base / "vapoursynth/plugins/vsngu"
        sibling.mkdir(parents=True)
        fallback = self.base / "other/plugins"
        api = SimpleNamespace(get_plugin_dir=lambda: str(fallback))
        with (patch.object(cli, "__file__", str(self.base / "vsngu/cli.py")),
              patch.dict(sys.modules, {"vapoursynth": api})):
            self.assertEqual(cli.default_weights_root(), fallback / "vsngu/weights")
            for name in ("vsngu.so", "vsngu.dll"):
                with self.subTest(name=name), patch.dict(sys.modules, {"vapoursynth": None}):
                    plugin = sibling / name
                    plugin.touch()
                    self.assertEqual(cli.default_weights_root(), sibling / "weights")
                    plugin.unlink()

    def test_default_path_without_plugin_directory(self):
        for api in (None, SimpleNamespace(get_plugin_dir=lambda: ""),
                    SimpleNamespace(get_plugin_dir=lambda: " \t\n")):
            with (self.subTest(api=api),
                  patch.object(cli, "__file__", str(self.base / "vsngu/cli.py")),
                  patch.dict(sys.modules, {"vapoursynth": api}),
                  self.assertRaisesRegex(ValueError, "cannot locate.*--weights-root")):
                cli.default_weights_root()

    def test_missing_tool_resources_fail_before_download(self):
        package = self.base / "python/vsngu"
        for resources in (self.base, package / "_bundled"):
            for bundled in sorted(BUNDLE):
                path = resources / bundled.removeprefix("vsngu/_bundled/")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()

            with patch.object(cli, "__file__", str(package / "cli.py")):
                self.assertEqual(cli.tools_dir(), resources / "tools")
                for bundled in sorted(BUNDLE):
                    relative = bundled.removeprefix("vsngu/_bundled/")
                    with self.subTest(resources=resources, missing=relative):
                        path = resources / relative
                        path.unlink()
                        self.assertIn(relative, self.check_rejected(self.root, "missing NGU weight tool resources"))
                        path.touch()

    def test_download_streams_with_videohelp_headers(self):
        # Incompressible data makes the response span several read chunks.
        source = archive(self.base / "source.zip", os.urandom(3 * cli.CHUNK_BYTES))
        data = source.read_bytes()
        copied = self.base / "downloaded.zip"

        class ObservedBody:
            def __init__(self, body):
                self.body = body
                self.requests = []

            def read(self, count):
                self.requests.append(count)
                return self.body.read(count)

        real_copy = cli.copy_bounded
        observed = []

        def copy(body, *args):
            wrapped = ObservedBody(body)
            observed.append(wrapped.requests)
            return real_copy(wrapped, *args)

        with (http_server({"/zip": (200, {"Content-Length": len(data)}, data)}) as (url, seen),
              patch.object(cli, "copy_bounded", side_effect=copy)):
            cli.download(url + "/zip", copied)
        self.assertEqual(copied.read_bytes(), data)
        self.assertEqual(len(observed), 1)
        self.assertGreater(len(observed[0]), 2)
        self.assertTrue(all(0 < n <= cli.CHUNK_BYTES for n in observed[0]))
        (path, headers), = seen
        self.assertEqual(path, "/zip")
        self.assertEqual(headers["Referer"], "https://www.videohelp.com/software/madVR")
        self.assertIn("Mozilla/5.0", headers["User-Agent"])

    def test_download_preserves_headers_across_http_redirect(self):
        data = archive(self.base / "source.zip").read_bytes()
        copied = self.base / "downloaded.zip"
        with (http_server({"/zip": (200, {"Content-Length": len(data)}, data)}) as (target, target_seen),
              http_server({"/redirect": (302, {"Location": target + "/zip"}, b"")}) as (url, seen)):
            cli.download(url + "/redirect", copied)
        self.assertEqual(copied.read_bytes(), data)
        for requests, expected_path in ((seen, "/redirect"), (target_seen, "/zip")):
            (path, headers), = requests
            self.assertEqual(path, expected_path)
            self.assertEqual(headers["User-Agent"], cli.USER_AGENT)
            self.assertEqual(headers["Referer"], cli.REFERER)

    def test_download_rejects_bad_http_and_cleans_up(self):
        routes = {
            "/bad-length": (200, {"Content-Length": "not-an-integer"}, cli.ZIP_MAGIC),
            "/large": (200, {"Content-Length": cli.MAX_ZIP_BYTES + 1}, b""),
            "/stream-large": (200, {}, cli.ZIP_MAGIC + b"x" * 512),
            "/html": (200, {}, b"<html>not a ZIP</html>"),
            "/short": (200, {"Content-Length": 100}, cli.ZIP_MAGIC),
            "/partial": (206, {}, cli.ZIP_MAGIC),
            "/redirect": (302, {"Location": "ftp://127.0.0.1/zip"}, b""),
        }
        with http_server(routes) as (url, seen), patch.object(cli, "MAX_ZIP_BYTES", 128):
            for route, message in (("/bad-length", "invalid HTTP Content-Length: not-an-integer"),
                                   ("/large", "limit"), ("/stream-large", "limit"),
                                   ("/html", "ZIP magic"), ("/short", "incomplete download"),
                                   ("/partial", "status: 206"), ("/redirect", "http:// or https://")):
                with self.subTest(route=route):
                    _, error = self.invoke("--url", url + route, status=1)
                    self.assertIn(message, error)
                    self.assertFalse(self.root.exists())
        self.assertEqual([path for path, _ in seen], list(routes))

        for url in ("file:///somewhere/madVR.zip", "ftp://example.org/file", "https:///missing-host"):
            with self.subTest(url=url), patch.object(cli, "download") as download:
                _, error = self.invoke("--url", url, status=1)
                self.assertIn("http:// or https://", error)
                download.assert_not_called()

    def test_local_ax_zip_and_http_zip_pipeline(self):
        pe = self.base / "madVR64.ax"
        pe.write_bytes(PAYLOAD)
        zipped = archive(self.base / "source.zip")

        def fake_tool(script, *args):
            output = args[-1]
            if script.name == "extract_ngu_models.py":
                self.assertEqual(args[0].read_bytes(), PAYLOAD)
                self.assertEqual(args[0].name, "madVR64.ax")
            else:
                self.assertEqual(script.name, "convert_ngu_fp16.py")
                self.assertTrue((args[1] / "manifest.json").is_file())
            output.mkdir()
            (output / "manifest.json").write_text("{}")

        with (patch.object(cli, "SOURCE_SIZE", len(PAYLOAD)),
              patch.object(cli, "SOURCE_SHA256", hashlib.sha256(PAYLOAD).hexdigest()),
              patch.object(cli, "run_tool", side_effect=fake_tool) as run):
            for source in (pe, zipped):
                with self.subTest(source=source.name), patch.object(cli, "download") as download:
                    out, _ = self.invoke("--source", source)
                    download.assert_not_called()
                    self.assertIn("Reading", out)
                    self.assertIn("SHA-256", out)
                    self.assertIn("Extracting FP32", out)
                    self.assertIn("to FP16", out)
                    self.assertIn("Done", out)
                    self.assertEqual(run.call_count, 2)
                    run.reset_mock()
                    shutil.rmtree(self.root)

            with http_server({"/zip": (200, {}, zipped.read_bytes())}) as (url, seen):
                out, _ = self.invoke("--url", url + "/zip")
            self.assertIn("Downloading", out)
            self.assertEqual(len(seen), 1)
            self.assertTrue((self.root / "weights/manifest.json").is_file())
            self.assertTrue((self.root / "weights-fp16/manifest.json").is_file())
            self.assertFalse((self.base / "nested").exists())

    def test_zip_exact_name_wrong_size_wrong_sha_and_cleanup(self):
        no_root = archive(self.base / "no-root.zip", exact=False)
        duplicate = archive(self.base / "duplicate.zip")
        with zipfile.ZipFile(duplicate, "a") as z, warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Duplicate name:", category=UserWarning)
            z.writestr(cli.SOURCE_NAME, PAYLOAD)
        short = archive(self.base / "wrong-size.zip", b"small")
        wrong = archive(self.base / "wrong-hash.zip", b"x" * len(PAYLOAD))
        ax = self.base / "local.ax"
        ax.write_bytes(b"x" * len(PAYLOAD))
        with (patch.object(cli, "SOURCE_SIZE", len(PAYLOAD)),
              patch.object(cli, "SOURCE_SHA256", hashlib.sha256(PAYLOAD).hexdigest()),
              patch.object(cli, "run_tool") as run):
            for path, message in ((no_root, "root entry"), (duplicate, "root entry"),
                                  (short, "size"), (wrong, "SHA-256 mismatch"), (ax, "SHA-256 mismatch")):
                with self.subTest(path=path):
                    _, error = self.invoke("--source", path, status=1)
                    self.assertIn(message, error)
                    self.assertFalse(self.root.exists())
            run.assert_not_called()

    def test_zip_rejects_encryption_unsupported_compression_and_oversized_stream(self):
        # Use tiny single-member ZIPs so the test works without a crypto or
        # LZMA backend and never needs a large payload to exercise the cap.
        def write_zip(name, data, header_format, local_offset, central_offset, value):
            path = self.base / name
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr(cli.SOURCE_NAME, data)
            raw = bytearray(path.read_bytes())
            central = raw.index(b"PK\x01\x02")
            struct.pack_into(header_format, raw, local_offset, value)
            struct.pack_into(header_format, raw, central + central_offset, value)
            path.write_bytes(raw)
            return path

        encrypted = write_zip("encrypted.zip", PAYLOAD, "<H", 6, 8, 1)
        lzma = write_zip("lzma.zip", PAYLOAD, "<H", 8, 10, zipfile.ZIP_LZMA)
        oversized = write_zip("oversized.zip", PAYLOAD + b"x", "<I", 22, 24, len(PAYLOAD))
        # Matching the checksum of the allowed prefix must not hide overflow.
        prefix_crc = self.base / "oversized-prefix-crc.zip"
        raw = bytearray(oversized.read_bytes())
        struct.pack_into("<I", raw, 14, zlib.crc32(PAYLOAD))
        struct.pack_into("<I", raw, raw.index(b"PK\x01\x02") + 16, zlib.crc32(PAYLOAD))
        prefix_crc.write_bytes(raw)

        with (patch.object(cli, "SOURCE_SIZE", len(PAYLOAD)),
              patch.object(cli, "SOURCE_SHA256", hashlib.sha256(PAYLOAD).hexdigest()),
              patch.object(cli, "run_tool") as run):
            for path, message in ((encrypted, "unsupported ZIP compression or encryption"),
                                  (lzma, "unsupported ZIP compression or encryption"),
                                  (oversized, "byte limit"), (prefix_crc, "Bad CRC-32")):
                with self.subTest(path=path):
                    _, error = self.invoke("--source", path, status=1)
                    self.assertIn(message, error)
                    self.assertFalse(self.root.exists())
            run.assert_not_called()

    def test_publish_failure_rolls_back_both_archives(self):
        source = archive(self.base / "source.zip")
        self.root.mkdir()
        marker = self.root / "user-data"
        marker.write_text("unchanged")

        def fake_tool(script, *args):
            output = args[-1]
            output.mkdir()
            (output / "data").write_text("weights")
            (output / "manifest.json").write_text("{}")

        rename = Path.rename

        def fail_second_manifest(path, destination):
            if destination == self.root / "weights-fp16/manifest.json":
                self.assertTrue((self.root / "weights/manifest.json").is_file())
                self.assertTrue((self.root / "weights-fp16/data").is_file())
                raise OSError("simulated failure publishing FP16")
            return rename(path, destination)

        with (patch.object(cli, "SOURCE_SIZE", len(PAYLOAD)),
              patch.object(cli, "SOURCE_SHA256", hashlib.sha256(PAYLOAD).hexdigest()),
              patch.object(cli, "run_tool", side_effect=fake_tool),
              patch.object(Path, "rename", new=fail_second_manifest)):
            _, error = self.invoke("--source", source, status=1)
        self.assertIn("simulated failure publishing FP16", error)
        self.assertEqual(list(self.root.iterdir()), [marker])
        self.assertEqual(marker.read_text(), "unchanged")

    def test_run_tool_ignores_pythonpath_and_does_not_write_bytecode(self):
        tools = self.base / "tools"
        tools.mkdir()
        for name in ("extract_ngu_models.py", "extract_madvr.py"):
            shutil.copyfile(ROOT / "tools" / name, tools / name)
        shadow = self.base / "shadow"
        shadow.mkdir()
        marker = self.base / "shadow-imported"
        (shadow / "extract_madvr.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
        (shadow / "sitecustomize.py").write_text("import extract_madvr\n")

        with patch.dict(os.environ, {"PYTHONPATH": str(shadow)}):
            cli.run_tool(tools / "extract_ngu_models.py", "--help")
        self.assertFalse(marker.exists())
        self.assertFalse((tools / "__pycache__").exists())

    def test_both_archives_are_idempotent(self):
        for name in ("weights", "weights-fp16"):
            path = self.root / name / "manifest.json"
            path.parent.mkdir(parents=True)
            path.write_text("unchanged")
        with patch.object(cli, "download") as download, patch.object(cli, "run_tool") as run:
            out, _ = self.invoke("--source", self.base / "does-not-exist.ax")
            self.assertIn("already exist", out)
        download.assert_not_called()
        run.assert_not_called()
        for manifest in self.root.glob("*/manifest.json"):
            self.assertEqual(manifest.read_text(), "unchanged")

    def test_existing_fp32_converts_real_archive_without_downloading(self):
        full = self.root / "weights"
        shutil.copytree(ARGS.fp32, full)
        before = (full / "manifest.json").read_bytes()
        with patch.object(cli, "download") as download:
            out, _ = self.invoke()
        self.assertIn("Using existing FP32 archive", out)
        download.assert_not_called()
        self.assertEqual((full / "manifest.json").read_bytes(), before)
        self.check_fp16(self.root)

    def check_fp16(self, root):
        half = root / "weights-fp16"
        self.assertEqual((half / "manifest.json").read_bytes(), EXPECTED_FP16.read_bytes())
        self.assertEqual(len(list(half.rglob("*.f16le"))), 3284)

    def test_failed_conversion_leaves_existing_fp32_untouched(self):
        full = self.root / "weights"
        full.mkdir(parents=True)
        (full / "manifest.json").write_text("incomplete")
        with patch.object(cli, "download") as download:
            _, error = self.invoke(status=1)
        self.assertIn("convert_ngu_fp16", error)
        download.assert_not_called()
        self.assertEqual((full / "manifest.json").read_text(), "incomplete")
        self.assertFalse((self.root / "weights-fp16").exists())

    def test_preflight_conflicts(self):
        (self.root / "weights").mkdir(parents=True)
        marker = self.root / "weights/user-data"
        marker.write_text("preserve")
        self.check_rejected(self.root, "without a regular manifest.json")
        self.assertEqual(marker.read_text(), "preserve")
        marker.unlink()
        self.check_rejected(self.root, "without a regular manifest.json")
        (self.root / "weights").rmdir()
        half = self.root / "weights-fp16"
        half.mkdir()
        (half / "manifest.json").touch()
        self.check_rejected(self.root, "FP16-only")

    def test_preflight_path_aliases_and_archive_ancestors(self):
        self.check_rejected(self.base / "not-created/../target", "parent traversal")
        self.assertFalse((self.base / "not-created").exists())
        for manifest, suffix in ((True, "f32le"), (False, "f32le"), (False, "f16le")):
            with self.subTest(manifest=manifest, suffix=suffix):
                old = self.base / f"existing-{manifest}-{suffix}"
                (old / "model").mkdir(parents=True)
                (old / "model" / f"stage00.weights.{suffix}").touch()
                if manifest:
                    (old / "manifest.json").touch()
                self.check_rejected(old, "parameter archive")
                self.check_rejected(old / "sub/more", "parameter archive")

    def test_preflight_symlinks(self):
        alias = self.base / "alias"
        external = self.base / "external"
        try:
            alias.symlink_to(external, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlinks unavailable: {exc}")
        self.check_rejected(alias / "sub", "symlink")
        self.root.mkdir()
        (self.root / "weights").symlink_to(external, target_is_directory=True)
        self.check_rejected(self.root, "symlink")
        self.assertFalse(external.exists())

    def test_preflight_permission_error_gives_actionable_hint(self):
        denied = self.base / "read-only"
        denied.mkdir()
        # Also exercise the error path under root or on filesystems where
        # chmod cannot make this directory unwritable.
        denied.chmod(0o555)
        try:
            if os.access(denied, os.W_OK):
                with patch.object(cli.tempfile, "TemporaryDirectory", side_effect=PermissionError(denied)):
                    error = self.check_rejected(denied / "new-root", "permission denied")
            else:
                error = self.check_rejected(denied / "new-root", "permission denied")
        finally:
            denied.chmod(0o755)
        self.assertIn("--weights-root", error)
        self.assertIn("elevated permissions", error)
        self.assertEqual(list(denied.iterdir()), [])

    def test_offline_wheel_and_installed_entry_points(self):
        missing = [name for name in ("pip", "scikit_build_core") if importlib.util.find_spec(name) is None]
        if missing:
            reason = f"offline wheel test requires missing Python modules: {', '.join(missing)}"
            if os.environ.get("NGU_REQUIRE_WHEEL_TEST") == "1":
                self.fail(f"{reason} (NGU_REQUIRE_WHEEL_TEST=1)")
            self.skipTest(reason)

        wheels = self.base / "wheels"
        prefix = self.base / "prefix"
        env = {**os.environ, "PIP_NO_INDEX": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
               "PIP_NO_CACHE_DIR": "1", "PYTHONDONTWRITEBYTECODE": "1"}
        env.pop("PYTHONPATH", None)
        defines = ["FETCHCONTENT_FULLY_DISCONNECTED=ON", *ARGS.wheel_define]
        self.run_process(sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps",
                         "--no-build-isolation", "--wheel-dir", wheels,
                         *(f"--config-settings=cmake.define.{value}" for value in defines), ROOT, env=env)
        wheel, = wheels.glob("*.whl")
        with zipfile.ZipFile(wheel) as z:
            names = set(z.namelist())
            plugin = f"vapoursynth/plugins/vsngu/{ARGS.plugin_name}"
            self.assertIn(plugin, names)
            self.assertEqual({name for name in names if name.startswith("vsngu/")},
                             BUNDLE | {"vsngu/__init__.py", "vsngu/__main__.py", "vsngu/cli.py"})
            entry_points, = [name for name in names if name.endswith(".dist-info/entry_points.txt")]
            self.assertIn(b"vsngu-weights = vsngu.cli:main", z.read(entry_points))
            self.assertFalse(any(name.endswith((".ax", ".f32le", ".f16le")) for name in names))

        self.run_process(sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                         "--ignore-installed", "--prefix", prefix, wheel, env=env)
        config = sysconfig.get_paths(vars={"base": str(prefix), "platbase": str(prefix)})
        env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys([config["purelib"], config["platlib"]]))
        script_dir = Path(config["scripts"])
        script = script_dir / ("vsngu-weights.exe" if os.name == "nt" else "vsngu-weights")
        for cmd in ((script,), (sys.executable, "-m", "vsngu")):
            self.assertIn("--weights-root", self.run_process(*cmd, "--help", env=env))

        if ARGS.madvr:
            # Real wheel, real local AX, and wheel-first default weights root.
            self.run_process(script, "--source", ARGS.madvr, env=env)
            plugin_path, = [Path(p) / plugin for p in env["PYTHONPATH"].split(os.pathsep)
                            if (Path(p) / plugin).is_file()]
            root = plugin_path.parent / "weights"
            self.assertEqual(len(list((root / "weights").rglob("*.f32le"))), 3284)
        else:
            root = self.root
            shutil.copytree(ARGS.fp32, root / "weights")
            self.run_process(script, "--weights-root", root, env=env)
        self.check_fp16(root)

    def test_real_local_zip_round_trip(self):
        if ARGS.madvr is None:
            self.skipTest("supply CMake NGU_MADVR_SOURCE for genuine madVR extraction")
        zipped = archive(self.base / "source.zip", ARGS.madvr.read_bytes())
        with patch.object(cli, "download") as download:
            self.invoke("--source", zipped)
        download.assert_not_called()
        self.assertEqual(len(list((self.root / "weights").rglob("*.f32le"))), 3284)
        self.check_fp16(self.root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", type=Path, required=True)
    parser.add_argument("--madvr", type=lambda path: Path(path).resolve())
    parser.add_argument("--plugin-name", default="vsngu.dll" if os.name == "nt" else "vsngu.so")
    parser.add_argument("--wheel-define", action="append", default=[])
    ARGS = parser.parse_args()
    unittest.main(argv=[__file__, "-v"])
