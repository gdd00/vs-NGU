#!/usr/bin/env python3
"""Independent stdlib scalar oracle and integration tests for FP16 experiments."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
import unittest

import vapoursynth as vs
from half_reference import fma as half_fma

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {28: "aa", 32: "soft", 36: "standard", 40: "sharp"}
QUALITIES = ("low", "medium", "high", "very_high")


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f16(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def add(a, b):
    return f32(a + b)


def mul(a, b):
    return f32(a * b)


def scalar_graph(source, width, height, params, aa, half=False):
    """Independent graph: exact integer half FMA, separate half baseline/combine."""
    rounding = f16 if half else f32
    add = lambda a, b: rounding(a + b)
    mul = lambda a, b: rounding(a * b)
    mac = half_fma if half else lambda a, b, c: add(c, mul(a, b))
    body = params[:-3 if aa else -1]
    def column_order(i):
        if not half or len(body) == 3 or i == 0:
            return False
        if len(body) == 5:
            return bool(i % 2)
        return not (len(body) == 6 and i == 5)
    head_column = half and (len(body) in (5, 6) or (len(body) == 8 and aa))
    if half:
        params = [(stage, [f16(v) for v in w], [f16(v) for v in b])
                  for stage, w, b in params]
    def sample(values, w, h, channels, x, y, c=0):
        return values[(min(max(y, 0), h - 1) * w + min(max(x, 0), w - 1)) * channels + c]

    def conv(values, stage, weights, bias, activation=True, column=False):
        ci, co = stage["in_channels"], stage["out_channels"]
        out = []
        for y in range(height):
            for x in range(width):
                for oc in range(co):
                    value = bias[oc]
                    taps = ((ky, kx, ic) for kx in range(3) for group in range(0, ci, 4)
                            for ky in range(3) for ic in range(group, group + 4)) if column else (
                            (ky, kx, ic) for ky in range(3) for kx in range(3) for ic in range(ci))
                    for ky, kx, ic in taps:
                        value = mac(sample(values, width, height, ci, x + kx - 1, y + ky - 1, ic),
                                    weights[(oc * ci + ic) * 9 + ky * 3 + kx], value)
                    out.append(mul(value, rounding(0.1)) if activation and value < 0 else value)
        return out

    features = source
    for i, stage in enumerate(body):
        features = conv(features, *stage, column=column_order(i))
    if aa:
        heads = params[-3:]
        outputs = [conv(features, heads[h][0], heads[h][1], heads[lane][2], False, head_column)
                   for lane, h in enumerate((1, 0, 2))]
        out = []
        for y in range(height * 2):
            for x in range(width * 2):
                ix, iy = x // 2, y // 2
                v = sample(source, width, height, 1, ix, iy)
                i = iy * width + ix
                if x % 2 and y % 2:
                    v = add(add(add(v, sample(source, width, height, 1, ix + 1, iy)),
                                sample(source, width, height, 1, ix, iy + 1)),
                            sample(source, width, height, 1, ix + 1, iy + 1))
                    v = add(mul(v, 0.25), outputs[2][i])
                elif x % 2:
                    v = add(mul(add(v, sample(source, width, height, 1, ix + 1, iy)), 0.5), outputs[1][i])
                elif y % 2:
                    v = add(mul(add(v, sample(source, width, height, 1, ix, iy + 1)), 0.5), outputs[0][i])
                out.append(v)
        return out

    ca, cb, cc, cd = [struct.unpack("<f", struct.pack("<I", b))[0]
                      for b in (0x3f5f3333, 0x3e766666, 0xbdaccccd, 0xbce66666)]

    if half:
        ca, cb, cc, cd = [f16(v) for v in (ca, cb, cc, cd)]

    def baseline(values, w, h, horizontal):
        ow, oh = w * (2 if horizontal else 1), h * (1 if horizontal else 2)
        out = []
        for y in range(oh):
            for x in range(ow):
                pos = x if horizontal else y
                first = pos // 2 - (1 if pos % 2 else 2)
                coeffs = (cc, ca, cb, cd) if pos % 2 else (cd, cb, ca, cc)
                value = 0.0
                for k, coeff in enumerate(coeffs):
                    value = add(value, mul(coeff, sample(values, w, h, 1,
                        first + k if horizontal else x, y if horizontal else first + k)))
                out.append(value)
        return out

    base = baseline(baseline(source, width, height, True), width * 2, height, False)
    stage, weights, bias = params[-1]
    ci = stage["in_channels"]
    out = []
    for y in range(height * 2):
        for x in range(width * 2):
            value = bias[0]
            if half:
                # Scalar source/output geometry, independent of GPU packing.
                taps = ((dy, dx, ic) for dx in (-1, 0, 1) for group in range(0, ci, 4)
                        for dy in (-1, 0, 1) for ic in range(group, group + 4)) if head_column else (
                        (dy, dx, ic) for dy in (-1, 0, 1) for dx in (-1, 0, 1) for ic in range(ci))
                for dy, dx, ic in taps:
                    ky, kx = y % 2 + 1 - 2 * dy, x % 2 + 1 - 2 * dx
                    w = weights[ic * 16 + ky * 4 + kx] if 0 <= ky < 4 and 0 <= kx < 4 else 0
                    value = mac(sample(features, width, height, ci,
                        x // 2 + dx, y // 2 + dy, ic), w, value)
            else:
                for ic in range(ci):
                    for ky in range((y + 1) % 2, 4, 2):
                        for kx in range((x + 1) % 2, 4, 2):
                            value = add(value, mul(sample(features, width, height, ci,
                                (x + 1 - kx) // 2, (y + 1 - ky) // 2, ic), weights[ic * 16 + ky * 4 + kx]))
            out.append(add(base[y * width * 2 + x], value))
    return out


def source_clip(core, width, height, frames, bits=16, raw=False):
    template = core.std.BlankClip(width=width, height=height, length=len(frames),
                                 format=vs.GRAYH if bits == 16 else vs.GRAYS,
                                 fpsnum=24000, fpsden=1001)
    code = "H" if raw else ("e" if bits == 16 else "f")

    def fill(n, f):
        out = f.copy()
        ptr, stride = out.get_write_ptr(0).value, out.get_stride(0)
        for y in range(height):
            data = struct.pack(f"<{width}{code}", *frames[n][y * width:(y + 1) * width])
            ctypes.memmove(ptr + y * stride, data, len(data))
        out.props["_TestStamp"] = n
        out.props["_Range"] = vs.RANGE_FULL
        out.props["_SARNum"], out.props["_SARDen"] = 4, 3
        return out

    return core.std.ModifyFrame(template, clips=template, selector=fill)


def frame_values(frame):
    code = "e" if frame.format.bits_per_sample == 16 else "f"
    return list(struct.unpack(f"<{frame.width * frame.height}{code}", bytes(frame[0])))


def parameters(model, root, reduced):
    result = []
    for stage in model["stages"]:
        arrays = []
        for kind in ("weights", "bias"):
            data = (root / model["id"] / f'stage{stage["index"]:02d}.{kind}.f32le').read_bytes()
            spec = stage[kind]
            if len(data) != spec["size"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
                raise ValueError("parameter integrity failure")
            values = list(struct.unpack(f"<{len(data) // 4}f", data))
            if kind == "bias":
                values = [f32(v / 255.0) for v in values]
            arrays.append([f16(v) for v in values] if reduced else values)
        result.append((stage, *arrays))
    return result


def difference_8bit(actual, reference):
    assert len(actual) == len(reference) and actual
    errors = sorted(abs(a - b) * 255 for a, b in zip(actual, reference))
    return {"max_abs_8bit": errors[-1],
            "mean_abs_8bit": sum(errors) / len(errors),
            "rmse_8bit": math.sqrt(sum(e * e for e in errors) / len(errors)),
            "p95_abs_8bit": errors[math.ceil(len(errors) * 0.95) - 1],
            "p99_abs_8bit": errors[math.ceil(len(errors) * 0.99) - 1]}


class FP16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = vs.core
        cls.core.num_threads = 8
        cls.core.std.LoadPlugin(path=str(ARGS.plugin.resolve()))
        index = json.loads((ROOT / "models/madvr09217-index.json").read_text())
        cls.models = [m for m in index["models"] if m["scale"] == 2]
        cls.metrics = []
        cls.quality_metrics = []
        cls.backend_max_error = 0.0
        cls.backend_error_by_format = {}

    def output(self, source, model, precision=None, backend="cpu", weights=None):
        options = {} if precision is None else {"weights_precision": precision}
        return self.core.ngu.NGU2x(
            source, weights=str(ARGS.weights if weights is None else weights),
            mode=FAMILIES[model["family_id"]],
            quality=QUALITIES[model["quality"]], option=model["selector_option"],
            backend=backend, **options)

    def check_frame(self, frame, expected, bits, precision, backend, exact=False):
        self.assertEqual(frame.format.id, vs.GRAYH if bits == 16 else vs.GRAYS)
        self.assertEqual(frame.props["_NGUPrecision"], "fp16" if bits == 16 and backend == "vulkan" else "fp32")
        self.assertEqual(frame.props["_NGUWeightPrecision"], precision)
        self.assertEqual(frame.props["_NGUBackend"], backend)
        self.assertEqual(frame.props["_Range"], vs.RANGE_FULL)
        self.assertEqual((frame.props["_SARNum"], frame.props["_SARDen"]), (4, 3))
        values = frame_values(frame)
        self.assertEqual(len(values), len(expected))
        for actual, ref in zip(values, expected):
            self.assertTrue(math.isfinite(actual) and math.isfinite(ref))
            if exact:
                self.assertEqual(struct.pack("<e" if bits == 16 else "<f", actual),
                                 struct.pack("<e" if bits == 16 else "<f", ref))
            else:
                tolerance = 2e-5 + 2e-5 * abs(ref)
                self.assertLessEqual(abs(actual - ref), tolerance)
            error = abs(actual - ref)
            type(self).backend_max_error = max(type(self).backend_max_error, error)
            key = f"{bits}/{precision}/{backend}"
            self.backend_error_by_format[key] = max(self.backend_error_by_format.get(key, 0), error)
        return values

    def test_all_72_models_against_independent_scalar(self):
        self.assertEqual(len(self.models), 72)
        original = [f32(v) for v in (-0.12345679, 1.2345679, 0.33333334, 0.0, 0.75, -0.5)]
        for model in self.models:
            actual_gpu = {}
            params = {p: parameters(model, ARGS.weights / "weights", p == "fp16") for p in ("fp32", "fp16")}
            for bits in (32, 16):
                precision = "fp16" if bits == 16 else "fp32"
                samples = original if bits == 32 else [f16(v) for v in original]
                source = source_clip(self.core, 3, 2, [samples], bits)
                gpu = self.core.std.GPUUpload(source)
                # Keep the weight-only scalar metric; mismatched archives are
                # no longer legal plugin calls.
                references = {p: scalar_graph(samples, 3, 2, params[p], model["family_id"] == 28)
                              for p in ("fp32", "fp16")}
                ref = references[precision]
                stored = [f16(v) for v in ref] if bits == 16 else ref
                for backend, clip in (("cpu", source), ("vulkan", gpu)):
                    with self.subTest(model=model["id"], bits=bits, precision=precision, backend=backend):
                        out = self.output(clip, model, backend=backend)
                        self.assertEqual((out.width, out.height, out.fps_num, out.fps_den, out.num_frames),
                                         (6, 4, 24000, 1001, 1))
                        self.assertEqual(out.gpu_resident, backend == "vulkan")
                        if backend == "vulkan":
                            out = self.core.std.GPUDownload(out)
                        frame = out.get_frame(0)
                        self.assertEqual(frame.props["_NGUModel"], model["id"])
                        expected = scalar_graph(samples, 3, 2, params[precision],
                            model["family_id"] == 28, half=True) if backend == "vulkan" and bits == 16 else stored
                        if backend == "vulkan" and bits == 16:
                            self.assertNotEqual(expected, stored)  # Detect FP32 compute + half boundary casts.
                        values = self.check_frame(frame, expected, bits, precision, backend,
                                                  exact=backend == "cpu" or bits == 16)
                        if backend == "vulkan":
                            actual_gpu[bits, precision] = values
                errors = [abs(a - b) for a, b in zip(references["fp16"], references["fp32"])]
                self.metrics.append({"model": model["id"], "input_bits": bits,
                                     "weight_only_max_abs_8bit": max(errors) * 255,
                                     "weight_only_rmse_8bit": math.sqrt(sum(e * e for e in errors) / len(errors)) * 255})
            quality = difference_8bit(actual_gpu[16, "fp16"], actual_gpu[32, "fp32"])
            self.assertGreater(quality["max_abs_8bit"], 0)
            self.quality_metrics.append({"model": model["id"], "fixture": "extended_range_3x2", **quality})

    def test_odd_sizes_and_tile_boundaries(self):
        representatives = [m for m in self.models if m["selector_option"] == 0]
        rng = random.Random(1643)
        for width, height in ((1, 1), (1, 5), (5, 1), (5, 3), (6, 4), (33, 13), (65, 25)):
            samples = [f16(rng.uniform(-0.5, 1.5)) for _ in range(width * height)]
            source = source_clip(self.core, width, height, [samples], bits=32)
            gpu = self.core.std.GPUUpload(source)
            for model in representatives:
                with self.subTest(size=(width, height), model=model["id"]):
                    cpu = self.output(source, model)
                    out = self.output(gpu, model, backend="vulkan")
                    expected = frame_values(cpu.get_frame(0))
                    frame = self.core.std.GPUDownload(out).get_frame(0)
                    self.check_frame(frame, expected, 32, "fp32", "vulkan")

    def test_quality_in_8bit_units(self):
        self.quality_image(257, 145)

    def test_large_nonconstant_quality_in_8bit_units(self):
        self.quality_image(1919, 1079)

    def quality_image(self, width, height):
        rng = random.Random(1163)
        # One deterministic image with flat areas, gradients, edges, fine
        # patterns and noise. Synthetic only: not a claim about real video.
        samples = []
        quarter = width // 4
        for y in range(height):
            for x in range(width):
                if x < quarter:
                    v = (x + y) / (quarter - 1 + height - 1)
                elif x < 2 * quarter:
                    v = 0.125 if y < height // 2 else 0.875
                elif x < 3 * quarter:
                    v = 0.25 if (x + y) % 2 else 0.75
                else:
                    v = rng.random()
                samples.append(f32(v))
        source = source_clip(self.core, width, height, [samples], bits=32)
        half = source_clip(self.core, width, height, [[f16(v) for v in samples]])
        gpu32, gpu16 = self.core.std.GPUUpload(source), self.core.std.GPUUpload(half)
        for model in (m for m in self.models if m["selector_option"] == 0 and
                      (width == 257 or m["family_id"] in (28, 40))):
            full = self.core.std.GPUDownload(self.output(gpu32, model, "fp32", "vulkan")).get_frame(0)
            reduced = self.core.std.GPUDownload(self.output(gpu16, model, "fp16", "vulkan")).get_frame(0)
            a, b = frame_values(reduced), frame_values(full)
            self.assertTrue(all(math.isfinite(v) for v in a + b))
            metric = difference_8bit(a, b)
            self.assertEqual(reduced.props["_NGUPrecision"], "fp16")
            self.assertEqual(full.props["_NGUPrecision"], "fp32")
            self.quality_metrics.append({"model": model["id"], "fixture": f"texture_{width}x{height}", **metric})

    def test_all_finite_half_codes_preserved_at_aa_native_phase(self):
        codes = [v for v in range(65536) if v & 0x7c00 != 0x7c00]
        self.assertEqual(len(codes), 256 * 248)
        source = source_clip(self.core, 256, 248, [codes], raw=True)
        model = next(m for m in self.models if m["id"] == "aa_2x_low")
        for precision in (None, "fp16"):
            output = self.output(self.core.std.GPUUpload(source), model, precision, "vulkan")
            frame = self.core.std.GPUDownload(output).get_frame(0)
            actual = struct.unpack(f"<{frame.width * frame.height}H", bytes(frame[0]))
            native = [actual[(y * 2) * frame.width + x * 2] for y in range(248) for x in range(256)]
            self.assertEqual(native, codes)  # Includes every subnormal, -0 and +/-65504.
        # Only the AA native phase is checked here: odd-phase FP16 overflow is
        # allowed on GPU, as in the existing GPU non-finite propagation contract.

    def test_validation_defaults_and_explicit_residency(self):
        model = self.models[0]
        for bits in (16, 32):
            source = source_clip(self.core, 3, 2, [[0.375] * 6], bits)
            expected = "fp16" if bits == 16 else "fp32"
            other = "fp32" if bits == 16 else "fp16"
            for backend, clip in (("cpu", source), ("vulkan", self.core.std.GPUUpload(source))):
                for precision in (None, expected):
                    out = self.output(clip, model, precision, backend)
                    if backend == "vulkan":
                        out = self.core.std.GPUDownload(out)
                    frame = out.get_frame(0)
                    self.assertEqual(frame.props["_NGUWeightPrecision"], expected)
                    self.assertEqual(frame.props["_NGUPrecision"],
                                     expected if backend == "vulkan" else "fp32")
                # Mismatches must fail before any archive access, even if no
                # weights exist at all.
                with self.assertRaisesRegex(vs.Error, "weights_precision conflicts.*requires " + expected):
                    self.output(clip, model, other, backend, weights="/nonexistent/ngu-weights")
                for weights in ("", "weights\0trailing"):
                    with self.assertRaisesRegex(vs.Error, "weights must be a non-empty string without NUL"):
                        self.output(clip, model, backend=backend, weights=weights)
                for precision in ("", "FP16", "auto", "fp16\0", "bf16"):
                    with self.assertRaisesRegex(vs.Error, "weights_precision"):
                        self.output(clip, model, precision, backend)
                for child in ("weights", "weights-fp16"):
                    with self.assertRaisesRegex(vs.Error, "missing .* weights archive.*common root"):
                        self.output(clip, model, backend=backend, weights=ARGS.weights / child)
            with self.assertRaisesRegex(vs.Error, "GPUUpload"):
                self.output(source, model, backend="vulkan")
            with self.assertRaisesRegex(vs.Error, "GPUDownload"):
                self.output(self.core.std.GPUUpload(source), model, backend="cpu")
        for fmt in (vs.GRAY8, vs.GRAY16, vs.RGBH, vs.RGBS, vs.YUV444PH, vs.YUV444PS):
            source = self.core.std.BlankClip(width=4, height=4, format=fmt)
            with self.assertRaisesRegex(vs.Error, "GRAYS/GRAYH"):
                self.output(source, model)

    def test_cpu_nonfinite_overflow_and_recovery(self):
        model = next(m for m in self.models if m["id"] == "sharp_2x_low")
        for bad in (math.nan, math.inf, -math.inf):
            source = source_clip(self.core, 3, 2, [[bad] + [0.5] * 5, [0.5] * 6])
            out = self.output(source, model)
            with self.assertRaisesRegex(vs.Error, "NaN/Inf"):
                out.get_frame(0)
            self.assertTrue(all(math.isfinite(v) for v in frame_values(out.get_frame(1))))
        source = source_clip(self.core, 3, 2, [[-65504, 65504, -65504] * 2, [0.5] * 6])
        out = self.output(source, model)
        with self.assertRaisesRegex(vs.Error, "FP16 overflow"):
            out.get_frame(0)
        self.assertTrue(all(math.isfinite(v) for v in frame_values(out.get_frame(1))))

    def test_manifestless_plugin_autoload(self):
        model = "sharp_2x_low"
        with tempfile.TemporaryDirectory() as tmp:
            plugins = Path(tmp) / "plugins"
            directory = plugins / "vsngu"
            directory.mkdir(parents=True)
            plugin = directory / ARGS.plugin.name
            shutil.copy2(ARGS.plugin, plugin)
            for child in ("weights", "weights-fp16"):
                shutil.copytree(ARGS.weights / child / model, directory / "weights" / child / model)
            # Create the core after setting the extra path, without changing
            # the existing test core or calling LoadPlugin in the child.
            script = textwrap.dedent("""
                from pathlib import Path
                import sys
                import unittest
                import vapoursynth as vs

                check = unittest.TestCase()
                core = vs.core
                core.num_threads = 2
                check.assertEqual(Path(core.ngu.plugin_path).resolve(), Path(sys.argv[1]).resolve())
                for fmt, precision in ((vs.GRAYS, "fp32"), (vs.GRAYH, "fp16")):
                    clip = core.std.BlankClip(width=3, height=2, length=1, format=fmt, color=0.375)
                    implicit = core.ngu.NGU2x(clip, quality="low").get_frame(0)
                    explicit = core.ngu.NGU2x(clip, quality="low", weights=sys.argv[2]).get_frame(0)
                    check.assertEqual((implicit.width, implicit.height, implicit.format.id), (6, 4, fmt))
                    check.assertEqual(implicit.props["_NGUWeightPrecision"], precision)
                    check.assertEqual(implicit.props["_NGUBackend"], "cpu")
                    check.assertEqual(bytes(implicit[0]), bytes(explicit[0]))
                """)
            result = subprocess.run(
                [sys.executable, "-c", script, str(plugin.resolve()), str(ARGS.weights.resolve())],
                cwd=tmp, env={**os.environ, "VAPOURSYNTH_EXTRA_PLUGIN_PATH": str(plugins.resolve())},
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_format_selects_only_its_archive_without_fallback(self):
        model = self.models[0]
        for bits in (16, 32):
            precision = "fp16" if bits == 16 else "fp32"
            selected_child, other_child = ("weights-fp16", "weights") if bits == 16 else ("weights", "weights-fp16")
            extension = ".f16le" if bits == 16 else ".f32le"
            source = source_clip(self.core, 3, 2, [[0.375] * 6], bits)
            for backend, clip in (("cpu", source), ("vulkan", self.core.std.GPUUpload(source))):
                with self.subTest(bits=bits, backend=backend), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    for child in (selected_child, other_child):
                        shutil.copytree(ARGS.weights / child / model["id"], root / child / model["id"])
                    # A corrupt opposite archive must not be opened. Removing
                    # it entirely must also leave the selected path usable.
                    ignored = next((root / other_child).rglob("*.weights.*"))
                    ignored.write_bytes(b"must never be read")
                    for _ in range(2):
                        out = self.output(clip, model, backend=backend, weights=root)
                        if backend == "vulkan":
                            out = self.core.std.GPUDownload(out)
                        self.assertEqual(out.get_frame(0).props["_NGUWeightPrecision"], precision)
                        if (root / other_child).exists():
                            shutil.rmtree(root / other_child)
                    shutil.copytree(ARGS.weights / other_child / model["id"], root / other_child / model["id"])
                    selected = root / selected_child / model["id"] / ("stage00.weights" + extension)
                    selected.unlink()
                    with self.assertRaisesRegex(vs.Error, "cannot open parameter file.*" + selected_child + ".*" + extension):
                        self.output(clip, model, backend=backend, weights=root)
                    shutil.rmtree(root / selected_child)
                    with self.assertRaisesRegex(vs.Error, "missing " + precision.upper() + " weights archive.*no fallback"):
                        self.output(clip, model, backend=backend, weights=root)

    def test_native_half_archive_integrity_and_not_reopened(self):
        model = self.models[0]
        source = source_clip(self.core, 3, 2, [[0.375] * 6])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ARGS.weights / "weights-fp16" / model["id"],
                            root / "weights-fp16" / model["id"])
            self.assertEqual(list(root.rglob("*.f32le")), [])
            selected = root / "weights-fp16" / model["id"] / "stage00.weights.f16le"
            original = selected.read_bytes()
            for bad in (bytes([original[0] ^ 1]) + original[1:], original[:-1], original + b"\0"):
                selected.write_bytes(bad)
                with self.assertRaisesRegex(vs.Error, "SHA-256|size mismatch"):
                    self.output(source, model, weights=root)
            selected.write_bytes(original)
            out = self.output(source, model, weights=root)
        self.assertTrue(all(math.isfinite(v) for v in frame_values(out.get_frame(0))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True, help="common weights root")
    parser.add_argument("--report", type=Path)
    ARGS = parser.parse_args()
    result = unittest.main(argv=[__file__, "-v"], exit=False).result
    report = {
        "schema_version": 2,
        "precision": {"vulkan": {"GRAYH": "fp16 operations, accumulators, parameters and intermediates",
                                  "GRAYS": "fp32 operations, accumulators, parameters and intermediates"},
                      "cpu": "unchanged fp32 arithmetic with optional half I/O",
                      "weights": "GRAYH reads only weights-fp16/*.f16le; GRAYS reads only weights/*.f32le from a common root",
                      "vulkan_half_comparison": "current schedules bit exact against independent integer binary16 FMA graph; baseline/combine rounded separately",
                      "shader_guard": "every compiled SPIR-V module rejects floating types other than the selected width"},
        "quality_units": "8-bit pixel value units: abs(actual-native_fp32)*255, no clipping",
        "quality_limitations": "synthetic fixtures only; current GPU only; differences include input, parameter and arithmetic quantization",
        "quality_metrics": FP16Tests.quality_metrics,
        "scalar_fixture": {"input_size": [3, 2],
                           "values": [-0.12345679, 1.2345679, 0.33333334, 0, 0.75, -0.5]},
        "fp16_archive": json.loads((ARGS.weights / "weights-fp16/manifest.json").read_text())["summary"],
        "fp16_metadata_sha256": hashlib.sha256((ROOT / "models/madvr09217-fp16-index.json").read_bytes()).hexdigest(),
        "plugin_sha256": hashlib.sha256(ARGS.plugin.read_bytes()).hexdigest(),
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for folder in ("src", "tests") for p in sorted((ROOT / folder).glob("*"))
                          if p.is_file()},
        "tests_run": result.testsRun,
        "passed": result.wasSuccessful(),
        "scalar_graphs": len(FP16Tests.metrics) * 2 + len(FP16Tests.models),
        "scalar_plugin_comparisons": len(FP16Tests.metrics) * 2,
        "backend_max_abs_8bit": FP16Tests.backend_max_error * 255,
        "backend_error_by_format_8bit": {k: v * 255 for k, v in FP16Tests.backend_error_by_format.items()},
        "weight_only_max_abs_8bit": max((m["weight_only_max_abs_8bit"] for m in FP16Tests.metrics), default=0),
        "full_fp16_max_abs_8bit": max((m["max_abs_8bit"] for m in FP16Tests.quality_metrics), default=0),
        "metrics": FP16Tests.metrics,
    }
    print(json.dumps({k: v for k, v in report.items() if k not in ("source_sha256", "metrics", "quality_metrics")}, indent=2))
    if ARGS.report:
        ARGS.report.parent.mkdir(parents=True, exist_ok=True)
        with ARGS.report.open("x") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
