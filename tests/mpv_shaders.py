#!/usr/bin/env python3
"""Test generated NGU shaders in real mpv gpu-next/Vulkan windows.

Requires a working display, mpv, ffmpeg, a C++20 compiler and OpenSSL headers
for the CPU oracle. Logs include compiled GLSL (private weights); keep local.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import re
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import export_mpv_shaders as export

# Not workgroup multiples: exercises all borders and partial tiles.
W, H = 82, 62

LUA = """
local function capture()
    mp.unregister_event(capture)
    local shot, err = mp.command_native({'screenshot-raw', 'window', 'rgba64'})
    if not shot then
        mp.msg.error(err or 'screenshot failed')
        mp.commandv('quit', '2')
        return
    end
    local file = assert(io.open('render.rgba64', 'wb'))
    for y = 0, shot.h - 1 do
        local row = y * shot.stride
        assert(file:write(shot.data:sub(row + 1, row + shot.w * 8)))
    end
    assert(file:close())
    mp.commandv('quit', '0')
end
mp.register_event('playback-restart', capture)
"""


def fixture(work, bits, colored=False):
    name = f"input-{bits}" + ("-color" if colored else "")
    peak = (1 << bits) - 1
    rng = random.Random(728)
    # Ramps, deterministic noise, single-pixel lines, diagonals, sharp
    # contrast changes and samples that don't land on the 8-bit grid.
    def y(x, y):
        if y < 19:
            value = 0.10 + 0.79 * (x + y) / (W + 17)
        elif y < 36:
            value = rng.uniform(0.05, 0.96)
        else:
            value = 0.14 if x + y // 2 < W * 0.75 else 0.85
            if x % 17 == 0:
                value = 1 - value
        return round(value * peak)

    luma = [y(x, j) for j in range(H) for x in range(W)]
    uv = W * H // 4
    cb = [1 << (bits-1)] * uv
    cr = cb[:]
    if colored:
        # A chroma edge on the subsampled grid, plus a horizontal stripe.
        cb = [round(peak * (0.43 if x < 23 else 0.58))
              for j in range(H//2) for x in range(W//2)]
        cr = [round(peak * (0.54 if 11 <= j < 22 else 0.44))
              for j in range(H//2) for x in range(W//2)]

    def pack(values, code):
        return struct.pack(f"<{len(values)}{code}", *values)

    (work / f"{name}.u16").write_bytes(pack(luma, "H"))
    pixfmt = "yuv420p" if bits == 8 else f"yuv420p{bits}le"
    frame = pack(luma + cb + cr, "B" if bits == 8 else "H")
    clip = work / f"{name}.nut"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pixel_format", pixfmt,
                    "-video_size", f"{W}x{H}", "-framerate", "30", "-color_range", "pc", "-i", "-",
                    "-c:v", "copy", str(clip)],
                   input=frame * 3, check=True)
    decoded = subprocess.check_output(
        ["ffmpeg", "-v", "error", "-i", str(clip), "-frames:v", "1", "-c:v", "copy",
         "-f", "rawvideo", "-"])
    assert decoded == frame, "FFmpeg changed the source samples before the test"
    return clip, work / f"{name}.u16"


def play(args, dest, clip, shaders, geometry=(W*2, H*2), *, shot=False, fullscreen=False):
    dest.mkdir()
    cmd = ["mpv", "--no-config", "--vo=gpu-next", "--gpu-api=vulkan",
           f"--gpu-context={args.gpu_context}", "--msg-level=all=warn,vo/gpu-next=trace",
           "--log-file=mpv.log", "--gpu-shader-cache=no", "--untimed", "--no-audio",
           "--no-resume-playback", "--keepaspect=no", "--osd-level=0",
           "--target-prim=bt.709", "--target-trc=srgb", "--target-colorspace-hint=no",
           "--target-contrast=inf", "--tone-mapping=clip", "--hdr-compute-peak=no",
           "--treat-srgb-as-power22=no", "--video-output-levels=full", "--sdr-adjust-gamma=no",
           "--vf=format=colormatrix=bt.709:colorlevels=full:primaries=bt.709:"
           "gamma=srgb:chroma-location=mpeg2/4/h264",
           "--icc-profile-auto=no", "--dither-depth=no", "--scale=bilinear", "--cscale=bilinear",
           "--dscale=bilinear"]
    cmd += ["--fullscreen"] if fullscreen else [f"--geometry={geometry[0]}x{geometry[1]}"]
    cmd += [f"--glsl-shaders-append={p.resolve()}" for p in shaders]
    if shot:
        (dest / "shot.lua").write_text(LUA)
        cmd += ["--pause", "--script=shot.lua"]
    else:
        cmd += ["--frames=1"]
    with (dest / "mpv.txt").open("w") as out:
        subprocess.run(cmd + [str(clip.resolve())], cwd=dest, stdout=out,
                       stderr=subprocess.STDOUT, check=True, timeout=60)
    log = (dest / "mpv.log").read_text()
    errors = re.findall(r"^.*\[(?:e|f)\].*$", log, re.M)
    if errors:
        raise AssertionError(f"{dest}/mpv.log: {errors[0]}")
    if fullscreen:
        # Wayland may present the initial source-sized window before the
        # compositor acknowledges fullscreen. Check only after the resize.
        for m in re.finditer(r"Window size: (\d+)x(\d+) ", log):
            if int(m[1]) > W + 2 or int(m[2]) > H + 2:
                log = log[m.start():]
                break
        else:
            raise AssertionError(f"{dest}/mpv.log: fullscreen size not realized")
    elif f"Window size: {geometry[0]}x{geometry[1]} " not in log:
        raise AssertionError(f"{dest}/mpv.log: requested window size not realized")
    if shot:
        raw = (dest / "render.rgba64").read_bytes()
        if len(raw) != 8 * geometry[0] * geometry[1]:
            raise AssertionError(f"{dest}: unexpected screenshot dimensions")
        pixels = struct.unpack(f"<{len(raw)//2}H", raw)
        return log, tuple(v for i, v in enumerate(pixels) if i % 4 != 3)
    return log, None


def shader_labels(shader):
    labels = re.findall(r"^//!DESC (.*)$", shader, re.M)
    assert re.findall(r"^//!HOOK (.*)$", shader, re.M) == ["LUMA"] * len(labels)
    assert re.findall(r"^//!WHEN (.*)$", shader, re.M) == [export.WHEN] * len(labels)
    assert shader.count("//!COMPUTE ") == len(labels)
    assert 1 <= len(labels) <= 16
    return labels


def check_hooks(log, labels, run):
    pattern = r"Executing hook pass \d+ on stage 'LUMA': (NGU .*?)\n(.*?)(?=Executing hook pass|\Z)"
    sections = re.findall(pattern, log, flags=re.S)
    events = [(label, "Saving output texture" in section, "Skipping hook due to condition" in section)
              for label, section in sections]
    assert set(events) == {(label, run, not run) for label in labels}, events
    success = "shaderc compile status 'success' (0 errors, 0 warnings)"
    compiled = {label for label, section in sections
                if success in section.split("Saving output texture", 1)[0]}
    if run:
        assert compiled == set(labels), f"missing SPIR-V compiles for {set(labels) - compiled}"
    return len(events), len(compiled)


def compile_reference(work):
    header = work / "ngu_catalog.hpp"
    gen = export.REPO / "tools/generate_ngu_catalog.py"
    subprocess.run([sys.executable, str(gen), "--index", str(export.INDEX),
                    "--fp16-index", str(export.HALF_INDEX), "--output", str(header)], check=True)
    reference = work / "mpv_reference"
    src = export.REPO / "src"
    subprocess.run(["c++", "-std=c++20", "-O2", "-ffp-contract=off", "-I", str(src),
                    "-I", str(work), str(export.REPO / "tests/mpv_reference.cpp"),
                    str(src / "ngu_cpu.cpp"), str(src / "ngu_model.cpp"),
                    "-lcrypto", "-o", str(reference)], check=True)
    return reference


def cpu_grid(runner, root, model, precision, bits, source, result):
    subprocess.run([str(runner), str(root), model, precision, str(W), str(H), str(bits),
                    str(source), str(result)], check=True)
    return result.read_bytes()


def reference_shader(grid, offset=False, capture=False):
    # Test-only CPU oracle upload. Production shaders do not use TEXTURE
    # declarations. Render the same presentation path on the same YUV clip.
    sample = "texelFetch(CPU_Y, ivec2(gl_GlobalInvocationID.xy), 0).r"
    if capture:
        sample = f"(0.25 + 0.5 * {sample})"
    return f"""//!TEXTURE CPU_Y
//!SIZE {2*W} {2*H}
//!FORMAT r32f
//!FILTER NEAREST
{grid.hex()}

//!HOOK LUMA
//!DESC CPU oracle
//!BIND CPU_Y
//!WIDTH LUMA.w 2 *
//!HEIGHT LUMA.h 2 *
//!COMPONENTS 1
""" + ("//!OFFSET -0.5 -0.5\n" if offset else "") + f"""//!COMPUTE 8 8
void hook() {{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, imageSize(out_image)))) return;
    imageStore(out_image, p, vec4({sample}, 0.0, 0.0, 1.0));
}}
"""


def capture_shader(aa):
    # Restore coordinate zero only in the diagnostic raw-grid readback.
    # Compress signed excursions into displayable full-range Y, not clamped.
    return """//!HOOK LUMA
//!DESC Read back the raw Y sample grid
//!BIND LUMA
//!COMPONENTS 1
""" + ("//!OFFSET 0.5 0.5\n" if aa else "") + """vec4 hook() {
    return vec4(0.25 + 0.5 * LUMA_tex(LUMA_pos).r, 0.0, 0.0, 1.0);
}
"""


def stats(delta):
    return max(delta), sum(delta) / len(delta)


def luma_error(rgb, grid):
    assert max(abs(r - g) + abs(g - b) for r, g, b in zip(*[iter(rgb)] * 3)) <= 4
    cpu = struct.unpack(f"<{len(grid)//4}f", grid)
    actual = [(u / 65535 - 0.25) * 2 for u in rgb[::3]]
    return stats([255 * abs(a - b) for a, b in zip(actual, cpu, strict=True)])


def color_error(actual, expected):
    # Per-channel comparison so equal-luma chroma shifts cannot disappear.
    return stats([abs(a - b) / 257 for a, b in zip(actual, expected, strict=True)])


def pixels(args, work, models):
    runner = compile_reference(work)
    colored, color_y = fixture(work, 8, colored=True)
    for bits in args.bit_depth:
        clip, yplane = fixture(work, bits)
        for i, (model, shader, labels) in enumerate(models):
            model_id = model["id"]
            base = work / f"{model_id}-{bits}"
            base.mkdir()
            grid = cpu_grid(runner, args.weights_root, model_id, args.precision, bits,
                            yplane, base / "cpu.f32")
            if i == 0:
                # Check the readback is identity (apart from FBO rounding)
                # before using it to recover raw luma samples from gray.
                ref = base / "cpu.glsl"
                ref.write_text(reference_shader(grid, capture=True))
                _, calibration = play(args, base / "cpu", clip, [ref], shot=True)
                error, _ = luma_error(calibration, grid)
                assert error <= 0.27, f"non-identity readback ({error:.5f} units): {base}"
            capture = base / "capture.glsl"
            capture.write_text(capture_shader(model["family_id"] == 28))
            log, actual = play(args, base / "gpu", clip, [shader, capture], shot=True)
            check_hooks(log, labels, True)
            maximum, mae = luma_error(actual, grid)
            print(f"Y {model_id} {bits:2d}-bit max={maximum:.5f} MAE={mae:.5f}", flush=True)
            native_half = "#define NGU_NATIVE_FP16 1" in shader.read_text()
            limit = 2.5 if native_half else 1.0
            assert maximum <= limit, f"{model_id}/{bits}: error exceeds {limit}/255 ({base})"
            if native_half:
                assert mae <= 0.25, f"{model_id}/{bits}: native FP16 MAE exceeds 0.25/255 ({base})"
    if any(m[0]["family_id"] == 28 for m in models):
        model, shader, labels = next(m for m in models if m[0]["family_id"] == 28)
        base = work / "aa-alignment"
        base.mkdir()
        grid = cpu_grid(runner, args.weights_root, model["id"], args.precision, 8,
                        color_y, base / "cpu.f32")
        correct, wrong = base / "aligned.glsl", base / "unaligned.glsl"
        correct.write_text(reference_shader(grid, offset=True))
        wrong.write_text(reference_shader(grid))
        log, actual = play(args, base / "gpu", colored, [shader], shot=True)
        check_hooks(log, labels, True)
        _, expected = play(args, base / "cpu", colored, [correct], shot=True)
        _, unaligned = play(args, base / "negative-control", colored, [wrong], shot=True)
        assert "Aligning plane 0: {-0.500000 -0.500000 " in log
        # left-sited 4:2:0: shifted by a quarter of the luma offset in
        # chroma texels when the renderer rounds LUMA's reference rect.
        assert "Aligning plane 1: {0.250000 0.000000 41.250000 31.000000} -> " \
               "{0.125000 -0.125000 41.125000 30.875000}" in log
        amax, amae = color_error(actual, expected)
        bmax, bmae = color_error(actual, unaligned)
        print(f"AA phase (rendered color): correct max={amax:.5f} MAE={amae:.5f}; "
              f"zero-offset control max={bmax:.5f} MAE={bmae:.5f}", flush=True)
        alimit = 2.5 if "#define NGU_NATIVE_FP16 1" in shader.read_text() else 1.0
        assert amax <= alimit and bmax > 1.0 and bmae > 4 * amae


def suite(args, work):
    work = work.resolve()
    models = []
    all_models = export.selected_models(export.INDEX)
    if args.model and set(args.model) - {m["id"] for m in all_models}:
        raise ValueError("unknown --model ID")
    for model in all_models:
        if args.model and model["id"] not in args.model:
            continue
        shader = args.shaders / export.model_path(model)
        labels = shader_labels(shader.read_text())
        models.append((model, shader, labels))
    if args.suite in ("all", "smoke"):
        clip, _ = fixture(work, 8)
        for m, shader, labels in models:
            log, _ = play(args, work / f"smoke-{m['id']}", clip, [shader])
            events, compiles = check_hooks(log, labels, True)
            print(f"smoke {m['id']}: {len(labels)} passes, {events} dispatches, "
                  f"{compiles} successful NGU compiles", flush=True)
        _, shader, labels = models[0]
        for case, size, active in (("1to1", (W, H), False),
                                   ("down", (W-4, H-4), False),
                                   ("within-tolerance", (W+2, H+2), False),
                                   ("wider", (W+4, H), True),
                                   ("taller", (W, H+4), True),
                                   ("fullscreen", None, True)):
            log, _ = play(args, work / case, clip, [shader], size, fullscreen=size is None)
            n, _ = check_hooks(log, labels, active)
            print(f"{case}: {n if active else 0} dispatched, {0 if active else n} skipped", flush=True)
    if args.suite in ("all", "pixels"):
        pixels(args, work, models)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shaders", type=Path, default=Path("~/.config/mpv/shaders/NGU"))
    p.add_argument("--weights-root", type=Path, default=export.REPO / "weights")
    p.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    p.add_argument("--gpu-context", default="x11vk")
    p.add_argument("--suite", choices=("all", "smoke", "pixels"), default="all")
    p.add_argument("--model", action="append", help="test only these index IDs (repeatable)")
    p.add_argument("--bit-depth", type=int, choices=(8, 10, 12), action="append")
    p.add_argument("--artifacts", type=Path, help="new directory to retain private logs and captures")
    args = p.parse_args()
    args.shaders = args.shaders.expanduser().resolve()
    args.weights_root = args.weights_root.expanduser().resolve()
    args.bit_depth = args.bit_depth or [8, 10]
    if args.artifacts:
        args.artifacts.mkdir(mode=0o700)
        suite(args, args.artifacts)
    else:
        with tempfile.TemporaryDirectory(prefix="vsngu-mpv-") as tmp:
            suite(args, Path(tmp))


if __name__ == "__main__":
    main()
