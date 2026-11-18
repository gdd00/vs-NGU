#!/usr/bin/env python3
"""Export private, self-contained mpv gpu-next NGU 2x LUMA shaders."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile

from generate_ngu_catalog import FAMILIES, INDEX_SHA256, QUALITIES, selected_models

REPO = Path(__file__).resolve().parents[1]
INDEX = REPO / "models/madvr09217-index.json"
HALF_INDEX = REPO / "models/madvr09217-fp16-index.json"
WHEN = "OUTPUT.w LUMA.w 2 + > OUTPUT.h LUMA.h 2 + > +"
# One workgroup processes 8x4 source positions (32 invocations).
BX, BY = 8, 4

# The extension macro advertises compiler support, NOT Vulkan shaderFloat16.
# Keep native half tiles opt-in; all accumulators always remain FP32.
HALF_DECL = """#if defined(NGU_NATIVE_FP16) && defined(GL_EXT_shader_explicit_arithmetic_types_float16)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#define TILE4 f16vec4
#else
#define TILE4 vec4
#endif
"""


def model_path(model: dict) -> Path:
    family = FAMILIES[model["family_id"]]
    quality = QUALITIES[model["quality"]]
    option = model["selector_option"]
    if family == "sharp" and option:
        return Path(f"sharp/options/{quality}_option{option:02d}.glsl")
    return Path(f"{family}/{quality}.glsl")


def half_catalog(path: Path) -> dict:
    index = json.loads(path.read_text(encoding="utf-8"))
    if (index["schema_version"] != 1 or index["dtype"] != "ieee754-binary16-le"
            or index["source_index_sha256"] != INDEX_SHA256
            or index["rounding"] != "nearest-even"
            or index["bias_transform"] != "fp32(raw/255) then fp16-rne"):
        raise ValueError("invalid FP16 metadata contract")
    return {m["id"]: m["stages"] for m in index["models"]}


def parameter(path: Path, metadata: dict, count: int, fp16: bool) -> tuple[float, ...]:
    code, stride = ("e", 2) if fp16 else ("f", 4)
    size = count * stride
    if metadata["size"] != size:
        raise ValueError(f"parameter extent in index does not match graph: {path}")
    with path.open("rb") as source:
        data = source.read(size + 1)
    if len(data) != size:
        raise ValueError(f"parameter size mismatch: {path}")
    if hashlib.sha256(data).hexdigest() != metadata["sha256"]:
        raise ValueError(f"parameter SHA-256 mismatch: {path}")
    values = struct.unpack(f"<{count}{code}", data)
    if not all(map(math.isfinite, values)):
        raise ValueError(f"non-finite parameter: {path}")
    return values


def load_layers(model: dict, root: Path, half: dict | None = None) -> list:
    fp16 = half is not None
    model_id = model["id"]
    archive = root / ("weights-fp16" if fp16 else "weights")
    stages = model["stages"]
    aa = model["family_id"] == 28
    num_body = len(stages) - (3 if aa else 1)
    if not 1 <= num_body <= 8 or (fp16 and len(half[model_id]) != len(stages)):
        raise ValueError(f"invalid native-2x graph: {model_id}")
    layers = []
    previous = 1
    for i, stage in enumerate(stages):
        ci, co, k = (stage[key] for key in ("in_channels", "out_channels", "kernel_size"))
        if (stage["index"] != i or ci != previous
                or (i < num_body and (k != 3 or co not in (8, 12, 16)))
                or (i >= num_body and (k != (3 if aa else 4) or co != 1))):
            raise ValueError(f"invalid native-2x stage: {model_id}/{i}")
        if i < num_body:
            previous = co
        if fp16:
            hs = half[model_id][i]
            if any(hs[key] != stage[key] for key in
                   ("index", "in_channels", "out_channels", "kernel_size")):
                raise ValueError(f"FP16 model shape mismatch: {model_id}/{i}")
        values = []
        for kind, n in (("weights", ci * co * k * k), ("bias", co)):
            name = f"{model_id}/stage{i:02d}.{kind}.f{'16' if fp16 else '32'}le"
            metadata = stage[kind]
            if fp16:
                entry = hs[kind]
                if entry["path"] != name or entry["source_sha256"] != metadata["sha256"]:
                    raise ValueError(f"FP16 metadata mismatch: {name}")
                metadata = entry
            values.append(parameter(archive / name, metadata, n, fp16))
        w, b = values
        if not fp16:
            b = tuple(struct.unpack("<f", struct.pack("<f", v / 255.0))[0] for v in b)
        layers.append((ci, co, w, b))
    return layers


def vec(values) -> str:
    # 9 significant digits round-trip a binary32. Force GLSL float literals.
    numbers = [format(v, ".9g") for v in values]
    return "vec4(" + ", ".join(v if "." in v or "e" in v else v + ".0" for v in numbers) + ")"


def header(label, inputs, save=None, size="2 *", comps=4, offset=None):
    lines = ["//!HOOK LUMA", f"//!DESC {label}"]
    lines.extend(f"//!BIND {name}" for name in inputs)
    if save:
        lines.append(f"//!SAVE {save}")
    lines.extend([f"//!WIDTH LUMA.w {size}".rstrip(), f"//!HEIGHT LUMA.h {size}".rstrip(),
                  f"//!COMPONENTS {comps}", f"//!WHEN {WHEN}"])
    if offset:
        lines.append(f"//!OFFSET {offset}")
    outscale = 2 if size == "2 *" else 1
    lines.append(f"//!COMPUTE {BX*outscale} {BY*outscale} {BX} {BY}")
    return "\n".join(lines) + "\n"


def fetch(name):
    # Exact point sampling with clamp-to-edge; equivalent to name_tex at texel
    # centers, explicitly multiplying by the normalization factor exactly once.
    return f"""vec4 read_{name}(ivec2 p) {{
    return texelFetch({name}_raw, clamp(p, ivec2(0), ivec2({name}_size) - 1), 0) * {name}_mul;
}}
"""


def tile(source, groups, lo, hi, native_fp16, packed_input=False):
    """Cooperatively stage each input texel once, including the halo."""
    tw, th = BX + hi - lo, BY + hi - lo
    glsl = ("#define NGU_NATIVE_FP16 1\n" if native_fp16 else "") + HALF_DECL if packed_input else ""
    tile_type = "TILE4" if packed_input else "vec4"
    coord = (f"2 * clamp(p, ivec2(0), ivec2({source}_size) / 2 - 1) + ivec2(g & 1, g >> 1)"
             if packed_input else "p")
    return glsl + fetch(source) + f"""
shared {tile_type} samples[{tw * th * groups}];
void fill_tile() {{
    ivec2 start = ivec2(gl_WorkGroupID.xy) * ivec2({BX}, {BY});
    for (int i = int(gl_LocalInvocationIndex); i < {tw * th * groups}; i += {BX * BY}) {{
        int pos = i / {groups};
        int g = i % {groups};
        ivec2 p = start + ivec2(pos % {tw}, pos / {tw}) + ivec2({lo});
        samples[i] = {tile_type}(read_{source}({coord}));
    }}
    barrier();
}}
vec4 sample_tile(ivec2 delta, int g) {{
    ivec2 p = ivec2(gl_LocalInvocationID.xy) + delta - ivec2({lo});
    return vec4(samples[(p.y * {tw} + p.x) * {groups} + g]);
}}
"""


def spatial_point():
    return f"""    fill_tile();
    ivec2 p = ivec2(gl_WorkGroupID.xy) * ivec2({BX}, {BY}) + ivec2(gl_LocalInvocationID.xy);
"""


def correlate(label, source, save, ci, co, w, b, body, native_fp16):
    # 2x2 spatial packing: group g of four channels is in texel
    # 2*p + (g%2, g/2). 8/12-channel layers leave unused groups unwritten,
    # and consumers never address them.
    groups = (ci + 3) // 4
    outputs = co // 4
    code = [header(label, [source, "LUMA"] if source != "LUMA" else ["LUMA"],
                   save, size="2 *" if body else ""),
            tile(source, groups, -1, 1, native_fp16, packed_input=ci != 1),
            "void hook() {\n", spatial_point(),
            "    if (any(greaterThanEqual(p, ivec2(LUMA_size)))) return;\n"]
    for g in range(outputs):
        code.append(f"    precise vec4 s{g} = {vec(b[4*g:4*g+4])};\n")
    for tap in range(9):
        for ig in range(groups):
            code.append(f"    vec4 v{tap}_{ig} = sample_tile(ivec2({tap%3-1}, {tap//3-1}), {ig});\n")
            for lane, ic in enumerate(range(4 * ig, min(ci, 4 * ig + 4))):
                for g in range(outputs):
                    # Emit immediate constants (no dynamically indexed constant
                    # arrays, which some drivers materialize per invocation).
                    weights = vec(w[(oc * ci + ic) * 9 + tap] for oc in range(4*g, 4*g+4))
                    code.append(f"    s{g} += v{tap}_{ig}[{lane}] * {weights};\n")
    for g in range(outputs):
        if body:
            code.append(f"    imageStore(out_image, 2 * p + ivec2({g%2}, {g//2}), "
                        f"mix(s{g}, 0.1 * s{g}, lessThan(s{g}, vec4(0.0))));\n")
        else:
            code.append(f"    imageStore(out_image, p, s{g});\n")
    return "".join(code) + "}\n"


def residual(label, source, layer, native_fp16):
    ci, _, w, b = layer
    code = [header(label, [source, "LUMA"], "NGU_HEAD", size="2 +"),
            tile(source, ci // 4, -1, 0, native_fp16, packed_input=True),
            "void hook() {\n", spatial_point(),
            "    if (any(greaterThanEqual(p, imageSize(out_image)))) return;\n",
            f"    precise vec4 s = {vec(b * 4)};\n"]
    for ig in range(ci // 4):
        decl = "vec4 " if ig == 0 else ""
        for dy in range(2):
            for dx in range(2):
                code.append(f"    {decl}v{dx}{dy} = sample_tile(ivec2({-dx}, {-dy}), {ig});\n")
        for ic in range(ig * 4, ig * 4 + 4):
            for dy in range(2):
                for dx in range(2):
                    weights = vec(w[ic * 16 + (lane//2 + dy*2)*4 + lane%2 + dx*2]
                                  for lane in range(4))
                    code.append(f"    s += v{dx}{dy}[{ic%4}] * {weights};\n")
    return "".join(code) + "    imageStore(out_image, p, s);\n}\n"


BASELINE = r"""
// Horizontal first, then vertical. These are the binary32 coefficients
// from baseline_pass(), not a generic bicubic interpolator.
vec4 taps(int parity) {
    float a = uintBitsToFloat(0x3f5f3333u);
    float b = uintBitsToFloat(0x3e766666u);
    float c = uintBitsToFloat(0xbdaccccdu);
    float d = uintBitsToFloat(0xbce66666u);
    return parity == 0 ? vec4(d, b, a, c) : vec4(c, a, b, d);
}
float combine(ivec2 phase, ivec2 p) {
    vec4 cx = taps(phase.x), cy = taps(phase.y);
    precise float base = 0.0;
    for (int y = 0; y < 4; ++y) {
        precise float bx = 0.0;
        for (int x = 0; x < 4; ++x)
            bx += cx[x] * sample_tile(phase - ivec2(2) + ivec2(x, y), 0).r;
        base += cy[y] * bx;
    }
    return base + read_NGU_HEAD(p + phase)[3 - phase.x - 2 * phase.y];
}
"""

AA = r"""
float combine(ivec2 phase, ivec2 p) {
    float v = sample_tile(ivec2(0), 0).r;
    if (phase == ivec2(0)) return v;
    vec3 h = read_NGU_HEAD(p).xyz;
    if (phase.y == 0) return (v + sample_tile(ivec2(1, 0), 0).r) * 0.5 + h.y;
    if (phase.x == 0) return (v + sample_tile(ivec2(0, 1), 0).r) * 0.5 + h.x;
    precise float avg = ((v + sample_tile(ivec2(1, 0), 0).r)
                            + sample_tile(ivec2(0, 1), 0).r)
                            + sample_tile(ivec2(1, 1), 0).r;
    return avg * 0.25 + h.z;
}
"""


def render(model_id, layers, aa, native_fp16=False):
    nbody = len(layers) - (3 if aa else 1)
    passes = []
    source = "LUMA"
    label = "NGU " + model_id
    for i, (ci, co, w, b) in enumerate(layers[:nbody]):
        save = f"NGU_F{i}"
        passes.append(correlate(f"{label} body {i + 1}", source, save, ci, co, w, b,
                                True, native_fp16 and i > 0))
        source = save
    if aa:
        heads = layers[nbody:]
        # make_aa_head swaps the first two kernels but not the biases!
        ci = heads[0][0]
        w = heads[1][2] + heads[0][2] + heads[2][2] + (0.0,) * (ci * 9)
        b = tuple(h[3][0] for h in heads) + (0.0,)
        passes.append(correlate(f"{label} AA heads", source, "NGU_HEAD", ci, 4,
                                w, b, False, native_fp16))
    else:
        passes.append(residual(f"{label} residual", source, layers[-1], native_fp16))

    # AA texel (0,0) is at the first input sample center. The texture's
    # sampling rectangle therefore starts half an output texel *earlier*.
    # Do not interpolate AA here: let gpu-next align the planes and scale once.
    final = [header(f"{label} output", ["LUMA", "NGU_HEAD"], comps=1,
                    offset="-0.5 -0.5" if aa else None),
             tile("LUMA", 1, 0 if aa else -2, 1 if aa else 2, False), fetch("NGU_HEAD"),
             AA if aa else BASELINE, "void hook() {\n", spatial_point(),
             """    if (any(greaterThanEqual(p, ivec2(LUMA_size)))) return;
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) {
            ivec2 phase = ivec2(x, y);
            imageStore(out_image, 2 * p + phase, vec4(combine(phase, p), 0.0, 0.0, 1.0));
        }
}
"""]
    return ("// Generated from user-local weights. Do not redistribute.\n"
            "// Load one NGU shader at a time.\n\n"
            + "\n".join(passes + ["".join(final)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, default=REPO / "weights",
                        help="common root of weights/ and weights-fp16/ (default: repo/weights)")
    parser.add_argument("--output", type=Path, default=Path("~/.config/mpv/shaders/NGU"),
                        help="output directory (default: ~/.config/mpv/shaders/NGU)")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32",
                        help="weight archive; FP32 arithmetic with either archive")
    parser.add_argument("--model", action="append", metavar="ID",
                        help="export only this index ID (repeatable; default: all 72)")
    parser.add_argument("--native-fp16", action="store_true",
                        help="opt in to native half shared tiles (requires GPU shaderFloat16)")
    args = parser.parse_args()
    try:
        models = selected_models(INDEX)
        if args.model:
            missing = set(args.model) - {m["id"] for m in models}
            if missing:
                raise ValueError("unknown native-2x model: " + ", ".join(sorted(missing)))
            models = [m for m in models if m["id"] in args.model]
        half = half_catalog(HALF_INDEX) if args.precision == "fp16" else None
        # Verify the complete requested set before changing any installed files.
        root = args.weights_root.expanduser()
        prepared = [(m, load_layers(m, root, half)) for m in models]
        target = args.output.expanduser()
        for model, layers in prepared:
            dest = target / model_path(model)
            dest.parent.mkdir(parents=True, exist_ok=True)
            glsl = render(model["id"], layers, model["family_id"] == 28, args.native_fp16)
            # Never expose a partially written shader. Keep the user's weights
            # private even when their umask permits group/world reads.
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=dest.parent,
                                             delete=False) as file:
                temp = Path(file.name)
                try:
                    file.write(glsl)
                    file.close()
                    temp.replace(dest)
                finally:
                    temp.unlink(missing_ok=True)
        print(f"Exported {len(prepared)} {args.precision.upper()} NGU shaders to {target}")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
