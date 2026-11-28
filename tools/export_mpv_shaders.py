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
MAX_PORTABLE_SMEM_BYTES = 32768

# The extension macro advertises compiler support, NOT Vulkan shaderFloat16.
# Keep native half tiles and convolution accumulators opt-in; luma baseline
# interpolation and final combine always remain FP32.
TYPE_DECL = """#if defined(NGU_NATIVE_FP16) && defined(GL_EXT_shader_explicit_arithmetic_types_float16)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#define ACC float16_t
#define ACC2 f16vec2
#define ACC3 f16vec3
#define ACC4 f16vec4
#else
#define ACC float
#define ACC2 vec2
#define ACC3 vec3
#define ACC4 vec4
#endif
"""

# Horizontal first, then vertical. Exact binary32 coefficients from baseline_pass().
BASELINE_HELPERS = """
vec4 ngu_taps(int parity) {
    float a = uintBitsToFloat(0x3f5f3333u);
    float b = uintBitsToFloat(0x3e766666u);
    float c = uintBitsToFloat(0xbdaccccdu);
    float d = uintBitsToFloat(0xbce66666u);
    return parity == 0 ? vec4(d, b, a, c) : vec4(c, a, b, d);
}
"""

# For each 3x3 input tap (sy, sx) of the 4x4 transpose residual head, list the
# non-zero (subpixel_index, weight_offset) pairs in (tl=0, tr=1, bl=2, br=3).
_TRANSPOSE_TAP_MAP: dict[tuple[int, int], list[tuple[int, int]]] = {}
for _sy in range(3):
    for _sx in range(3):
        _entries = []
        for _sub, (_py_p, _px_p) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
            _dy = 1 + _py_p - _sy
            _dx = 1 + _px_p - _sx
            if 0 <= _dy <= 1 and 0 <= _dx <= 1:
                _wi_rel = (1 - _py_p + _dy * 2) * 4 + (1 - _px_p + _dx * 2)
                _entries.append((_sub, _wi_rel))
        _TRANSPOSE_TAP_MAP[(_sy, _sx)] = _entries


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


def f_lit(v: float) -> str:
    # 9 significant digits round-trip binary32; force GLSL float literal syntax.
    s = format(v, ".9g")
    if "." not in s and "e" not in s:
        s += ".0"
    return s


def acc3(values) -> str:
    return "ACC3(" + ", ".join(f_lit(v) for v in values) + ")"


def acc4(values) -> str:
    return "ACC4(" + ", ".join(f_lit(v) for v in values) + ")"


def tex_dims(co: int) -> tuple[str, str, int, int]:
    """Planar RGBA sub-plane layout for 8-ch (2x1), 12-ch (3x1), and 16-ch (2x2)."""
    g = co // 4
    if g == 2:
        return "2 *", "", 2, 1
    if g == 3:
        return "3 *", "", 3, 1
    return "2 *", "2 *", 2, 2


def emit_header(label: str, binds: list[str], save: str | None,
                w_expr: str, h_expr: str, comps: int,
                bw: int, bh: int, tw: int, th: int,
                offset: str | None = None) -> str:
    lines = ["//!HOOK LUMA", f"//!DESC {label}"]
    for b in binds:
        lines.append(f"//!BIND {b}")
    if save:
        lines.append(f"//!SAVE {save}")
    lines.append(f"//!WIDTH LUMA.w {w_expr}".rstrip())
    lines.append(f"//!HEIGHT LUMA.h {h_expr}".rstrip())
    lines.append(f"//!COMPONENTS {comps}")
    lines.append(f"//!WHEN {WHEN}")
    if offset:
        lines.append(f"//!OFFSET {offset}")
    lines.append(f"//!COMPUTE {bw} {bh} {tw} {th}")
    return "\n".join(lines) + "\n"


def emit_clamp_bounds(cur_w: int, cur_h: int, halo_x: int, halo_y: int, pfx: str, ind: str) -> str:
    return (
        f"{ind}int {pfx}x_lo = max(0, {halo_x} - wg_base.x);\n"
        f"{ind}int {pfx}x_hi = min({cur_w - 1}, W - 1 + {halo_x} - wg_base.x);\n"
        f"{ind}int {pfx}y_lo = max(0, {halo_y} - wg_base.y);\n"
        f"{ind}int {pfx}y_hi = min({cur_h - 1}, H - 1 + {halo_y} - wg_base.y);\n"
    )


def emit_store_tex(outs: int, dst_x: str, dst_y: str, r: int, ind: str) -> str:
    """Coalesced unit-stride planar sub-plane imageStore for each 4-ch group."""
    lines = []
    for g in range(outs):
        val = f"vec4(a{r}_{g})"
        if outs in (2, 3):
            x_expr = dst_x if g == 0 else f"({dst_x}) + {g} * W"
            coord = f"ivec2({x_expr}, {dst_y})"
        else:
            x_expr = dst_x if (g & 1) == 0 else f"({dst_x}) + W"
            y_expr = dst_y if (g >> 1) == 0 else f"({dst_y}) + H"
            coord = f"ivec2({x_expr}, {y_expr})"
        lines.append(f"{ind}imageStore(out_image, {coord}, {val});\n")
    return "".join(lines)


def emit_load_planar_mlp(source: str, gin: int, in_w: int, in_h: int,
                         in_halo: int, nthreads: int, ind: str = "    ") -> str:
    """Load all gin planar sub-planes per pixel in one burst (gin x fewer div/mod/clamp)."""
    in_plane = in_w * in_h
    lines = [
        f"{ind}for (int i = tid; i < {in_plane}; i += {nthreads}) {{\n",
        f"{ind}    int ix = i % {in_w}, iy = i / {in_w};\n",
        f"{ind}    ivec2 sp = clamp(wg_base + ivec2(ix - {in_halo}, iy - {in_halo}), ivec2(0), ivec2(W - 1, H - 1));\n",
    ]
    for g in range(gin):
        if gin in (2, 3):
            cx = "sp.x" if g == 0 else f"sp.x + {g} * W"
            coord = f"ivec2({cx}, sp.y)"
        else:
            cx = "sp.x" if (g & 1) == 0 else "sp.x + W"
            cy = "sp.y" if (g >> 1) == 0 else "sp.y + H"
            coord = f"ivec2({cx}, {cy})"
        off = "" if g == 0 else f"{g * in_plane} + "
        lines.append(f"{ind}    s_feat[{off}i] = ACC4(texelFetch({source}_raw, {coord}, 0) * {source}_mul);\n")
    lines.append(f"{ind}}}\n")
    return "".join(lines)


def emit_load_planar_flat(source: str, gin: int, in_w: int, in_h: int,
                          in_halo: int, nthreads: int, ind: str = "    ") -> str:
    """Balanced flat planar sub-plane loader when in_plane is near nthreads."""
    in_plane = in_w * in_h
    total = in_plane * gin
    if gin in (2, 3):
        ce = "ivec2(sp.x + g * W, sp.y)"
    else:
        ce = "ivec2(sp.x + (g & 1) * W, sp.y + (g >> 1) * H)"
    return (
        f"{ind}for (int i = tid; i < {total}; i += {nthreads}) {{\n"
        f"{ind}    int g = i / {in_plane}, rem = i % {in_plane};\n"
        f"{ind}    int ix = rem % {in_w}, iy = rem / {in_w};\n"
        f"{ind}    ivec2 sp = clamp(wg_base + ivec2(ix - {in_halo}, iy - {in_halo}), ivec2(0), ivec2(W - 1, H - 1));\n"
        f"{ind}    s_feat[i] = ACC4(texelFetch({source}_raw, {ce}, 0) * {source}_mul);\n"
        f"{ind}}}\n"
    )


def emit_conv_1ch_rows(w, b, co: int, luma_w: int, x_var: str, y0_var: str,
                       py: int, has_y_guard: bool, pfx: str, ind: str = "        ") -> str:
    outs = co // 4
    lines = []
    for dx in range(3):
        lines.append(f"{ind}int cx{dx} = clamp({x_var} + {dx}, {pfx}x_lo, {pfx}x_hi);\n")
    for dy in range(py + 2):
        lines.append(f"{ind}int cy{dy} = clamp({y0_var} + {dy}, {pfx}y_lo, {pfx}y_hi) * {luma_w};\n")
    for r in range(py):
        for g in range(outs):
            lines.append(f"{ind}ACC4 a{r}_{g} = {acc4(b[4 * g:4 * g + 4])};\n")
    for ky in range(py + 2):
        guarded = has_y_guard and (ky == py + 1)
        if guarded:
            lines.append(f"{ind}if (has_y1) {{\n")
        inner = ind + "    " if guarded else ind
        for kx in range(3):
            lines.append(f"{inner}ACC s_{ky}_{kx} = ACC(s_luma[cy{ky} + cx{kx}]);\n")
            for r in range(py):
                if r <= ky <= r + 2:
                    tap = (ky - r) * 3 + kx
                    for g in range(outs):
                        wv = acc4(w[(oc * 1 + 0) * 9 + tap] for oc in range(4 * g, 4 * g + 4))
                        lines.append(f"{inner}a{r}_{g} = fma(ACC4(s_{ky}_{kx}), {wv}, a{r}_{g});\n")
        if guarded:
            lines.append(f"{ind}}}\n")
    for r in range(py):
        for g in range(outs):
            lines.append(f"{ind}a{r}_{g} = max(a{r}_{g}, a{r}_{g} * ACC(0.1));\n")
    return "".join(lines)


def emit_conv_mch_rows(w, b, ci: int, co: int, in_w: int, in_h: int,
                       x_var: str, y0_var: str, py: int, has_y_guard: bool,
                       pfx: str, relu: bool = True, ind: str = "        ") -> str:
    ins = ci // 4
    outs = co // 4
    in_plane = in_w * in_h
    lines = []
    for dx in range(3):
        lines.append(f"{ind}int cx{dx} = clamp({x_var} + {dx}, {pfx}x_lo, {pfx}x_hi);\n")
    for dy in range(py + 2):
        lines.append(f"{ind}int cy{dy} = clamp({y0_var} + {dy}, {pfx}y_lo, {pfx}y_hi) * {in_w};\n")
    for r in range(py):
        for g in range(outs):
            lines.append(f"{ind}ACC4 a{r}_{g} = {acc4(b[4 * g:4 * g + 4])};\n")
    lines.append(f"{ind}ACC4 v;\n")
    for ig in range(ins):
        base_off = ig * in_plane
        for kx in range(3):
            for ky in range(py + 2):
                guarded = has_y_guard and (ky == py + 1)
                if guarded:
                    lines.append(f"{ind}if (has_y1) {{\n")
                inner = ind + "    " if guarded else ind
                lines.append(f"{inner}v = s_feat[{base_off} + cy{ky} + cx{kx}];\n")
                for lane, swz in enumerate(("xxxx", "yyyy", "zzzz", "wwww")):
                    ic = ig * 4 + lane
                    for r in range(py):
                        if r <= ky <= r + 2:
                            tap = (ky - r) * 3 + kx
                            for g in range(outs):
                                wv = acc4(w[(oc * ci + ic) * 9 + tap] for oc in range(4 * g, 4 * g + 4))
                                lines.append(f"{inner}a{r}_{g} = fma(v.{swz}, {wv}, a{r}_{g});\n")
                if guarded:
                    lines.append(f"{ind}}}\n")
    if relu:
        for r in range(py):
            for g in range(outs):
                lines.append(f"{ind}a{r}_{g} = max(a{r}_{g}, a{r}_{g} * ACC(0.1));\n")
    return "".join(lines)


def emit_transpose_head_rows(w_res, b_res, ci: int, in_plane: int, py: int,
                             hcx_pfx: str, hcy_pfx: str, ind: str) -> str:
    """Single-v non-zero-only 4x4 transpose conv across py rows (zero gather shuffles)."""
    ins = ci // 4
    swz4 = ("xxxx", "yyyy", "zzzz", "wwww")
    comp = ("x", "y", "z", "w")
    lines = [f"{ind}ACC4 res{r} = {acc4(b_res * 4)};\n" for r in range(py)]
    lines.append(f"{ind}{{\n{ind}    ACC4 v;\n")
    for ig in range(ins):
        boff = ig * in_plane
        for kx in range(3):
            for ky in range(py + 2):
                lines.append(f"{ind}    v = s_feat[{boff} + {hcy_pfx}{ky} + {hcx_pfx}{kx}];\n")
                for r in range(py):
                    sy = ky - r
                    if 0 <= sy <= 2:
                        entries = _TRANSPOSE_TAP_MAP[(sy, kx)]
                        if len(entries) == 4:
                            for lane in range(4):
                                ic = ig * 4 + lane
                                wv = acc4(w_res[ic * 16 + wi] for _, wi in entries)
                                lines.append(f"{ind}    res{r} = fma(v.{swz4[lane]}, {wv}, res{r});\n")
                        elif len(entries) == 2:
                            (s0, w0r), (s1, w1r) = entries
                            dst_swz = comp[s0] + comp[s1]
                            for lane in range(4):
                                ic = ig * 4 + lane
                                vsw = comp[lane] * 2
                                wa = f_lit(w_res[ic * 16 + w0r])
                                wb = f_lit(w_res[ic * 16 + w1r])
                                lines.append(f"{ind}    res{r}.{dst_swz} = fma(v.{vsw}, ACC2({wa}, {wb}), res{r}.{dst_swz});\n")
                        else:
                            s0, w0r = entries[0]
                            dc = comp[s0]
                            for lane in range(4):
                                ic = ig * 4 + lane
                                wa = f_lit(w_res[ic * 16 + w0r])
                                lines.append(f"{ind}    res{r}.{dc} = fma(v.{comp[lane]}, ACC({wa}), res{r}.{dc});\n")
    lines.append(f"{ind}}}\n")
    return "".join(lines)


def emit_aa_head_rows(heads: list, in_plane: int, py: int,
                      hcx_pfx: str, hcy_pfx: str, ind: str) -> str:
    """3-channel ACC3 AA head across py rows (eliminates wasted 4th zero channel)."""
    ci = heads[0][0]
    ins = ci // 4
    # make_aa_head swaps the first two kernels but not the biases!
    w_aa = heads[1][2] + heads[0][2] + heads[2][2]
    b_aa = (heads[0][3][0], heads[1][3][0], heads[2][3][0])
    swz3 = ("xxx", "yyy", "zzz", "www")
    lines = [f"{ind}ACC3 h{r} = {acc3(b_aa)};\n" for r in range(py)]
    lines.append(f"{ind}{{\n{ind}    ACC4 v;\n")
    for ig in range(ins):
        boff = ig * in_plane
        for kx in range(3):
            for ky in range(py + 2):
                lines.append(f"{ind}    v = s_feat[{boff} + {hcy_pfx}{ky} + {hcx_pfx}{kx}];\n")
                for lane in range(4):
                    ic = ig * 4 + lane
                    for r in range(py):
                        if r <= ky <= r + 2:
                            tap = (ky - r) * 3 + kx
                            wv = acc3(w_aa[(oc * ci + ic) * 9 + tap] for oc in range(3))
                            lines.append(f"{ind}    h{r} = fma(v.{swz3[lane]}, {wv}, h{r});\n")
    lines.append(f"{ind}}}\n")
    return "".join(lines)


def emit_shbase_populate(tw_tile: int, th_tile: int, luma_w: int, luma_halo: int,
                         nthreads: int, ind: str = "        ") -> str:
    """Cooperatively compute FP32 horizontal 4-tap baseline once per tile row."""
    hbase_w = tw_tile
    hbase_h = th_tile + 4
    return (
        f"{ind}vec4 hcx0 = ngu_taps(0), hcx1 = ngu_taps(1);\n"
        f"{ind}for (int i = tid; i < {hbase_w * hbase_h}; i += {nthreads}) {{\n"
        f"{ind}    int bx = i % {hbase_w}, by = i / {hbase_w};\n"
        f"{ind}    int bly = clamp(by + {luma_halo - 2}, lm_y_lo, lm_y_hi) * {luma_w};\n"
        f"{ind}    float v0 = s_luma[bly + clamp(bx + {luma_halo - 2}, lm_x_lo, lm_x_hi)];\n"
        f"{ind}    float v1 = s_luma[bly + clamp(bx + {luma_halo - 1}, lm_x_lo, lm_x_hi)];\n"
        f"{ind}    float v2 = s_luma[bly + clamp(bx + {luma_halo}, lm_x_lo, lm_x_hi)];\n"
        f"{ind}    float v3 = s_luma[bly + clamp(bx + {luma_halo + 1}, lm_x_lo, lm_x_hi)];\n"
        f"{ind}    float v4 = s_luma[bly + clamp(bx + {luma_halo + 2}, lm_x_lo, lm_x_hi)];\n"
        f"{ind}    s_hbase[i] = vec2(dot(vec4(v0, v1, v2, v3), hcx0), dot(vec4(v1, v2, v3, v4), hcx1));\n"
        f"{ind}}}\n"
        f"{ind}barrier();\n"
    )


def emit_sharp_combine_rows(luma_w: int, luma_halo: int, tw_tile: int, th_tile: int,
                            py: int, ox_var: str, oy0_var: str, check_th: bool,
                            use_shbase: bool, ind: str) -> str:
    lines = []
    if use_shbase:
        lines.append(f"{ind}vec4 cy0 = ngu_taps(0), cy1 = ngu_taps(1);\n")
        for r in range(py):
            y_expr = oy0_var if r == 0 else f"{oy0_var} + {r}"
            cond = f"{y_expr} < {th_tile} && wg_base.y + {y_expr} < H" if check_th else f"wg_base.y + {y_expr} < H"
            lines.append(f"{ind}if ({cond}) {{\n")
            for k in range(5):
                lines.append(f"{ind}    vec2 b{k} = s_hbase[({y_expr} + {k}) * {tw_tile} + {ox_var}];\n")
            lines.append(f"{ind}    vec2 top = b0 * cy0.x + b1 * cy0.y + b2 * cy0.z + b3 * cy0.w;\n")
            lines.append(f"{ind}    vec2 bot = b1 * cy1.x + b2 * cy1.y + b3 * cy1.z + b4 * cy1.w;\n")
            lines.append(f"{ind}    vec4 base4 = vec4(top, bot) + vec4(res{r});\n")
            lines.append(f"{ind}    ivec2 p = wg_base + ivec2({ox_var}, {y_expr});\n")
            lines.append(f"{ind}    imageStore(out_image, 2 * p + ivec2(0, 0), vec4(base4.x, 0.0, 0.0, 1.0));\n")
            lines.append(f"{ind}    imageStore(out_image, 2 * p + ivec2(1, 0), vec4(base4.y, 0.0, 0.0, 1.0));\n")
            lines.append(f"{ind}    imageStore(out_image, 2 * p + ivec2(0, 1), vec4(base4.z, 0.0, 0.0, 1.0));\n")
            lines.append(f"{ind}    imageStore(out_image, 2 * p + ivec2(1, 1), vec4(base4.w, 0.0, 0.0, 1.0));\n")
            lines.append(f"{ind}}}\n")
    else:
        lines.append(f"{ind}vec4 cx0 = ngu_taps(0), cx1 = ngu_taps(1);\n")
        for d in range(5):
            lines.append(f"{ind}int blx{d} = clamp({ox_var} + {luma_halo - 2 + d}, lm_x_lo, lm_x_hi);\n")
        for ky in range(py + 4):
            lines.append(f"{ind}int bly{ky} = clamp({oy0_var} + {luma_halo - 2 + ky}, lm_y_lo, lm_y_hi) * {luma_w};\n")
            lines.append(f"{ind}vec2 hr{ky};\n")
            lines.append(f"{ind}{{ float v0 = s_luma[bly{ky} + blx0], v1 = s_luma[bly{ky} + blx1], v2 = s_luma[bly{ky} + blx2], v3 = s_luma[bly{ky} + blx3], v4 = s_luma[bly{ky} + blx4];\n")
            lines.append(f"{ind}  hr{ky} = vec2(dot(vec4(v0, v1, v2, v3), cx0), dot(vec4(v1, v2, v3, v4), cx1)); }}\n")
        for r in range(py):
            y_expr = oy0_var if r == 0 else f"{oy0_var} + {r}"
            cond = f"{y_expr} < {th_tile} && wg_base.y + {y_expr} < H" if check_th else f"wg_base.y + {y_expr} < H"
            lines.append(f"{ind}if ({cond}) {{\n")
            lines.append(f"{ind}    vec2 top = hr{r} * cx0.x + hr{r + 1} * cx0.y + hr{r + 2} * cx0.z + hr{r + 3} * cx0.w;\n")
            lines.append(f"{ind}    vec2 bot = hr{r + 1} * cx1.x + hr{r + 2} * cx1.y + hr{r + 3} * cx1.z + hr{r + 4} * cx1.w;\n")
            lines.append(f"{ind}    vec4 base4 = vec4(top, bot) + vec4(res{r});\n")
            lines.append(f"{ind}    ivec2 p = wg_base + ivec2({ox_var}, {y_expr});\n")
            lines.append(f"{ind}    imageStore(out_image, 2 * p + ivec2(0, 0), vec4(base4.x, 0.0, 0.0, 1.0));\n")
            lines.append(f"{ind}    imageStore(out_image, 2 * p + ivec2(1, 0), vec4(base4.y, 0.0, 0.0, 1.0));\n")
            lines.append(f"{ind}    imageStore(out_image, 2 * p + ivec2(0, 1), vec4(base4.z, 0.0, 0.0, 1.0));\n")
            lines.append(f"{ind}    imageStore(out_image, 2 * p + ivec2(1, 1), vec4(base4.w, 0.0, 0.0, 1.0));\n")
            lines.append(f"{ind}}}\n")
    return "".join(lines)


def emit_aa_combine_rows(luma_w: int, luma_halo: int, th_tile: int, py: int,
                         ox_var: str, oy0_var: str, check_th: bool, ind: str) -> str:
    lines = [
        f"{ind}int lx0 = clamp({ox_var} + {luma_halo}, lm_x_lo, lm_x_hi), lx1 = clamp({ox_var} + {luma_halo + 1}, lm_x_lo, lm_x_hi);\n"
    ]
    for ky in range(py + 1):
        lines.append(f"{ind}int ly{ky} = clamp({oy0_var} + {luma_halo + ky}, lm_y_lo, lm_y_hi) * {luma_w};\n")
        lines.append(f"{ind}float sl{ky}_0 = s_luma[ly{ky} + lx0], sl{ky}_1 = s_luma[ly{ky} + lx1];\n")
    for r in range(py):
        y_expr = oy0_var if r == 0 else f"{oy0_var} + {r}"
        cond = f"{y_expr} < {th_tile} && wg_base.y + {y_expr} < H" if check_th else f"wg_base.y + {y_expr} < H"
        lines.append(f"{ind}if ({cond}) {{\n")
        lines.append(f"{ind}    vec3 fh = vec3(h{r});\n")
        lines.append(f"{ind}    ivec2 p = wg_base + ivec2({ox_var}, {y_expr});\n")
        lines.append(f"{ind}    imageStore(out_image, 2 * p, vec4(sl{r}_0, 0.0, 0.0, 1.0));\n")
        lines.append(f"{ind}    imageStore(out_image, 2 * p + ivec2(1, 0), vec4((sl{r}_0 + sl{r}_1) * 0.5 + fh.y, 0.0, 0.0, 1.0));\n")
        lines.append(f"{ind}    imageStore(out_image, 2 * p + ivec2(0, 1), vec4((sl{r}_0 + sl{r + 1}_0) * 0.5 + fh.x, 0.0, 0.0, 1.0));\n")
        lines.append(f"{ind}    imageStore(out_image, 2 * p + ivec2(1, 1), vec4(((sl{r}_0 + sl{r}_1) + sl{r + 1}_0 + sl{r + 1}_1) * 0.25 + fh.z, 0.0, 0.0, 1.0));\n")
        lines.append(f"{ind}}}\n")
    return "".join(lines)


def emit_p0_pass(label: str, save: str, tw_tile: int, th_tile: int, py_b1: int,
                 stage0, stage1, native_fp16: bool) -> str:
    """b0 + b1 pass with workgroup aligned 100% to the heavier b1 layer and coalesced planar output."""
    assert th_tile % py_b1 == 0
    wg_x, wg_y = tw_tile, th_tile // py_b1
    nthreads = wg_x * wg_y
    ci0, co0, w0, b0 = stage0
    ci1, co1, w1, b1 = stage1
    assert ci0 == 1
    g0, g1 = co0 // 4, co1 // 4
    b0_w, b0_h = tw_tile + 2, th_tile + 2
    b0_plane = b0_w * b0_h
    luma_w, luma_h = tw_tile + 4, th_tile + 4

    smem_fp32 = luma_w * luma_h * 4 + b0_plane * g0 * 16
    if smem_fp32 > MAX_PORTABLE_SMEM_BYTES:
        raise ValueError(f"{label}: shared memory {smem_fp32} exceeds {MAX_PORTABLE_SMEM_BYTES}")

    w_ex, h_ex, sx, sy = tex_dims(co1)
    hdr = emit_header(label, ["LUMA"], save, w_ex, h_ex, 4,
                      sx * tw_tile, sy * th_tile, wg_x, wg_y)
    code = [hdr]
    if native_fp16:
        code.append("#define NGU_NATIVE_FP16 1\n")
    code.append(TYPE_DECL)
    code.append(f"shared float s_luma[{luma_w * luma_h}];\n")
    code.append(f"shared ACC4 s_feat[{b0_plane * g0}];\n")
    code.append(f"""void hook() {{
    int tid = int(gl_LocalInvocationIndex);
    int lx = int(gl_LocalInvocationID.x), ly = int(gl_LocalInvocationID.y);
    int W = int(LUMA_size.x), H = int(LUMA_size.y);
    ivec2 wg_base = ivec2(gl_WorkGroupID.xy) * ivec2({tw_tile}, {th_tile});
    for (int i = tid; i < {luma_w * luma_h}; i += {nthreads}) {{
        int ix = i % {luma_w}, iy = i / {luma_w};
        ivec2 sp = clamp(wg_base + ivec2(ix - 2, iy - 2), ivec2(0), ivec2(W - 1, H - 1));
        s_luma[i] = texelFetch(LUMA_raw, sp, 0).r * LUMA_mul;
    }}
    barrier();
    {{
""")
    code.append(emit_clamp_bounds(luma_w, luma_h, 2, 2, "b0_", "        "))
    ny_pairs = (b0_h + 1) // 2
    npairs = b0_w * ny_pairs
    has_odd = (b0_h % 2) != 0
    if npairs <= nthreads:
        code.append(f"        if (tid < {npairs}) {{\n")
        code.append(f"            int ox = tid % {b0_w}, oy0 = (tid / {b0_w}) * 2;\n")
        if has_odd:
            code.append(f"            bool has_y1 = (oy0 + 1 < {b0_h});\n")
        code.append(emit_conv_1ch_rows(w0, b0, co0, luma_w, "ox", "oy0", 2, has_odd, "b0_", "            "))
        for g in range(g0):
            code.append(f"            s_feat[{g * b0_plane} + oy0 * {b0_w} + ox] = a0_{g};\n")
        if has_odd:
            code.append("            if (has_y1) {\n")
        for g in range(g0):
            code.append(f"                s_feat[{g * b0_plane} + (oy0 + 1) * {b0_w} + ox] = a1_{g};\n")
        if has_odd:
            code.append("            }\n")
        code.append("        }\n")
    else:
        code.append(f"        for (int pair_id = tid; pair_id < {npairs}; pair_id += {nthreads}) {{\n")
        code.append(f"            int ox = pair_id % {b0_w}, oy0 = (pair_id / {b0_w}) * 2;\n")
        if has_odd:
            code.append(f"            bool has_y1 = (oy0 + 1 < {b0_h});\n")
        code.append(emit_conv_1ch_rows(w0, b0, co0, luma_w, "ox", "oy0", 2, has_odd, "b0_", "            "))
        for g in range(g0):
            code.append(f"            s_feat[{g * b0_plane} + oy0 * {b0_w} + ox] = a0_{g};\n")
        if has_odd:
            code.append("            if (has_y1) {\n")
        for g in range(g0):
            code.append(f"                s_feat[{g * b0_plane} + (oy0 + 1) * {b0_w} + ox] = a1_{g};\n")
        if has_odd:
            code.append("            }\n")
        code.append("        }\n")
    code.append("    }\n    barrier();\n    {\n")
    code.append(emit_clamp_bounds(b0_w, b0_h, 1, 1, "b1_", "        "))
    code.append("        int oy0 = ly" + ("" if py_b1 == 1 else f" * {py_b1}") + ";\n")
    code.append(emit_conv_mch_rows(w1, b1, ci1, co1, b0_w, b0_h, "lx", "oy0", py_b1, False, "b1_", True, "        "))
    code.append("        if (wg_base.x + lx < W) {\n")
    for r in range(py_b1):
        y_expr = "oy0" if r == 0 else f"oy0 + {r}"
        code.append(f"            if (wg_base.y + {y_expr} < H) {{\n")
        code.append(emit_store_tex(g1, "wg_base.x + lx", f"wg_base.y + {y_expr}", r, "                "))
        code.append("            }\n")
    code.append("        }\n    }\n}\n")
    return "".join(code)


def emit_2l_warp_pass(label: str, source: str, save: str,
                      tw_tile: int, th_tile: int, stage0, stage1,
                      native_fp16: bool) -> str:
    """Warp-aligned 2-layer body pass with planar MLP loader, ColumnGroupRows, and coalesced output."""
    wg_x, wg_y = tw_tile + 2, th_tile + 2
    nthreads = wg_x * wg_y
    in_w, in_h = tw_tile + 4, th_tile + 4
    ci0, co0, w0, b0 = stage0
    ci1, co1, w1, b1 = stage1
    g0, g1 = co0 // 4, co1 // 4
    gin = ci0 // 4
    in_plane = in_w * in_h
    mid_plane = wg_x * wg_y
    max_feat = max(in_plane * gin, mid_plane * g0)
    if max_feat * 16 > MAX_PORTABLE_SMEM_BYTES:
        raise ValueError(f"{label}: shared memory {max_feat * 16} exceeds {MAX_PORTABLE_SMEM_BYTES}")

    w_ex, h_ex, sx, sy = tex_dims(co1)
    hdr = emit_header(label, [source, "LUMA"], save, w_ex, h_ex, 4,
                      sx * tw_tile, sy * th_tile, wg_x, wg_y)

    code = [hdr]
    if native_fp16:
        code.append("#define NGU_NATIVE_FP16 1\n")
    code.append(TYPE_DECL)
    code.append(f"shared ACC4 s_feat[{max_feat}];\n")
    code.append(f"""void hook() {{
    int tid = int(gl_LocalInvocationIndex);
    int lx = int(gl_LocalInvocationID.x), ly = int(gl_LocalInvocationID.y);
    int W = int(LUMA_size.x), H = int(LUMA_size.y);
    ivec2 wg_base = ivec2(gl_WorkGroupID.xy) * ivec2({tw_tile}, {th_tile});
""")
    code.append(emit_load_planar_mlp(source, gin, in_w, in_h, 2, nthreads, "    "))
    code.append("    barrier();\n")
    code.append(emit_clamp_bounds(in_w, in_h, 2, 2, "s0_", "    "))
    code.append("    int s0_cx0 = clamp(lx, s0_x_lo, s0_x_hi), s0_cx1 = clamp(lx + 1, s0_x_lo, s0_x_hi), s0_cx2 = clamp(lx + 2, s0_x_lo, s0_x_hi);\n")
    code.append(f"    int s0_cy0 = clamp(ly, s0_y_lo, s0_y_hi) * {in_w}, s0_cy1 = clamp(ly + 1, s0_y_lo, s0_y_hi) * {in_w}, s0_cy2 = clamp(ly + 2, s0_y_lo, s0_y_hi) * {in_w};\n")
    for g in range(g0):
        code.append(f"    ACC4 r_{g} = {acc4(b0[4 * g:4 * g + 4])};\n")
    code.append("    {\n        ACC4 v;\n")
    for ig in range(gin):
        base_off = ig * in_plane
        for kx in range(3):
            for ky in range(3):
                tap = ky * 3 + kx
                code.append(f"        v = s_feat[{base_off} + s0_cy{ky} + s0_cx{kx}];\n")
                for lane, swz in enumerate(("xxxx", "yyyy", "zzzz", "wwww")):
                    ic = ig * 4 + lane
                    for g in range(g0):
                        wv = acc4(w0[(oc * ci0 + ic) * 9 + tap] for oc in range(4 * g, 4 * g + 4))
                        code.append(f"        r_{g} = fma(v.{swz}, {wv}, r_{g});\n")
    code.append("    }\n")
    for g in range(g0):
        code.append(f"    r_{g} = max(r_{g}, r_{g} * ACC(0.1));\n")
    code.append("    barrier();\n")
    for g in range(g0):
        code.append(f"    s_feat[{g * mid_plane} + ly * {wg_x} + lx] = r_{g};\n")
    code.append("    barrier();\n")

    code.append(f"    if (lx < {tw_tile} && ly < {th_tile}) {{\n")
    code.append("        ivec2 dst_p = wg_base + ivec2(lx, ly);\n")
    code.append("        if (dst_p.x < W && dst_p.y < H) {\n")
    code.append(emit_clamp_bounds(wg_x, wg_y, 1, 1, "s1_", "            "))
    code.append("            int s1_cx0 = clamp(lx, s1_x_lo, s1_x_hi), s1_cx1 = clamp(lx + 1, s1_x_lo, s1_x_hi), s1_cx2 = clamp(lx + 2, s1_x_lo, s1_x_hi);\n")
    code.append(f"            int s1_cy0 = clamp(ly, s1_y_lo, s1_y_hi) * {wg_x}, s1_cy1 = clamp(ly + 1, s1_y_lo, s1_y_hi) * {wg_x}, s1_cy2 = clamp(ly + 2, s1_y_lo, s1_y_hi) * {wg_x};\n")
    for g in range(g1):
        code.append(f"            ACC4 a0_{g} = {acc4(b1[4 * g:4 * g + 4])};\n")
    code.append("            ACC4 v;\n")
    for ig in range(g0):
        base_off = ig * mid_plane
        for kx in range(3):
            for ky in range(3):
                tap = ky * 3 + kx
                code.append(f"            v = s_feat[{base_off} + s1_cy{ky} + s1_cx{kx}];\n")
                for lane, swz in enumerate(("xxxx", "yyyy", "zzzz", "wwww")):
                    ic = ig * 4 + lane
                    for g in range(g1):
                        wv = acc4(w1[(oc * ci1 + ic) * 9 + tap] for oc in range(4 * g, 4 * g + 4))
                        code.append(f"            a0_{g} = fma(v.{swz}, {wv}, a0_{g});\n")
    for g in range(g1):
        code.append(f"            a0_{g} = max(a0_{g}, a0_{g} * ACC(0.1));\n")
    code.append(emit_store_tex(g1, "dst_p.x", "dst_p.y", 0, "            "))
    code.append("        }\n    }\n}\n")
    return "".join(code)


def emit_head_pass(label: str, source: str, tw_tile: int, th_tile: int,
                   final_layers: list, aa: bool, native_fp16: bool) -> str:
    """Standalone head+combine pass (16x16) with planar loader, single-v transpose, and s_hbase."""
    wg_x, wg_y = tw_tile, th_tile
    nthreads = wg_x * wg_y
    ci = final_layers[0][0]
    gin = ci // 4
    in_w, in_h = tw_tile + 2, th_tile + 2
    in_plane = in_w * in_h
    comb_lo, comb_hi = (0, 1) if aa else (-2, 2)
    luma_w = tw_tile + comb_hi - comb_lo
    luma_h = th_tile + comb_hi - comb_lo
    luma_halo = -comb_lo
    use_shbase = not aa
    hbase_elems = tw_tile * (th_tile + 4) if use_shbase else 0

    smem_fp32 = luma_w * luma_h * 4 + in_plane * gin * 16 + hbase_elems * 8
    if smem_fp32 > MAX_PORTABLE_SMEM_BYTES:
        raise ValueError(f"{label}: shared memory {smem_fp32} exceeds {MAX_PORTABLE_SMEM_BYTES}")

    hdr = emit_header(label, [source, "LUMA"], None, "2 *", "2 *", 1,
                      2 * tw_tile, 2 * th_tile, wg_x, wg_y,
                      offset="-0.5 -0.5" if aa else None)
    code = [hdr]
    if native_fp16:
        code.append("#define NGU_NATIVE_FP16 1\n")
    code.append(TYPE_DECL)
    if not aa:
        code.append(BASELINE_HELPERS)
    code.append(f"shared float s_luma[{luma_w * luma_h}];\n")
    code.append(f"shared ACC4 s_feat[{in_plane * gin}];\n")
    if use_shbase:
        code.append(f"shared vec2 s_hbase[{hbase_elems}];\n")

    code.append(f"""void hook() {{
    int tid = int(gl_LocalInvocationIndex);
    int lx = int(gl_LocalInvocationID.x), ly = int(gl_LocalInvocationID.y);
    int W = int(LUMA_size.x), H = int(LUMA_size.y);
    ivec2 wg_base = ivec2(gl_WorkGroupID.xy) * ivec2({tw_tile}, {th_tile});
    for (int i = tid; i < {luma_w * luma_h}; i += {nthreads}) {{
        int ix = i % {luma_w}, iy = i / {luma_w};
        ivec2 sp = clamp(wg_base + ivec2(ix - {luma_halo}, iy - {luma_halo}), ivec2(0), ivec2(W - 1, H - 1));
        s_luma[i] = texelFetch(LUMA_raw, sp, 0).r * LUMA_mul;
    }}
""")
    code.append(emit_load_planar_flat(source, gin, in_w, in_h, 1, nthreads, "    "))
    code.append("    barrier();\n")
    code.append(emit_clamp_bounds(in_w, in_h, 1, 1, "hd_", "    "))
    code.append(emit_clamp_bounds(luma_w, luma_h, luma_halo, luma_halo, "lm_", "    "))
    if use_shbase:
        code.append("    {\n")
        code.append(emit_shbase_populate(tw_tile, th_tile, luma_w, luma_halo, nthreads, "        "))
        code.append("    }\n")
    code.append("    if (wg_base.x + lx < W && wg_base.y + ly < H) {\n")
    code.append("        int hcx0 = clamp(lx, hd_x_lo, hd_x_hi), hcx1 = clamp(lx + 1, hd_x_lo, hd_x_hi), hcx2 = clamp(lx + 2, hd_x_lo, hd_x_hi);\n")
    code.append(f"        int hcy0 = clamp(ly, hd_y_lo, hd_y_hi) * {in_w}, hcy1 = clamp(ly + 1, hd_y_lo, hd_y_hi) * {in_w}, hcy2 = clamp(ly + 2, hd_y_lo, hd_y_hi) * {in_w};\n")
    if aa:
        code.append(emit_aa_head_rows(final_layers, in_plane, 1, "hcx", "hcy", "        "))
        code.append(emit_aa_combine_rows(luma_w, luma_halo, th_tile, 1, "lx", "ly", False, "        "))
    else:
        _, _, w_res, b_res = final_layers[0]
        code.append(emit_transpose_head_rows(w_res, b_res, ci, in_plane, 1, "hcx", "hcy", "        "))
        code.append(emit_sharp_combine_rows(luma_w, luma_halo, tw_tile, th_tile, 1, "lx", "ly", False, True, "        "))
    code.append("    }\n}\n")
    return "".join(code)


def emit_warp_fused_pass(label: str, source: str, save: str | None,
                         tw_tile: int, th_tile: int, py: int,
                         stages: list, final_layers: list | None, aa: bool,
                         native_fp16: bool, use_shbase: bool = False) -> str:
    """Pure warp-stripe fused multi-layer pass (wg_x = max stage width == 32, zero %/ in conv/head)."""
    m = len(stages)
    is_final = final_layers is not None
    halo_end = 1 if is_final else 0
    stage_dims = []
    for k in range(m):
        hk = halo_end + m - 1 - k
        stage_dims.append((tw_tile + 2 * hk, th_tile + 2 * hk, hk))
    in_halo = halo_end + m
    in_w, in_h = tw_tile + 2 * in_halo, th_tile + 2 * in_halo

    wg_x = stage_dims[0][0]
    wg_y = (stage_dims[0][1] + py - 1) // py
    nthreads = wg_x * wg_y

    need_luma = (source == "LUMA") or is_final
    if is_final:
        comb_lo = 0 if aa else -2
        comb_hi = 1 if aa else 2
        if source == "LUMA":
            luma_lo = min(-in_halo, comb_lo)
            luma_hi = max(in_halo, comb_hi)
        else:
            luma_lo, luma_hi = comb_lo, comb_hi
    else:
        luma_lo, luma_hi = -in_halo, in_halo
    luma_w = tw_tile + luma_hi - luma_lo
    luma_h = th_tile + luma_hi - luma_lo
    luma_halo = -luma_lo

    shbase_active = is_final and (not aa) and use_shbase
    hbase_elems = tw_tile * (th_tile + 4) if shbase_active else 0

    feat_sizes = [1]
    if source != "LUMA":
        ci0 = stages[0][0]
        feat_sizes.append(in_w * in_h * (ci0 // 4))
    for k in range(m):
        _, co_k, _, _ = stages[k]
        if k < m - 1 or is_final:
            wk, hk, _ = stage_dims[k]
            feat_sizes.append(wk * hk * (co_k // 4))
    max_feat = max(feat_sizes)

    smem_fp32 = (luma_w * luma_h * 4 if need_luma else 0) + max_feat * 16 + hbase_elems * 8
    if smem_fp32 > MAX_PORTABLE_SMEM_BYTES:
        raise ValueError(f"{label}: shared memory {smem_fp32} exceeds {MAX_PORTABLE_SMEM_BYTES}")

    binds = ["LUMA"] if source == "LUMA" else [source, "LUMA"]
    if is_final:
        hdr = emit_header(label, binds, None, "2 *", "2 *", 1,
                          2 * tw_tile, 2 * th_tile, wg_x, wg_y,
                          offset="-0.5 -0.5" if aa else None)
    else:
        co_last = stages[-1][1]
        w_ex, h_ex, sx, sy = tex_dims(co_last)
        hdr = emit_header(label, binds, save, w_ex, h_ex, 4,
                          sx * tw_tile, sy * th_tile, wg_x, wg_y)

    code = [hdr]
    if native_fp16:
        code.append("#define NGU_NATIVE_FP16 1\n")
    code.append(TYPE_DECL)
    if is_final and not aa:
        code.append(BASELINE_HELPERS)
    if need_luma:
        code.append(f"shared float s_luma[{luma_w * luma_h}];\n")
    code.append(f"shared ACC4 s_feat[{max_feat}];\n")
    if shbase_active:
        code.append(f"shared vec2 s_hbase[{hbase_elems}];\n")

    code.append(f"""void hook() {{
    int tid = int(gl_LocalInvocationIndex);
    int lx = int(gl_LocalInvocationID.x), ly = int(gl_LocalInvocationID.y);
    int y0 = ly * {py};
    int W = int(LUMA_size.x), H = int(LUMA_size.y);
    ivec2 wg_base = ivec2(gl_WorkGroupID.xy) * ivec2({tw_tile}, {th_tile});
""")
    if need_luma:
        code.append(f"""    for (int i = tid; i < {luma_w * luma_h}; i += {nthreads}) {{
        int ix = i % {luma_w}, iy = i / {luma_w};
        ivec2 sp = clamp(wg_base + ivec2(ix - {luma_halo}, iy - {luma_halo}), ivec2(0), ivec2(W - 1, H - 1));
        s_luma[i] = texelFetch(LUMA_raw, sp, 0).r * LUMA_mul;
    }}
""")
    if source != "LUMA":
        gin = stages[0][0] // 4
        code.append(emit_load_planar_mlp(source, gin, in_w, in_h, in_halo, nthreads, "    "))
    code.append("    barrier();\n")

    cur_w, cur_h, cur_halo = in_w, in_h, in_halo
    for k, (ci, co, w, b) in enumerate(stages):
        wk, hk, halo_k = stage_dims[k]
        outs = co // 4
        out_plane = wk * hk
        writes_smem = (k < m - 1) or is_final
        needs_inplace = (k > 0 or source != "LUMA") and writes_smem
        pfx = f"b{k}_"
        need_x = wk < wg_x
        need_y = hk < wg_y * py
        conds = []
        if need_x:
            conds.append(f"lx < {wk}")
        if need_y:
            conds.append(f"y0 < {hk}")
        cond_str = " && ".join(conds)
        has_odd_h = (hk % py) != 0

        code.append("    {\n")
        if k == 0 and source == "LUMA":
            code.append(emit_clamp_bounds(luma_w, luma_h, luma_halo, luma_halo, pfx, "        "))
        else:
            code.append(emit_clamp_bounds(cur_w, cur_h, cur_halo, cur_halo, pfx, "        "))
        if needs_inplace:
            for r in range(py):
                for g in range(outs):
                    code.append(f"        ACC4 r{r}_{g} = ACC4(0.0);\n")
        if cond_str:
            code.append(f"        if ({cond_str}) {{\n")
        ind = "            " if cond_str else "        "
        if has_odd_h:
            code.append(f"{ind}bool has_y1 = (y0 + 1 < {hk});\n")

        if k == 0 and source == "LUMA":
            off = luma_halo - halo_k - 1
            code.append(f"{ind}int lx_in = lx + {off}, ly0_in = y0 + {off};\n")
            code.append(emit_conv_1ch_rows(w, b, co, luma_w, "lx_in", "ly0_in", py, has_odd_h, pfx, ind))
        else:
            code.append(emit_conv_mch_rows(w, b, ci, co, cur_w, cur_h, "lx", "y0", py, has_odd_h, pfx, True, ind))

        if needs_inplace:
            for r in range(py):
                for g in range(outs):
                    code.append(f"{ind}r{r}_{g} = a{r}_{g};\n")
            if cond_str:
                code.append("        }\n")
            code.append("        barrier();\n")
            if cond_str:
                code.append(f"        if ({cond_str}) {{\n")
            for r in range(py):
                if need_y and r > 0:
                    code.append(f"{ind}if (y0 + {r} < {hk}) {{\n")
                    for g in range(outs):
                        code.append(f"{ind}    s_feat[{g * out_plane} + (y0 + {r}) * {wk} + lx] = r{r}_{g};\n")
                    code.append(f"{ind}}}\n")
                else:
                    y_expr = "y0" if r == 0 else f"(y0 + {r})"
                    for g in range(outs):
                        code.append(f"{ind}s_feat[{g * out_plane} + {y_expr} * {wk} + lx] = r{r}_{g};\n")
            if cond_str:
                code.append("        }\n")
            code.append("        barrier();\n    }\n")
        elif writes_smem:
            for r in range(py):
                if need_y and r > 0:
                    code.append(f"{ind}if (y0 + {r} < {hk}) {{\n")
                    for g in range(outs):
                        code.append(f"{ind}    s_feat[{g * out_plane} + (y0 + {r}) * {wk} + lx] = a{r}_{g};\n")
                    code.append(f"{ind}}}\n")
                else:
                    y_expr = "y0" if r == 0 else f"(y0 + {r})"
                    for g in range(outs):
                        code.append(f"{ind}s_feat[{g * out_plane} + {y_expr} * {wk} + lx] = a{r}_{g};\n")
            if cond_str:
                code.append("        }\n")
            code.append("        barrier();\n    }\n")
        else:
            code.append(f"{ind}if (wg_base.x + lx < W) {{\n")
            for r in range(py):
                y_expr = "y0" if r == 0 else f"y0 + {r}"
                cond_r = f"{y_expr} < {hk} && " if (need_y and r > 0) else ""
                code.append(f"{ind}    if ({cond_r}wg_base.y + {y_expr} < H) {{\n")
                code.append(emit_store_tex(outs, "wg_base.x + lx", f"wg_base.y + {y_expr}", r, ind + "        "))
                code.append(f"{ind}    }}\n")
            code.append(f"{ind}}}\n")
            if cond_str:
                code.append("        }\n")
            code.append("    }\n")
        cur_w, cur_h, cur_halo = wk, hk, halo_k

    if is_final:
        in_plane = cur_w * cur_h
        code.append("    {\n")
        code.append(emit_clamp_bounds(cur_w, cur_h, cur_halo, cur_halo, "hd_", "        "))
        code.append(emit_clamp_bounds(luma_w, luma_h, luma_halo, luma_halo, "lm_", "        "))
        if shbase_active:
            code.append(emit_shbase_populate(tw_tile, th_tile, luma_w, luma_halo, nthreads, "        "))
        code.append(f"        if (lx < {tw_tile} && y0 < {th_tile} && wg_base.x + lx < W) {{\n")
        code.append("            int hcx0 = clamp(lx, hd_x_lo, hd_x_hi), hcx1 = clamp(lx + 1, hd_x_lo, hd_x_hi), hcx2 = clamp(lx + 2, hd_x_lo, hd_x_hi);\n")
        for dy in range(py + 2):
            code.append(f"            int hcy{dy} = clamp(y0 + {dy}, hd_y_lo, hd_y_hi) * {cur_w};\n")
        if aa:
            code.append(emit_aa_head_rows(final_layers, in_plane, py, "hcx", "hcy", "            "))
            code.append(emit_aa_combine_rows(luma_w, luma_halo, th_tile, py, "lx", "y0", True, "            "))
        else:
            ci, _, w_res, b_res = final_layers[0]
            code.append(emit_transpose_head_rows(w_res, b_res, ci, in_plane, py, "hcx", "hcy", "            "))
            code.append(emit_sharp_combine_rows(luma_w, luma_halo, tw_tile, th_tile, py, "lx", "y0", True, shbase_active, "            "))
        code.append("        }\n    }\n")

    code.append("}\n")
    return "".join(code)


def emit_fused_pass(label: str, source: str, save: str,
                    tw_tile: int, th_tile: int, wg_x: int, wg_y: int,
                    stages: list, native_fp16: bool) -> str:
    """Cooperative 256-thread fused 3-layer body pass (high occupancy for native-FP16 medium pass 1)."""
    nthreads = wg_x * wg_y
    m = len(stages)
    stage_dims = []
    for k in range(m):
        hk = m - 1 - k
        stage_dims.append((tw_tile + 2 * hk, th_tile + 2 * hk, hk))
    in_halo = m
    in_w, in_h = tw_tile + 2 * in_halo, th_tile + 2 * in_halo
    luma_w, luma_h, luma_halo = in_w, in_h, in_halo

    feat_sizes = [1]
    for k in range(m - 1):
        _, co_k, _, _ = stages[k]
        wk, hk, _ = stage_dims[k]
        feat_sizes.append(wk * hk * (co_k // 4))
    max_feat = max(feat_sizes)
    smem_fp32 = luma_w * luma_h * 4 + max_feat * 16
    if smem_fp32 > MAX_PORTABLE_SMEM_BYTES:
        raise ValueError(f"{label}: shared memory {smem_fp32} exceeds {MAX_PORTABLE_SMEM_BYTES}")

    co_last = stages[-1][1]
    w_ex, h_ex, sx, sy = tex_dims(co_last)
    hdr = emit_header(label, ["LUMA"], save, w_ex, h_ex, 4,
                      sx * tw_tile, sy * th_tile, wg_x, wg_y)
    code = [hdr]
    if native_fp16:
        code.append("#define NGU_NATIVE_FP16 1\n")
    code.append(TYPE_DECL)
    code.append(f"shared float s_luma[{luma_w * luma_h}];\n")
    code.append(f"shared ACC4 s_feat[{max_feat}];\n")
    code.append(f"""void hook() {{
    int tid = int(gl_LocalInvocationIndex);
    int W = int(LUMA_size.x), H = int(LUMA_size.y);
    ivec2 wg_base = ivec2(gl_WorkGroupID.xy) * ivec2({tw_tile}, {th_tile});
    for (int i = tid; i < {luma_w * luma_h}; i += {nthreads}) {{
        int lx = i % {luma_w}, ly = i / {luma_w};
        ivec2 sp = clamp(wg_base + ivec2(lx - {luma_halo}, ly - {luma_halo}), ivec2(0), ivec2(W - 1, H - 1));
        s_luma[i] = texelFetch(LUMA_raw, sp, 0).r * LUMA_mul;
    }}
    barrier();
""")
    cur_w, cur_h, cur_halo = in_w, in_h, in_halo
    for k, (ci, co, w, b) in enumerate(stages):
        wk, hk, halo_k = stage_dims[k]
        outs = co // 4
        writes_smem = k < m - 1
        needs_inplace = k > 0 and writes_smem
        pfx = f"b{k}_"
        bounds_code = emit_clamp_bounds(cur_w, cur_h, cur_halo, cur_halo, pfx, "        ")
        ny_pairs = (hk + 1) // 2
        npairs = wk * ny_pairs
        has_odd_h = (hk % 2) != 0

        if npairs <= nthreads:
            code.append("    {\n        int pair_id = tid;\n")
            code.append(bounds_code)
            if needs_inplace:
                for g in range(outs):
                    code.append(f"        ACC4 r0_{g} = ACC4(0.0), r1_{g} = ACC4(0.0);\n")
            if npairs < nthreads:
                code.append(f"        if (pair_id < {npairs}) {{\n")
            code.append(f"        int ox = pair_id % {wk}, oy0 = (pair_id / {wk}) * 2;\n")
            if has_odd_h:
                code.append(f"        bool has_y1 = (oy0 + 1 < {hk});\n")
            code.append(emit_conv_mch_rows(w, b, ci, co, cur_w, cur_h, "ox", "oy0", 2, has_odd_h, pfx, True))
            if needs_inplace:
                for g in range(outs):
                    code.append(f"        r0_{g} = a0_{g}; r1_{g} = a1_{g};\n")
                if npairs < nthreads:
                    code.append("        }\n")
                code.append("        barrier();\n")
                out_plane = wk * hk
                if npairs < nthreads:
                    code.append(f"        if (pair_id < {npairs}) {{\n")
                code.append(f"            int ox2 = pair_id % {wk}, oy2 = (pair_id / {wk}) * 2;\n")
                for g in range(outs):
                    code.append(f"            s_feat[{g * out_plane} + oy2 * {wk} + ox2] = r0_{g};\n")
                if has_odd_h:
                    code.append(f"            if (oy2 + 1 < {hk}) {{\n")
                for g in range(outs):
                    code.append(f"                s_feat[{g * out_plane} + (oy2 + 1) * {wk} + ox2] = r1_{g};\n")
                if has_odd_h:
                    code.append("            }\n")
                if npairs < nthreads:
                    code.append("        }\n")
                code.append("        barrier();\n    }\n")
            else:
                code.append("        if (wg_base.x + ox < W) {\n")
                code.append("            if (wg_base.y + oy0 < H) {\n")
                code.append(emit_store_tex(outs, "wg_base.x + ox", "wg_base.y + oy0", 0, "                "))
                code.append("            }\n")
                cond1 = "has_y1 && " if has_odd_h else ""
                code.append(f"            if ({cond1}wg_base.y + oy0 + 1 < H) {{\n")
                code.append(emit_store_tex(outs, "wg_base.x + ox", "wg_base.y + oy0 + 1", 1, "                "))
                code.append("            }\n        }\n")
                if npairs < nthreads:
                    code.append("        }\n")
                code.append("    }\n")
        else:
            out_plane = wk * hk
            code.append("    {\n")
            code.append(bounds_code)
            code.append(f"        for (int pair_id = tid; pair_id < {npairs}; pair_id += {nthreads}) {{\n")
            code.append(f"            int ox = pair_id % {wk}, oy0 = (pair_id / {wk}) * 2;\n")
            if has_odd_h:
                code.append(f"            bool has_y1 = (oy0 + 1 < {hk});\n")
            code.append(emit_conv_1ch_rows(w, b, co, luma_w, "ox", "oy0", 2, has_odd_h, pfx, "            "))
            for g in range(outs):
                code.append(f"            s_feat[{g * out_plane} + oy0 * {wk} + ox] = a0_{g};\n")
            if has_odd_h:
                code.append("            if (has_y1) {\n")
            for g in range(outs):
                code.append(f"                s_feat[{g * out_plane} + (oy0 + 1) * {wk} + ox] = a1_{g};\n")
            if has_odd_h:
                code.append("            }\n")
            code.append("        }\n    }\n    barrier();\n")
        cur_w, cur_h, cur_halo = wk, hk, halo_k
    code.append("}\n")
    return "".join(code)


def render(model_id: str, layers: list, aa: bool, native_fp16: bool = False) -> str:
    nbody = len(layers) - (3 if aa else 1)
    body = layers[:nbody]
    heads = layers[nbody:]
    label = f"NGU {model_id}"
    passes = []

    # AA texel (0,0) is at the first input sample center. The texture's
    # sampling rectangle therefore starts half an output texel earlier (-0.5).
    if nbody == 3:
        # Low: 1 single warp-stripe fused pass (0 intermediate textures)
        passes.append(emit_warp_fused_pass(label, "LUMA", None,
                                           26, 22 if aa else 18, 2,
                                           body, heads, aa, native_fp16, use_shbase=not aa))
    elif nbody == 5:
        # Medium: 2 fused passes (1 planar 2x1 8-ch texture)
        if native_fp16:
            passes.append(emit_fused_pass(f"{label} pass 1", "LUMA", "NGU_F0",
                                          24, 16, 32, 8, body[:3], native_fp16))
            passes.append(emit_warp_fused_pass(f"{label} pass 2", "NGU_F0", None,
                                               28, 18 if aa else 16, 2,
                                               body[3:], heads, aa, native_fp16, use_shbase=not aa))
        else:
            passes.append(emit_warp_fused_pass(f"{label} pass 1", "LUMA", "NGU_F0",
                                               28, 14, 2, body[:3], None, aa, native_fp16))
            passes.append(emit_warp_fused_pass(f"{label} pass 2", "NGU_F0", None,
                                               28, 18, 2, body[3:], heads, aa, native_fp16, use_shbase=False))
    elif nbody == 6:
        # High: 4 fused passes reusing 2 ping-pong planar 3x1 12-ch textures
        passes.append(emit_p0_pass(f"{label} pass 1", "NGU_F0",
                                   32, 16 if native_fp16 else 12, 2 if native_fp16 else 1,
                                   body[0], body[1], native_fp16))
        passes.append(emit_2l_warp_pass(f"{label} pass 2", "NGU_F0", "NGU_F1",
                                        30, 14, body[2], body[3], native_fp16))
        passes.append(emit_2l_warp_pass(f"{label} pass 3", "NGU_F1", "NGU_F0",
                                        30, 14, body[4], body[5], native_fp16))
        passes.append(emit_head_pass(f"{label} pass 4", "NGU_F0",
                                     16, 16, heads, aa, native_fp16))
    elif nbody == 8:
        # VeryHigh: 5 fused passes reusing 2 ping-pong planar 2x2 16-ch textures
        passes.append(emit_p0_pass(f"{label} pass 1", "NGU_F0",
                                   32, 12, 1, body[0], body[1], native_fp16))
        passes.append(emit_2l_warp_pass(f"{label} pass 2", "NGU_F0", "NGU_F1",
                                        30, 10, body[2], body[3], native_fp16))
        passes.append(emit_2l_warp_pass(f"{label} pass 3", "NGU_F1", "NGU_F0",
                                        30, 10, body[4], body[5], native_fp16))
        passes.append(emit_2l_warp_pass(f"{label} pass 4", "NGU_F0", "NGU_F1",
                                        30, 10, body[6], body[7], native_fp16))
        passes.append(emit_head_pass(f"{label} pass 5", "NGU_F1",
                                     16, 16, heads, aa, native_fp16))
    else:
        raise ValueError(f"unexpected body stage count: {nbody}")

    return ("// Generated from user-local weights. Do not redistribute.\n"
            "// Load one NGU shader at a time.\n\n"
            + "\n".join(passes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, default=REPO / "weights",
                        help="common root of weights/ and weights-fp16/ (default: repo/weights)")
    parser.add_argument("--output", type=Path, default=Path("~/.config/mpv/shaders/NGU"),
                        help="output directory (default: ~/.config/mpv/shaders/NGU)")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32",
                        help="weight archive (default: fp32)")
    parser.add_argument("--model", action="append", metavar="ID",
                        help="export only this index ID (repeatable; default: all 72)")
    parser.add_argument("--native-fp16", action="store_true",
                        help="opt in to native half tiles and convolution accumulators (requires GPU shaderFloat16)")
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
