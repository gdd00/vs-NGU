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
#define ACC4 f16vec4
#else
#define ACC float
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


def acc4(values) -> str:
    return "ACC4(" + ", ".join(f_lit(v) for v in values) + ")"


def tex_dims(co: int) -> tuple[str, str, int, int]:
    """Dense RGBA packing for 8-ch (2x1), 12-ch (3x1), and 16-ch (2x2) intermediates."""
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
        f"{ind}int {pfx}x_hi = min({cur_w - 1}, int(LUMA_size.x) - 1 + {halo_x} - wg_base.x);\n"
        f"{ind}int {pfx}y_lo = max(0, {halo_y} - wg_base.y);\n"
        f"{ind}int {pfx}y_hi = min({cur_h - 1}, int(LUMA_size.y) - 1 + {halo_y} - wg_base.y);\n"
    )


def emit_conv_1ch_rows(w, b, co: int, luma_w: int, x_var: str, y0_var: str,
                       py: int, has_y_guard: bool, pfx: str) -> str:
    outs = co // 4
    lines = []
    for dx in range(3):
        lines.append(f"        int cx{dx} = clamp({x_var} + {dx}, {pfx}x_lo, {pfx}x_hi);\n")
    for dy in range(py + 2):
        lines.append(f"        int cy{dy} = clamp({y0_var} + {dy}, {pfx}y_lo, {pfx}y_hi) * {luma_w};\n")
    for r in range(py):
        for g in range(outs):
            lines.append(f"        ACC4 a{r}_{g} = {acc4(b[4 * g:4 * g + 4])};\n")
    for ky in range(py + 2):
        guarded = has_y_guard and (ky == py + 1)
        if guarded:
            lines.append("        if (has_y1) {\n")
        ind = "            " if guarded else "        "
        for kx in range(3):
            lines.append(f"{ind}ACC s_{ky}_{kx} = ACC(s_luma[cy{ky} + cx{kx}]);\n")
            for r in range(py):
                if r <= ky <= r + 2:
                    tap = (ky - r) * 3 + kx
                    for g in range(outs):
                        wv = acc4(w[(oc * 1 + 0) * 9 + tap] for oc in range(4 * g, 4 * g + 4))
                        lines.append(f"{ind}a{r}_{g} = fma(ACC4(s_{ky}_{kx}), {wv}, a{r}_{g});\n")
        if guarded:
            lines.append("        }\n")
    for r in range(py):
        for g in range(outs):
            lines.append(f"        a{r}_{g} = max(a{r}_{g}, a{r}_{g} * ACC(0.1));\n")
    return "".join(lines)


def emit_conv_mch_rows(w, b, ci: int, co: int, in_w: int, in_h: int,
                       x_var: str, y0_var: str, py: int, has_y_guard: bool,
                       pfx: str, relu: bool = True) -> str:
    ins = ci // 4
    outs = co // 4
    in_plane = in_w * in_h
    lines = []
    for dx in range(3):
        lines.append(f"        int cx{dx} = clamp({x_var} + {dx}, {pfx}x_lo, {pfx}x_hi);\n")
    for dy in range(py + 2):
        lines.append(f"        int cy{dy} = clamp({y0_var} + {dy}, {pfx}y_lo, {pfx}y_hi) * {in_w};\n")
    for r in range(py):
        for g in range(outs):
            lines.append(f"        ACC4 a{r}_{g} = {acc4(b[4 * g:4 * g + 4])};\n")
    lines.append("        ACC4 v;\n")
    for ig in range(ins):
        base_off = ig * in_plane
        for kx in range(3):
            for ky in range(py + 2):
                guarded = has_y_guard and (ky == py + 1)
                if guarded:
                    lines.append("        if (has_y1) {\n")
                ind = "            " if guarded else "        "
                lines.append(f"{ind}v = s_feat[{base_off} + cy{ky} + cx{kx}];\n")
                for lane, swz in enumerate(("xxxx", "yyyy", "zzzz", "wwww")):
                    ic = ig * 4 + lane
                    for r in range(py):
                        if r <= ky <= r + 2:
                            tap = (ky - r) * 3 + kx
                            for g in range(outs):
                                wv = acc4(w[(oc * ci + ic) * 9 + tap] for oc in range(4 * g, 4 * g + 4))
                                lines.append(f"{ind}a{r}_{g} = fma(v.{swz}, {wv}, a{r}_{g});\n")
                if guarded:
                    lines.append("        }\n")
    if relu:
        for r in range(py):
            for g in range(outs):
                lines.append(f"        a{r}_{g} = max(a{r}_{g}, a{r}_{g} * ACC(0.1));\n")
    return "".join(lines)


def emit_store_tex(outs: int, dst_x: str, dst_y: str, r: int, ind: str) -> str:
    lines = []
    for g in range(outs):
        val = f"vec4(a{r}_{g})"
        if outs == 2:
            coord = f"ivec2(2 * ({dst_x}) + {g}, {dst_y})"
        elif outs == 3:
            coord = f"ivec2(3 * ({dst_x}) + {g}, {dst_y})"
        else:
            coord = f"ivec2(2 * ({dst_x}) + {g & 1}, 2 * ({dst_y}) + {g >> 1})"
        lines.append(f"{ind}imageStore(out_image, {coord}, {val});\n")
    return "".join(lines)


def emit_2l_warp_pass(label: str, source: str, save: str,
                      tw_tile: int, th_tile: int, stage0, stage1,
                      native_fp16: bool) -> str:
    """Warp-aligned 2-layer body pass (wg = (tw_tile+2) x (th_tile+2), 1 px/thread)."""
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
    if gin == 2:
        coord_expr = "ivec2(2 * sp.x + g, sp.y)"
    elif gin == 3:
        coord_expr = "ivec2(3 * sp.x + g, sp.y)"
    else:
        coord_expr = "ivec2(2 * sp.x + (g & 1), 2 * sp.y + (g >> 1))"

    code = [hdr]
    if native_fp16:
        code.append("#define NGU_NATIVE_FP16 1\n")
    code.append(TYPE_DECL)
    code.append(f"shared ACC4 s_feat[{max_feat}];\n")
    code.append(f"""void hook() {{
    int tid = int(gl_LocalInvocationIndex);
    int lx = int(gl_LocalInvocationID.x), ly = int(gl_LocalInvocationID.y);
    ivec2 wg_base = ivec2(gl_WorkGroupID.xy) * ivec2({tw_tile}, {th_tile});
    for (int i = tid; i < {in_plane * gin}; i += {nthreads}) {{
        int g = i / {in_plane}, rem = i % {in_plane};
        int ix = rem % {in_w}, iy = rem / {in_w};
        ivec2 sp = clamp(wg_base + ivec2(ix - 2, iy - 2), ivec2(0), ivec2(LUMA_size) - 1);
        s_feat[i] = ACC4(texelFetch({source}_raw, {coord_expr}, 0) * {source}_mul);
    }}
    barrier();
""")
    code.append(emit_clamp_bounds(in_w, in_h, 2, 2, "s0_", "    "))
    code.append("    int s0_cx0 = clamp(lx, s0_x_lo, s0_x_hi), s0_cx1 = clamp(lx + 1, s0_x_lo, s0_x_hi), s0_cx2 = clamp(lx + 2, s0_x_lo, s0_x_hi);\n")
    code.append(f"    int s0_cy0 = clamp(ly, s0_y_lo, s0_y_hi) * {in_w}, s0_cy1 = clamp(ly + 1, s0_y_lo, s0_y_hi) * {in_w}, s0_cy2 = clamp(ly + 2, s0_y_lo, s0_y_hi) * {in_w};\n")
    for g in range(g0):
        code.append(f"    ACC4 r_{g} = {acc4(b0[4 * g:4 * g + 4])};\n")
    code.append("    {\n        ACC4 v;\n")
    for ig in range(gin):
        base_off = ig * in_plane
        for ky in range(3):
            for kx in range(3):
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
    code.append("        if (dst_p.x < int(LUMA_size.x) && dst_p.y < int(LUMA_size.y)) {\n")
    code.append(emit_clamp_bounds(wg_x, wg_y, 1, 1, "s1_", "            "))
    code.append("            int s1_cx0 = clamp(lx, s1_x_lo, s1_x_hi), s1_cx1 = clamp(lx + 1, s1_x_lo, s1_x_hi), s1_cx2 = clamp(lx + 2, s1_x_lo, s1_x_hi);\n")
    code.append(f"            int s1_cy0 = clamp(ly, s1_y_lo, s1_y_hi) * {wg_x}, s1_cy1 = clamp(ly + 1, s1_y_lo, s1_y_hi) * {wg_x}, s1_cy2 = clamp(ly + 2, s1_y_lo, s1_y_hi) * {wg_x};\n")
    for g in range(g1):
        code.append(f"            ACC4 a0_{g} = {acc4(b1[4 * g:4 * g + 4])};\n")
    code.append("            ACC4 v;\n")
    for ig in range(g0):
        base_off = ig * mid_plane
        for ky in range(3):
            for kx in range(3):
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


def emit_fused_pass(label: str, source: str, save: str | None,
                    tw_tile: int, th_tile: int, wg_x: int, wg_y: int,
                    stages: list, final_layers: list | None, aa: bool,
                    native_fp16: bool, last_py: int = 2) -> str:
    nthreads = wg_x * wg_y
    m = len(stages)
    is_final = final_layers is not None
    halo_end = 1 if is_final else 0

    stage_dims = []
    for k in range(m):
        hk = halo_end + m - 1 - k
        stage_dims.append((tw_tile + 2 * hk, th_tile + 2 * hk, hk))
    in_halo = halo_end + m
    in_w, in_h = tw_tile + 2 * in_halo, th_tile + 2 * in_halo

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

    feat_sizes = [1]
    if source != "LUMA":
        ci0 = stages[0][0] if m > 0 else final_layers[0][0]
        feat_sizes.append(in_w * in_h * (ci0 // 4))
    for k in range(m):
        _, co_k, _, _ = stages[k]
        if k < m - 1 or is_final:
            wk, hk, _ = stage_dims[k]
            feat_sizes.append(wk * hk * (co_k // 4))
    max_feat = max(feat_sizes)

    smem_fp32 = (luma_w * luma_h * 4 if need_luma else 0) + (max_feat * 16 if max_feat > 1 else 0)
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
    if max_feat > 1:
        code.append(f"shared ACC4 s_feat[{max_feat}];\n")

    code.append("void hook() {\n")
    code.append("    int tid = int(gl_LocalInvocationIndex);\n")
    code.append(f"    ivec2 wg_base = ivec2(gl_WorkGroupID.xy) * ivec2({tw_tile}, {th_tile});\n")

    if need_luma:
        total_luma = luma_w * luma_h
        code.append(f"""    for (int i = tid; i < {total_luma}; i += {nthreads}) {{
        int lx = i % {luma_w}, ly = i / {luma_w};
        ivec2 sp = clamp(wg_base + ivec2(lx - {luma_halo}, ly - {luma_halo}), ivec2(0), ivec2(LUMA_size) - 1);
        s_luma[i] = texelFetch(LUMA_raw, sp, 0).r * LUMA_mul;
    }}
""")

    cur_w, cur_h, cur_halo = in_w, in_h, in_halo
    if source != "LUMA":
        ci0 = stages[0][0] if m > 0 else final_layers[0][0]
        gin = ci0 // 4
        in_plane = in_w * in_h
        total_in = in_plane * gin
        if gin == 2:
            coord_expr = "ivec2(2 * sp.x + g, sp.y)"
        elif gin == 3:
            coord_expr = "ivec2(3 * sp.x + g, sp.y)"
        else:
            coord_expr = "ivec2(2 * sp.x + (g & 1), 2 * sp.y + (g >> 1))"
        code.append(f"""    for (int i = tid; i < {total_in}; i += {nthreads}) {{
        int g = i / {in_plane}, rem = i % {in_plane};
        int lx = rem % {in_w}, ly = rem / {in_w};
        ivec2 sp = clamp(wg_base + ivec2(lx - {in_halo}, ly - {in_halo}), ivec2(0), ivec2(LUMA_size) - 1);
        s_feat[i] = ACC4(texelFetch({source}_raw, {coord_expr}, 0) * {source}_mul);
    }}
""")
    code.append("    barrier();\n")

    for k, (ci, co, w, b) in enumerate(stages):
        wk, hk, halo_k = stage_dims[k]
        outs = co // 4
        writes_smem = (k < m - 1) or is_final
        needs_inplace_sync = (k > 0 or source != "LUMA") and writes_smem
        pfx = f"b{k}_"

        if k == 0 and source == "LUMA":
            bounds_code = emit_clamp_bounds(luma_w, luma_h, luma_halo, luma_halo, pfx, "        ")
        else:
            bounds_code = emit_clamp_bounds(cur_w, cur_h, cur_halo, cur_halo, pfx, "        ")

        if (not writes_smem) and wk == wg_x and hk == wg_y * last_py:
            code.append("    {\n")
            code.append(bounds_code)
            code.append("        int ox = int(gl_LocalInvocationID.x);\n")
            code.append(f"        int oy0 = int(gl_LocalInvocationID.y) * {last_py};\n")
            code.append(emit_conv_mch_rows(w, b, ci, co, cur_w, cur_h,
                                           "ox", "oy0", last_py, False, pfx, relu=True))
            code.append("        if (wg_base.x + ox < int(LUMA_size.x)) {\n")
            for r in range(last_py):
                code.append(f"            if (wg_base.y + oy0 + {r} < int(LUMA_size.y)) {{\n")
                code.append(emit_store_tex(outs, "wg_base.x + ox", f"wg_base.y + oy0 + {r}", r, "                "))
                code.append("            }\n")
            code.append("        }\n    }\n")
            cur_w, cur_h, cur_halo = wk, hk, halo_k
            continue

        ny_pairs = (hk + 1) // 2
        npairs = wk * ny_pairs
        has_odd_h = (hk % 2) != 0

        if npairs <= nthreads:
            code.append("    {\n        int pair_id = tid;\n")
            code.append(bounds_code)
            if needs_inplace_sync:
                for g in range(outs):
                    code.append(f"        ACC4 r0_{g} = ACC4(0.0), r1_{g} = ACC4(0.0);\n")
            if npairs < nthreads:
                code.append(f"        if (pair_id < {npairs}) {{\n")
            if (wk & (wk - 1)) == 0:
                sh = wk.bit_length() - 1
                code.append(f"        int ox = pair_id & {wk - 1}, oy0 = (pair_id >> {sh}) * 2;\n")
            else:
                code.append(f"        int ox = pair_id % {wk}, oy0 = (pair_id / {wk}) * 2;\n")
            if has_odd_h:
                code.append(f"        bool has_y1 = (oy0 + 1 < {hk});\n")

            if k == 0 and source == "LUMA":
                off = luma_halo - halo_k - 1
                code.append(f"        int lx = ox + {off}, ly0 = oy0 + {off};\n")
                code.append(emit_conv_1ch_rows(w, b, co, luma_w, "lx", "ly0", 2, has_odd_h, pfx))
            else:
                code.append(emit_conv_mch_rows(w, b, ci, co, cur_w, cur_h, "ox", "oy0", 2, has_odd_h, pfx, relu=True))

            if needs_inplace_sync:
                for g in range(outs):
                    code.append(f"        r0_{g} = a0_{g}; r1_{g} = a1_{g};\n")
                if npairs < nthreads:
                    code.append("        }\n")
                code.append("        barrier();\n")
                out_plane = wk * hk
                if npairs < nthreads:
                    code.append(f"        if (pair_id < {npairs}) {{\n")
                if (wk & (wk - 1)) == 0:
                    sh = wk.bit_length() - 1
                    code.append(f"            int ox2 = pair_id & {wk - 1}, oy2 = (pair_id >> {sh}) * 2;\n")
                else:
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
            elif writes_smem:
                out_plane = wk * hk
                for g in range(outs):
                    code.append(f"        s_feat[{g * out_plane} + oy0 * {wk} + ox] = a0_{g};\n")
                if has_odd_h:
                    code.append("        if (has_y1) {\n")
                for g in range(outs):
                    code.append(f"            s_feat[{g * out_plane} + (oy0 + 1) * {wk} + ox] = a1_{g};\n")
                if has_odd_h:
                    code.append("        }\n")
                if npairs < nthreads:
                    code.append("        }\n")
                code.append("        barrier();\n    }\n")
            else:
                code.append("        if (wg_base.x + ox < int(LUMA_size.x)) {\n")
                code.append("            if (wg_base.y + oy0 < int(LUMA_size.y)) {\n")
                code.append(emit_store_tex(outs, "wg_base.x + ox", "wg_base.y + oy0", 0, "                "))
                code.append("            }\n")
                cond1 = "has_y1 && " if has_odd_h else ""
                code.append(f"            if ({cond1}wg_base.y + oy0 + 1 < int(LUMA_size.y)) {{\n")
                code.append(emit_store_tex(outs, "wg_base.x + ox", "wg_base.y + oy0 + 1", 1, "                "))
                code.append("            }\n        }\n")
                if npairs < nthreads:
                    code.append("        }\n")
                code.append("    }\n")
        else:
            off = luma_halo - halo_k - 1
            out_plane = wk * hk
            code.append("    {\n")
            code.append(bounds_code)
            code.append(f"        for (int pair_id = tid; pair_id < {npairs}; pair_id += {nthreads}) {{\n")
            code.append(f"            int ox = pair_id % {wk}, oy0 = (pair_id / {wk}) * 2;\n")
            if has_odd_h:
                code.append(f"            bool has_y1 = (oy0 + 1 < {hk});\n")
            code.append(f"            int lx = ox + {off}, ly0 = oy0 + {off};\n")
            code.append(emit_conv_1ch_rows(w, b, co, luma_w, "lx", "ly0", 2, has_odd_h, pfx))
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

    if is_final:
        total_out = tw_tile * th_tile
        in_plane = cur_w * cur_h
        one_per_thread = (tw_tile == wg_x and th_tile == wg_y)
        code.append("    {\n")
        code.append(emit_clamp_bounds(cur_w, cur_h, cur_halo, cur_halo, "hd_", "        "))
        code.append(emit_clamp_bounds(luma_w, luma_h, luma_halo, luma_halo, "lm_", "        "))
        if one_per_thread:
            code.append("        int ox = int(gl_LocalInvocationID.x), oy = int(gl_LocalInvocationID.y);\n")
            code.append("        ivec2 p = wg_base + ivec2(ox, oy);\n")
            code.append("        if (p.x < int(LUMA_size.x) && p.y < int(LUMA_size.y)) {\n")
        else:
            code.append(f"        for (int idx = tid; idx < {total_out}; idx += {nthreads}) {{\n")
            if (tw_tile & (tw_tile - 1)) == 0:
                sh = tw_tile.bit_length() - 1
                code.append(f"            int ox = idx & {tw_tile - 1}, oy = idx >> {sh};\n")
            else:
                code.append(f"            int ox = idx % {tw_tile}, oy = idx / {tw_tile};\n")
            code.append("            ivec2 p = wg_base + ivec2(ox, oy);\n")
            code.append("            if (p.x >= int(LUMA_size.x) || p.y >= int(LUMA_size.y)) continue;\n")
        code.append("            int hcx0 = clamp(ox, hd_x_lo, hd_x_hi), hcx1 = clamp(ox + 1, hd_x_lo, hd_x_hi), hcx2 = clamp(ox + 2, hd_x_lo, hd_x_hi);\n")
        code.append(f"            int hcy0 = clamp(oy, hd_y_lo, hd_y_hi) * {cur_w}, hcy1 = clamp(oy + 1, hd_y_lo, hd_y_hi) * {cur_w}, hcy2 = clamp(oy + 2, hd_y_lo, hd_y_hi) * {cur_w};\n")
        if aa:
            heads = final_layers
            ci = heads[0][0]
            ins = ci // 4
            # make_aa_head swaps the first two kernels but not the biases!
            w_aa = heads[1][2] + heads[0][2] + heads[2][2] + (0.0,) * (ci * 9)
            b_aa = (heads[0][3][0], heads[1][3][0], heads[2][3][0], 0.0)
            code.append(f"            ACC4 h = {acc4(b_aa)};\n")
            for ig in range(ins):
                base_off = ig * in_plane
                for ky in range(3):
                    for kx in range(3):
                        tap = ky * 3 + kx
                        code.append(f"            {{\n                ACC4 v = s_feat[{base_off} + hcy{ky} + hcx{kx}];\n")
                        for lane, swz in enumerate(("xxxx", "yyyy", "zzzz", "wwww")):
                            ic = ig * 4 + lane
                            wv = acc4(w_aa[(oc * ci + ic) * 9 + tap] for oc in range(4))
                            code.append(f"                h = fma(v.{swz}, {wv}, h);\n")
                        code.append("            }\n")
            code.append(f"""            int lx0 = clamp(ox + {luma_halo}, lm_x_lo, lm_x_hi), lx1 = clamp(ox + {luma_halo + 1}, lm_x_lo, lm_x_hi);
            int ly0 = clamp(oy + {luma_halo}, lm_y_lo, lm_y_hi) * {luma_w}, ly1 = clamp(oy + {luma_halo + 1}, lm_y_lo, lm_y_hi) * {luma_w};
            float s00 = s_luma[ly0 + lx0], s10 = s_luma[ly0 + lx1];
            float s01 = s_luma[ly1 + lx0], s11 = s_luma[ly1 + lx1];
            vec3 fh = vec3(h.xyz);
            imageStore(out_image, 2 * p, vec4(s00, 0.0, 0.0, 1.0));
            imageStore(out_image, 2 * p + ivec2(1, 0), vec4((s00 + s10) * 0.5 + fh.y, 0.0, 0.0, 1.0));
            imageStore(out_image, 2 * p + ivec2(0, 1), vec4((s00 + s01) * 0.5 + fh.x, 0.0, 0.0, 1.0));
            imageStore(out_image, 2 * p + ivec2(1, 1), vec4(((s00 + s10) + s01 + s11) * 0.25 + fh.z, 0.0, 0.0, 1.0));
""")
        else:
            ci, _, w_res, b_res = final_layers[0]
            ins = ci // 4
            code.append(f"            ACC4 res = {acc4(b_res * 4)};\n")
            for ig in range(ins):
                base_off = ig * in_plane
                code.append("            {\n")
                for sy in range(3):
                    for sx in range(3):
                        code.append(f"                ACC4 f{sy}{sx} = s_feat[{base_off} + hcy{sy} + hcx{sx}];\n")
                for lane in range(4):
                    ic = ig * 4 + lane
                    for dy in range(2):
                        for dx in range(2):
                            f_terms = []
                            w_terms = []
                            for py_p in range(2):
                                for px_p in range(2):
                                    fy = 1 + py_p - dy
                                    fx = 1 + px_p - dx
                                    f_terms.append(f"f{fy}{fx}[{lane}]")
                                    wi = ic * 16 + (1 - py_p + dy * 2) * 4 + (1 - px_p + dx * 2)
                                    w_terms.append(w_res[wi])
                            code.append(f"                res = fma(ACC4({', '.join(f_terms)}), {acc4(w_terms)}, res);\n")
                code.append("            }\n")
            code.append(f"""            vec4 cx0 = ngu_taps(0), cx1 = ngu_taps(1);
            vec4 cy0 = cx0, cy1 = cx1;
            int blx0 = clamp(ox + {luma_halo - 2}, lm_x_lo, lm_x_hi);
            int blx1 = clamp(ox + {luma_halo - 1}, lm_x_lo, lm_x_hi);
            int blx2 = clamp(ox + {luma_halo}, lm_x_lo, lm_x_hi);
            int blx3 = clamp(ox + {luma_halo + 1}, lm_x_lo, lm_x_hi);
            int blx4 = clamp(ox + {luma_halo + 2}, lm_x_lo, lm_x_hi);
            float r0[5], r1[5];
            for (int ky = 0; ky < 5; ++ky) {{
                int bly = clamp(oy + {luma_halo - 2} + ky, lm_y_lo, lm_y_hi) * {luma_w};
                float v0 = s_luma[bly + blx0], v1 = s_luma[bly + blx1], v2 = s_luma[bly + blx2], v3 = s_luma[bly + blx3], v4 = s_luma[bly + blx4];
                r0[ky] = dot(vec4(v0, v1, v2, v3), cx0);
                r1[ky] = dot(vec4(v1, v2, v3, v4), cx1);
            }}
            vec4 base4 = vec4(
                dot(vec4(r0[0], r0[1], r0[2], r0[3]), cy0),
                dot(vec4(r1[0], r1[1], r1[2], r1[3]), cy0),
                dot(vec4(r0[1], r0[2], r0[3], r0[4]), cy1),
                dot(vec4(r1[1], r1[2], r1[3], r1[4]), cy1)
            ) + vec4(res);
            imageStore(out_image, 2 * p + ivec2(0, 0), vec4(base4.x, 0.0, 0.0, 1.0));
            imageStore(out_image, 2 * p + ivec2(1, 0), vec4(base4.y, 0.0, 0.0, 1.0));
            imageStore(out_image, 2 * p + ivec2(0, 1), vec4(base4.z, 0.0, 0.0, 1.0));
            imageStore(out_image, 2 * p + ivec2(1, 1), vec4(base4.w, 0.0, 0.0, 1.0));
""")
        code.append("        }\n    }\n")

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
        # Low: 1 single fused pass (0 intermediate textures)
        passes.append(emit_fused_pass(label, "LUMA", None, 24, 14, 32, 8,
                                      body, heads, aa, native_fp16))
    elif nbody == 5:
        # Medium: 2 fused passes (1 dense 2x1 8-ch texture)
        passes.append(emit_fused_pass(f"{label} pass 1", "LUMA", "NGU_F0",
                                      24, 16, 32, 8, body[:3], None, aa, native_fp16))
        passes.append(emit_fused_pass(f"{label} pass 2", "NGU_F0", None,
                                      24, 14, 32, 8, body[3:], heads, aa, native_fp16))
    elif nbody == 6:
        # High: 4 fused passes (3 dense 3x1 12-ch textures)
        passes.append(emit_fused_pass(f"{label} pass 1", "LUMA", "NGU_F0",
                                      32, 16, 32, 8, body[:2], None, aa, native_fp16, last_py=2))
        passes.append(emit_2l_warp_pass(f"{label} pass 2", "NGU_F0", "NGU_F1",
                                        30, 14, body[2], body[3], native_fp16))
        passes.append(emit_2l_warp_pass(f"{label} pass 3", "NGU_F1", "NGU_F2",
                                        30, 14, body[4], body[5], native_fp16))
        passes.append(emit_fused_pass(f"{label} pass 4", "NGU_F2", None,
                                      16, 16, 16, 16, [], heads, aa, native_fp16))
    elif nbody == 8:
        # VeryHigh: 5 fused passes (4 2x2 16-ch textures)
        passes.append(emit_fused_pass(f"{label} pass 1", "LUMA", "NGU_F0",
                                      32, 12, 32, 6, body[:2], None, aa, native_fp16, last_py=2))
        passes.append(emit_2l_warp_pass(f"{label} pass 2", "NGU_F0", "NGU_F1",
                                        30, 10, body[2], body[3], native_fp16))
        passes.append(emit_2l_warp_pass(f"{label} pass 3", "NGU_F1", "NGU_F2",
                                        30, 10, body[4], body[5], native_fp16))
        passes.append(emit_2l_warp_pass(f"{label} pass 4", "NGU_F2", "NGU_F3",
                                        30, 10, body[6], body[7], native_fp16))
        passes.append(emit_fused_pass(f"{label} pass 5", "NGU_F3", None,
                                      16, 16, 16, 16, [], heads, aa, native_fp16))
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
