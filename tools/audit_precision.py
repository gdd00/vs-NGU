#!/usr/bin/env python3
"""Audit actual NGU_DUMP_SHADERS compiler outputs (including FP16 FMA types)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct


def audit(path):
    data = path.read_bytes()
    words = struct.unpack(f"<{len(data)//4}I", data)
    if len(words) < 5 or words[0] != 0x07230203:
        raise ValueError(f"{path}: invalid SPIR-V")
    instructions, i = [], 5
    while i < len(words):
        count, opcode = words[i] >> 16, words[i] & 65535
        if not count or i + count > len(words):
            raise ValueError(f"{path}: invalid instruction at {i}")
        instructions.append((opcode, words[i+1:i+count]))
        i += count
    expected = 16 if path.name.startswith("fp16-") else 32
    floating = {a[0]: a[1] for op, a in instructions if op == 22}
    vectors = {a[0]: a[1] for op, a in instructions if op == 23}
    if not floating or set(floating.values()) != {expected}:
        raise ValueError(f"{path}: unexpected floating types {floating}")
    sets = {a[0] for op, a in instructions if op == 11 and
            struct.pack(f"<{len(a)-1}I", *a[1:]).split(b"\0")[0] == b"GLSL.std.450"}
    fmas = [a for op, a in instructions if op == 12 and a[2] in sets and a[3] == 50]
    widths = [floating[vectors.get(a[0], a[0])] for a in fmas]
    if expected == 16 and (not fmas or set(widths) != {16}):
        raise ValueError(f"{path}: native half FMA missing or wrong type")
    conversions = sum(op == 115 for op, _ in instructions)  # OpFConvert
    if conversions:
        raise ValueError(f"{path}: unexpected floating conversion")
    return {"module": path.name, "sha256": hashlib.sha256(data).hexdigest(),
            "glsl_sha256": hashlib.sha256(path.with_suffix(".glsl").read_bytes()).hexdigest(),
            "float_widths": sorted(set(floating.values())), "fma_count": len(fmas),
            "fma_widths": sorted(set(widths)), "fconvert_count": conversions,
            "no_contraction_count": sum(op == 71 and len(a) > 1 and a[1] == 42
                                        for op, a in instructions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    modules = [audit(p) for p in sorted(args.directory.rglob("*.spv"))]
    if not modules or not any(m["float_widths"] == [16] for m in modules):
        raise ValueError("no FP16 compiler output found")
    report = {"modules": modules,
              "fp16_modules": sum(m["float_widths"] == [16] for m in modules),
              "fp32_modules": sum(m["float_widths"] == [32] for m in modules)}
    if args.report:
        with args.report.open("x") as f:
            json.dump(report, f, indent=2)
            f.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "modules"}))


if __name__ == "__main__":
    main()
