#!/usr/bin/env python3
"""Generate ONLY native-2x metadata (not weights) from the existing schema-3 index.

Builds without raw/ or extracted/. Does not extract/export parameters or execute
PE code. The runtime opens and hashes only the selected model's archived files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FAMILIES = {28: "aa", 32: "soft", 36: "standard", 40: "sharp"}
QUALITIES = ("low", "medium", "high", "very_high")

INDEX_SHA256 = "da7412175e69448e0f94fa1c1836c8ad4e154365d91934171daf09cdfb835ac6"


def selected_models(index: Path) -> list[dict]:
    if hashlib.sha256(index.read_bytes()).hexdigest() != INDEX_SHA256:
        raise ValueError("requires the audited schema-3 index SHA-256")
    catalog = json.loads(index.read_text(encoding="utf-8"))
    selected = [m for m in catalog["models"] if m["scale"] == 2]
    if len(selected) != 72:
        raise ValueError("incomplete native-2x model set")
    for m in selected:
        if any(s["expand_2x"] for s in m["stages"]):
            raise ValueError("2x runtime must not contain hidden spatial expansion")
    return selected


def generate(index: Path, fp16_index: Path) -> str:
    selected = selected_models(index)
    half = json.loads(fp16_index.read_text(encoding="utf-8"))
    if (half["schema_version"] != 1 or half["dtype"] != "ieee754-binary16-le" or
            half["source_index_sha256"] != INDEX_SHA256 or
            half["bias_transform"] != "fp32(raw/255) then fp16-rne"):
        raise ValueError("invalid FP16 metadata contract")
    half_models = {m["id"]: m for m in half["models"]}
    lines = ["// Generated metadata only. No parameter values; do not edit.",
             '#include "ngu_model.hpp"', "namespace ngu::catalog {"]
    for m in selected:
        lines.append(f'inline constexpr StageSpec {m["id"]}_stages[] = {{')
        for s in m["stages"]:
            files = []
            for kind in ("weights", "bias"):
                b = s[kind]
                name = f'{m["id"]}/stage{s["index"]:02d}.{kind}.f32le'
                files.append(f'{{"{name}", {b["size"]}, "{b["sha256"]}"}}')
            hs = half_models[m["id"]]["stages"][s["index"]]
            for key in ("index", "in_channels", "out_channels", "kernel_size"):
                if hs[key] != s[key]:
                    raise ValueError("FP16 model shape mismatch")
            for kind in ("weights", "bias"):
                b = hs[kind]
                expected_path = f'{m["id"]}/stage{s["index"]:02d}.{kind}.f16le'
                if (b["path"] != expected_path or b["size"] * 2 != s[kind]["size"] or
                        b["source_sha256"] != s[kind]["sha256"] or len(b["sha256"]) != 64):
                    raise ValueError("FP16 parameter metadata mismatch")
                files.append(f'{{"{b["path"]}", {b["size"]}, "{b["sha256"]}"}}')
            lines.append(f'    {{{s["in_channels"]}, {s["out_channels"]}, {s["kernel_size"]}, ' +
                         ", ".join(files) + "},")
        lines.append("};")
    lines.append("inline constexpr ModelSpec models[] = {")
    for m in selected:
        lines.append(f'    {{"{m["id"]}", "{FAMILIES[m["family_id"]]}", ' +
                     f'"{QUALITIES[m["quality"]]}", {m["selector_option"]}, ' +
                     f'{m["id"]}_stages}},')
    lines.extend(["};", "} // namespace ngu::catalog", ""])
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", required=True, type=Path)
    p.add_argument("--fp16-index", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    result = generate(args.index, args.fp16_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
