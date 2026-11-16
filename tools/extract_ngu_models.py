#!/usr/bin/env python3
"""Extract all indexed NGU parameter variants from the exact madVR64 0.92.17.

Default operation is entirely offline and never executes PE code. The OPTIONAL
--verify-native helper executes the audited parameter selectors, restricted to
hard-coded valid inputs, after a SHA-256 whitelist check of its stdin snapshot.
Parameter recovery does not establish complete NGU rendering equivalence.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

from extract_madvr import FormatError, PE, sha256

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "models" / "madvr09217-index.json"
SOURCE_SHA256 = "9f684ba984344d5f578e1590d7d0a62b6b93386598fdfee6b5ae0aa560bf7164"
SOURCE_SIZE = 10645576
SELECTOR_RVA = 0x40e0
SELECTOR_CODE_END = 0xf3a7
SELECTOR_TABLE_END = 0x1068c
SELECTOR_CODE_SHA256 = "5e0899d8d5e35bb31420c4ab0b5c418fcb80a6bb5054f2cc8a4eec26f0850327"
FAMILIES = {28: "aa", 32: "soft", 36: "standard", 40: "sharp"}
QUALITIES = ("low", "medium", "high", "very_high")
AUXILIARY_ID = "auxiliary_selector_family13"
SELECTORS = [
    {"id": 0, "rva": SELECTOR_RVA, "code_end_rva": SELECTOR_CODE_END,
     "table_end_rva": SELECTOR_TABLE_END, "code_sha256": SELECTOR_CODE_SHA256,
     "literal_parameter_rva_count": 2313},
    {"id": 1, "rva": 0x5bc20, "code_end_rva": 0x5fb1b, "table_end_rva": 0x6018c,
     "code_sha256": "44021d64ca961b22f4faf429653a7a32af45b854e3a6087fed73c5697ec587d0",
     "literal_parameter_rva_count": 800},
    {"id": 2, "rva": 0x7c3b0, "code_end_rva": 0x7cfec, "table_end_rva": 0x7d14c,
     "code_sha256": "488285432ea1465bdf760f0fc27637513178a5c3195ae6dd1ac410b3f5cbe372",
     "literal_parameter_rva_count": 154},
]


def check_source(data: bytes) -> None:
    if len(data) != SOURCE_SIZE or sha256(data) != SOURCE_SHA256:
        raise FormatError("unsupported source: requires exact madVR64.ax 0.92.17 SHA-256")


def all_models(catalog: dict) -> list[dict]:
    return catalog["models"] + catalog["auxiliary_models"]


def model_id(selector: int, family: int, scale: int, quality: int, option: int) -> str:
    if selector == 1:
        return f"auxiliary_selector5bc20_mode{family:02d}_{QUALITIES[quality]}"
    if selector == 2:
        return f"auxiliary_selector7c3b0_mode{family:02d}"
    if family == 13:
        return AUXILIARY_ID
    base = f"{FAMILIES[family]}_{scale}x_{QUALITIES[quality]}"
    return base if option == 0 else f"{base}_option{option:02d}"


def expected_model_keys() -> set[tuple[int, ...]]:
    return {(0, f, s, q, o) for f in FAMILIES for s in (2, 4)
            for q in range(4) for o in (range(15) if f == 40 else (0,))
            if not (f == 28 and s == 4)}


def expected_auxiliary_keys() -> set[tuple[int, ...]]:
    return ({(0, 13, 2, 0, 0)} |
            {(1, mode, 0, q, 0) for q in range(4) for mode in range(12 if q == 0 else 16)} |
            {(2, mode, 0, 0, 0) for mode in range(11)})


def parameter_kernel_size(selector: int, family: int, out_channels: int, expand: bool) -> int:
    """Parameter extent, not a guess from the selector's expansion flag.

    The flag only marks hidden spatial expansion. Non-AA NGU output upload at
    VA 4a4a13f0..4a4a143b ALSO consumes 16 weights per input channel, using
    Cubicd10/11/12. AA uses three separate 3x3 heads (Cubicp1*).
    Auxiliary heads are kept at their previously indexed extents; these have
    not been audited as NGU upsampling graphs.
    """
    return 4 if expand or (selector == 0 and family in (32, 36, 40) and out_channels == 1) else 3


def records(catalog: dict, *, include_auxiliary: bool = False) -> list[tuple[int, ...]]:
    selected = all_models(catalog) if include_auxiliary else catalog["models"]
    return [
        (m["selector_id"], m["family_id"], m["scale"], m["quality"], m["selector_option"],
         s["index"], s["in_channels"], s["out_channels"],
         s["weights"]["rva"], s["bias"]["rva"], int(s["expand_2x"]))
        for m in selected for s in m["stages"]
    ]


def parameter_rvas(catalog: dict) -> set[int]:
    return {s[k]["rva"] for m in all_models(catalog) for s in m["stages"]
            for k in ("weights", "bias")}


def rva_digest(rvas: set[int]) -> str:
    return sha256(b"".join(struct.pack("<I", r) for r in sorted(rvas)))


def parameter_summary(selected: list[dict]) -> dict:
    blobs = [s[k] for m in selected for s in m["stages"] for k in ("weights", "bias")]
    contents = {b["sha256"]: b["size"] for b in blobs}
    union_bytes, end = 0, 0
    for start, stop in sorted((b["rva"], b["rva"] + b["size"]) for b in blobs):
        union_bytes += max(0, stop - max(start, end))
        end = max(end, stop)
    return {
        "model_count": len(selected),
        "stage_count": sum(len(m["stages"]) for m in selected),
        "parameter_files": len(blobs),
        "float32_values_per_use": sum(b["size"] // 4 for b in blobs),
        "distinct_parameter_rvas": len({b["rva"] for b in blobs}),
        "distinct_parameter_spans": len({(b["rva"], b["size"]) for b in blobs}),
        "unique_source_bytes": union_bytes,
        "distinct_parameter_contents": len(contents),
        "unique_content_bytes": sum(contents.values()),
    }


def validate_catalog(catalog: dict) -> None:
    if (catalog.get("schema_version") != 3 or
        catalog.get("source_sha256") != SOURCE_SHA256 or
        catalog.get("selector_rva") != SELECTOR_RVA or
        catalog.get("selector_end_rva") != SELECTOR_CODE_END or
        catalog.get("selector_table_end_rva") != SELECTOR_TABLE_END or
        catalog.get("selector_code_sha256") != SELECTOR_CODE_SHA256 or
        catalog.get("selectors") != SELECTORS):
        raise FormatError("unsupported model index")
    expected = expected_model_keys()
    actual = set()
    expected_auxiliary = expected_auxiliary_keys()
    if len(catalog["auxiliary_models"]) != len(expected_auxiliary):
        raise FormatError("incomplete auxiliary selector coverage")
    for m in all_models(catalog):
        key = (m["selector_id"], m["family_id"], m["scale"], m["quality"], m["selector_option"])
        auxiliary = m in catalog["auxiliary_models"]
        if (key in actual or
            (key not in expected_auxiliary if auxiliary else key not in expected)):
            raise FormatError("unknown or duplicate model")
        actual.add(key)
        selector, f, scale, quality, option = key
        if m["id"] != model_id(*key):
            raise FormatError("model id mismatch")
        aliases = list(range(15)) if selector == 0 and f in (28, 32, 36) else [option]
        if m["equivalent_options_0_to_14"] != aliases:
            raise FormatError("incorrect option aliases")
        count = (7 if selector == 2 else (4, 6, 7, 9)[quality] if selector == 1 else
                 9 if auxiliary else (6, 8, 9, 11)[quality] if f == 28 else
                 ((4, 6, 7, 9) if scale == 2 else (8, 10, 10, 12))[quality])
        if len(m["stages"]) != count:
            raise FormatError("stage count mismatch")
        for i, stage in enumerate(m["stages"]):
            ci, co = stage["in_channels"], stage["out_channels"]
            if stage["index"] != i or ci not in (1, 8, 12, 16) or co not in (1, 8, 12, 16):
                raise FormatError("invalid stage channels/index")
            if type(stage["expand_2x"]) is not bool:
                raise FormatError("invalid spatial expansion flag")
            k = parameter_kernel_size(selector, f, co, stage["expand_2x"])
            if stage["kernel_size"] != k or (stage["expand_2x"] and scale != 4):
                raise FormatError("kernel/scale mismatch")
            for label, size in (("weights", ci * co * k * k * 4), ("bias", co * 4)):
                blob = stage[label]
                if (type(blob["rva"]) is not int or blob["rva"] < 0x220000 or
                    blob["size"] != size or len(blob["sha256"]) != 64 or
                    any(c not in "0123456789abcdef" for c in blob["sha256"])):
                    raise FormatError("invalid parameter span")
    if actual != expected | expected_auxiliary:
        raise FormatError("incomplete all-option model set")
    coverage = catalog["coverage"]
    domains = {
        "family_ids": [28, 32, 36, 40], "scale_values": [2, 4],
        "quality_values": [0, 1, 2, 3], "option_values": list(range(15)),
        "layer_values": list(range(32)), "ignored_option_families": [28, 32, 36],
        "sharp_extra_options": list(range(1, 15)),
        "fallback_options_checked": [-2147483648, -1, 15, 2147483647],
        "auxiliary_family_id": 13,
    }
    if any(coverage.get(k) != v for k, v in domains.items()):
        raise FormatError("incomplete selector domain coverage")
    rvas = parameter_rvas(catalog)
    by_selector = coverage["selector_parameter_coverage"]
    if len(by_selector) != len(SELECTORS):
        raise FormatError("incomplete selector coverage")
    for spec, entry in zip(SELECTORS, by_selector):
        found = {s[k]["rva"] for m in all_models(catalog) if m["selector_id"] == spec["id"]
                 for s in m["stages"] for k in ("weights", "bias")}
        if (entry["selector_id"] != spec["id"] or
            entry["static_rva_count"] != spec["literal_parameter_rva_count"] or
            len(found) != entry["static_rva_count"] or
            rva_digest(found) != entry["parameter_rvas_sha256"]):
            raise FormatError("selector parameter address coverage mismatch")
    if (coverage["static_parameter_rva_count"] != len(rvas) or
        coverage["parameter_rvas_sha256"] != rva_digest(rvas) or
        coverage["missing_parameter_rvas"] != [] or
        coverage["unexpected_parameter_rvas"] != []):
        raise FormatError("parameter address coverage mismatch")
    if catalog["summary"] != {
        "ngu": parameter_summary(catalog["models"]),
        "auxiliary": parameter_summary(catalog["auxiliary_models"]),
        "combined": parameter_summary(all_models(catalog)),
    }:
        raise FormatError("parameter summary mismatch")


def load_catalog(path: Path = CATALOG) -> dict:
    try:
        catalog = json.loads(path.read_text(encoding="utf-8"))
        validate_catalog(catalog)
        return catalog
    except (KeyError, TypeError, ValueError, struct.error) as e:
        raise FormatError(f"malformed model index: {e}") from e


def native_records(data: bytes, helper: Path) -> list[tuple[int, ...]]:
    # Check BEFORE spawning any executable; stdin is the checked bytes, not a
    # pathname reopened later. This helper is not a sandbox for arbitrary PEs.
    check_source(data)
    result = subprocess.run([str(helper.resolve()), "--all-parameters"], input=data,
                            capture_output=True, timeout=15, check=False)
    if result.returncode:
        raise FormatError("native selector verifier failed: " +
                          result.stderr.decode("utf-8", errors="replace"))
    try:
        observed = [tuple(map(int, line.split())) for line in result.stdout.decode("ascii").splitlines()]
    except (ValueError, UnicodeDecodeError) as e:
        raise FormatError("invalid native verifier output") from e
    if not observed or any(len(row) != 11 for row in observed):
        raise FormatError("invalid native verifier columns")
    return observed


def verify_native(data: bytes, catalog: dict, helper: Path) -> None:
    check_source(data)
    validate_catalog(catalog)
    if sorted(native_records(data, helper)) != sorted(records(catalog, include_auxiliary=True)):
        raise FormatError("native selector differs from model index")


def parameter_bytes(pe: PE, blob: dict) -> bytes:
    raw = pe.reader.take(pe.file_offset(blob["rva"], blob["size"]), blob["size"])
    if sha256(raw) != blob["sha256"]:
        raise FormatError("parameter hash mismatch")
    if not all(math.isfinite(v[0]) for v in struct.iter_unpack("<f", raw)):
        raise FormatError("non-finite parameter")
    return raw


def extract(source: Path, output: Path, model: str | None = None,
            native_helper: Path | None = None) -> dict:
    # mkdir(parents=True) can create a missing component before processing
    # '..'. Reject traversal and use the checked canonical path for all writes.
    if ".." in output.parts:
        raise FormatError(f"refusing parent traversal ('..') in output: {output}")
    resolved_output = output.resolve()
    # Runtime archives may omit the manifest. Check resolved ancestors too,
    # so an explicit output cannot create a nested archive through an alias.
    for parent in resolved_output.parents:
        if ((parent / "manifest.json").is_file() or
                any(parent.glob("*/stage*.f32le")) or any(parent.glob("*/stage*.f16le"))):
            raise FormatError(f"refusing output inside parameter archive: {parent} (output: {output})")
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise FormatError(f"refusing output through symlink: {output}")
    output = resolved_output
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FormatError(f"output must be a new or empty directory: {output}")
    data = source.read_bytes()
    check_source(data)
    catalog = load_catalog()
    if native_helper is not None:
        verify_native(data, catalog, native_helper)
    selected = [m for m in all_models(catalog) if model is None or m["id"] == model]
    if not selected:
        raise FormatError(f"unknown model id: {model}")
    pe = PE(data)
    # Validate and gather ALL requested blobs before writing anything.
    files = {}
    exported = copy.deepcopy(selected)
    contents = {}
    spans = []
    for m in exported:
        for stage in m["stages"]:
            for label in ("weights", "bias"):
                blob = stage[label]
                path = f'{m["id"]}/stage{stage["index"]:02d}.{label}.f32le'
                files[path] = parameter_bytes(pe, blob)
                blob["exported_file"] = path
                # Preserve every mode's logical files, and explicitly identify
                # shared content without counting aliases as new trained models.
                entry = contents.setdefault(blob["sha256"], {"size": blob["size"], "files": []})
                entry["files"].append(path)
                spans.append((blob["rva"], blob["rva"] + blob["size"], path))
    # Optimized scalar biases may occupy part of another parameter array. Same
    # starting RVA is not a whole-blob alias when lengths differ.
    shared_storage = []
    previous = []
    for start, stop, path in sorted(spans):
        previous = [p for p in previous if p[1] > start]
        for a, b, other in previous:
            shared_storage.append({"files": [other, path], "rva": start,
                                   "size": min(b, stop) - start})
        previous.append((start, stop, path))
    manifest = {
        "schema_version": 3, "source_path": str(source),
        "source_sha256": SOURCE_SHA256, "index_sha256": sha256(CATALOG.read_bytes()),
        "native_selector_verified": native_helper is not None,
        "models": [m for m in exported if m["selector_id"] == 0 and m["family_id"] in FAMILIES],
        "auxiliary_models": [m for m in exported if m["selector_id"] != 0 or m["family_id"] not in FAMILIES],
        "summary": parameter_summary(exported),
        "parameter_contents": contents,
        "shared_parameter_storage": shared_storage,
        "selection": "all_indexed" if model is None else model,
        "index_coverage": copy.deepcopy(catalog["coverage"]),
        "scope": "Index covers all parameter branches of three audited selectors, including Sharp options 1..14. "
                 "Export may be restricted by --model. "
                 "72 auxiliary models are preserved separately, not labelled NGU; their output extents remain "
                 "unaudited. Schema 3 corrects 136 non-AA NGU output heads from 3x3 to 4x4. "
                 "Full graph and pixel equivalence remain unverified.",
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = output / name
        path.parent.mkdir(exist_ok=True)
        with path.open("xb") as file:
            file.write(content)
    with (output / "manifest.json").open("x", encoding="utf-8") as file:
        file.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--weights-root", type=Path, default=ROOT / "weights",
                        help="common root (default: <project>/weights); FP32 goes in its weights/ child")
    parser.add_argument("--output", type=Path,
                        help="explicit FP32 archive directory (overrides <weights-root>/weights)")
    parser.add_argument("--model", help="e.g. sharp_2x_low_option01; default: all 140 NGU variants + 72 auxiliary models")
    parser.add_argument("--verify-native", type=Path, metavar="HELPER",
                        help="OPTIONAL: compare all variants against the audited, input-restricted native selectors")
    args = parser.parse_args(argv)
    output = args.output if args.output is not None else args.weights_root / "weights"
    try:
        manifest = extract(args.source, output, args.model, args.verify_native)
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        print(f"models: {e}", file=sys.stderr)
        return 1
    print(f'{len(manifest["models"])} NGU variants + {len(manifest["auxiliary_models"])} auxiliary, ' +
          f'{manifest["summary"]["stage_count"]} parameter stages exported')
    print(output / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
