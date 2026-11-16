#!/usr/bin/env python3
"""Convert the verified full NGU archive to real little-endian binary16 files.

Weights are RNE binary16; biases are scaled raw/255 in FP32 before RNE binary16.
Only metadata/digests may be redistributed. Parameter files remain user-supplied.
No source files are modified; an existing output directory is never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

from generate_ngu_catalog import INDEX_SHA256

ROOT = Path(__file__).resolve().parents[1]
BIAS_TRANSFORM = "fp32(raw/255) then fp16-rne"


def encode_parameter(data: bytes, kind: str) -> bytes:
    if kind not in ("weights", "bias") or not data or len(data) % 4:
        raise ValueError("invalid FP32 parameter extent/type")
    values = struct.unpack(f"<{len(data) // 4}f", data)
    if not all(map(math.isfinite, values)):
        raise ValueError("non-finite FP32 parameter")
    if kind == "bias":
        values = [struct.unpack("<f", struct.pack("<f", v / 255.0))[0] for v in values]
    try:
        return struct.pack(f"<{len(values)}e", *values)
    except (OverflowError, struct.error) as exc:
        raise ValueError("FP16 parameter overflow") from exc


def convert_archive(source: Path, output: Path, index: Path = ROOT / "models/madvr09217-index.json",
                    metadata_output: Path | None = None) -> dict:
    # Reject traversal before mkdir can create an intermediate directory.
    # Keep the checked canonical paths for both archive and metadata writes,
    # but inspect original symlinks too (including dangling ones).
    destinations = [output] if metadata_output is None else [output, metadata_output]
    resolved_destinations = []
    for destination in destinations:
        if ".." in destination.parts:
            raise ValueError(f"refusing parent traversal ('..') in output: {destination}")
        resolved = destination.resolve()
        for parent in resolved.parents:
            if ((parent / "manifest.json").is_file() or
                    any(parent.glob("*/stage*.f32le")) or any(parent.glob("*/stage*.f16le"))):
                raise ValueError(f"refusing output inside parameter archive: {parent} (output: {destination})")
        if any(path.is_symlink() for path in (destination, *destination.parents)):
            raise ValueError(f"refusing output through symlink: {destination}")
        resolved_destinations.append(resolved)
    source, output = source.resolve(), resolved_destinations[0]
    if metadata_output is not None:
        metadata_output = resolved_destinations[1]
        if (metadata_output == output or output in metadata_output.parents or
                metadata_output in output.parents):
            raise ValueError("metadata output must be separate from archive output")
    if output == source or source in output.parents:
        raise ValueError("output must be separate from the original archive")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output: {output}")
    if metadata_output is not None and metadata_output.exists():
        raise FileExistsError(f"refusing to overwrite metadata: {metadata_output}")
    if not source.is_dir():
        raise ValueError(f"missing FP32 archive: {source}; expected <weights-root>/weights")
    index_bytes = index.read_bytes()
    if hashlib.sha256(index_bytes).hexdigest() != INDEX_SHA256:
        raise ValueError("requires the verified schema-3 source index")
    catalog = json.loads(index_bytes)
    staged = {}
    converted_models = []
    source_bytes = 0
    for model in catalog["models"] + catalog["auxiliary_models"]:
        stages = []
        for stage in model["stages"]:
            entry = {"index": stage["index"], "in_channels": stage["in_channels"],
                     "out_channels": stage["out_channels"], "kernel_size": stage["kernel_size"]}
            for kind in ("weights", "bias"):
                name = f'{model["id"]}/stage{stage["index"]:02d}.{kind}'
                original = (source / (name + ".f32le")).read_bytes()
                spec = stage[kind]
                if len(original) != spec["size"] or hashlib.sha256(original).hexdigest() != spec["sha256"]:
                    raise ValueError(f"source parameter size/SHA-256 mismatch: {name}")
                converted = encode_parameter(original, kind)
                path = name + ".f16le"
                staged[path] = converted
                source_bytes += len(original)
                entry[kind] = {"path": path, "size": len(converted),
                               "sha256": hashlib.sha256(converted).hexdigest(),
                               "source_sha256": spec["sha256"]}
            stages.append(entry)
        converted_models.append({"id": model["id"], "stages": stages})
    metadata = {
        "schema_version": 1, "dtype": "ieee754-binary16-le",
        "source_index_sha256": INDEX_SHA256, "rounding": "nearest-even",
        "bias_transform": BIAS_TRANSFORM,
        "summary": {"models": len(converted_models), "parameter_files": len(staged),
                    "source_parameter_bytes": source_bytes,
                    "parameter_bytes": sum(map(len, staged.values()))},
        "models": converted_models,
    }
    # All originals have been verified and all values converted before writing.
    encoded_metadata = json.dumps(metadata, indent=2) + "\n"
    output.mkdir(parents=True)
    for name, data in staged.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            file.write(data)
    with (output / "manifest.json").open("x", encoding="utf-8") as file:
        file.write(encoded_metadata)
    if metadata_output is not None:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        with metadata_output.open("x", encoding="utf-8") as file:
            file.write(encoded_metadata)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, default=ROOT / "weights",
                        help="common root (default: <project>/weights)")
    parser.add_argument("--source", type=Path,
                        help="explicit FP32 archive (default: <weights-root>/weights)")
    parser.add_argument("--output", type=Path,
                        help="explicit FP16 archive (default: <weights-root>/weights-fp16)")
    parser.add_argument("--index", type=Path, default=ROOT / "models/madvr09217-index.json")
    parser.add_argument("--metadata-output", type=Path,
                        help="optional metadata copy; must be separate from the archive output")
    args = parser.parse_args(argv)
    source = args.source if args.source is not None else args.weights_root / "weights"
    output = args.output if args.output is not None else args.weights_root / "weights-fp16"
    try:
        metadata = convert_archive(source, output, args.index, args.metadata_output)
    except (OSError, ValueError) as exc:
        print(f"fp16: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"output": str(output.resolve()), **metadata["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
