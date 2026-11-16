#!/usr/bin/env python3
"""Offline shader extraction for the user-supplied madVR 0.92.17 binaries.

Does not load or execute the PE. Extracted code is third-party material; this
tool does not grant permission to use or redistribute it. Standard library only.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import zlib

MAX_INFLATED = 64 * 1024 * 1024


class FormatError(ValueError):
    pass


class Reader:
    def __init__(self, data: bytes):
        self.data = data

    def take(self, offset: int, size: int) -> bytes:
        if offset < 0 or size < 0 or offset > len(self.data) - size:
            raise FormatError(f"out-of-bounds read: {offset:#x}+{size:#x}")
        return self.data[offset:offset + size]

    def unpack(self, fmt: str, offset: int) -> tuple:
        return struct.unpack(fmt, self.take(offset, struct.calcsize(fmt)))

    def u32(self, offset: int) -> int:
        return self.unpack("<I", offset)[0]

    def string(self, offset: int) -> str:
        self.take(offset, 1)
        end = self.data.find(b"\0", offset)
        if end < 0:
            raise FormatError("unterminated string")
        try:
            return self.data[offset:end].decode("ascii")
        except UnicodeDecodeError as e:
            raise FormatError("non-ASCII string") from e


@dataclass(frozen=True)
class Resource:
    path: tuple[str | int, ...]
    rva: int
    offset: int
    data: bytes


class PE:
    def __init__(self, data: bytes):
        self.reader = r = Reader(data)
        if r.take(0, 2) != b"MZ":
            raise FormatError("not a PE: missing MZ")
        pe = r.u32(0x3c)
        if r.take(pe, 4) != b"PE\0\0":
            raise FormatError("missing PE signature")
        self.machine, count = r.unpack("<HH", pe + 4)
        opt_size = r.unpack("<H", pe + 20)[0]
        opt = pe + 24
        magic = r.unpack("<H", opt)[0]
        if magic not in (0x10b, 0x20b):
            raise FormatError("unsupported optional header")
        dd = 96 if magic == 0x10b else 112
        if opt_size < dd + 24 or r.u32(opt + dd - 4) < 3:
            raise FormatError("missing resource data directory")
        self.headers_size = r.u32(opt + 60)
        self.resource_rva, self.resource_size = r.unpack("<II", opt + dd + 16)
        if not self.resource_rva or not self.resource_size:
            raise FormatError("no resources")
        r.take(opt, opt_size)
        self.sections = []
        for i in range(count):
            sec = opt + opt_size + i * 40
            name, virtual_size, rva, raw_size, raw = r.unpack("<8sIIII", sec)
            r.take(sec, 40)
            r.take(raw, raw_size)
            self.sections.append((name.rstrip(b"\0"), rva, virtual_size, raw, raw_size))

    def file_offset(self, rva: int, size: int) -> int:
        if rva < 0 or size < 0:
            raise FormatError("negative RVA or size")
        if rva < self.headers_size and size <= self.headers_size - rva:
            self.reader.take(rva, size)
            return rva
        matches = []
        for _, start, _, raw, raw_size in self.sections:
            delta = rva - start
            if 0 <= delta < raw_size and size <= raw_size - delta:
                matches.append(raw + delta)
        if len(matches) != 1:
            raise FormatError(f"unmapped or ambiguous file-backed RVA: {rva:#x}")
        return matches[0]

    def resources(self) -> list[Resource]:
        base = self.file_offset(self.resource_rva, self.resource_size)
        tree = Reader(self.reader.take(base, self.resource_size))
        found: list[Resource] = []
        paths = set()

        def name(value: int) -> str | int:
            if not value & 0x80000000:
                return value
            pos = value & 0x7fffffff
            length = tree.unpack("<H", pos)[0]
            try:
                return tree.take(pos + 2, 2 * length).decode("utf-16le")
            except UnicodeDecodeError as e:
                raise FormatError("invalid resource name") from e

        def walk(pos: int, path: tuple, ancestors: tuple):
            if pos in ancestors or len(path) > 2:
                raise FormatError("cyclic or over-deep resource directory")
            tree.take(pos, 16)
            named, ids = tree.unpack("<HH", pos + 12)
            tree.take(pos + 16, (named + ids) * 8)
            for i in range(named + ids):
                key, value = tree.unpack("<II", pos + 16 + i * 8)
                next_path = path + (name(key),)
                target = value & 0x7fffffff
                if value & 0x80000000:
                    walk(target, next_path, ancestors + (pos,))
                else:
                    if len(next_path) != 3 or next_path in paths:
                        raise FormatError("invalid or duplicate resource leaf")
                    rva, size, _, _ = tree.unpack("<IIII", target)
                    offset = self.file_offset(rva, size)
                    found.append(Resource(next_path, rva, offset,
                                          self.reader.take(offset, size)))
                    paths.add(next_path)

        walk(0, (), ())
        return found


def unique_offset(data: bytes, signature: bytes) -> int:
    pos = data.find(signature)
    if pos < 0 or data.find(signature, pos + 1) >= 0:
        raise FormatError("missing or ambiguous cipher table signature")
    return pos


class ShaderCipher:
    """The binary's Blowfish variant, NOT interoperable with standard Blowfish.

    Its F function indexes S0 with the LOW byte (and S3 with the HIGH byte).
    Blocks are two LE uint32s, CBC with a zero IV, no padding. A final partial
    block is unchanged. Tables are read from the input, not bundled here.
    """
    def __init__(self, p: list[int], s: list[int], key: bytes):
        if len(p) != 18 or len(s) != 1024 or not key or any(x >= 128 for x in key):
            raise FormatError("invalid cipher tables or unsupported non-ASCII key")
        self.p, self.s = p.copy(), s.copy()
        for i in range(18):
            word = bytes(key[(4 * i + j) % len(key)] for j in range(4))
            self.p[i] ^= int.from_bytes(word, "big")
        left = right = 0
        for table in (self.p, self.s):
            for i in range(0, len(table), 2):
                left, right = self.block(left, right, self.p)
                table[i:i + 2] = left, right

    @classmethod
    def from_binary(cls, data: bytes) -> tuple[ShaderCipher, dict]:
        p_off = unique_offset(data, struct.pack("<4I", 0x243f6a88, 0x85a308d3,
                                               0x13198a2e, 0x03707344))
        s_off = unique_offset(data, struct.pack("<4I", 0xd1310ba6, 0x98dfb5ac,
                                               0x2ffd72db, 0xd01adfb7))
        r = Reader(data)
        cipher = cls(list(r.unpack("<18I", p_off)),
                     list(r.unpack("<1024I", s_off)), b"ShaderPwd")
        return cipher, {"p_table_file_offset": p_off, "s_table_file_offset": s_off}

    def f(self, x: int) -> int:
        s = self.s
        return ((((s[x & 255] + s[256 + ((x >> 8) & 255)]) & 0xffffffff)
                 ^ s[512 + ((x >> 16) & 255)]) + s[768 + (x >> 24)]) & 0xffffffff

    def block(self, left: int, right: int, subkeys: list[int]) -> tuple[int, int]:
        for k in subkeys[:16]:
            left ^= k
            right ^= self.f(left)
            left, right = right, left
        left, right = right, left
        return left ^ subkeys[17], right ^ subkeys[16]

    def decrypt(self, data: bytes) -> bytes:
        out = bytearray(data)
        prev_left = prev_right = 0
        keys = self.p[::-1]
        for i in range(0, len(data) // 8 * 8, 8):
            a, b = struct.unpack_from("<II", data, i)
            left, right = self.block(a, b, keys)
            struct.pack_into("<II", out, i, left ^ prev_left, right ^ prev_right)
            prev_left, prev_right = a, b
        return bytes(out)


def inflate(data: bytes, limit: int = MAX_INFLATED) -> bytes:
    try:
        decoder = zlib.decompressobj()
        out = decoder.decompress(data, limit + 1)
        if len(out) > limit or decoder.unconsumed_tail:
            raise FormatError("inflated shader pack exceeds limit")
        if not decoder.eof or decoder.unused_data:
            raise FormatError("truncated zlib stream or trailing data")
        return out
    except zlib.error as e:
        raise FormatError(f"shader pack zlib checksum/format error: {e}") from e


@dataclass(frozen=True)
class Shader:
    name: str
    offset: int  # relative to the (decompressed, for SHADER) resource pack
    source_crc: int  # CRC of HLSL SOURCE, not of the bytecode!
    code: bytes


def parse_pack(data: bytes) -> list[Shader]:
    r = Reader(data)
    pos = 0
    shaders = []
    names = set()
    while pos < len(data):
        name_size = r.u32(pos)
        pos += 4
        if not 2 <= name_size <= 1024:
            raise FormatError("invalid shader name length")
        raw_name = r.take(pos, name_size)
        pos += name_size
        if raw_name[-1] != 0 or b"\0" in raw_name[:-1]:
            raise FormatError("invalid NUL-terminated shader name")
        try:
            name = raw_name[:-1].decode("ascii")
        except UnicodeDecodeError as e:
            raise FormatError("non-ASCII shader name") from e
        if not re.fullmatch(r"[A-Za-z0-9_\[\].,-]+", name) or name in names:
            raise FormatError(f"unsafe or duplicate shader name: {name!r}")
        crc, size = r.unpack("<II", pos)
        pos += 8
        if not size:
            raise FormatError("empty shader")
        code = r.take(pos, size)
        shaders.append(Shader(name, pos, crc, code))
        names.add(name)
        pos += size
    if not shaders:
        raise FormatError("empty shader pack")
    return shaders


def parse_ctab(data: bytes) -> dict:
    r = Reader(data)
    size, creator, version, count, info, flags, target = r.unpack("<7I", 0)
    if size != 28:
        raise FormatError("unsupported CTAB header")
    r.take(info, count * 20)
    constants = []
    for i in range(count):
        name, reg_set, reg_index, reg_count, _, type_pos, default = r.unpack(
            "<IHHHHII", info + i * 20)
        cls, kind, rows, cols, elements, members, member_info = r.unpack("<6HI", type_pos)
        constants.append({
            "name": r.string(name), "register_set": reg_set,
            "register_index": reg_index, "register_count": reg_count,
            "type": {"class": cls, "kind": kind, "rows": rows, "columns": cols,
                     "elements": elements, "struct_members": members,
                     "member_info_offset": member_info},
            "default_offset": default,
        })
    return {"creator": r.string(creator), "target": r.string(target),
            "version_token": version, "flags": flags, "constants": constants}


def legacy_metadata(code: bytes) -> dict:
    r = Reader(code)
    if len(code) % 4 or r.u32(0) not in (0xffff0300, 0xfffe0300):
        raise FormatError("expected shader model 3 bytecode")
    pos = 4
    opcodes: Counter = Counter()
    tables = []
    while pos < len(code):
        token = r.u32(pos)
        op = token & 0xffff
        if op == 0xffff:
            if token != 0x0000ffff or pos + 4 != len(code):
                raise FormatError("invalid END or trailing shader bytes")
            return {"model": "ps_3_0" if r.u32(0) >> 16 == 0xffff else "vs_3_0",
                    "instruction_count": sum(opcodes.values()),
                    "opcodes": {f"0x{k:04x}": v for k, v in sorted(opcodes.items())},
                    "constant_tables": tables}
        if op == 0xfffe:
            payload = r.take(pos + 4, ((token >> 16) & 0x7fff) * 4)
            if payload.startswith(b"CTAB"):
                tables.append(parse_ctab(payload[4:]))
            pos += 4 + len(payload)
        else:
            length = (token >> 24) & 15
            r.take(pos, (length + 1) * 4)
            opcodes[op] += 1
            pos += (length + 1) * 4
    raise FormatError("missing shader END")


def dxbc_metadata(code: bytes) -> dict:
    r = Reader(code)
    if r.take(0, 4) != b"DXBC":
        raise FormatError("missing DXBC header")
    version, size, count = r.unpack("<III", 20)
    if version != 1 or size != len(code):
        raise FormatError("invalid DXBC size/version")
    r.take(32, count * 4)
    chunks = []
    model = None
    for i in range(count):
        offset = r.u32(32 + i * 4)
        if offset < 32 + count * 4:
            raise FormatError("DXBC chunk overlaps header")
        try:
            fourcc = r.take(offset, 4).decode("ascii")
        except UnicodeDecodeError as e:
            raise FormatError("non-ASCII DXBC chunk identifier") from e
        size = r.u32(offset + 4)
        payload = r.take(offset + 8, size)
        for previous in chunks:
            if offset < previous["offset"] + previous["size"] + 8 and previous["offset"] < offset + size + 8:
                raise FormatError("overlapping DXBC chunks")
        chunks.append({"fourcc": fourcc, "offset": offset, "size": size})
        if fourcc in ("SHDR", "SHEX"):
            cr = Reader(payload)
            v, words = cr.unpack("<II", 0)
            if words * 4 != size:
                raise FormatError("invalid DXBC token count")
            stage = {0: "ps", 1: "vs", 5: "cs"}.get(v >> 16, str(v >> 16))
            model = f"{stage}_{(v >> 4) & 15}_{v & 15}"
    if model is None:
        raise FormatError("no DXBC shader code")
    return {"model": model, "chunks": chunks}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def analyze(data: bytes) -> tuple[dict, dict[str, list[Shader]]]:
    pe = PE(data)
    resources = pe.resources()
    cipher, tables = ShaderCipher.from_binary(data)
    packs = {}
    resource_info = {}
    for kind in ("SHADER", "COMPUTE"):
        matches = [res for res in resources if res.path[:2] == (kind, "DATA")]
        if len(matches) != 1:
            raise FormatError(f"expected one {kind}/DATA resource")
        resource = matches[0]
        decoded = inflate(cipher.decrypt(resource.data)) if kind == "SHADER" else resource.data
        packs[kind] = parse_pack(decoded)
        resource_info[kind] = {
            "path": list(resource.path), "file_offset": resource.offset,
            "rva": resource.rva, "stored_size": len(resource.data),
            "stored_sha256": sha256(resource.data), "decoded_size": len(decoded),
            "decoded_sha256": sha256(decoded),
        }
    entries = []
    for kind, shaders in packs.items():
        for shader in shaders:
            metadata = legacy_metadata(shader.code) if kind == "SHADER" else dxbc_metadata(shader.code)
            entries.append({
                "pack": kind, "name": shader.name, "pack_offset": shader.offset,
                "size": len(shader.code), "sha256": sha256(shader.code),
                "hlsl_source_crc32": f"{shader.source_crc:08x}", **metadata,
            })
    manifest = {
        "schema_version": 1, "source_sha256": sha256(data), "source_size": len(data),
        "machine": f"0x{pe.machine:04x}", "cipher_tables": tables,
        "resources": resource_info, "shader_count": len(entries), "shaders": entries,
        "scope": "Shader extraction only. Network weights, host graph and NGU equivalence NOT verified.",
    }
    return manifest, packs


def extract(source: Path, output: Path, all_shaders: bool = False) -> dict:
    # Analyze everything before creating output; don't leave a green partial manifest.
    data = source.read_bytes()
    manifest, packs = analyze(data)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FormatError("output must be a new or empty directory")
    if sum(s.name == "NGUPass" for s in packs["SHADER"]) != 1:
        raise FormatError("NGUPass not found uniquely")
    output.mkdir(parents=True, exist_ok=True)
    for entry, shader in zip(manifest["shaders"],
                             [s for shaders in packs.values() for s in shaders]):
        if all_shaders or shader.name == "NGUPass":
            path = f'{entry["pack"].lower()}/{shader.name}.bin'
            target = output / path
            target.parent.mkdir(exist_ok=True)
            target.write_bytes(shader.code)
            entry["exported_file"] = path
    manifest["source_path"] = str(source)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="madVR .ax PE file")
    parser.add_argument("--output", required=True, type=Path, help="new or empty output directory")
    parser.add_argument("--all", action="store_true", help="export all shaders (default: NGUPass only)")
    args = parser.parse_args()
    try:
        manifest = extract(args.source, args.output, args.all)
    except (OSError, FormatError) as e:
        print(f"extract: {e}", file=sys.stderr)
        return 1
    ngu = next(x for x in manifest["shaders"] if x["name"] == "NGUPass")
    print(f'{manifest["shader_count"]} shaders catalogued; NGUPass {ngu["size"]} bytes, sha256 {ngu["sha256"]}')
    print(args.output / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
