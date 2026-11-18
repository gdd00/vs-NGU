"""Install the FP32 and FP16 weights for the VapourSynth NGU plugin."""

from __future__ import annotations

import argparse
from copy import copy
import errno
import hashlib
from http.client import HTTPException
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile
import zlib


DEFAULT_URL = "https://www.videohelp.com/download/madVR09217.zip"
REFERER = "https://www.videohelp.com/software/madVR"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
MAX_ZIP_BYTES = 128 * 1024 * 1024
CHUNK_BYTES = 64 * 1024
ZIP_MAGIC = b"PK\x03\x04"
SOURCE_NAME = "madVR64.ax"
# Same pinned release as tools/extract_ngu_models.py, which rechecks the
# snapshot before parsing the PE. Neither path loads or executes the PE.
SOURCE_SIZE = 10645576
SOURCE_SHA256 = "9f684ba984344d5f578e1590d7d0a62b6b93386598fdfee6b5ae0aa560bf7164"


def default_weights_root() -> Path:
    package_parent = Path(__file__).resolve().parents[1]
    wheel_plugin = package_parent / "vapoursynth/plugins/vsngu"
    if any((wheel_plugin / name).is_file() for name in ("vsngu.so", "vsngu.dll")):
        return wheel_plugin / "weights"
    message = "cannot locate the VapourSynth plugin directory; use --weights-root"
    try:
        from vapoursynth import get_plugin_dir

        plugin_dir = get_plugin_dir()
        if isinstance(plugin_dir, str) and not plugin_dir.strip():
            raise ValueError(message)
        return Path(plugin_dir) / "vsngu/weights"
    except (ImportError, OSError, RuntimeError, TypeError) as exc:
        raise ValueError(message) from exc


def tools_dir() -> Path:
    package = Path(__file__).resolve().parent
    bundled = package / "_bundled"
    # Both locations have the tools/ and models/ relative paths expected
    # by the unmodified standalone scripts.
    base = bundled if bundled.is_dir() else package.parents[1]
    required = (
        "tools/extract_madvr.py",
        "tools/extract_ngu_models.py",
        "tools/convert_ngu_fp16.py",
        "tools/generate_ngu_catalog.py",
        "models/madvr09217-index.json",
    )
    missing = [name for name in required if not (base / name).is_file()]
    if missing:
        raise ValueError(f"missing NGU weight tool resources in {base}: {', '.join(missing)}")
    return base / "tools"


def path_info(path: Path) -> stat.stat_result | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if (stat.S_ISLNK(info.st_mode) or
            getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
        raise ValueError(f"refusing symlink or reparse point in target path: {path}")
    return info


def archive_exists(path: Path) -> bool:
    info = path_info(path)
    if info is None:
        return False
    if stat.S_ISDIR(info.st_mode):
        manifest = path_info(path / "manifest.json")
        if manifest is not None and stat.S_ISREG(manifest.st_mode):
            return True
    raise ValueError(f"refusing to overwrite existing path without a regular manifest.json: {path}")


def inspect_root(root: Path) -> tuple[Path, bool, bool]:
    """Return the closest existing directory and whether each archive exists."""
    parent_dir = Path(root.anchor)
    for path in (*reversed(root.parents), root):
        info = path_info(path)
        if info is None:
            continue
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"target path component is not a directory: {path}")
        if ((path / "manifest.json").is_file() or
                any(path.glob("*/stage*.f32le")) or any(path.glob("*/stage*.f16le"))):
            raise ValueError(f"refusing to write inside a parameter archive: {path}")
        parent_dir = path

    fp32 = archive_exists(root / "weights")
    fp16 = archive_exists(root / "weights-fp16")
    if fp16 and not fp32:
        raise ValueError(f"refusing to overwrite an existing FP16-only archive: {root / 'weights-fp16'}")
    return parent_dir, fp32, fp16


def check_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError(f"expected an http:// or https:// URL: {url!r}")


class HTTPOnlyRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def copy_bounded(stream, path: Path, limit: int, prefix: bytes = b"") -> int:
    total = len(prefix)
    if total > limit:
        raise ValueError(f"{path.name} exceeds the {limit}-byte limit")
    with path.open("xb") as out:
        out.write(prefix)
        while chunk := stream.read(min(CHUNK_BYTES, limit - total + 1)):
            total += len(chunk)
            if total > limit:
                raise ValueError(f"{path.name} exceeds the {limit}-byte limit")
            out.write(chunk)
    return total


def zip_magic(stream) -> bytes:
    if stream.read(len(ZIP_MAGIC)) != ZIP_MAGIC:
        raise ValueError("invalid ZIP magic; expected a madVR ZIP archive")
    return ZIP_MAGIC


def download(url: str, path: Path) -> None:
    check_url(url)
    request = Request(url, headers={"User-Agent": USER_AGENT, "Referer": REFERER})
    with build_opener(HTTPOnlyRedirect()).open(request, timeout=30) as response:
        if response.status != 200:
            raise ValueError(f"unexpected HTTP response status: {response.status}")
        declared = response.headers.get("Content-Length")
        try:
            length = int(declared) if declared is not None else None
        except ValueError as exc:
            raise ValueError(f"invalid HTTP Content-Length: {declared}") from exc
        if length is not None:
            if length > MAX_ZIP_BYTES:
                raise ValueError(f"download exceeds the {MAX_ZIP_BYTES}-byte limit")
            if length < len(ZIP_MAGIC):
                raise ValueError(f"invalid HTTP Content-Length: {declared}")
        total = copy_bounded(response, path, MAX_ZIP_BYTES, zip_magic(response))
        if length is not None and total != length:
            raise ValueError(f"incomplete download: expected {length} bytes, received {total}")


def prepare_source(source: Path, destination: Path) -> None:
    info = source.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"source must be a regular file: {source}")
    if source.suffix.lower() == ".zip":
        if info.st_size > MAX_ZIP_BYTES:
            raise ValueError(f"ZIP exceeds the {MAX_ZIP_BYTES}-byte limit")
        with source.open("rb") as stream:
            zip_magic(stream)
            with zipfile.ZipFile(stream) as archive:
                found = [entry for entry in archive.infolist() if entry.orig_filename == SOURCE_NAME]
                if len(found) != 1:
                    raise ValueError(f"ZIP must contain exactly one root entry named {SOURCE_NAME}")
                entry = found[0]
                if entry.file_size != SOURCE_SIZE:
                    raise ValueError(f"incorrect {SOURCE_NAME} size in ZIP: expected {SOURCE_SIZE} bytes")
                if entry.flag_bits & 1 or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise ValueError(f"unsupported ZIP compression or encryption for {SOURCE_NAME}")
                # zipfile normally truncates the stream at the declared size.
                # Allow one byte of lookahead so copy_bounded detects overruns,
                # without extracting entries by pathname or raising the output cap.
                probe = copy(entry)
                probe.file_size += 1
                try:
                    with archive.open(probe) as ax:
                        size = copy_bounded(ax, destination, SOURCE_SIZE)
                except NotImplementedError as exc:
                    raise ValueError(f"unsupported ZIP member: {exc}") from exc
    elif source.suffix.lower() == ".ax":
        if info.st_size != SOURCE_SIZE:
            raise ValueError(f"incorrect {SOURCE_NAME} size: expected {SOURCE_SIZE} bytes")
        with source.open("rb") as stream:
            size = copy_bounded(stream, destination, SOURCE_SIZE)
    else:
        raise ValueError("--source must be a .zip or .ax file")

    if size != SOURCE_SIZE:
        raise ValueError(f"incorrect {SOURCE_NAME} size: expected {SOURCE_SIZE} bytes, received {size}")
    print(f"Verifying {SOURCE_NAME} SHA-256...", flush=True)
    with destination.open("rb") as ax:
        digest = hashlib.file_digest(ax, "sha256").hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"{SOURCE_NAME} SHA-256 mismatch; expected the exact madVR 0.92.17 binary")


def run_tool(script: Path, *arguments: str | Path) -> None:
    # Use standalone scripts in their own directory/import context. Avoid
    # writes to the installed package, and ignore PYTHONPATH/user site imports.
    result = subprocess.run([sys.executable, "-E", "-s", "-B", str(script), *map(str, arguments)],
                            capture_output=True, text=True)
    if result.returncode:
        reason = result.stderr.strip() or result.stdout.strip() or f"exit status {result.returncode}"
        raise ValueError(f"{script.stem}: {reason}")


def publish(root: Path, staged: list[Path]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    inspect_root(root)
    created = []
    try:
        # mkdir reserves each archive name without replacing even an empty
        # directory that appeared during the download or conversion.
        for archive in staged:
            output = root / archive.name
            output.mkdir()
            created.append(output)
        for archive, output in zip(staged, created):
            for entry in archive.iterdir():
                if entry.name != "manifest.json":
                    entry.rename(output / entry.name)
            (archive / "manifest.json").rename(output / "manifest.json")
    except BaseException:
        for output in reversed(created):
            shutil.rmtree(output)
        raise


def install(root: Path, source: Path | None, url: str) -> None:
    if ".." in root.parts:
        raise ValueError(f"refusing parent traversal ('..') in --weights-root: {root}")
    root = root.absolute()
    print(f"Target weights root: {root}", flush=True)
    parent_dir, fp32, fp16 = inspect_root(root)
    if fp32 and fp16:
        print("FP32 and FP16 archives already exist; nothing to do.")
        return

    scripts = tools_dir()
    # Creating the staging directory is also a real write-permission check,
    # before downloading anything. Stage on the target filesystem.
    with tempfile.TemporaryDirectory(prefix=".vsngu-", dir=parent_dir) as temp:
        work = Path(temp)
        full, half = work / "weights", work / "weights-fp16"
        if fp32:
            full = root / "weights"
            print(f"Using existing FP32 archive: {full}", flush=True)
        else:
            if source is None:
                source = work / "madVR09217.zip"
                print(f"Downloading {url}...", flush=True)
                download(url, source)
            else:
                print(f"Reading {source}...", flush=True)
            pe = work / SOURCE_NAME
            prepare_source(source, pe)
            print("Extracting FP32 weights...", flush=True)
            run_tool(scripts / "extract_ngu_models.py", pe, "--output", full)
        print("Converting FP32 weights to FP16...", flush=True)
        run_tool(scripts / "convert_ngu_fp16.py", "--source", full, "--output", half)
        publish(root, [half] if fp32 else [full, half])
    print(f"Done: FP32 and FP16 weights installed in {root}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vsngu-weights", description=__doc__)
    parser.add_argument("--weights-root", type=Path,
                        help="common root containing weights/ and weights-fp16/ (default: plugin's weights/)")
    parser.add_argument("--source", type=Path,
                        help="local madVR09217.zip or madVR64.ax instead of downloading")
    parser.add_argument("--url", default=DEFAULT_URL, help="download URL (default: %(default)s)")
    args = parser.parse_args(argv)
    try:
        check_url(args.url)
        install(args.weights_root or default_weights_root(), args.source, args.url)
    except (OSError, ValueError, HTTPException, zipfile.BadZipFile, zlib.error) as exc:
        message = str(exc)
        if isinstance(exc, PermissionError) or (isinstance(exc, OSError) and exc.errno == errno.EROFS):
            message = (f"permission denied: {exc}; use --weights-root with a writable directory "
                       "or rerun with elevated permissions")
        print(f"vsngu-weights: {message}", file=sys.stderr)
        return 1
    return 0
