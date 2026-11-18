# vs-NGU

A VapourSynth implementation of madVR's NGU (Next Generation Upscaling) 2x upscaler, with a CPU backend and a Vulkan backend (FP32 or FP16).

The plugin is written against **madVR 0.92.17**. It does **not** ship any NGU weights. You extract them yourself from your own copy of `madVR64.ax`. Each parameter file is checked against a SHA-256 digest compiled into the plugin, so only weights extracted from the exact 0.92.17 build will load.

## Installation

Prebuilt wheels for Windows x64 and Linux x86_64 need Python 3.12+ and VapourSynth R82+. Windows and Linux use the same command; pip picks the matching wheel automatically:

```bash
pip install --extra-index-url https://gdd00.github.io/vs-NGU/simple/ vapoursynth-ngu
```

pip installs the plugin at `vapoursynth/plugins/vsngu/vsngu.so` (Windows: `vsngu.dll`) inside site-packages, where VapourSynth autoloads it. The wheel also ships the `vsngu-weights` command.

The weights are not included. To omit the `weights` argument in your scripts, they must end up in the `weights/` directory next to the plugin library:

```
vapoursynth/plugins/            # <plugin-dir>
└── vsngu/
    ├── vsngu.so                # Windows: vsngu.dll
    └── weights/                # common weights root (default)
        ├── weights/            # FP32
        └── weights-fp16/       # FP16
```

Note the two levels of `weights/`: `vsngu/weights/weights/` holds the FP32 archive. Locate `<plugin-dir>` with:

```bash
python -c "import vapoursynth as vs; print(vs.get_plugin_dir())"
```

This prints the `vapoursynth/plugins` directory in site-packages; the plugin is in its `vsngu/` subdirectory. You can also keep the weights anywhere else and pass the common root explicitly with `weights=`. To fill this layout, continue with [Weights](#weights).

## Weights

The plugin does not download, extract or convert weights, neither at install time nor at runtime. Prepare them once, either with `vsngu-weights` or with the tools from this repository.

### With `vsngu-weights` (recommended)

`vsngu-weights` (also available as `python -m vsngu`) is pure Python and does not need a clone of this repository:

```bash
vsngu-weights            # or: python -m vsngu
```

It downloads `madVR09217.zip` from VideoHelp, verifies the SHA-256 of the `madVR64.ax` inside, extracts the FP32 archive and converts it to FP16. The result, `weights/` and `weights-fp16/`, goes into the default weights root `<plugin-dir>/vsngu/weights/` (see [Installation](#installation)).

Options:

- `--source path/to/madVR09217.zip`: use a local ZIP (or `madVR64.ax`) instead of downloading.
- `--weights-root DIR`: write to another common weights root.
- `--url URL`: download from another URL (default `https://www.videohelp.com/download/madVR09217.zip`).

Existing archives are never overwritten. If both are already present the command does nothing; if only `weights/` exists it only creates `weights-fp16/`. If the plugin lives in a system directory without write access (such as `/usr/lib/...`), rerun with elevated permissions or pass a writable `--weights-root` and use `weights=` in your script.

### From the repository

Download madVR **0.92.17** from [videohelp.com/software/madVR](https://www.videohelp.com/software/madVR) and unpack it. Then run both tools from this repository with the same `--weights-root`:

```bash
# 1. Extract the FP32 archive into <weights-root>/weights/
python tools/extract_ngu_models.py path/to/madvr/raw/madVR09217/madVR64.ax --weights-root weights

# 2. Convert it to an FP16 archive in <weights-root>/weights-fp16/
python tools/convert_ngu_fp16.py --weights-root weights
```

`--weights-root` defaults to `weights/` in the repository. To install straight into the default plugin location, pass `<plugin-dir>/vsngu/weights` instead, or move the generated `weights/` and `weights-fp16/` directories there afterwards.

Both tools need only the Python standard library. Extraction reads the PE file offline and never runs code from it. Neither tool overwrites an existing output directory.

You end up with one common weights root:

```
weights/            # common weights root: the path you pass as weights=
├── weights/        # FP32, used for GRAYS input
└── weights-fp16/   # FP16, used for GRAYH input
```

The common root is the directory that *contains* `weights/` and `weights-fp16/`. Do not pass either child directory as the root.

## Usage

Prototype:

`core.ngu.NGU2x(vnode clip[, string weights, string mode = "sharp", string quality = "medium", int option = 0, string backend = "auto", string weights_precision])`

Output is exactly 2x the input width and height. The plugin adds these frame properties: `_NGUModel`, `_NGUPrecision`, `_NGUWeightPrecision` and `_NGUBackend`.

Arguments:

- `vnode clip`: Input clip. It must have a constant format and constant dimensions, and only **GRAYS** (float32) or **GRAYH** (float16) single-plane clips are supported. The plugin does no colour conversion. Pass in the luma (or any other plane) yourself, normalised to the usual float range.
  - The input sample type chooses the precision:
    - GRAYS uses the FP32 weights from `weights/`.
    - GRAYH uses the FP16 weights from `weights-fp16/`.
  - Where the clip lives chooses the backend:
    - A CPU-resident clip runs on the CPU backend.
    - A GPU-resident clip (from `std.GPUUpload`, VapourSynth R82+) runs on the Vulkan backend.
- `string weights`: Optional path to the common weights root described above. If omitted, NGU2x reads from the `weights/` directory next to the loaded plugin library (for a pip install, `vapoursynth/plugins/vsngu/weights/`). It does not download, extract, or convert weights.
- `string mode`: NGU variant. One of `"aa"`, `"soft"`, `"standard"`, `"sharp"`. Default `"sharp"`.
- `string quality`: One of `"low"`, `"medium"`, `"high"`, `"very_high"`. Default `"medium"`.
- `int option`: Range `0..14`, default `0`. Selects one of the alternative Sharp parameter sets that madVR uses internally.
  - **Only used when `mode="sharp"`**. madVR's own selector ignores this value for the other modes, so it is forced to `0` there.
- `string backend`: One of `"auto"`, `"cpu"`, `"vulkan"` (case sensitive). Default `"auto"`.
  - `"auto"`: picks the backend from the input clip's residency.
  - `"cpu"`: an assertion that the input is CPU-resident. The plugin never adds an implicit transfer, so a GPU clip is rejected; use `std.GPUDownload` explicitly.
  - `"vulkan"`: an assertion that the input is GPU-resident. A CPU clip is rejected; use `std.GPUUpload` explicitly.
- `string weights_precision`: `"fp32"` or `"fp16"`. Deprecated and only kept for compatibility. It only checks the value against the input format: `"fp16"` alongside GRAYS (or `"fp32"` alongside GRAYH) raises an error. It is best left out.

Precision and backend combinations:

| Input | Backend | Weights        | Compute                                                       |
|-------|---------|----------------|---------------------------------------------------------------|
| GRAYS | cpu     | `weights/`     | FP32                                                          |
| GRAYH | cpu     | `weights-fp16/`| FP16 weights; samples converted to float, written back as FP16 |
| GRAYS | vulkan  | `weights/`     | FP32 shaders                                                  |
| GRAYH | vulkan  | `weights-fp16/`| Native FP16 shaders                                           |

Vulkan requirements:

- **FP32 path**: the device must support FP32 signed-zero/Inf/NaN preservation and RTE rounding.
- **FP16 path**: the device must support `shaderFloat16`, `storageBuffer16BitAccess`, and FP16 signed-zero/Inf/NaN preservation, RTE rounding and denorm preservation.

Example:

```python
import vapoursynth as vs
core = vs.core

y = core.std.ShufflePlanes(src, 0, vs.GRAY).resize.Point(format=vs.GRAYS)

# CPU, FP32, weights from the default root vapoursynth/plugins/vsngu/weights/
up = core.ngu.NGU2x(y, mode="sharp", quality="high")

# Pass weights="path/to/weights" (the common root) to override the default root
up = core.ngu.NGU2x(y, weights="path/to/weights", mode="sharp", quality="high")

# Vulkan, FP16 (VapourSynth R82+), weights from the default root
gpu = core.std.GPUUpload(y.resize.Point(format=vs.GRAYH))
up = core.ngu.NGU2x(gpu, mode="sharp", quality="very_high", option=14)
up = core.std.GPUDownload(up)
```

See also [examples/ngu2x.vpy](examples/ngu2x.vpy).

## Building

Requirements:

- CMake 3.20+ and a C++20 compiler
- VapourSynth R82+ headers, found through pkg-config or the `vapoursynth` Python package
- OpenSSL (libcrypto). Linux only; Windows uses BCrypt.
- Vulkan 1.4 headers. If they are missing, CMake fetches them automatically.

```bash
cmake -S . -B build/native -G Ninja \
    -D CMAKE_BUILD_TYPE=Release \
    -D NGU_NATIVE_ARCH=ON \
    -D BUILD_TESTING=OFF
cmake --build build/native
cmake --install build/native
```

The build output is `build/native/vsngu.so` on Linux or `build/native/vsngu.dll` on Windows. Native installation uses `${CMAKE_INSTALL_LIBDIR}/vapoursynth/vsngu/` for library artifacts and `${CMAKE_INSTALL_BINDIR}/vapoursynth/vsngu/` for runtime artifacts. The default weights root is still the `weights/` directory next to the installed library.

To build a wheel locally:

```bash
pip install .
```

## License

The plugin source code is under the MIT license. madVR and the NGU weights belong to their respective owners. This project does not distribute them and grants no rights to them.
