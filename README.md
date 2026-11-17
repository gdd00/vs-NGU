# vs-NGU

A VapourSynth implementation of madVR's NGU (Next Generation Upscaling) 2x upscaler, with a CPU backend and a Vulkan backend (FP32 or FP16).

The plugin is written against **madVR 0.92.17**. It does **not** ship any NGU weights. You extract them yourself from your own copy of `madVR64.ax`. Each parameter file is checked against a SHA-256 digest compiled into the plugin, so only weights extracted from the exact 0.92.17 build will load.

## Weights

Download madVR **0.92.17** from [videohelp.com/software/madVR](https://www.videohelp.com/software/madVR) and unpack it. Then run:

```bash
# 1. Extract the FP32 archive into weights/weights/
python tools/extract_ngu_models.py path/to/madvr/raw/madVR09217/madVR64.ax --weights-root weights

# 2. Convert it to an FP16 archive in weights/weights-fp16/
python tools/convert_ngu_fp16.py --weights-root weights
```

Both tools need only the Python standard library. Extraction reads the PE file offline and never runs code from it. Neither tool overwrites an existing output directory.

You end up with one common root, which is what the `weights` argument expects:

```
weights/
├── weights/        # FP32, used for GRAYS input
└── weights-fp16/   # FP16, used for GRAYH input
```

## Usage

Prototype:

`core.ngu.NGU2x(vnode clip, string weights[, string mode = "sharp", string quality = "medium", int option = 0, string backend = "auto", string weights_precision])`

Output is exactly 2x the input width and height. The plugin adds these frame properties: `_NGUModel`, `_NGUPrecision`, `_NGUWeightPrecision` and `_NGUBackend`.

Arguments:

- `vnode clip`: Input clip. It must have a constant format and constant dimensions, and only **GRAYS** (float32) or **GRAYH** (float16) single-plane clips are supported. The plugin does no colour conversion. Pass in the luma (or any other plane) yourself, normalised to the usual float range.
  - The input sample type chooses the precision:
    - GRAYS uses the FP32 weights from `weights/`.
    - GRAYH uses the FP16 weights from `weights-fp16/`.
  - Where the clip lives chooses the backend:
    - A CPU-resident clip runs on the CPU backend.
    - A GPU-resident clip (from `std.GPUUpload`, VapourSynth R82+) runs on the Vulkan backend.
- `string weights`: Path to the common weights root described above. Required.
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

# CPU, FP32
up = core.ngu.NGU2x(y, weights="path/to/weights", mode="sharp", quality="high")

# Vulkan, FP16 (VapourSynth R82+)
gpu = core.std.GPUUpload(y.resize.Point(format=vs.GRAYH))
up = core.ngu.NGU2x(gpu, weights="path/to/weights", mode="sharp", quality="very_high", option=14)
up = core.std.GPUDownload(up)
```

See also [examples/ngu2x.vpy](examples/ngu2x.vpy).

## Installation

Prebuilt wheels for Windows x64 and Linux x86_64 need Python 3.12+ and VapourSynth R82+. pip installs the plugin into VapourSynth's `plugins` directory inside site-packages, where VapourSynth autoloads it.

Windows and Linux use the same command; pip picks the matching wheel automatically:

```bash
pip install --extra-index-url https://gdd00.github.io/vs-NGU/simple/ vapoursynth-ngu
```

The weights are not included. See [Weights](#weights).

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

The output is `build/native/vsngu.so` on Linux or `build/native/vsngu.dll` on Windows.

CMake options:

- `NGU_NATIVE_ARCH` (default `ON`): adds `-march=native` on GCC/Clang. The binary is then tuned for the build machine and may not run on other CPUs. Turn it off for portable builds; release wheels are built with `OFF`. It does nothing under MSVC.
- `BUILD_TESTING` (default `ON`): builds the test suite. The tests also need the VapourSynth core library and the extracted weights (`-D NGU_WEIGHTS_ROOT=path/to/weights`). Run them with `ctest --test-dir build/native`.
- `NGU_SANITIZE` (default `OFF`): builds with ASan and UBSan.
- `NGU_DUMP_SHADERS` (default `OFF`): enables shader dumps for precision audits, written to the `NGU_SHADER_DUMP_DIR` directory.

To build a wheel locally:

```bash
pip install .
```

## License

The plugin source code is under the MIT license. madVR and the NGU weights belong to their respective owners. This project does not distribute them and grants no rights to them.
