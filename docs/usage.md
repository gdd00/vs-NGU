# Usage

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
- `string weights`: Optional path to the common weights root described in [Weights](installation.md#weights). If omitted, NGU2x reads from the `weights/` directory next to the loaded plugin library (for a pip install, `vapoursynth/plugins/vsngu/weights/`). It does not download, extract, or convert weights.
- `string mode`: NGU variant. One of `"aa"`, `"soft"`, `"standard"`, `"sharp"`. Default `"sharp"`.
  - `"aa"` output is shifted by half an output pixel; the other modes are center-aligned. See [`aa` mode alignment](#aa-mode-alignment).
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

# Encoding with mode="aa": compensate the half-pixel shift in the same
# resize that brings the 2x output to the target size (here 1920x1080)
up = core.ngu.NGU2x(y, mode="aa", quality="high")
out = core.resize.Spline36(up, 1920, 1080, src_left=-0.5, src_top=-0.5)
# or, with fmtconv:
out = core.fmtc.resample(up, 1920, 1080, sx=-0.5, sy=-0.5, kernel="spline36")

# With an extra crop offset given in input (1x) coordinates:
# scale it to the 2x grid, then add the -0.5 correction
src_left, src_top = 1.0, 0.5
out = core.resize.Spline36(up, 1920, 1080, src_left=src_left * 2 - 0.5, src_top=src_top * 2 - 0.5)
```

See also [examples/ngu2x.vpy](../examples/ngu2x.vpy).

## `aa` mode alignment

Only `mode="aa"` has a sub-pixel offset. Its 2x output grid is corner-aligned (top-left): every source pixel `(x, y)` is copied to output `(2x, 2y)`, so the whole image is shifted up and to the left by **0.5 output pixels** (0.25 input pixels). `"soft"`, `"standard"` and `"sharp"` are strictly center-aligned, with zero offset.

- **Real-time playback**: not affected; no compensation is usually needed.
- **Encoding**: when you resize the 2x output to the target resolution (downscaling back, supersampling, or scaling to any other size), **always add the offset to that resize filter**. As in `nnedi3_resample`, fold the center correction into the same resampling pass instead of adding a separate shift filter, which would interpolate twice and add blur. If you keep the 2x output without a later resize and need strict center alignment, correct it with `core.resize.Bicubic(up, src_left=-0.5, src_top=-0.5)` (or `Spline36`).

Offset parameters for the resize that follows `NGU2x(..., mode="aa")`:

| Filter | Parameters |
|--------|------------|
| `core.resize.*` (`Spline36`, `Bicubic`, `Lanczos`, ...) | `src_left=-0.5, src_top=-0.5` |
| `core.fmtc.resample` (as used by `nnedi3_resample`) | `sx=-0.5, sy=-0.5` |

If you also crop or shift by `src_left` / `src_top` in **input** (1x) coordinates, convert them to the 2x grid and then add the correction: use `src_left * 2 - 0.5` and `src_top * 2 - 0.5` (`sx` / `sy` for `fmtc.resample`). Any `src_width` / `src_height` must likewise be doubled.
