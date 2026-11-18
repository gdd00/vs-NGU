# mpv gpu-next shaders

Generate the 72 native-2× luma shaders from your own extracted archive:

```sh
python3 tools/export_mpv_shaders.py --weights-root ./weights
mpv --vo=gpu-next --gpu-api=vulkan \
  --glsl-shader="$HOME/.config/mpv/shaders/NGU/standard/high.glsl" video.mkv
```

`--weights-root` names the directory containing `weights/` (FP32) and/or
`weights-fp16/` (FP16), not the precision-specific inner directory.
The exporter checks the byte count and SHA-256 against the selected index
before writing shaders. The default is the FP32 archive; `--precision fp16`
uses the indexed binary16 coefficients (with already scaled biases), still
with FP32 accumulators. Use `--model <index-id>` to select one or more models
and `--output <dir>` to put them elsewhere. The generated files **contain
the local coefficients**; do not commit or redistribute them or shader logs.

The default output directory is `~/.config/mpv/shaders/NGU/`. It contains
`aa/`, `soft/`, `standard/` and `sharp/`, each with four quality files.
Sharp's additional 56 variants are in
`sharp/options/<quality>_option01.glsl` through `<quality>_option14.glsl`.
Load **one** NGU file at a time. The exporter doesn't change mpv settings.

Shaders run 5–10 LUMA-only compute passes with clamped shared-memory tiles.
Each source pixel uses a 2×2 block of RGBA texels to hold up to 16 feature
channels in a single texture. Normalized luma is read with `LUMA_mul`;
mpv still handles chroma
sampling and color conversion. Every pass runs only when the target width
**or** height exceeds the original luma dimension by more than two pixels.
AA keeps the raw corner-aligned NGU grid and uses `OFFSET -0.5 -0.5` to
describe it to libplacebo's plane alignment; other modes use zero offset.

By default the shared tiles use FP32, even with FP16 intermediate render
targets. With Vulkan `shaderFloat16` support, `--native-fp16` opts into
`GL_EXT_shader_explicit_arithmetic_types_float16` for the packed shared
tiles. The GLSL fallback handles compilers without the extension; a compiler
advertising it does **not** guarantee the GPU feature, so leave the flag off
on unverified devices.

```sh
python3 tests/mpv_shaders_unit.py
python3 tests/mpv_shaders.py --suite smoke
python3 tests/mpv_shaders.py --suite pixels --model aa_2x_high
```

The smoke suite runs real Vulkan windows (config ignored), checks successful
SPIR-V compilation and dispatch for every model, and checks skip/activation
thresholds. The pixel suite also requires FFmpeg, C++20 and OpenSSL headers;
it recovers the raw Y samples from a calibrated 16-bit neutral-gray
readback from mpv and compares them to the production FP32 CPU oracle.
For AA it separately tests
left-sited 4:2:0 color edges with the real offset and a zero-offset negative
control. Use `--gpu-context waylandvk` if X11/Vulkan isn't available and
`--artifacts /path/outside/repo` to retain private logs and readbacks.
