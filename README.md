# vs-NGU

A VapourSynth implementation of madVR's NGU (Next Generation Upscaling) 2x upscaler, with a CPU backend and a Vulkan backend (FP32 or FP16).

The plugin is written against **madVR 0.92.17**. It does **not** ship any NGU weights. You extract them yourself from your own copy of `madVR64.ax`. Each parameter file is checked against a SHA-256 digest compiled into the plugin, so only weights extracted from the exact 0.92.17 build will load.

## Documentation

- [Installation](docs/installation.md)
- [Usage](docs/usage.md)
- [mpv gpu-next shaders](docs/mpv-shaders.md)
- [Building](docs/building.md)

## License

The plugin source code is under the MIT license. madVR and the NGU weights belong to their respective owners. This project does not distribute them and grants no rights to them.
