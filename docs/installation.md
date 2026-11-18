# Installation

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
