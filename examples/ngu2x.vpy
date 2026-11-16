"""Runnable CPU GRAYS / GRAYH NGU2x example; replace the demonstration source.

vspipe --arg mode=sharp --arg quality=very_high --arg option=14 examples/ngu2x.vpy --
FP16: vspipe --arg bits=16 examples/ngu2x.vpy --
weights=<common-root> contains weights/ (GRAYS) and weights-fp16/ (GRAYH).
No source conversion or generic scaler is used by NGU2x.
"""
from pathlib import Path
import vapoursynth as vs

script = Path(__file__)
# VSScript can chdir to the script directory while retaining a relative __file__.
if not script.is_absolute() and not script.is_file():
    script = Path(script.name)
root = script.resolve().parents[1]

core = vs.core
plugin_path = str(globals().get("plugin", root/"build/native/vsngu.so"))
weights_path = str(globals().get("weights", root/"weights"))
mode_name = str(globals().get("mode", "sharp"))
quality_name = str(globals().get("quality", "medium"))
option_number = int(globals().get("option", "0"))
sample_bits = int(globals().get("bits", "32"))
if sample_bits not in (16, 32):
    raise ValueError("bits must be 16 or 32")
# Legacy argument is only an assertion; the plugin rejects a format mismatch.
legacy_options = {}
if "weights_precision" in globals():
    legacy_options["weights_precision"] = str(globals()["weights_precision"])
if not any(p.namespace == "ngu" for p in core.plugins()):
    core.std.LoadPlugin(path=plugin_path)

# Demonstration source. For real processing supply a constant CPU GRAYS / GRAYH
# node here. Input normalisation / decoding / colour conversion is the caller's.
source = core.std.BlankClip(width=64, height=36, format=vs.GRAYH if sample_bits == 16 else vs.GRAYS,
                            length=10, color=.375)
result = core.ngu.NGU2x(source, weights=weights_path, mode=mode_name,
                        quality=quality_name, option=option_number, **legacy_options)
result.set_output()
