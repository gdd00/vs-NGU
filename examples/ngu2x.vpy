"""Runnable CPU GRAYS / GRAYH NGU2x example; replace the demonstration source.

vspipe --arg mode=sharp --arg quality=very_high --arg option=14 examples/ngu2x.vpy --
FP16: vspipe --arg bits=16 examples/ngu2x.vpy --
weights=<common-root> contains weights/ (GRAYS) and weights-fp16/ (GRAYH).
No source conversion or generic scaler is used by NGU2x.

AA downscale for encoding (128x72 to 64x36):
vspipe --arg mode=aa --arg target_width=64 --arg target_height=36 examples/ngu2x.vpy --
Use - instead of -- to pipe raw frames to an encoder.

Only mode="aa" shifts the image up and left by 0.5 pixels (-0.5 on both
axes of the 2x output grid). The soft, standard and sharp modes are centered.
No correction is needed for real-time playback. For encoding, pass
src_left=-0.5 and src_top=-0.5 to the downscaler in the same call as the
target size, as in nnedi3_resample, rather than shifting in a separate pass.
Omit both target arguments to keep the original 2x output.
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
target_w = globals().get("target_width")
target_h = globals().get("target_height")
if (target_w is None) != (target_h is None):
    raise ValueError("target_width and target_height must be set together")
if target_w is not None:
    target_w, target_h = int(target_w), int(target_h)
    if target_w <= 0 or target_h <= 0:
        raise ValueError("target_width and target_height must be positive")
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

if target_w is not None:
    # Compensate in NGU2x output pixels while downscaling, in one pass.
    aa_shift = -0.5 if mode_name == "aa" else 0.0
    # Equivalent geometry for AA with fmtconv (on a supported input format):
    # result = core.fmtc.resample(result, w=target_w, h=target_h,
    #                             sx=-0.5, sy=-0.5, kernel="spline36")
    result = core.resize.Spline36(result, width=target_w, height=target_h,
                                  src_left=aa_shift, src_top=aa_shift)
result.set_output()
