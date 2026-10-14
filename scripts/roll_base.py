"""Roll film base from the per-frame audit of roll_exposure.py.

Runs estimate_base_roll with the EXIF nominal exposure times (used) and with exact 1/3-stop values
(sensitivity), and writes the per-frame base in each frame's raw signal for convert.py --roll-base.
The base is measured on the analysis path (CFA-aware); for Bayer frames it agrees with the render path
to 0.05 % (experiments/own_roll01/exposure/REPORT.md).

  roll_base.py FRAMES_JSON OUT_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np

from filmcolor.base import estimate_base_roll


def main(frames_json, out_json):
    d = json.loads(Path(frames_json).read_text())
    F = d["frames"]
    cand = {n: f["base"]["rgb"] for n, f in F.items()}
    res = {}
    for norm in ("exposure_nominal", "exposure_third_stop"):
        t = {n: f["exif"][norm] for n, f in F.items()}
        res[norm] = (estimate_base_roll(cand, t, t_ref=d["t_ref"]), t)
    used, t = res["exposure_nominal"]
    alt, t_alt = res["exposure_third_stop"]
    frames = {}
    for n in sorted(F):
        own = np.array(cand[n])
        b = np.array(used.for_exposure(t[n]))
        d_own = np.log10(b / own)  # density of the frame's own interior estimate above the roll base
        frames[n] = dict(exposure=t[n], base_rgb=b.tolist(),
                         base_rgb_third_stop=list(alt.for_exposure(t_alt[n])),
                         own_interior_estimate=own.tolist(), own_minus_roll_density=d_own.tolist(),
                         own_colour_error_RG_BG=[float(d_own[0] - d_own[1]), float(d_own[2] - d_own[1])])
    out = dict(source=str(frames_json), method="estimate_base_roll on per-frame interior auto estimates",
               normalisation="exposure_nominal", roll=used.to_dict(), roll_third_stop=alt.to_dict(),
               frames=frames)
    Path(out_json).write_text(json.dumps(out, indent=1))
    print("members", used.members, "used", used.used, "rgb_norm", [round(x, 4) for x in used.rgb_norm])
    print("third-stop members", alt.members == used.members, "rgb_norm", [round(x, 4) for x in alt.rgb_norm])
    print("warnings", used.warnings)
    print("wrote", out_json)


if __name__ == "__main__":
    main(*sys.argv[1:3])
