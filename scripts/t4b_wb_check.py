"""T4b white-balance strength check: global colour cast of the app's automatic render vs the reference print.

  t4b_wb_check.py OUT_JSON ROLL_FOLDER REL=REF_JPG [REL=REF_JPG ...]

Each frame is opened as the app opens it (automatic base choice, T4b white point); NOREF_V3 is then estimated with
``wb_v3_strength`` in STRENGTHS (0 = no per-frame correction: only the look's prior) and the preview rendered.
Measured: mean CIELAB a*, b* (D50) over the central 80 % of our render (sRGB ICC of the export) and of the
reference (its embedded ICC), and the distance between the two means (``d_ab``). A global mean ignores framing and
alignment; it measures the overall cast only. References are checked against the frozen sets first.
"""
import json
import sys
from pathlib import Path

import numpy as np

from filmcolor.app import State
from filmcolor.colorimetry import MatrixTRCProfile, xyz_to_lab
from filmcolor.compare import load_lab
from filmcolor.export import icc_profile
from filmcolor.frozen import check_source
from filmcolor.noref import NOREF_V3, estimate_params_v3, estimation_input

STRENGTHS = (0.0, 0.5, 0.85)


def central_mean_ab(lab):
    h, w = lab.shape[:2]
    c = lab[int(0.1 * h): int(0.9 * h), int(0.1 * w): int(0.9 * w)]
    return [float(c[..., 1].mean()), float(c[..., 2].mean())]


def main(out_json: str, folder: str, pairs: list[str]):
    prof = MatrixTRCProfile(icc_profile("srgb"))
    st = State(Path(folder), Path("/tmp/fc_t4b_out"), allow_frozen=False, max_side=1000)
    st.roll_info()
    res = {}
    for pair in pairs:
        rel, ref = pair.split("=", 1)
        check_source(Path(ref))
        ref_ab = central_mean_ab(load_lab(ref, factor=8))
        st.open(rel, "color")
        pv = st.preview
        P4, m4 = estimation_input(pv.full_positive, pv.image_region)
        row = dict(ref=Path(ref).name, ref_ab=ref_ab, strengths={})
        for s in STRENGTHS:
            est = estimate_params_v3(P4, m4, dict(NOREF_V3, wb_v3_strength=s))
            ab = central_mean_ab(xyz_to_lab(prof.to_xyz(pv.render(est["params"]))))
            row["strengths"][str(s)] = dict(temp=est["params"]["temp"], tint=est["params"]["tint"], ab=ab,
                                            d_ab=float(np.hypot(ab[0] - ref_ab[0], ab[1] - ref_ab[1])))
        res[rel] = row
        print(rel, {s: round(v["d_ab"], 2) for s, v in row["strengths"].items()}, flush=True)
    Path(out_json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
