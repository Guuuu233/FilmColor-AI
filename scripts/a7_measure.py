"""A7R5 rebate frames, step 2: measured film base vs the engine's estimates (read-only, exploratory).

On the 1/8 linear caches of a7_inspect.py (same decode as the engine's render path):
  backlight  pixels at >= 95 % of the frame's maximum G: sprocket holes and film-free edges (no pixel saturates)
  rebate     hand-placed rectangles in the unexposed gaps between frames (REBATE, % of width / height, chosen by
             eye on the grid previews before any number below was computed); per rectangle the median and the
             coefficient of variation, and left vs right gap where both exist
  measured base density  D_meas = -log10(rebate / backlight) per channel (base + fog of the processed film)
Engine estimates in the same units: the per-frame automatic base of pipeline.positive (interior search, i.e.
what a scan without rebate would give) and estimate_base_roll over the six frames (one session, identical
exposure 1/40 s f/9 ISO 100, so no exposure normalisation). Densities relative to the backlight:
D = -log10(base / backlight); error = D_est - D_meas (+ = estimate denser than the film base).

  a7_measure.py OUT_JSON ARW [ARW ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

from filmcolor.base import estimate_base_auto, estimate_base_roll
from filmcolor.frozen import load_cache
from filmcolor.invert import base_from_regions
from filmcolor.paths import data_dir
from filmcolor.pipeline import positive
from filmcolor.raw import linearize, saturation_mask
from filmcolor.rawpy_io import demosaic_linear, load_raw

# (x0, x1, y0, y1) in % of width / height
REBATE = {
    "DSC01683": dict(left=(2.0, 6.0, 10, 85), right=(97.0, 99.0, 10, 85)),
    "DSC01684": dict(left=(1.0, 5.0, 10, 85), right=(96.5, 99.0, 10, 85)),
    "DSC01688": dict(left=(1.5, 4.0, 10, 90), right=(96.0, 99.0, 10, 90)),
    "DSC01692": dict(left=(2.0, 5.0, 10, 85)),
    "DSC01694": dict(left=(4.0, 7.0, 10, 85), right=(98.0, 99.5, 10, 85)),
    "DSC01695": dict(left=(3.5, 7.0, 10, 85)),
}


# picture interior with an inward margin (x0, x1, y0, y1 in %), placed by eye on the grid previews: the base search
# restricted to it emulates a scan cropped inside the picture (no rebate, no sprocket holes, no film-free edge)
IMAGE = {
    "DSC01683": (10, 93, 6, 90), "DSC01684": (9, 92, 5, 91), "DSC01688": (8, 92, 8, 92),
    "DSC01692": (9, 94, 5, 90), "DSC01694": (11, 95, 5, 91), "DSC01695": (11, 95, 5, 91),
}


def interior_base(path, box):
    """pipeline._camera_input's automatic base with the block search limited to ``box``."""
    f = load_raw(path)
    H, W = f.cfa.shape
    x0, x1, y0, y1 = box
    crop = (int(H * y0 / 100), int(W * x0 / 100), int(H * (y1 - y0) / 100), int(W * (x1 - x0) / 100))
    est = estimate_base_auto(linearize(f), f.colors, saturation_mask(f), f.pattern.shape[0], crop=crop)
    del f
    return np.array(base_from_regions(demosaic_linear(path), est.regions)["rgb"]), est.warnings


def rect(T, r):
    h, w = T.shape[:2]
    x0, x1, y0, y1 = r
    return T[int(h * y0 / 100):int(h * y1 / 100), int(w * x0 / 100):int(w * x1 / 100)].reshape(-1, 3)


def dens(x, ref):
    return (-np.log10(np.asarray(x) / np.asarray(ref))).round(4).tolist()


def main(out_json, *arws):
    rows, own, inner, t, rows_meta = {}, {}, {}, {}, {}
    for a in arws:
        stem = Path(a).stem
        T = load_cache(data_dir("s3_a7") / f"{stem}_T8.npy").astype(np.float64)
        flat = T.reshape(-1, 3)
        bl = flat[flat[:, 1] >= 0.95 * flat[:, 1].max()]
        backlight = np.median(bl, 0)
        reb = {}
        for side, r in REBATE[stem].items():
            v = rect(T, r)
            reb[side] = dict(median=np.median(v, 0).tolist(), cv=(v.std(0) / v.mean(0)).round(4).tolist(), n=len(v))
        rebate = np.median(np.array([x["median"] for x in reb.values()]), 0)
        p = positive(a)
        b_own = np.array(p["base_info"]["rgb_render"])
        del p
        b_int, w_int = interior_base(a, IMAGE[stem])
        own[stem], inner[stem], t[stem] = b_own.tolist(), b_int.tolist(), 1.0
        rows[stem] = dict(backlight=backlight.tolist(), backlight_n=len(bl), backlight_chroma_spread=
                          (np.percentile(bl, 90, 0) / np.percentile(bl, 10, 0)).round(4).tolist(),
                          rebate=reb, rebate_used=rebate.tolist(), D_meas=dens(rebate, backlight),
                          left_right_D=dens(reb["left"]["median"], reb["right"]["median"]) if "right" in reb else None,
                          b_own=b_own.tolist(), D_own=dens(b_own, backlight),
                          err_own=(np.array(dens(b_own, backlight)) - np.array(dens(rebate, backlight))).round(4).tolist(),
                          b_int=b_int.tolist(), int_warnings=w_int, D_int=dens(b_int, backlight),
                          err_int=(np.array(dens(b_int, backlight)) - np.array(dens(rebate, backlight))).round(4).tolist())
        print(stem, json.dumps({k: rows[stem][k] for k in ("D_meas", "D_own", "err_own", "D_int", "err_int")}), flush=True)
    for key, cand in (("roll", own), ("roll_int", inner)):
        rb = estimate_base_roll(cand, t)
        b_roll = np.array(rb.rgb_norm)
        for stem, r in rows.items():
            r[f"D_{key}"] = dens(b_roll, r["backlight"])
            r[f"err_{key}"] = (np.array(r[f"D_{key}"]) - np.array(r["D_meas"])).round(4).tolist()
        rows_meta[key] = dict(used=rb.used, members=rb.members, warnings=rb.warnings)
    D_meas = np.array([r["D_meas"] for r in rows.values()])
    bls = np.array([r["backlight"] for r in rows.values()])
    summary = dict(
        roll_estimates=rows_meta,
        D_meas_mean=D_meas.mean(0).round(4).tolist(), D_meas_range=(D_meas.max(0) - D_meas.min(0)).round(4).tolist(),
        backlight_range_D=(np.log10(bls.max(0) / bls.min(0))).round(4).tolist(),
        **{f"{k}_median": np.median([r[k] for r in rows.values()], 0).round(4).tolist()
           for k in ("err_own", "err_roll", "err_int", "err_roll_int")})
    Path(out_json).write_text(json.dumps(dict(summary=summary, frames=rows), indent=1))
    print(json.dumps(summary, indent=1))
    for stem, r in rows.items():
        print(stem, "err_int", r["err_int"], "err_roll_int", r["err_roll_int"], "int warnings", len(r["int_warnings"]),
              "rebate cv", [x["cv"] for x in r["rebate"].values()])


if __name__ == "__main__":
    main(*sys.argv[1:])
