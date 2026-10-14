"""Does a frame need a large exposure because of the base engine's white point? Re-invert with
lower white-point percentiles and refit (6 and 7reg). Research tool: sees the NLP reference.

  white_point_probe.py SIDECAR_DIR COMPARE_JSON OUT_JSON STEM ...
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from filmcolor.compare import Alignment, aligned_pair, block_mean, load_lab  # noqa: E402
from filmcolor.fit import fit_reference, render_lab  # noqa: E402
from filmcolor.invert import InversionParams, density, estimate_white_point, invert, region_mask  # noqa: E402
from filmcolor.raw import saturation_mask  # noqa: E402
from filmcolor.render import RenderParams  # noqa: E402
from filmcolor.rawpy_io import demosaic_linear, load_raw  # noqa: E402

PCTS = (99.9, 99.0, 97.0, 95.0, 90.0)
MARGIN = 0.03


def core(a):
    h, w = a.shape[:2]
    my, mx = int(h * MARGIN), int(w * MARGIN)
    return a[my : h - my, mx : w - mx].reshape(-1, 3)


def main(sidecar_dir, compare_json, out_json, *stems):
    comp = json.loads(Path(compare_json).read_text())
    out = {}
    for stem in stems:
        sc = json.loads((Path(sidecar_dir) / f"{stem}.json").read_text())
        p = sc["inversion"]["params"]
        f = load_raw(sc["input"]["path"])
        sat = saturation_mask(f)
        mask = ~sat & region_mask(sat.shape, tuple(sc["image_region"]["region"]))
        D = density(demosaic_linear(sc["input"]["path"]), p["base"])
        ref4 = load_lab(comp[stem]["reference"], 4)
        al = Alignment(**comp[stem]["alignment"])
        rows = []
        for pct in PCTS:
            d_hi = estimate_white_point(D, mask, pct)
            P = invert(D, InversionParams(base=p["base"], d_hi=d_hi, gamma=p["gamma"]))
            P8, R8 = aligned_pair(block_mean(P, 4).astype(np.float32), ref4, al)
            Pc, Rc = core(P8).astype(np.float32), core(R8)
            row = dict(white_pct=pct, d_hi=list(d_hi),
                       ref_share_Lstar_above_97=float(np.mean(Rc[:, 0] > 97)))
            for mode in ("6", "7reg"):
                r = fit_reference(Pc, Rc, mode, n_starts=3)
                lab, _ = render_lab(Pc[None], RenderParams(**r["params"]))
                row[mode] = dict(de2000_median=r["de2000_median"], ev=r["params"]["ev"], midtone=r["params"]["midtone"],
                                 contrast=r["params"]["contrast"], shoulder=r["params"]["shoulder"], at_bound=r["at_bound"],
                                 ours_share_Lstar_above_97=float(np.mean(lab[0][:, 0] > 97)))
            rows.append(row)
            print(stem, pct, "6:", round(row["6"]["de2000_median"], 2), row["6"]["at_bound"],
                  "| 7reg:", round(row["7reg"]["de2000_median"], 2), "ev", round(row["7reg"]["ev"], 2),
                  "| L*>97 ref", round(row["ref_share_Lstar_above_97"], 3), flush=True)
        out[stem] = rows
    Path(out_json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
