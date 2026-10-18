"""Base-inversion diagnosis: residuals shared by the two references (experiments/s3_diag).

The shop and user grids are both crops of the same full-frame 1/8 grid of the negative (diag_variants.py records
each crop's start), so their pixels correspond on the overlap. For each fit (fixed look, ub7) and inversion (B0,
B1), with r_s / r_u the Lab residuals (ours - reference) against the shop / user print, each after its own fit:
  shared = (r_s + r_u) / 2   error both graders disagree with in the same direction (candidate inversion error)
  split  = (r_s - r_u) / 2   the part where they disagree with each other (candidate taste difference)
Reported over the overlap and per tonal zone of the mean reference L*: RMS and mean of shared / split per Lab
axis, the per-pixel correlation of r_s and r_u, and the dE2000 between the two prints themselves.

  diag_two_refs.py VARIANTS_DIR OUT_JSON STEM [STEM ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from diag_common import ZONES  # noqa: E402

from filmcolor.colorimetry import delta_e2000  # noqa: E402
from filmcolor.frozen import load_cache  # noqa: E402
from filmcolor.paths import data_dir  # noqa: E402

AXES = ("L", "a", "b")


def overlap(start_s, shape_s, start_u, shape_u):
    y0, x0 = max(start_s[0], start_u[0]), max(start_s[1], start_u[1])
    y1 = min(start_s[0] + shape_s[0], start_u[0] + shape_u[0])
    x1 = min(start_s[1] + shape_s[1], start_u[1] + shape_u[1])
    if y1 <= y0 or x1 <= x0:
        raise ValueError("the two reference grids do not overlap")
    sl = lambda st: np.s_[y0 - st[0] : y1 - st[0], x0 - st[1] : x1 - st[1]]  # noqa: E731
    return sl(start_s), sl(start_u), (y1 - y0) * (x1 - x0)


def summary(sh, sp, rs, ru):
    out = {}
    for i, ax in enumerate(AXES):
        c = np.corrcoef(rs[:, i], ru[:, i])[0, 1]
        out[ax] = dict(shared_mean=float(sh[:, i].mean()), shared_rms=float(np.sqrt((sh[:, i] ** 2).mean())),
                       split_mean=float(sp[:, i].mean()), split_rms=float(np.sqrt((sp[:, i] ** 2).mean())),
                       corr=float(c))
    return out


def main(variants_dir, out_json, *stems):
    rdir = data_dir("s3_diag", "resid")
    cache = data_dir("s3_ml", "cache", "ext_own")
    ucache = data_dir("s3_ml", "cache_user", "ext_own")
    res = {}
    for stem in stems:
        vs = json.loads((Path(variants_dir) / f"{stem}_shop.json").read_text())
        vu = json.loads((Path(variants_dir) / f"{stem}_user.json").read_text())
        ss, su, n = overlap(vs["grid"]["start"], vs["grid_shape"], vu["grid"]["start"], vu["grid_shape"])
        Rs = load_cache(cache / f"{stem}_R8.npy").astype(np.float64)[ss].reshape(-1, 3)
        Ru = load_cache(ucache / f"{stem}_R8.npy").astype(np.float64)[su].reshape(-1, 3)
        Rm = (Rs + Ru) / 2
        de_prints = delta_e2000(Rs, Ru)
        fr = dict(overlap_pixels=n, prints_de2000_median=float(np.median(de_prints)),
                  prints_mean_diff_shop_minus_user={ax: float((Rs - Ru)[:, i].mean()) for i, ax in enumerate(AXES)})
        for key in ("B0_fixed_look", "B0_ub7", "B1_fixed_look", "B1_ub7"):
            rs = load_cache(rdir / f"{stem}_shop_{key}.npy").astype(np.float64)[ss].reshape(-1, 3)
            ru = load_cache(rdir / f"{stem}_user_{key}.npy").astype(np.float64)[su].reshape(-1, 3)
            sh, sp = (rs + ru) / 2, (rs - ru) / 2
            d = dict(all=summary(sh, sp, rs, ru), zones={})
            for name, lo, hi in ZONES:
                m = (Rm[:, 0] >= lo) & (Rm[:, 0] < hi)
                if m.sum() >= 200:
                    d["zones"][name] = dict(share=float(m.mean()), **summary(sh[m], sp[m], rs[m], ru[m]))
            fr[key] = d
        res[stem] = fr
        a = fr["B0_ub7"]["all"]
        print(stem, "overlap", n, "prints dE %.1f" % fr["prints_de2000_median"],
              "B0 ub7 shared mean a %+.1f b %+.1f (rms a %.1f b %.1f) split rms a %.1f b %.1f" % (
                  a["a"]["shared_mean"], a["b"]["shared_mean"], a["a"]["shared_rms"], a["b"]["shared_rms"],
                  a["a"]["split_rms"], a["b"]["split_rms"]), flush=True)
    Path(out_json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
