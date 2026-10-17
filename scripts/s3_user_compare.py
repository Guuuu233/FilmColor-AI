"""Summary of the two-grader comparison (s3_user_labels.py), per roll and overall.

Q1: user minus shop P4 parameters (scale units): mean, spread over frames, and the offset share mean^2 /
(mean^2 + variance) (1 = a pure roll offset, 0 = purely frame-specific).
Q2: dE2000 to the user's print of the fixed look, of the shop-fitted P4 parameters, of the shop parameters shifted
by the roll's mean user-minus-shop offset estimated on the OTHER frames of the roll (leave one frame out: what a
known roll-level correction would recover), and of the user's own P4 label.
Q3: user P4 label against the 7-parameter ceiling ub7 (mean dE).

  s3_user_compare.py USER_LABEL_DIR OUT_JSON
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

from filmcolor.colorimetry import delta_e2000
from filmcolor.fit import SCALES, render_lab
from filmcolor.frozen import load_cache
from filmcolor.paths import data_dir
from filmcolor.render import BOUNDS, RenderParams

FREE = ("ev", "temp", "tint", "midtone")


def main(label_dir, out_json):
    rows = [json.loads(p.read_text()) for p in sorted(Path(label_dir).glob("*.json"))]
    for r in rows:
        r["diff"] = {k: (r["p4"]["params"][k] - r["shop_p4_params"][k]) / SCALES[k] for k in FREE}
    out = {}
    for roll in sorted({r["roll"] for r in rows}) + ["all"]:
        g = [r for r in rows if roll == "all" or r["roll"] == roll]
        q1 = {}
        for k in FREE:
            d = np.array([r["diff"][k] for r in g])
            q1[k] = dict(mean=float(d.mean()), sd=float(d.std()), offset_share=float(d.mean() ** 2 / (d.mean() ** 2 + d.var())))
        loo = []
        for r in g if roll != "all" else []:
            others = [o for o in g if o is not r]
            p = dict(r["shop_p4_params"])
            for k in FREE:
                p[k] = float(np.clip(p[k] + SCALES[k] * np.mean([o["diff"][k] for o in others]), *BOUNDS[k]))
            cache = data_dir("s3_ml", "cache_user", "ext_own")
            stem = r["id"].split(":")[1]
            P, R = load_cache(cache / f"{stem}_P8.npy").reshape(-1, 3), load_cache(cache / f"{stem}_R8.npy").reshape(-1, 3)
            lab, _ = render_lab(P[None], RenderParams(**p))
            r["shop_shifted_de"] = float(np.median(delta_e2000(lab[0], R.astype(np.float64))))
            loo.append(r["shop_shifted_de"])
        med = lambda xs: float(st.median(xs)) if xs else None  # noqa: E731
        out[roll] = dict(
            n=len(g), q1_user_minus_shop_scaled=q1,
            q2_de_to_user_print=dict(fixed_look=med([r["fixed_look"]["de2000_median"] for r in g]),
                                     shop_p4=med([r["shop_p4_on_user"]["de2000_median"] for r in g]),
                                     shop_p4_plus_roll_offset_loo=med(loo) if loo else med([r.get("shop_shifted_de") for r in rows if "shop_shifted_de" in r]),
                                     user_p4_label=med([r["p4"]["de2000_median"] for r in g])),
            shop_p4_on_shop_print=med([r["shop_p4_de"] for r in g]),
            q3_user_p4_minus_ub7_mean_dE=dict(median=med([r["p4"]["de2000_mean"] - r["ub7"]["de2000_mean"] for r in g]),
                                              share_over_2=float(np.mean([r["p4"]["de2000_mean"] - r["ub7"]["de2000_mean"] > 2 for r in g]))),
            alignment_ncc_min=min(r["alignment"]["ncc"] for r in g), ref_kinds=sorted({r["ref_kind"] for r in g}))
    Path(out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for roll, s in out.items():
        q2 = s["q2_de_to_user_print"]
        print(f"{roll:30s} n={s['n']:2d} | Q1 " + " ".join(f"{k} {v['mean']:+.2f}±{v['sd']:.2f} ({v['offset_share']:.2f})" for k, v in s["q1_user_minus_shop_scaled"].items()),
              "| Q2 fixed %.2f shop %.2f shop+offset %s user %.2f | shop on shop %.2f | Q3 %.2f (>2: %.0f%%) | ncc min %.3f" % (
                  q2["fixed_look"], q2["shop_p4"], "%.2f" % q2["shop_p4_plus_roll_offset_loo"] if q2["shop_p4_plus_roll_offset_loo"] else "-",
                  q2["user_p4_label"], s["shop_p4_on_shop_print"], s["q3_user_p4_minus_ub7_mean_dE"]["median"],
                  100 * s["q3_user_p4_minus_ub7_mean_dE"]["share_over_2"], s["alignment_ncc_min"]))


if __name__ == "__main__":
    main(*sys.argv[1:3])
