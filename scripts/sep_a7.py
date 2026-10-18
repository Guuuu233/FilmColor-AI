"""Separating the colour and the level role of the film base (experiments/s3_separate/DESIGN.md), A7R5 rebate frames.

No reference prints exist for these frames; the measured base BM (rebate, experiments/s3_a7/measure.json) is the
physical anchor. Bases in the same linear units (1/8 caches of a7_inspect.py, one session, t = 1):
  B0    per-frame automatic base, search limited to the picture interior (a7_measure.py b_int; a scan cropped
        inside the picture)
  B1    estimate_base_roll over the six interior bases (recomputed here exactly as a7_measure.py roll_int)
  L0C1, L1C0  as sep_check.py (level = channel-mean log base)
  BM    measured base (rebate median)
White point: 99.9th percentile of -log10 T over the picture interior of the 1/8 cache (same for every variant;
d_hi = that + log10 b). Metrics over the picture interior (a7_measure.py IMAGE): share below black; fixed look and
level matched (ev only, median L* = that of B0 under the fixed look) renders compared with BM's render under the
same reading: dE2000 median and a*b* distance median ("how far from the measured-base result").

  sep_a7.py OUT_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from a7_measure import IMAGE, rect  # noqa: E402
from diag_common import LOOK, positive_T, render_lab  # noqa: E402
from sep_check import bases, match_ev, median_L  # noqa: E402

from filmcolor.base import estimate_base_roll  # noqa: E402
from filmcolor.colorimetry import delta_e2000  # noqa: E402
from filmcolor.frozen import load_cache  # noqa: E402
from filmcolor.invert import EPS  # noqa: E402
from filmcolor.paths import data_dir  # noqa: E402
from filmcolor.render import RenderParams  # noqa: E402


def lab_of(P, params):
    lab, _, _ = render_lab(P[None], RenderParams(**params), with_pre_clip=True)
    return lab[0]


def main(out_json):
    m = json.loads(Path("experiments/s3_a7/measure.json").read_text())["frames"]
    inner = {s: r["b_int"] for s, r in m.items()}
    rb = estimate_base_roll(inner, {s: 1.0 for s in inner})
    broll = np.array(rb.rgb_norm)
    out = dict(roll=dict(used=rb.used, members=rb.members, warnings=rb.warnings, base=broll.tolist()), frames={})
    for stem, r in m.items():
        T = rect(load_cache(data_dir("s3_a7") / f"{stem}_T8.npy").astype(np.float64), IMAGE[stem])
        pct = np.percentile(-np.log10(np.maximum(T, EPS)), 99.9, axis=0)
        bs = bases(np.array(r["b_int"]), broll) | {"BM": np.array(r["rebate_used"])}
        Ps = {k: positive_T(T, b, (0, 0, 0), pct + np.log10(b)) for k, b in bs.items()}
        target = median_L(Ps["B0"], LOOK["params"])
        labs, row = {}, {}
        for k, P in Ps.items():
            ev, clamped = match_ev(P, target)
            labs[k] = dict(fixed=lab_of(P, LOOK["params"]), matched=lab_of(P, LOOK["params"] | dict(ev=ev)))
            row[k] = dict(base=bs[k].tolist(), D_vs_BM=(-np.log10(bs[k] / bs["BM"])).round(4).tolist(),
                          below_black_any=float((P < 0).any(-1).mean()),
                          below_black=[float(x) for x in (P < 0).mean(0)], ev_matched=ev, ev_clamped=clamped)
        for k in Ps:
            for reading in ("fixed", "matched"):
                a, b = labs[k][reading], labs["BM"][reading]
                row[k][f"{reading}_dE_vs_BM"] = float(np.median(delta_e2000(a, b)))
                row[k][f"{reading}_ab_vs_BM"] = float(np.median(np.hypot(*(a[:, 1:] - b[:, 1:]).T)))
                row[k][f"{reading}_L_median"] = float(np.median(a[:, 0]))
        out["frames"][stem] = row
        print(stem, " ".join(f"{k} m {row[k]['matched_dE_vs_BM']:.2f}/{row[k]['matched_ab_vs_BM']:.2f} "
                             f"bb {row[k]['below_black_any']:.3f}" for k in Ps), flush=True)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:2])
