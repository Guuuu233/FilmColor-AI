"""Roll-base validation v1, stability of the roll base itself (experiments/s3_rollbase_v1).

Inputs are the per-frame automatic bases and EXIF exposure times recorded by diag_roll_base.py; no image, cache or
reference is read. For each roll, the reference estimate (estimate_base_roll defaults) is compared with:
  loo:<frame>    the estimate without one frame (every frame whose removal changes anything is listed)
  ev_true        exposure normalised by the true third-stop shutter time 2^(round(3 log2 t)/3) instead of the
                 nominal EXIF value (Nikon writes 1/15 for 1/16 s, 1/13 for 2^-11/3 s, ...)
  n_use=k        median of the k most transparent cluster frames, k = 1 .. members
  chroma_tol=x, level_window=x   the cluster thresholds halved and doubled
Reported per alternative: base change in density per channel (log10 alt/ref, + = denser), its chromaticity part
(R-G, B-G), and how many frames change side of the 0.1 D colour-disagreement rule
(max(|o_R - o_G|, |o_B - o_G|) >= 0.1, o = log10(roll base / own base)).

  rb_stability.py ROLL_BASE_JSON OUT_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np

from filmcolor.base import estimate_base_roll

SPLIT = 0.1


def colour_error(own, roll):
    o = np.log10(np.asarray(roll) / np.asarray(own))
    return float(max(abs(o[0] - o[1]), abs(o[2] - o[1])))


def summarise(rb, ref, cand, t):
    b, b0 = np.array(rb.for_exposure(1.0)), np.array(ref.for_exposure(1.0))
    d = np.log10(b / b0)
    flips = [n for n in cand
             if (colour_error(cand[n], rb.for_exposure(t[n])) >= SPLIT) != (colour_error(cand[n], ref.for_exposure(t[n])) >= SPLIT)]
    return dict(used=rb.used, members=len(rb.members), warnings=rb.warnings, d_density=d.round(4).tolist(),
                d_chroma_RG_BG=[round(float(d[0] - d[1]), 4), round(float(d[2] - d[1]), 4)], trigger_flips=flips)


def main(rb_json, out_json):
    src = json.loads(Path(rb_json).read_text())
    out = dict(source=rb_json, split=SPLIT, rolls={})
    for roll, r in src["rolls"].items():
        fr = {n: v for n, v in r["frames"].items() if "base_own" in v}
        cand = {n: v["base_own"] for n, v in fr.items()}
        t = {n: float(v["exif"]["ExposureTime"]) for n, v in fr.items()}
        ref = estimate_base_roll(cand, t)
        alts = {}
        for n in cand:
            c2, t2 = {k: v for k, v in cand.items() if k != n}, {k: v for k, v in t.items() if k != n}
            try:
                rb = estimate_base_roll(c2, t2, t_ref=ref.t_ref)
            except ValueError as e:
                alts[f"loo:{n}"] = dict(error=str(e))
                continue
            s = summarise(rb, ref, c2, t2)
            if s["used"] != ref.used or s["members"] != len(ref.members):
                alts[f"loo:{n}"] = s
        t_true = {n: 2.0 ** (round(3 * np.log2(v)) / 3) for n, v in t.items()}
        alts["ev_true"] = summarise(estimate_base_roll(cand, t_true, t_ref=2.0 ** (round(3 * np.log2(ref.t_ref)) / 3)),
                                    ref, cand, t_true) | dict(
            nominal_vs_true_D=sorted({round(float(np.log10(t[n] / t_true[n])), 4) for n in t}))
        for k in range(1, len(ref.members) + 1):
            alts[f"n_use={k}"] = summarise(estimate_base_roll(cand, t, t_ref=ref.t_ref, n_use=k), ref, cand, t)
        for key, base_val in (("chroma_tol", 0.05), ("level_window", 0.1)):
            for x in (base_val / 2, base_val * 2):
                try:
                    alts[f"{key}={x}"] = summarise(estimate_base_roll(cand, t, t_ref=ref.t_ref, **{key: x}), ref, cand, t)
                except ValueError as e:
                    alts[f"{key}={x}"] = dict(error=str(e))
        trig = sorted(n for n in cand if colour_error(cand[n], ref.for_exposure(t[n])) >= SPLIT)
        out["rolls"][roll] = dict(frames=len(cand), used=ref.used, members=ref.members,
                                  exposure_times=sorted(set(t.values())), triggered=trig, alternatives=alts)
        print(roll, "frames", len(cand), "used", ref.used, "members", len(ref.members), "triggered", len(trig))
        for k, v in alts.items():
            print("  ", k, json.dumps(v))
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
