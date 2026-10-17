"""Label quality summary for the S3 labels (s3_labels.py output).

  s3_label_report.py LABEL_DIR OUT_JSON

Groups: experiment A domains (nlp, shop, blueneg) and experiment B development (shop + ext_own), plus ext_own
alone. For each: expressiveness of fixed look / P4 / P3 / v1 / 7-parameter ceiling (median and mean dE2000,
objective), P4's gap to the ceiling, parameters at a bound, near-optimal multi-start spread, per-roll label
distribution (between-roll share of variance), clipping, below-black and film-base warnings.
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

FITS = ("fixed_look", "p4", "p3", "v1", "ub7")
FREE = {"p4": ("ev", "temp", "tint", "midtone"), "p3": ("temp", "tint", "midtone"), "ub7": ("ev", "temp", "tint", "sat", "contrast", "midtone", "shoulder")}
SPREAD_FLAG = 0.5  # near-optimal starts more than half a scale unit apart = different parameters, similar fit
CLIP_FLAG, BLACK_FLAG = 0.02, 0.05


def q(xs, p):
    return float(np.percentile(xs, p)) if len(xs) else None


def between_share(vals_by_roll):
    """Share of the label variance that lies between rolls (rolls with one frame contribute to both)."""
    allv = [v for vs in vals_by_roll.values() for v in vs]
    if len(allv) < 3:
        return None
    m = np.mean(allv)
    between = sum(len(vs) * (np.mean(vs) - m) ** 2 for vs in vals_by_roll.values())
    total = sum((v - m) ** 2 for v in allv)
    return float(between / total) if total > 0 else None


def summarise(rows):
    out = dict(n=len(rows), rolls=len({r["roll"] for r in rows}))
    out["expressiveness"] = {k: dict(median_dE=q([r[k]["de2000_median"] for r in rows], 50),
                                     mean_dE=q([r[k]["de2000_mean"] for r in rows], 50),
                                     objective=q([r[k]["objective"] for r in rows if "objective" in r[k]], 50))
                             for k in FITS}
    gap_med = [r["p4"]["de2000_median"] - r["ub7"]["de2000_median"] for r in rows]
    gap_mean = [r["p4"]["de2000_mean"] - r["ub7"]["de2000_mean"] for r in rows]
    gap_obj = [r["p4"]["objective"] - r["ub7"]["objective"] for r in rows]
    gain = [r["fixed_look"]["de2000_mean"] - r["p4"]["de2000_mean"] for r in rows]
    out["p4_vs_ceiling"] = dict(
        gap_median_dE=dict(p50=q(gap_med, 50), p90=q(gap_med, 90)),
        gap_mean_dE=dict(p50=q(gap_mean, 50), p90=q(gap_mean, 90), share_over_1=float(np.mean(np.array(gap_mean) > 1)),
                         share_over_2=float(np.mean(np.array(gap_mean) > 2))),
        gap_objective=dict(p50=q(gap_obj, 50), p90=q(gap_obj, 90)),
        share_of_fixed_to_ceiling_gain_reached=q([(r["fixed_look"]["de2000_mean"] - r["p4"]["de2000_mean"]) /
                                                  max(r["fixed_look"]["de2000_mean"] - r["ub7"]["de2000_mean"], 1e-6)
                                                  for r in rows], 50),
        fixed_minus_p4_mean_dE=dict(p50=q(gain, 50)))
    out["at_bound"] = {k: {p: float(np.mean([p in r[k]["at_bound"] for r in rows])) for p in FREE[k]} for k in ("p4", "p3", "ub7")}
    out["multistart"] = {k: dict(near_optimal_median=q([r[k]["n_near_optimal"] for r in rows], 50),
                                 share_spread_over_flag={p: float(np.mean([r[k]["spread_near_optimal"][p] > SPREAD_FLAG for r in rows]))
                                                         for p in FREE[k]})
                         for k in ("p4", "p3", "ub7")}
    by_roll = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for p in FREE["p4"]:
            by_roll[p][r["roll"]].append(r["p4"]["params"][p])
    out["p4_labels"] = {p: dict(median=q([r["p4"]["params"][p] for r in rows], 50),
                                p10=q([r["p4"]["params"][p] for r in rows], 10), p90=q([r["p4"]["params"][p] for r in rows], 90),
                                between_roll_share=between_share(by_roll[p]),
                                roll_means={roll: round(float(np.mean(v)), 3) for roll, v in sorted(by_roll[p].items())})
                        for p in FREE["p4"]}
    clip = [r["p4"]["highlight"]["clipped_where_ref_not_clipped_frac"] for r in rows]
    black = [r["p4"]["diagnostics"].get("below_zero_after_colour", 0.0) for r in rows]
    warn = [bool(r["cache"]["base"].get("estimate", {}).get("warnings")) for r in rows]
    out["anomalies"] = dict(
        p4_new_clip_over_2pct=[r["id"] for r, c in zip(rows, clip) if c > CLIP_FLAG],
        p4_below_black_over_5pct=[r["id"] for r, b in zip(rows, black) if b > BLACK_FLAG],
        fixed_look_below_black_over_5pct=[r["id"] for r in rows if r["fixed_look"]["diagnostics"].get("below_zero_after_colour", 0) > BLACK_FLAG],
        base_warning_share=float(np.mean(warn)),
        p4_mean_dE_with_base_warning=q([r["p4"]["de2000_mean"] for r, w in zip(rows, warn) if w], 50),
        p4_mean_dE_without_base_warning=q([r["p4"]["de2000_mean"] for r, w in zip(rows, warn) if not w], 50),
        worst_p4_mean_dE=[(r["id"], round(r["p4"]["de2000_mean"], 2), round(r["ub7"]["de2000_mean"], 2))
                          for r in sorted(rows, key=lambda r: -r["p4"]["de2000_mean"])[:8]],
        largest_gap_to_ceiling=[(r["id"], round(g, 2)) for g, r in sorted(zip(gap_mean, rows), key=lambda t: -t[0])[:8]])
    return out


def main(label_dir, out_json):
    rows = [json.loads(p.read_text()) for p in sorted(Path(label_dir).glob("*/*.json"))]
    groups = {"A:nlp": [r for r in rows if r["domain"] == "nlp"], "A:shop": [r for r in rows if r["domain"] == "shop"],
              "A:blueneg": [r for r in rows if r["domain"] == "blueneg"],
              "B:dev (shop + ext_own)": [r for r in rows if "B" in r["experiments"]],
              "ext_own": [r for r in rows if r["domain"] == "ext_own"]}
    res = {g: summarise(rs) for g, rs in groups.items() if rs}
    res["seconds_per_frame_median"] = st.median(r["seconds"] for r in rows)
    Path(out_json).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    for g, s in res.items():
        if not isinstance(s, dict):
            continue
        e = s["expressiveness"]
        print(f"{g:24s} n={s['n']:3d} rolls={s['rolls']:2d} |",
              " ".join(f"{k} {e[k]['median_dE']:.2f}/{e[k]['mean_dE']:.2f}" for k in FITS),
              "| P4-ceiling mean dE p50 %.2f p90 %.2f" % (s["p4_vs_ceiling"]["gap_mean_dE"]["p50"], s["p4_vs_ceiling"]["gap_mean_dE"]["p90"]))


if __name__ == "__main__":
    main(*sys.argv[1:3])
