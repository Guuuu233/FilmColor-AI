"""Tables for experiments/s2_noref/REPORT.md from the stored results (no fitting, no estimation).

  summarize_noref_v2.py ABLATION_JSON ORACLE_JSON
"""
import collections
import json
import statistics as st
import sys

import numpy as np
from scipy.stats import spearmanr

from filmcolor.render import LUMA, RenderParams, render_full

ANOMALOUS = {"DSC_3822", "DSC_3858"}
GROUPS = ["look", "brightness", "wb", "tone", "sat", "all"]


def lum(y, p):
    """Display-linear luminance given to a grey scene pixel of luminance ``y``."""
    v, _, _ = render_full(np.full((1, 1, 3), y, np.float32), RenderParams(**p))
    v = np.clip(v, 0, 1)
    v = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)
    return float(v[0, 0] @ LUMA)


def main(abl_path, orc_path):
    L, O = json.load(open(abl_path)), json.load(open(orc_path))["results"]
    look = L["look_params"]
    stages = {s: {(r["set"], r["stem"]): r for r in v["results"]} for s, v in L["stages"].items()}

    print("## stages (median dE2000, anomalous frames excluded)\n")
    print("| stage | groups | NLP | shop | better / worse vs kept | max clip_not_ref | kept |")
    print("|---|---|---|---|---|---|---|")
    for s, v in L["stages"].items():
        res = stages[s]
        med = {w: st.median(r["eval"]["de2000_median"] for (ww, stem), r in res.items()
                            if ww == w and "eval" in r and stem not in ANOMALOUS) for w in ("nlp", "shop")}
        clip = max(r["eval"]["clip_not_ref"] for r in res.values() if "eval" in r and r["stem"] not in ANOMALOUS)
        d = v.get("decision")
        bw = f"{d['better']} / {d['worse']}" if d else "-"
        print(f"| {s} | {', '.join(v['groups']) or 'look only'}{' (brighten only)' if v['brighten_only'] else ''} "
              f"| {med['nlp']:.2f} | {med['shop']:.2f} | {bw} | {clip:.3f} | {'yes' if s in L['kept'] else 'no'} |")
    print(f"\nspread band for D: {L.get('spread_band_source')}")

    orc = collections.defaultdict(dict)
    for r in O:
        orc[(r["set"], r["stem"])][r["group"]] = r
    print("\n## per-group oracle (median dE2000, anomalous frames excluded)\n")
    print("| set | " + " | ".join(GROUPS) + " |")
    print("|---|" + "---|" * len(GROUPS))
    for w in ("nlp", "shop"):
        rows = [v for (s, stem), v in orc.items() if s == w and stem not in ANOMALOUS]
        print(f"| {w} ({len(rows)}) | " + " | ".join(f"{st.median(v[g]['eval']['de2000_median'] for v in rows):.2f}"
                                                    for g in GROUPS) + " |")
    best = collections.Counter(max(GROUPS[1:5], key=lambda g: v["look"]["eval"]["de2000_median"] - v[g]["eval"]["de2000_median"])
                               for (s, stem), v in orc.items() if stem not in ANOMALOUS)
    print(f"\nlargest single-group gain (frames): {dict(best)}")

    A, B1, B2, C = (stages[s] for s in ("A", "B1", "B2", "C"))
    need_b, est_b, conf_b, need = [], [], [], collections.defaultdict(list)
    for k, o in orc.items():
        if k[1] in ANOMALOUS:
            continue
        y = A[k]["estimate"]["stats"]["median"]
        nb = np.log2(lum(y, o["brightness"]["params"]) / lum(y, look))
        need_b.append(nb)
        est_b.append(np.log2(lum(y, B2[k]["estimate"]["params"]) / lum(y, look)))
        conf_b.append((B1[k]["estimate"]["confidence"]["exposure"],
                       abs(nb - np.log2(lum(y, B1[k]["estimate"]["params"]) / lum(y, look)))))
        pc = C[k]["estimate"]["params"]
        for c in ("temp", "tint"):
            need[c].append((o["wb"]["params"][c] - look[c], pc[c] - look[c]))
    print("\n## does the estimator see what the oracle needs (anomalous frames excluded)\n")
    print("| quantity | needed p10 .. p90 | Spearman (needed, estimated) |")
    print("|---|---|---|")
    r = spearmanr(need_b, est_b)
    print(f"| brightness of the median pixel, stops (B2) | {np.percentile(need_b, 10):+.2f} .. {np.percentile(need_b, 90):+.2f} "
          f"| {r.statistic:+.2f} (p = {r.pvalue:.2f}) |")
    for c, v in need.items():
        n, e = zip(*v)
        r = spearmanr(n, e)
        print(f"| {c} (C) | {np.percentile(n, 10):+.2f} .. {np.percentile(n, 90):+.2f} | {r.statistic:+.2f} (p = {r.pvalue:.2f}) |")
    c, err = zip(*conf_b)
    r = spearmanr(c, err)
    lo = [e for cc, e in conf_b if cc < 1]
    hi = [e for cc, e in conf_b if cc == 1]
    print(f"\nexposure confidence (B1) vs |brightness error|: Spearman {r.statistic:+.2f} (p = {r.pvalue:.2f}); "
          f"median error at confidence < 1: {np.median(lo) if lo else float('nan'):.2f} stops (n = {len(lo)}), "
          f"at confidence 1: {np.median(hi):.2f} stops (n = {len(hi)})")
    print("\n## input-quality gate (stage A estimates)\n")
    for k, r in A.items():
        e = r["estimate"]
        if e["status"] != "ok" or e["flags"]:
            print(f"- {k[1]}: {e['status']} {e['flags']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
