"""Summary of eval_blueneg.py output: does the frozen fixed look transfer to the BlueNeg scanner?

usage: summarize_blueneg.py eval.json [out.json]
"""
import json
import sys
from collections import defaultdict

import numpy as np

DEV = dict(  # median dE2000 on the development sets (experiments/s2_noref/REPORT.md)
    default=dict(nlp=11.89, shop=12.51),
    global_look_v1=dict(nlp=6.37, shop=6.66, nlp_cross=6.91, shop_cross=7.40),
)


def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None


def main():
    ev = json.load(open(sys.argv[1]))
    rows = ev["frames"]
    ok = [r for r in rows if r["registration"]["ncc"] >= ev["ncc_min"]]
    out = dict(n_frames=len(rows), n_registered=len(ok), ncc_min=ev["ncc_min"],
               excluded=[(r["frame"], round(r["registration"]["ncc"], 3)) for r in rows if r not in ok], dev=DEV)
    variants = [k for k, x in rows[0].items() if isinstance(x, dict) and "metrics" in x]
    out["variants"] = variants
    for v in variants:
        m = [r[v]["metrics"] for r in ok]
        d = [r[v]["decompose"] for r in ok]
        out[v] = dict(
            de2000_median=med([x["de2000_median"] for x in m]),
            mean_dL=med([x["mean_dL"] for x in m]), mean_da=med([x["mean_da"] for x in m]),
            mean_db=med([x["mean_db"] for x in m]), chroma_ratio=med([x["chroma_ratio"] for x in m]),
            upper_tone=med([x["tone"] for x in d]), upper_tone_ab=med([x["tone_ab"] for x in d]),
            qc=dict((s, sum(r[v]["qc"] == s for r in ok)) for s in ("pass", "warn", "fail")),
        )
    de = lambda r, v: r[v]["metrics"]["de2000_median"]  # noqa: E731
    out["better_than_default_frames"] = {v: sum(de(r, v) < de(r, "default") for r in ok)
                                         for v in variants if v != "default" and "default" in variants}
    if "auto" in variants:
        out["auto_statuses"] = dict((s, sum(r["auto"]["auto"] == s for r in ok)) for s in {r["auto"]["auto"] for r in ok})
    by_film = defaultdict(list)
    for r in ok:
        by_film[r["film"]].append(r)
    out["by_film"] = {f: dict(n=len(rs), **{v: med([de(r, v) for r in rs]) for v in variants})
                      for f, rs in sorted(by_film.items())}
    out["per_frame"] = []
    for r in rows:
        p = dict(frame=r["frame"], film=r["film"], ncc=round(r["registration"]["ncc"], 3),
                 rot90=r["registration"]["rot90"], mirror=r["registration"]["mirror"])
        for v in variants:
            mm = r[v]["metrics"]
            p[v] = dict(dE=round(mm["de2000_median"], 2), dL=round(mm["mean_dL"], 1), da=round(mm["mean_da"], 1),
                        db=round(mm["mean_db"], 1), upper_tone_ab=round(r[v]["decompose"]["tone_ab"], 2))
        out["per_frame"].append(p)
    print(json.dumps({k: v for k, v in out.items() if k != "per_frame"}, indent=1, ensure_ascii=False))
    for p in out["per_frame"]:
        print(p)
    if len(sys.argv) > 2:
        json.dump(out, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
