"""Per-source tables of T4D_REPORT.md from scripts/t4d_source_eval.py outputs.

  t4d_summary.py BASE=FILE[,FILE...]:VARIANT  OTHER=FILE[,FILE...]:VARIANT  [...]

Each argument names a column: the frames of the listed files (merged; a later file replaces an earlier one's frame)
under one variant. The first column is the baseline of the better / worse counts. Frames are compared on the ids
present in every column, without errors and with alignment NCC >= ncc_min (0.7). Per source (nlp, shop, blueneg,
ext_own) and overall: median and mean of the per-frame median dE2000, median of the per-frame mean dL*, median |dL*|,
median chroma ratio, median of our / the reference's median L* and C*, frames better / worse than the baseline by
more than 0.25, and frames with backlit weight > 0 (or the T4c flag).
"""
import json
import sys

import numpy as np

DOMAINS = ("nlp", "shop", "blueneg", "ext_own")


def load(arg):
    name, rest = arg.split("=", 1)
    files, variant = rest.rsplit(":", 1)
    frames, ncc_min = {}, 0.7
    meta = []
    for fn in files.split(","):
        d = json.load(open(fn))
        ncc_min = d.get("ncc_min", ncc_min)
        meta.append(f"{fn.split('/')[-1]} {d['git_commit'][:7]} dirty={d['src_dirty']}")
        for k, v in d["frames"].items():
            frames[k] = v
    ok = {k: dict(v, v=v["variants"][variant]) for k, v in frames.items()
          if "error" not in v and v.get("ncc", 0) >= ncc_min and variant in v.get("variants", {})}
    return name, ok, meta, {k for k, v in frames.items() if "error" in v or v.get("ncc", 0) < ncc_min}


def stats(rows):
    de = np.array([r["v"]["metrics"]["de2000_median"] for r in rows])
    dl = np.array([r["v"]["metrics"]["mean_dL"] for r in rows])
    cr = np.array([r["v"]["metrics"]["chroma_ratio"] or np.nan for r in rows], float)
    oL = np.array([r["v"]["ours_L_median"] for r in rows])
    rL = np.array([r["v"]["ref_L_median"] for r in rows])
    oC = np.array([r["v"]["ours_C_median"] for r in rows])
    rC = np.array([r["v"]["ref_C_median"] for r in rows])
    bl = sum(1 for r in rows if (r["v"].get("backlit") or {}).get("weight", 0) > 0
             or "backlit_midtone" in (r["v"].get("flags") or []))
    return dict(n=len(rows), de_med=np.median(de), de_mean=de.mean(), dl_med=np.median(dl), adl_med=np.median(np.abs(dl)),
                cr_med=np.nanmedian(cr), oL=np.median(oL), rL=np.median(rL), oC=np.median(oC), rC=np.median(rC), backlit=bl)


def main(args):
    cols = [load(a) for a in args]
    common = set.intersection(*(set(c[1]) for c in cols))
    for name, _, meta, bad in cols:
        print(f"# {name}: {'; '.join(meta)}; left out (error / NCC < 0.7): {len(bad)}")
    base = cols[0]
    for dom in DOMAINS + ("all",):
        ids = sorted(k for k in common if dom == "all" or base[1][k]["domain"] == dom)
        if not ids:
            continue
        print(f"\n## {dom} ({len(ids)} frames)")
        print("| column | dE median | dE mean | dL median | abs dL median | chroma ratio | L* ours / ref | C* ours / ref | "
              "better / worse vs " + base[0] + " | backlit |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        b_de = {k: base[1][k]["v"]["metrics"]["de2000_median"] for k in ids}
        for name, fr, _, _ in cols:
            s = stats([fr[k] for k in ids])
            d = np.array([fr[k]["v"]["metrics"]["de2000_median"] - b_de[k] for k in ids])
            bw = f"{int((d < -0.25).sum())} / {int((d > 0.25).sum())}" if name != base[0] else "-"
            print(f"| {name} | {s['de_med']:.2f} | {s['de_mean']:.2f} | {s['dl_med']:+.2f} | {s['adl_med']:.2f} | "
                  f"{s['cr_med']:.2f} | {s['oL']:.1f} / {s['rL']:.1f} | {s['oC']:.1f} / {s['rC']:.1f} | {bw} | {s['backlit']} |")


if __name__ == "__main__":
    main(sys.argv[1:])
