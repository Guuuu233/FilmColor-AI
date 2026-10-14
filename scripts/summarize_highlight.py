"""Compare reference fits without (S2-2) and with (S2-6) the highlight constraint, re-rendering both
parameter sets on the same aligned pixels (P8 / R8 caches written by sensitivity.py).

  summarize_highlight.py OLD_FIT_DIR NEW_FIT_DIR CACHE_DIR
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

from filmcolor.colorimetry import delta_e2000
from filmcolor.fit import reference_clipped, render_lab
from filmcolor.render import RenderParams

MODES = ("6", "7", "7reg")


def measure(P, R, params):
    lab, _, pre = render_lab(P[None], RenderParams(**params), with_pre_clip=True)
    de = delta_e2000(lab[0], R)
    over = pre[0] > 1
    ref_clip = reference_clipped(R)
    return dict(de_median=float(np.median(de)), clipped=float(over.any(-1).mean()),
                clipped_not_ref=float((over & ~ref_clip).any(-1).mean()), ref_clipped=float(ref_clip.any(-1).mean()))


def main(old_dir, new_dir, cache):
    old_dir, new_dir, cache = Path(old_dir), Path(new_dir), Path(cache)
    rows = {}
    for p in sorted(new_dir.glob("_DSF*.json")):
        stem = p.stem
        new, old = json.loads(p.read_text()), json.loads((old_dir / p.name).read_text())
        P = np.load(cache / f"{stem}_P8.npy").reshape(-1, 3)
        R = np.load(cache / f"{stem}_R8.npy").reshape(-1, 3)
        rows[stem] = {m: dict(old=measure(P, R, old["fits"][m]["params"]), new=measure(P, R, new["fits"][m]["params"]),
                              new_fit=new["fits"][m]) for m in MODES}
    print("| 帧 | 参考图裁切% | " + " | ".join(f"{m}: ΔE 旧→新 | {m}: 裁切%（参考未裁）旧→新" for m in MODES) + " |")
    print("|---|---|" + "---|---|" * len(MODES))
    for stem, r in rows.items():
        cells = []
        for m in MODES:
            o, n = r[m]["old"], r[m]["new"]
            cells += [f"{o['de_median']:.2f}→{n['de_median']:.2f}",
                      f"{100 * o['clipped_not_ref']:.1f}→{100 * n['clipped_not_ref']:.1f}"]
        print(f"| {stem} | {100 * r['7']['old']['ref_clipped']:.1f} | " + " | ".join(cells) + " |")
    print()
    for m in MODES:
        o = [r[m]["old"] for r in rows.values()]
        n = [r[m]["new"] for r in rows.values()]
        fits = [r[m]["new_fit"] for r in rows.values()]
        print(m, "median dE old", round(st.median(x["de_median"] for x in o), 2), "new", round(st.median(x["de_median"] for x in n), 2),
              "| max clipped-not-ref% old", round(100 * max(x["clipped_not_ref"] for x in o), 1),
              "new", round(100 * max(x["clipped_not_ref"] for x in n), 1),
              "| frames > 2% old", sum(x["clipped_not_ref"] > 0.02 for x in o), "new", sum(x["clipped_not_ref"] > 0.02 for x in n),
              "| max near-optimal spread new", round(max(max(f["spread_near_optimal"].values()) for f in fits), 2),
              "| bound hits new", sorted({k for f in fits for k in f["at_bound"]}),
              sum(len(f["at_bound"]) for f in fits))


if __name__ == "__main__":
    main(*sys.argv[1:4])
