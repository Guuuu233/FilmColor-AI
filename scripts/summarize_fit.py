"""Summarise fit_nlp.py results into a Markdown table and aggregate statistics."""
import json
import statistics as st
import sys
from pathlib import Path

MODES = ("6", "7", "7reg")


def main(fit_dir):
    rows = [json.loads(p.read_text()) for p in sorted(Path(fit_dir).glob("_DSF*.json"))]
    print("| 帧 | 默认 | 6 | 7 | 7reg | 7 的 ev / midtone | 7reg 的 ev / midtone | 碰边界（6 / 7 / 7reg） | 近优起点数（6/7/7reg） | 最大离散（6/7/7reg） | 截负值%（6/7/7reg） | cos(ev,midtone) 默认 / 7 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        f = r["fits"]
        spread = [max(f[m]["spread_near_optimal"].values()) for m in MODES]
        bz = [100 * f[m]["diagnostics"]["below_zero_after_colour"] for m in MODES]
        print(f"| {r['stem']} | {r['default']['de2000_median']:.1f} | " + " | ".join(f"{f[m]['de2000_median']:.2f}" for m in MODES)
              + f" | {f['7']['params']['ev']:+.2f} / {f['7']['params']['midtone']:.3f} | {f['7reg']['params']['ev']:+.2f} / {f['7reg']['params']['midtone']:.3f}"
              + " | " + " / ".join(",".join(f[m]["at_bound"]) or "—" for m in MODES)
              + " | " + "/".join(str(f[m]["n_near_optimal"]) for m in MODES)
              + " | " + "/".join(f"{s:.2f}" for s in spread)
              + " | " + "/".join(f"{b:.1f}" for b in bz)
              + f" | {r['conditioning_default']['cosine']['ev/midtone']:.2f} / {r['conditioning_fit7']['cosine']['ev/midtone']:.2f} |")
    colour = [r for r in rows if r["stem"] != "_DSF9228"]
    print()
    for label, sel in (("all 12", rows), ("colour 11", colour)):
        print(label, "median of medians: default", round(st.median(r["default"]["de2000_median"] for r in sel), 2),
              *[f"{m} {st.median(r['fits'][m]['de2000_median'] for r in sel):.2f}" for m in MODES])
    gain = [r["fits"]["6"]["de2000_median"] - r["fits"]["7"]["de2000_median"] for r in rows]
    print("6 minus 7 (median dE):", [round(g, 2) for g in gain])
    bounds = {m: {} for m in MODES}
    for r in rows:
        for m in MODES:
            for k in r["fits"][m]["at_bound"]:
                bounds[m][k] = bounds[m].get(k, 0) + 1
    print("bound hits by parameter:", bounds)
    print("sv ratio default / fit7:", [(r["stem"][-4:], round(r["conditioning_default"]["sv_ratio"], 3), round(r["conditioning_fit7"]["sv_ratio"], 3)) for r in rows])
    print("other |cos| > 0.8 at fit7:", sorted({k for r in rows for k, v in r["conditioning_fit7"]["cosine"].items() if v > 0.8 and k != "ev/midtone"}))


if __name__ == "__main__":
    main(sys.argv[1])
