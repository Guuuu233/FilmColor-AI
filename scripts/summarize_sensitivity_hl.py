"""Summarise the S2-6 sensitivity run (sensitivity.py s2-6; highlight constraint on) as Markdown.

  summarize_sensitivity_hl.py S2_6_JSON [S2_3_JSON]

With the S2-3 result, each λ row also shows the unconstrained median ΔE and worst unwanted clipping.
"""
import json
import statistics as st
import sys
from collections import defaultdict

from filmcolor.fit import SCALES
from filmcolor.render import PER_IMAGE

BASE = "lam=0.05 hw=10.0"


def shift(a, b):
    """Largest parameter difference in (base) scale units."""
    return max(abs(a[k] - b[k]) / SCALES[k] for k in PER_IMAGE)


def unwanted_clip(fit):
    h = fit.get("highlight")
    return h["clipped_where_ref_not_clipped_frac"] if h else fit["diagnostics"]["above_white"]


def row(rs):
    spreads = [max(r["fit"]["spread_near_optimal"].values()) for r in rs]
    clip = [unwanted_clip(r["fit"]) for r in rs]
    return dict(de=st.median(r["fit"]["de2000_median"] for r in rs), spread_med=st.median(spreads), spread_max=max(spreads),
                all5=sum(r["fit"]["n_near_optimal"] == 5 for r in rs), bounds=sum(len(r["fit"]["at_bound"]) for r in rs),
                clip_max=max(clip), clip_frames=sum(c > 0.02 for c in clip),
                ev=st.median(abs(r["fit"]["params"]["ev"]) for r in rs))


def main(path, old_path=None):
    by = defaultdict(dict)
    for r in json.load(open(path)):
        by[r["config"]][r["stem"]] = r
    old = defaultdict(dict)
    if old_path:
        for r in json.load(open(old_path)):
            old[r["config"]][r["stem"]] = r
    stems = sorted(by[BASE])

    print("## 正则强度（高光权重 10）\n")
    print("| λ | ΔE 中位数 | 最大离散 中位 / 最大 | 5 起点全近优 | 碰边界 | 参考未裁处裁切% 最大 | 裁切 >2% 张数 | |ev| 中位 |"
          + (" 无约束 S2-3：ΔE / 裁切% 最大 |" if old_path else ""))
    print("|---|---|---|---|---|---|---|---|" + ("---|" if old_path else ""))
    for name in [c for c in by if c.endswith("hw=10.0")]:
        lam = name.split()[0]
        r = row([by[name][s] for s in stems])
        line = (f"| {lam[4:]} | {r['de']:.2f} | {r['spread_med']:.2f} / {r['spread_max']:.2f} | {r['all5']} | {r['bounds']} | "
                f"{100 * r['clip_max']:.1f} | {r['clip_frames']} | {r['ev']:.2f} |")
        if old_path and lam in old:
            o = row([old[lam][s] for s in stems])
            line += f" {o['de']:.2f} / {100 * o['clip_max']:.1f} |"
        print(line)

    print("\n## 高光权重（λ=0.05；参数变化相对 hw=10 的解）\n")
    print("| hw | ΔE 中位数 | 参考未裁处裁切% 最大 | 裁切 >2% 张数 | 最大离散 最大 | 参数最大变化（尺度单位）中位 / 最大 |")
    print("|---|---|---|---|---|---|")
    hws = sorted((c for c in by if c.startswith("lam=0.05 ")), key=lambda c: float(c.split("hw=")[1]))
    for name in hws:
        r = row([by[name][s] for s in stems])
        sh = [shift(by[name][s]["fit"]["params"], by[BASE][s]["fit"]["params"]) for s in stems]
        print(f"| {name.split('hw=')[1]} | {r['de']:.2f} | {100 * r['clip_max']:.1f} | {r['clip_frames']} | {r['spread_max']:.2f} | "
              f"{st.median(sh):.2f} / {max(sh):.2f} |")

    print("\n## 逐张：参考未裁处裁切%（λ=0.05）\n")
    print("| 帧 | " + " | ".join(f"hw={n.split('hw=')[1]}" for n in hws) + " |")
    print("|---|" + "---|" * len(hws))
    for s in stems:
        print(f"| {s} | " + " | ".join(f"{100 * unwanted_clip(by[n][s]['fit']):.1f}" for n in hws) + " |")


if __name__ == "__main__":
    main(*sys.argv[1:3])
