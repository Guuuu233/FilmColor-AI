"""Summarise the clipped-area ablation (sensitivity.py SET area) as Markdown.

  summarize_area.py SENSITIVITY_JSON [--exclude STEM ...]

Parameter changes are in (base) scale units, relative to the area weight 0 solution (the S2-6 candidate).
Texture: highlight_texture ratio (ours / reference local detail in the reference's highlights; 1 = as in the
reference); frames without enough highlight pixels are left out of the texture columns.
"""
import json
import statistics as st
import sys
from collections import defaultdict

from filmcolor.fit import SCALES
from filmcolor.render import PER_IMAGE


def main(path, exclude=()):
    by = defaultdict(dict)
    for r in json.load(open(path)):
        by[r["config"]][r["stem"]] = r["fit"]
    names = sorted(by, key=lambda c: float(c.split("area=")[1]))
    base = names[0]
    stems = sorted(s for s in by[base] if s not in exclude)
    print(f"{len(stems)} 帧" + (f"（排除 {', '.join(sorted(exclude))}）" if exclude else "") + "\n")
    print("| 面积权重 | ΔE 中位数 | 近优解离散 最大 | 5 起点全近优 | 碰边界 | 裁切% 最大 | 裁切 >2% 张数 | 层次比值 中位 / 最低 | 参数变化 中位 / 最大 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for name in names:
        fs = [by[name][s] for s in stems]
        clip = [f["highlight"]["clipped_where_ref_not_clipped_frac"] for f in fs]
        tex = [f["texture"]["ratio"] for f in fs if f["texture"]["ratio"] is not None]
        sh = [max(abs(f["params"][k] - by[base][s]["params"][k]) / SCALES[k] for k in PER_IMAGE) for f, s in zip(fs, stems)]
        print(f"| {name.split('area=')[1]} | {st.median(f['de2000_median'] for f in fs):.2f} | "
              f"{max(max(f['spread_near_optimal'].values()) for f in fs):.2f} | {sum(f['n_near_optimal'] == 5 for f in fs)} | "
              f"{sum(len(f['at_bound']) for f in fs)} | {100 * max(clip):.1f} | {sum(c > 0.02 for c in clip)} | "
              f"{st.median(tex):.2f} / {min(tex):.2f} | {st.median(sh):.2f} / {max(sh):.2f} |")
    print(f"\n层次比值有效帧数：{sum(by[base][s]['texture']['ratio'] is not None for s in stems)}\n")
    print("| 帧 | " + " | ".join(f"area={n.split('area=')[1]}：ΔE / 裁切% / 层次" for n in names) + " |")
    print("|---|" + "---|" * len(names))
    for s in sorted(by[base]):
        cells = []
        for n in names:
            f = by[n][s]
            t = f["texture"]["ratio"]
            cells.append(f"{f['de2000_median']:.2f} / {100 * f['highlight']['clipped_where_ref_not_clipped_frac']:.1f} / "
                         + (f"{t:.2f}" if t is not None else "—"))
        print(f"| {s}{'（异常帧）' if s in exclude else ''} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    args = sys.argv[1:]
    excl = args[args.index("--exclude") + 1:] if "--exclude" in args else []
    main(args[0], tuple(excl))
