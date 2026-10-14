"""Summarise a 2 x 2 lambda x highlight-weight cross check (sensitivity.py SET cross) as Markdown.

  summarize_cross.py SENSITIVITY_JSON [--exclude STEM ...]

Parameter changes are in (base) scale units, relative to the lambda=0.05, weight 30 solution.
"""
import json
import statistics as st
import sys
from collections import defaultdict

from filmcolor.fit import SCALES
from filmcolor.render import PER_IMAGE

REF = "lam=0.05 hw=30.0"


def unwanted_clip(fit):
    return fit["highlight"]["clipped_where_ref_not_clipped_frac"]


def main(path, exclude=()):
    by = defaultdict(dict)
    for r in json.load(open(path)):
        by[r["config"]][r["stem"]] = r["fit"]
    stems = sorted(s for s in by[REF] if s not in exclude)
    print(f"{len(stems)} 帧" + (f"（排除 {', '.join(sorted(exclude))}）" if exclude else "") + "\n")
    print("| λ | hw | ΔE 中位数 | 近优解离散 中位 / 最大 | 5 起点全近优 | 碰边界 | 裁切% 最大 | 裁切 >2% 张数 | 参数变化 中位 / 最大 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for lam in ("0.02", "0.05"):
        for hw in ("10.0", "30.0"):
            fs = [by[f"lam={lam} hw={hw}"][s] for s in stems]
            sp = [max(f["spread_near_optimal"].values()) for f in fs]
            clip = [unwanted_clip(f) for f in fs]
            sh = [max(abs(f["params"][k] - by[REF][s]["params"][k]) / SCALES[k] for k in PER_IMAGE)
                  for f, s in zip(fs, stems)]
            print(f"| {lam} | {hw[:-2]} | {st.median(f['de2000_median'] for f in fs):.2f} | {st.median(sp):.2f} / {max(sp):.2f} | "
                  f"{sum(f['n_near_optimal'] == 5 for f in fs)} | {sum(len(f['at_bound']) for f in fs)} | {100 * max(clip):.1f} | "
                  f"{sum(c > 0.02 for c in clip)} | {st.median(sh):.2f} / {max(sh):.2f} |")
    worst = [(s, c) for s in stems for c in by if unwanted_clip(by[c][s]) > 0.02]
    if worst:
        print("\n裁切 >2% 的帧：" + "，".join(f"{s} {c} {100 * unwanted_clip(by[c][s]):.1f}%" for s, c in worst))


if __name__ == "__main__":
    args = sys.argv[1:]
    excl = args[args.index("--exclude") + 1:] if "--exclude" in args else []
    main(args[0], tuple(excl))
