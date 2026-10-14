"""Summarise sensitivity.py results (Markdown)."""
import json
import statistics as st
import sys
from collections import defaultdict

from filmcolor.fit import SCALES
from filmcolor.render import PER_IMAGE

BASE = "lam=0.05"


def shift(a, b):
    """Largest parameter difference in (base) scale units."""
    return max(abs(a[k] - b[k]) / SCALES[k] for k in PER_IMAGE)


def main(path):
    res = json.load(open(path))
    by = defaultdict(dict)
    for r in res:
        by[r["config"]][r["stem"]] = r
    stems = sorted(by[BASE])
    med = lambda xs: st.median(xs)  # noqa: E731

    print("## 正则强度\n")
    print("| λ | ΔE 中位数（12 张中位） | 最大离散 中位 / 最大 | 5 起点全近优的张数 | 碰边界次数 | |ev| 中位 | 最小缩放奇异值 中位 | 截负值% 最大 |")
    print("|---|---|---|---|---|---|---|---|")
    for name in [c for c in by if c.startswith("lam=")]:
        rs = [by[name][s] for s in stems]
        spreads = [max(r["fit"]["spread_near_optimal"].values()) for r in rs]
        print(f"| {name[4:]} | {med(r['fit']['de2000_median'] for r in rs):.2f} | {med(spreads):.2f} / {max(spreads):.2f} | "
              f"{sum(r['fit']['n_near_optimal'] == 5 for r in rs)} | {sum(len(r['fit']['at_bound']) for r in rs)} | "
              f"{med(abs(r['fit']['params']['ev']) for r in rs):.2f} | {med(min(r['conditioning']['sv_scaled']) for r in rs):.3f} | "
              f"{100 * max(r['fit']['diagnostics']['below_zero_after_colour'] for r in rs):.1f} |")

    print("\n## 尺度扰动（相对 λ=0.05、原尺度的解）\n")
    print("| 配置 | 参数最大变化（尺度单位）中位 / 最大 | 变化最大的参数（出现次数） | ΔE 变化 中位 / 最大 |")
    print("|---|---|---|---|")
    for name in [c for c in by if c.startswith("scale")]:
        sh, dde, which = [], [], defaultdict(int)
        for s in stems:
            a, b = by[name][s]["fit"], by[BASE][s]["fit"]
            sh.append(shift(a["params"], b["params"]))
            dde.append(a["de2000_median"] - b["de2000_median"])
            which[max(PER_IMAGE, key=lambda k: abs(a["params"][k] - b["params"][k]) / SCALES[k])] += 1
        print(f"| {name} | {med(sh):.2f} / {max(sh):.2f} | {dict(which)} | {med(dde):+.2f} / {max(dde, key=abs):+.2f} |")

    print("\n## 反差上界\n")
    print("| 配置 | ΔE 中位数 | 相对上界 2.0 的 ΔE 变化 中位 / 最好 | contrast 中位 / 最大 | 仍顶到新上界的张数 | 截负值% 最大 | 高于白% 最大 |")
    print("|---|---|---|---|---|---|---|")
    for name in [c for c in by if c.startswith("contrast")]:
        ref = "lam=" + name.split("lam=")[1]
        rs = [by[name][s]["fit"] for s in stems]
        d = [by[name][s]["fit"]["de2000_median"] - by[ref][s]["fit"]["de2000_median"] for s in stems]
        print(f"| {name} | {med(r['de2000_median'] for r in rs):.2f} | {med(d):+.2f} / {min(d):+.2f} | "
              f"{med(r['params']['contrast'] for r in rs):.2f} / {max(r['params']['contrast'] for r in rs):.2f} | "
              f"{sum('contrast' in r['at_bound'] for r in rs)} | {100 * max(r['diagnostics']['below_zero_after_colour'] for r in rs):.1f} | "
              f"{100 * max(r['diagnostics']['above_white'] for r in rs):.1f} |")

    print("\n## 输入子窗口（85%，相对整图 λ=0.05 的解）\n")
    print("| 帧 | 3 个子窗口的参数最大变化（尺度单位） | ΔE 中位数 整图 / 子窗口 |")
    print("|---|---|---|")
    allsh = []
    for s in stems:
        b = by[BASE][s]["fit"]
        sh = [shift(by[f"window{w}"][s]["fit"]["params"], b["params"]) for w in range(3)]
        allsh += sh
        print(f"| {s} | {' / '.join(f'{x:.2f}' for x in sh)} | {b['de2000_median']:.2f} / "
              f"{' / '.join(f'{by[f'window{w}'][s]['fit']['de2000_median']:.2f}' for w in range(3))} |")
    print(f"\n子窗口参数变化：中位 {med(allsh):.2f}，最大 {max(allsh):.2f}（尺度单位）")


if __name__ == "__main__":
    main(sys.argv[1])
