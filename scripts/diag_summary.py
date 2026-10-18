"""Base-inversion diagnosis: tables for experiments/s3_diag/REPORT.md (written to TABLES.md and summary.json).

Reads the records of diag_roll_base.py, diag_variants.py (main and b1 sets), diag_two_refs.py and diag_rollcheck.py.
Frame-level comparisons are descriptive (counts and medians per roll); no significance is claimed.

  diag_summary.py DIAG_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

FRAMES = ("A002", "A008", "A019", "A023", "A140", "A196")
ROLL_NAMES = {"own-e5207-20240401-dsc6161": "5207 dsc6161", "own-e5207-20240323-dsc8242": "5207 dsc8242",
              "own-g200-20240320-dsc4889": "Gold 200"}
REACHED = 0.1  # G density of the frame's own base above the roll base below which the frame "reached the base"
COLOUR_SPLIT = 0.1  # exploratory split on the base chromaticity disagreement, chosen after seeing the roll check


def colour_error(x):
    o = x["own_minus_roll_density"]
    return max(abs(o[0] - o[1]), abs(o[2] - o[1]))


def load(p):
    return json.loads(Path(p).read_text())


def f2(x):
    return "–" if x is None else f"{x:.2f}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main(d):
    d = Path(d)
    md, summary = [], {}
    roll = load(d / "roll_base.json")
    var = {}
    for p in sorted((d / "variants").glob("*.json")):
        stem, ref = p.stem.split("_")[:2]
        var.setdefault((stem, ref), {}).update(load(p)["variants"])
    refs = [(s, r) for s in FRAMES for r in ("shop", "user") if (s, r) in var]

    # 1. base offsets of the diagnosis frames
    rows = []
    for s in FRAMES:
        for rname, r in roll["rolls"].items():
            if s in r["frames"]:
                fr = r["frames"][s]
                rows.append([s, ROLL_NAMES[rname], *(f"{x:+.2f}" for x in fr["own_minus_roll_density"]),
                             *(f"{x:+.2f}" for x in fr["own_colour_error_RG_BG"])])
    md += ["## T1 逐帧片基比整卷片基密多少（D，>0 = 更密）",
           table(["帧", "卷", "R", "G", "B", "R−G", "B−G"], rows)]

    # 2. dE2000 medians per variant
    cols = [("B0", ""), ("B1", ""), ("B1c", ""), ("B0w4", ""), ("B1w4", ""), ("MCt_B1", ""), ("MCp_B1", "")]
    for fit in ("p4", "ub7", "fixed_look"):
        rows = [[f"{s} {('店家' if r == 'shop' else '用户')}", *(f2(var[(s, r)][v][fit]["de2000_median"]) for v, _ in cols)]
                for s, r in refs]
        md += [f"## T2-{fit} ΔE2000 中位数", table(["帧 / 参考", *(v for v, _ in cols)], rows)]

    # 3. residual colour after ub7, B0 vs B1
    rows = []
    for s, r in refs:
        row = [f"{s} {('店家' if r == 'shop' else '用户')}"]
        for v in ("B0", "B1"):
            m = var[(s, r)][v]["ub7"]
            row += [f"{m['dL_median']:+.1f}", f"{m['da_mean']:+.1f}", f"{m['db_mean']:+.1f}", f2(m["dh_abs_median_deg"]),
                    f"{100 * m['clipped_where_ref_not_frac']:.2f}%"]
        rows.append(row)
    md += ["## T3 7 参数上限之后的残差（B0 | B1）：ΔL* 中位数、Δa* Δb* 均值、|Δh| 中位数（°）、参考未裁切处我们裁切的像素",
           table(["帧 / 参考", "B0 ΔL", "Δa", "Δb", "|Δh|", "裁切", "B1 ΔL", "Δa", "Δb", "|Δh|", "裁切"], rows)]
    rows = []
    for s, r in refs:
        for v in ("B0", "B1"):
            z = var[(s, r)][v]["ub7"]["zones"]
            rows.append([f"{s} {('店家' if r == 'shop' else '用户')}", v,
                         *(f"{z[k]['share']:.2f} / {f2(z[k].get('de2000_median'))} / {z[k].get('db_mean', float('nan')):+.1f}"
                           for k in ("shadow", "mid", "high"))])
    md += ["## T4 分影调（参考 L* <35 / 35–70 / >70）：像素占比 / ΔE 中位数 / Δb* 均值（7 参数上限）",
           table(["帧 / 参考", "反转", "暗部", "中间调", "高光"], rows)]
    rows = []
    for s, r in refs:
        for v in ("B0", "B1"):
            bins = var[(s, r)][v]["ub7"]["density_bins"]
            rows.append([f"{s} {('店家' if r == 'shop' else '用户')}", v,
                         *(f"{b['da_mean']:+.1f} / {b['db_mean']:+.1f}" for b in bins)])
    md += ["## T5 按负片 G 密度五等分（低 → 高 = 正片暗 → 亮）的 Δa* / Δb* 均值（7 参数上限）",
           table(["帧 / 参考", "反转", "1", "2", "3", "4", "5"], rows)]

    # 4. diagnostic fits that see the reference
    rows = []
    for s, r in refs:
        row = [f"{s} {('店家' if r == 'shop' else '用户')}"]
        for v in ("B0", "B1"):
            x = var[(s, r)][v]
            g = x["ub7_chan_gamma"]["chan_gamma"]
            row += [f2(x["ub7"]["de2000_median"]), f2(x["ub7_chan_gamma"]["de2000_median"]), f"{g[0]:.2f}/{g[2]:.2f}",
                    f2(x["ub7_p_matrix"]["de2000_median"])]
        rows.append(row)
    md += ["## T6 诊断拟合（看参考图）：7 参数上限，加逐通道反差（拟合的 R/B 系数），加 P 空间 3×3 矩阵",
           table(["帧 / 参考", "B0 ub7", "+反差", "R/B", "+矩阵", "B1 ub7", "+反差", "R/B", "+矩阵"], rows)]

    # 5. two references
    tr = load(d / "two_refs.json")
    rows = []
    for s, fr in tr.items():
        row = [s, f"{fr['prints_de2000_median']:.1f}"]
        for k in ("B0_ub7", "B1_ub7"):
            a = fr[k]["all"]
            row += [f"{a['a']['shared_rms']:.1f}/{a['b']['shared_rms']:.1f}", f"{a['a']['split_rms']:.1f}/{a['b']['split_rms']:.1f}",
                    f"{a['a']['corr']:+.2f}/{a['b']['corr']:+.2f}"]
        rows.append(row)
    md += ["## T7 两种参考共有与分歧的残差（a*/b*，7 参数上限各自拟合后）",
           table(["帧", "两成品之间 ΔE", "B0 共有 RMS", "B0 分歧 RMS", "B0 相关", "B1 共有 RMS", "B1 分歧 RMS", "B1 相关"], rows)]

    # 6. roll-wide check
    rc = [load(p) for p in sorted((d / "rollcheck").glob("*.json"))]
    if rc:
        rows, rows_g, sm = [], [], {}
        for rname in ROLL_NAMES:
            for ref in ("shop", "user"):
                recs = [x for x in rc if x["roll"] == rname and x["ref_kind"] == ref]
                if not recs:
                    continue
                for group, sel in (("全部", lambda x: True), ("已碰到片基", lambda x: x["own_minus_roll_density"][1] < REACHED),
                                   ("未碰到片基", lambda x: x["own_minus_roll_density"][1] >= REACHED)):
                    g = [x for x in recs if sel(x)]
                    if not g:
                        continue
                    row = [ROLL_NAMES[rname], "店家" if ref == "shop" else "用户", group, len(g)]
                    for fit in ("p4", "ub7"):
                        med = {v: float(np.median([x["variants"][v][fit]["de2000_median"] for x in g])) for v in ("B0", "B0w4", "B1", "B1w4")}
                        dd = np.array([x["variants"]["B1"][fit]["de2000_median"] - x["variants"]["B0"][fit]["de2000_median"] for x in g])
                        row += [*(f2(med[v]) for v in ("B0", "B0w4", "B1", "B1w4")),
                                f"{int((dd < 0).sum())}/{len(dd)}", f"{dd.max():+.2f}"]
                        sm[f"{rname}|{ref}|{group}|{fit}"] = dict(n=len(g), median=med, b1_better=int((dd < 0).sum()),
                                                                 worst_b1_minus_b0=float(dd.max()))
                    rows.append(row)
                gam = np.array([x["variants"]["B1"]["ub7_chan_gamma"]["chan_gamma"] for x in recs])
                gain = np.array([x["variants"]["B1"]["ub7"]["de2000_median"] - x["variants"]["B1"]["ub7_chan_gamma"]["de2000_median"] for x in recs])
                rows_g.append([ROLL_NAMES[rname], "店家" if ref == "shop" else "用户", len(recs),
                               *(f"{np.median(gam[:, i]):.2f} [{np.percentile(gam[:, i], 25):.2f}, {np.percentile(gam[:, i], 75):.2f}]" for i in (0, 2)),
                               f"{int((gam[:, 0] > 1).sum())}/{len(gam)}", f"{int((gam[:, 2] > 1).sum())}/{len(gam)}", f"{np.median(gain):.2f}"])
        # exploratory split (chosen after seeing the roll results): chromaticity disagreement of the frame's own base
        # with the roll base, a reference-free quantity
        rows_c = []
        for label, sel in ((f"≥ {COLOUR_SPLIT} D", lambda e: e >= COLOUR_SPLIT), (f"< {COLOUR_SPLIT} D", lambda e: e < COLOUR_SPLIT)):
            for rname in (*ROLL_NAMES, None):
                g = [x for x in rc if (rname is None or x["roll"] == rname) and sel(colour_error(x))]
                if not g:
                    continue
                for fit in ("p4", "ub7"):
                    dd = np.array([x["variants"]["B1"][fit]["de2000_median"] - x["variants"]["B0"][fit]["de2000_median"] for x in g])
                    rows_c.append([label, ROLL_NAMES.get(rname, "三卷合计"), fit, len(g), f"{int((dd < 0).sum())}/{len(dd)}",
                                   f"{np.median(dd):+.2f}", f"{dd.min():+.2f}", f"{dd.max():+.2f}"])
        md += ["## T10 按逐帧片基与整卷片基的色度差分层（max(|R−G|, |B−G|)，探索性：分界看过结果后才定）：B1 − B0 的 ΔE 变化（帧×参考）",
               table(["色度差", "卷", "拟合", "组数", "B1 更好", "中位数", "最好", "最差"], rows_c)]
        md += ["## T8 整卷检验：各卷 ΔE2000 中位数（B0 / B0w4 / B1 / B1w4），B1 优于 B0 的帧数，B1 相对 B0 的最大变差",
               table(["卷", "参考", "组", "帧", "P4 B0", "B0w4", "B1", "B1w4", "B1 更好", "最大变差",
                      "ub7 B0", "B0w4", "B1", "B1w4", "B1 更好", "最大变差"], rows),
               "## T9 整卷：B1 上参考要求的逐通道反差系数（中位数 [四分位]），以及它带来的 ΔE 下降中位数",
               table(["卷", "参考", "帧", "R 系数", "B 系数", "R>1", "B>1", "ΔE 下降"], rows_g)]
        summary["rollcheck"] = sm
    (d / "TABLES.md").write_text("\n\n".join(md) + "\n")
    (d / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print("wrote", d / "TABLES.md", len(rc), "rollcheck records")


if __name__ == "__main__":
    main(*sys.argv[1:])
