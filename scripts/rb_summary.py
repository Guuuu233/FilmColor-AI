"""Roll-base validation v1: score the pre-registered rule R1 (experiments/s3_rollbase_v1/PREREG.md).

Reads the diag_rollcheck.py outputs (B0 and B1 per frame against the shop print). R1 takes B1 where the base
chroma disagreement max(|o_R - o_G|, |o_B - o_G|) >= 0.1 D (o = log10(roll base / own base)), else B0. Primary
metric: fixed-look dE2000 median; Delta = R1 - B0 per frame. Criteria exactly as PREREG section 6.

  rb_summary.py ROLLCHECK_DIR OUT_JSON OUT_MD
"""
import json
import sys
from pathlib import Path

import numpy as np

SPLIT, SEVERE, CLIP_PP, MIN_TRIG = 0.1, 2.0, 0.02, 3


def colour_error(o):
    return max(abs(o[0] - o[1]), abs(o[2] - o[1]))


def frame_row(x):
    c = colour_error(x["own_minus_roll_density"])
    trig = c >= SPLIT
    b0, b1 = x["variants"]["B0"], x["variants"]["B1"]
    r1 = b1 if trig else b0
    row = dict(id=x["id"], roll=x["roll"], chroma_disagreement=round(c, 4), triggered=trig,
               own_minus_roll_density=[round(v, 3) for v in x["own_minus_roll_density"]])
    for name, v in (("B0", b0), ("B1", b1), ("R1", r1)):
        f = v["fixed_look"]
        row[name] = dict(fixed=round(f["de2000_median"], 3), p4=round(v["p4"]["de2000_median"], 3),
                         ub7=round(v["ub7"]["de2000_median"], 3), dL=round(f["dL_median"], 2),
                         dh=None if f["dh_abs_median_deg"] is None else round(f["dh_abs_median_deg"], 1),
                         below_black=[round(b, 4) for b in v["below_black"]],
                         clipped=round(f.get("clipped_where_ref_not_frac", float("nan")), 4),
                         zones={k: round(z["de2000_median"], 2) for k, z in f["zones"].items() if "de2000_median" in z})
    row["delta_fixed"] = round(row["R1"]["fixed"] - row["B0"]["fixed"], 3)
    row["delta_p4"] = round(row["R1"]["p4"] - row["B0"]["p4"], 3)
    row["delta_fixed_B1"] = round(row["B1"]["fixed"] - row["B0"]["fixed"], 3)
    row["new_clipping"] = bool(max(np.subtract(row["R1"]["below_black"], row["B0"]["below_black"])) > CLIP_PP)
    return row


def judge(rows):
    trig = [r for r in rows if r["triggered"]]
    d_all = [r["delta_fixed"] for r in rows]
    d_trig = [r["delta_fixed"] for r in trig]
    severe = [r["id"] for r in rows if r["delta_fixed"] > SEVERE]
    clip = [r["id"] for r in rows if r["new_clipping"]]
    s = dict(frames=len(rows), triggered=len(trig), trigger_rate=round(len(trig) / len(rows), 3),
             median_delta_all=round(float(np.median(d_all)), 3),
             median_delta_triggered=round(float(np.median(d_trig)), 3) if trig else None,
             improved_triggered=sum(d < 0 for d in d_trig), severe=severe, new_clipping=clip,
             median_delta_B1_all=round(float(np.median([r["delta_fixed_B1"] for r in rows])), 3),
             median_delta_p4_triggered=round(float(np.median([r["delta_p4"] for r in trig])), 3) if trig else None,
             median_fixed=dict((k, round(float(np.median([r[k]["fixed"] for r in rows])), 3)) for k in ("B0", "R1", "B1")),
             worst=sorted(rows, key=lambda r: -r["delta_fixed"])[:3])
    s["worst"] = [dict(id=r["id"], delta_fixed=r["delta_fixed"], triggered=r["triggered"],
                       chroma_disagreement=r["chroma_disagreement"]) for r in s["worst"]]
    if len(trig) >= MIN_TRIG:
        checks = dict(median_triggered_below_0=s["median_delta_triggered"] < 0,
                      two_thirds_improved=s["improved_triggered"] >= 2 * len(trig) / 3,
                      at_most_one_severe=len(severe) <= 1, median_all_le_0_1=s["median_delta_all"] <= 0.1)
        s["kind"] = "evidence"
    else:
        checks = dict(no_severe=not severe, no_new_clipping=not clip)
        s["kind"] = "harm-only"
    s["checks"], s["passed"] = checks, all(checks.values())
    return s


def main(rc_dir, out_json, out_md):
    rows = [frame_row(json.loads(p.read_text())) for p in sorted(Path(rc_dir).glob("*_shop.json"))]
    rolls = sorted({r["roll"] for r in rows})
    per = {roll: judge([r for r in rows if r["roll"] == roll]) for roll in rolls}
    stage = all(s["passed"] for s in per.values()) and any(s["kind"] == "evidence" for s in per.values())
    verdict = "passed" if stage else ("insufficient evidence" if all(s["passed"] for s in per.values()) else "failed")
    Path(out_json).write_text(json.dumps(dict(prereg="experiments/s3_rollbase_v1/PREREG.md", verdict=verdict,
                                              rolls=per, frames=rows), indent=1, ensure_ascii=False))
    L = [f"# 卷级片基验证 v1：结果表（`rb_summary.py` 生成）", "", f"阶段判定：**{verdict}**", ""]
    for roll, s in per.items():
        L += [f"## {roll}", "",
              f"- 帧数 {s['frames']}，触发 {s['triggered']}（{s['trigger_rate']:.0%}），判定类型 {s['kind']}，"
              f"{'通过' if s['passed'] else '未通过'}：{s['checks']}",
              f"- 固定风格 ΔE 中位数：B0 {s['median_fixed']['B0']}、R1 {s['median_fixed']['R1']}、B1 {s['median_fixed']['B1']}",
              f"- Δ（R1 − B0）中位数：全部帧 {s['median_delta_all']}，触发帧 {s['median_delta_triggered']}，"
              f"触发帧变好 {s['improved_triggered']}/{s['triggered']}；B1 全换的 Δ 中位数 {s['median_delta_B1_all']}；"
              f"触发帧 P4 的 Δ 中位数 {s['median_delta_p4_triggered']}",
              f"- 严重退化（Δ > +{SEVERE}）：{s['severe'] or '无'}；新增裁切：{s['new_clipping'] or '无'}",
              f"- 最差 3 帧：{s['worst']}", "",
              "| 帧 | 色度差 | 触发 | 固定 B0 | 固定 R1 | Δ | 固定 B1 | P4 B0 | P4 R1 | ub7 B0 | ub7 R1 | ΔL B0→R1 | \\|Δh\\| B0→R1 | 黑点下 B0 | 黑点下 R1 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in (r for r in rows if r["roll"] == roll):
            b0, r1 = r["B0"], r["R1"]
            L.append(f"| {r['id']} | {r['chroma_disagreement']:.2f} | {'是' if r['triggered'] else ''} | {b0['fixed']:.2f} | "
                     f"{r1['fixed']:.2f} | {r['delta_fixed']:+.2f} | {r['B1']['fixed']:.2f} | {b0['p4']:.2f} | {r1['p4']:.2f} | "
                     f"{b0['ub7']:.2f} | {r1['ub7']:.2f} | {b0['dL']:+.1f}→{r1['dL']:+.1f} | {b0['dh']}→{r1['dh']} | "
                     f"{max(b0['below_black']):.1%} | {max(r1['below_black']):.1%} |")
        L.append("")
    Path(out_md).write_text("\n".join(L) + "\n")
    print(verdict)
    for roll, s in per.items():
        print(roll, json.dumps({k: v for k, v in s.items() if k != "worst"}, ensure_ascii=False), s["worst"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
