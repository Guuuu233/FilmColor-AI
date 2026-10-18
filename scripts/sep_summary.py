"""Summary of the base colour / level separation (experiments/s3_separate/DESIGN.md): sep_check.py records + sep_a7.py.

Groups: S3 diagnosis rolls (shop prints, user edits), validation v1 rolls, BlueNeg intact rolls (registration
NCC < 0.7 excluded as PREREG_BLUENEG.md). Per group, variant and reading (fixed look / level matched / P4): medians
of dE2000, a*b* distance, dL*, share below black; differences to B0 (median, improved, worse by > 2). Subsets by how
far the per-frame base is from the roll base: chroma part (max |c0 - c1| over channels, the "colour" that L0C1
replaces) and level part (|l0 - l1|, what L1C0 replaces), split at 0.1 D. Reading rules of DESIGN.md section 6 are
applied to the level-matched medians of the subset where both parts differ.

  sep_summary.py CHECK_DIR A7_JSON OUT_JSON OUT_MD
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

VARIANTS = ("B0", "B1", "L0C1", "L1C0")
READINGS = ("fixed_look", "matched", "p4")
EXCLUDE = {"blueneg:199807A-21-orlando-disney"}  # NCC 0.575 < 0.7
SPLIT = 0.1


def group_of(r):
    if r["id"].startswith("blueneg:"):
        return "BlueNeg 冲印"
    if r["roll"] in ("own-e5207-20240401-dsc6233", "shop_r01"):
        return "v1 两卷 店家"
    return "S3 诊断三卷 " + ("店家" if r["ref_kind"] == "shop" else "用户")


def med(x):
    return round(float(np.median(x)), 3) if len(x) else None


def defined(r, v):
    return not r["variants"][v].get("undefined")


def de(r, v, rd):
    """dE of a variant; an undefined inversion counts as infinitely worse."""
    return r["variants"][v][rd]["de2000_median"] if defined(r, v) else float("inf")


def stats(rows):
    """Medians over the frames where the variant is defined; undefined ones are counted, and count as worse > 2."""
    out = {}
    for v in VARIANTS:
        ok = [r for r in rows if defined(r, v)]
        for rd in READINGS:
            out[f"{v}/{rd}"] = dict(n=len(rows), undefined=len(rows) - len(ok),
                                    dE=med([de(r, v, rd) for r in ok]),
                                    ab=med([r["variants"][v][rd]["ab_median"] for r in ok]),
                                    dL=med([r["variants"][v][rd]["dL_median"] for r in ok]),
                                    below_black=med([r["variants"][v]["below_black_any"] for r in ok]))
            if v != "B0":
                d = [de(r, v, rd) - de(r, "B0", rd) for r in rows]
                out[f"{v}/{rd}"].update(d_vs_B0=med([x for x in d if np.isfinite(x)]), better=int(sum(x < 0 for x in d)),
                                        worse2=int(sum(x > 2 for x in d)))
    return out


def reading_rule(s):
    """DESIGN.md section 6 on level-matched medians: which part of the roll base carries the effect."""
    g = lambda v: s[f"{v}/matched"]["dE"]
    if any(g(v) is None for v in VARIANTS):
        return None
    colour = abs(g("L0C1") - g("B1")) < abs(g("L0C1") - g("B0")) and abs(g("L1C0") - g("B0")) < abs(g("L1C0") - g("B1"))
    level = abs(g("L1C0") - g("B1")) < abs(g("L1C0") - g("B0")) and abs(g("L0C1") - g("B0")) < abs(g("L0C1") - g("B1"))
    return "colour" if colour and not level else "level" if level and not colour else "mixed"


def main(check_dir, a7_json, out_json, out_md):
    rows = [json.loads(p.read_text()) for p in sorted(Path(check_dir).glob("*.json"))]
    rows = [r for r in rows if r["id"] not in EXCLUDE]
    for r in rows:
        r["chroma_part"] = float(max(abs(x) for x in r["chroma_diff"]))
        r["level_part"] = abs(r["level_diff"])
    groups = defaultdict(list)
    for r in rows:
        groups[group_of(r)].append(r)
    res = dict(n=len(rows), groups={})
    for g, rs in sorted(groups.items()):
        subsets = {
            "all": rs,
            "chroma_part>=0.1": [r for r in rs if r["chroma_part"] >= SPLIT],
            "level_part>=0.1": [r for r in rs if r["level_part"] >= SPLIT],
            "both>=0.1": [r for r in rs if r["chroma_part"] >= SPLIT and r["level_part"] >= SPLIT],
            "both<0.1": [r for r in rs if r["chroma_part"] < SPLIT and r["level_part"] < SPLIT],
        }
        res["groups"][g] = {k: dict(n=len(v), stats=stats(v) if v else None,
                                   rule=reading_rule(stats(v)) if v else None) for k, v in subsets.items()}
        worst = sorted(rs, key=lambda r: -max(de(r, v, "matched") - de(r, "B0", "matched") for v in VARIANTS[1:]))[:5]
        res["groups"][g]["worst_matched"] = [dict(id=r["id"], ref=r["ref_kind"], **{
            v: round(de(r, v, "matched"), 2) if defined(r, v) else "undefined" for v in VARIANTS}) for r in worst]
    a7 = json.loads(Path(a7_json).read_text())
    res["a7r5"] = {s: {v: {k: round(x, 3) for k, x in fr[v].items()
                           if k.endswith(("dE_vs_BM", "ab_vs_BM", "below_black_any"))} | dict(D_vs_BM=fr[v]["D_vs_BM"])
                       for v in fr} for s, fr in a7["frames"].items()}
    Path(out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1))

    L = ["# 片基颜色 / 亮度分离：结果表（`sep_summary.py` 生成）", ""]
    for g, sub in res["groups"].items():
        L += [f"## {g}", ""]
        for k in ("all", "chroma_part>=0.1", "level_part>=0.1", "both>=0.1", "both<0.1"):
            s = sub[k]
            if not s["n"]:
                L += [f"- {k}：0 帧", ""]
                continue
            L += [f"**{k}**（{s['n']} 帧；明度对齐口径的解读：{s['rule']}）", "",
                  "| 变体 | 读法 | ΔE 中位数 | a\\*b\\* 中位数 | ΔL\\* 中位数 | 黑点以下 | 相对 B0 的 Δ 中位数 | 变好 | 变差 > 2（含无定义） | 无定义 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for v in VARIANTS:
                for rd in READINGS:
                    x = s["stats"][f"{v}/{rd}"]
                    L.append(f"| {v} | {rd} | {x['dE']} | {x['ab']} | {x['dL']} | {x['below_black']} | "
                             f"{x.get('d_vs_B0', '–')} | {x.get('better', '–')} | {x.get('worse2', '–')} | {x['undefined']} |")
            L.append("")
        L += ["明度对齐口径下相对 B0 变差最多的帧：", "", "| 帧 | 参考 | B0 | B1 | L0C1 | L1C0 |", "|---|---|---|---|---|---|"]
        L += [f"| {w['id']} | {w['ref']} | {w['B0']} | {w['B1']} | {w['L0C1']} | {w['L1C0']} |" for w in sub["worst_matched"]]
        L.append("")
    L += ["## A7R5（与实测片基 BM 的渲染结果比较，明度对齐口径）", "",
          "| 帧 | " + " | ".join(f"{v} ΔE / ab / 黑点下" for v in ("B0", "B1", "L0C1", "L1C0")) + " |",
          "|---|---|---|---|---|"]
    for s, fr in res["a7r5"].items():
        L.append(f"| {s} | " + " | ".join(f"{fr[v]['matched_dE_vs_BM']} / {fr[v]['matched_ab_vs_BM']} / "
                                         f"{fr[v]['below_black_any']}" for v in ("B0", "B1", "L0C1", "L1C0")) + " |")
    Path(out_md).write_text("\n".join(L) + "\n")
    print(json.dumps({g: {k: (v["n"], v["rule"]) for k, v in sub.items() if k != "worst_matched"}
                      for g, sub in res["groups"].items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
