"""T5b §4: write data/manifests/material_polarity_20261105.json, positives filed as negatives (data only).
Usage: python -I scripts/t5b_material_polarity.py
Single files get their SHA256 through check_source(allow_frozen=False) (read for hashing only, not decoded); folders
get their NEF count. The material itself is read-only and is not moved. The evidence is the research side's
(T5a audit, 3rd edition, 2026-11-05) and the committed T5a device / Batch 19-21 records (d_hi_G).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filmcolor.frozen import check_source  # noqa: E402

MATERIAL = Path("/Volumes/waizhiyingpan/胶片负片素材")
OUT = ROOT / "data/manifests/material_polarity_20261105.json"
T5A = ROOT / "experiments/t4_auto_ui/t5a"
RESEARCH = "research (FilmColor_AI_T5a_审计_2026-11-05 第 3 版)"


def d_hi_g():
    dev = json.loads((T5A / "devices_candidate.json").read_text())["new_tiff"]
    bat = json.loads((T5A / "batches_candidate.json").read_text())["frames"]
    out = {"01_彩色负片_未反相/A850.tif": dev["A850.tif"]["inversion"]["d_hi"][1]}
    for rel in ("01_彩色负片_未反相/A851.tif", "01_彩色负片_未反相/A852.tif", "02_黑白负片_未反相/B094.tif"):
        out[rel] = bat[rel]["inversion"]["d_hi"][1]
    return out


def main():
    dg = d_hi_g()
    entries = []
    for rel in ("01_彩色负片_未反相/A850.tif", "01_彩色负片_未反相/A851.tif", "01_彩色负片_未反相/A852.tif"):
        entries.append(dict(rel=rel, kind="positive_image", sha256=check_source(MATERIAL / rel, allow_frozen=False),
                            evidence=f"不反相时颜色正常；T5a d_hi_G {dg[rel]:.2f} D（input_density_implausible）",
                            checked_by=RESEARCH))
    for rel in ("02_黑白负片_未反相/B094.tif", "02_黑白负片_未反相/B040.tif"):
        ev = (f"不反相时天空亮、水面暗；T5a d_hi_G {dg[rel]:.2f} D，manual_review_required" if rel in dg else
              "与 B094 像素统计相同（SHA256 不同）；不反相时天空亮、水面暗")
        entries.append(dict(rel=rel, kind="positive_image", sha256=check_source(MATERIAL / rel, allow_frozen=False),
                            evidence=ev + "；来自 pixls 帖子 52582，标题写 “B&W Negative”，文件内容是正像",
                            checked_by=RESEARCH))
    for d in ("Eastman 5294(尼康数字化)-", "Eastman 5294(尼康数字化)-1", "Eastman 5294(尼康数字化)-2"):
        entries.append(dict(rel=d, kind="reversal_film",
                            n_nef=sum(p.suffix.lower() == ".nef" for p in (MATERIAL / d / "原件").iterdir()),
                            evidence="Kodak 5294 为 EKTACHROME 100D 彩色反转片；机内 JPG 为正像",
                            checked_by="user (2026-11-05) + " + RESEARCH))
    d = "foma pan100(d67 尼康数字化)-"
    entries.append(dict(rel=d, kind="reversal_film",
                        n_nef=sum(p.suffix.lower() == ".nef" for p in (MATERIAL / d / "原件").iterdir()),
                        evidence="黑白负片反转冲洗：`逐张调整灰阶` 26/26 张成片（DSC_7400–7425）与反相输出明暗相反，"
                                 "NCC 0.84–0.98，比第二候选高 ≥ 0.15；同卷另一目录名为“负片正冲”"
                                 "（foma pan100(d67 负片正冲 尼康数字化)-，只有 12 张成片，没有原片）",
                        checked_by=RESEARCH))
    doc = dict(id="material_polarity_20261105", created="2026-11-05",
               purpose="误归为负片的正片；挑选负片盲测和诊断素材时排除。素材只读，未移动。",
               root=str(MATERIAL), kinds=dict(positive_image="已反相的正像（edited / 转换后的图像）",
                                              reversal_film="反转片或反转冲洗的胶片，原片即正像"),
               entries=entries)
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(doc, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
