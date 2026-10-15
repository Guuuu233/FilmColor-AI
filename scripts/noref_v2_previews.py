"""Preview sheets for the S2-7 report: default | noref_v2 (global_look_v1) | v1 reference fit | reference,
one row per paired development frame, plus a single downsized contact sheet.

  noref_v2_previews.py ABLATION_JSON OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
from noref_eval import SETS  # noqa: E402

from filmcolor.colorimetry import lab_to_xyz, xyz_to_srgb_preview  # noqa: E402
from filmcolor.compare import resize_bilinear  # noqa: E402
from filmcolor.render import RenderParams, render  # noqa: E402

ROW_H = 160


def main(abl, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    L = json.load(open(abl))
    final = {(r["set"], r["stem"]): r["estimate"]["params"] for r in L["stages"]["A"]["results"]}
    rows = []
    for (w, stem), params in final.items():
        cache = SETS[w]["eval_cache"]
        if not (Path(cache) / f"{stem}_R8.npy").exists():
            continue
        P8, R8 = np.load(f"{cache}/{stem}_P8.npy"), np.load(f"{cache}/{stem}_R8.npy")
        cols = [render(P8, RenderParams())[0], render(P8, RenderParams(**params))[0],
                render(P8, RenderParams(**L["v1_params"][w][stem]))[0], np.clip(xyz_to_srgb_preview(lab_to_xyz(R8)), 0, 1)]
        gap = np.ones((P8.shape[0], 4, 3))
        sheet = np.concatenate(sum([[c, gap] for c in cols], [])[:-1], 1)
        tifffile.imwrite(out_dir / f"{stem}.tif", (np.clip(sheet, 0, 1) * 255 + 0.5).astype(np.uint8))
        h, wd = sheet.shape[:2]
        rows.append(resize_bilinear(sheet, (ROW_H, round(wd * ROW_H / h))))
    width = max(r.shape[1] for r in rows)
    pad = [np.concatenate([r, np.ones((ROW_H, width - r.shape[1], 3))], 1) for r in rows]
    contact = np.concatenate(sum([[r, np.ones((4, width, 3))] for r in pad], [])[:-1], 0)
    tifffile.imwrite(out_dir / "contact.tif", (np.clip(contact, 0, 1) * 255 + 0.5).astype(np.uint8))
    print(len(rows), "frames ->", out_dir)


if __name__ == "__main__":
    main(*sys.argv[1:])
