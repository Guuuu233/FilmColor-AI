"""Side-by-side check of S3 labels on the aligned 1/8 grid: reference, fixed look, and the renders of the
P4 / P3 / 7-parameter ceiling fits (sRGB previews), one row per frame. For looking at labels and worst cases.

  s3_label_sheet.py LABEL_DIR OUT_JPG ID [ID ...]      (ID as in the frame list, e.g. ext:A002)
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from filmcolor.colorimetry import lab_to_xyz, xyz_to_srgb_preview
from filmcolor.fit import render_lab
from filmcolor.frozen import load_cache
from filmcolor.render import RenderParams

COLS = ("reference", "fixed_look", "p4", "p3", "ub7")
W = 360


def preview(lab):
    rgb = np.clip(xyz_to_srgb_preview(lab_to_xyz(lab)), 0, 1)
    im = Image.fromarray((rgb * 255 + 0.5).astype(np.uint8))
    return im.resize((W, int(W * im.height / im.width)))


def main(label_dir, out, *ids):
    rows = []
    for fid in ids:
        domain, stem = fid.split(":", 1)
        domain = {"ext": "ext_own"}.get(domain, domain)
        lab = json.loads((Path(label_dir) / domain / f"{stem}.json").read_text())
        cache = Path(lab["cache"]["dir"])
        P, R = load_cache(cache / f"{stem}_P8.npy"), load_cache(cache / f"{stem}_R8.npy")
        tiles = [preview(R)]
        for k in COLS[1:]:
            tiles.append(preview(render_lab(P, RenderParams(**lab[k]["params"]))[0]))
        h = tiles[0].height + 16
        row = Image.new("RGB", (W * len(COLS), h), "white")
        d = ImageDraw.Draw(row)
        for i, (k, t) in enumerate(zip(COLS, tiles)):
            row.paste(t, (i * W, 16))
            txt = k if k == "reference" else f"{k} dE med {lab[k]['de2000_median']:.1f} mean {lab[k]['de2000_mean']:.1f}"
            d.text((i * W + 4, 2), (fid + "  " if i == 0 else "") + txt, fill="black")
        rows.append(row)
    sheet = Image.new("RGB", (W * len(COLS), sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    print(out)


if __name__ == "__main__":
    main(*sys.argv[1:])
