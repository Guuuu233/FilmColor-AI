"""A7R5 rebate frames: where a7_measure.py measured (review of 2026-10-18; read-only, images not in git).

On the 1/8 linear caches (gamma-encoded for display): rebate rectangles (cyan, a7_measure.REBATE), picture interior
(yellow, a7_measure.IMAGE, the interior-only base search), backlight pixels (red: >= 95 % of the frame's maximum G,
the pixels a7_measure.py takes as light without film). One contact sheet with the six frames.

  a7_roi_sheet.py OUT_JPG
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from a7_measure import IMAGE, REBATE  # noqa: E402

from filmcolor.frozen import load_cache  # noqa: E402
from filmcolor.paths import data_dir  # noqa: E402

TILE_W = 900


def tile(stem):
    T = load_cache(data_dir("s3_a7") / f"{stem}_T8.npy").astype(np.float64)
    h, w = T.shape[:2]
    v = (np.clip(T / np.percentile(T, 99.9), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    bl = T[..., 1] >= 0.95 * T[..., 1].max()
    v[bl] = (255, 0, 0)
    img = Image.fromarray(v)
    d = ImageDraw.Draw(img)
    box = lambda r: (w * r[0] / 100, h * r[2] / 100, w * r[1] / 100, h * r[3] / 100)
    for side, r in REBATE[stem].items():
        d.rectangle(box(r), outline=(0, 255, 255), width=4)
    d.rectangle(box(IMAGE[stem]), outline=(255, 255, 0), width=4)
    img = img.resize((TILE_W, int(h * TILE_W / w)))
    ImageDraw.Draw(img).text((8, 8), f"{stem}  backlight px {int(bl.sum())}", fill=(255, 255, 255))
    return img


def main(out):
    tiles = [tile(s) for s in REBATE]
    cols = 2
    th = max(t.height for t in tiles)
    sheet = Image.new("RGB", (cols * TILE_W, ((len(tiles) + 1) // cols) * th), "black")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * TILE_W, (i // cols) * th))
    sheet.save(out, quality=88)


if __name__ == "__main__":
    main(*sys.argv[1:2])
