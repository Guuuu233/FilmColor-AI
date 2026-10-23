"""T3a viewing aid: enlarged edge bands of the 1/8 caches, for placing the rebate, backlight and picture boxes by eye
(read-only; images git-ignored). No statistic of the pixels is computed or shown.

  t3a_zoom.py CANDIDATES_JSON OUT_DIR

Per frame two images, same rendering as the t3a_inspect.py grid (gamma-encoded negative normalised by the frame's
99.9 % value, saturated blocks magenta, G >= 95 % of the frame's maximum red), enlarged x4:
  <stem>_tb.jpg  top band (y 0-12 %) above the bottom band (y 88-100 %), x 20-80 %
  <stem>_lr.jpg  left band (x 0-12 %) beside the right band (x 88-100 %), y 15-85 %
Coordinates are % of the whole frame: lines every 0.5 % across a band (labels every 1 %), every 5 % along it.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from filmcolor.frozen import load_cache
from filmcolor.paths import data_dir

SCALE, BAND = 4, 12


def render(T8, S8):
    v = (np.clip(T8 / np.percentile(T8, 99.9), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    v[T8[..., 1] >= 0.95 * T8[..., 1].max()] = (255, 0, 0)
    v[S8 > 0] = (255, 0, 255)
    return v


def panel(v, x0, x1, y0, y1):
    """Crop (% of the frame), enlarge, draw the grid in frame coordinates."""
    h, w = v.shape[:2]
    c0, c1, r0, r1 = int(w * x0 / 100), int(w * x1 / 100), int(h * y0 / 100), int(h * y1 / 100)
    img = Image.fromarray(v[r0:r1, c0:c1]).resize(((c1 - c0) * SCALE, (r1 - r0) * SCALE), Image.NEAREST)
    d = ImageDraw.Draw(img)
    px = lambda i: (w * i / 100 - c0) * SCALE
    py = lambda i: (h * i / 100 - r0) * SCALE
    across_x = (x1 - x0) <= BAND  # left / right band: fine lines along x
    for k in range(0, 201):
        i = k / 2
        for horiz, pos, lo, hi in ((False, px(i), x0, x1), (True, py(i), y0, y1)):
            if not lo <= i <= hi:
                continue
            fine = across_x != horiz
            if not fine and i % 5:
                continue
            col = (0, 200, 0) if i % 5 == 0 else (0, 110, 170) if i % 1 == 0 else (0, 60, 90)
            if horiz:
                d.line([(0, pos), (img.width, pos)], fill=col)
                if (fine and i % 1 == 0) or i % 5 == 0:
                    d.text((2, pos + 1), f"{i:g}", fill=(255, 255, 0))
            else:
                d.line([(pos, 0), (pos, img.height)], fill=col)
                if (fine and i % 1 == 0) or i % 5 == 0:
                    d.text((pos + 1, 2), f"{i:g}", fill=(255, 255, 0))
    return img


def main(candidates, out_dir, stem=None, *box):
    """With STEM X0 X1 Y0 Y1: one enlarged region <stem>_x<X0>-<X1>_y<Y0>-<Y1>.jpg instead (e.g. a gap between
    frames inside the scan); fine lines along the narrower side."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = data_dir("t3_inversion", "T8")
    if stem:
        v = render(load_cache(cache / f"{stem}_T8.npy").astype(np.float64), load_cache(cache / f"{stem}_S8.npy"))
        x0, x1, y0, y1 = map(float, box)
        panel(v, x0, x1, y0, y1).save(out / f"{stem}_x{x0:g}-{x1:g}_y{y0:g}-{y1:g}.jpg", quality=88)
        return
    for c in json.loads(Path(candidates).read_text())["frames"]:
        if "excluded" in c:
            continue
        v = render(load_cache(cache / f"{c['stem']}_T8.npy").astype(np.float64), load_cache(cache / f"{c['stem']}_S8.npy"))
        top, bot = panel(v, 20, 80, 0, BAND), panel(v, 20, 80, 100 - BAND, 100)
        tb = Image.new("RGB", (top.width, top.height + bot.height + 12), "white")
        tb.paste(top, (0, 0))
        tb.paste(bot, (0, top.height + 12))
        tb.save(out / f"{c['stem']}_tb.jpg", quality=88)
        left, right = panel(v, 0, BAND, 15, 85), panel(v, 100 - BAND, 100, 15, 85)
        lr = Image.new("RGB", (left.width + right.width + 12, left.height), "white")
        lr.paste(left, (0, 0))
        lr.paste(right, (left.width + 12, 0))
        lr.save(out / f"{c['stem']}_lr.jpg", quality=88)
        print(c["stem"], tb.size, lr.size, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
