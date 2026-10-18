"""Read-only look at new material: one contact-sheet tile per file, plus an edge check on TIFF negatives.

Edge check: on a negative the clearest (brightest) pixels are the base; if the outer strip of the frame
is as bright as the interior 99.9th percentile in every channel, the frame likely shows unexposed rebate.

  audit_new_sheet.py ROOT OUT_JPG REL [REL ...]
"""
import sys
from pathlib import Path

import numpy as np
import rawpy
import tifffile
from io import BytesIO
from PIL import Image, ImageDraw

TILE = 360


def tile(path):
    if path.suffix.lower() in (".tif", ".tiff"):
        a = tifffile.imread(path)[::8, ::8, :3].astype(np.float64)
        h, w = a.shape[:2]
        s = max(2, min(h, w) // 50)
        edge = np.concatenate([a[:s].reshape(-1, 3), a[-s:].reshape(-1, 3),
                               a[:, :s].reshape(-1, 3), a[:, -s:].reshape(-1, 3)])
        inner = a[s:-s, s:-s].reshape(-1, 3)
        info = dict(edge_p50=np.percentile(edge, 50, axis=0).round(0).tolist(),
                    inner_p999=np.percentile(inner, 99.9, axis=0).round(0).tolist(),
                    max=a.reshape(-1, 3).max(0).tolist())
        img = Image.fromarray((np.clip(a / a.max(), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8))
    elif path.suffix.lower() in (".jpg", ".jpeg", ".png"):
        img = Image.open(path)
        img.draft("RGB", (TILE * 2, TILE * 2))
        info = {}
    else:
        with rawpy.imread(str(path)) as r:
            t = r.extract_thumb()
        img = Image.open(BytesIO(t.data)) if t.format == rawpy.ThumbFormat.JPEG else Image.fromarray(t.data)
        info = {}
    img.thumbnail((TILE, TILE))
    return img, info


def main(root, out, *rels):
    root = Path(root)
    tiles = []
    for rel in rels:
        img, info = tile(root / rel)
        print(rel, info)
        tiles.append((rel, img))
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * TILE, rows * (TILE + 20)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (rel, img) in enumerate(tiles):
        x, y = (i % cols) * TILE, (i // cols) * (TILE + 20)
        sheet.paste(img.convert("RGB"), (x, y + 20))
        d.text((x + 4, y + 4), rel.split("/")[-1] + " " + rel.split("/")[0][:16], fill="black")
    sheet.save(out, quality=88)


if __name__ == "__main__":
    main(*sys.argv[1:])
