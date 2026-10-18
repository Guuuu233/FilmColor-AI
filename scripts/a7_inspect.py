"""A7R5 rebate frames, step 1: previews with a coordinate grid and saturation statistics (read-only).

Decodes each ARW the way the engine's camera path does (rawpy_io.demosaic_linear, raw.saturation_mask), stores a
1/8 linear block-mean cache for later region measurements (s3_a7/<stem>_T8.npy, provenance sidecar) and writes a
gamma-encoded preview with grid lines every 5 % (labels every 10 %) and saturated pixels in magenta, so that
sprocket holes (backlight, no film), rebate (processed, unexposed film) and image can be located by eye.

  a7_inspect.py OUT_DIR ARW [ARW ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from filmcolor.compare import block_mean
from filmcolor.frozen import check_source, save_cache
from filmcolor.paths import data_dir
from filmcolor.raw import saturation_mask
from filmcolor.rawpy_io import demosaic_linear, load_raw


def main(out_dir, *arws):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = data_dir("s3_a7")
    stats = {}
    for a in arws:
        sha = check_source(a)
        fr = load_raw(a)
        sat = saturation_mask(fr)
        del fr
        rgb = demosaic_linear(a)
        T8 = block_mean(rgb, 8).astype(np.float32)
        S8 = block_mean(sat.astype(np.float32), 8).astype(np.float32)
        stem = Path(a).stem
        save_cache(cache / f"{stem}_T8.npy", T8, [sha])
        save_cache(cache / f"{stem}_S8.npy", S8, [sha])
        stats[stem] = dict(sha256=sha, shape=list(sat.shape), sat_frac=float(sat.mean()),
                           max=rgb.reshape(-1, 3).max(0).tolist(), p999=np.percentile(T8.reshape(-1, 3), 99.9, 0).tolist())
        del rgb
        v = np.clip(T8 / np.percentile(T8, 99.9), 0, 1) ** (1 / 2.2)
        img = Image.fromarray((v * 255).astype(np.uint8))
        px = np.array(img)
        px[S8 > 0] = (255, 0, 255)
        img = Image.fromarray(px)
        d = ImageDraw.Draw(img)
        h, w = T8.shape[:2]
        for i in range(0, 101, 5):
            x, y = int(w * i / 100), int(h * i / 100)
            col = (0, 255, 0) if i % 10 == 0 else (0, 110, 0)
            d.line([(x, 0), (x, h)], fill=col)
            d.line([(0, y), (w, y)], fill=col)
            if i % 10 == 0:
                d.text((x + 2, 2), str(i), fill=(255, 255, 0))
                d.text((2, y + 2), str(i), fill=(255, 255, 0))
        img.save(out / f"{stem}_grid.jpg", quality=90)
        print(stem, json.dumps(stats[stem]), flush=True)
    (out / "inspect.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
