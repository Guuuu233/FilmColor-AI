"""Compare analysis path (CFA-aware block means) with render path (demosaiced block means)."""
import json
import sys
import time

import numpy as np

from filmcolor.raw import linearize, saturation_mask, superpixel
from filmcolor.rawpy_io import demosaic_linear, load_raw

BLOCK = 48


def block_means(rgb, block):
    h, w = (rgb.shape[0] // block) * block, (rgb.shape[1] // block) * block
    return rgb[:h, :w].reshape(h // block, block, w // block, block, 3).mean(axis=(1, 3))


def compare(path, block=BLOCK, min_level=0.02):
    f = load_raw(path)
    lin, col, sat = linearize(f), f.colors, saturation_mask(f)
    sp, sp_sat = superpixel(lin, col, sat, f.pattern.shape[0], block)
    t0 = time.time()
    dm = demosaic_linear(path)
    t_dm = time.time() - t0
    if dm.shape[:2] != f.cfa.shape:
        raise ValueError(f"render shape {dm.shape[:2]} != cfa shape {f.cfa.shape}")
    bm = block_means(dm, block)
    # skip saturated blocks and near-black blocks (render path clips negatives to 0)
    ok = ~sp_sat & np.all(sp > min_level, -1)
    ratio = bm[ok] / sp[ok]
    return dict(
        file=str(path), block=block, n_blocks=int(ok.sum()), demosaic_seconds=round(t_dm, 1),
        median_ratio=[float(x) for x in np.median(ratio, 0)],
        p01_ratio=[float(x) for x in np.percentile(ratio, 1, 0)],
        p99_ratio=[float(x) for x in np.percentile(ratio, 99, 0)],
    )


if __name__ == "__main__":
    out = [compare(p) for p in sys.argv[1:]]
    print(json.dumps(out, indent=1))
