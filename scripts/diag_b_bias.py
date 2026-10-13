"""Locate the analysis-vs-render channel bias: native CFA sites vs interpolated sites, flat vs textured blocks."""
import json
import sys

import numpy as np

from filmcolor.raw import linearize, saturation_mask, superpixel
from filmcolor.rawpy_io import demosaic_linear, load_raw

BLOCK = 48


def diag(path):
    f = load_raw(path)
    lin, col, sat = linearize(f), f.colors, saturation_mask(f)
    dm = demosaic_linear(path)
    out = dict(file=str(path))
    # 1. native sites: rendered value of channel c where the CFA sampled c
    for c, name in enumerate("RGB"):
        m = (col == c) & ~sat & (lin > 0.02)
        r = dm[..., c][m] / lin[m]
        out[f"native_{name}"] = dict(median=float(np.median(r)), p01=float(np.percentile(r, 1)),
                                     p99=float(np.percentile(r, 99)), exact_frac=float(np.mean(np.abs(r - 1) < 1e-3)))
    # 2. block ratio split by block texture (rel std of the block in the render)
    n = f.pattern.shape[0]
    sp, sp_sat = superpixel(lin, col, sat, n, BLOCK)
    hb, wb = sp.shape[:2]
    blk = dm[: hb * BLOCK, : wb * BLOCK].reshape(hb, BLOCK, wb, BLOCK, 3)
    bm = blk.mean(axis=(1, 3))
    cv = (blk.std(axis=(1, 3)) / np.maximum(bm, 1e-6)).max(-1)
    ok = ~sp_sat & np.all(sp > 0.02, -1)
    ratio = bm / sp
    qs = np.quantile(cv[ok], [0, 0.25, 0.5, 0.75, 1])
    out["by_texture_quartile"] = []
    for i in range(4):
        sel = ok & (cv >= qs[i]) & (cv <= qs[i + 1])
        out["by_texture_quartile"].append(dict(cv_range=[float(qs[i]), float(qs[i + 1])], n=int(sel.sum()),
                                               median_ratio=[float(x) for x in np.median(ratio[sel], 0)]))
    return out


if __name__ == "__main__":
    print(json.dumps([diag(p) for p in sys.argv[1:]], indent=1))
