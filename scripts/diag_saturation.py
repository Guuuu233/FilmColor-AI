"""Locate saturated / near-saturated samples and summarise per-channel distributions (hypothesis check)."""
import sys
import numpy as np, tifffile
from filmcolor.raw import linearize, saturation_mask, superpixel
from filmcolor.rawpy_io import load_raw

for name in sys.argv[1:]:
    f = load_raw(f"data/raw/public/nlp_official/{name}")
    lin, col = linearize(f), f.colors
    sat = saturation_mask(f)
    near = saturation_mask(f, f.white - 0.03 * (f.white - f.black.max()))  # within 3 % of white
    print(f"== {name}")
    for c, k in enumerate("RGB"):
        v = lin[col == c]
        print(f"  {k}: p50={np.percentile(v,50):.4f} p99={np.percentile(v,99):.4f} p99.9={np.percentile(v,99.9):.4f} max={v.max():.4f} "
              f"sat={int((sat & (col==c)).sum())} near={int((near & (col==c)).sum())}")
    ys, xs = np.nonzero(near)
    if ys.size:
        H, W = lin.shape
        print(f"  near-sat bbox rows {ys.min()}-{ys.max()} cols {xs.min()}-{xs.max()} (image {H}x{W}); "
              f"frac within 10% of an image edge: {np.mean((ys<H*.1)|(ys>H*.9)|(xs<W*.1)|(xs>W*.9)):.2f}")
    sp, ss = superpixel(lin, col, near, 6, 24)
    g = np.log2(np.clip(sp[..., 1], 1e-4, None)); g = (g - g.min()) / (g.max() - g.min())
    vis = np.repeat(g[..., None], 3, -1)
    vis[ss] = (1, 0, 0)
    tifffile.imwrite(f"experiments/s1_diag/sat_{name}.tif", (vis * 255).astype(np.uint8))
