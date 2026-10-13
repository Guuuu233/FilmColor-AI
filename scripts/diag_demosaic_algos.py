import sys
import numpy as np
import rawpy
from filmcolor.raw import linearize, saturation_mask, superpixel
from filmcolor.rawpy_io import demosaic_linear, load_raw
B = 48
p = sys.argv[1]
f = load_raw(p); lin, col, sat = linearize(f), f.colors, saturation_mask(f)
sp, ss = superpixel(lin, col, sat, 6, B); hb, wb = sp.shape[:2]
ok = ~ss & np.all(sp > 0.02, -1)
for name, algo in [("default", None), ("LINEAR", rawpy.DemosaicAlgorithm.LINEAR), ("VNG", rawpy.DemosaicAlgorithm.VNG)]:
    dm = demosaic_linear(p, algo)
    bm = dm[:hb*B, :wb*B].reshape(hb, B, wb, B, 3).mean(axis=(1, 3))
    r = bm / sp
    print(name, "all", np.round(np.median(r[ok], 0), 4))
    bq = np.quantile(sp[ok][:, 2], [0, .25, .5, .75, 1])
    for i in range(4):
        s = ok & (sp[..., 2] >= bq[i]) & (sp[..., 2] <= bq[i+1])
        print("   B level", np.round(bq[i:i+2], 3), np.round(np.median(r[s], 0), 4))
