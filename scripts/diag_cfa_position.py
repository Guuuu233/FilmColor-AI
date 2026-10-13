import sys, json
import numpy as np
from filmcolor.raw import linearize, saturation_mask
from filmcolor.rawpy_io import load_raw
from filmcolor.base import estimate_base_auto
for p in sys.argv[1:]:
    f = load_raw(p); lin, col, sat = linearize(f), f.colors, saturation_mask(f)
    est = estimate_base_auto(lin, col, sat, 6, crop=f.crop)
    acc = np.zeros((6, 6)); cnt = np.zeros((6, 6))
    for t, l, h, w in est.regions:  # flat base blocks, aligned to the 6x6 pattern (block=24)
        a = lin[t:t+h, l:l+w].reshape(h//6, 6, w//6, 6)
        acc += a.mean(axis=(0, 2)); cnt += 1
    pos = acc / cnt
    line = [p.split("/")[-1]]
    for c, n in enumerate("RGB"):
        v = pos[f.pattern == c]
        line.append(f"{n}: n_pos={v.size} spread={(v.max()-v.min())/v.mean():.2%}")
    print("  ".join(line))
