"""Direct dE2000 between the shop's print and the user's own print of the same frame (two-grader comparison):
both prints were aligned to the same full-frame 1/8 grid (cache records), so their overlap is cut out on that
grid. Frames where either print is rotated are skipped. Registration is checked by the L* correlation, and the
dE is also given on a 4x coarser grid (insensitive to a one-cell misregistration).

  s3_user_vs_shop_prints.py USER_LABEL_DIR OUT_JSON
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

from filmcolor.colorimetry import delta_e2000
from filmcolor.compare import block_mean
from filmcolor.frozen import load_cache
from filmcolor.paths import data_dir

MARGIN = 0.03  # as sensitivity.margin_crop


def precrop(n):
    """Size before the margin crop that leaves n cells."""
    return next(H for H in range(n, 2 * n) if H - 2 * int(H * MARGIN) == n)


def top_left(al, R):
    return al["offset"][0] + int(precrop(R.shape[0]) * MARGIN), al["offset"][1] + int(precrop(R.shape[1]) * MARGIN)


def main(label_dir, out_json):
    out = {}
    for p in sorted(Path(label_dir).glob("*.json")):
        u = json.loads(p.read_text())
        stem = u["id"].split(":")[1]
        sc = data_dir("s3_ml", "cache", "ext_own")
        s = json.loads((sc / f"{stem}.json").read_text())
        if s["alignment"]["rot90"] or u["alignment"]["rot90"]:
            continue
        Rs = load_cache(sc / f"{stem}_R8.npy").astype(np.float64)
        Ru = load_cache(data_dir("s3_ml", "cache_user", "ext_own") / f"{stem}_R8.npy").astype(np.float64)
        (ys, xs), (yu, xu) = top_left(s["alignment"], Rs), top_left(u["alignment"], Ru)
        y0, x0 = max(ys, yu), max(xs, xu)
        y1, x1 = min(ys + Rs.shape[0], yu + Ru.shape[0]), min(xs + Rs.shape[1], xu + Ru.shape[1])
        a, b = Rs[y0 - ys : y1 - ys, x0 - xs : x1 - xs], Ru[y0 - yu : y1 - yu, x0 - xu : x1 - xu]
        La, Lb = a[..., 0] - a[..., 0].mean(), b[..., 0] - b[..., 0].mean()
        d = a.reshape(-1, 3) - b.reshape(-1, 3)
        out[u["id"]] = dict(roll=u["roll"], ncc=float((La * Lb).sum() / np.sqrt((La ** 2).sum() * (Lb ** 2).sum())),
                            de_median=float(np.median(delta_e2000(a.reshape(-1, 3), b.reshape(-1, 3)))),
                            de_median_4x=float(np.median(delta_e2000(block_mean(a, 4).reshape(-1, 3), block_mean(b, 4).reshape(-1, 3)))),
                            shop_minus_user=dict(dL=float(d[:, 0].mean()), da=float(d[:, 1].mean()), db=float(d[:, 2].mean())))
    Path(out_json).write_text(json.dumps(out, indent=1))
    for roll in sorted({v["roll"] for v in out.values()}):
        g = [v for v in out.values() if v["roll"] == roll]
        print(roll, len(g), "ncc min %.3f" % min(v["ncc"] for v in g), "dE %.2f (4x %.2f)" % (st.median(v["de_median"] for v in g), st.median(v["de_median_4x"] for v in g)),
              "shop-user dL %.1f da %.1f db %.1f" % tuple(st.mean(v["shop_minus_user"][k] for v in g) for k in ("dL", "da", "db")))


if __name__ == "__main__":
    main(*sys.argv[1:3])
