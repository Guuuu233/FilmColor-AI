"""Is the group-pack TIFF still linear in the BlueNeg scanner DNG it was converted from? (read-only)

For each (TIFF, DNG) pair of the same frame: per-channel quantiles (0.5..99.5 %) of both, fitted as
log T = a + s log D (s = 1: pure scale, linear; s != 1: a tone curve); then, if the frames align after
resizing to a common 1/16 grid, a per-pixel fit T = M D (3x3, no offset) with its R^2 and the share of
off-diagonal weight (channel mixing).

  audit_ds_linearity.py TIFF DNG [TIFF DNG ...]
"""
import json
import sys

import numpy as np
import tifffile

from filmcolor.scanner import load_scanner

Q = np.linspace(0.5, 99.5, 60)


def grid(a, h, w):
    """Area-average an (H, W, 3) image onto an (h, w) grid (edges cropped to a multiple)."""
    H, W = a.shape[:2]
    by, bx = H // h, W // w
    return a[:by * h, :bx * w].reshape(h, by, w, bx, 3).mean((1, 3))


def main(*paths):
    out = []
    for tif, dng in zip(paths[::2], paths[1::2]):
        T = tifffile.imread(tif)[..., :3].astype(np.float64)
        f = load_scanner(dng)
        D = np.asarray(f.rgb, dtype=np.float64)
        r = dict(tif=tif, dng=dng, tif_shape=T.shape[:2], dng_shape=D.shape[:2],
                 tif_max=T.reshape(-1, 3).max(0).tolist(), dng_max=D.reshape(-1, 3).max(0).tolist())
        fits = []
        for c in range(3):
            qt, qd = np.percentile(T[..., c], Q), np.percentile(D[..., c], Q)
            ok = (qt > 0) & (qd > 0)
            s, a = np.polyfit(np.log(qd[ok]), np.log(qt[ok]), 1)
            res = np.log(qt[ok]) - (a + s * np.log(qd[ok]))
            fits.append(dict(slope=round(s, 4), scale=round(float(np.exp(a)), 5), rms=round(float(res.std()), 4)))
        r["quantile_fit"] = fits
        ar_t, ar_d = T.shape[1] / T.shape[0], D.shape[1] / D.shape[0]
        r["aspect"] = [round(ar_t, 4), round(ar_d, 4)]
        if abs(ar_t / ar_d - 1) < 0.01:
            # common 1/16 grid: removes the resolution difference (averaging narrows quantiles) and lets pixels pair up
            h, w = T.shape[0] // 16, T.shape[1] // 16
            tg, dg0 = grid(T, h, w), grid(D, h, w)
            best = None
            for name, dg in (("none", dg0), ("lr", dg0[:, ::-1]), ("ud", dg0[::-1]), ("rot180", dg0[::-1, ::-1])):
                t, d = tg.reshape(-1, 3), dg.reshape(-1, 3)
                M, *_ = np.linalg.lstsq(d, t, rcond=None)
                r2 = 1 - ((t - d @ M) ** 2).sum(0) / ((t - t.mean(0)) ** 2).sum(0)
                if best is None or r2.mean() > best[1].mean():
                    best = (name, r2, M, t, d)
            name, r2, M, t, d = best
            Mn = np.abs(M) / np.abs(M).sum(0)
            k = (t * d).sum(0) / (d * d).sum(0)  # per-channel pure scale, no mixing
            r2_diag = 1 - ((t - d * k) ** 2).sum(0) / ((t - t.mean(0)) ** 2).sum(0)
            gfit = []
            for c in range(3):
                qt, qd = np.percentile(t[:, c], Q), np.percentile(d[:, c], Q)
                s, a = np.polyfit(np.log(qd), np.log(qt), 1)
                gfit.append(round(s, 4))
            r["pixel_fit"] = dict(flip=name, M=M.round(1).tolist(), r2=r2.round(5).tolist(),
                                  offdiag_share=round(float(1 - np.trace(Mn) / 3), 4),
                                  diag_scale=k.round(1).tolist(), r2_diag=r2_diag.round(5).tolist(),
                                  grid_quantile_slope=gfit)
        out.append(r)
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
