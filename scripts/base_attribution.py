"""S3-5: how much of the BlueNeg cross-device error comes from the film base estimate.

Per frame, at 1/8 resolution (the grid eval_blueneg.py registers on), re-render the frozen `auto`
output from the negative with a modified film base and compare with the print (same registration
and metrics as eval_blueneg.py). A base change b' = b * 10**-delta shifts every density by +delta,
so the white-point percentile shifts by exactly +delta and d_hi is taken from the full-resolution
sidecar plus that shift (no low-resolution re-estimate).

  baseline        sidecar base and render parameters (consistency check against eval3.json)
  sens            per-channel base density +-0.02, +-0.05 D
  oracle_base     3 log-base offsets (+-0.3 D) fitted to the print       } use the print: attribution
  oracle_base_chroma  zero-mean offsets only (base colour, black point unchanged)  }
  oracle_wb       temp, tint fitted to the print, base unchanged         } only, never a configuration
  oracle_tone     ev, midtone fitted to the print, base unchanged        }
  roll_swap       base chromaticity of the other sampled frame of the same roll (no reference used)

usage: base_attribution.py selection.json DATA_ROOT CONV_DIR eval3.json out.json [frame ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from filmcolor.colorimetry import D50, delta_e2000, xyz_to_lab
from filmcolor.compare import block_mean, metrics, resize_bilinear
from filmcolor.invert import InversionParams, density, invert
from filmcolor.render import BOUNDS, RenderParams, render_full
from filmcolor.scanner import load_scanner

F = 8
SENS = (-0.05, -0.02, 0.02, 0.05)
SRGB_D50 = np.array([[0.4360747, 0.3850649, 0.1430804],
                     [0.2225045, 0.7168786, 0.0606169],
                     [0.0139322, 0.0971045, 0.7141733]])


def enc_to_lab(e):
    lin = np.where(e <= 0.04045, e / 12.92, ((e + 0.055) / 1.055) ** 2.4)
    return xyz_to_lab(lin @ SRGB_D50.T, white=D50)


def print_lab(path):
    import tifffile
    a = tifffile.imread(path).astype(np.float64)[..., :3] / 65535
    return enc_to_lab(a)


def orient(img, k, flip):
    return np.rot90(img[:, ::-1] if flip else img, k)


class Frame:
    def __init__(self, fr, root, conv, reg):
        sc = json.load(open(conv / "auto" / f"{fr['filename']}.json"))
        self.base = np.array(sc["base"]["rgb_render"], np.float64)
        p = sc["inversion"]["params"]
        self.d_hi, self.gamma, self.exposure = np.array(p["d_hi"]), p["gamma"], p["exposure"]
        self.rp = sc["render"]["params"]
        neg = load_scanner(root / fr["negative_path"])
        self.neg = block_mean(neg.rgb, F).astype(np.float32)
        del neg
        ref = print_lab(root / fr["printed_path"])
        r = block_mean(orient(ref, reg["rot90"], reg["mirror"]), reg["factor"])
        shape = (int(round(r.shape[0] * reg["scale"])), int(round(r.shape[1] * reg["scale"])))
        self.ref = resize_bilinear(r, shape)
        self.win = (reg["offset"][0], reg["offset"][1], shape[0], shape[1])

    def lab(self, base=None, rp=None):
        b = self.base if base is None else np.asarray(base, np.float64)
        shift = -np.log10(b / self.base)  # density shift of the whole image
        p = InversionParams(base=tuple(b), d_lo=(0.0, 0.0, 0.0), d_hi=tuple(self.d_hi + shift),
                            gamma=self.gamma, exposure=self.exposure)
        P = invert(density(self.neg, b), p) * np.float32(self.exposure)
        e, _, _ = render_full(P, RenderParams.from_dict(rp or self.rp))
        y, x, h, w = self.win
        return enc_to_lab(e[y : y + h, x : x + w].astype(np.float64))

    def score(self, **kw):
        m = metrics(self.lab(**kw), self.ref)
        return {k: m[k] for k in ("de2000_median", "mean_dL", "mean_da", "mean_db")}

    def fast_cost(self, **kw):  # mean dE2000 on a 2x coarser grid, for the optimiser
        a, r = block_mean(self.lab(**kw), 2), block_mean(self.ref, 2)
        return float(delta_e2000(a.reshape(-1, 3), r.reshape(-1, 3)).mean())


def fit(fun, x0, bounds, step):
    """Nelder-Mead with an explicit initial simplex (scipy's default step is 0.00025 at x0 = 0)."""
    x0 = np.asarray(x0, float)
    simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[i] * step[i] for i in range(len(x0))])
    lo, hi = np.array(bounds).T
    simplex = np.clip(simplex, lo, hi)
    res = minimize(fun, x0, method="Nelder-Mead", bounds=bounds,
                   options=dict(xatol=1e-3, fatol=1e-3, maxiter=400, initial_simplex=simplex))
    return res.x


def run(fr, root, conv, reg):
    f = Frame(fr, root, conv, reg)
    out = dict(frame=fr["filename"], roll=fr["roll_id"], base=f.base.tolist(), baseline=f.score())
    out["sens"] = {f"{'RGB'[c]}{d:+.2f}": f.score(base=f.base * 10 ** -(d * (np.arange(3) == c))) for c in range(3) for d in SENS}
    x = fit(lambda v: f.fast_cost(base=f.base * 10 ** -v), np.zeros(3), [(-0.3, 0.3)] * 3, [0.05] * 3)
    out["oracle_base"] = dict(delta_D=x.tolist(), **f.score(base=f.base * 10 ** -x))
    # chromatic part only: per-channel offsets with zero mean (a common offset moves the black point,
    # which acts like a tone parameter rather than a base colour error)
    chroma = lambda v: np.array([v[0], -(v[0] + v[1]), v[1]])  # noqa: E731
    x = fit(lambda v: f.fast_cost(base=f.base * 10 ** -chroma(v)), np.zeros(2), [(-0.2, 0.2)] * 2, [0.03, 0.03])
    out["oracle_base_chroma"] = dict(delta_D=chroma(x).tolist(), **f.score(base=f.base * 10 ** -chroma(x)))
    wb = lambda v: dict(f.rp, temp=float(v[0]), tint=float(v[1]))  # noqa: E731
    x = fit(lambda v: f.fast_cost(rp=wb(v)), np.array([f.rp["temp"], f.rp["tint"]]), [BOUNDS["temp"], BOUNDS["tint"]], [0.1, 0.1])
    out["oracle_wb"] = dict(temp=float(x[0]), tint=float(x[1]), **f.score(rp=wb(x)))
    tone = lambda v: dict(f.rp, ev=float(v[0]), midtone=float(v[1]))  # noqa: E731
    x = fit(lambda v: f.fast_cost(rp=tone(v)), np.array([f.rp["ev"], f.rp["midtone"]]), [BOUNDS["ev"], BOUNDS["midtone"]], [0.3, 0.03])
    out["oracle_tone"] = dict(ev=float(x[0]), midtone=float(x[1]), **f.score(rp=tone(x)))
    out["_frame"] = f
    return out


def main():
    sel, root, conv = json.load(open(sys.argv[1])), Path(sys.argv[2]), Path(sys.argv[3])
    regs = {r["frame"]: r["registration"] for r in json.load(open(sys.argv[4]))["frames"]}
    out_path, only = sys.argv[5], set(sys.argv[6:])
    frames = [fr for fr in sel["frames"] if not only or fr["filename"] in only]
    res = []
    for fr in frames:
        r = run(fr, root, conv, regs[fr["filename"]])
        print(r["frame"], "baseline %.2f" % r["baseline"]["de2000_median"],
              "| base %.2f chroma %.2f wb %.2f tone %.2f" % tuple(r[k]["de2000_median"] for k in ("oracle_base", "oracle_base_chroma", "oracle_wb", "oracle_tone")),
              "| dD", np.round(r["oracle_base"]["delta_D"], 3), flush=True)
        res.append(r)
    # roll swap: the other sampled frame's base chromaticity, keeping this frame's green level
    by_roll = {}
    for r in res:
        by_roll.setdefault(r["roll"], []).append(r)
    for rs in by_roll.values():
        if len(rs) != 2:
            continue
        for a, b in (rs, rs[::-1]):
            other = np.array(b["base"])
            swapped = other / other[1] * a["base"][1]
            a["roll_swap"] = dict(source=b["frame"], base=swapped.tolist(), **a["_frame"].score(base=swapped))
    for r in res:
        r.pop("_frame")
    json.dump(dict(method=__doc__.split("usage:")[0].strip(), frames=res), open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main()
