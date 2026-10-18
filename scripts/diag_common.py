"""Shared pieces of the base-inversion diagnosis (experiments/s3_diag). Experimental code: the engine
(filmcolor.invert / render / fit) is imported, never changed.

positive_T   the engine inversion (invert.invert) written out with two experimental options: per-channel contrast
             factors ``chan_gamma`` (exponent R * g_c, each channel still 0 at its black point and 1 at its white
             point; g = (1, 1, 1) is the engine) and a 3x3 ``t_matrix`` applied to the base-normalised
             transmittance T / b before the density (rows sum to 1, so base stays at density 0)
grid_T       T on the reference-aligned grid of an S3 cache (shop: P4 grid, alignment factor 2; user: X8 grid,
             factor 1), 3 % margin, reproduced from the stored alignment
metrics      dE2000, lightness, hue, clipping, tonal zones of one rendering against one reference
fit_extra    the ub7 objective (fit.fit_reference mode "7", lam 0, highlight 30, area 10) over the seven render
             parameters plus extra parameters (channel contrast or a P-space matrix), from the ub7 optimum
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from filmcolor.colorimetry import delta_e2000  # noqa: E402
from filmcolor.compare import block_mean  # noqa: E402
from filmcolor.fit import (FD_STEP, FIT_CONFIG_V1, HUBER_DELTA, MAXITER, SCALES, clip_area, highlight_penalty,  # noqa: E402
                           reference_clipped, render_lab)
from filmcolor.invert import EPS  # noqa: E402
from filmcolor.render import BOUNDS, PER_IMAGE, RenderParams  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from sensitivity import MARGIN, margin_crop  # noqa: E402

LOOK = json.loads(Path("src/filmcolor/configs/global_look_v1.json").read_text())
ZONES = (("shadow", 0, 35), ("mid", 35, 70), ("high", 70, 101))  # reference L*
HUE_MIN_CHROMA = 10.0
XYZ_SRGB_D65 = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                         [0.0193339, 0.1191920, 0.9503041]])


def positive_T(T, base, d_lo, d_hi, gamma=0.6, chan_gamma=(1.0, 1.0, 1.0), t_matrix=None):
    t = np.asarray(T, np.float64) / np.asarray(base, np.float64)
    if t_matrix is not None:
        t = t @ np.asarray(t_matrix, np.float64).T
    D = -np.log10(np.maximum(t, EPS))
    lo, hi = np.asarray(d_lo, np.float64), np.asarray(d_hi, np.float64)
    s = (D - lo) / (hi - lo)
    R = (hi[1] - lo[1]) / gamma
    k = R * np.asarray(chan_gamma, np.float64)
    floor = 10.0 ** -k
    return ((10.0 ** (k * (s - 1)) - floor) / (1 - floor)).astype(np.float32)


def camera_to_srgb(rgb_xyz_matrix):
    """dcraw / LibRaw convention: rgb_xyz_matrix maps XYZ (D65) to camera RGB; returns camera RGB -> linear
    sRGB with rows summing to 1 (camera white -> sRGB white)."""
    cam_rgb = np.asarray(rgb_xyz_matrix, np.float64)[:3] @ XYZ_SRGB_D65
    cam_rgb = cam_rgb / cam_rgb.sum(1, keepdims=True)
    return np.linalg.inv(cam_rgb)


def margin_shape_candidates(n_out):
    return [n for n in range(n_out, int(n_out / (1 - 2 * MARGIN)) + 3) if n - 2 * int(n * MARGIN) == n_out]


def grid_T(T_in, al: dict, out_shape, check_P=None, check_fn=None):
    """``T_in`` at the alignment's input resolution; crop as compare.aligned_pair, then margin_crop to
    ``out_shape``. The pre-margin size is not stored; every size compatible with ``out_shape`` is tried and,
    when they give different crops, the one whose positive (``check_fn``) best matches ``check_P`` is taken."""
    a = block_mean(T_in, al["factor"])
    y, x = al["offset"]
    tried = {}
    for h in margin_shape_candidates(out_shape[0]):
        for w in margin_shape_candidates(out_shape[1]):
            g = margin_crop(a[y : y + h, x : x + w])
            if g.shape[:2] == tuple(out_shape[:2]):
                tried[(y + int(h * MARGIN), x + int(w * MARGIN))] = g
    if not tried:
        raise ValueError("no pre-margin size reproduces the cached grid")
    if len(tried) == 1 or check_P is None:
        (start, g), = list(tried.items())[:1]
        return g, dict(start=list(start), n_candidates=len(tried))
    scores = {k: float(np.corrcoef(np.log(np.maximum(check_fn(g), 1e-4)).ravel(),
                                   np.log(np.maximum(check_P, 1e-4)).ravel())[0, 1]) for k, g in tried.items()}
    best = max(scores, key=scores.get)
    return tried[best], dict(start=list(best), n_candidates=len(tried), corr={f"{k[0]},{k[1]}": v for k, v in scores.items()})


def lab_lch(lab):
    C = np.hypot(lab[..., 1], lab[..., 2])
    h = np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))
    return C, h


def metrics(lab, ref, pre_clip=None, L_density=None):
    """``lab``, ``ref``: (N, 3); ``pre_clip``: display-linear before the white clip (N, 3); ``L_density``:
    (N,) a per-pixel density used to bin residuals (G channel of the negative, fixed per frame)."""
    de = delta_e2000(lab, ref)
    d = lab - ref
    C, h = lab_lch(lab)
    Cr, hr = lab_lch(ref)
    hue_ok = (Cr > HUE_MIN_CHROMA) & (C > HUE_MIN_CHROMA / 2)
    dh = (h - hr + 180) % 360 - 180
    out = dict(de2000_median=float(np.median(de)), de2000_mean=float(de.mean()), de2000_p90=float(np.percentile(de, 90)),
               dL_median=float(np.median(d[:, 0])), dL_abs_median=float(np.median(np.abs(d[:, 0]))),
               da_mean=float(d[:, 1].mean()), db_mean=float(d[:, 2].mean()),
               hue_n=int(hue_ok.sum()),
               dh_mean_deg=float(np.degrees(np.angle(np.exp(1j * np.radians(dh[hue_ok])).mean()))) if hue_ok.any() else None,
               dh_abs_median_deg=float(np.median(np.abs(dh[hue_ok]))) if hue_ok.any() else None,
               chroma_ratio_median=float(np.median(C[hue_ok] / Cr[hue_ok])) if hue_ok.any() else None)
    if pre_clip is not None:
        rc = reference_clipped(ref)
        over = pre_clip > 1
        out.update(clipped_frac=float(over.any(-1).mean()), ref_clipped_frac=float(rc.any(-1).mean()),
                   clipped_where_ref_not_frac=float((over & ~rc).any(-1).mean()))
    zones = {}
    for name, lo, hi in ZONES:
        m = (ref[:, 0] >= lo) & (ref[:, 0] < hi)
        zones[name] = dict(share=float(m.mean()), **(dict(
            de2000_median=float(np.median(de[m])), dL_mean=float(d[m, 0].mean()),
            da_mean=float(d[m, 1].mean()), db_mean=float(d[m, 2].mean())) if m.sum() >= 50 else {}))
    out["zones"] = zones
    if L_density is not None:
        q = np.quantile(L_density, np.linspace(0, 1, 6))
        bins = []
        for i in range(5):
            m = (L_density >= q[i]) & (L_density <= q[i + 1])
            bins.append(dict(density_range=[float(q[i]), float(q[i + 1])], de2000_median=float(np.median(de[m])),
                             dL_mean=float(d[m, 0].mean()), da_mean=float(d[m, 1].mean()), db_mean=float(d[m, 2].mean())))
        out["density_bins"] = bins
    return out


def evaluate(P, ref, params, L_density=None):
    lab, diag, pre = render_lab(P[None], RenderParams(**params), with_pre_clip=True)
    m = metrics(lab[0], ref, pre[0], L_density)
    m["below_zero_after_colour"] = diag["below_zero_after_colour"]
    return m, lab[0]


EXTRA = {
    # per-channel contrast factors of the inversion (G fixed at 1): log g_R, log g_B
    "chan_gamma": dict(n=2, scale=0.1, bounds=(np.log(0.5), np.log(2.0))),
    # P-space matrix, rows sum to 1: the six off-diagonal entries (diagonal = 1 - row sum of off-diagonals)
    "p_matrix": dict(n=6, scale=0.1, bounds=(-0.8, 0.8)),
}


def matrix_of(v):
    m = np.zeros((3, 3))
    m[~np.eye(3, dtype=bool)] = v
    m[np.eye(3, dtype=bool)] = 1 - m.sum(1)
    return m


def fit_extra(kind, T, inv, ref, start_params, n_sample=20000, seed=0):
    """Minimise the ub7 objective over the seven render parameters and the ``kind`` extras. ``T`` (N, 3) and
    ``inv`` (positive_T keyword arguments without chan_gamma) define the positive; chan_gamma changes the
    inversion, p_matrix is applied to P before the render. Starts at ``start_params`` (the ub7 optimum without
    extras) with extras at their neutral values, so the result is never worse than ub7 on the objective.
    Diagnostic only: it sees the reference."""

    def P_of(e, rows=slice(None)):
        if e is None or kind == "p_matrix":
            return positive_T(T[rows], **inv)
        return positive_T(T[rows], **inv, chan_gamma=(float(np.exp(e[0])), 1.0, float(np.exp(e[1]))))

    c = FIT_CONFIG_V1
    ex = EXTRA[kind]
    rng = np.random.default_rng(seed)
    N = len(ref)
    idx = np.sort(rng.choice(N, min(n_sample, N), replace=False))
    Rs = ref[idx]
    allowed = reference_clipped(ref)[idx]
    d0 = np.array([start_params[k] for k in PER_IMAGE])
    sc = np.array([SCALES[k] for k in PER_IMAGE])
    blo, bhi = np.array([BOUNDS[k][0] for k in PER_IMAGE]), np.array([BOUNDS[k][1] for k in PER_IMAGE])
    lo = np.concatenate([(blo - d0) / sc, np.full(ex["n"], ex["bounds"][0] / ex["scale"])])
    hi = np.concatenate([(bhi - d0) / sc, np.full(ex["n"], ex["bounds"][1] / ex["scale"])])
    P_fixed = P_of(None, idx) if kind == "p_matrix" else None

    def split(u):
        theta = np.clip(d0 + sc * u[:7], blo, bhi)
        e = u[7:] * ex["scale"]
        return dict(zip(PER_IMAGE, map(float, theta))), e

    def P_sample(e):
        if kind == "chan_gamma":
            return P_of(e, idx)
        return P_fixed @ matrix_of(e).T.astype(np.float32)

    def objective(u):
        theta, e = split(u)
        if kind == "p_matrix" and np.linalg.cond(matrix_of(e)) > 10:
            return 1e6
        lab, _, pre = render_lab(P_sample(e)[None], RenderParams(**theta), with_pre_clip=True)
        de = delta_e2000(lab[0], Rs)
        return float((np.sqrt(de ** 2 + HUBER_DELTA ** 2) - HUBER_DELTA).mean()
                     + c["highlight_weight"] * highlight_penalty(pre[0], allowed) + c["area_weight"] * clip_area(pre[0], allowed))

    u0 = np.zeros(7 + ex["n"])
    r = minimize(objective, u0, method="L-BFGS-B", bounds=list(zip(lo, hi)), options=dict(eps=FD_STEP, maxiter=MAXITER))
    theta, e = split(r.x)
    P = P_of(e) if kind == "chan_gamma" else P_of(None) @ matrix_of(e).T
    extra = (dict(chan_gamma=[float(np.exp(e[0])), 1.0, float(np.exp(e[1]))]) if kind == "chan_gamma"
             else dict(p_matrix=matrix_of(e).tolist()))
    return dict(params=theta | dict(matrix=[[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]), objective=float(r.fun),
                objective_start=float(objective(u0)), success=bool(r.success), **extra), P.astype(np.float32)
