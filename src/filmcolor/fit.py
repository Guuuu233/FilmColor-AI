"""Reference-image parameter fitting (Sprint 2 research tool).

Finds render parameters that bring our output closest to a reference rendering of the same negative
(e.g. NLP). It sees the reference, so it measures what the parameter space can express; it is NOT an
automatic-correction baseline. Differences are "relative to the reference output" (docs/SPRINT2_PARAMS.md).

Objective: mean pseudo-Huber of CIEDE2000, sqrt(dE^2 + delta^2) - delta, between our rendering (read
with our ICC profile) and the reference Lab, plus an optional penalty lam * sum(((theta - default) /
scale)^2). A plain mean dE has a kink wherever pixel errors reach zero, which stalls gradient-based
optimisation; pseudo-Huber is smooth there and still linear (robust) for large errors. Optimised with
L-BFGS-B in the scaled variables u = (theta - default) / scale, from several fixed starts.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .colorimetry import MatrixTRCProfile, delta_e2000, xyz_to_lab
from .export import icc_profile
from .render import BOUNDS, PER_IMAGE, RenderParams, render

SCALES = dict(ev=1.0, temp=0.25, tint=0.25, sat=0.25, contrast=0.25, midtone=0.05, shoulder=0.25)
DEFAULTS = {k: getattr(RenderParams(), k) for k in PER_IMAGE}
MODES = {
    "6": dict(free=tuple(k for k in PER_IMAGE if k != "ev"), lam=0.0),
    "7": dict(free=PER_IMAGE, lam=0.0),
    "7reg": dict(free=PER_IMAGE, lam=0.05),
}
FD_STEP = 2e-3     # finite-difference step in scaled units
HUBER_DELTA = 1.0  # dE units
NEAR_OPTIMAL = 0.1  # starts within this many dE of the best count as "also optimal"
_OUR_PROFILE = MatrixTRCProfile(icc_profile("srgb"))


def render_lab(P, params: RenderParams, mask=None):
    """Lab (D50) of our sRGB-encoded rendering as a colour-managed viewer would read it."""
    e, diag = render(P, params, mask)
    return xyz_to_lab(_OUR_PROFILE.to_xyz(e.astype(np.float64))), diag


def fit_reference(P, ref_lab, mode: str = "7", *, n_starts: int = 5, seed: int = 0,
                  n_sample: int = 20000, lam: float | None = None) -> dict:
    """``P``: scene-linear positive pixels (..., 3); ``ref_lab``: reference Lab on the same pixels."""
    cfg = MODES[mode]
    free, lam = list(cfg["free"]), cfg["lam"] if lam is None else lam
    P2, R2 = np.asarray(P, np.float32).reshape(-1, 3), np.asarray(ref_lab, np.float64).reshape(-1, 3)
    if len(P2) != len(R2):
        raise ValueError("P and ref_lab must cover the same pixels")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(P2), min(n_sample, len(P2)), replace=False))
    Ps, Rs = P2[idx][None], R2[idx]
    d0 = np.array([DEFAULTS[k] for k in free])
    sc = np.array([SCALES[k] for k in free])
    blo, bhi = np.array([BOUNDS[k][0] for k in free]), np.array([BOUNDS[k][1] for k in free])
    lo, hi = (blo - d0) / sc, (bhi - d0) / sc

    def params_of(u):
        theta = np.clip(d0 + sc * u, blo, bhi)  # clip after the transform: rounding can step past a bound
        return RenderParams(**{k: float(v) for k, v in zip(free, theta)})

    def objective(u):
        lab, _ = render_lab(Ps, params_of(u))
        de = delta_e2000(lab[0], Rs)
        return float((np.sqrt(de ** 2 + HUBER_DELTA ** 2) - HUBER_DELTA).mean() + lam * np.sum(u ** 2))

    starts = [np.zeros(len(free))] + [rng.uniform(lo, hi) for _ in range(n_starts - 1)]
    runs = []
    for u0 in starts:
        r = minimize(objective, u0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                     options=dict(eps=FD_STEP, maxiter=300))
        runs.append(dict(theta=params_of(r.x).to_dict(), u=r.x,
                         objective=float(r.fun), success=bool(r.success), nfev=int(r.nfev)))
    best = min(runs, key=lambda r: r["objective"])
    near = [r for r in runs if r["objective"] <= best["objective"] + NEAR_OPTIMAL]
    U = np.array([r["u"] for r in near])
    p = params_of(best["u"])
    lab, diag = render_lab(P2[None], p)
    de = delta_e2000(lab[0], R2)
    rng_u = hi - lo
    at_bound = [k for k, u, l, h, w in zip(free, best["u"], lo, hi, rng_u) if min(u - l, h - u) <= 0.01 * w]
    return dict(
        mode=mode, free=free, lam=lam, params=p.to_dict(),
        de2000_mean=float(de.mean()), de2000_median=float(np.median(de)), de2000_p90=float(np.percentile(de, 90)),
        objective=best["objective"], diagnostics=diag, at_bound=at_bound,
        n_near_optimal=len(near),
        spread_near_optimal={k: float(U[:, i].max() - U[:, i].min()) for i, k in enumerate(free)},  # scaled units
        runs=[{k: v for k, v in r.items() if k != "u"} for r in runs],
    )


def param_conditioning(P, params: RenderParams, names=PER_IMAGE, n_sample: int = 20000, seed: int = 0) -> dict:
    """Column-normalised Jacobian of the rendered Lab w.r.t. scaled parameters at ``params``:
    min/max singular value ratio and pairwise |cosine| between parameter effects."""
    P2 = np.asarray(P, np.float32).reshape(-1, 3)
    idx = np.sort(np.random.default_rng(seed).choice(len(P2), min(n_sample, len(P2)), replace=False))
    Ps = P2[idx][None]
    base, _ = render_lab(Ps, params)
    cols = []
    for k in names:
        h = 0.05 * SCALES[k]
        v = getattr(params, k)
        v2 = v + h if v + h <= BOUNDS[k][1] else v - h
        out, _ = render_lab(Ps, RenderParams(**(params.to_dict() | {k: v2})))
        cols.append(((out - base) / (v2 - v) * SCALES[k]).ravel())
    J = np.stack(cols, 1)
    norms = np.linalg.norm(J, axis=0)
    Jn = J / np.where(norms > 0, norms, 1)
    s = np.linalg.svd(Jn, compute_uv=False)
    C = np.abs(Jn.T @ Jn)
    return dict(sv_ratio=float(s.min() / s.max()),
                cosine={f"{names[i]}/{names[j]}": float(C[i, j]) for i in range(len(names)) for j in range(i + 1, len(names))})
