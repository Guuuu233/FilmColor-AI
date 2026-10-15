"""Reference-image parameter fitting (Sprint 2 research tool).

Finds render parameters that bring our output closest to a reference rendering of the same negative
(e.g. NLP). It sees the reference, so it measures what the parameter space can express; it is NOT an
automatic-correction baseline. Differences are "relative to the reference output" (docs/SPRINT2_PARAMS.md).

Objective: mean pseudo-Huber of CIEDE2000, sqrt(dE^2 + delta^2) - delta, between our rendering (read
with our ICC profile) and the reference Lab, plus an optional penalty lam * sum(((theta - default) /
scale)^2). A plain mean dE has a kink wherever pixel errors reach zero, which stalls gradient-based
optimisation; pseudo-Huber is smooth there and still linear (robust) for large errors. Optimised with
L-BFGS-B in the scaled variables u = (theta - default) / scale, from several fixed starts.

Highlight constraint (S2-6): CIEDE2000 barely penalises pushing a near-white area to pure white (white vs
light grey is ~2-3 dE), so the plain objective traded sky texture for midtone match (_DSF9237: 18.9 % of
the image clipped vs 0.5 % in NLP). The objective adds highlight_weight * mean over pixels of
sum_c h(stops above white before the clip), h(s) = sqrt(s^2 + d^2) - d (smooth, ~linear), on channels the
reference itself does not clip: the reference Lab is expressed in our output encoding (exact inverse of our
matrix/TRC profile) and a channel at or above REF_CLIP_ENCODED counts as clipped in the reference (a white
sky, or one channel of a saturated colour), where our clipping is not penalised. highlight_weight is dE per stop per pixel; the
default 10 equates pushing a pixel 0.1 stop past white with a 1 dE colour error (a convention, see
experiments/s2_sensitivity_hl). highlight_weight = 0 reproduces the S2-2 / S2-3 fitter.

Clipped-area constraint (S2-6c): the depth penalty misses a wide area pushed only slightly past white
(DSC_3825 sky on the shop reference: 7.9 % clipped at weight 30). area_weight * clip_area adds the soft share
of pixels past white (0 = off, the default). Labels for later work use FIT_CONFIG_V1 via fit_reference_v1.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .colorimetry import MatrixTRCProfile, delta_e2000, lab_to_xyz, xyz_to_lab
from .export import icc_profile
from .render import BOUNDS, PER_IMAGE, RenderParams, overexposure_stops, render_full, srgb_eotf

SCALES = dict(ev=1.0, temp=0.25, tint=0.25, sat=0.25, contrast=0.25, midtone=0.05, shoulder=0.25)
DEFAULTS = {k: getattr(RenderParams(), k) for k in PER_IMAGE}
MODES = {
    "6": dict(free=tuple(k for k in PER_IMAGE if k != "ev"), lam=0.0),
    "7": dict(free=PER_IMAGE, lam=0.0),
    "7reg": dict(free=PER_IMAGE, lam=0.05),
}
FD_STEP = 2e-3     # finite-difference step in scaled units
MAXITER = 300      # L-BFGS-B iterations per start
HUBER_DELTA = 1.0  # dE units
NEAR_OPTIMAL = 0.1  # starts within this many dE of the best count as "also optimal"
HIGHLIGHT_WEIGHT = 10.0  # dE per stop past white per pixel
HIGHLIGHT_DELTA = 0.05   # stops; smoothing of the highlight hinge
CLIP_AREA_RAMP = 0.05    # stops past white over which a channel's clipped-area count rises from 0 to 1
REF_CLIP_ENCODED = 0.99  # reference channels at or above this (our sRGB encoding) are clipped in the reference
_OUR_PROFILE = MatrixTRCProfile(icc_profile("srgb"))


def render_lab(P, params: RenderParams, mask=None, *, with_pre_clip: bool = False):
    """Lab (D50) of our sRGB-encoded rendering as a colour-managed viewer would read it (and optionally the
    display-linear values before the white clip)."""
    e, diag, pre = render_full(P, params, mask)
    lab = xyz_to_lab(_OUR_PROFILE.to_xyz(e.astype(np.float64)))
    return (lab, diag, pre) if with_pre_clip else (lab, diag)


def reference_clipped(ref_lab, threshold: float = REF_CLIP_ENCODED) -> np.ndarray:
    """(N, 3) bool: reference channels at or above ``threshold`` when expressed in our output encoding."""
    xyz = lab_to_xyz(np.asarray(ref_lab, np.float64).reshape(-1, 3))
    lin = np.linalg.solve(_OUR_PROFILE.matrix, xyz.T).T
    return lin >= float(srgb_eotf(np.float64(threshold)))


def highlight_penalty(pre_clip, allowed=None, delta: float = HIGHLIGHT_DELTA) -> float:
    """Mean over pixels of sum_c h(stops above white), h(s) = sqrt(s^2 + delta^2) - delta; channels where
    ``allowed`` (N, 3) is True (clipped in the reference) are not penalised."""
    s = overexposure_stops(np.asarray(pre_clip, np.float64)).reshape(-1, 3)
    h = np.sqrt(s ** 2 + delta ** 2) - delta
    if allowed is not None:
        h = np.where(np.asarray(allowed).reshape(-1, 3), 0.0, h)
    return float(h.sum(-1).mean())


def clip_area(pre_clip, allowed=None, ramp: float = CLIP_AREA_RAMP) -> float:
    """Soft share of pixels pushed past white in any channel not clipped in the reference (``allowed``).
    A channel counts 0 at or below white and rises smoothly (smoothstep in stops) to 1 at ``ramp`` stops
    past white; a pixel counts 1 - prod_c(1 - r_c). Complements ``highlight_penalty``, which grows with
    depth: a wide area pushed only slightly past white costs it almost nothing (S2-6b, DSC_3825 sky)."""
    t = np.clip(overexposure_stops(np.asarray(pre_clip, np.float64)).reshape(-1, 3) / ramp, 0.0, 1.0)
    r = t * t * (3 - 2 * t)
    if allowed is not None:
        r = np.where(np.asarray(allowed).reshape(-1, 3), 0.0, r)
    return float((1 - np.prod(1 - r, axis=-1)).mean())


def fit_reference(P, ref_lab, mode: str = "7", *, n_starts: int = 5, seed: int = 0,
                  n_sample: int = 20000, lam: float | None = None, scales: dict | None = None,
                  highlight_weight: float = HIGHLIGHT_WEIGHT, area_weight: float = 0.0) -> dict:
    """``P``: scene-linear positive pixels (..., 3); ``ref_lab``: reference Lab on the same pixels.
    ``lam`` / ``scales`` override the mode's penalty strength and the parameter scales;
    ``highlight_weight`` sets the highlight depth constraint and ``area_weight`` (dE per unit image share)
    the clipped-area constraint (0 = off)."""
    for name, w in (("highlight_weight", highlight_weight), ("area_weight", area_weight)):
        if not (np.isfinite(w) and w >= 0):
            raise ValueError(f"{name} must be >= 0, got {w}")
    scales = SCALES | (scales or {})
    cfg = MODES[mode]
    free, lam = list(cfg["free"]), cfg["lam"] if lam is None else lam
    P2, R2 = np.asarray(P, np.float32).reshape(-1, 3), np.asarray(ref_lab, np.float64).reshape(-1, 3)
    if len(P2) != len(R2):
        raise ValueError("P and ref_lab must cover the same pixels")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(P2), min(n_sample, len(P2)), replace=False))
    Ps, Rs = P2[idx][None], R2[idx]
    ref_clip = reference_clipped(R2)
    allowed_s = ref_clip[idx]
    d0 = np.array([DEFAULTS[k] for k in free])
    sc = np.array([scales[k] for k in free])
    blo, bhi = np.array([BOUNDS[k][0] for k in free]), np.array([BOUNDS[k][1] for k in free])
    lo, hi = (blo - d0) / sc, (bhi - d0) / sc

    def params_of(u):
        theta = np.clip(d0 + sc * u, blo, bhi)  # clip after the transform: rounding can step past a bound
        return RenderParams(**{k: float(v) for k, v in zip(free, theta)})

    def objective(u):
        lab, _, pre = render_lab(Ps, params_of(u), with_pre_clip=True)
        de = delta_e2000(lab[0], Rs)
        hl = highlight_weight * highlight_penalty(pre[0], allowed_s) if highlight_weight else 0.0
        hl += area_weight * clip_area(pre[0], allowed_s) if area_weight else 0.0
        return float((np.sqrt(de ** 2 + HUBER_DELTA ** 2) - HUBER_DELTA).mean() + hl + lam * np.sum(u ** 2))

    starts = [np.zeros(len(free))] + [rng.uniform(lo, hi) for _ in range(n_starts - 1)]
    runs = []
    for u0 in starts:
        r = minimize(objective, u0, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                     options=dict(eps=FD_STEP, maxiter=MAXITER))
        runs.append(dict(theta=params_of(r.x).to_dict(), u=r.x,
                         objective=float(r.fun), success=bool(r.success), nfev=int(r.nfev)))
    best = min(runs, key=lambda r: r["objective"])
    near = [r for r in runs if r["objective"] <= best["objective"] + NEAR_OPTIMAL]
    U = np.array([r["u"] for r in near])
    p = params_of(best["u"])
    lab, diag, pre = render_lab(P2[None], p, with_pre_clip=True)
    de = delta_e2000(lab[0], R2)
    over = pre[0] > 1
    rng_u = hi - lo
    at_bound = [k for k, u, l, h, w in zip(free, best["u"], lo, hi, rng_u) if min(u - l, h - u) <= 0.01 * w]
    return dict(
        mode=mode, free=free, lam=lam, scales={k: scales[k] for k in free}, params=p.to_dict(),
        de2000_mean=float(de.mean()), de2000_median=float(np.median(de)), de2000_p90=float(np.percentile(de, 90)),
        objective=best["objective"], diagnostics=diag, at_bound=at_bound,
        highlight=dict(weight=highlight_weight, ref_clip_encoded=REF_CLIP_ENCODED, delta_stops=HIGHLIGHT_DELTA,
                       clipped_frac=float(over.any(-1).mean()), ref_clipped_frac=float(ref_clip.any(-1).mean()),
                       clipped_where_ref_not_clipped_frac=float((over & ~ref_clip).any(-1).mean()),
                       penalty=highlight_penalty(pre[0], ref_clip),
                       area_weight=area_weight, area_ramp_stops=CLIP_AREA_RAMP, area=clip_area(pre[0], ref_clip)),
        n_near_optimal=len(near),
        spread_near_optimal={k: float(U[:, i].max() - U[:, i].min()) for i, k in enumerate(free)},  # scaled units
        runs=[{k: v for k, v in r.items() if k != "u"} for r in runs],
    )


FIT_CONFIG_V1 = dict(
    # Frozen 2026-10-15 (S2-6c) for consistent reference-fit labels; a research convention, not proven optimal
    # (experiments/s2_hl_area/REPORT.md). Labels from different configs must not be mixed.
    id="fit_config_v1", mode="7reg", lam=0.05, highlight_weight=30.0, area_weight=10.0,
    scales=dict(ev=1.0, temp=0.25, tint=0.25, sat=0.25, contrast=0.25, midtone=0.05, shoulder=0.25),
    bounds=dict(ev=(-3.0, 3.0), temp=(-1.0, 1.0), tint=(-1.0, 1.0), sat=(0.0, 2.0), contrast=(0.5, 2.0),
                midtone=(0.05, 0.5), shoulder=(0.0, 1.0)),
    objective="mean pseudo-Huber(dE2000) + highlight_weight * highlight_penalty + area_weight * clip_area "
              "+ lam * sum(((theta - default) / scale)^2)",
    huber_delta=1.0, highlight_delta_stops=0.05, area_ramp_stops=0.05, ref_clip_encoded=0.99,
    n_starts=5, n_sample=20000, seed=0, fd_step=2e-3, maxiter=300, near_optimal=0.1,
)


def fit_reference_v1(P, ref_lab) -> dict:
    """``fit_reference`` under FIT_CONFIG_V1; refuses to run if the module's scales, bounds or constants
    differ from the frozen values (e.g. inside ``experimental_bounds``)."""
    c = FIT_CONFIG_V1
    current = dict(scales=SCALES, bounds={k: tuple(BOUNDS[k]) for k in PER_IMAGE}, huber_delta=HUBER_DELTA,
                   highlight_delta_stops=HIGHLIGHT_DELTA, area_ramp_stops=CLIP_AREA_RAMP,
                   ref_clip_encoded=REF_CLIP_ENCODED, fd_step=FD_STEP, maxiter=MAXITER, near_optimal=NEAR_OPTIMAL,
                   lam=MODES[c["mode"]]["lam"])
    for k, v in current.items():
        if v != c[k]:
            raise RuntimeError(f"fit_config_v1: {k} is {v}, frozen value {c[k]}")
    r = fit_reference(P, ref_lab, c["mode"], n_starts=c["n_starts"], seed=c["seed"], n_sample=c["n_sample"],
                      lam=c["lam"], highlight_weight=c["highlight_weight"], area_weight=c["area_weight"])
    r["fit_config"] = c["id"]
    return r


def param_conditioning(P, params: RenderParams, names=PER_IMAGE, n_sample: int = 20000, seed: int = 0) -> dict:
    """Jacobian of the rendered Lab w.r.t. scaled parameters at ``params``: singular values of the
    scaled Jacobian (RMS Lab change per pixel per scale unit), and of its column-normalised form
    (min/max ratio: collinearity) with pairwise |cosine| between parameter effects."""
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
    sv_scaled = np.linalg.svd(J / np.sqrt(len(idx)), compute_uv=False)
    norms = np.linalg.norm(J, axis=0)
    Jn = J / np.where(norms > 0, norms, 1)
    s = np.linalg.svd(Jn, compute_uv=False)
    C = np.abs(Jn.T @ Jn)
    return dict(sv_ratio=float(s.min() / s.max()), sv_scaled=[float(x) for x in sv_scaled],
                cosine={f"{names[i]}/{names[j]}": float(C[i, j]) for i in range(len(names)) for j in range(i + 1, len(names))})
