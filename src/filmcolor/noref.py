"""Reference-free per-image render parameters (S2-7, config NOREF_V1). Pure numpy.

Input: the scene-linear positive P as the renderer sees it (``positive * inversion exposure``; continuous,
not clipped at the white point), a mask of image pixels to take statistics from, and optionally the
``frame_stats`` of other frames of the same roll. Never sees a reference image or reference-fit labels.
Design: docs/S2_7_NOREF_DESIGN.md. Parameter groups are estimated separately, each with a confidence:

- input quality: frames with too little tonal range get the defaults and a "needs review" flag.
- white balance (temp, tint): median chroma of low-saturation mid-tone pixels, shrunk by a confidence built
  from how many such pixels there are, how consistent they are and whether image quadrants agree; the rest
  of the way towards the roll estimate (if consistent) or 0. "Not enough evidence" means no change.
- exposure (ev): the median luminance is moved into an allowed band (not to a single target), capped so
  the highlights (p99.5) end at most ``headroom`` stops past white; then the smallest shoulder that keeps the
  share of pixels pushed past white within ``clip_budget`` (ev is lowered if even the full shoulder does not).
- contrast: the encoded p2..p98 luminance spread is moved into an allowed band.
- sat: a fixed default, lowered for images that are already very colourful. midtone stays at its default.

Global hyperparameters are tuned on the development sets only (data/manifests/SOURCES.md) and frozen with
the config id; development-set scores are not evidence of generalisation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .render import BOUNDS, LUMA, RenderParams, render_full

NOREF_V1 = dict(
    id="noref_v1",
    n_sample=200_000,             # pixels (fixed-seed subsample) used for statistics
    min_luma=0.002,               # ignore pixels darker than this (noise, below black)
    low_info_range_stops=3.0,     # p1..p99 luminance range below this: low-information frame
    median_band=(0.12, 0.30),     # allowed display-linear median luminance after ev
    headroom=1.5,                 # stops the p99.5 luminance may end past white before the shoulder
    ev_limits=(-3.0, 2.5),
    clip_budget=0.005,            # share of pixels allowed past white after the shoulder
    shoulder_steps=21,            # shoulder grid 0..1
    spread_band=(0.62, 0.80),     # allowed encoded p2..p98 luminance spread
    contrast_limits=(0.7, 1.8),
    sat=1.2, sat_colourful=0.9,   # default sat; images with p90 chroma (log2) above this get less
    wb_luma_pct=(20, 95),         # mid-tone pixels used for white balance
    wb_radius=0.3,                # log2 chroma radius of "low-saturation" candidates around neutral
    wb_share=(0.05, 0.30),        # candidate share mapping to confidence 0..1
    wb_mad=(0.06, 0.15),          # candidate chroma MAD mapping to confidence 1..0
    wb_conflict=0.15,             # quadrant estimates further apart than this (log2): conflict
    wb_conflict_factor=0.3,
    wb_max_shift=0.3,             # cap on |temp|, |tint|
    roll_consistency=0.1,         # roll estimates with weighted MAD below this are trusted fully
)


def _sample(P, mask, n, seed=0):
    px = np.asarray(P, np.float32)[mask] if mask is not None else np.asarray(P, np.float32).reshape(-1, 3)
    if len(px) > n:
        px = px[np.sort(np.random.default_rng(seed).choice(len(px), n, replace=False))]
    return px


def _chroma(px):
    """log2(R/G), log2(B/G)."""
    g = np.maximum(px[:, 1], 1e-6)
    return np.stack([np.log2(np.maximum(px[:, 0], 1e-6) / g), np.log2(np.maximum(px[:, 2], 1e-6) / g)], 1)


def _ramp(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def wb_from_chroma(c):
    """temp, tint that move chroma ``c`` = (log2 R/G, log2 B/G) to neutral (render.wb_gains)."""
    return float(-(c[0] - c[1])), float(-(c[0] + c[1]) / 2)


def _wb_candidates(px, cfg):
    y = px @ LUMA.astype(np.float32)
    lo, hi = np.percentile(y, cfg["wb_luma_pct"])
    mid = (y >= max(lo, cfg["min_luma"])) & (y <= hi)
    c = _chroma(px[mid])
    cand = np.linalg.norm(c, axis=1) <= cfg["wb_radius"]
    return c[cand], int(mid.sum())


def frame_stats(P, mask=None, cfg=NOREF_V1) -> dict:
    """Per-frame statistics, also used as roll context for other frames."""
    px = _sample(P, mask, cfg["n_sample"])
    if not np.all(np.isfinite(px)):
        raise ValueError("positive contains NaN or infinite values")
    y = px @ LUMA.astype(np.float32)
    yv = y[y > cfg["min_luma"]]
    if len(yv) < 100:
        raise ValueError("too few pixels above black for statistics")
    p1, p50, p99, p995 = (float(x) for x in np.percentile(yv, [1, 50, 99, 99.5]))
    cand, n_mid = _wb_candidates(px, cfg)
    est = np.median(cand, 0) if len(cand) else np.zeros(2)
    mad = float(np.median(np.linalg.norm(cand - est, axis=1))) if len(cand) else None
    return dict(median=p50, p1=p1, p99=p99, p995=p995, range_stops=float(np.log2(p99 / max(p1, cfg["min_luma"]))),
                above_white=float(np.mean(px.max(1) > 1)), dark_share=float(np.mean(y <= cfg["min_luma"])),
                wb_chroma=[float(x) for x in est], wb_share=len(cand) / max(n_mid, 1), wb_mad=mad)


def estimate_wb(P, mask, cfg=NOREF_V1, roll=None) -> dict:
    px = _sample(P, mask, cfg["n_sample"])
    cand, n_mid = _wb_candidates(px, cfg)
    share = len(cand) / max(n_mid, 1)
    diag = dict(share=share, n_candidates=len(cand))
    if len(cand) < 50:
        est, conf = np.zeros(2), 0.0
        diag["reason"] = "too few low-saturation mid-tone pixels"
    else:
        est = np.median(cand, 0)
        mad = float(np.median(np.linalg.norm(cand - est, axis=1)))
        conf = _ramp(share, *cfg["wb_share"]) * (1 - _ramp(mad, *cfg["wb_mad"]))
        diag.update(chroma=[float(x) for x in est], mad=mad)
        # quadrant agreement: a frame lit by two different sources gives no single answer
        quads = _quadrant_estimates(P, mask, cfg)
        if len(quads) >= 2:
            spread = max(float(np.linalg.norm(a - b)) for i, a in enumerate(quads) for b in quads[i + 1:])
            diag["quadrant_spread"] = spread
            if spread > cfg["wb_conflict"]:
                conf *= cfg["wb_conflict_factor"]
                diag["conflict"] = True
    target = conf * est
    roll_info = None
    if roll:
        ests = np.array([r["wb_chroma"] for r in roll])
        w = np.array([_ramp(r["wb_share"], *cfg["wb_share"]) for r in roll]) + 1e-9
        centre = np.array([_weighted_median(ests[:, i], w) for i in range(2)])
        spread = _weighted_median(np.linalg.norm(ests - centre, axis=1), w)
        roll_conf = 1 - _ramp(spread, cfg["roll_consistency"], 3 * cfg["roll_consistency"])
        target = target + (1 - conf) * roll_conf * centre
        roll_info = dict(chroma=[float(x) for x in centre], spread=float(spread), confidence=roll_conf, n=len(roll))
    temp, tint = wb_from_chroma(target)
    m = cfg["wb_max_shift"]
    return dict(temp=float(np.clip(temp, -m, m)), tint=float(np.clip(tint, -m, m)), confidence=conf,
                roll=roll_info, diagnostics=diag)


def _quadrant_estimates(P, mask, cfg):
    h, w = P.shape[:2]
    out = []
    for ys in (slice(0, h // 2), slice(h // 2, h)):
        for xs in (slice(0, w // 2), slice(w // 2, w)):
            m = mask[ys, xs] if mask is not None else None
            cand, _ = _wb_candidates(_sample(P[ys, xs], m, cfg["n_sample"] // 4), cfg)
            if len(cand) >= 50:
                out.append(np.median(cand, 0))
    return out


def _weighted_median(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, c[-1] / 2)])


def _clip_share(px, p: RenderParams):
    _, _, pre = render_full(px, p)
    return float(np.mean(pre.max(-1) > 1))


def _encoded_spread(px, p: RenderParams):
    e, _, _ = render_full(px, p)
    y = e @ LUMA.astype(np.float32)
    lo, hi = np.percentile(y, [2, 98])
    return float(hi - lo)


def estimate_params(P, mask=None, cfg=NOREF_V1, roll=None) -> dict:
    """Seven render parameters for one frame, with per-group confidence, flags and diagnostics.
    ``P``: (H, W, 3) positive; ``mask``: (H, W) bool image pixels (None = all); ``roll``: list of
    ``frame_stats`` of the other frames of the same roll (optional)."""
    P = np.asarray(P, np.float32)
    if mask is not None and mask.shape != P.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} != image shape {P.shape[:2]}")
    st = frame_stats(P, mask, cfg)
    flags, conf = [], {}
    if st["range_stops"] < cfg["low_info_range_stops"]:
        flags.append("low_information: needs review")
        return dict(config=cfg["id"], params=RenderParams().to_dict(), flags=flags,
                    confidence=dict(overall=0.0), stats=st, diagnostics={})
    if st["above_white"] > 0.05:
        flags.append("white_point_unrepresentative")  # highlights largely outside the white-point region

    wb = estimate_wb(P, mask, cfg, roll)
    conf["wb"] = wb["confidence"]
    px = _sample(P, mask, cfg["n_sample"] // 4, seed=1)

    # exposure: median into the band, highlights capped at `headroom` stops past white
    lo, hi = cfg["median_band"]
    m = st["median"]
    ev_mid = float(np.log2(np.clip(m, lo, hi) / m))
    ev_hl = float(np.log2(2 ** cfg["headroom"] / st["p995"]))
    ev = float(np.clip(min(ev_mid, ev_hl), *cfg["ev_limits"]))
    conf["exposure"] = 1.0 - _ramp(st["above_white"], 0.05, 0.25)
    base = dict(temp=wb["temp"], tint=wb["tint"])
    shoulder, ev_reduced = None, 0.0
    for _ in range(60):  # lower ev by 0.1 stop until some shoulder keeps clipping within budget
        for s in np.linspace(0, 1, cfg["shoulder_steps"]):
            if _clip_share(px, RenderParams(ev=ev, shoulder=float(s), **base)) <= cfg["clip_budget"]:
                shoulder = float(s)
                break
        if shoulder is not None or ev <= cfg["ev_limits"][0]:
            break
        ev, ev_reduced = max(ev - 0.1, cfg["ev_limits"][0]), ev_reduced + 0.1
    if shoulder is None:
        shoulder = 1.0
        flags.append("highlight_budget_not_met")
    if ev_reduced:
        flags.append(f"ev_lowered_for_highlights:{ev_reduced:.1f}")

    # contrast: encoded spread into the band
    p0 = dict(ev=ev, shoulder=shoulder, **base)
    spread = _encoded_spread(px, RenderParams(**p0))
    s_lo, s_hi = cfg["spread_band"]
    c_lo, c_hi = cfg["contrast_limits"]
    contrast = 1.0
    if spread < s_lo:
        contrast = _solve_contrast(px, p0, s_lo, 1.0, c_hi)
    elif spread > s_hi:
        contrast = _solve_contrast(px, p0, s_hi, c_lo, 1.0)

    # saturation
    c = _chroma(px[(px @ LUMA.astype(np.float32)) > cfg["min_luma"]])
    p90 = float(np.percentile(np.linalg.norm(c, axis=1), 90))
    sat = cfg["sat"] if p90 <= cfg["sat_colourful"] else 1 + (cfg["sat"] - 1) * cfg["sat_colourful"] / p90

    params = RenderParams(ev=ev, temp=wb["temp"], tint=wb["tint"], sat=float(sat), contrast=float(contrast),
                          shoulder=shoulder)
    _check_bounds(params)
    conf["overall"] = min(conf["exposure"], 1.0)
    return dict(config=cfg["id"], params=params.to_dict(), flags=flags, confidence=conf, stats=st,
                diagnostics=dict(wb=wb, ev_mid=ev_mid, ev_highlights=ev_hl, spread_before=spread,
                                 chroma_p90=p90, clip_share=_clip_share(px, params)))


def _solve_contrast(px, p0, target, lo, hi, iters=20):
    """Bisection on contrast for the encoded spread (monotonic increasing in contrast)."""
    f = lambda c: _encoded_spread(px, RenderParams(contrast=c, **p0)) - target  # noqa: E731
    if f(lo) > 0:
        return lo
    if f(hi) < 0:
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return float((lo + hi) / 2)


def _check_bounds(p: RenderParams):
    for k, (lo, hi) in BOUNDS.items():
        if not lo <= getattr(p, k) <= hi:
            raise AssertionError(f"{k}={getattr(p, k)} outside {lo, hi}")


# --- noref_v2: fixed global look + per-group adjustments, enabled one group at a time (S2-7 ablation) ---

CONFIG_DIR = Path(__file__).parent / "configs"
GROUPS = ("brightness", "wb", "tone", "sat")

NOREF_V2 = dict(
    NOREF_V1,
    id="noref_v2",
    look="global_look_v1",
    # groups adjusted per frame; the others keep the look's value. Empty: no group passed the pre-set keep rule
    # in the 2026-10-15 ablation (experiments/s2_noref/REPORT.md), so noref_v2 = look + input-quality gate.
    groups=(),
    wp_invalid_stops=1.0,         # p99 this many stops past white: white point unusable, manual review
    brighten_only=False,          # brightness: only lift frames below the band (high-key frames left alone)
    exp_shift_conf=(1.0, 2.5),    # |brightness shift| (stops) mapping to exposure confidence 1..0
    spread_band=None,             # tone: encoded p2..p98 spread band; set from development inputs before use
    contrast_floor=0.7,
)


def load_look(name: str = "global_look_v1") -> dict:
    look = json.loads((CONFIG_DIR / f"{name}.json").read_text())
    if look.get("id") != name:
        raise ValueError(f"look file {name}.json has id {look.get('id')!r}")
    RenderParams.from_dict(look["params"])  # validates names and bounds
    return look


def estimate_params_v2(P, mask=None, cfg=NOREF_V2, roll=None, look=None) -> dict:
    """noref_v2 / noref_v3 dispatcher: when ``cfg["id"] == "noref_v3"``, runs the adaptive print estimator
    ``estimate_params_v3``; otherwise runs ``noref_v2`` (start from the fixed look and adjust only the groups in
    ``cfg["groups"]``)."""
    if cfg.get("id") == "noref_v3":
        return estimate_params_v3(P, mask, cfg, roll, look)
    P = np.asarray(P, np.float32)
    if mask is not None and mask.shape != P.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} != image shape {P.shape[:2]}")
    unknown = set(cfg["groups"]) - set(GROUPS)
    if unknown:
        raise ValueError(f"unknown parameter groups: {sorted(unknown)}")
    look = look or load_look(cfg["look"])
    lp = dict(look["params"])
    st = frame_stats(P, mask, cfg)
    out = dict(config=cfg["id"], look=look["id"], groups=list(cfg["groups"]), stats=st, diagnostics={})
    reasons = []
    if st["range_stops"] < cfg["low_info_range_stops"]:
        reasons.append(f"low_information: tonal range {st['range_stops']:.2f} stops")
    if st["p99"] > 2 ** cfg["wp_invalid_stops"]:
        reasons.append(f"white_point_invalid: p99 {np.log2(st['p99']):.2f} stops past the inversion white point")
    if reasons:
        return dict(out, status="manual_review_required", params=RenderParams(**lp).to_dict(), flags=reasons,
                    confidence=dict(overall=0.0))
    flags = ["white_point_unrepresentative"] if st["above_white"] > 0.05 else []
    conf = {}
    p = dict(lp)
    px = _sample(P, mask, cfg["n_sample"] // 4, seed=1)
    g = cfg["groups"]

    if "brightness" in g:  # scene-linear median (after the look's ev) into the band, highlights capped
        lo, hi = cfg["median_band"]
        hi = np.inf if cfg["brighten_only"] else hi
        m = st["median"] * 2 ** lp["ev"]
        shift = min(float(np.log2(np.clip(m, lo, hi) / m)),
                    float(np.log2(2 ** cfg["headroom"] / (st["p995"] * 2 ** lp["ev"]))))
        p["ev"] = float(np.clip(lp["ev"] + shift, *cfg["ev_limits"]))
        conf["exposure"] = (1 - _ramp(st["above_white"], 0.05, 0.25)) * (1 - _ramp(abs(shift), *cfg["exp_shift_conf"]))
        out["diagnostics"]["brightness_shift"] = shift
    if "wb" in g:  # offset from the frame's neutral estimate (shrunk to the roll or 0), added to the look
        wb = estimate_wb(P, mask, cfg, roll)
        p["temp"], p["tint"] = lp["temp"] + wb["temp"], lp["tint"] + wb["tint"]
        conf["wb"] = wb["confidence"]
        out["diagnostics"]["wb"] = wb
    if "tone" in g:  # shoulder no lower than the look's, raised (then ev lowered) to keep clipping in budget
        s_grid = np.linspace(lp["shoulder"], 1.0, cfg["shoulder_steps"])
        shoulder, lowered = None, 0.0
        for _ in range(60):
            shoulder = next((float(s) for s in s_grid
                             if _clip_share(px, RenderParams(**dict(p, shoulder=float(s)))) <= cfg["clip_budget"]), None)
            if shoulder is not None or p["ev"] <= cfg["ev_limits"][0]:
                break
            p["ev"], lowered = max(p["ev"] - 0.1, cfg["ev_limits"][0]), lowered + 0.1
        if shoulder is None:
            shoulder = 1.0
            flags.append("highlight_budget_not_met")
        if lowered:
            flags.append(f"ev_lowered_for_highlights:{lowered:.1f}")
        p["shoulder"] = shoulder
        if cfg["spread_band"] is None:
            raise ValueError("tone group needs cfg['spread_band'] (set from development inputs)")
        spread = _encoded_spread(px, RenderParams(**p))
        s_lo, s_hi = cfg["spread_band"]
        if spread > s_hi:  # only lowered: the look's contrast is already at its upper bound
            p["contrast"] = _solve_contrast(px, {k: v for k, v in p.items() if k != "contrast"}, s_hi,
                                            cfg["contrast_floor"], lp["contrast"])
        elif spread < s_lo:
            flags.append("tonal_spread_below_band")
        out["diagnostics"]["spread_before"] = spread
    if "sat" in g:  # the look's saturation boost, reduced for frames that are already very colourful
        c = _chroma(px[(px @ LUMA.astype(np.float32)) > cfg["min_luma"]])
        p90 = float(np.percentile(np.linalg.norm(c, axis=1), 90))
        p["sat"] = float(1 + (lp["sat"] - 1) * min(1.0, cfg["sat_colourful"] / p90))
        out["diagnostics"]["chroma_p90"] = p90

    params = RenderParams(**p)
    _check_bounds(params)
    conf["overall"] = conf.get("exposure", 1.0)
    out["diagnostics"]["clip_share"] = _clip_share(px, params)
    return dict(out, status="ok", params=params.to_dict(), flags=flags, confidence=conf)


def estimation_input(P, image_region, f: int = 4):
    """The estimator's input as in the development runs: ``f`` x ``f`` block mean of the positive (as
    ``compare.block_mean``) and the image region (top, left, height, width) as a mask at that scale."""
    h, w = (P.shape[0] // f) * f, (P.shape[1] // f) * f
    P4 = np.asarray(P, np.float32)[:h, :w].reshape(h // f, f, w // f, f, 3).mean(axis=(1, 3)).astype(np.float32)
    t, l, rh, rw = (int(x) // f for x in image_region)
    mask = np.zeros(P4.shape[:2], bool)
    mask[t : t + rh, l : l + rw] = True
    return P4, mask


def encoded_spread(P, mask, params: dict, cfg=NOREF_V2) -> float:
    """Encoded p2..p98 luminance spread of a frame under ``params`` (input statistic, no reference)."""
    return _encoded_spread(_sample(P, mask, cfg["n_sample"] // 4, seed=1), RenderParams(**params))


# --- noref_v3: NegPy (marcinz606/NegPy) + Darktable negadoctor inspired textured-cell auto estimator (T4 v2) ---

NOREF_V3 = dict(
    NOREF_V1,
    id="noref_v3",
    rev="t4b-2026-10-24",          # T4b: contrast / shadow decisions on the encoded output, wb_v3, contrast centre 1.30
    look="global_look_v1",
    low_info_range_stops=0.6,      # allow open-sky / desert / snow frames with paper-black floor (p99/p1 >= 1.5x)
    cell_size=8,                   # grid cell side for textured-cell activity metering
    activity_thr_d=0.05,           # NegPy activity threshold: within-cell std(-log10 Y) >= 0.05 D
    min_textured_share=0.08,       # fallback to top 25% most active cells when fewer than 8% exceed activity_thr_d
    anchor_target_density=0.75,    # NegPy print target density 0.75 D (reflectance 10^-0.75 ≈ 0.178 = 18% grey)
    anchor_band_d=0.08,            # comfort band around target density (preserves high-key / low-key mood)
    base_contrast=1.30,            # natural print contrast centre (UI contrast 0); replaces fixed contrast=2.0
    contrast_bounds_v3=(1.05, 1.38),  # contrast is only lowered (to >= 1.05) for very wide, both-ends-clipped frames
    contrast_spread_max=0.78,      # encoded luminance p90 - p10 above this, with both ends clipped: lower contrast
    end_clip_share=0.005,          # "clipped end": share of encoded luminance <= 0.02 (black) or >= 0.98 (white)
    base_sat=1.02,                 # restrained saturation
    base_midtone=0.22,             # comfortable mid-grey placement
    base_shoulder=0.15,            # gentle baseline highlight roll-off
    target_encoded_band=(0.45, 0.52),  # target sRGB-encoded median luminance band
    shadow_p25_floor=0.12,         # shadows lifted only when encoded luminance p25 is below this (crushed frame)
    shadow_gain=2.5, shadow_max=0.35,  # shadows = min(shadow_max, shadow_gain * (floor - p25)); blacks stay 0
    clip_budget=0.002,             # <= 0.2 % highlight clipping
    headroom=1.2,                  # max stops p99.5 may exceed white before shoulder
    ev_limits=(-2.0, 2.5),
    wb_prior_weight=0.55,          # prior from global_look_v1, applied for the share (1 - wb confidence)
    wb_max_shift=0.35,
    wb_v3_zones=(((70, 96), 0.35, 0.6), ((25, 75), 0.25, 0.4)),  # (luma pct range, chroma radius log2, weight):
                                   # bright near-neutral (cloud, road, white surfaces) and low-saturation mid-grey
    wb_v3_min_px=200,              # pixels a zone needs to count
    wb_v3_huber=0.08,              # Huber scale (log2 chroma) of the robust zone location
    wb_v3_share=(0.01, 0.08),      # candidate share mapping to confidence 0..1
    wb_v3_mad=(0.10, 0.35),        # candidate MAD mapping to confidence factor 1..0.5 (never 0)
    wb_v3_conflict_factor=0.75,    # quadrant disagreement lowers confidence by this factor (never to 0)
    wb_v3_strength=0.85,           # share of the measured residual cast removed at full confidence
    wp_invalid_stops=1.5,
)


def _textured_cells(P: np.ndarray, mask: np.ndarray | None, cfg: dict) -> dict:
    """NegPy-style textured-cell metering: partition ``P`` into ``cell_size x cell_size`` cells inside ``mask``,
    measure within-cell log-density activity ``std(-log10 Y)``, and return textured cell means and quantiles.
    Units: ``-log10`` of the scene-linear positive luminance, i.e. density below the inversion white point (0 at the
    white point, 0.75 = 17.8 % of it). A normal scene spans 1.5..2.5 of these, so they are not print densities: the
    T4b contrast and shadow decisions use the encoded output instead (``_encoded_quantiles``)."""
    cs = int(cfg.get("cell_size", 8))
    h, w = P.shape[:2]
    if h < cs * 2 or w < cs * 2:
        cs = max(1, min(h, w) // 4)
    hc, wc = h // cs, w // cs
    Pc = np.asarray(P[: hc * cs, : wc * cs], np.float32).reshape(hc, cs, wc, cs, 3)
    Y = np.maximum(Pc @ LUMA.astype(np.float32), 1e-4)
    D = -np.log10(Y)
    std_D = D.std(axis=(1, 3))
    mean_D = D.mean(axis=(1, 3))
    mean_rgb = Pc.mean(axis=(1, 3))
    if mask is not None:
        mc = mask[: hc * cs, : wc * cs].reshape(hc, cs, wc, cs).mean(axis=(1, 3)) > 0.5
    else:
        mc = np.ones((hc, wc), bool)
    valid = mc & np.isfinite(std_D) & (mean_rgb.max(axis=-1) > cfg["min_luma"])
    if not valid.any():
        valid = np.ones((hc, wc), bool)
    thr = float(cfg.get("activity_thr_d", 0.05))
    tex = valid & (std_D >= thr)
    tex_share = float(tex.sum() / max(valid.sum(), 1))
    if tex_share < float(cfg.get("min_textured_share", 0.08)):
        q75 = float(np.percentile(std_D[valid], 75))
        tex = valid & (std_D >= q75)
    dt = mean_D[tex]
    p2, p5, p10, p50, p90, p95, p98 = (float(x) for x in np.percentile(dt, [2, 5, 10, 50, 90, 95, 98]))
    in_5_95 = dt[(dt >= p5) & (dt <= p95)]
    mean_5_95 = float(in_5_95.mean()) if len(in_5_95) else p50
    anchor_d = 0.5 * (mean_5_95 + 0.5 * (p5 + p95))
    return dict(
        textured_share=tex_share,
        n_textured=int(tex.sum()),
        anchor_d=anchor_d,
        span_10_90_d=float(p90 - p10),
        p2_bright_d=p2,
        p10_d=p10,
        p50_d=p50,
        p90_d=p90,
        p98_dark_d=p98,
        textured_rgb=mean_rgb[tex],
    )


def _huber_location(c, k: float, iters: int = 10) -> np.ndarray:
    """Huber M-estimate of the 2-D location of chroma samples ``c`` (n, 2), starting from the median."""
    m = np.median(c, 0)
    for _ in range(iters):
        r = np.linalg.norm(c - m, axis=1)
        w = np.minimum(1.0, k / np.maximum(r, 1e-9))
        m = (w[:, None] * c).sum(0) / w.sum()
    return m


def estimate_wb_v3(px, cfg=None) -> dict:
    """T4b white balance on sample pixels ``px`` (n, 3) of the positive: in each zone of ``cfg["wb_v3_zones"]``
    (luminance percentile range, chroma radius, weight) the near-neutral pixels give a robust (Huber) residual cast;
    zones with at least ``wb_v3_min_px`` pixels are combined by weight. Confidence comes from the candidate share and
    spread and is lowered, never zeroed, when image quadrants disagree (a frame with sky above and grass below is not
    evidence against a cast). temp / tint remove ``wb_v3_strength`` x confidence of the cast."""
    cfg = cfg or NOREF_V3
    px = np.asarray(px, np.float32)
    y = px @ LUMA.astype(np.float32)
    ok = y > cfg["min_luma"]
    px, y = px[ok], y[ok]
    diag = dict(n=int(len(px)), zones=[])
    if len(px) < cfg["wb_v3_min_px"]:
        return dict(temp=0.0, tint=0.0, confidence=0.0, chroma=[0.0, 0.0], diagnostics=dict(diag, reason="too few pixels"))
    c = _chroma(px)
    norm = np.linalg.norm(c, axis=1)
    ests, weights, cands = [], [], []
    for (q_lo, q_hi), radius, weight in cfg["wb_v3_zones"]:
        lo, hi = np.percentile(y, [q_lo, q_hi])
        sel = (y >= lo) & (y <= hi) & (norm <= radius)
        zd = dict(luma_pct=[q_lo, q_hi], radius=radius, n=int(sel.sum()), share=float(sel.sum() / max(len(y), 1)))
        if sel.sum() >= cfg["wb_v3_min_px"]:
            est = _huber_location(c[sel], cfg["wb_v3_huber"])
            zd.update(chroma=[float(v) for v in est])
            ests.append(est), weights.append(weight), cands.append(c[sel])
        diag["zones"].append(zd)
    if not ests:
        return dict(temp=0.0, tint=0.0, confidence=0.0, chroma=[0.0, 0.0],
                    diagnostics=dict(diag, reason="no near-neutral zone with enough pixels"))
    w = np.asarray(weights) / np.sum(weights)
    est = (w[:, None] * np.asarray(ests)).sum(0)
    allc = np.concatenate(cands)
    mad = float(np.median(np.linalg.norm(allc - est, axis=1)))
    share = float(sum(z["share"] for z in diag["zones"] if "chroma" in z) / len(diag["zones"]))
    conf = _ramp(share, *cfg["wb_v3_share"]) * (1 - 0.5 * _ramp(mad, *cfg["wb_v3_mad"]))
    if len(ests) == 2:
        diag["zone_disagreement"] = float(np.linalg.norm(ests[0] - ests[1]))
    quads = []
    n = len(px)
    # quadrants are approximated by sample order: _sample keeps row-major order, so halves of the sample are the
    # top and bottom of the frame; left / right are not separable here, so only top vs bottom is checked
    for half in (slice(0, n // 2), slice(n // 2, n)):
        ch, yh, nh = c[half], y[half], norm[half]
        lo, hi = np.percentile(yh, [25, 96])
        selh = (yh >= lo) & (yh <= hi) & (nh <= max(r for _, r, _ in cfg["wb_v3_zones"]))
        if selh.sum() >= cfg["wb_v3_min_px"]:
            quads.append(_huber_location(ch[selh], cfg["wb_v3_huber"]))
    if len(quads) == 2:
        diag["top_bottom_spread"] = float(np.linalg.norm(quads[0] - quads[1]))
        if diag["top_bottom_spread"] > cfg["wb_conflict"]:
            conf *= cfg["wb_v3_conflict_factor"]
            diag["conflict"] = True
    temp, tint = wb_from_chroma(cfg["wb_v3_strength"] * conf * est)
    m = cfg["wb_max_shift"]
    diag.update(share=share, mad=mad)
    return dict(temp=float(np.clip(temp, -m, m)), tint=float(np.clip(tint, -m, m)), confidence=float(conf),
                chroma=[float(v) for v in est], diagnostics=diag)


def _encoded_quantiles(px, p: RenderParams, qs=(2, 50, 98)):
    e, _, _ = render_full(px, p)
    y = e @ LUMA.astype(np.float32)
    return [float(x) for x in np.percentile(y, qs)]


def estimate_params_v3(P, mask=None, cfg=NOREF_V3, roll=None, look=None) -> dict:
    """Textured-cell adaptive print estimator (NOREF_V3, T4 v2, inspired by marcinz606/NegPy & negadoctor):

    1. **Textured Cells**: filters grid cells by local density activity ``std(-log10 Y) >= 0.05 D`` so large
       uniform skies, water or borders do not bias metering.
    2. **Auto Density (ev)**: pulls the textured P5..P95 anchor density towards ``anchor_target_density = 0.75 D``
       (18 % mid-grey) with a comfort band, then refines so sRGB-encoded median luminance lands in the target band.
    3. **Contrast** (T4b): starts at the print centre ``base_contrast = 1.30`` (UI contrast 0); lowered (to >= 1.05)
       only when the encoded luminance p90 - p10 exceeds ``contrast_spread_max`` and both ends clip.
    4. **Highlight Hold (highlights & shoulder)**: when bright textured cells ``P2`` approach the white point or
       full-frame clipping exceeds ``clip_budget`` (0.2 %), writes an explicit negative ``highlights`` burn and/or a
       raised ``shoulder``.
    5. **Shadow Hold** (T4b): positive ``shadows`` only when the encoded luminance p25 is below ``shadow_p25_floor``;
       ``blacks`` is never set automatically.
    6. **White balance** (T4b, ``estimate_wb_v3``): robust residual cast of bright near-neutral and low-saturation
       mid-grey pixels; the look's prior covers only the share the evidence does not.

    T4b (2026-10-24) replaced the T4 v2 contrast / shadow / paper-black rules, which compared ``-log10`` of the
    scene-linear positive (``_textured_cells``) with print-density thresholds and so lowered contrast to its floor
    and lifted shadows and blacks on almost every frame.
    """
    P = np.asarray(P, np.float32)
    if mask is not None and mask.shape != P.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} != image shape {P.shape[:2]}")
    look = look or load_look(cfg["look"])
    lp = dict(look["params"])
    st = frame_stats(P, mask, cfg)
    out = dict(config=cfg["id"], config_rev=cfg.get("rev"), look=look["id"], groups=["brightness", "wb", "tone", "sat"],
               stats=st, diagnostics={})
    reasons = []
    if st["range_stops"] < cfg["low_info_range_stops"]:
        reasons.append(f"low_information: tonal range {st['range_stops']:.2f} stops")
    if st["p99"] > 2 ** cfg["wp_invalid_stops"]:
        reasons.append(f"white_point_invalid: p99 {np.log2(st['p99']):.2f} stops past the inversion white point")
    base_p = dict(
        ev=0.0,
        temp=float(lp["temp"] * cfg["wb_prior_weight"]),
        tint=float(lp["tint"] * cfg["wb_prior_weight"]),
        sat=float(cfg["base_sat"]),
        contrast=float(cfg["base_contrast"]),
        midtone=float(cfg["base_midtone"]),
        shoulder=float(cfg["base_shoulder"]),
        highlights=0.0, shadows=0.0, whites=0.0, blacks=0.0, vibrance=0.0,
    )
    if reasons:
        return dict(out, status="manual_review_required", params=RenderParams(**base_p).to_dict(),
                    flags=reasons, confidence=dict(overall=0.0))

    flags = ["white_point_unrepresentative"] if st["above_white"] > 0.05 else []
    px = _sample(P, mask, cfg["n_sample"] // 4, seed=1)
    tc = _textured_cells(P, mask, cfg)
    out["diagnostics"]["textured_cells"] = {k: v for k, v in tc.items() if k != "textured_rgb"}

    # 1. White balance (T4b): robust bright-neutral + mid-grey residual cast (estimate_wb_v3); the look's prior only
    #    for the share the evidence does not cover
    wb = estimate_wb_v3(px, cfg)
    m_wb = cfg["wb_max_shift"]
    base_p["temp"] = float(np.clip((1 - wb["confidence"]) * base_p["temp"] + wb["temp"], -m_wb, m_wb))
    base_p["tint"] = float(np.clip((1 - wb["confidence"]) * base_p["tint"] + wb["tint"], -m_wb, m_wb))
    out["diagnostics"]["wb"] = wb

    # 2. Saturation / vibrance: reduce saturation slightly on already high-chroma frames
    c_all = _chroma(px[(px @ LUMA.astype(np.float32)) > cfg["min_luma"]])
    p90 = float(np.percentile(np.linalg.norm(c_all, axis=1), 90)) if len(c_all) else 0.3
    if p90 > cfg["sat_colourful"]:
        base_p["sat"] = float(1.0 + (cfg["base_sat"] - 1.0) * (cfg["sat_colourful"] / p90))
    else:
        base_p["vibrance"] = 0.08
    out["diagnostics"]["chroma_p90"] = p90

    # 3. Contrast starts at the print centre base_contrast (T4b); lowered after exposure only for very wide frames
    c_lo, c_hi = cfg["contrast_bounds_v3"]
    base_p["contrast"] = float(np.clip(cfg["base_contrast"], c_lo, c_hi))

    # 4. Auto Density (NegPy textured anchor_d vs anchor_target_density = 0.75 D, with high-key protection)
    target_d = float(cfg["anchor_target_density"])
    band_d = float(cfg["anchor_band_d"])
    anc_d = tc["anchor_d"]
    if anc_d > target_d + band_d:
        # Underexposed / dark textured cells: lift strongly towards target_d
        ev_anc = (anc_d - (target_d + band_d * 0.4)) / 0.30103
    elif anc_d < target_d - band_d:
        # High-key open sky / desert / snow (anc_d < 0.67 D): preserve high-key mood, pull back at most -0.25 EV
        ev_anc = max(-0.25, (anc_d - (target_d - band_d)) * 0.35 / 0.30103)
    else:
        ev_anc = (anc_d - target_d) * 0.4 / 0.30103
    base_p["ev"] = float(np.clip(ev_anc, *cfg["ev_limits"]))

    # Refine EV: lift dark frames into target_encoded_band (0.45..0.52); high-key frames allowed up to 0.76
    t_lo, t_hi = cfg["target_encoded_band"]
    p2_0, med0, p98_0 = _encoded_quantiles(px, RenderParams(**base_p))
    t_hi_eff = 0.76 if anc_d < 0.65 else t_hi
    if med0 < t_lo or med0 > t_hi_eff:
        target_m = t_lo + 0.03 if med0 < t_lo else t_hi_eff - 0.02
        lo_ev, hi_ev = cfg["ev_limits"]
        for _ in range(18):
            mid_ev = 0.5 * (lo_ev + hi_ev)
            _, m_cur, _ = _encoded_quantiles(px, RenderParams(**dict(base_p, ev=mid_ev)))
            if m_cur < target_m:
                lo_ev = mid_ev
            else:
                hi_ev = mid_ev
        base_p["ev"] = float(np.clip(0.5 * (lo_ev + hi_ev), *cfg["ev_limits"]))

    # 5. NegPy Highlight Hold: check bright textured cells P2 after EV shift, plus full-frame clip_budget
    p2_shifted_d = tc["p2_bright_d"] - 0.30103 * base_p["ev"]
    if p2_shifted_d < 0.12:
        # Automatic highlight burn written directly to the highlights slider
        base_p["highlights"] = float(np.clip(-1.8 * (0.12 - p2_shifted_d), -0.65, 0.0))

    s_grid = np.linspace(base_p["shoulder"], 0.85, 15)
    hl_grid = sorted({round(base_p["highlights"], 3), -0.20, -0.40, -0.65}, reverse=True)
    found = False
    for hl in hl_grid:
        if hl > base_p["highlights"]:
            continue
        for s in s_grid:
            cand_p = RenderParams(**dict(base_p, shoulder=float(s), highlights=float(hl)))
            if _clip_share(px, cand_p) <= cfg["clip_budget"]:
                base_p["shoulder"], base_p["highlights"] = float(s), float(hl)
                found = True
                break
        if found:
            break
    ev_lowered = 0.0
    if not found:
        base_p["shoulder"], base_p["highlights"] = 0.85, -0.65
        for _ in range(30):
            if _clip_share(px, RenderParams(**base_p)) <= cfg["clip_budget"] or base_p["ev"] <= cfg["ev_limits"][0]:
                break
            base_p["ev"] = float(max(base_p["ev"] - 0.1, cfg["ev_limits"][0]))
            ev_lowered += 0.1
    if ev_lowered > 0:
        flags.append(f"ev_lowered_for_highlights:{ev_lowered:.1f}")

    # 6. Contrast relief (T4b, encoded output): only a very wide frame (encoded luminance p90 - p10 above
    #    contrast_spread_max) with both ends clipped gets a lower contrast, down to the lower bound
    def ends(p):
        e, _, _ = render_full(px, RenderParams(**p))
        y = e @ LUMA.astype(np.float32)
        q10, q90 = np.percentile(y, [10, 90])
        return float(q90 - q10), float(np.mean(y <= 0.02)), float(np.mean(y >= 0.98))

    spread, black_end, white_end = ends(base_p)
    out["diagnostics"]["contrast_check"] = dict(spread_10_90=spread, black_end=black_end, white_end=white_end)
    if spread > cfg["contrast_spread_max"] and min(black_end, white_end) > cfg["end_clip_share"]:
        lo_c, hi_c = c_lo, base_p["contrast"]
        if ends(dict(base_p, contrast=lo_c))[0] > cfg["contrast_spread_max"]:
            base_p["contrast"] = lo_c
        else:
            for _ in range(14):  # bisection: spread increases with contrast
                mid_c = 0.5 * (lo_c + hi_c)
                lo_c, hi_c = (mid_c, hi_c) if ends(dict(base_p, contrast=mid_c))[0] <= cfg["contrast_spread_max"] else (lo_c, mid_c)
            base_p["contrast"] = float(lo_c)
        flags.append("contrast_lowered_wide_frame")

    # 7. Shadow Hold (T4b, encoded output): shadows lifted only on crushed frames (encoded luminance p25 below
    #    shadow_p25_floor); blacks are never set automatically (paper-black / matte stays a user choice)
    p2, p25, p50, p98 = _encoded_quantiles(px, RenderParams(**base_p), qs=(2, 25, 50, 98))
    out["diagnostics"]["encoded_p25_before_shadows"] = p25
    if p25 < cfg["shadow_p25_floor"]:
        base_p["shadows"] = float(np.clip(cfg["shadow_gain"] * (cfg["shadow_p25_floor"] - p25), 0.0, cfg["shadow_max"]))

    # Round slider-visible values cleanly
    for k in ("ev", "temp", "tint", "sat", "contrast", "shoulder", "highlights", "shadows", "whites", "blacks", "vibrance"):
        base_p[k] = round(float(base_p[k]), 4)

    params = RenderParams(**base_p)
    _check_bounds(params)
    p2_f, p25_f, p50_f, p98_f = _encoded_quantiles(px, params, qs=(2, 25, 50, 98))
    clip_f = _clip_share(px, params)
    out["diagnostics"].update(
        encoded_p2=p2_f, encoded_p25=p25_f, encoded_median=p50_f, encoded_p98=p98_f,
        spread_before=float(p98 - p2), clip_share=clip_f,
    )
    conf = dict(wb=wb["confidence"], exposure=float(1.0 - _ramp(st["above_white"], 0.05, 0.25)), overall=1.0)
    return dict(out, status="ok", params=params.to_dict(), flags=flags, confidence=conf)
