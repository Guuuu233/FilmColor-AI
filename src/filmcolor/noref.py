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
    """noref_v2: start from the fixed look (``look``, default ``cfg["look"]``) and adjust only the groups in
    ``cfg["groups"]``. Frames whose input statistics make the base inversion untrustworthy (too little
    tonal range, or highlights far past the inversion white point) get status ``manual_review_required``
    and the look unchanged: no per-frame enhancement on top of a suspect inversion."""
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
