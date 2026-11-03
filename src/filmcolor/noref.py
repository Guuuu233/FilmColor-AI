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

from .holder import film_box, holder_mask
from .render import BOUNDS, EXTENDED_BOUNDS, IDENTITY, LUMA, RenderParams, render_full

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


def estimate_params_v2(P, mask=None, cfg=NOREF_V2, roll=None, look=None, *, span_d_g: float | None = None) -> dict:
    """noref_v2 / noref_v3 dispatcher: when ``cfg["id"] == "noref_v3"``, runs the adaptive print estimator
    ``estimate_params_v3`` (``span_d_g``: the inversion's G density range); otherwise runs ``noref_v2`` (start from the
    fixed look and adjust only the groups in ``cfg["groups"]``)."""
    if cfg.get("id") == "noref_v3":
        return estimate_params_v3(P, mask, cfg, roll, look, span_d_g=span_d_g)
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
    rev="t4v-2026-11-03",          # T4v: TIFF transfer curves, 8-bit input and pre-stretched signal reconstruction;
                                   # T4u: pre-stretched TIFFs, unsupported positive tint and narrow-blue rescue;
                                   # T4t: zero-clipped white points, slide mounts and conflicting tint evidence;
                                   # T4s: scanner TIFF, global zero-escape evidence and B&W roll routing;
                                   # T4r: wide-bright over-lift exemption and sparse / conflicting WB gates;
                                   # T4q: fixed spatial radius and bounded darkening of lifted shadows;
                                   # T4p: bounded highlight EV drop, pre-hold rescue evidence and bright-sky split;
                                   # T4m: thin-negative darkening without backlit lift, strong cyan confidence floor;
                                   # T4l: metering inside the film box and off full black strips, wider ev lowering
                                   # on frames bright all over, too short a density range for manual review;
                                   # T4k: thin-negative ev cap, warm-scene cooling damped, green tint countered,
                                   # cool mid greys keep the grey world undamped (the task list's rebate filter,
                                   # backlit bright share and cool-cast temp cap measured and off by default); T4j: no sky damping on a uniform cool cast (bright, mid-grey and grey
                                   # world all cool), auto levels off by default; T4i (not adopted): sky split, auto levels;
                                   # T4h: sky damping on the grey-world blue axis only, milder portrait;
                                   # T4g: adaptive WB and softer portrait / natural presets; T4f: trimean anchor,
                                   # textured EV refinement (at most -0.20 EV down), wider backlit, sky WB damping;
                                   # T4e: standard back to the T4c band, backlit weight narrowed and relative to the
                                   # preset; T4d: neutral wb fallback, style presets, continuous backlit weight;
                                   # T4c: camera matrix, chroma_preserve, backlit midtone; T4b: encoded-output decisions
    look="global_look_v1",         # recorded only: wb_prior_weight 0 (T4d) applies none of it
    preset="standard",             # STYLE_PRESETS name this config is (NOREF_V3 itself is the standard preset)
    low_info_range_stops=0.6,      # allow open-sky / desert / snow frames with paper-black floor (p99/p1 >= 1.5x)
    cell_size=8,                   # grid cell side for textured-cell activity metering
    activity_thr_d=0.05,           # NegPy activity threshold: within-cell std(-log10 Y) >= 0.05 D
    min_textured_share=0.08,       # fallback to top 25% most active cells when fewer than 8% exceed activity_thr_d
    anchor_target_density=0.75,    # NegPy print target density 0.75 D (reflectance 10^-0.75 ≈ 0.178 = 18% grey)
    anchor_band_d=0.08,            # comfort band around target density (preserves high-key / low-key mood)
    base_contrast=1.30,            # standard print contrast (UI contrast 0)
    contrast_bounds_v3=(1.05, 1.38),  # contrast is only lowered (to >= 1.05) for very wide, both-ends-clipped frames;
                                   # the upper bound never cuts a preset's base_contrast
    contrast_spread_max=0.78,      # encoded luminance p90 - p10 above this, with both ends clipped: lower contrast
    end_clip_share=0.005,          # "clipped end": share of encoded luminance <= 0.02 (black) or >= 0.98 (white)
    base_sat=1.02,                 # restrained saturation
    base_vibrance=0.08,            # vibrance on frames that are not already colourful (chroma p90 <= sat_colourful)
    base_midtone=0.23,             # mid-grey placement (UI midtone 0)
    base_shoulder=0.22,            # baseline highlight roll-off
    base_shadows=0.0, base_highlights=0.0, base_slope_r=0.0, base_slope_b=0.0,
    base_blacks=0.0,               # blacks of the preset (a constant; never estimated from the frame)
    temp_offset=0.0, tint_offset=0.0,  # preset colour offsets added after the white balance estimate
    base_tint=0.015,               # T4k: added to every preset's tint offset (counter the set's green lean)
    # T4k (off by default: it also drops dark foregrounds, A012 4.02 -> 9.16): deep-black rebate / gaps between frames
    # inside the central region are not metered: a cell counts only when its mean max channel exceeds cell_min_luma and
    # cell_bright_px_share of its pixels have max channel above cell_min_luma_px (fewer than cell_min_textured textured
    # cells left: the cells without this filter)
    cell_border_filter=False, cell_min_luma=0.022, cell_bright_px_share=0.80, cell_min_luma_px=0.015, cell_min_textured=50,
    # T4k (off by default, with cell_border_filter): the EV refinement's p98 ceiling and the backlit check use the
    # pixels with max channel above tone_min_luma (fewer than wb_v3_min_px: all); bright share (for the backlit and thin
    # checks) = share of the tone pixels with linear luminance above bright_luma
    tone_border_filter=False, tone_min_luma=0.020, bright_luma=0.45,
    # T4k (off by default: 0 better / 2 worse): the backlit weight is multiplied by ramp(bright share;
    # backlit_sky_share_lo, backlit_min_sky_share)
    backlit_sky_check=False, backlit_sky_share_lo=0.04, backlit_min_sky_share=0.12,
    # T4m: a thin negative (G density range below thin_span_max and few bright pixels) gets ev capped at
    # clip((span - thin_span_zero) / thin_span_ramp x thin_ev_max, thin_ev_min, thin_ev_max), including negative ev;
    # below thin_blacks_span, blacks is lowered by clip((thin_blacks_span - span) x thin_blacks_gain, 0, thin_blacks_max).
    # When applied below thin_suppress_backlit_span, the backlit lift is suppressed (its whites came from stretching).
    thin_neg_check=True, thin_span_max=0.62, thin_bright_share_max=0.42, thin_span_zero=0.46, thin_span_ramp=0.16,
    thin_ev_min=-0.45, thin_ev_max=0.60, thin_blacks_span=0.56, thin_blacks_gain=1.0, thin_blacks_max=0.20,
    thin_suppress_backlit_span=0.58,
    # T4l metering frame (_metering_mask): with an opaque holder covering >= meter_holder_min_share of the frame, the
    # mask is cut to the film box (holder.film_box) inset by meter_box_inset of its height / width on each side;
    # then rows / columns at the mask's edges with >= meter_strip_share of their mask pixels at max channel <=
    # meter_strip_luma (full black strips: rebate, gaps between frames) are cut, scanning inward from each side, plus
    # meter_strip_margin more
    meter_holder_box=True, meter_holder_min_share=0.01, meter_box_inset=0.08,
    meter_trim_strips=True, meter_strip_luma=0.012, meter_strip_share=0.80, meter_strip_margin=2,
    meter_min_share=0.25,          # the metering mask keeps at least this share of the original mask, else unchanged
    # T4l (off by default: 4 better / 10 worse on the 185 dev frames due to low-contrast ext_own frames): lowering the
    # ev on frames bright all over (textured-cell encoded p25 >= ev_down_q25[0]) by up to
    # ev_refine_down_max + ev_down_extra x ramp(p25; ev_down_q25), instead of ev_refine_down_max alone (A013, a sky over
    # dark ground, keeps 0.20)
    ev_down_adaptive=False, ev_down_q25=(0.42, 0.55), ev_down_extra=0.45,
    min_span_d_g=0.22,             # T4l: inversion G density range below this: low information, manual review; None = off
    target_encoded_band=(0.45, 0.52),  # target sRGB-encoded median luminance of the textured cells (T4g; T4d
                                   # 0.48..0.55 and T4c / T4e 0.45..0.52 on the full frame)
    ev_refine_down_max=0.20,       # T4f: the EV refinement lowers exposure by at most this, and not below -this
    ev_highlight_max_drop=0.60,    # T4p: limit global darkening for sparse border / lamp highlights in Step 5
    # T4q: after Highlight Hold, pull back large EV gains when Step 4b's encoded shadows were already bright
    overlift_damp_check=True, overlift_ev_min=1.50, overlift_ev_pivot=1.30, overlift_ev_ramp=0.50,
    overlift_q25_min=0.35, overlift_q25_pivot=0.33, overlift_gain=3.2, overlift_max_pull=0.65,
    overlift_blacks_gain=0.10, overlift_blacks_max=0.06,
    overlift_max_span=1.15, overlift_max_bright_share=0.05,  # T4r: exempt wide-range bright scenes
    ev_refine_p98_max=0.96,        # T4f: the EV refinement brightens only while full-frame encoded p98 stays <= this
    ev_anchor_p98_cap=True,        # T4f: a positive anchor ev is held to the same p98 ceiling (the task list puts the
                                   # ceiling on the refinement only, which left A012 at the anchor's +2.50 EV)
    shadow_p25_floor=0.12,         # shadows lifted only when encoded luminance p25 is below this (crushed frame)
    shadow_gain=2.5, shadow_max=0.35,  # shadows = min(shadow_max, shadow_gain * (floor - p25)); blacks stay 0
    clip_budget=0.002,             # <= 0.2 % highlight clipping
    headroom=1.2,                  # max stops p99.5 may exceed white before shoulder
    ev_limits=(-2.0, 2.5),
    wb_prior_weight=0.0,           # T4d: the share (1 - wb confidence) falls back to neutral (0, 0), not to
                                   # global_look_v1 (fitted on nlp_official + shop_r01 references; T4b-T4c used 0.55)
    wb_max_shift=0.50,
    wb_v3_adaptive_share=0.12, wb_v3_adaptive_pct=15.0,
    wb_v3_gw_weight=0.45, wb_v3_gw_threshold=0.18, wb_v3_gw_huber=0.18,
    wb_v3_gw_sky_red=0.75,         # T4h: clear-sky grey-world estimate keeps this share of its red axis (blue axis
                                   # x wb_v3_sky_weight)
    wb_v3_zones=(((70, 96), 0.35, 0.6), ((25, 75), 0.25, 0.4)),  # (luma pct range, chroma radius log2, weight):
                                   # bright near-neutral (cloud, road, white surfaces) and low-saturation mid-grey
    wb_v3_min_px=200,              # pixels a zone needs to count
    wb_v3_huber=0.08,              # Huber scale (log2 chroma) of the robust zone location
    wb_v3_share=(0.01, 0.08),      # candidate share mapping to confidence 0..1
    wb_v3_mad=(0.10, 0.35),        # candidate MAD mapping to confidence factor 1..0.5 (never 0)
    wb_v3_conflict_factor=0.75,    # quadrant disagreement lowers confidence by this factor (never to 0)
    wb_v3_strength=0.65,           # share of the measured residual cast removed at full confidence (T4b-T4e 0.85)
    wb_v3_sky=(-0.05, 0.05),       # T4f: bright zone reads as clear sky when its log2 R/G < [0] and log2 B/G > [1]
    wb_v3_sky_weight=0.35,         # T4f: then its weight is multiplied by this (the mid-grey zone leads)
    # T4j: a uniform cool cast, the bright zone, the mid-grey zone and the midtone grey world each with log2 R/G below
    # [0] and log2 B/G above [1] of its pair, is not damped as a sky; it gets grey-world weight wb_v3_cast_gw_weight,
    # confidence at least wb_v3_cast_conf and strength wb_v3_cast_strength. False = the T4h rule.
    wb_v3_cool_cast_check=True,
    wb_v3_cool_z0=(-0.05, 0.05), wb_v3_cool_z1=(-0.10, 0.10), wb_v3_cool_gw=(-0.20, 0.25),
    wb_v3_cast_gw_weight=0.60, wb_v3_cast_conf=0.80, wb_v3_cast_strength=0.80,
    wb_v3_cast_max_tb_spread=0.22, wb_v3_cast_max_zone_dis=0.22, wb_v3_cast_max_bright_share=0.45,
    wb_v3_cast_max_bright_share_no_tb=0.12,  # T4r: sparse neutral evidence cannot supply a half-frame comparison
    # T4m: neither zone has neutral evidence and the combined log2 R/G is strongly negative: confidence at least this
    wb_v3_cyan_conf_check=True, wb_v3_cyan_narrow_max=0.04, wb_v3_cyan_r_max=-0.85, wb_v3_cyan_min_conf=0.72,
    wb_v3_cyan_b_max=0.20, wb_v3_cyan_max_zone_dis=0.60,  # T4r: do not boost conflicting scene colours
    wb_v3_cyan_min_bright_share=0.10,  # T4t: dark monochrome foliage does not support a strong cyan correction
    wb_v3_tint_max_zone_dis=1.25, wb_v3_tint_conflict_damp=0.55,  # T4t: damp positive tint on extreme zone conflicts
    wb_v3_no_narrow_max_tint=0.18,  # T4u: cap positive tint without narrow neutral evidence or a cast boost
    wb_v3_cast_max_temp=None,      # T4k: a cool-cast frame's temp at most this (0.38 measured: _DSF8948 8.17 -> 9.21);
                                   # None = up to wb_max_shift (T4j)
    # T4k: mid-grey zone and grey world both strongly cool (log2 R/G below [0], B/G above [1] of wb_v3_undamp_z1 /
    # wb_v3_undamp_gw) with a top-bottom spread below wb_v3_undamp_tb_max (no sky over ground): no grey-world sky damping
    wb_v3_gw_cool_undamp=True, wb_v3_undamp_z1=(-0.15, 0.15), wb_v3_undamp_gw=(-0.20, 0.25), wb_v3_undamp_tb_max=0.08,
    wb_v3_cool_damp=0.55,          # T4k: a negative temp (towards blue: warm walls, lamp light) is multiplied by this
    wb_v3_tint_boost=1.20,         # T4k: a positive tint (towards magenta, against green) is multiplied by this
    # T4i auto levels on the existing sliders (task list 2.1, scheme A; off by default since T4j, measured in T4i):
    # - slope_r / slope_b: per-channel log2 ratio to G at the auto_slope_pct percentiles (pixels with max > 0.003);
    #   when highlight minus shadow ratio differs by >= auto_slope_min_delta, the channel's gamma is matched to G over
    #   the G log range, at most +-auto_slope_max, added to the preset's slope
    auto_slope=False, auto_slope_pct=(2.0, 98.0), auto_slope_min_delta=0.10, auto_slope_max=0.08,
    # - contrast: encoded luminance p90 - p10 below auto_contrast_spread raises contrast by auto_contrast_gain x the
    #   shortfall, at most auto_contrast_max (the task list gives the cap and threshold; the gain is ours)
    auto_contrast=False, auto_contrast_spread=0.42, auto_contrast_gain=1.3, auto_contrast_max=0.22,
    # - blacks: encoded luminance p2 above auto_blacks_p2 adds -clip((p2 - auto_blacks_offset) x auto_blacks_gain, 0,
    #   auto_blacks_max) to the preset's blacks, within +-auto_blacks_bound
    auto_blacks=False, auto_blacks_p2=0.06, auto_blacks_offset=0.05, auto_blacks_gain=1.1, auto_blacks_max=0.28,
    auto_blacks_bound=0.35,
    wp_invalid_stops=1.5,
    matrix=None,                   # colour matrix rendered with (app: camera_color_matrix of the RAW); None = identity
    camera_matrix_mix=0.40,        # share of the camera matrix the app renders camera RAW with
    chroma_preserve=0.40,          # shoulder & midtone partly on luminance (RenderParams.chroma_preserve; T4e 0.35)
    # Backlit weight (T4f: wider than T4e's 0.78 / 0.32 / 0.06; T4e: bright background and dark foreground only; T4d
    # also fired on a wide p25..p90 gap; T4c was 1 when p90 >= 0.80 and p25 <= 0.30), on the encoded luminance:
    # w = ramp(p90; backlit_p90, backlit_p90 + backlit_ramp) x ramp(backlit_p25 - p25; 0, backlit_ramp_p25)
    backlit_p90=0.72,              # bright background: weight starts at this p90, full at p90 + backlit_ramp
    backlit_p25=0.40,              # dark foreground: starts below this p25, full at p25 - backlit_ramp_p25
    backlit_ramp=0.08, backlit_ramp_p25=0.10,  # T4f task list formula: 0.72..0.80 and 0.40 - p25 in 0..0.10
    # Relative to the preset (T4e; T4d moved to absolute 1.55 / 0.42 / 0.35, equal for every preset at full weight):
    backlit_contrast_step=0.10, backlit_contrast_cap=1.65,  # contrast + w x step, at most cap (T4e step 0.12)
    backlit_shoulder_step=0.15, backlit_shoulder_cap=0.55,  # shoulder + w x step, at most cap
    backlit_midtone_step=0.09, backlit_midtone_cap=0.38,    # midtone towards the one lifting p25 to backlit_p25_target,
                                                            # searched up to min(cap, base midtone + step), x w
    backlit_p25_target=0.34,       # T4f (T4e 0.28)
    # T4o: sparse highlights cap EV while q90 misses the ordinary backlit gate
    backlit_capped_rescue=True, backlit_capped_min_span=1.00,
    backlit_capped_med_band=(0.24, 0.40), backlit_capped_q25_band=(0.08, 0.25), backlit_capped_q90_min=0.45,
    backlit_near_min_span=1.15, backlit_near_q90_band=(0.68, 0.72), backlit_near_q25_band=(0.15, 0.22),
    backlit_rescue_shadows_step=0.16,
)


# Style presets (T4d): overrides of NOREF_V3 for the automatic starting point; the estimate itself (exposure, white
# balance, highlight / shadow hold, backlit weight) runs the same way on each. Values from the T4g task list; they are
# named after looks the task list associates with them (NLP tone profiles, Frontier / Noritsu scans, optical prints),
# not measured against those systems.
STYLE_PRESETS = {
    "standard": dict(label="标准 Standard", cfg=dict()),
    "natural": dict(label="自然 Natural", cfg=dict(
        target_encoded_band=(0.46, 0.53), base_midtone=0.24, base_contrast=1.18, base_shoulder=0.35, base_sat=0.96,
        base_vibrance=0.04, chroma_preserve=0.35, camera_matrix_mix=0.35, base_shadows=0.06, base_highlights=-0.05)),
    "portrait": dict(label="人像 Portrait", cfg=dict(
        target_encoded_band=(0.49, 0.56), base_midtone=0.26, base_contrast=1.16, base_shoulder=0.42, base_sat=0.94,
        base_vibrance=0.05, chroma_preserve=0.38, camera_matrix_mix=0.38, temp_offset=-0.01, tint_offset=0.03,
        base_shadows=0.12, base_highlights=-0.10, base_slope_r=-0.02, base_slope_b=0.02)),
    "vivid": dict(label="鲜明 Vivid", cfg=dict(
        target_encoded_band=(0.45, 0.52), base_midtone=0.23, base_contrast=1.46, base_shoulder=0.28, base_sat=1.14,
        base_vibrance=0.18, chroma_preserve=0.58, camera_matrix_mix=0.58, base_blacks=-0.04,
        base_highlights=-0.05, base_slope_r=0.02, base_slope_b=-0.02, temp_offset=0.02)),
    "print_classic": dict(label="经典印相 Print", cfg=dict(
        target_encoded_band=(0.42, 0.49), base_midtone=0.21, base_contrast=1.40, base_shoulder=0.18, base_sat=1.02,
        base_vibrance=0.04, chroma_preserve=0.35, camera_matrix_mix=0.35, base_blacks=-0.07, temp_offset=0.01)),
}


def preset_config(name: str, cfg: dict | None = None) -> dict:
    """NOREF_V3 (or ``cfg``) with the overrides of style preset ``name`` (ValueError for an unknown name)."""
    if name not in STYLE_PRESETS:
        raise ValueError(f"unknown style preset {name!r}; one of {sorted(STYLE_PRESETS)}")
    return dict(cfg or NOREF_V3, **STYLE_PRESETS[name]["cfg"], preset=name)


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

    def textured(v):
        t = v & (std_D >= thr)
        share = float(t.sum() / max(v.sum(), 1))
        if share < float(cfg.get("min_textured_share", 0.08)):
            t = v & (std_D >= float(np.percentile(std_D[v], 75)))
        return t, share

    tex, tex_share = textured(valid)
    border = None
    if cfg.get("cell_border_filter"):
        # T4k: a cell on the black rebate or a gap between frames is mostly pixels far below white; its edge with the
        # image is textured and would be metered as the frame's shadows (_DSF9239: anchor 1.62 D instead of 1.04 D)
        lit = (Pc.max(axis=-1) > cfg["cell_min_luma_px"]).mean(axis=(1, 3))
        keep = valid & (mean_rgb.max(axis=-1) > cfg["cell_min_luma"]) & (lit >= cfg["cell_bright_px_share"])
        border = dict(dropped=int((valid & ~keep).sum()), used=False)
        if keep.any():
            tex_k, share_k = textured(keep)
            if tex_k.sum() >= cfg["cell_min_textured"]:
                tex, tex_share = tex_k, share_k
                border["used"] = True
    dt = mean_D[tex]
    p2, p10, p25, p50, p75, p90, p98 = (float(x) for x in np.percentile(dt, [2, 10, 25, 50, 75, 90, 98]))
    anchor_d = 0.5 * p50 + 0.25 * (p25 + p75)  # Tukey trimean (T4f; T4 v2-T4e averaged the P5..P95 mean and midrange)
    return dict(
        textured_share=tex_share,
        n_textured=int(tex.sum()),
        border_cells=border,
        anchor_d=anchor_d,
        span_10_90_d=float(p90 - p10),
        p2_bright_d=p2,
        p10_d=p10,
        p25_d=p25,
        p50_d=p50,
        p75_d=p75,
        p90_d=p90,
        p98_dark_d=p98,
        textured_rgb=mean_rgb[tex],
    )


def _metering_mask(P: np.ndarray, mask: np.ndarray | None, cfg: dict) -> tuple[np.ndarray | None, dict | None]:
    """T4l: the estimator's mask without an opaque holder's surroundings (inset film box) and full black strips at its
    edges (rebate, gaps between frames). A dark subject inside the frame (A012's foreground) is not a full row or
    column of near-black pixels, so it stays. Returns (mask, diagnostics); diagnostics None when both are off."""
    if not (cfg.get("meter_holder_box") or cfg.get("meter_trim_strips")):
        return mask, None
    h, w = P.shape[:2]
    m0 = mask if mask is not None else np.ones((h, w), bool)
    m = m0.copy()
    diag = dict(mask_share_before=float(m0.mean()))
    if cfg.get("meter_holder_box"):
        hold = holder_mask(P)
        share = float(hold.mean())
        diag.update(holder_share=share)
        if share >= cfg["meter_holder_min_share"]:
            t, l, bh, bw = film_box(hold)
            dt, dl = int(round(cfg["meter_box_inset"] * bh)), int(round(cfg["meter_box_inset"] * bw))
            box = np.zeros_like(m)
            box[t + dt : t + bh - dt, l + dl : l + bw - dl] = True
            m &= box
            diag.update(film_box=[t, l, bh, bw], box_inset=[dt, dl])
    if cfg.get("meter_trim_strips") and m.any():
        black = P.max(axis=-1) <= cfg["meter_strip_luma"]
        trimmed = dict(top=0, bottom=0, left=0, right=0)
        for side in trimmed:
            while m.any():
                rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
                if side in ("top", "bottom"):
                    i = rows[0] if side == "top" else rows[-1]
                    line, sel = black[i], m[i]
                else:
                    i = cols[0] if side == "left" else cols[-1]
                    line, sel = black[:, i], m[:, i]
                if not sel.any() or float(line[sel].mean()) < cfg["meter_strip_share"]:
                    break
                if side in ("top", "bottom"):
                    m[i] = False
                else:
                    m[:, i] = False
                trimmed[side] += 1
            if trimmed[side]:  # the strip's soft edge
                rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
                k = int(cfg["meter_strip_margin"])
                if side == "top" and len(rows):
                    m[rows[0] : rows[0] + k] = False
                elif side == "bottom" and len(rows):
                    m[max(rows[-1] - k + 1, 0) : rows[-1] + 1] = False
                elif side == "left" and len(cols):
                    m[:, cols[0] : cols[0] + k] = False
                elif side == "right" and len(cols):
                    m[:, max(cols[-1] - k + 1, 0) : cols[-1] + 1] = False
        diag.update(strips=trimmed)
    if m.sum() < cfg["meter_min_share"] * m0.sum():
        diag.update(used=False, reason="too little left")
        return mask, diag
    diag.update(used=True, mask_share_after=float(m.mean()))
    return m, diag


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
    evidence against a cast). T4g expands sparse zones to their lowest-chroma 15% and blends a midtone Huber estimate;
    a clear-sky bright zone gets less weight, and a clear-sky grey-world estimate keeps its red axis (x
    ``wb_v3_gw_sky_red``) with only its blue axis damped (T4h). T4j: when the bright zone, the mid-grey zone and the
    grey world all point to a strong cool cast (``wb_v3_cool_*``: red low, blue high), the frame is cool all over, not
    a sky: no damping, grey-world weight ``wb_v3_cast_gw_weight``, confidence at least ``wb_v3_cast_conf`` and strength
    ``wb_v3_cast_strength`` (T4k: temp at most ``wb_v3_cast_max_temp``). T4k: a strongly cool mid-grey zone and grey
    world without a top-bottom split keep the grey world undamped; a negative temp is multiplied by ``wb_v3_cool_damp``
    and a positive tint by ``wb_v3_tint_boost``. temp / tint remove ``wb_v3_strength`` x confidence of the cast."""
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
    ests, weights, cands, found = [], [], [], []
    sky_r, sky_b = cfg.get("wb_v3_sky", (-np.inf, np.inf))
    for i, ((q_lo, q_hi), radius, weight) in enumerate(cfg["wb_v3_zones"]):
        lo, hi = np.percentile(y, [q_lo, q_hi])
        zone = (y >= lo) & (y <= hi)
        narrow_share = float(np.mean(norm[zone] <= radius)) if zone.any() else 0.0
        radius_eff = radius
        if zone.any() and narrow_share < cfg["wb_v3_adaptive_share"]:
            radius_eff = max(radius, float(np.percentile(norm[zone], cfg["wb_v3_adaptive_pct"])))
        sel = zone & (norm <= radius_eff)
        zd = dict(luma_pct=[q_lo, q_hi], radius=radius, radius_effective=radius_eff, narrow_share=narrow_share,
                  n=int(sel.sum()), share=float(sel.sum() / max(len(y), 1)))
        if sel.sum() >= cfg["wb_v3_min_px"]:
            est = _huber_location(c[sel], cfg["wb_v3_huber"])
            zd.update(chroma=[float(v) for v in est])
            ests.append(est), weights.append(weight), cands.append(c[sel]), found.append(i)
        diag["zones"].append(zd)
    if not ests:
        return dict(temp=0.0, tint=0.0, confidence=0.0, chroma=[0.0, 0.0],
                    diagnostics=dict(diag, reason="no near-neutral zone with enough pixels"))
    lo, hi = np.percentile(y, [15, 88])
    mid = (y >= lo) & (y <= hi)
    est_gw = _huber_location(c[mid], cfg["wb_v3_gw_huber"]) if mid.sum() >= cfg["wb_v3_min_px"] else None

    def cool(v, key):  # log2 R/G below and log2 B/G above the thresholds of cfg[key]
        return bool(v[0] < cfg[key][0] and v[1] > cfg[key][1])

    quads = []
    n = len(px)
    # quadrants are approximated by sample order: _sample keeps row-major order, so halves of the sample are the
    # top and bottom of the frame; left / right are not separable here, so only top vs bottom is checked
    r_quad = max(r for _, r, _ in cfg["wb_v3_zones"])
    for half in (slice(0, n // 2), slice(n // 2, n)):
        ch, yh, nh = c[half], y[half], norm[half]
        lo, hi = np.percentile(yh, [25, 96])
        selh = (yh >= lo) & (yh <= hi) & (nh <= r_quad)
        if selh.sum() >= cfg["wb_v3_min_px"]:
            quads.append(_huber_location(ch[selh], cfg["wb_v3_huber"]))
    if len(quads) == 2:
        diag["top_bottom_spread"] = float(np.linalg.norm(quads[0] - quads[1]))
    if len(ests) == 2:
        diag["zone_disagreement"] = float(np.linalg.norm(ests[0] - ests[1]))
    bright_px_share = float(np.mean(y > cfg.get("bright_luma", 0.45)))
    br_thr = (cfg.get("wb_v3_cast_max_bright_share", 0.45) if "top_bottom_spread" in diag
              else cfg.get("wb_v3_cast_max_bright_share_no_tb", 0.12))
    split_sky = (diag.get("zone_disagreement", 0.0) > cfg.get("wb_v3_cast_max_zone_dis", np.inf)
                 and (diag.get("top_bottom_spread", 0.0) > cfg.get("wb_v3_cast_max_tb_spread", np.inf)
                      or bright_px_share >= br_thr))
    diag.update(split_sky=bool(split_sky), bright_px_share=bright_px_share)
    # T4j: a uniform cool cast (_DSF8948: bright [-0.10, +0.17], mid-grey [-0.22, +0.31], grey world [-0.46, +0.61]);
    # a sky over a warm or green ground has a mid-grey zone that is not cool (A008 [-0.12, -0.08]) and keeps the damping
    cool_cast = bool(cfg.get("wb_v3_cool_cast_check", False) and found == [0, 1] and est_gw is not None
                     and cool(ests[0], "wb_v3_cool_z0") and cool(ests[1], "wb_v3_cool_z1")
                     and cool(est_gw, "wb_v3_cool_gw") and not split_sky)
    diag["cool_cast"] = cool_cast
    for k, i in enumerate(found):
        if i == 0 and not cool_cast and ests[k][0] < sky_r and ests[k][1] > sky_b:
            # T4f: a bright zone with red low and blue high reads as clear sky, not as a cool cast
            weights[k] = weights[k] * float(cfg["wb_v3_sky_weight"])
            diag["zones"][i].update(clear_sky=True)
        diag["zones"][i].update(weight=float(weights[k]))
    w = np.asarray(weights) / np.sum(weights)
    est = (w[:, None] * np.asarray(ests)).sum(0)
    # T4k: mid-grey zone and grey world both strongly cool on a frame without a sky over ground (_DSF8951): the grey
    # world is a cast, not a sky
    gw_undamp = bool(cfg.get("wb_v3_gw_cool_undamp") and est_gw is not None and 1 in found
                     and cool(ests[found.index(1)], "wb_v3_undamp_z1") and cool(est_gw, "wb_v3_undamp_gw")
                     and diag.get("top_bottom_spread", np.inf) < cfg["wb_v3_undamp_tb_max"])
    diag["gw_cool_undamped"] = gw_undamp
    if est_gw is not None:
        sparse = any(z["narrow_share"] < cfg["wb_v3_adaptive_share"] for z in diag["zones"])
        gw_weight = float(cfg["wb_v3_gw_weight"]) if sparse or np.linalg.norm(est_gw) > cfg["wb_v3_gw_threshold"] else 0.0
        if cool_cast:
            gw_weight = float(cfg["wb_v3_cast_gw_weight"])
        diag["grey_world"] = dict(chroma=[float(v) for v in est_gw], weight=gw_weight)
        if not (cool_cast or gw_undamp) and est_gw[0] < sky_r and est_gw[1] > sky_b:
            # T4h: in a clear-sky direction only the blue axis is damped (T4g damped the whole weight), so a cyan cast
            # (red low) is still corrected without pulling the sky yellow
            est_gw = np.array([est_gw[0] * float(cfg["wb_v3_gw_sky_red"]), est_gw[1] * float(cfg["wb_v3_sky_weight"])])
            diag["grey_world"].update(clear_sky=True, chroma_used=[float(v) for v in est_gw])
        est = (1 - gw_weight) * est + gw_weight * est_gw
    allc = np.concatenate(cands)
    mad = float(np.median(np.linalg.norm(allc - est, axis=1)))
    share = float(sum(z["share"] for z in diag["zones"] if "chroma" in z) / len(diag["zones"]))
    conf = _ramp(share, *cfg["wb_v3_share"]) * (1 - 0.5 * _ramp(mad, *cfg["wb_v3_mad"]))
    if diag.get("top_bottom_spread", 0.0) > cfg["wb_conflict"]:
        conf *= cfg["wb_v3_conflict_factor"]
        diag["conflict"] = True
    strength = cfg["wb_v3_strength"]
    if cool_cast:
        conf = max(conf, float(cfg["wb_v3_cast_conf"]))
        strength = cfg["wb_v3_cast_strength"]
    elif (cfg.get("wb_v3_cyan_conf_check") and len(diag["zones"]) == 2
          and all(z["narrow_share"] < cfg["wb_v3_cyan_narrow_max"] for z in diag["zones"])
          and est[0] < cfg["wb_v3_cyan_r_max"]
          and est[1] <= cfg.get("wb_v3_cyan_b_max", 0.20)
          and diag.get("zone_disagreement", 0.0) <= cfg.get("wb_v3_cyan_max_zone_dis", 0.60)
          and bright_px_share >= cfg.get("wb_v3_cyan_min_bright_share", 0.0)):
        conf = max(conf, float(cfg["wb_v3_cyan_min_conf"]))
        diag["cyan_conf_boost"] = True
    temp, tint = wb_from_chroma(strength * conf * est)
    diag["temp_tint_raw"] = [float(temp), float(tint)]
    # T4k: cooling a warm scene (warm wall, lamp light) is damped; a correction against green is strengthened
    if temp < 0:
        temp *= float(cfg.get("wb_v3_cool_damp", 1.0))
    if tint > 0:
        if diag.get("zone_disagreement", 0.0) > cfg.get("wb_v3_tint_max_zone_dis", np.inf):
            tint *= float(cfg.get("wb_v3_tint_conflict_damp", 0.55))
            diag["tint_conflict_damped"] = True
        elif (not cool_cast and not diag.get("cyan_conf_boost")
              and len(diag["zones"]) == 2
              and all(z["narrow_share"] < cfg["wb_v3_cyan_narrow_max"] for z in diag["zones"])
              and tint > float(cfg.get("wb_v3_no_narrow_max_tint", np.inf))):
            tint = float(cfg.get("wb_v3_no_narrow_max_tint", np.inf))
            diag["no_narrow_tint_capped"] = True
        else:
            tint *= float(cfg.get("wb_v3_tint_boost", 1.0))
    if cool_cast and cfg.get("wb_v3_cast_max_temp") is not None:
        temp = min(temp, float(cfg["wb_v3_cast_max_temp"]))  # T4k: T4j pushed cool-cast frames to +0.50
    m = cfg["wb_max_shift"]
    diag.update(share=share, mad=mad)
    return dict(temp=float(np.clip(temp, -m, m)), tint=float(np.clip(tint, -m, m)), confidence=float(conf),
                chroma=[float(v) for v in est], diagnostics=diag)


def _encoded_quantiles(px, p: RenderParams, qs=(2, 50, 98)):
    e, _, _ = render_full(px, p)
    y = e @ LUMA.astype(np.float32)
    return [float(x) for x in np.percentile(y, qs)]


def estimate_params_v3(P, mask=None, cfg=NOREF_V3, roll=None, look=None, *, span_d_g: float | None = None) -> dict:
    """Textured-cell adaptive print estimator (NOREF_V3, T4 v2, inspired by marcinz606/NegPy & negadoctor):

    1. **Textured Cells**: filters grid cells by local density activity ``std(-log10 Y) >= 0.05 D`` so large
       uniform skies, water or borders do not bias metering.
    2. **Auto Density (ev)**: pulls the textured anchor density (T4f: trimean) towards ``anchor_target_density =
       0.75 D`` (18 % mid-grey) with a comfort band, then refines so the encoded median luminance of the textured cells
       (T4f; the full frame before) lands in the target band: brightening under a full-frame p98 ceiling, lowering
       by at most 0.20 EV.
    3. **Contrast** (T4b): starts at the print centre ``base_contrast`` (standard 1.30); lowered (to >= 1.05)
       only when the encoded luminance p90 - p10 exceeds ``contrast_spread_max`` and both ends clip.
    4. **Highlight Hold (highlights & shoulder)**: when bright textured cells ``P2`` approach the white point or
       full-frame clipping exceeds ``clip_budget`` (0.2 %), writes an explicit negative ``highlights`` burn and/or a
       raised ``shoulder``.
    5. **Shadow Hold** (T4b): positive ``shadows`` only when the encoded luminance p25 is below ``shadow_p25_floor``;
       ``blacks`` starts at the preset's constant ``base_blacks`` (0 for standard).
    5b. **Auto levels** (T4i, on the existing sliders; off by default since T4j): ``slope_r`` / ``slope_b`` follow G between the shadow and
       highlight percentiles of each channel; a narrow frame (encoded p90 - p10 < ``auto_contrast_spread``) gets more
       contrast; lifted shadows (encoded p2 > ``auto_blacks_p2``) a negative ``blacks`` on top of the preset's.
    6. **White balance** (T4b, ``estimate_wb_v3``): robust residual cast of bright near-neutral and low-saturation
       mid-grey pixels; the share the evidence does not cover gets ``wb_prior_weight`` x the look (0 since T4d:
       neutral), then the preset's ``temp_offset`` / ``tint_offset``.
    7. **Colour and backlit frames** (T4c-T4e): renders with ``cfg["matrix"]`` (the app passes the RAW's camera
       matrix) and ``chroma_preserve``; a backlit frame (bright background with a dark foreground on the encoded
       output; continuous weight since T4d) gets, in proportion to the weight, a higher midtone, shoulder and contrast,
       each a step above the preset's own value (T4e), instead of more exposure.
    8. **Style presets** (T4d): ``preset_config(name)`` overrides the base values (band, midtone, contrast, shoulder,
       saturation, vibrance, blacks, colour offsets, camera matrix share, chroma_preserve); steps 1-7 are the same.
    9. **Borders and thin negatives** (T4k): cells and pixels on black rebate / gaps between frames are not metered;
       the backlit weight needs a bright share; with ``span_d_g`` (the inversion's G density range, white point minus
       black point) below ``thin_span_max`` the ev is capped and blacks lowered (thin / night negatives,
       whose highlights the inversion already stretched to white). T4m permits a negative cap and suppresses the
       backlit lift below ``thin_suppress_backlit_span``.
    10. **Metering frame and wider lowering** (T4l): the mask is cut to the film box inside an opaque holder (inset) and
       off full black strips at its edges (``_metering_mask``); a frame whose textured cells are bright down to their
       p25 may be lowered by more than ``ev_refine_down_max``; a G density range below ``min_span_d_g`` is left for
       manual review.

    T4b (2026-10-24) replaced the T4 v2 contrast / shadow / paper-black rules, which compared ``-log10`` of the
    scene-linear positive (``_textured_cells``) with print-density thresholds and so lowered contrast to its floor
    and lifted shadows and blacks on almost every frame.
    """
    P = np.asarray(P, np.float32)
    if mask is not None and mask.shape != P.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} != image shape {P.shape[:2]}")
    look = look or load_look(cfg["look"])
    lp = dict(look["params"])
    mask, metering = _metering_mask(P, mask, cfg)
    st = frame_stats(P, mask, cfg)
    out = dict(config=cfg["id"], config_rev=cfg.get("rev"), preset=cfg.get("preset"), look=look["id"],
               groups=["brightness", "wb", "tone", "sat"], stats=st, diagnostics={})
    if metering is not None:
        out["diagnostics"]["metering"] = metering
    reasons = []
    if st["range_stops"] < cfg["low_info_range_stops"]:
        reasons.append(f"low_information: tonal range {st['range_stops']:.2f} stops")
    if cfg.get("min_span_d_g") is not None and span_d_g is not None and span_d_g < cfg["min_span_d_g"]:
        # T4l (A1141, out of focus): the inversion stretched less than this G density range to the white point
        reasons.append(f"low_information: density range {span_d_g:.2f} D < {cfg['min_span_d_g']:.2f} D")
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
        highlights=float(cfg["base_highlights"]), shadows=float(cfg["base_shadows"]), whites=0.0,
        blacks=float(cfg.get("base_blacks", 0.0)), vibrance=0.0,
        slope_r=float(cfg["base_slope_r"]), slope_b=float(cfg["base_slope_b"]),
        matrix=cfg.get("matrix") or IDENTITY, chroma_preserve=float(cfg.get("chroma_preserve", 0.0)),
    )
    if reasons:
        return dict(out, status="manual_review_required", params=RenderParams(**base_p).to_dict(),
                    flags=reasons, confidence=dict(overall=0.0))

    flags = ["white_point_unrepresentative"] if st["above_white"] > 0.05 else []
    px = _sample(P, mask, cfg["n_sample"] // 4, seed=1)
    tc = _textured_cells(P, mask, cfg)
    # T4k: tone quantiles without deep-black rebate / frame gaps; bright share for the backlit and thin checks
    px_tone = px
    if cfg.get("tone_border_filter"):
        lit_px = px[px.max(axis=-1) > cfg["tone_min_luma"]]
        if len(lit_px) >= cfg["wb_v3_min_px"]:
            px_tone = lit_px
    bright_share = float(np.mean((px_tone @ LUMA.astype(np.float32)) > cfg.get("bright_luma", 0.45)))
    out["diagnostics"]["tone_pixels"] = dict(n=int(len(px_tone)), n_all=int(len(px)), bright_share=bright_share)
    out["diagnostics"]["textured_cells"] = {k: v for k, v in tc.items() if k != "textured_rgb"}

    # 1. White balance (T4b): robust bright-neutral + mid-grey residual cast (estimate_wb_v3); the look's prior only
    #    for the share the evidence does not cover
    wb = estimate_wb_v3(px, cfg)
    m_wb = cfg["wb_max_shift"]
    base_p["temp"] = float(np.clip((1 - wb["confidence"]) * base_p["temp"] + wb["temp"], -m_wb, m_wb))
    base_p["tint"] = float(np.clip((1 - wb["confidence"]) * base_p["tint"] + wb["tint"], -m_wb, m_wb))
    # the preset's colour offsets (T4d) come on top of the estimate, inside the slider bounds
    base_p["temp"] = float(np.clip(base_p["temp"] + cfg.get("temp_offset", 0.0), *BOUNDS["temp"]))
    base_p["tint"] = float(np.clip(base_p["tint"] + cfg.get("tint_offset", 0.0) + cfg.get("base_tint", 0.0),
                                   *BOUNDS["tint"]))
    out["diagnostics"]["wb"] = wb
    auto_lv = out["diagnostics"]["auto_levels"] = {}

    # 1b. Auto slopes (T4i): log2 R/G and B/G at the shadow and highlight percentiles of each channel; a channel whose
    #     ratio changes by >= auto_slope_min_delta between them gets the gamma that follows G over G's log range
    #     (white balance gains shift both ends equally, so they do not enter)
    if cfg.get("auto_slope"):
        px_v = px[px.max(-1) > 0.003]
        if len(px_v) >= cfg["wb_v3_min_px"]:
            q_lo, q_hi = (np.maximum(np.percentile(px_v, q, axis=0), 1e-4) for q in cfg["auto_slope_pct"])
            r_lo, r_hi = np.log2(q_lo[[0, 2]] / q_lo[1]), np.log2(q_hi[[0, 2]] / q_hi[1])
            g_range = float(np.log2(q_hi[1] / q_lo[1]))
            auto_lv["slope"] = dict(shadow_log2_rg_bg=r_lo.tolist(), highlight_log2_rg_bg=r_hi.tolist(),
                                    g_log2_range=g_range)
            for k, name in enumerate(("slope_r", "slope_b")):
                delta = float(r_hi[k] - r_lo[k])
                if abs(delta) >= cfg["auto_slope_min_delta"] and g_range > 0.5:
                    s = float(np.clip(np.log2(max(1.0 + delta / g_range, 0.25)) / 0.65,
                                      -cfg["auto_slope_max"], cfg["auto_slope_max"]))
                    base_p[name] = float(np.clip(base_p[name] + s, *EXTENDED_BOUNDS[name]))
                    auto_lv["slope"][name] = s

    # 2. Saturation / vibrance: reduce saturation slightly on already high-chroma frames
    c_all = _chroma(px[(px @ LUMA.astype(np.float32)) > cfg["min_luma"]])
    p90 = float(np.percentile(np.linalg.norm(c_all, axis=1), 90)) if len(c_all) else 0.3
    if p90 > cfg["sat_colourful"]:
        base_p["sat"] = float(1.0 + (cfg["base_sat"] - 1.0) * (cfg["sat_colourful"] / p90))
    else:
        base_p["vibrance"] = float(cfg.get("base_vibrance", 0.08))
    out["diagnostics"]["chroma_p90"] = p90

    # 3. Contrast starts at the print centre base_contrast (T4b); lowered after exposure only for very wide frames
    c_lo, c_hi = cfg["contrast_bounds_v3"]
    base_p["contrast"] = float(np.clip(cfg["base_contrast"], c_lo, max(c_hi, cfg["base_contrast"])))

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

    # Refine EV (T4f) on the encoded median of the textured cells (cell-mean RGB), not of the full frame, whose median
    # sits in the sky on frames with a large sky (T4c-T4e pulled A013 down to -1.08 EV that way):
    # - textured median below the band: brighten to t_lo + 0.03, while full-frame encoded p98 stays <= ev_refine_p98_max;
    # - above the band (high-key frames: above 0.76): lower towards t_hi - 0.02 by at most ev_refine_down_max, and
    #   not below -ev_refine_down_max.
    # With ev_anchor_p98_cap a positive anchor ev is first held to the same ceiling (not below 0): on backlit frames
    # (A012, A023) the dark textured foreground alone gives the anchor +2.50 EV, which burns the sky.
    t_lo, t_hi = cfg["target_encoded_band"]
    tex_px = tc["textured_rgb"]
    p98_max = float(cfg["ev_refine_p98_max"])

    def bisect_ev(lo_ev, hi_ev, below):  # largest ev in [lo_ev, hi_ev] with below(ev) true; below falls with ev
        for _ in range(18):
            mid_ev = 0.5 * (lo_ev + hi_ev)
            lo_ev, hi_ev = (mid_ev, hi_ev) if below(mid_ev) else (lo_ev, mid_ev)
        return float(lo_ev)

    def p98_ok(ev):
        return _encoded_quantiles(px_tone, RenderParams(**dict(base_p, ev=ev)), qs=(98,))[0] <= p98_max

    refine = dict(ev_anchor_raw=base_p["ev"])
    if cfg.get("ev_anchor_p98_cap") and base_p["ev"] > 0 and not p98_ok(base_p["ev"]):
        base_p["ev"] = bisect_ev(0.0, base_p["ev"], p98_ok)
        refine.update(anchor_capped_by_p98=True)
    ev0 = base_p["ev"]
    med_tex = _encoded_quantiles(tex_px, RenderParams(**base_p), qs=(50,))[0]
    t_hi_eff = 0.76 if anc_d < 0.65 else t_hi
    refine.update(ev_anchor=ev0, med_tex_before=med_tex)

    if med_tex < t_lo:
        tgt = t_lo + 0.03
        ev_up = bisect_ev(ev0, cfg["ev_limits"][1],
                          lambda ev: _encoded_quantiles(tex_px, RenderParams(**dict(base_p, ev=ev)), qs=(50,))[0] < tgt)
        if not p98_ok(ev_up):
            ev_up = bisect_ev(ev0, ev_up, p98_ok)
            refine.update(capped_by_p98=True)
        base_p["ev"] = float(max(ev0, ev_up))
    elif med_tex > t_hi_eff:
        tgt, dmax = t_hi_eff - 0.02, float(cfg["ev_refine_down_max"])
        if cfg.get("ev_down_adaptive"):
            # T4l: textured shadows bright too (encoded p25 high) is a frame bright all over, not a sky over dark ground
            q25_tex = _encoded_quantiles(tex_px, RenderParams(**base_p), qs=(25,))[0]
            dmax += float(cfg["ev_down_extra"]) * _ramp(q25_tex, *cfg["ev_down_q25"])
            refine.update(q25_tex=q25_tex, down_max=dmax)
        floor = min(ev0, max(ev0 - dmax, -dmax))
        base_p["ev"] = bisect_ev(floor, ev0,
                                 lambda ev: _encoded_quantiles(tex_px, RenderParams(**dict(base_p, ev=ev)), qs=(50,))[0] <= tgt)
    base_p["ev"] = float(np.clip(base_p["ev"], *cfg["ev_limits"]))
    refine.update(ev=base_p["ev"], med_tex_after=_encoded_quantiles(tex_px, RenderParams(**base_p), qs=(50,))[0])
    out["diagnostics"]["ev_refine"] = refine

    # 4a. Thin negative: the inversion stretched a short G density range (night, underexposed) to the white point
    #     already; T4m lets the cap go negative to darken very thin negatives
    thin = {}
    if cfg.get("thin_neg_check") and span_d_g is not None:
        thin = dict(span_d_g=float(span_d_g), bright_share=bright_share, applied=False)
        if span_d_g < cfg["thin_span_max"] and bright_share < cfg["thin_bright_share_max"]:
            cap = float(np.clip((span_d_g - cfg["thin_span_zero"]) / cfg["thin_span_ramp"] * cfg["thin_ev_max"],
                                cfg.get("thin_ev_min", 0.0), cfg["thin_ev_max"]))
            thin.update(applied=True, ev_cap=float(cap), ev_before=base_p["ev"])
            base_p["ev"] = float(min(base_p["ev"], cap))
            if span_d_g < cfg["thin_blacks_span"]:
                lower = float(np.clip((cfg["thin_blacks_span"] - span_d_g) * cfg["thin_blacks_gain"], 0.0,
                                      cfg["thin_blacks_max"]))
                base_p["blacks"] = float(np.clip(base_p["blacks"] - lower, *EXTENDED_BOUNDS["blacks"]))
                thin.update(blacks_lowered=lower)
        out["diagnostics"]["thin_negative"] = thin

    # 4b. Backlit midtone (continuous weight w in [0, 1] since T4d; T4e: bright background and dark foreground only,
    #     and relative to the preset): contrast and shoulder get w x their step above the preset's own values (capped),
    #     the midtone moves w of the way to the midtone (bisection up to min(cap, preset midtone + step)) that lifts p25
    #     to backlit_p25_target, instead of more ev pushing the sky into white
    q25, q90 = _encoded_quantiles(px_tone, RenderParams(**base_p), qs=(25, 90))
    rmp = float(cfg["backlit_ramp"])
    rmp25 = float(cfg.get("backlit_ramp_p25", rmp))
    w_bl = _ramp(q90, cfg["backlit_p90"], cfg["backlit_p90"] + rmp) * _ramp(cfg["backlit_p25"] - q25, 0.0, rmp25)
    out["diagnostics"]["backlit_check"] = dict(p25=q25, p90=q90, weight=w_bl)
    if cfg.get("backlit_sky_check") and w_bl > 0:
        # T4k: a dark foreground needs a bright background to be backlit (_DSF9239: black frame gaps alone gave p25)
        w_bl *= _ramp(bright_share, cfg["backlit_sky_share_lo"], cfg["backlit_min_sky_share"])
        out["diagnostics"]["backlit_check"].update(weight=w_bl, bright_share=bright_share)
    if thin.get("applied") and span_d_g < cfg.get("thin_suppress_backlit_span", 0.58):
        out["diagnostics"]["backlit_check"].update(thin_suppressed=True, weight_before_thin=w_bl, weight=0.0)
        w_bl = 0.0
    backlit = w_bl > 0
    if backlit:
        base_p["shoulder"] = min(float(cfg["backlit_shoulder_cap"]),
                                 base_p["shoulder"] + w_bl * float(cfg["backlit_shoulder_step"]))
        if cfg["base_contrast"] >= 1.25:
            base_p["contrast"] = min(float(cfg["backlit_contrast_cap"]),
                                     base_p["contrast"] + w_bl * float(cfg["backlit_contrast_step"]))
        lo_m = m0 = base_p["midtone"]
        hi_m = max(m0, min(float(cfg["backlit_midtone_cap"]), m0 + float(cfg["backlit_midtone_step"])))
        if _encoded_quantiles(px_tone, RenderParams(**dict(base_p, midtone=hi_m)), qs=(25,))[0] <= cfg["backlit_p25_target"]:
            m_t = hi_m
        else:
            for _ in range(14):  # p25 increases with midtone
                mid_m = 0.5 * (lo_m + hi_m)
                if _encoded_quantiles(px_tone, RenderParams(**dict(base_p, midtone=mid_m)), qs=(25,))[0] < cfg["backlit_p25_target"]:
                    lo_m = mid_m
                else:
                    hi_m = mid_m
            m_t = float(hi_m)
        base_p["midtone"] = float(m0 + w_bl * (m_t - m0))
        out["diagnostics"]["backlit_check"].update(midtone_target=m_t)
        flags.append("backlit_midtone")

    # 5. NegPy Highlight Hold: check bright textured cells P2 after EV shift, plus full-frame clip_budget
    p2_shifted_d = tc["p2_bright_d"] - 0.30103 * base_p["ev"]
    if p2_shifted_d < 0.12:
        # Automatic highlight burn written directly to the highlights slider
        base_p["highlights"] = min(base_p["highlights"], float(np.clip(-1.8 * (0.12 - p2_shifted_d), -0.65, 0.0)))

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
        max_drop = float(cfg.get("ev_highlight_max_drop", 3.0))
        for _ in range(30):
            if (_clip_share(px, RenderParams(**base_p)) <= cfg["clip_budget"]
                    or base_p["ev"] <= cfg["ev_limits"][0] or ev_lowered + 1e-9 >= max_drop):
                break
            step = min(0.1, max(0.0, max_drop - ev_lowered))
            if step <= 1e-9:
                break
            base_p["ev"] = float(max(base_p["ev"] - step, cfg["ev_limits"][0]))
            ev_lowered = round(ev_lowered + step, 4)
    out["diagnostics"]["highlight_hold"] = dict(ev_lowered=ev_lowered, max_drop=cfg.get("ev_highlight_max_drop", 3.0))
    if ev_lowered > 0:
        flags.append(f"ev_lowered_for_highlights:{ev_lowered:.1f}")

    # T4o: sparse highlights can cap EV without reaching the ordinary q90 backlit threshold.
    if w_bl == 0.0 and cfg.get("backlit_capped_rescue") and span_d_g is not None:
        med_after = (min(float(refine["med_tex_after"]),
                         _encoded_quantiles(tex_px, RenderParams(**base_p), qs=(50,))[0])
                     if ev_lowered > 0 else float(refine["med_tex_after"]))
        q25_cur, q90_cur = _encoded_quantiles(px_tone, RenderParams(**base_p), qs=(25, 90))
        q25_res = q25_cur if ev_lowered > 0 else float(q25)
        q90_res = max(float(q90), q90_cur)
        hl_capped = bool(refine.get("capped_by_p98") or refine.get("anchor_capped_by_p98") or ev_lowered > 0)
        med_lo, med_hi = cfg["backlit_capped_med_band"]
        q25_lo, q25_hi = cfg["backlit_capped_q25_band"]
        near_lo, near_hi = cfg["backlit_near_q90_band"]
        near_q25_lo, near_q25_hi = cfg["backlit_near_q25_band"]
        c1 = (span_d_g >= cfg["backlit_capped_min_span"] and hl_capped and med_lo <= med_after < med_hi
              and q25_lo <= q25_res <= q25_hi and q90_res >= cfg["backlit_capped_q90_min"])
        c2 = (span_d_g >= cfg["backlit_near_min_span"] and near_lo <= q90_res < near_hi
              and near_q25_lo <= q25_res <= near_q25_hi)
        if c1 or c2:
            before_rescue = dict(base_p)
            w_res = min(0.85, max((q90_res - 0.44) / 0.16, 0.40)) * min(1.0, (0.26 - q25_res) / 0.08)
            base_p["midtone"] = float(min(cfg["backlit_midtone_cap"], base_p["midtone"] + w_res * 0.08))
            base_p["shoulder"] = float(min(cfg["backlit_shoulder_cap"], base_p["shoulder"] + w_res * 0.12))
            base_p["shadows"] = float(max(base_p["shadows"], w_res * float(cfg["backlit_rescue_shadows_step"])))
            # Recheck Highlight Hold after lifting; stronger shoulder first, otherwise retain the pre-rescue result.
            if _clip_share(px, RenderParams(**base_p)) > cfg["clip_budget"]:
                for shoulder in np.linspace(base_p["shoulder"], 0.85, 15):
                    base_p["shoulder"] = float(shoulder)
                    if _clip_share(px, RenderParams(**base_p)) <= cfg["clip_budget"]:
                        break
                else:
                    base_p = before_rescue
                    w_res = 0.0
            out["diagnostics"]["backlit_check"].update(rescue_weight=float(w_res), rescue_c1=bool(c1), rescue_c2=bool(c2),
                                                       rescue_med_before=med_after, rescue_q25=q25_res, rescue_q90=q90_res)
            if w_res > 0:
                flags.append("backlit_midtone")

    # T4q: use Step 4b's shadow evidence after the backlit / highlight decisions, so their inputs stay unchanged
    ol_wide_bright = (span_d_g is not None and span_d_g >= cfg.get("overlift_max_span", 1.15)
                      and bright_share >= cfg.get("overlift_max_bright_share", 0.05))
    if (cfg.get("overlift_damp_check") and not ol_wide_bright and base_p["ev"] >= cfg["overlift_ev_min"]
            and q25 >= cfg["overlift_q25_min"]):
        ev0_ol = float(base_p["ev"])
        ramp_ev = float(np.clip((ev0_ol - cfg["overlift_ev_pivot"]) / cfg["overlift_ev_ramp"], 0.0, 1.0))
        pull = float(np.clip(cfg["overlift_gain"] * (q25 - cfg["overlift_q25_pivot"]) * ramp_ev,
                             0.0, cfg["overlift_max_pull"]))
        if pull > 0.0:
            blk_drop = float(min(cfg["overlift_blacks_max"], pull * cfg["overlift_blacks_gain"]))
            base_p["ev"] = float(max(cfg["ev_limits"][0], ev0_ol - pull))
            base_p["blacks"] = float(np.clip(base_p["blacks"] - blk_drop, *EXTENDED_BOUNDS["blacks"]))
            out["diagnostics"]["overlift_damp"] = dict(
                applied=True, ev_before=ev0_ol, ev_after=base_p["ev"], pull=pull,
                q25_before=float(q25), blacks_lowered=blk_drop)

    # 6. Contrast relief (T4b, encoded output): only a very wide frame (encoded luminance p90 - p10 above
    #    contrast_spread_max) with both ends clipped gets a lower contrast, down to the lower bound
    def ends(p):
        e, _, _ = render_full(px, RenderParams(**p))
        y = e @ LUMA.astype(np.float32)
        q10, q90 = np.percentile(y, [10, 90])
        return float(q90 - q10), float(np.mean(y <= 0.02)), float(np.mean(y >= 0.98))

    spread, black_end, white_end = ends(base_p)
    out["diagnostics"]["contrast_check"] = dict(spread_10_90=spread, black_end=black_end, white_end=white_end)
    if not backlit and spread > cfg["contrast_spread_max"] and min(black_end, white_end) > cfg["end_clip_share"]:
        lo_c, hi_c = c_lo, base_p["contrast"]
        if ends(dict(base_p, contrast=lo_c))[0] > cfg["contrast_spread_max"]:
            base_p["contrast"] = lo_c
        else:
            for _ in range(14):  # bisection: spread increases with contrast
                mid_c = 0.5 * (lo_c + hi_c)
                lo_c, hi_c = (mid_c, hi_c) if ends(dict(base_p, contrast=mid_c))[0] <= cfg["contrast_spread_max"] else (lo_c, mid_c)
            base_p["contrast"] = float(lo_c)
        flags.append("contrast_lowered_wide_frame")
    # 6b. Auto contrast (T4i): a narrow frame (encoded luminance p90 - p10 below auto_contrast_spread) gets more
    #     contrast in proportion to the shortfall, at most auto_contrast_max
    if cfg.get("auto_contrast") and spread < cfg["auto_contrast_spread"]:
        inc = float(min(cfg["auto_contrast_max"], cfg["auto_contrast_gain"] * (cfg["auto_contrast_spread"] - spread)))
        base_p["contrast"] = float(min(BOUNDS["contrast"][1], base_p["contrast"] + inc))
        auto_lv["contrast"] = inc

    # 7. Shadow Hold (T4b, encoded output): shadows lifted only on crushed frames (encoded luminance p25 below
    #    shadow_p25_floor); a positive blacks (paper-black / matte) stays a user choice
    p2, p25, p50, p98 = _encoded_quantiles(px, RenderParams(**base_p), qs=(2, 25, 50, 98))
    out["diagnostics"]["encoded_p25_before_shadows"] = p25
    if p25 < cfg["shadow_p25_floor"]:
        base_p["shadows"] = max(base_p["shadows"], float(np.clip(
            cfg["shadow_gain"] * (cfg["shadow_p25_floor"] - p25), 0.0, cfg["shadow_max"])))
    # 7b. Auto blacks (T4i): darkest 2 % lifted off black (encoded luminance p2 above auto_blacks_p2) get a negative
    #     blacks on top of the preset's
    if cfg.get("auto_blacks"):
        p2_b = _encoded_quantiles(px, RenderParams(**base_p), qs=(2,))[0]
        if p2_b > cfg["auto_blacks_p2"]:
            ab = -float(np.clip((p2_b - cfg["auto_blacks_offset"]) * cfg["auto_blacks_gain"], 0.0,
                                cfg["auto_blacks_max"]))
            bound = float(cfg["auto_blacks_bound"])
            base_p["blacks"] = float(np.clip(base_p["blacks"] + ab, -bound, bound))
            auto_lv["blacks"] = dict(p2=p2_b, blacks=ab)

    # Round slider-visible values cleanly
    for k in ("ev", "temp", "tint", "sat", "contrast", "midtone", "shoulder", "highlights", "shadows", "whites", "blacks",
              "vibrance", "slope_r", "slope_b"):
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
