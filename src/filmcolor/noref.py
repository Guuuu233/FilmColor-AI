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
from .render import (BOUNDS, EXTENDED_BOUNDS, IDENTITY, LUMA, LUMA_F32, RenderParams, RenderParamsT19,
                     T19_BOUNDS, ZERO_MATRIX, _all_finite, _max3, pre_shoulder_linear,
                     render_fast_full, render_t19, shoulder_curve)

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
    arr = np.asarray(P, np.float32)
    flat = arr.reshape(-1, 3)
    if mask is None or bool(mask.all()):
        if len(flat) > n:
            return flat[np.sort(np.random.default_rng(seed).choice(len(flat), n, replace=False))]
        return flat
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return np.empty((0, 3), dtype=np.float32)
    if len(idx) > n:
        idx = idx[np.sort(np.random.default_rng(seed).choice(len(idx), n, replace=False))]
    return flat[idx]


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
    y = px @ LUMA_F32
    lo, hi = np.percentile(y, cfg["wb_luma_pct"])
    mid = (y >= max(lo, cfg["min_luma"])) & (y <= hi)
    c = _chroma(px[mid])
    cand = np.linalg.norm(c, axis=1) <= cfg["wb_radius"]
    return c[cand], int(mid.sum())


def frame_stats(P, mask=None, cfg=NOREF_V1) -> dict:
    """Per-frame statistics, also used as roll context for other frames."""
    px = _sample(P, mask, cfg["n_sample"])
    if len(px) == 0:
        raise ValueError("too few pixels above black for statistics")
    if not _all_finite(px):
        raise ValueError("positive contains NaN or infinite values")
    y = px @ LUMA_F32
    yv = y[y > cfg["min_luma"]]
    if len(yv) < 100:
        raise ValueError("too few pixels above black for statistics")
    p1, p50, p99, p995 = (float(x) for x in np.percentile(yv, [1, 50, 99, 99.5]))
    cand, n_mid = _wb_candidates(px, cfg)
    est = np.median(cand, 0) if len(cand) else np.zeros(2)
    mad = float(np.median(np.linalg.norm(cand - est, axis=1))) if len(cand) else None
    return dict(median=p50, p1=p1, p99=p99, p995=p995, range_stops=float(np.log2(p99 / max(p1, cfg["min_luma"]))),
                above_white=float(np.mean(_max3(px) > 1)), dark_share=float(np.mean(y <= cfg["min_luma"])),
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
    x = np.asarray(px, np.float32)
    if not _all_finite(x):
        raise ValueError("input contains NaN or infinite values")
    x_lin, _ = pre_shoulder_linear(x, p, fast_pow=True)
    eff_s = float(np.clip(p.shoulder + max(0.0, -0.45 * p.highlights) + max(0.0, -0.25 * p.whites), 0.0, 1.0))
    return float(np.mean(shoulder_curve(_max3(x_lin), eff_s) > 1))


_ORIG_CLIP_SHARE = _clip_share


def _encoded_spread(px, p: RenderParams):
    e, _, _, _ = render_fast_full(px, p)
    y = e @ LUMA_F32
    lo, hi = np.quantile(y, [0.02, 0.98])
    return float(hi - lo)


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
        c = _chroma(px[(px @ LUMA_F32) > cfg["min_luma"]])
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
    arr = np.asarray(P, np.float32)
    h, w = (arr.shape[0] // f) * f, (arr.shape[1] // f) * f
    oh, ow = h // f, w // f
    P4 = np.empty((oh, ow, 3), dtype=np.float32)
    step = f * 64
    for r0 in range(0, h, step):
        r1 = min(h, r0 + step)
        b0, b1 = r0 // f, r1 // f
        P4[b0:b1] = arr[r0:r1, :w].reshape(b1 - b0, f, ow, f, 3).mean(axis=(1, 3))
    t, l, rh, rw = (int(x) // f for x in image_region)
    mask = np.zeros((oh, ow), bool)
    mask[t : t + rh, l : l + rw] = True
    return P4, mask


# --- noref_v3: NegPy (marcinz606/NegPy) + Darktable negadoctor inspired textured-cell auto estimator (T4 v2) ---

NOREF_V3 = dict(
    NOREF_V1,
    id="noref_v3",
    rev="t4z-2026-11-05",
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
    # T5i: tighten thin-negative EV cap when extreme positive anchor EV lifts dark underexposed negatives too high
    thin_neg_high_anc_check=True, thin_high_anc_min=1.50, thin_high_anc_pivot=1.30,
    thin_high_anc_med0_max=0.46, thin_high_anc_q25_min=0.08,
    thin_high_anc_ev_gain=0.35, thin_high_anc_max_pull=0.40, thin_high_anc_blacks_gain=0.14,
    # T4x: stretched high-q25 floors that bypass the ordinary thin-negative bright-share gate
    thin_flat_check=True, thin_flat_span_max=0.72, thin_flat_q25_min=0.62,
    thin_flat_iqr_max=0.20, thin_flat_iqr_pivot=0.20, thin_flat_ev_min=-0.65,
    thin_flat_ev_gain=5.0, thin_flat_blacks_max=0.18,
    # T4x: dense overexposed plateaus with enough full-frame range to separate them from flat clouds
    ultra_flat_check=True, ultra_flat_min_range_stops=1.20, ultra_flat_q25_min=0.72,
    ultra_flat_iqr_max=0.068, ultra_flat_iqr_pivot=0.068, ultra_flat_ev_gain=18.0,
    ultra_flat_max_pull=0.55, ultra_flat_blacks_gain=0.55, ultra_flat_blacks_max=0.30,
    ultra_flat_contrast_gain=0.85, ultra_flat_contrast_cap=1.75,
    # T5d: flare / fog pedestal lifting the darkest textured cells above mid-grey (A1125.TIF: p98_dark_d = 0.514 D)
    lifted_floor_check=True, lifted_floor_span_max=0.84, lifted_floor_ev_max=-0.22, lifted_floor_p98_dark_d_max=0.56,
    lifted_floor_p98_dark_d_pivot=0.58, lifted_floor_ev_gain=11.0, lifted_floor_max_pull=0.70,
    lifted_floor_blacks_gain=0.45, lifted_floor_blacks_max=0.30,
    lifted_floor_contrast_gain=0.65, lifted_floor_contrast_cap=1.75,
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
    overlift_short_span_max=0.77, overlift_short_ev_pivot=1.10, overlift_short_ev_ramp=0.45,
    overlift_short_q25_pivot=0.25, overlift_short_gain=4.2, overlift_short_max_pull=0.95,
    overlift_bl_coupled_min_w=0.20, overlift_bl_coupled_ev_pivot=1.10, overlift_bl_coupled_ev_ramp=0.45,
    overlift_bl_coupled_q25_pivot=0.22, overlift_bl_coupled_gain=3.8,
    overlift_blacks_gain=0.10, overlift_blacks_max=0.06,
    overlift_max_span=1.15, overlift_max_bright_share=0.05,  # T4r: exempt wide-range bright scenes
    # T4w / T5i: moderate EV on short-range frames with lifted shadows, allowing for the preset's shadow lift
    overlift_mod_check=True, overlift_mod_ev_min=0.66, overlift_mod_ev_pivot=0.52, overlift_mod_ev_ramp=0.45,
    overlift_mod_q25_min=0.40, overlift_mod_q25_pivot=0.37,
    overlift_mod_color_q25_min=0.365, overlift_mod_color_q25_pivot=0.35,
    overlift_mod_max_bright_share=0.20,
    overlift_mod_gain=5.2, overlift_mod_max_pull=0.62, overlift_mod_shadow_comp=0.35,
    overlift_mod_hi_ev_min=1.35, overlift_mod_hi_q25_min=0.40, overlift_mod_hi_ev_pivot=0.58, overlift_mod_hi_ev_ramp=0.55,
    overlift_mod_hi_q25_pivot=0.38, overlift_mod_hi_gain=4.0, overlift_mod_hi_max_pull=0.45,
    # T4y / T5i: short-span bright scenes with shadows already lifted under positive EV (including p98-capped anchors)
    overlift_bright_check=True, overlift_bright_span_max=0.76, overlift_bright_min_share=0.20,
    overlift_bright_ev_min=0.45, overlift_bright_ev_pivot=0.30, overlift_bright_ev_ramp=0.45,
    overlift_bright_anc_max=1.40, overlift_bright_med0_min=0.46,
    overlift_bright_q25_band=(0.475, 0.68), overlift_bright_q25_pivot=0.45,
    overlift_bright_gain=3.2, overlift_bright_max_pull=0.48,
    overlift_bright_cap_check=True, overlift_bright_min_anc_drop=0.115, overlift_bright_cap_med0_min=0.41,
    overlift_bright_cap_q25_min=0.36, overlift_bright_cap_q25_pivot=0.34,
    # T4z: damp simultaneous backlit midtone and high-EV lifts in sparse-light night scenes
    overlift_night_bl_check=True, overlift_night_bl_min_weight=0.65,
    overlift_night_bl_max_bright_share=0.090,
    overlift_night_bl_ev_min=1.50, overlift_night_bl_ev_pivot=1.35, overlift_night_bl_ev_ramp=0.55,
    overlift_night_bl_q25_min=0.175, overlift_night_bl_q25_pivot=0.155,
    overlift_night_bl_hi_w_min=0.95, overlift_night_bl_hi_ev_min=1.60,
    overlift_night_bl_hi_q25_min=0.08, overlift_night_bl_hi_q25_pivot=0.06,
    overlift_night_bl_gain=3.6, overlift_night_bl_max_pull=0.55,
    # T5i: damp mid-EV low-bright scenes where strong backlit lift and positive EV double-lift already-bright textures
    overlift_mid_bl_check=True, overlift_mid_bl_min_weight=0.85,
    overlift_mid_bl_max_bright_share=0.090,
    overlift_mid_bl_ev_min=1.20, overlift_mid_bl_ev_pivot=1.05, overlift_mid_bl_ev_ramp=0.45,
    overlift_mid_bl_med0_min=0.50, overlift_mid_bl_q25_min=0.18, overlift_mid_bl_q25_pivot=0.15,
    overlift_mid_bl_gain=3.6, overlift_mid_bl_max_pull=0.45,
    # T5i: damp short-span backlit double-lift when positive anchor EV and backlit midtone fire simultaneously
    overlift_short_bl_check=True, overlift_short_bl_span_max=0.85, overlift_short_bl_min_weight=0.50,
    overlift_short_bl_ev_min=0.48, overlift_short_bl_anc_min=1.05, overlift_short_bl_med0_max=0.55,
    overlift_short_bl_q25_min=0.12, overlift_short_bl_ev_pivot=0.30, overlift_short_bl_ev_ramp=0.40,
    overlift_short_bl_span_pivot=0.90, overlift_short_bl_span_ramp=0.22,
    overlift_short_bl_gain=0.95, overlift_short_bl_max_pull=0.80, overlift_short_bl_mid_damp=0.65,
    # T5i: damp medium-span backlit double-lift on color negatives with high anchor EV and dark initial textures
    overlift_med_span_bl_check=True, overlift_med_span_bl_span_min=0.85, overlift_med_span_bl_span_max=1.10,
    overlift_med_span_bl_min_bright_share=0.10, overlift_med_span_bl_max_bright_share=0.35,
    overlift_med_span_bl_min_weight=0.78,
    overlift_med_span_bl_ev_min=0.70, overlift_med_span_bl_ev_max=1.75,
    overlift_med_span_bl_anc_min=1.45, overlift_med_span_bl_med0_max=0.49, overlift_med_span_bl_q25_min=0.16,
    overlift_med_span_bl_ev_pivot=0.60, overlift_med_span_bl_ev_ramp=0.45,
    overlift_med_span_bl_gain=0.40, overlift_med_span_bl_max_pull=0.36, overlift_med_span_bl_mid_damp=0.45,
    # T5i: anchor milky shadow floor on short-span bright color negatives (NegPy print toe / fade floor offset)
    color_lifted_floor_check=True, color_lifted_floor_span_max=0.79, color_lifted_floor_min_bright_share=0.20,
    color_lifted_floor_ev_max=0.10, color_lifted_floor_min_p98_dark_d=0.90,
    color_lifted_floor_short_span=0.56, color_lifted_floor_min_anchor_d=0.45,
    color_lifted_floor_p2_min=0.29, color_lifted_floor_p2_pivot=0.26, color_lifted_floor_shadow_comp=0.30,
    color_lifted_floor_ev_gain=2.20, color_lifted_floor_ev_max_drop=0.32,
    color_lifted_floor_blacks_gain=1.40, color_lifted_floor_blacks_max=0.22,
    color_lifted_floor_contrast_gain=1.00, color_lifted_floor_contrast_max=0.18,
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
    # T5i: damp cool-cast temp boost when bright neutral evidence is abundant and mid-grey is not strongly blue
    wb_bright_cool_damp_check=True, wb_bright_cool_min_br=0.20, wb_bright_cool_min_ns0=0.15,
    wb_bright_cool_max_z1_b=0.22, wb_bright_cool_min_gw_r=-0.80, wb_bright_cool_factor=0.55,
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
    # T5j (C11..C16): Darkroom print S-curve, NegPy Print-Toe BPC, Boyack-Juenger low-key damping,
    # residual overlift convergence, multi-zone WB crossover, and chroma-damped C-41 dye crosstalk unmixing
    t5j_wb_crossover_check=True,
    t5j_dye_unmix_check=True, t5j_dye_unmix_max_cp90=0.73,
    t5j_boyack_lowkey_check=True,
    t5j_overlift_conv_check=True,
    t5j_paper_white_check=True,
    t5j_print_toe_bpc_check=True,
    # T5k (C17..C20): Experimental follow-on hooks (disabled by default to preserve strict zero-regression T5j)
    t5k_bpc_check=True,
    t5k_soft_clip_check=True,
    t5k_dye_unmix_v2_check=True,
    t5k_blue_sat_wb_check=True,
    t5k_uncal_fog_check=True,
    # T6: Adaptive print contrast, paper-white release, shadow-toe anchoring, and uncalibrated overlift damping
    t6_print_contrast_check=True,
    t6_paper_white_release_check=True,
    t6_shadow_toe_check=True,
    t6_uncal_overlift_check=True,
    # T6 v2: Calibrated linear-light BPC + shoulder release + uncalibrated C-41 print/unmix/WB model
    t6_v2_cal_bpc_shoulder_check=True,
    t6_v2_uncal_print_check=True,
    # T6 v3: Calibrated C-41 unmix/cross50 + bright-gate/contrast-1.65 + uncalibrated midtone/pull/contrast-1.70
    t6_v3_cal_refine_check=True,
    t6_v3_uncal_refine_check=True,
    # T6 v4: Smooth span-aware calibrated midtone/EV/sat + uncalibrated contrast-1.40/sat-0.86
    t6_v4_cal_refine_check=True,
    t6_v4_uncal_refine_check=True,
    # T7: Orthogonal Softplus H&D curve, dual-zone crossover, highlight/dark-frame guards, skin protection
    t7_hl_dark_guard_check=True,
    # T8: Shadow-toe / contrast decoupling, uncalibrated dark midtone guard, calibrated chroma guard, uncal BPC guard
    t8_uncal_mid_guard_check=True,
    t8_cal_chroma_guard_check=True,
    t8_uncal_bpc_guard_check=True,
    # T9: Calibrated deep-blacks EV release, uncalibrated dark-scene EV damping, uncalibrated WB centering
    t9_cal_deep_blk_ev_check=True,
    t9_uncal_dark_ev_check=True,
    t9_uncal_wb_center_check=True,
    # T10: High-key p2 guard, uncal sat restore
    t10_cal_deep_blk_p2_guard_check=True,
    t10_cal_highkey_guard_check=True,
    # T11: Calibrated high-med/contrast/midtone/BPC refinements
    t11_cal_bpc_damp_check=True,
    # T12: Calibrated deep-blacks EV floor, uncalibrated WB v2, uncalibrated adaptive sat
    t12_uncal_span_wb_check=True,
    t12_uncal_span_ev_check=True,
    t12_uncal_span_sat_check=True,
    t12_uncal_lowkey_guard_check=True,
    # T13: Configurable uncal print/WB/sat/tone & cal WB/EV hooks (all off/neutral by default)
    t13_uncal_cnt_cap=1.48,
    t13_uncal_cnt_p2_min=0.08,
    t13_uncal_cnt_spread_max=0.68,
    t13_uncal_paper_white_check=True,
    t13_uncal_paper_white_thr=0.90,
    t13_uncal_paper_white_max=0.18,
    t13_uncal_sat_base=0.86,
    t13_uncal_sat_slope=0.14,
    t13_uncal_sat_min=0.84,
    t13_uncal_sat_max=0.96,
    t13_uncal_vibrance_add=0.0,
    t13_uncal_wb_slope=0.78,
    t13_uncal_wb_max=0.46,
    t13_uncal_tint_base=0.04,
    t13_uncal_tint_slope=0.0,
    t13_uncal_ev_slope=0.54,
    t13_uncal_ev_max=0.36,
    # T14: uncalibrated per-channel slope correction (blue crossover vs span_d_g, red contrast match)
    t14_uncal_slope_check=True,
    t14_uncal_slope_b_base=0.12,
    t14_uncal_slope_b_slope=0.22,
    t14_uncal_slope_b_min=0.04,
    t14_uncal_slope_b_max=0.26,
    t14_uncal_slope_r_add=-0.09,
    # T15: calibrated camera-scan global contrast refinement (smooth contrast lift, capped)
    t15_cal_cnt_check=True,
    t15_cal_cnt_add=0.08,
    t15_cal_cnt_cap=1.75,
    # T16: calibrated span-aware EV down-refinement cap, severe-cyan sky GW undamp, and uncalibrated WB strength
    t16_cal_span_ev_down_check=True,
    t16_cal_span_lo=0.82,
    t16_cal_span_width=0.28,
    t16_cal_ev_down_min=0.08,
    t16_cal_sev_cyan_check=True,
    t16_cal_sev_cyan_r_max=-1.20,
    t16_uncal_wb_str_check=True,
    t16_uncal_wb_strength=0.69,
    # T17: calibrated C-41 cross-unmix weight + non-sky cyan rescue; uncalibrated WB confidence curve + warm de-overshoot
    t17_cal_cross_weight=0.62,
    t17_cal_nonsky_cyan_check=True,
    t17_nonsky_z1_r_max=-0.10,
    t17_nonsky_z1_b_min=0.08,
    t17_nonsky_min_span=0.80,
    t17_uncal_wb_conf_gamma=0.85,
    t17_uncal_wb_deov_check=True,
    t17_deov_pivot=0.14,
    t17_deov_gain=0.28,
    t17_deov_max=0.09,
)

# Neutral-preserving (row-sum = 1.0) C-41 secondary coupler crosstalk unmixing matrix in linear Rec.709 RGB:
# removes cyan-dye green/blue and magenta-dye blue parasitic cross-coupling without shifting D65 white/grey.
C41_CROSS_UNMIX_MATRIX = (
    (1.058406, -0.009586, -0.048820),
    (-0.008839, 0.986508, 0.022331),
    (-0.109867, 0.194770, 0.915097),
)

# T5k (K4): calibrated camera-scan C-41 dye unmixing refinement (neutral-preserving, row-sum = 1.0)
C41_DYE_UNMIX_K4_MATRIX = (
    (1.077900, -0.024600, -0.053300),
    (-0.014100, 1.018900, -0.004800),
    (-0.003800, 0.017200, 0.986700),
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
    Pc_flat = np.swapaxes(Pc, 1, 2).reshape(hc, wc, cs * cs, 3)
    Y = np.maximum(Pc_flat @ LUMA_F32, np.float32(1e-4))
    D = -np.log10(Y)
    mean_D = D.mean(axis=-1)
    std_D = np.sqrt(np.maximum(((D - mean_D[..., None]) ** 2).mean(axis=-1), np.float32(0.0)))
    mean_rgb = Pc_flat.mean(axis=2)
    if mask is not None:
        mc = np.swapaxes(mask[: hc * cs, : wc * cs].reshape(hc, cs, wc, cs), 1, 2).reshape(hc, wc, cs * cs).mean(axis=-1) > 0.5
    else:
        mc = np.ones((hc, wc), bool)
    valid = mc & np.isfinite(std_D) & (_max3(mean_rgb) > cfg["min_luma"])
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
    if len(dt) == 0:
        dt = mean_D.ravel()
    if len(dt) == 0:
        dt = np.zeros(1, dtype=np.float32)
    p2, p10, p25, p50, p75, p90, p98 = (float(x) for x in np.quantile(dt, [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]))
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
            fb = film_box(hold)
            if fb is not None:
                t, l, bh, bw = fb
                dt, dl = int(round(cfg["meter_box_inset"] * bh)), int(round(cfg["meter_box_inset"] * bw))
                box = np.zeros_like(m)
                if bh > 2 * dt and bw > 2 * dl:
                    box[t + dt : t + bh - dt, l + dl : l + bw - dl] = True
                m &= box
                m &= ~hold
                diag.update(film_box=[t, l, bh, bw], box_inset=[dt, dl])
            else:
                m[:] = False
    if cfg.get("meter_trim_strips") and m.any():
        strip_luma = np.float32(cfg["meter_strip_luma"])
        strip_share = float(cfg["meter_strip_share"])
        k_margin = int(cfg["meter_strip_margin"])
        p_black = P.max(axis=-1) <= strip_luma
        trimmed = dict(top=0, bottom=0, left=0, right=0)
        for side in trimmed:
            rows = np.flatnonzero(m.any(1))
            cols = np.flatnonzero(m.any(0))
            if len(rows) == 0 or len(cols) == 0:
                break
            if side == "top":
                for i in rows:
                    sel = m[i]
                    if float(p_black[i, sel].mean()) < strip_share:
                        break
                    m[i] = False
                    trimmed[side] += 1
            elif side == "bottom":
                for i in rows[::-1]:
                    sel = m[i]
                    if float(p_black[i, sel].mean()) < strip_share:
                        break
                    m[i] = False
                    trimmed[side] += 1
            elif side == "left":
                for i in cols:
                    sel = m[:, i]
                    if float(p_black[sel, i].mean()) < strip_share:
                        break
                    m[:, i] = False
                    trimmed[side] += 1
            else:
                for i in cols[::-1]:
                    sel = m[:, i]
                    if float(p_black[sel, i].mean()) < strip_share:
                        break
                    m[:, i] = False
                    trimmed[side] += 1
            if trimmed[side]:  # the strip's soft edge
                rows2, cols2 = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
                if side == "top" and len(rows2):
                    m[rows2[0] : rows2[0] + k_margin] = False
                elif side == "bottom" and len(rows2):
                    m[max(rows2[-1] - k_margin + 1, 0) : rows2[-1] + 1] = False
                elif side == "left" and len(cols2):
                    m[:, cols2[0] : cols2[0] + k_margin] = False
                elif side == "right" and len(cols2):
                    m[:, max(cols2[-1] - k_margin + 1, 0) : cols2[-1] + 1] = False
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


def estimate_wb_v3(px, cfg=None, *, span_d_g: float | None = None) -> dict:
    """Multi-zone Huber white balance with grey-world blend and sky/cast guards."""
    cfg = cfg or NOREF_V3
    px = np.asarray(px, np.float32)
    y = px @ LUMA_F32
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
    # T4k: mid-grey zone and grey world both strongly cool on a frame without a sky over ground (_DSF8951): the grey
    # world is a cast, not a sky
    gw_undamp = bool(cfg.get("wb_v3_gw_cool_undamp") and est_gw is not None and 1 in found
                     and cool(ests[found.index(1)], "wb_v3_undamp_z1") and cool(est_gw, "wb_v3_undamp_gw")
                     and diag.get("top_bottom_spread", np.inf) < cfg["wb_v3_undamp_tb_max"])
    diag["gw_cool_undamped"] = gw_undamp
    is_cal = cfg.get("matrix") is not None
    nonsky_cyan = False
    if cfg.get("t17_cal_nonsky_cyan_check") and is_cal and not split_sky and not (cool_cast or gw_undamp):
        min_sp = cfg.get("t17_nonsky_min_span")
        if min_sp is None or (span_d_g is not None and span_d_g >= float(min_sp)):
            z1_est = ests[found.index(1)] if 1 in found else None
            if z1_est is not None and z1_est[0] < float(cfg.get("t17_nonsky_z1_r_max", -0.10)) and z1_est[1] > float(cfg.get("t17_nonsky_z1_b_min", 0.08)):
                nonsky_cyan = True
    sev_cyan = bool(
        is_cal
        and cfg.get("t16_cal_sev_cyan_check")
        and est_gw is not None
        and not (cool_cast or gw_undamp)
        and est_gw[0] < sky_r
        and est_gw[1] > sky_b
        and est_gw[0] < float(cfg.get("t16_cal_sev_cyan_r_max", -1.20))
    )
    sky_weight = 1.0 if (sev_cyan or nonsky_cyan) else float(cfg["wb_v3_sky_weight"])
    gw_sky_red = 1.0 if (sev_cyan or nonsky_cyan) else float(cfg["wb_v3_gw_sky_red"])
    if sev_cyan:
        diag["t16_cal_sev_cyan"] = True
    if nonsky_cyan:
        diag["t17_cal_nonsky_cyan"] = True
    for k, i in enumerate(found):
        if i == 0 and not cool_cast and ests[k][0] < sky_r and ests[k][1] > sky_b:
            # T4f: a bright zone with red low and blue high reads as clear sky, not as a cool cast
            weights[k] = weights[k] * sky_weight
            diag["zones"][i].update(clear_sky=True)
        diag["zones"][i].update(weight=float(weights[k]))
    w = np.asarray(weights) / np.sum(weights)
    est = (w[:, None] * np.asarray(ests)).sum(0)
    if est_gw is not None:
        sparse = any(z["narrow_share"] < cfg["wb_v3_adaptive_share"] for z in diag["zones"])
        gw_weight = float(cfg["wb_v3_gw_weight"]) if sparse or np.linalg.norm(est_gw) > cfg["wb_v3_gw_threshold"] else 0.0
        if cool_cast:
            gw_weight = float(cfg["wb_v3_cast_gw_weight"])
        diag["grey_world"] = dict(chroma=[float(v) for v in est_gw], weight=gw_weight)
        if not (cool_cast or gw_undamp) and est_gw[0] < sky_r and est_gw[1] > sky_b:
            # T4h: in a clear-sky direction only the blue axis is damped (T4g damped the whole weight), so a cyan cast
            # (red low) is still corrected without pulling the sky yellow
            est_gw = np.array([est_gw[0] * gw_sky_red, est_gw[1] * sky_weight])
            diag["grey_world"].update(clear_sky=True, chroma_used=[float(v) for v in est_gw])
        est = (1 - gw_weight) * est + gw_weight * est_gw
    allc = np.concatenate(cands)
    mad = float(np.median(np.linalg.norm(allc - est, axis=1)))
    share = float(sum(z["share"] for z in diag["zones"] if "chroma" in z) / len(diag["zones"]))
    conf = _ramp(share, *cfg["wb_v3_share"]) * (1 - 0.5 * _ramp(mad, *cfg["wb_v3_mad"]))
    if diag.get("top_bottom_spread", 0.0) > cfg["wb_conflict"]:
        conf *= cfg["wb_v3_conflict_factor"]
        diag["conflict"] = True
    if not is_cal and cfg.get("t17_uncal_wb_conf_gamma") is not None and not split_sky and conf > 0.0:
        conf = float(conf ** float(cfg["t17_uncal_wb_conf_gamma"]))
    strength = (
        float(cfg.get("t16_uncal_wb_strength", 0.72))
        if (not is_cal and cfg.get("t16_uncal_wb_str_check"))
        else cfg["wb_v3_strength"]
    )
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


def _pre_shoulder_max(px: np.ndarray, p_dict: dict) -> np.ndarray:
    """Max channel of linear positive right before shoulder_curve (invariant across shoulder & highlights)."""
    rp = RenderParams(**p_dict)
    x_lin, _ = pre_shoulder_linear(np.asarray(px, np.float32), rp, fast_pow=True)
    return _max3(x_lin)


def _encoded_quantiles(px, p: RenderParams, qs=(2, 50, 98)):
    e, _, _, _ = render_fast_full(px, p)
    y = e @ LUMA_F32
    return [float(x) for x in np.quantile(y, [q / 100.0 for q in qs])]


def _compose_matrix(cur_m, factor_m):
    m = np.asarray(factor_m, dtype=np.float64) @ np.asarray(cur_m, dtype=np.float64)
    m = m - (m.sum(1, keepdims=True) - 1.0) / 3.0
    return [list(map(float, r)) for r in m]


def _apply_wb_and_unmix_v3(
    base_p: dict, wd: dict, zs: list, gw: list, p90: float,
    span_d_g: float | None, bright_share: float, wb_conf: float,
    has_cam_matrix: bool, cfg: dict, pg_diag: dict,
) -> None:
    """C15 multi-zone WB crossover + C16 chroma-damped C-41 cross-hue dye unmixing."""
    t_off = float(cfg.get("tint_offset", 0.0))
    if cfg.get("t5j_wb_crossover_check") and p90 > 1e-4 and len(zs) >= 2 and "chroma" in zs[0] and "chroma" in zs[1]:
        z0, z1 = zs[0]["chroma"], zs[1]["chroma"]
        if z0[1] <= -0.205 and z1[1] <= -0.12 and max(z0[0], z1[0]) <= 0.005 and base_p["tint"] < 0.25 + t_off:
            t_add = float(np.clip((-0.5 * (z0[1] + z1[1]) - 0.10) * 0.60, 0.0, 0.085))
            if t_add > 0.0:
                base_p["tint"] = float(np.clip(base_p["tint"] + t_add, *BOUNDS["tint"]))
                pg_diag["wb_warm_fluor_tint"] = round(t_add, 4)
        elif (not wd.get("split_sky") and not wd.get("cool_cast")
              and bright_share < 0.24 and z1[1] >= 0.16 and z1[0] <= -0.18
              and gw[1] >= 0.28 and gw[0] <= -0.28):
            c_str = min(z1[1] - 0.10, -z1[0] - 0.12)
            temp_add = float(np.clip(c_str * 1.4, 0.0, 0.22))
            tint_add = float(np.clip((-z1[0] - 0.15) * 0.45, 0.0, 0.06))
            base_p["temp"] = float(np.clip(base_p["temp"] + temp_add, *BOUNDS["temp"]))
            base_p["tint"] = float(np.clip(base_p["tint"] + tint_add, *BOUNDS["tint"]))
            pg_diag["wb_shaded_cool"] = dict(temp_add=round(temp_add, 4), tint_add=round(tint_add, 4))

    cam_mix = float(cfg.get("camera_matrix_mix", 0.40))
    cp_max = float(cfg.get("t5j_dye_unmix_max_cp90", 0.73))
    if cam_mix > 0.45:
        cp_max = min(cp_max, 0.68)
    if (cfg.get("t5j_dye_unmix_check") and has_cam_matrix and span_d_g is not None
            and 1e-4 < p90 <= cp_max and span_d_g >= 0.58 and wb_conf >= 0.90
            and not wd.get("cool_cast") and not wd.get("cyan_conf_boost")
            and gw[0] >= -0.18 and gw[1] <= 0.22):
        w_unmix = float(np.clip((cp_max + 0.02 - p90) / 0.40, 0.30, 1.0))
        m_eff = (1.0 - w_unmix) * np.eye(3) + w_unmix * np.asarray(C41_CROSS_UNMIX_MATRIX, dtype=np.float64)
        base_p["matrix"] = _compose_matrix(base_p["matrix"], m_eff)
        pg_diag["dye_unmix_weight"] = round(w_unmix, 4)


def _apply_overlift_convergence_v3(
    base_p: dict, tc: dict, ol: dict, refine: dict, ev_lowered: float,
    w_bl: float, q25: float, bright_share: float, span_d_g: float | None,
    enc_p2_eff: float, enc_p98: float, spread: float, p90: float,
    has_cam_matrix: bool, thin_applied: bool, lifted_applied: bool,
    cfg: dict, pg_diag: dict,
) -> None:
    """C13 Boyack-Juenger low-key damping + C14 residual overlift & narrow-fog convergence."""
    ev_off = float(base_p["ev"])
    q25_sh = 0.35 * max(0.0, float(cfg.get("base_shadows", 0.0)))
    base_cnt = float(cfg["base_contrast"])
    anc_drop = float(refine["ev_anchor_raw"]) - float(refine["ev"])
    if (cfg.get("t5j_boyack_lowkey_check") and cfg.get("overlift_damp_check", True)
            and p90 > 1e-4 and span_d_g is not None
            and span_d_g >= 0.82 and bright_share < 0.06
            and ev_off >= 1.35 and float(refine["ev_anchor_raw"]) >= 1.80
            and ev_lowered == 0.0 and w_bl < 0.25 and q25 >= 0.17 + q25_sh
            and (anc_drop < 0.30 or span_d_g < 1.12)):
        pull_lk = float(np.clip((ev_off - 1.15) * 0.38, 0.0, 0.48))
        if pull_lk > 0.0:
            base_p["ev"] = float(max(cfg["ev_limits"][0], base_p["ev"] - pull_lk))
            base_p["blacks"] = float(np.clip(base_p["blacks"] - min(0.05, pull_lk * 0.10), *EXTENDED_BOUNDS["blacks"]))
            pg_diag["boyack_lowkey_pull"] = round(pull_lk, 4)

    if not cfg.get("t5j_overlift_conv_check"):
        return
    if ol.get("mode") == "moderate" and ol.get("ev_before", 2.0) < 1.34 and base_p["ev"] >= 0.35 and enc_p2_eff >= 0.11:
        extra = float(np.clip((base_p["ev"] - 0.30) * 0.25, 0.0, 0.22))
        base_p["ev"] = float(max(cfg["ev_limits"][0], base_p["ev"] - extra))
        base_p["blacks"] = float(np.clip(base_p["blacks"] - extra * 0.12, *EXTENDED_BOUNDS["blacks"]))
        pg_diag["overlift_conv"] = dict(mode="moderate", extra_pull=round(extra, 4))
    elif (not ol.get("applied") and span_d_g is not None and span_d_g < 0.76
          and bright_share >= 0.25 and 0.28 <= base_p["ev"] < 0.45
          and q25 >= 0.47 + q25_sh and w_bl == 0.0):
        extra = float(np.clip((base_p["ev"] - 0.15) * 0.55, 0.0, 0.22))
        base_p["ev"] = float(max(cfg["ev_limits"][0], base_p["ev"] - extra))
        base_p["blacks"] = float(np.clip(base_p["blacks"] - 0.03, *EXTENDED_BOUNDS["blacks"]))
        pg_diag["overlift_conv"] = dict(mode="short_span_gap", extra_pull=round(extra, 4))
    elif ol.get("mode") == "bright_short_span" and enc_p2_eff >= 0.22 and base_p["ev"] >= 0.25:
        extra = float(np.clip((base_p["ev"] - 0.10) * 0.45, 0.0, 0.24))
        base_p["ev"] = float(max(cfg["ev_limits"][0], base_p["ev"] - extra))
        base_p["blacks"] = float(np.clip(base_p["blacks"] - 0.04, *EXTENDED_BOUNDS["blacks"]))
        pg_diag["overlift_conv"] = dict(mode="bright_short_span", extra_pull=round(extra, 4))
    elif (p90 > 1e-4 and not lifted_applied and not thin_applied
          and base_cnt >= 1.25
          and base_p["ev"] <= 0.05 and bright_share < 0.20
          and 0.20 <= spread <= 0.28 and enc_p2_eff >= 0.21 and tc["p98_dark_d"] >= 1.20):
        drop = float(np.clip((0.30 - spread) * 2.2, 0.0, 0.24))
        base_p["ev"] = float(max(cfg["ev_limits"][0], base_p["ev"] - drop))
        base_p["blacks"] = float(np.clip(base_p["blacks"] - 0.14, *EXTENDED_BOUNDS["blacks"]))
        base_p["contrast"] = float(min(BOUNDS["contrast"][1], min(base_cnt + 0.25, base_p["contrast"] + 0.18)))
        if cfg.get("t5j_print_toe_bpc_check"):
            toe = float(np.clip((enc_p2_eff - 0.15) * 0.40, 0.0, 0.085))
            if has_cam_matrix and enc_p98 <= 0.855:
                base_p["shoulder"] = float(max(0.06, base_p["shoulder"] - 0.15))
                base_p["tone_curve"] = {"rgb": [[0.0, 0.0], [0.28, round(0.28 - toe, 4)], [0.58, round(0.58 - toe * 0.25, 4)], [0.84, 0.88], [1.0, 1.0]]}
            elif toe >= 0.01:
                base_p["tone_curve"] = {"rgb": [[0.0, 0.0], [0.26, round(0.26 - toe * 0.7, 4)], [0.58, 0.58], [1.0, 1.0]]}
        pg_diag["overlift_conv"] = dict(mode="narrow_fog_floor", ev_drop=round(drop, 4))


def _apply_paper_white_and_toe_v3(
    base_p: dict, tc: dict, wd: dict, span_d_g: float | None, bright_share: float,
    w_bl: float, q25: float, enc_p2_eff: float, enc_med: float, enc_p98: float,
    spread: float, hl_unburned: bool, has_cam_matrix: bool, thin_applied: bool,
    lifted_applied: bool, cfg: dict, pg_diag: dict,
) -> None:
    """C11 Paper-White Ceiling Release + C12 Print-Toe BPC tone-curve placement."""
    sh_off = max(0.0, float(cfg.get("base_shadows", 0.0)))
    q25_sh = 0.35 * sh_off
    base_cnt = float(cfg["base_contrast"])
    has_tc = bool(base_p.get("tone_curve"))

    blk_q25_comp = 0.55 * min(0.0, float(cfg.get("base_blacks", 0.0)))
    if (cfg.get("t5j_paper_white_check") and not has_tc and has_cam_matrix and span_d_g is not None
            and float(cfg["base_sat"]) <= 1.05):
        if (span_d_g >= 0.78 and enc_p98 <= 0.875 and base_p["ev"] <= 0.35
                and q25 >= 0.42 + q25_sh + blk_q25_comp and hl_unburned and w_bl == 0.0 and not wd.get("cool_cast")
                and not lifted_applied and not thin_applied):
            if bright_share >= 0.12 and tc["p98_dark_d"] >= 0.92 and tc["span_10_90_d"] >= 0.55:
                base_p["shoulder"] = float(max(0.05, base_p["shoulder"] - 0.16))
                toe = float(np.clip((enc_p2_eff - 0.08) * 0.28, 0.0, 0.055)) if enc_p2_eff >= 0.10 else 0.0
                pts = ([[0.0, 0.0], [0.24, round(0.24 - toe, 4)], [0.54, 0.54], [0.82, 0.87], [1.0, 1.0]]
                       if toe > 0.004 else [[0.0, 0.0], [0.52, 0.52], [0.82, 0.87], [1.0, 1.0]])
                base_p["tone_curve"] = {"rgb": pts}
                if spread < 0.50 and base_p["contrast"] <= base_cnt + 0.02:
                    inc = min(0.22, (0.50 - spread) * 0.75)
                    base_p["contrast"] = float(min(BOUNDS["contrast"][1], base_p["contrast"] + inc))
                has_tc = True
                pg_diag["paper_white_curve"] = "bright_cam_print"
            elif (bright_share >= 0.06 and enc_med >= 0.48 + 0.20 * sh_off
                  and tc["p98_dark_d"] >= 1.60 and tc["span_10_90_d"] >= 0.80):
                base_p["shoulder"] = float(max(0.06, base_p["shoulder"] - 0.16))
                toe = max(0.025, float(np.clip((enc_p2_eff - 0.06) * 0.45, 0.0, 0.045)))
                base_p["tone_curve"] = {"rgb": [[0.0, 0.0], [0.25, round(0.25 - toe, 4)], [0.54, 0.54], [0.82, 0.87], [1.0, 1.0]]}
                if spread < 0.52 and base_p["contrast"] <= base_cnt + 0.02:
                    inc = min(0.22, (0.54 - spread) * 2.2)
                    base_p["contrast"] = float(min(BOUNDS["contrast"][1], base_p["contrast"] + inc))
                has_tc = True
                pg_diag["paper_white_curve"] = "mid_cam_print"
        if (not has_tc and tc["p98_dark_d"] >= 2.50 and tc["span_10_90_d"] >= 1.50
                and 0.56 <= span_d_g < 0.72 and enc_med <= 0.48 + 0.20 * sh_off
                and enc_p98 <= 0.86 and 0.02 <= base_p["ev"] <= 0.25):
            base_p["ev"] = float(max(cfg["ev_limits"][0], base_p["ev"] - 0.08))
            base_p["shoulder"] = float(max(0.08, base_p["shoulder"] - 0.15))
            base_p["contrast"] = float(min(BOUNDS["contrast"][1], min(base_cnt + 0.25, base_p["contrast"] + 0.14)))
            base_p["tone_curve"] = {"rgb": [[0.0, 0.0], [0.25, 0.21], [0.55, 0.55], [0.82, 0.87], [1.0, 1.0]]}
            has_tc = True
            pg_diag["paper_white_curve"] = "hi_density_cam_print"

    if (cfg.get("t5j_print_toe_bpc_check") and not has_tc and span_d_g is not None
            and 0.58 <= span_d_g < 1.15 and enc_p2_eff >= 0.165 and enc_p98 >= 0.80 and tc["p98_dark_d"] >= 1.35
            and (base_p["ev"] >= 0.28 or (has_cam_matrix and w_bl >= 0.40 and base_p["ev"] >= 0.18 and enc_p98 <= 0.87))
            and not lifted_applied):
        if has_cam_matrix and w_bl >= 0.40 and enc_p98 <= 0.87:
            base_p["shoulder"] = float(max(0.10, base_p["shoulder"] - 0.18))
            if base_p.get("highlights", 0.0) < float(cfg["base_highlights"]):
                base_p["highlights"] = float(min(float(cfg["base_highlights"]), base_p["highlights"] + 0.16))
            base_p["ev"] = float(max(cfg["ev_limits"][0], base_p["ev"] - 0.06))
            toe = float(np.clip((enc_p2_eff - 0.11) * 0.42, 0.0, 0.055))
            base_p["tone_curve"] = {"rgb": [[0.0, 0.0], [0.25, round(0.25 - toe, 4)], [0.55, 0.55], [0.82, 0.87], [1.0, 1.0]]}
        else:
            toe = float(np.clip((enc_p2_eff - 0.12) * 0.36, 0.0, 0.068))
            if toe >= 0.008:
                base_p["tone_curve"] = {"rgb": [[0.0, 0.0], [0.25, round(0.25 - toe, 4)], [0.56, round(0.56 - toe * 0.22, 4)], [1.0, 1.0]]}
        if base_p.get("tone_curve") and spread < 0.58 and base_p["contrast"] <= base_cnt + 0.08:
            inc = min(0.25, (0.58 - spread) * 0.65)
            base_p["contrast"] = float(min(BOUNDS["contrast"][1], base_p["contrast"] + inc))
        pg_diag["print_toe_bpc"] = True


def _apply_cal_grade_v3(
    base_p: dict, px: np.ndarray, wd: dict, zs: list, gw: list,
    span_d_g: float | None, spread: float, p90: float, wb_conf: float,
    enc_p2: float, enc_med: float, enc_p2_post_eff: float, enc_p98_post: float,
    lin_q2_raw: float, lin_q98_raw: float, cfg: dict, pg_diag: dict,
) -> None:
    """Calibrated camera-scan print grade (T5k..T15 CAL branch)."""
    if (cfg.get("t5k_bpc_check") and not wd.get("cool_cast")
            and "wb_warm_fluor_tint" not in pg_diag and wb_conf >= 0.85
            and enc_p98_post >= 0.90 and enc_p2_post_eff >= 0.25 and base_p["ev"] <= 0.30):
        lin_p2 = lin_q2_raw * (2.0 ** float(base_p["ev"]))
        if lin_p2 >= 0.018:
            bpc_val = float(np.clip((lin_p2 - 0.012) * 0.35, 0.0, 0.028))
            if bpc_val > 0.0:
                base_p["bpc"] = round(bpc_val, 4)
                pg_diag["t5k_bpc"] = round(bpc_val, 4)

    if cfg.get("t5k_soft_clip_check") and not base_p.get("tone_curve") and base_p["shoulder"] >= 0.16 and base_p["ev"] <= 1.10:
        base_p["soft_clip"] = 0.78
        pg_diag["t5k_soft_clip"] = 0.78

    if (cfg.get("t5k_dye_unmix_v2_check") and span_d_g is not None and span_d_g >= 0.58
            and wb_conf >= (0.85 if cfg.get("t6_v2_cal_bpc_shoulder_check") else 0.89)
            and not wd.get("cool_cast") and not wd.get("cyan_conf_boost")
            and gw[0] >= -0.18 and -0.25 <= gw[1] <= 0.22 and enc_p98_post >= 0.85
            and float(cfg.get("base_sat", 1.0)) <= 1.12):
        base_p["matrix"] = _compose_matrix(base_p["matrix"], C41_DYE_UNMIX_K4_MATRIX)
        pg_diag["t5k_dye_unmix_v2"] = True

    if cfg.get("t5k_blue_sat_wb_check"):
        if base_p["temp"] >= 0.495 and gw[1] >= 0.45 and wd.get("cool_cast"):
            tc_cur = dict(base_p.get("tone_curve") or {})
            tc_cur["b"] = [[0.0, 0.0], [0.25, 0.205], [0.5, 0.435], [0.75, 0.705], [1.0, 1.0]]
            base_p["tone_curve"] = tc_cur
            pg_diag["t5k_blue_sat_curve"] = True
        elif "wb_shaded_cool" in pg_diag and len(zs) >= 2 and "chroma" in zs[1]:
            z1 = zs[1]["chroma"]
            c_str = min(z1[1] - 0.10, -z1[0] - 0.12)
            extra = float(np.clip(c_str * 0.6, 0.0, 0.10))
            if extra > 0.0:
                base_p["temp"] = float(np.clip(base_p["temp"] + extra, *BOUNDS["temp"]))
                pg_diag["t5k_shaded_cool_extra"] = round(extra, 4)

    if cfg.get("t6_print_contrast_check") and p90 > 1e-4:
        has_tc_now = bool(base_p.get("tone_curve"))
        c_before = float(base_p["contrast"])
        if not has_tc_now:
            if spread < 0.68 and base_p["contrast"] < 1.80:
                base_p["contrast"] = float(min(1.80, base_p["contrast"] + (0.68 - spread) * 1.8))
                pg_diag["t6_cal_contrast_add"] = round(float(base_p["contrast"]) - c_before, 4)
            if cfg.get("t6_paper_white_release_check") and enc_p98_post < 0.93:
                w_add = float(min(0.25, (0.94 - enc_p98_post) * 1.35))
                if w_add > 0.0:
                    base_p["whites"] = float(min(0.40, base_p.get("whites", 0.0) + w_add))
                    pg_diag["t6_paper_white_whites"] = round(w_add, 4)
        elif spread < 0.52 and base_p["contrast"] < 1.60:
            base_p["contrast"] = float(min(1.60, base_p["contrast"] + (0.52 - spread) * 1.1))
            pg_diag["t6_tc_contrast_add"] = round(float(base_p["contrast"]) - c_before, 4)

    if cfg.get("t6_shadow_toe_check") and p90 > 1e-4 and enc_p2_post_eff > 0.07 and enc_p98_post > 0.78:
        blk_pull = float(min(0.18, (enc_p2_post_eff - 0.06) * 0.75))
        if blk_pull > 0.0:
            base_p["blacks"] = float(np.clip(base_p["blacks"] - blk_pull, *EXTENDED_BOUNDS["blacks"]))
            pg_diag["t6_shadow_toe_blacks"] = round(blk_pull, 4)

    if cfg.get("t6_v2_cal_bpc_shoulder_check") and p90 > 1e-4:
        if (not wd.get("cool_cast") and enc_p2_post_eff > 0.06 and enc_p98_post > 0.78
                and "t6_shadow_toe_blacks" not in pg_diag):
            lin_p2 = lin_q2_raw * (2.0 ** float(base_p["ev"]))
            bpc_v = float(np.clip((enc_p2_post_eff - 0.06) * 0.08, 0.0, min(0.014, max(0.0, lin_p2 * 0.70))))
            if bpc_v > float(base_p.get("bpc", 0.0)):
                base_p["bpc"] = round(bpc_v, 4)
                pg_diag["t6_v2_cal_bpc"] = round(bpc_v, 4)
        if not bool(base_p.get("tone_curve")) and enc_p98_post < 0.94 and base_p["shoulder"] > 0.06:
            shd_drop = float(min(0.12, (0.95 - enc_p98_post) * 0.8))
            if shd_drop > 0.0:
                base_p["shoulder"] = float(max(0.06, base_p["shoulder"] - shd_drop))
                pg_diag["t6_v2_shoulder_release"] = round(shd_drop, 4)

    if cfg.get("t6_v3_cal_refine_check") and p90 > 1e-4:
        post_v2_params = RenderParams(**{
            **base_p,
            **{k: round(float(base_p[k]), 4) for k in (
                "ev", "temp", "tint", "sat", "contrast", "midtone", "shoulder",
                "highlights", "shadows", "whites", "blacks", "vibrance", "slope_r", "slope_b"
            )},
        })
        (enc_med_post_v2,) = _encoded_quantiles(px, post_v2_params, qs=(50,))
        if enc_med_post_v2 >= 0.56:
            hk_guard = cfg.get("t10_cal_highkey_guard_check") and enc_p2_post_eff >= 0.25
            shd_mult, wht_mult = (0.5, 0.35) if hk_guard else (1.0, 0.7)
            shd_rel = float(pg_diag.get("t6_v2_shoulder_release", 0.0))
            if shd_rel > 0.0:
                shd_add = shd_rel * shd_mult
                base_p["shoulder"] = float(base_p["shoulder"]) + shd_add
                pg_diag["t6_v3_bright_shd_restore"] = round(shd_add, 4)
            wht_add = float(pg_diag.get("t6_paper_white_whites", 0.0))
            if wht_add > 0.0:
                wht_damp = wht_add * wht_mult
                base_p["whites"] = float(base_p["whites"]) - wht_damp
                pg_diag["t6_v3_bright_wht_damp"] = round(wht_damp, 4)
        if float(base_p["contrast"]) > 1.65 and "t6_cal_contrast_add" in pg_diag:
            base_p["contrast"] = 1.65
            pg_diag["t6_v3_cal_cnt_cap"] = 1.65
        if not pg_diag.get("t5k_dye_unmix_v2"):
            base_p["matrix"] = _compose_matrix(base_p["matrix"], C41_DYE_UNMIX_K4_MATRIX)
            pg_diag["t6_v3_cal_k4_unmix"] = True
        w_cr = float(cfg.get("t17_cal_cross_weight", 0.5))
        m_c50 = (1.0 - w_cr) * np.eye(3) + w_cr * np.asarray(C41_CROSS_UNMIX_MATRIX, dtype=np.float64)
        base_p["matrix"] = _compose_matrix(base_p["matrix"], m_c50)
        pg_diag["t6_v3_cal_cross50"] = True
        if cfg.get("t6_v4_cal_refine_check"):
            if float(base_p["midtone"]) <= 0.235:
                m_t = 0.225
                if span_d_g is not None and span_d_g < 0.58:
                    m_t -= min(0.035, (0.58 - float(span_d_g)) * 0.25)
                if enc_med_post_v2 > 0.55:
                    m_t -= min(0.012, (enc_med_post_v2 - 0.55) * 0.06)
                base_p["midtone"] = float(round(max(0.185, m_t), 4))
            if span_d_g is not None and span_d_g > 1.05 and enc_p98_post < 0.90:
                ev_add = float(min(0.15, (float(span_d_g) - 1.05) * 0.5))
                base_p["ev"] = float(np.clip(base_p["ev"] + ev_add, *cfg["ev_limits"]))
            base_p["sat"] = round(float(base_p["sat"]) * 0.92, 4)
            if cfg.get("t8_cal_chroma_guard_check"):
                dc_cal = float(pg_diag.get("t6_cal_contrast_add", 0.0))
                if dc_cal > 0.25:
                    base_p["sat"] = round(float(base_p["sat"]) * max(0.88, 1.0 - 0.16 * (dc_cal - 0.25)), 4)
            base_p["chroma_preserve"] = max(float(base_p.get("chroma_preserve", 0.0)), 0.55)
            pg_diag["t6_v4_cal_refine"] = True

    if cfg.get("t7_hl_dark_guard_check") and p90 > 1e-4:
        lin_p98_ev = lin_q98_raw * (2.0 ** float(base_p["ev"]))
        if float(base_p["ev"]) > 0.35 and enc_med > 0.50 and enc_p2 > 0.25:
            if lin_q98_raw > 0.80 and lin_p98_ev > 1.65:
                ev_cap = float(max(0.30, np.log2(1.45 / lin_q98_raw)))
                if ev_cap < float(base_p["ev"]):
                    pg_diag["t7_hl_ev_cap"] = round(float(base_p["ev"]) - ev_cap, 4)
                    base_p["ev"] = ev_cap
        if enc_p2 > 0.48 and enc_med > 0.50:
            lin_p2 = lin_q2_raw * (2.0 ** float(base_p["ev"]))
            if lin_p2 > 0.025:
                bpc_cap = min(0.025, lin_p2 * 0.30) if cfg.get("t11_cal_bpc_damp_check") else min(0.055, lin_p2 * 0.50)
                bpc_lift = float(np.clip((enc_p2_post_eff - 0.30) * 0.22, 0.0, bpc_cap))
                if bpc_lift > float(base_p.get("bpc", 0.0)):
                    base_p["bpc"] = round(bpc_lift, 4)
                    pg_diag["t7_bpc_lift"] = round(bpc_lift, 4)

    if cfg.get("t9_cal_deep_blk_ev_check") and p90 > 1e-4:
        p2_ok = (not cfg.get("t10_cal_deep_blk_p2_guard_check")) or (enc_p2_post_eff < 0.34)
        if p2_ok and float(base_p.get("blacks", 0.0)) <= -0.15 and float(base_p.get("ev", 0.0)) <= -0.10:
            base_p["ev"] = round(float(base_p["ev"]) - 0.12, 4)
            pg_diag["t9_cal_deep_blk_ev"] = -0.12

    if cfg.get("t15_cal_cnt_check") and p90 > 1e-4:
        cnt_cap = float(cfg.get("t15_cal_cnt_cap", 1.75))
        cnt_add = float(cfg.get("t15_cal_cnt_add", 0.08))
        c_cur = float(base_p["contrast"])
        if cnt_add != 0.0 and c_cur < cnt_cap:
            c_new = float(min(cnt_cap, c_cur + cnt_add))
            base_p["contrast"] = c_new
            pg_diag["t15_cal_cnt_add"] = round(c_new - c_cur, 4)


def _apply_uncal_grade_v3(
    base_p: dict, span_d_g: float | None, bright_share: float, q25: float,
    spread: float, p90: float, enc_p2: float, enc_p2_eff: float,
    enc_p2_post_eff: float, enc_med: float, enc_p98: float,
    lin_q10_raw: float, lin_q50_raw: float, lin_q90_raw: float, lin_q98_raw: float,
    out: dict, cfg: dict, pg_diag: dict,
) -> None:
    """Uncalibrated scanner print grade (T5k..T14 UNCAL branch)."""
    base_cnt = float(cfg["base_contrast"])
    q25_sh = 0.35 * max(0.0, float(cfg.get("base_shadows", 0.0)))

    if (cfg.get("t5k_uncal_fog_check") and cfg.get("color_lifted_floor_check", True)
            and span_d_g is not None and p90 > 1e-4 and not base_p.get("tone_curve")
            and not out["diagnostics"].get("ultra_flat_bright", {}).get("applied")
            and not out["diagnostics"].get("lifted_floor", {}).get("applied")
            and not out["diagnostics"].get("auto_levels")
            and base_p["contrast"] <= base_cnt + 0.02
            and enc_p2_eff >= 0.30 and enc_p98 <= 0.85 and (enc_p98 - enc_p2) <= 0.50):
        toe = float(np.clip((enc_p2_eff - 0.22) * 0.45, 0.0, 0.085))
        if toe > 0.0:
            base_p["tone_curve"] = {
                "rgb": [[0.0, 0.0], [0.25, round(0.25 - toe, 4)], [0.55, round(0.55 - toe * 0.35, 4)], [1.0, 1.0]]
            }
            c_inc = min(0.25, (0.52 - (enc_p98 - enc_p2)) * 0.8)
            base_p["contrast"] = float(min(BOUNDS["contrast"][1], base_p["contrast"] + c_inc))
            pg_diag["t5k_uncal_fog_toe"] = round(toe, 4)

    wb_temp_init = float(base_p.get("temp", 0.0))
    if cfg.get("t6_uncal_overlift_check") and p90 > 1e-4:
        med_min = 0.42 if cfg.get("t8_uncal_mid_guard_check") else 0.35
        if base_p["ev"] > 0.10 and enc_med >= med_min and q25 >= 0.22 + q25_sh:
            pull = float(np.clip((base_p["ev"] - 0.05) * 0.65, 0.0, 0.65))
            if pull > 0.0:
                base_p["ev"] = float(max(cfg["ev_limits"][0], base_p["ev"] - pull))
                base_p["blacks"] = float(np.clip(base_p["blacks"] - min(0.08, pull * 0.12), *EXTENDED_BOUNDS["blacks"]))
                pg_diag["t6_uncal_overlift_pull"] = round(pull, 4)

    if cfg.get("t6_print_contrast_check") and p90 > 1e-4:
        has_tc_now = bool(base_p.get("tone_curve"))
        c_before = float(base_p["contrast"])
        if not has_tc_now and spread < 0.74 and base_p["contrast"] < 1.85:
            base_p["contrast"] = float(min(1.85, base_p["contrast"] + (0.74 - spread) * 1.9))
        elif has_tc_now and spread < 0.55 and base_p["contrast"] < 1.65:
            base_p["contrast"] = float(min(1.65, base_p["contrast"] + (0.55 - spread) * 1.4))
        dc = max(0.0, float(base_p["contrast"]) - c_before)
        if dc > 0.0:
            base_p["sat"] = float(max(0.65, base_p["sat"] * (1.0 - 0.30 * dc)))
            pg_diag["t6_uncal_contrast_add"] = round(dc, 4)

    if cfg.get("t6_v2_uncal_print_check") and p90 > 1e-4:
        mid_pre_v2 = float(base_p["midtone"])
        if cfg.get("t8_uncal_mid_guard_check"):
            if enc_med >= 0.52:
                base_p["midtone"] = 0.19
            elif enc_med >= 0.36:
                t_m = (enc_med - 0.36) / (0.52 - 0.36)
                base_p["midtone"] = float(round(0.220 - t_m * (0.220 - 0.190), 4))
            else:
                base_p["midtone"] = float(max(0.20, min(mid_pre_v2, 0.225)))
        elif enc_med >= 0.30:
            base_p["midtone"] = 0.19
        else:
            base_p["midtone"] = float(max(0.16, mid_pre_v2 - 0.03))
        base_p["chroma_preserve"] = max(float(base_p.get("chroma_preserve", 0.0)), 0.75)
        m_u = np.asarray(C41_DYE_UNMIX_K4_MATRIX, dtype=np.float64) @ np.asarray(C41_CROSS_UNMIX_MATRIX, dtype=np.float64)
        base_p["matrix"] = _compose_matrix(base_p["matrix"], m_u)
        base_p["temp"] = float(np.clip(base_p["temp"] + 0.15, *BOUNDS["temp"]))
        base_p["tint"] = float(np.clip(base_p["tint"] + 0.05, *BOUNDS["tint"]))
        bpc_p2_thr = 0.13 if cfg.get("t8_uncal_bpc_guard_check") else 0.08
        bpc_u_cap = 0.012 if cfg.get("t8_uncal_bpc_guard_check") else 0.020
        if enc_p2_post_eff > bpc_p2_thr:
            bpc_u = float(np.clip((enc_p2_post_eff - bpc_p2_thr) * 0.08, 0.0, bpc_u_cap))
            if bpc_u > float(base_p.get("bpc", 0.0)):
                base_p["bpc"] = round(bpc_u, 4)
                pg_diag["t6_v2_uncal_bpc"] = round(bpc_u, 4)
        pg_diag["t6_v2_uncal_print"] = True

        if cfg.get("t6_v3_uncal_refine_check"):
            if 0.30 <= enc_med < 0.42 and bright_share < 0.12:
                base_p["midtone"] = float(min(mid_pre_v2, 0.205))
            if enc_med >= 0.58 or (bright_share >= 0.30 and enc_med >= 0.48):
                pull = min(0.45, max(0.0, (enc_med - 0.48) * 1.1) + max(0.0, bright_share - 0.25) * 0.5)
                base_p["ev"] = float(max(cfg["ev_limits"][0], base_p["ev"] - pull))
                base_p["midtone"] = 0.185
                pg_diag["t6_v3_uncal_pull"] = round(pull, 4)
            if float(base_p["contrast"]) > 1.70:
                base_p["contrast"] = 1.70
                pg_diag["t6_v3_uncal_cnt_cap"] = 1.70
        if cfg.get("t6_v4_uncal_refine_check"):
            dark_uncal = cfg.get("t7_hl_dark_guard_check") and enc_med < 0.35
            u_cnt_cap = 1.40
            if cfg.get("t13_uncal_cnt_cap") is not None:
                if enc_p2_eff >= float(cfg.get("t13_uncal_cnt_p2_min", 0.0)) and spread < float(cfg.get("t13_uncal_cnt_spread_max", 1.0)):
                    u_cnt_cap = float(cfg["t13_uncal_cnt_cap"])
            if not dark_uncal and float(base_p["contrast"]) > u_cnt_cap:
                base_p["contrast"] = u_cnt_cap
                pg_diag["t6_v4_uncal_cnt_cap"] = u_cnt_cap
            pw_thr = float(cfg.get("t13_uncal_paper_white_thr", 0.89))
            if cfg.get("t13_uncal_paper_white_check") and enc_p98 < pw_thr:
                w_add_u = float(np.clip((pw_thr + 0.01 - enc_p98) * 1.15, 0.0, float(cfg.get("t13_uncal_paper_white_max", 0.16))))
                if w_add_u > 0.0:
                    base_p["whites"] = float(min(0.35, float(base_p.get("whites", 0.0)) + w_add_u))
                    pg_diag["t13_uncal_whites"] = round(w_add_u, 4)
            if float(cfg.get("t13_uncal_vibrance_add", 0.0)) != 0.0:
                base_p["vibrance"] = float(np.clip(float(base_p.get("vibrance", 0.0)) + float(cfg["t13_uncal_vibrance_add"]), -0.5, 0.6))
            if cfg.get("t12_uncal_span_sat_check"):
                sp = float(span_d_g) if span_d_g is not None else 0.80
                sat_m = float(np.clip(
                    float(cfg.get("t13_uncal_sat_base", 0.86)) + float(cfg.get("t13_uncal_sat_slope", 0.14)) * (sp - 0.80),
                    float(cfg.get("t13_uncal_sat_min", 0.84)),
                    float(cfg.get("t13_uncal_sat_max", 0.96)),
                ))
                base_p["sat"] = round(float(base_p["sat"]) * sat_m, 4)
                pg_diag["t12_uncal_span_sat"] = round(sat_m, 4)
            else:
                base_p["sat"] = round(float(base_p["sat"]) * 0.86, 4)
            if enc_med < 0.40 and bright_share < 0.12:
                is_lk = False
                if cfg.get("t12_uncal_lowkey_guard_check"):
                    lk_skew = (lin_q90_raw - lin_q50_raw) / max(lin_q50_raw - lin_q10_raw, 1e-5)
                    lk_dr = float(np.log2(max(lin_q98_raw, 1e-5) / max(lin_q10_raw, 1e-5)))
                    if lin_q10_raw < 0.045 and lk_skew > 3.0 and lk_dr > 4.5 and bright_share < 0.05:
                        is_lk = True
                        base_p["ev"] = float(max(cfg["ev_limits"][0], float(base_p["ev"]) - 0.18))
                        pg_diag["t12_uncal_lowkey_guard"] = True
                if not is_lk:
                    m_floor = 0.220 if dark_uncal else 0.200
                    if float(base_p["midtone"]) < m_floor:
                        base_p["midtone"] = m_floor
            if cfg.get("t14_uncal_slope_check"):
                sp = float(span_d_g) if span_d_g is not None else 0.80
                sb_add = float(np.clip(
                    float(cfg.get("t14_uncal_slope_b_base", 0.12))
                    + float(cfg.get("t14_uncal_slope_b_slope", 0.22)) * (sp - 0.70),
                    float(cfg.get("t14_uncal_slope_b_min", 0.04)),
                    float(cfg.get("t14_uncal_slope_b_max", 0.26)),
                ))
                sr_add = float(cfg.get("t14_uncal_slope_r_add", -0.09))
                base_p["slope_b"] = round(float(np.clip(float(base_p.get("slope_b", 0.0)) + sb_add, -0.5, 0.5)), 4)
                base_p["slope_r"] = round(float(np.clip(float(base_p.get("slope_r", 0.0)) + sr_add, -0.5, 0.5)), 4)
                pg_diag["t14_uncal_slope_b_add"] = round(sb_add, 4)
                pg_diag["t14_uncal_slope_r_add"] = round(sr_add, 4)
            pg_diag["t6_v4_uncal_refine"] = True

    if cfg.get("t7_hl_dark_guard_check") and p90 > 1e-4:
        lin_p98_ev = lin_q98_raw * (2.0 ** float(base_p["ev"]))
        if float(base_p["ev"]) > 0.35 and enc_med > 0.50 and enc_p2 > 0.25 and lin_q98_raw > 0.80 and lin_p98_ev > 1.65:
            ev_cap = float(max(0.30, np.log2(1.45 / lin_q98_raw)))
            if ev_cap < float(base_p["ev"]):
                pg_diag["t7_hl_ev_cap"] = round(float(base_p["ev"]) - ev_cap, 4)
                base_p["ev"] = ev_cap

    if cfg.get("t9_uncal_dark_ev_check") and p90 > 1e-4:
        dark_share = float(out.get("stats", {}).get("dark_share", 0.0))
        if float(base_p.get("ev", 0.0)) > 0.70 and dark_share > 0.04:
            pull = min(0.35, (float(base_p["ev"]) - 0.70) * 0.45)
            base_p["ev"] = round(float(base_p["ev"]) - pull, 4)
            pg_diag["t9_uncal_dark_ev_pull"] = round(pull, 4)

    if cfg.get("t9_uncal_wb_center_check") and p90 > 1e-4:
        sp = float(span_d_g) if span_d_g is not None else 0.80
        if cfg.get("t12_uncal_span_wb_check"):
            wb_slp = float(cfg.get("t13_uncal_wb_slope", 0.65))
            wb_mx = float(cfg.get("t13_uncal_wb_max", 0.30))
            d_temp = 0.05 + float(np.clip(wb_slp * (sp - 0.80), -0.15, wb_mx))
            t_b = float(cfg.get("t13_uncal_tint_base", 0.04))
            t_s = float(cfg.get("t13_uncal_tint_slope", 0.0))
            d_tint = t_b + float(np.clip(t_s * (sp - 0.80), -0.02, 0.08))
            pg_diag["t12_uncal_span_wb"] = round(d_temp, 4)
        else:
            d_temp, d_tint = 0.05, 0.04
        if cfg.get("t17_uncal_wb_deov_check") and wb_temp_init > float(cfg.get("t17_deov_pivot", 0.12)):
            damp_t = min(float(cfg.get("t17_deov_max", 0.11)), (wb_temp_init - float(cfg.get("t17_deov_pivot", 0.12))) * float(cfg.get("t17_deov_gain", 0.32)))
            d_temp -= damp_t
            pg_diag["t17_uncal_wb_deov"] = round(damp_t, 4)
        base_p["temp"] = float(np.clip(float(base_p.get("temp", 0.0)) + d_temp, *BOUNDS["temp"]))
        base_p["tint"] = float(np.clip(float(base_p.get("tint", 0.0)) + d_tint, *BOUNDS["tint"]))
        pg_diag["t9_uncal_wb_center"] = True

    if cfg.get("t12_uncal_span_ev_check") and p90 > 1e-4:
        sp = float(span_d_g) if span_d_g is not None else 0.80
        d_ev = float(np.clip(float(cfg.get("t13_uncal_ev_slope", 0.35)) * (sp - 0.80), -0.10, float(cfg.get("t13_uncal_ev_max", 0.20))))
        base_p["ev"] = float(np.clip(float(base_p.get("ev", 0.0)) + d_ev, *cfg["ev_limits"]))
        pg_diag["t12_uncal_span_ev"] = round(d_ev, 4)


_ROUND_SLIDER_KEYS = (
    "ev", "temp", "tint", "sat", "contrast", "midtone", "shoulder",
    "highlights", "shadows", "whites", "blacks", "vibrance", "slope_r", "slope_b",
)

_PRINT_GRADE_CHECKS = (
    "t5j_wb_crossover_check", "t5j_dye_unmix_check", "t5j_boyack_lowkey_check",
    "t5j_overlift_conv_check", "t5j_paper_white_check", "t5j_print_toe_bpc_check",
    "t5k_bpc_check", "t5k_soft_clip_check", "t5k_dye_unmix_v2_check",
    "t5k_blue_sat_wb_check", "t5k_uncal_fog_check",
    "t6_print_contrast_check", "t6_paper_white_release_check",
    "t6_shadow_toe_check", "t6_uncal_overlift_check",
    "t6_v2_cal_bpc_shoulder_check", "t6_v2_uncal_print_check",
    "t6_v3_cal_refine_check", "t6_v3_uncal_refine_check",
    "t6_v4_cal_refine_check", "t6_v4_uncal_refine_check",
    "t7_hl_dark_guard_check", "t8_uncal_mid_guard_check",
    "t8_cal_chroma_guard_check", "t8_uncal_bpc_guard_check",
    "t9_cal_deep_blk_ev_check", "t9_uncal_dark_ev_check", "t9_uncal_wb_center_check",
    "t10_cal_deep_blk_p2_guard_check", "t10_cal_highkey_guard_check",
    "t11_cal_bpc_damp_check", "t12_uncal_span_wb_check", "t12_uncal_span_ev_check",
    "t12_uncal_span_sat_check", "t12_uncal_lowkey_guard_check",
    "t13_uncal_paper_white_check", "t14_uncal_slope_check", "t15_cal_cnt_check",
)


def _apply_print_grade_v3(
    base_p: dict,
    px: np.ndarray,
    tc: dict,
    wb: dict,
    p90: float,
    span_d_g: float | None,
    bright_share: float,
    q25: float,
    w_bl: float,
    refine: dict,
    ev_lowered: float,
    spread: float,
    out: dict,
    cfg: dict,
    px_y_stats: tuple[float, float, float, float, float] | None = None,
) -> None:
    """Darkroom print grade orchestrator (T5j..T15) across WB/unmix, paper-white, CAL, and UNCAL branches."""
    if not any(cfg.get(k) for k in _PRINT_GRADE_CHECKS):
        return

    pre_params = RenderParams(**{
        **base_p,
        **{k: round(float(base_p[k]), 4) for k in _ROUND_SLIDER_KEYS},
    })
    if px_y_stats is None:
        px_y_stats = tuple(float(v) for v in np.quantile(px @ LUMA_F32, [0.02, 0.10, 0.50, 0.90, 0.98]))
    lin_q2_raw, lin_q10_raw, lin_q50_raw, lin_q90_raw, lin_q98_raw = px_y_stats
    enc_p2, _, enc_med, enc_p98 = _encoded_quantiles(px, pre_params, qs=(2, 25, 50, 98))

    wd = wb["diagnostics"]
    zs = wd.get("zones", [])
    gw = wd.get("grey_world", {}).get("chroma", [0.0, 0.0])
    has_cam_matrix = cfg.get("matrix") is not None
    p2_sh = 0.30 * max(0.0, float(cfg.get("base_shadows", 0.0)))
    enc_p2_eff = enc_p2 - p2_sh
    hl_unburned = abs(float(base_p["highlights"]) - float(cfg["base_highlights"])) <= 1e-4
    ol = out["diagnostics"].get("overlift_damp", {})
    thin_applied = bool(out["diagnostics"].get("thin_negative", {}).get("applied"))
    lifted_applied = bool(out["diagnostics"].get("color_lifted_floor"))
    pg_diag: dict = {}

    _apply_wb_and_unmix_v3(base_p, wd, zs, gw, p90, span_d_g, bright_share, wb["confidence"], has_cam_matrix, cfg, pg_diag)
    _apply_overlift_convergence_v3(
        base_p, tc, ol, refine, ev_lowered, w_bl, q25, bright_share, span_d_g,
        enc_p2_eff, enc_p98, spread, p90, has_cam_matrix, thin_applied, lifted_applied, cfg, pg_diag,
    )
    _apply_paper_white_and_toe_v3(
        base_p, tc, wd, span_d_g, bright_share, w_bl, q25, enc_p2_eff, enc_med, enc_p98,
        spread, hl_unburned, has_cam_matrix, thin_applied, lifted_applied, cfg, pg_diag,
    )

    enc_p2_post, enc_p98_post = _encoded_quantiles(px, RenderParams(**base_p), qs=(2, 98))
    enc_p2_post_eff = enc_p2_post - p2_sh
    if has_cam_matrix:
        _apply_cal_grade_v3(
            base_p, px, wd, zs, gw, span_d_g, spread, p90, wb["confidence"],
            enc_p2, enc_med, enc_p2_post_eff, enc_p98_post, lin_q2_raw, lin_q98_raw, cfg, pg_diag,
        )
    else:
        _apply_uncal_grade_v3(
            base_p, span_d_g, bright_share, q25, spread, p90, enc_p2, enc_p2_eff,
            enc_p2_post_eff, enc_med, enc_p98, lin_q10_raw, lin_q50_raw, lin_q90_raw, lin_q98_raw,
            out, cfg, pg_diag,
        )

    if pg_diag:
        out["diagnostics"]["print_grade_v3"] = pg_diag


def _apply_overlift_damp_v3(
    base_p: dict, refine: dict, thin: dict, span_d_g: float | None,
    bright_share: float, q25: float, w_bl: float, ev_lowered: float,
    p90: float, out: dict, cfg: dict,
) -> None:
    """T4q..T4z overlift damping across backlit, moderate, bright short-span, and thin-high-anchor modes."""
    # T4q: use Step 4b's shadow evidence after the backlit / highlight decisions, so their inputs stay unchanged
    def _apply_overlift(ev_piv, ev_rmp, q25_piv, gain, max_pull, mode=None, **extra):
        ev0_ol = float(base_p["ev"])
        ramp_ev = float(np.clip((ev0_ol - ev_piv) / ev_rmp, 0.0, 1.0))
        pull = float(np.clip(gain * (q25 - q25_piv) * ramp_ev, 0.0, max_pull))
        if pull > 0.0:
            blk_drop = float(min(cfg["overlift_blacks_max"], pull * cfg["overlift_blacks_gain"]))
            base_p["ev"] = float(max(cfg["ev_limits"][0], ev0_ol - pull))
            base_p["blacks"] = float(np.clip(base_p["blacks"] - blk_drop, *EXTENDED_BOUNDS["blacks"]))
            d = dict(applied=True, **({"mode": mode} if mode else {}), ev_before=ev0_ol, ev_after=base_p["ev"],
                     pull=pull, q25_before=float(q25), **extra, blacks_lowered=blk_drop)
            out["diagnostics"]["overlift_damp"] = d

    sh_comp = float(cfg.get("overlift_mod_shadow_comp", 0.35)) * max(0.0, float(cfg.get("base_shadows", 0.0)))
    mod_q25_min = float(cfg.get("overlift_mod_color_q25_min", 0.365)) if p90 > 1e-4 else float(cfg["overlift_mod_q25_min"])
    mod_q25_piv = float(cfg.get("overlift_mod_color_q25_pivot", 0.35)) if p90 > 1e-4 else float(cfg["overlift_mod_q25_pivot"])
    ol_wide_bright = (span_d_g is not None and span_d_g >= cfg.get("overlift_max_span", 1.15)
                      and bright_share >= cfg.get("overlift_max_bright_share", 0.05))
    q25_lo, q25_hi = cfg["overlift_bright_q25_band"]
    if cfg.get("overlift_bright_cap_check") and float(refine["ev_anchor_raw"]) - base_p["ev"] >= cfg.get("overlift_bright_min_anc_drop", 0.115):
        q25_lo = min(q25_lo, float(cfg.get("overlift_bright_cap_q25_min", 0.36)))
        q25_piv_br = float(cfg.get("overlift_bright_cap_q25_pivot", 0.34))
    else:
        q25_piv_br = float(cfg["overlift_bright_q25_pivot"])
    if (cfg.get("overlift_damp_check") and not ol_wide_bright and base_p["ev"] >= cfg["overlift_ev_min"]
            and q25 >= cfg["overlift_q25_min"]):
        if span_d_g is not None and span_d_g < cfg.get("overlift_short_span_max", 0.77):
            _apply_overlift(cfg.get("overlift_short_ev_pivot", 1.10), cfg.get("overlift_short_ev_ramp", 0.45),
                            cfg.get("overlift_short_q25_pivot", 0.25), cfg.get("overlift_short_gain", 4.2),
                            cfg.get("overlift_short_max_pull", 0.95))
        elif (span_d_g is not None and 0.85 <= span_d_g < 1.10 and w_bl >= cfg.get("overlift_bl_coupled_min_w", 0.20)
              and bright_share < 0.045 and float(refine["med_tex_before"]) <= 0.555):
            _apply_overlift(cfg.get("overlift_bl_coupled_ev_pivot", 1.10), cfg.get("overlift_bl_coupled_ev_ramp", 0.45),
                            cfg.get("overlift_bl_coupled_q25_pivot", 0.22), cfg.get("overlift_bl_coupled_gain", 3.8),
                            cfg.get("overlift_max_pull", 0.65))
        else:
            _apply_overlift(cfg["overlift_ev_pivot"], cfg["overlift_ev_ramp"], cfg["overlift_q25_pivot"],
                            cfg["overlift_gain"], cfg["overlift_max_pull"])
    elif (cfg.get("overlift_mod_check") and span_d_g is not None
          and span_d_g < cfg.get("overlift_max_span", 1.15)
          and bright_share < cfg.get("overlift_mod_max_bright_share", 0.20)
          and base_p["ev"] >= cfg["overlift_mod_ev_min"]
          and q25 >= (float(cfg.get("overlift_mod_hi_q25_min", 0.40))
                      if base_p["ev"] >= cfg.get("overlift_mod_hi_ev_min", 1.35)
                      else mod_q25_min) + sh_comp):
        if base_p["ev"] >= cfg.get("overlift_mod_hi_ev_min", 1.35):
            _apply_overlift(cfg.get("overlift_mod_hi_ev_pivot", 0.58), cfg.get("overlift_mod_hi_ev_ramp", 0.55),
                            float(cfg.get("overlift_mod_hi_q25_pivot", 0.38)) + sh_comp, cfg.get("overlift_mod_hi_gain", 4.0),
                            cfg.get("overlift_mod_hi_max_pull", 0.45), mode="moderate")
        else:
            _apply_overlift(cfg["overlift_mod_ev_pivot"], cfg["overlift_mod_ev_ramp"],
                            mod_q25_piv + sh_comp, cfg["overlift_mod_gain"],
                            cfg["overlift_mod_max_pull"], mode="moderate")
    elif (cfg.get("overlift_bright_check") and span_d_g is not None
          and span_d_g < cfg["overlift_bright_span_max"]
          and bright_share >= cfg["overlift_bright_min_share"]
          and base_p["ev"] >= cfg["overlift_bright_ev_min"]
          and float(refine["ev_anchor_raw"]) <= cfg["overlift_bright_anc_max"]
          and (float(refine["med_tex_before"]) >= cfg["overlift_bright_med0_min"]
               or (cfg.get("overlift_bright_cap_check")
                   and float(refine["ev_anchor_raw"]) - base_p["ev"] >= cfg.get("overlift_bright_min_anc_drop", 0.115)
                   and float(refine["med_tex_before"]) >= cfg.get("overlift_bright_cap_med0_min", 0.41)))
          and (q25_lo + sh_comp) <= q25 <= (q25_hi + sh_comp)):
        _apply_overlift(cfg["overlift_bright_ev_pivot"], cfg["overlift_bright_ev_ramp"],
                        q25_piv_br + sh_comp, cfg["overlift_bright_gain"],
                        min(cfg["overlift_bright_max_pull"], float(base_p["ev"])), mode="bright_short_span")
    elif (cfg.get("overlift_night_bl_check")
          and w_bl >= cfg["overlift_night_bl_min_weight"]
          and bright_share < cfg["overlift_night_bl_max_bright_share"]
          and base_p["ev"] >= cfg["overlift_night_bl_ev_min"]
          and ev_lowered == 0.0
          and (q25 >= cfg["overlift_night_bl_q25_min"]
               or (w_bl >= cfg.get("overlift_night_bl_hi_w_min", 0.95)
                   and base_p["ev"] >= cfg.get("overlift_night_bl_hi_ev_min", 1.60)
                   and q25 >= cfg.get("overlift_night_bl_hi_q25_min", 0.08)))):
        q25_piv_nb = (cfg["overlift_night_bl_q25_pivot"]
                      if q25 >= cfg["overlift_night_bl_q25_min"]
                      else cfg.get("overlift_night_bl_hi_q25_pivot", 0.06))
        _apply_overlift(cfg["overlift_night_bl_ev_pivot"], cfg["overlift_night_bl_ev_ramp"],
                        q25_piv_nb, cfg["overlift_night_bl_gain"],
                        cfg["overlift_night_bl_max_pull"], mode="night_backlit_double", w_bl=float(w_bl))
    elif (cfg.get("overlift_mid_bl_check")
          and w_bl >= cfg.get("overlift_mid_bl_min_weight", 0.85)
          and bright_share < cfg.get("overlift_mid_bl_max_bright_share", 0.090)
          and base_p["ev"] >= cfg.get("overlift_mid_bl_ev_min", 1.20)
          and ev_lowered == 0.0
          and float(refine["med_tex_before"]) >= cfg.get("overlift_mid_bl_med0_min", 0.50)
          and q25 >= cfg.get("overlift_mid_bl_q25_min", 0.18)):
        _apply_overlift(cfg.get("overlift_mid_bl_ev_pivot", 1.05), cfg.get("overlift_mid_bl_ev_ramp", 0.45),
                        cfg.get("overlift_mid_bl_q25_pivot", 0.15), cfg.get("overlift_mid_bl_gain", 3.6),
                        cfg.get("overlift_mid_bl_max_pull", 0.45), mode="mid_backlit_double", w_bl=float(w_bl))
    elif (cfg.get("overlift_short_bl_check")
          and span_d_g is not None and span_d_g < cfg.get("overlift_short_bl_span_max", 0.85)
          and w_bl >= cfg.get("overlift_short_bl_min_weight", 0.50)
          and base_p["ev"] >= cfg.get("overlift_short_bl_ev_min", 0.48)
          and float(refine["ev_anchor_raw"]) >= cfg.get("overlift_short_bl_anc_min", 1.05)
          and float(refine["med_tex_before"]) <= cfg.get("overlift_short_bl_med0_max", 0.55)
          and q25 >= cfg.get("overlift_short_bl_q25_min", 0.12)):
        ev0_ol = float(base_p["ev"])
        ramp_ev = float(np.clip((ev0_ol - cfg.get("overlift_short_bl_ev_pivot", 0.30)) / cfg.get("overlift_short_bl_ev_ramp", 0.40), 0.0, 1.0))
        ramp_sp = float(np.clip((cfg.get("overlift_short_bl_span_pivot", 0.90) - span_d_g) / cfg.get("overlift_short_bl_span_ramp", 0.22), 0.25, 1.0))
        pull = float(np.clip(cfg.get("overlift_short_bl_gain", 0.95) * w_bl * ramp_ev * ramp_sp, 0.0, min(cfg.get("overlift_short_bl_max_pull", 0.80), ev0_ol)))
        if pull > 0.0:
            blk_drop = float(min(cfg["overlift_blacks_max"], pull * cfg["overlift_blacks_gain"]))
            mid_damp = float(cfg.get("overlift_short_bl_mid_damp", 0.0))
            if mid_damp > 0.0 and base_p["midtone"] > cfg["base_midtone"]:
                base_p["midtone"] = float(cfg["base_midtone"] + (1.0 - mid_damp * ramp_sp) * (base_p["midtone"] - cfg["base_midtone"]))
            base_p["ev"] = float(max(cfg["ev_limits"][0], ev0_ol - pull))
            base_p["blacks"] = float(np.clip(base_p["blacks"] - blk_drop, *EXTENDED_BOUNDS["blacks"]))
            out["diagnostics"]["overlift_damp"] = dict(
                applied=True, mode="short_span_backlit", ev_before=ev0_ol, ev_after=base_p["ev"],
                pull=pull, q25_before=float(q25), w_bl=float(w_bl), blacks_lowered=blk_drop
            )
    elif (cfg.get("overlift_med_span_bl_check") and p90 > 1e-4
          and span_d_g is not None
          and cfg.get("overlift_med_span_bl_span_min", 0.85) <= span_d_g < cfg.get("overlift_med_span_bl_span_max", 1.10)
          and ev_lowered == 0.0
          and ((cfg.get("overlift_med_span_bl_min_bright_share", 0.10) <= bright_share < cfg.get("overlift_med_span_bl_max_bright_share", 0.35)
                and w_bl >= cfg.get("overlift_med_span_bl_min_weight", 0.78)
                and cfg.get("overlift_med_span_bl_ev_min", 0.70) <= base_p["ev"] < cfg.get("overlift_med_span_bl_ev_max", 1.75)
                and float(refine["ev_anchor_raw"]) >= cfg.get("overlift_med_span_bl_anc_min", 1.45)
                and float(refine["med_tex_before"]) <= cfg.get("overlift_med_span_bl_med0_max", 0.49)
                and q25 >= cfg.get("overlift_med_span_bl_q25_min", 0.16))
               or (0.015 <= bright_share < 0.045
                   and 0.28 <= w_bl < cfg.get("overlift_med_span_bl_min_weight", 0.78)
                   and base_p["ev"] >= 1.50
                   and float(refine["ev_anchor_raw"]) >= 1.65
                   and float(refine["med_tex_before"]) <= 0.555
                   and q25 >= 0.25))):
        ev0_ol = float(base_p["ev"])
        ramp_ev = float(np.clip((ev0_ol - cfg.get("overlift_med_span_bl_ev_pivot", 0.60)) / cfg.get("overlift_med_span_bl_ev_ramp", 0.45), 0.0, 1.0))
        gain_ms = cfg.get("overlift_med_span_bl_gain", 0.40) if w_bl >= cfg.get("overlift_med_span_bl_min_weight", 0.78) else 0.90
        max_p_ms = cfg.get("overlift_med_span_bl_max_pull", 0.36) if w_bl >= cfg.get("overlift_med_span_bl_min_weight", 0.78) else 0.42
        pull = float(np.clip(gain_ms * w_bl * ramp_ev, 0.0, min(max_p_ms, ev0_ol)))
        if pull > 0.0:
            blk_drop = float(min(cfg["overlift_blacks_max"], pull * cfg["overlift_blacks_gain"]))
            mid_damp = float(cfg.get("overlift_med_span_bl_mid_damp", 0.45))
            if mid_damp > 0.0 and base_p["midtone"] > cfg["base_midtone"]:
                base_p["midtone"] = float(cfg["base_midtone"] + (1.0 - mid_damp) * (base_p["midtone"] - cfg["base_midtone"]))
            base_p["ev"] = float(max(cfg["ev_limits"][0], ev0_ol - pull))
            base_p["blacks"] = float(np.clip(base_p["blacks"] - blk_drop, *EXTENDED_BOUNDS["blacks"]))
            out["diagnostics"]["overlift_damp"] = dict(
                applied=True, mode="med_span_backlit", ev_before=ev0_ol, ev_after=base_p["ev"],
                pull=pull, q25_before=float(q25), w_bl=float(w_bl), blacks_lowered=blk_drop
            )
    elif (cfg.get("thin_neg_high_anc_check") and thin.get("applied")
          and float(refine["ev_anchor_raw"]) >= cfg.get("thin_high_anc_min", 1.50)
          and float(refine["med_tex_before"]) <= cfg.get("thin_high_anc_med0_max", 0.46)
          and q25 >= cfg.get("thin_high_anc_q25_min", 0.08)):
        ev0_tn = float(base_p["ev"])
        pull_tn = float(np.clip((float(refine["ev_anchor_raw"]) - cfg.get("thin_high_anc_pivot", 1.30)) * cfg.get("thin_high_anc_ev_gain", 0.35),
                                0.0, cfg.get("thin_high_anc_max_pull", 0.40)))
        if pull_tn > 0.0:
            blk_tn = float(pull_tn * cfg.get("thin_high_anc_blacks_gain", 0.14))
            base_p["ev"] = float(max(cfg["ev_limits"][0], ev0_tn - pull_tn))
            base_p["blacks"] = float(np.clip(base_p["blacks"] - blk_tn, *EXTENDED_BOUNDS["blacks"]))
            thin.update(high_anc_pull=pull_tn, ev_after_high_anc=base_p["ev"], blacks_lowered_high_anc=blk_tn)


def estimate_params_v3(P, mask=None, cfg=NOREF_V3, roll=None, look=None, *, span_d_g: float | None = None) -> dict:
    """Textured-cell adaptive print estimator (NOREF_V3)."""
    P = np.asarray(P, np.float32)
    if mask is not None and mask.shape != P.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} != image shape {P.shape[:2]}")
    look = look or load_look(cfg["look"])
    mask, metering = _metering_mask(P, mask, cfg)
    st = frame_stats(P, mask, cfg)
    if (st["range_stops"] < cfg["low_info_range_stops"]
            or (cfg.get("min_span_d_g") is not None and span_d_g is not None and span_d_g < cfg["min_span_d_g"])
            or st["p99"] > 2 ** cfg["wp_invalid_stops"]):
        return _estimate_from_features_v3(np.empty((0, 3), np.float32), {}, st, metering, cfg, look=look, span_d_g=span_d_g)
    px = _sample(P, mask, cfg["n_sample"] // 4, seed=1)
    tc = _textured_cells(P, mask, cfg)
    return _estimate_from_features_v3(px, tc, st, metering, cfg, look=look, span_d_g=span_d_g)


def _bisect_param(lo: float, hi: float, pred, steps: int = 18) -> float:
    """Largest value in [lo, hi] where monotone predicate ``pred(val)`` holds."""
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if pred(mid) else (lo, mid)
    return float(lo)


def _refine_ev_v3(base_p: dict, tex_px: np.ndarray, px_tone: np.ndarray, anc_d: float,
                  cfg: dict, span_d_g: float | None) -> tuple[dict, float]:
    t_lo, t_hi = cfg["target_encoded_band"]
    p98_max = float(cfg["ev_refine_p98_max"])
    p98_ok = lambda ev: _encoded_quantiles(px_tone, RenderParams(**dict(base_p, ev=ev)), qs=(98,))[0] <= p98_max

    refine = dict(ev_anchor_raw=base_p["ev"])
    if cfg.get("ev_anchor_p98_cap") and base_p["ev"] > 0 and not p98_ok(base_p["ev"]):
        base_p["ev"] = _bisect_param(0.0, base_p["ev"], p98_ok)
        refine.update(anchor_capped_by_p98=True)
    ev0 = base_p["ev"]
    med_tex = _encoded_quantiles(tex_px, RenderParams(**base_p), qs=(50,))[0]
    t_hi_eff = 0.76 if anc_d < 0.65 else t_hi
    refine.update(ev_anchor=ev0, med_tex_before=med_tex)

    if med_tex < t_lo:
        tgt = t_lo + 0.03
        ev_up = _bisect_param(ev0, cfg["ev_limits"][1],
                              lambda ev: _encoded_quantiles(tex_px, RenderParams(**dict(base_p, ev=ev)), qs=(50,))[0] < tgt)
        if not p98_ok(ev_up):
            ev_up = _bisect_param(ev0, ev_up, p98_ok)
            refine.update(capped_by_p98=True)
        base_p["ev"] = float(max(ev0, ev_up))
    elif med_tex > t_hi_eff:
        tgt, dmax = t_hi_eff - 0.02, float(cfg["ev_refine_down_max"])
        if cfg.get("t16_cal_span_ev_down_check") and cfg.get("matrix") is not None and span_d_g is not None:
            sp_lo = float(cfg.get("t16_cal_span_lo", 0.82))
            sp_w = float(cfg.get("t16_cal_span_width", 0.28))
            t_sp = float(np.clip((float(span_d_g) - sp_lo) / sp_w, 0.0, 1.0)) if sp_w > 0 else 0.0
            dm_lo = float(cfg.get("t16_cal_ev_down_min", 0.08))
            dmax = dmax - t_sp * (dmax - dm_lo)
            refine.update(t16_cal_ev_down_max=round(dmax, 4))
        if cfg.get("ev_down_adaptive"):
            q25_tex = _encoded_quantiles(tex_px, RenderParams(**base_p), qs=(25,))[0]
            dmax += float(cfg["ev_down_extra"]) * _ramp(q25_tex, *cfg["ev_down_q25"])
            refine.update(q25_tex=q25_tex, down_max=dmax)
        floor = min(ev0, max(ev0 - dmax, -dmax))
        base_p["ev"] = _bisect_param(floor, ev0,
                                     lambda ev: _encoded_quantiles(tex_px, RenderParams(**dict(base_p, ev=ev)), qs=(50,))[0] <= tgt)
    base_p["ev"] = float(np.clip(base_p["ev"], *cfg["ev_limits"]))
    refine.update(ev=base_p["ev"], med_tex_after=_encoded_quantiles(tex_px, RenderParams(**base_p), qs=(50,))[0])
    return refine, float(p98_max)


def _apply_thin_and_flat_v3(base_p: dict, st: dict, tc: dict, q25: float, q90: float,
                            span_d_g: float | None, bright_share: float, out: dict, cfg: dict) -> dict:
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

    iqr_25_90 = float(q90 - q25)
    if (cfg.get("thin_flat_check") and span_d_g is not None and not thin.get("applied")
            and span_d_g < cfg["thin_flat_span_max"] and base_p["ev"] <= 0.0
            and q25 >= cfg["thin_flat_q25_min"] and iqr_25_90 <= cfg["thin_flat_iqr_max"]):
        pull_f = float(np.clip((cfg["thin_flat_iqr_pivot"] - iqr_25_90) * cfg["thin_flat_ev_gain"],
                               0.0, max(0.0, base_p["ev"] - cfg["thin_flat_ev_min"])))
        if pull_f > 0.0:
            ev_b4 = float(base_p["ev"])
            blk_f = float(min(cfg["thin_flat_blacks_max"], pull_f * 0.45))
            base_p["ev"] = float(ev_b4 - pull_f)
            base_p["blacks"] = float(np.clip(base_p["blacks"] - blk_f, *EXTENDED_BOUNDS["blacks"]))
            thin.update(applied=True, mode="flat_bright", ev_before=ev_b4, ev_after=base_p["ev"],
                        pull=pull_f, blacks_lowered=blk_f, q25=float(q25), iqr_25_90=iqr_25_90)
            out["diagnostics"]["thin_negative"] = thin
    elif (cfg.get("ultra_flat_check") and st["range_stops"] >= cfg["ultra_flat_min_range_stops"]
          and base_p["ev"] <= 0.0 and q25 >= cfg["ultra_flat_q25_min"]
          and iqr_25_90 <= cfg["ultra_flat_iqr_max"]):
        pull_uf = float(np.clip((cfg["ultra_flat_iqr_pivot"] - iqr_25_90) * cfg["ultra_flat_ev_gain"],
                                0.0, cfg["ultra_flat_max_pull"]))
        if pull_uf > 0.0:
            ev_b4 = float(base_p["ev"])
            blk_uf = float(min(cfg["ultra_flat_blacks_max"], pull_uf * cfg["ultra_flat_blacks_gain"]))
            c_uf = float(min(cfg["ultra_flat_contrast_cap"],
                             base_p["contrast"] + pull_uf * cfg["ultra_flat_contrast_gain"]))
            base_p["ev"] = float(max(cfg["ev_limits"][0], ev_b4 - pull_uf))
            base_p["blacks"] = float(np.clip(base_p["blacks"] - blk_uf, *EXTENDED_BOUNDS["blacks"]))
            base_p["contrast"] = c_uf
            out["diagnostics"]["ultra_flat_bright"] = dict(
                applied=True, ev_before=ev_b4, ev_after=base_p["ev"], pull=pull_uf,
                blacks_lowered=blk_uf, contrast_after=c_uf, q25=float(q25), iqr_25_90=iqr_25_90)
    elif (cfg.get("lifted_floor_check") and span_d_g is not None and not thin.get("applied")
          and span_d_g < cfg["lifted_floor_span_max"] and st["range_stops"] >= cfg["ultra_flat_min_range_stops"]
          and base_p["ev"] <= cfg["lifted_floor_ev_max"]
          and tc["p98_dark_d"] <= cfg["lifted_floor_p98_dark_d_max"]):
        pull_lf = float(np.clip((cfg["lifted_floor_p98_dark_d_pivot"] - tc["p98_dark_d"]) * cfg["lifted_floor_ev_gain"],
                                0.0, cfg["lifted_floor_max_pull"]))
        if pull_lf > 0.0:
            ev_b4 = float(base_p["ev"])
            blk_lf = float(min(cfg["lifted_floor_blacks_max"], pull_lf * cfg["lifted_floor_blacks_gain"]))
            c_lf = float(min(cfg["lifted_floor_contrast_cap"],
                             base_p["contrast"] + pull_lf * cfg["lifted_floor_contrast_gain"]))
            base_p["ev"] = float(max(cfg["ev_limits"][0], ev_b4 - pull_lf))
            base_p["blacks"] = float(np.clip(base_p["blacks"] - blk_lf, *EXTENDED_BOUNDS["blacks"]))
            base_p["contrast"] = c_lf
            out["diagnostics"]["lifted_floor"] = dict(
                applied=True, ev_before=ev_b4, ev_after=base_p["ev"], pull=pull_lf,
                blacks_lowered=blk_lf, contrast_after=c_lf, p98_dark_d=float(tc["p98_dark_d"]))
    return thin


def _apply_backlit_v3(base_p: dict, px_tone: np.ndarray, q25: float, q90: float,
                      thin: dict, span_d_g: float | None, bright_share: float,
                      out: dict, cfg: dict, flags: list[str]) -> float:
    rmp = float(cfg["backlit_ramp"])
    rmp25 = float(cfg.get("backlit_ramp_p25", rmp))
    w_bl = _ramp(q90, cfg["backlit_p90"], cfg["backlit_p90"] + rmp) * _ramp(cfg["backlit_p25"] - q25, 0.0, rmp25)
    out["diagnostics"]["backlit_check"] = dict(p25=q25, p90=q90, weight=w_bl)
    if cfg.get("backlit_sky_check") and w_bl > 0:
        w_bl *= _ramp(bright_share, cfg["backlit_sky_share_lo"], cfg["backlit_min_sky_share"])
        out["diagnostics"]["backlit_check"].update(weight=w_bl, bright_share=bright_share)
    if thin.get("applied") and span_d_g < cfg.get("thin_suppress_backlit_span", 0.58):
        out["diagnostics"]["backlit_check"].update(thin_suppressed=True, weight_before_thin=w_bl, weight=0.0)
        w_bl = 0.0
    if w_bl > 0:
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
            for _ in range(14):
                mid_m = 0.5 * (lo_m + hi_m)
                if _encoded_quantiles(px_tone, RenderParams(**dict(base_p, midtone=mid_m)), qs=(25,))[0] < cfg["backlit_p25_target"]:
                    lo_m = mid_m
                else:
                    hi_m = mid_m
            m_t = float(hi_m)
        base_p["midtone"] = float(m0 + w_bl * (m_t - m0))
        out["diagnostics"]["backlit_check"].update(midtone_target=m_t)
        flags.append("backlit_midtone")
    return float(w_bl)


def _apply_highlight_hold_v3(base_p: dict, px: np.ndarray, tc: dict, out: dict, cfg: dict, flags: list[str]) -> float:
    p2_shifted_d = tc["p2_bright_d"] - 0.30103 * base_p["ev"]
    if p2_shifted_d < 0.12:
        base_p["highlights"] = min(base_p["highlights"], float(np.clip(-1.8 * (0.12 - p2_shifted_d), -0.65, 0.0)))

    s_grid = np.linspace(base_p["shoulder"], 0.85, 15)
    hl_grid = sorted({round(base_p["highlights"], 3), -0.20, -0.40, -0.65}, reverse=True)
    found = False
    ev_lowered = 0.0
    budget = float(cfg["clip_budget"])
    max_drop = float(cfg.get("ev_highlight_max_drop", 3.0))
    if _clip_share is _ORIG_CLIP_SHARE:
        x_max0 = _pre_shoulder_max(px, base_p)
        whites_pos = max(0.0, -0.25 * float(base_p["whites"]))
        for hl in hl_grid:
            if hl > base_p["highlights"]:
                continue
            hl_term = max(0.0, -0.45 * float(hl)) + whites_pos
            for s in s_grid:
                eff_s = float(np.clip(float(s) + hl_term, 0.0, 1.0))
                if float(np.mean(x_max0 > np.float32(1.0 + 7.0 * eff_s))) <= budget:
                    base_p["shoulder"], base_p["highlights"] = float(s), float(hl)
                    found = True
                    break
            if found:
                break
        if not found:
            base_p["shoulder"], base_p["highlights"] = 0.85, -0.65
            eff_s_max = float(np.clip(0.85 + max(0.0, -0.45 * (-0.65)) + whites_pos, 0.0, 1.0))
            lw_max = 1.0 + 7.0 * eff_s_max
            ev_init = float(base_p["ev"])
            for _ in range(30):
                if not base_p.get("vibrance"):
                    scale = np.float32(2.0 ** (float(base_p["ev"]) - ev_init))
                    cs = float(np.mean(x_max0 * scale > np.float32(lw_max)))
                else:
                    cs = _clip_share(px, RenderParams(**base_p))
                if cs <= budget or base_p["ev"] <= cfg["ev_limits"][0] or ev_lowered + 1e-9 >= max_drop:
                    break
                step = min(0.1, max(0.0, max_drop - ev_lowered))
                if step <= 1e-9:
                    break
                base_p["ev"] = float(max(base_p["ev"] - step, cfg["ev_limits"][0]))
                ev_lowered = round(ev_lowered + step, 4)
    else:
        for hl in hl_grid:
            if hl > base_p["highlights"]:
                continue
            for s in s_grid:
                if _clip_share(px, RenderParams(**dict(base_p, shoulder=float(s), highlights=float(hl)))) <= budget:
                    base_p["shoulder"], base_p["highlights"] = float(s), float(hl)
                    found = True
                    break
            if found:
                break
        if not found:
            base_p["shoulder"], base_p["highlights"] = 0.85, -0.65
            for _ in range(30):
                if (_clip_share(px, RenderParams(**base_p)) <= budget
                        or base_p["ev"] <= cfg["ev_limits"][0] or ev_lowered + 1e-9 >= max_drop):
                    break
                step = min(0.1, max(0.0, max_drop - ev_lowered))
                if step <= 1e-9:
                    break
                base_p["ev"] = float(max(base_p["ev"] - step, cfg["ev_limits"][0]))
                ev_lowered = round(ev_lowered + step, 4)
    out["diagnostics"]["highlight_hold"] = dict(ev_lowered=ev_lowered, max_drop=max_drop)
    if ev_lowered > 0:
        flags.append(f"ev_lowered_for_highlights:{ev_lowered:.1f}")
    return ev_lowered


def _apply_backlit_rescue_v3(base_p: dict, px: np.ndarray, px_tone: np.ndarray, tex_px: np.ndarray,
                             st: dict, refine: dict, w_bl: float, q25: float, q90: float,
                             ev_lowered: float, span_d_g: float | None, p98_max: float,
                             out: dict, cfg: dict, flags: list[str]) -> dict:
    if not (w_bl == 0.0 and cfg.get("backlit_capped_rescue") and span_d_g is not None):
        return base_p
    med_after = (min(float(refine["med_tex_after"]), _encoded_quantiles(tex_px, RenderParams(**base_p), qs=(50,))[0])
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
    c3 = (span_d_g >= 1.50 and st["dark_share"] >= 0.30 and q25_res < q25_lo
          and 0.15 <= q90_res < cfg["backlit_capped_q90_min"] and med_after >= 0.50)
    if c1 or c2 or c3:
        before_rescue = dict(base_p)
        w_res = (0.85 if c3 else min(0.85, max((q90_res - 0.44) / 0.16, 0.40)) * min(1.0, (0.26 - q25_res) / 0.08))
        if c3 and _encoded_quantiles(px_tone, RenderParams(**dict(base_p, ev=base_p["ev"] + 0.65)), qs=(98,))[0] <= p98_max:
            base_p["ev"] = float(min(cfg["ev_limits"][1], base_p["ev"] + 0.65))
        base_p["midtone"] = float(min(cfg["backlit_midtone_cap"], base_p["midtone"] + w_res * (0.12 if c3 else 0.08)))
        base_p["shoulder"] = float(min(cfg["backlit_shoulder_cap"], base_p["shoulder"] + w_res * 0.12))
        base_p["shadows"] = float(max(base_p["shadows"], 0.35 if c3 else w_res * float(cfg["backlit_rescue_shadows_step"])))
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
        if c3:
            out["diagnostics"]["backlit_check"]["rescue_c3"] = True
        if w_res > 0:
            flags.append("backlit_midtone")
    return base_p


def _apply_contrast_and_shadows_v3(base_p: dict, px: np.ndarray, tc: dict, backlit: bool,
                                   p90: float, bright_share: float, span_d_g: float | None,
                                   c_lo: float, out: dict, cfg: dict, auto_lv: dict, flags: list[str]) -> tuple[float, float, float]:
    def ends(p):
        e, _, _, _ = render_fast_full(px, RenderParams(**p))
        y = e @ LUMA_F32
        q10, q90_y = np.quantile(y, [0.10, 0.90])
        return float(q90_y - q10), float(np.mean(y <= np.float32(0.02))), float(np.mean(y >= np.float32(0.98)))

    spread, black_end, white_end = ends(base_p)
    out["diagnostics"]["contrast_check"] = dict(spread_10_90=spread, black_end=black_end, white_end=white_end)
    if not backlit and spread > cfg["contrast_spread_max"] and min(black_end, white_end) > cfg["end_clip_share"]:
        lo_c, hi_c = c_lo, base_p["contrast"]
        if ends(dict(base_p, contrast=lo_c))[0] > cfg["contrast_spread_max"]:
            base_p["contrast"] = lo_c
        else:
            base_p["contrast"] = _bisect_param(
                lo_c, hi_c, lambda c: ends(dict(base_p, contrast=c))[0] <= cfg["contrast_spread_max"], steps=14,
            )
        flags.append("contrast_lowered_wide_frame")
    if cfg.get("auto_contrast") and spread < cfg["auto_contrast_spread"]:
        inc = float(min(cfg["auto_contrast_max"], cfg["auto_contrast_gain"] * (cfg["auto_contrast_spread"] - spread)))
        base_p["contrast"] = float(min(BOUNDS["contrast"][1], base_p["contrast"] + inc))
        auto_lv["contrast"] = inc

    p2, p25, p50, p98 = _encoded_quantiles(px, RenderParams(**base_p), qs=(2, 25, 50, 98))
    out["diagnostics"]["encoded_p25_before_shadows"] = p25
    if p25 < cfg["shadow_p25_floor"]:
        base_p["shadows"] = max(base_p["shadows"], float(np.clip(
            cfg["shadow_gain"] * (cfg["shadow_p25_floor"] - p25), 0.0, cfg["shadow_max"])))
    if cfg.get("auto_blacks"):
        p2_b = _encoded_quantiles(px, RenderParams(**base_p), qs=(2,))[0]
        if p2_b > cfg["auto_blacks_p2"]:
            ab = -float(np.clip((p2_b - cfg["auto_blacks_offset"]) * cfg["auto_blacks_gain"], 0.0, cfg["auto_blacks_max"]))
            bound = float(cfg["auto_blacks_bound"])
            base_p["blacks"] = float(np.clip(base_p["blacks"] + ab, -bound, bound))
            auto_lv["blacks"] = dict(p2=p2_b, blacks=ab)
    elif (p90 <= 1e-4 and span_d_g is not None and span_d_g < 0.80 and base_p["ev"] <= 0.35
          and tc["p98_dark_d"] <= 1.32 and p2 > 0.25):
        ex_p2 = float(p2 - 0.22)
        ev_drop = float(np.clip(ex_p2 * 1.40, 0.0, 0.45))
        blk_drop = float(np.clip(ex_p2 * 1.00, 0.0, 0.28))
        con_boost = float(np.clip(ex_p2 * 0.80, 0.0, 0.25))
        base_p["ev"] = float(max(cfg["ev_limits"][0], base_p["ev"] - ev_drop))
        base_p["blacks"] = float(np.clip(base_p["blacks"] - blk_drop, *EXTENDED_BOUNDS["blacks"]))
        base_p["contrast"] = float(min(1.65, base_p["contrast"] + con_boost))
        out["diagnostics"]["bw_lifted_floor"] = dict(p2_before=float(p2), ev_drop=ev_drop, blacks_lowered=blk_drop, contrast_boost=con_boost)
    elif (cfg.get("color_lifted_floor_check") and p90 > 1e-4
          and span_d_g is not None and span_d_g < cfg.get("color_lifted_floor_span_max", 0.79)
          and bright_share >= cfg.get("color_lifted_floor_min_bright_share", 0.20)
          and base_p["ev"] <= cfg.get("color_lifted_floor_ev_max", 0.10)
          and tc["p98_dark_d"] >= cfg.get("color_lifted_floor_min_p98_dark_d", 0.90)
          and (span_d_g < cfg.get("color_lifted_floor_short_span", 0.56)
               or tc["anchor_d"] >= cfg.get("color_lifted_floor_min_anchor_d", 0.45))
          and p2 >= cfg.get("color_lifted_floor_p2_min", 0.29) + max(0.0, float(cfg["base_shadows"])) * float(cfg.get("color_lifted_floor_shadow_comp", 0.30))):
        p2_sh_comp = max(0.0, float(cfg["base_shadows"])) * float(cfg.get("color_lifted_floor_shadow_comp", 0.30))
        ex_p2 = float(p2 - (cfg.get("color_lifted_floor_p2_pivot", 0.26) + p2_sh_comp))
        ev_drop = float(np.clip(ex_p2 * cfg.get("color_lifted_floor_ev_gain", 2.20), 0.0, cfg.get("color_lifted_floor_ev_max_drop", 0.32)))
        blk_drop = float(np.clip(ex_p2 * cfg.get("color_lifted_floor_blacks_gain", 1.40), 0.0, cfg.get("color_lifted_floor_blacks_max", 0.22)))
        con_boost = float(np.clip(ex_p2 * cfg.get("color_lifted_floor_contrast_gain", 1.00), 0.0, cfg.get("color_lifted_floor_contrast_max", 0.18)))
        base_p["ev"] = float(max(cfg["ev_limits"][0], base_p["ev"] - ev_drop))
        base_p["blacks"] = float(np.clip(base_p["blacks"] - blk_drop, *EXTENDED_BOUNDS["blacks"]))
        base_p["contrast"] = float(min(BOUNDS["contrast"][1], base_p["contrast"] + con_boost))
        out["diagnostics"]["color_lifted_floor"] = dict(p2_before=float(p2), ev_drop=ev_drop, blacks_lowered=blk_drop, contrast_boost=con_boost)
    return spread, p2, p98


def _estimate_from_features_v3(
    px: np.ndarray, tc: dict, st: dict, metering: dict | None,
    cfg: dict = NOREF_V3, look: dict | None = None, *, span_d_g: float | None = None,
) -> dict:
    look = look or load_look(cfg["look"])
    lp = dict(look["params"])
    out = dict(config=cfg["id"], config_rev=cfg.get("rev"), preset=cfg.get("preset"), look=look["id"],
               groups=["brightness", "wb", "tone", "sat"], stats=st, diagnostics={})
    if metering is not None:
        out["diagnostics"]["metering"] = metering
    reasons = []
    if st["range_stops"] < cfg["low_info_range_stops"]:
        reasons.append(f"low_information: tonal range {st['range_stops']:.2f} stops")
    if cfg.get("min_span_d_g") is not None and span_d_g is not None and span_d_g < cfg["min_span_d_g"]:
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
        matrix=cfg.get("matrix") if cfg.get("matrix") is not None else IDENTITY, chroma_preserve=float(cfg.get("chroma_preserve", 0.0)),
    )
    if reasons:
        return dict(out, status="manual_review_required", params=RenderParams(**base_p).to_dict(),
                    flags=reasons, confidence=dict(overall=0.0))

    flags = ["white_point_unrepresentative"] if st["above_white"] > 0.05 else []
    px_tone = px
    if cfg.get("tone_border_filter"):
        lit_px = px[px.max(axis=-1) > cfg["tone_min_luma"]]
        if len(lit_px) >= cfg["wb_v3_min_px"]:
            px_tone = lit_px
    bright_share = float(np.mean((px_tone @ LUMA_F32) > cfg.get("bright_luma", 0.45)))
    out["diagnostics"]["tone_pixels"] = dict(n=int(len(px_tone)), n_all=int(len(px)), bright_share=bright_share)
    out["diagnostics"]["textured_cells"] = {k: v for k, v in tc.items() if k != "textured_rgb"}

    wb = estimate_wb_v3(px, cfg, span_d_g=span_d_g)
    if cfg.get("wb_bright_cool_damp_check") and wb["diagnostics"].get("cool_cast"):
        wd = wb["diagnostics"]
        br_wb = float(wd.get("bright_px_share", 0.0))
        ns0 = float(wd["zones"][0]["narrow_share"]) if len(wd.get("zones", [])) > 0 else 0.0
        z1_b = float(wd["zones"][1]["chroma"][1]) if len(wd.get("zones", [])) > 1 and "chroma" in wd["zones"][1] else 1.0
        gw_r = float(wd.get("grey_world", {}).get("chroma", [0.0, 0.0])[0])
        if (br_wb >= cfg.get("wb_bright_cool_min_br", 0.20)
                and ns0 >= cfg.get("wb_bright_cool_min_ns0", 0.15)
                and z1_b <= cfg.get("wb_bright_cool_max_z1_b", 0.22)
                and gw_r >= cfg.get("wb_bright_cool_min_gw_r", -0.80)):
            damp_f = float(cfg.get("wb_bright_cool_factor", 0.55))
            wb = dict(wb, temp=float(wb["temp"] * damp_f), diagnostics=dict(wd, bright_cool_damped=damp_f))
    m_wb = cfg["wb_max_shift"]
    base_p["temp"] = float(np.clip((1 - wb["confidence"]) * base_p["temp"] + wb["temp"], -m_wb, m_wb))
    base_p["tint"] = float(np.clip((1 - wb["confidence"]) * base_p["tint"] + wb["tint"], -m_wb, m_wb))
    base_p["temp"] = float(np.clip(base_p["temp"] + cfg.get("temp_offset", 0.0), *BOUNDS["temp"]))
    base_p["tint"] = float(np.clip(base_p["tint"] + cfg.get("tint_offset", 0.0) + cfg.get("base_tint", 0.0),
                                   *BOUNDS["tint"]))
    out["diagnostics"]["wb"] = wb
    auto_lv = out["diagnostics"]["auto_levels"] = {}

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

    c_all = _chroma(px[(px @ LUMA_F32) > cfg["min_luma"]])
    p90 = float(np.percentile(np.linalg.norm(c_all, axis=1), 90)) if len(c_all) else 0.3
    if p90 > cfg["sat_colourful"]:
        base_p["sat"] = float(1.0 + (cfg["base_sat"] - 1.0) * (cfg["sat_colourful"] / p90))
    else:
        base_p["vibrance"] = float(cfg.get("base_vibrance", 0.08))
    out["diagnostics"]["chroma_p90"] = p90

    c_lo, c_hi = cfg["contrast_bounds_v3"]
    base_p["contrast"] = float(np.clip(cfg["base_contrast"], c_lo, max(c_hi, cfg["base_contrast"])))

    target_d, band_d, anc_d = float(cfg["anchor_target_density"]), float(cfg["anchor_band_d"]), tc["anchor_d"]
    if anc_d > target_d + band_d:
        ev_anc = (anc_d - (target_d + band_d * 0.4)) / 0.30103
    elif anc_d < target_d - band_d:
        ev_anc = max(-0.25, (anc_d - (target_d - band_d)) * 0.35 / 0.30103)
    else:
        ev_anc = (anc_d - target_d) * 0.4 / 0.30103
    base_p["ev"] = float(np.clip(ev_anc, *cfg["ev_limits"]))

    tex_px = tc["textured_rgb"]
    refine, p98_max = _refine_ev_v3(base_p, tex_px, px_tone, anc_d, cfg, span_d_g)
    out["diagnostics"]["ev_refine"] = refine

    q25, q90 = _encoded_quantiles(px_tone, RenderParams(**base_p), qs=(25, 90))
    thin = _apply_thin_and_flat_v3(base_p, st, tc, q25, q90, span_d_g, bright_share, out, cfg)
    w_bl = _apply_backlit_v3(base_p, px_tone, q25, q90, thin, span_d_g, bright_share, out, cfg, flags)
    ev_lowered = _apply_highlight_hold_v3(base_p, px, tc, out, cfg, flags)
    base_p = _apply_backlit_rescue_v3(base_p, px, px_tone, tex_px, st, refine, w_bl, q25, q90, ev_lowered, span_d_g, p98_max, out, cfg, flags)

    _apply_overlift_damp_v3(base_p, refine, thin, span_d_g, bright_share, float(q25), float(w_bl), ev_lowered, p90, out, cfg)
    spread, p2, p98 = _apply_contrast_and_shadows_v3(base_p, px, tc, w_bl > 0, p90, bright_share, span_d_g, c_lo, out, cfg, auto_lv, flags)

    _apply_print_grade_v3(
        base_p, px, tc, wb, p90, span_d_g, bright_share, float(q25), float(w_bl),
        refine, ev_lowered, spread, out, cfg,
        px_y_stats=tuple(float(v) for v in np.quantile(px @ LUMA_F32, [0.02, 0.10, 0.50, 0.90, 0.98])),
    )

    for k in _ROUND_SLIDER_KEYS:
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


NOREF_T22 = dict(
    id="noref_t22",
    version="T22",
    weights_name="t20_v3",
    t18_version="t18",
    roll_alpha_wb=0.35,
    roll_alpha_tone=0.45,
    roll_k_shrink=4.0,
    roll_mad_wb_scale=0.12,
)
NOREF_T20 = dict(NOREF_T22, id="noref_t20")  # legacy alias
NOREF_T23 = dict(
    id="noref_t23",
    version="T23",
    weights_name="t23",
    t18_version="t23",
    roll_alpha_wb=0.35,
    roll_alpha_tone=0.45,
    roll_k_shrink=4.0,
    roll_mad_wb_scale=0.12,
)
NOREF_T24 = dict(
    id="noref_t24",
    version="T24",
    weights_name="t24",
    t18_version="t23",
    roll_alpha_wb=0.35,
    roll_alpha_tone=0.45,
    roll_k_shrink=4.0,
    roll_mad_wb_scale=0.12,
)

_T20_WEIGHTS_CACHE: dict[str, dict] = {}


def _load_t20_weights(name: str = "t20_v3") -> dict:
    cached = _T20_WEIGHTS_CACHE.get(name)
    if cached is not None:
        return cached
    p = CONFIG_DIR / f"{name}.json"
    data = json.loads(p.read_text())
    _T20_WEIGHTS_CACHE[name] = data
    return data


def _t20_interior_crop(P: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    H, W = P.shape[:2]
    if mask is not None and mask.shape[:2] == (H, W):
        ys, xs = np.where(mask.astype(bool))
        if len(ys) > 100:
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
        else:
            y0, y1, x0, x1 = 0, H, 0, W
    else:
        y0, y1, x0, x1 = int(0.04 * H), int(0.96 * H), int(0.04 * W), int(0.96 * W)
    dy = max(1, int(0.03 * (y1 - y0)))
    dx = max(1, int(0.03 * (x1 - x0)))
    y0i, y1i = min(y0 + dy, max(y1 - 1, y0 + 1)), max(y1 - dy, min(y0 + 1, H))
    x0i, x1i = min(x0 + dx, max(x1 - 1, x0 + 1)), max(x1 - dx, min(x0 + 1, W))
    return np.clip(np.asarray(P[y0i:y1i, x0i:x1i], dtype=np.float32), 1e-6, None)


def extract_t20_features(
    P_t18: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    d_hi=(0.85, 0.85, 0.85),
    d_lo=(0.0, 0.0, 0.0),
    matrix=IDENTITY,
) -> dict:
    """Extract clean holder-masked 2D spatial, luminance, and pre-matrix WB cues from T18-inverted ``P_t18``."""
    P_in = _t20_downsample_for_v3(_t20_interior_crop(P_t18, mask), max_side=360)
    hi, wi = P_in.shape[:2]

    # Pre-matrix WB & luminance features (exact space where wb_gains() operates)
    luma_pre = np.clip((P_in * LUMA_F32).sum(axis=-1), 1e-6, None)
    valid_pre = luma_pre > 1e-4
    if valid_pre.sum() < 100:
        valid_pre = np.ones_like(luma_pre, dtype=bool)
    z_pre = np.log2(luma_pre / 0.18).astype(np.float64)
    zq_pre5, zq_pre85, zq_pre95 = np.percentile(z_pre[valid_pre], [5, 85, 95])
    lr_pre = np.log2(np.maximum(P_in[..., 0], 1e-6) / np.maximum(P_in[..., 1], 1e-6))
    lb_pre = np.log2(np.maximum(P_in[..., 2], 1e-6) / np.maximum(P_in[..., 1], 1e-6))
    mid_pre = valid_pre & (z_pre > zq_pre5) & (z_pre < zq_pre95) & (luma_pre > 0.004)
    if mid_pre.sum() < 50:
        mid_pre = valid_pre
    c_gw_pre = [float(np.median(lr_pre[mid_pre])), float(np.median(lb_pre[mid_pre]))]
    wp_pre = valid_pre & (z_pre >= zq_pre85)
    if wp_pre.sum() < 30:
        wp_pre = mid_pre
    cmag_pre = np.hypot(lr_pre - c_gw_pre[0], lb_pre - c_gw_pre[1])
    w_wp = wp_pre * np.exp(-(cmag_pre ** 2) / 0.35)
    w_wp /= max(float(w_wp.sum()), 1e-8)
    c_wp_pre = [float((w_wp * lr_pre).sum()), float((w_wp * lb_pre).sum())]

    if hi >= 4 and wi >= 4:
        log_rgb = np.log2(P_in)
        gm = np.sqrt(
            (0.25 * (log_rgb[1:-1, 2:] - log_rgb[1:-1, :-2]) ** 2
             + 0.25 * (log_rgb[2:, 1:-1] - log_rgb[:-2, 1:-1]) ** 2).sum(-1) / 3.0
        )
        w_ge = (valid_pre[1:-1, 1:-1] & (luma_pre[1:-1, 1:-1] > 0.005)) * np.clip(gm, 0.0, 0.8)
        w_ge /= max(float(w_ge.sum()), 1e-8)
        c_ge_pre = [float((w_ge * lr_pre[1:-1, 1:-1]).sum()), float((w_ge * lb_pre[1:-1, 1:-1]).sum())]

        lap = np.abs(
            log_rgb[1:-1, 2:] + log_rgb[1:-1, :-2]
            + log_rgb[2:, 1:-1] + log_rgb[:-2, 1:-1] - 4.0 * log_rgb[1:-1, 1:-1]
        )
        lap_m = np.sqrt((lap ** 2).sum(-1) / 3.0)
        w_ge2 = (valid_pre[1:-1, 1:-1] & (luma_pre[1:-1, 1:-1] > 0.005)) * np.clip(lap_m, 0.0, 1.2)
        w_ge2 /= max(float(w_ge2.sum()), 1e-8)
        c_ge2_pre = [float((w_ge2 * lr_pre[1:-1, 1:-1]).sum()), float((w_ge2 * lb_pre[1:-1, 1:-1]).sum())]
    else:
        c_ge_pre = list(c_gw_pre)
        c_ge2_pre = list(c_gw_pre)

    tint_dev = 0.5 * (lr_pre + lb_pre) - float(np.median(0.5 * (lr_pre[mid_pre] + lb_pre[mid_pre])))
    w_pl = mid_pre * np.exp(-(tint_dev ** 2) / 0.06)
    w_pl /= max(float(w_pl.sum()), 1e-8)
    c_pl_pre = [float((w_pl * lr_pre).sum()), float((w_pl * lb_pre).sum())]

    # Single 6x8 spatial patch pass for WB voting + local block contrast (pre-matrix)
    gx_pre = np.zeros_like(z_pre)
    gy_pre = np.zeros_like(z_pre)
    if wi >= 3:
        gx_pre[:, 1:-1] = 0.5 * (z_pre[:, 2:] - z_pre[:, :-2])
    if hi >= 3:
        gy_pre[1:-1, :] = 0.5 * (z_pre[2:, :] - z_pre[:-2, :])
    grad_pre = np.hypot(gx_pre, gy_pre)
    p_lr, p_lb, p_wt, p_wn, p_zl, b_stds = [], [], [], [], [], []
    bh_p, bw_p = max(4, hi // 6), max(4, wi // 8)
    bright_pre = luma_pre > 0.004
    for by in range(0, hi - bh_p + 1, bh_p):
        for bx in range(0, wi - bw_p + 1, bw_p):
            vp = valid_pre[by:by + bh_p, bx:bx + bw_p]
            zb_all = z_pre[by:by + bh_p, bx:bx + bw_p]
            if vp.sum() > 20:
                b_stds.append(float(np.std(zb_all[vp])))
            mv = vp & bright_pre[by:by + bh_p, bx:bx + bw_p]
            if mv.sum() < 16:
                continue
            lrb = lr_pre[by:by + bh_p, bx:bx + bw_p][mv]
            lbb = lb_pre[by:by + bh_p, bx:bx + bw_p][mv]
            mlr, mlb = float(np.median(lrb)), float(np.median(lbb))
            c_std = float(np.hypot(np.std(lrb), np.std(lbb)))
            g_m = float(np.mean(grad_pre[by:by + bh_p, bx:bx + bw_p][mv]))
            z_m = float(np.median(zb_all[mv]))
            wt = (1.0 - np.exp(-g_m / 0.08)) * (1.0 - np.exp(-c_std / 0.06)) * np.exp(-max(0.0, -z_m - 1.5) ** 2 / 2.0)
            wn = np.exp(-((mlr - c_gw_pre[0]) ** 2 + (mlb - c_gw_pre[1]) ** 2) / 0.25) * (1.0 - np.exp(-g_m / 0.06))
            p_lr.append(mlr)
            p_lb.append(mlb)
            p_wt.append(max(wt, 1e-6))
            p_wn.append(max(wn, 1e-6))
            p_zl.append(z_m)
    if p_lr:
        p_lr_a = np.asarray(p_lr, dtype=np.float64)
        p_lb_a = np.asarray(p_lb, dtype=np.float64)
        wt_a = np.asarray(p_wt, dtype=np.float64)
        wt_sum = max(float(wt_a.sum()), 1e-8)
        wn_a = np.asarray(p_wn, dtype=np.float64)
        wn_a /= max(float(wn_a.sum()), 1e-8)
        c_patch_tex = [float((wt_a * p_lr_a).sum() / wt_sum), float((wt_a * p_lb_a).sum() / wt_sum)]
        c_patch_neut = [float((wn_a * p_lr_a).sum()), float((wn_a * p_lb_a).sum())]
        c_patch_med = [float(np.median(p_lr_a)), float(np.median(p_lb_a))]
        pl_a = np.asarray(p_zl, dtype=np.float64)
        wb_b = (1.0 - np.exp(-wt_a / max(float(np.median(wt_a)), 1e-4))) * np.exp(
            -((pl_a - float(np.percentile(pl_a, 85))) ** 2) / 0.8
        )
        wb_b = np.maximum(wb_b, 1e-8)
        wb_b /= wb_b.sum()
        c_patch_wp = [float((wb_b * p_lr_a).sum()), float((wb_b * p_lb_a).sum())]
    else:
        c_patch_tex = list(c_gw_pre)
        c_patch_neut = list(c_gw_pre)
        c_patch_med = list(c_gw_pre)
        c_patch_wp = list(c_wp_pre)
    block_std_med = float(np.median(b_stds)) if b_stds else 0.5

    # Center-vs-surround & top-vs-bottom on pre-matrix luma
    cy0, cy1 = hi // 4, (3 * hi) // 4
    cx0, cx1 = wi // 4, (3 * wi) // 4
    center_m = np.zeros_like(valid_pre)
    center_m[cy0:cy1, cx0:cx1] = True
    cm_v = center_m & valid_pre
    om_v = (~center_m) & valid_pre
    z_center_diff = float(np.median(z_pre[cm_v]) - np.median(z_pre[om_v])) if (cm_v.sum() > 20 and om_v.sum() > 20) else 0.0
    tm_v = valid_pre[: max(1, hi // 3), :]
    bm_v = valid_pre[min(hi - 1, (2 * hi) // 3):, :]
    top_bot_diff = (
        float(np.median(z_pre[: max(1, hi // 3), :][tm_v]) - np.median(z_pre[min(hi - 1, (2 * hi) // 3):, :][bm_v]))
        if (tm_v.sum() > 20 and bm_v.sum() > 20)
        else 0.0
    )

    # Working-space (post-matrix) luma, saliency, edge & chroma quantiles
    mat = np.asarray(matrix, dtype=np.float32)
    P_w = np.clip(P_in @ mat.T, 1e-6, None)
    luma = np.clip((P_w * LUMA_F32).sum(axis=-1), 1e-6, None)
    valid = luma > 1e-4
    if valid.sum() < 100:
        valid = np.ones_like(luma, dtype=bool)
    z = np.log2(luma / 0.18).astype(np.float64)
    zv = z[valid]
    zq_all = np.percentile(zv, [2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5]).astype(np.float64)
    zq, z_max99, z_max995 = zq_all[:9], float(zq_all[9]), float(zq_all[10])

    gx = np.zeros_like(z)
    gy = np.zeros_like(z)
    if wi >= 3:
        gx[:, 1:-1] = 0.5 * (z[:, 2:] - z[:, :-2])
    if hi >= 3:
        gy[1:-1, :] = 0.5 * (z[2:, :] - z[:-2, :])
    grad = np.hypot(gx, gy)
    grad_v = grad[valid]
    edge_mean = float(np.mean(grad_v))
    edge_p75 = float(np.percentile(grad_v, 75))

    yy = np.linspace(-1.0, 1.0, hi)[:, None]
    xx = np.linspace(-1.0, 1.0, wi)[None, :]
    w_center = np.exp(-1.1 * (xx ** 2 + yy ** 2))
    grad_norm = np.clip(grad / max(edge_p75, 0.05), 0.15, 2.5)
    w_sal = w_center * grad_norm * valid
    w_sal_s = w_sal / max(float(w_sal.sum()), 1e-8)
    z_flat = z.ravel()
    order = np.argsort(z_flat)
    z_sorted = z_flat[order]
    ws = np.cumsum(w_sal_s.ravel()[order])
    z_sal50 = float(z_sorted[min(int(np.searchsorted(ws, 0.5)), len(order) - 1)])

    w_fg = w_center * (np.clip(grad / max(edge_p75, 0.05), 0.05, 3.0) ** 1.6) * valid
    w_fg /= max(float(w_fg.sum()), 1e-8)
    ws_fg = np.cumsum(w_fg.ravel()[order])
    z_fg_sal25, z_fg_sal50, z_fg_sal75 = (
        float(z_sorted[min(int(np.searchsorted(ws_fg, q)), len(order) - 1)]) for q in (0.25, 0.50, 0.75)
    )

    p02_rgb, p05_rgb = np.percentile(P_w[valid], [2.0, 5.0], axis=0)
    sh_r_excess = float(np.log2(max(float(p05_rgb[0]), 1e-5) / max(float(p05_rgb[1]), 1e-5)))
    sh_b_excess = float(np.log2(max(float(p05_rgb[2]), 1e-5) / max(float(p05_rgb[1]), 1e-5)))
    sh_floor_luma = float(np.log2(max(float((p02_rgb * LUMA_F32).sum()), 1e-5) / 0.18))

    tex_m = valid & (grad > max(0.04, 0.35 * edge_p75))
    if tex_m.sum() < 50:
        tex_m = valid
    tex_z25, tex_z50, tex_z75 = np.percentile(z[tex_m], [25, 50, 75]).astype(np.float64)

    dark_flat = float(np.mean((z < -2.0) & (grad < 0.08) & valid))
    bright_flat = float(np.mean((z > 1.3) & (grad < 0.08) & valid))
    hl_clip_frac = float(np.mean((z > 2.0) & valid))

    lr = np.log2(np.maximum(P_w[..., 0], 1e-6) / np.maximum(P_w[..., 1], 1e-6))
    lb = np.log2(np.maximum(P_w[..., 2], 1e-6) / np.maximum(P_w[..., 1], 1e-6))
    mid_w = valid & (z > zq[1]) & (z < zq[7]) & (luma > 0.004)
    if mid_w.sum() < 50:
        mid_w = valid
    c_gw = [float(np.median(lr[mid_w])), float(np.median(lb[mid_w]))]
    ch_raw = np.hypot(lr[mid_w] - c_gw[0], lb[mid_w] - c_gw[1])
    ch_p75 = float(np.percentile(ch_raw, 75))

    top_half = np.zeros_like(valid)
    top_half[: max(1, hi // 2), :] = True
    sky_frac = float(np.mean(top_half & valid & (z > zq[4]) & ((lb - lr) > 0.20)))

    def _wb_from_c(w):
        ws = max(float(w.sum()), 1e-8)
        cr = float((w * lr_pre).sum() / ws)
        cb = float((w * lb_pre).sum() / ws)
        return -(cr - cb), -0.5 * (cr + cb)

    zq_w = zq_all[[1, 2, 3, 4, 5, 6, 7, 9]]  # [5, 10, 25, 50, 75, 90, 95, 99]
    sky_m = top_half & (z > zq_w[3]) & ((lb - lr) > 0.25)
    fol_m = (lr < -0.15) & (lb < -0.15)
    skin_m = (lr > 0.22) & (lb < -0.22)
    clean_m = valid & (z > zq_w[0]) & (z < zq_w[6]) & (~sky_m) & (~fol_m) & (~skin_m)
    if clean_m.sum() < 40:
        clean_m = valid & (z > zq_w[0]) & (z < zq_w[6])
    t_cneut, n_cneut = _wb_from_c(clean_m * np.exp(-(lr ** 2 + lb ** 2) / 0.35))

    w_cbright = valid & (z >= zq_w[4]) & (~sky_m)
    if w_cbright.sum() < 30:
        w_cbright = valid & (z >= zq_w[4])
    t_cbright, n_cbright = _wb_from_c(w_cbright * np.exp(-(lr ** 2 + lb ** 2) / 0.40))

    hi_m = valid_pre & (z >= zq_w[4]) & (luma > 0.01)
    lo_m = valid_pre & (z <= zq_w[2]) & (luma > 0.002)
    if hi_m.sum() < 30:
        hi_m = valid_pre
    if lo_m.sum() < 30:
        lo_m = valid_pre
    t_hi_wb, n_hi_wb = _wb_from_c(hi_m.astype(np.float64))
    t_lo_wb, n_lo_wb = _wb_from_c(lo_m.astype(np.float64))

    P_clip = np.clip(P_in, 1e-6, 4.0).astype(np.float64)
    m6 = np.power(np.mean(np.power(P_clip[valid_pre], 6.0), axis=0), 1.0 / 6.0)
    lr_s6 = float(np.log2(max(float(m6[0]), 1e-6) / max(float(m6[1]), 1e-6)))
    lb_s6 = float(np.log2(max(float(m6[2]), 1e-6) / max(float(m6[1]), 1e-6)))

    dh = np.asarray(d_hi, dtype=np.float64) - np.asarray(d_lo, dtype=np.float64)
    dg = max(float(dh[1]), 1e-4)
    r_u = float(dh[0]) / dg
    b_u = float(dh[2]) / dg

    return dict(
        t_cneut=t_cneut,
        n_cneut=n_cneut,
        t_cbright=t_cbright,
        n_cbright=n_cbright,
        t_hi_wb=t_hi_wb,
        n_hi_wb=n_hi_wb,
        t_lo_wb=t_lo_wb,
        n_lo_wb=n_lo_wb,
        t_hl_sh_diff=t_hi_wb - t_lo_wb,
        n_hl_sh_diff=n_hi_wb - n_lo_wb,
        t_sog6=-(lr_s6 - lb_s6),
        n_sog6=-0.5 * (lr_s6 + lb_s6),
        z_mean_log=float(np.log2(max(float(np.mean(luma[valid])), 1e-6) / 0.18)),
        z_rms=float(0.5 * np.log2(max(float(np.mean(luma[valid] ** 2)), 1e-12) / (0.18 ** 2))),
        z_trim=float(np.mean(np.clip(z[valid], zq_w[1], zq_w[5]))),
        key_asym=float((zq_w[5] - zq_w[3]) - (zq_w[3] - zq_w[1])),
        hl_head99=float(zq_w[7] - zq_w[4]),
        sh_tail25_05=float(zq_w[2] - zq_w[0]),
        b_u=b_u,
        log_span=float(np.log(max(dg, 0.08))),
        d_hi_g_sat=float(np.tanh((dg - 0.85) / 0.45)),
        z02=float(zq[0]),
        z10=float(zq[2]),
        z25=float(zq[3]),
        z50=float(zq[4]),
        z75=float(zq[5]),
        z90=float(zq[6]),
        z98=float(zq[8]),
        z_sal50=z_sal50,
        tex50=float(tex_z50),
        dark_flat=dark_flat,
        bright_flat=bright_flat,
        hl_clip=hl_clip_frac,
        sky_frac=sky_frac,
        span90=float(zq[6] - zq[2]),
        sh_tail=float(zq[3] - zq[0]),
        hi_head=float(zq[8] - zq[5]),
        edge_mean=edge_mean,
        block_std=block_std_med,
        z_center_diff=z_center_diff,
        top_bot_diff=top_bot_diff,
        ch_p75=ch_p75,
        d_hi_g=dg,
        r_u=r_u,
        t_gw=-(c_gw_pre[0] - c_gw_pre[1]),
        t_wp=-(c_wp_pre[0] - c_wp_pre[1]),
        t_ge=-(c_ge_pre[0] - c_ge_pre[1]),
        t_ge2=-(c_ge2_pre[0] - c_ge2_pre[1]),
        t_pl=-(c_pl_pre[0] - c_pl_pre[1]),
        n_gw=-0.5 * (c_gw_pre[0] + c_gw_pre[1]),
        n_wp=-0.5 * (c_wp_pre[0] + c_wp_pre[1]),
        n_ge=-0.5 * (c_ge_pre[0] + c_ge_pre[1]),
        n_ge2=-0.5 * (c_ge2_pre[0] + c_ge2_pre[1]),
        n_pl=-0.5 * (c_pl_pre[0] + c_pl_pre[1]),
        t_ptex=-(c_patch_tex[0] - c_patch_tex[1]),
        n_ptex=-0.5 * (c_patch_tex[0] + c_patch_tex[1]),
        t_pneut=-(c_patch_neut[0] - c_patch_neut[1]),
        n_pneut=-0.5 * (c_patch_neut[0] + c_patch_neut[1]),
        t_pmed=-(c_patch_med[0] - c_patch_med[1]),
        n_pmed=-0.5 * (c_patch_med[0] + c_patch_med[1]),
        t_pwp=-(c_patch_wp[0] - c_patch_wp[1]),
        n_pwp=-0.5 * (c_patch_wp[0] + c_patch_wp[1]),
        z_fg_sal50=z_fg_sal50,
        z_fg_sal_span=float(z_fg_sal75 - z_fg_sal25),
        z_max99=z_max99,
        z_max995=z_max995,
        sh_r_excess=sh_r_excess,
        sh_b_excess=sh_b_excess,
        sh_floor_luma=sh_floor_luma,
    )


def _t20_downsample_for_v3(P: np.ndarray, max_side: int = 320) -> np.ndarray:
    h, w = P.shape[:2]
    step = max(1, int(max(h, w) // max_side))
    if step <= 1:
        return P
    return np.ascontiguousarray(P[::step, ::step], dtype=np.float32)


def estimate_params_t20(
    P_t18: np.ndarray,
    mask: np.ndarray | None = None,
    cfg: dict = NOREF_T20,
    *,
    d_hi=(0.85, 0.85, 0.85),
    d_lo=(0.0, 0.0, 0.0),
    has_cam_matrix: bool = True,
    matrix=IDENTITY,
    P_v1: np.ndarray | None = None,
    cfg_v3: dict = NOREF_V3,
    _weights_dict: dict | None = None,
) -> dict:
    """T20 smooth C1 physics-based parameter estimator for T18-inverted negatives + T19 orthogonal paper curve."""
    if not _all_finite(np.asarray(P_t18)):
        raise ValueError("input contains NaN or infinite values")
    dh = np.asarray(d_hi, dtype=np.float64) - np.asarray(d_lo, dtype=np.float64)
    span_d_g = float(dh[1])
    w_name = cfg.get("weights_name", "t20_v3")
    orange_branch = bool(has_cam_matrix or (float(dh[2]) / max(float(dh[1]), 1e-4) > 0.80))
    branch_name = "cal" if orange_branch else "uncal"
    mat_tup = RenderParams(matrix=matrix).matrix
    P_anchor_in = _t20_interior_crop(P_v1 if P_v1 is not None else P_t18, mask)
    if min(P_anchor_in.shape[:2]) < 2:
        raise ValueError(f"frame too small for the automatic estimate: interior crop {P_anchor_in.shape[:2]}")
    P_anchor = _t20_downsample_for_v3(P_anchor_in, max_side=320)
    v3_cfg = dict(cfg_v3, matrix=mat_tup)
    res_v3 = estimate_params_v3(P_anchor, None, v3_cfg, span_d_g=span_d_g)
    pv17 = res_v3["params"]

    feats = extract_t20_features(P_t18, mask, d_hi=d_hi, d_lo=d_lo, matrix=mat_tup)
    feats.update(
        ev17=float(pv17["ev"]),
        tp17=float(pv17["temp"]),
        tn17=float(pv17["tint"]),
        c17=float(pv17["contrast"]),
        sh17=float(pv17["shoulder"]),
        blk17=float(pv17.get("blacks", 0.0)),
        sat17=float(pv17["sat"]),
        df_x_ev=float(feats["dark_flat"]) * max(float(pv17["ev"]), 0.0),
        sky_x_tge=float(feats["sky_frac"]) * (float(feats["t_ge"]) - float(feats["t_gw"])),
        ev_fg_hl=float(min(-feats["z_fg_sal50"], 2.3 - feats["z_max99"])),
        ev_backlit=float(max(-feats["z_center_diff"], 0.0) * feats["bright_flat"]),
        ev_hl_ceil=float(np.tanh(2.3 - feats["z_max995"])),
        wb_sky_t=float(np.tanh(2.5 * feats["sky_frac"]) * (feats["t_ge"] - feats["t_gw"])),
        wb_ch_t=float(np.tanh(feats["ch_p75"]) * (feats["t_pneut"] - feats["t_pmed"])),
        wb_ch_n=float(np.tanh(feats["ch_p75"]) * (feats["n_pneut"] - feats["n_pmed"])),
        t_tex_gw=float(feats["t_ptex"] - feats["t_gw"]),
        n_tex_gw=float(feats["n_ptex"] - feats["n_gw"]),
        t_wp_gw=float(feats["t_wp"] - feats["t_gw"]),
        n_wp_gw=float(feats["n_wp"] - feats["n_gw"]),
    )

    weights = _weights_dict if _weights_dict is not None else _load_t20_weights(w_name)
    col_sets = weights.get("branch_col_sets", {}).get(branch_name, weights["col_sets"])
    branch = weights["branches"][branch_name]
    soft_clip_z = weights.get("branch_soft_clip_z", {}).get(branch_name, weights.get("soft_clip_z"))
    ev_anchor = weights.get("ev_anchor", "ev17")

    raw_preds = {}
    for pn, cols in col_sets.items():
        spec = branch[pn]
        x = np.array([feats[c] for c in cols], dtype=np.float64)
        mu = np.asarray(spec["mean"], dtype=np.float64)
        sd = np.maximum(np.asarray(spec["scale"], dtype=np.float64), 1e-8)
        z_x = (x - mu) / sd
        xs = (soft_clip_z * np.tanh(z_x / soft_clip_z)) if soft_clip_z is not None else np.clip(z_x, -3.5, 3.5)
        raw_preds[pn] = float(spec["b"] + np.dot(np.asarray(spec["w"], dtype=np.float64), xs))

    ev_scale = float(weights.get("branch_ev_scale", {}).get(branch_name, 1.0))
    if ev_anchor:
        ev_raw = feats[ev_anchor] + ev_scale * np.tanh((raw_preds["ev"] - feats[ev_anchor]) / ev_scale)
    else:
        ev_raw = raw_preds["ev"]
    ev_p = float(np.clip(ev_raw, -2.5, 2.5))
    temp_p = float(np.clip(raw_preds["temp"], -1.0, 1.0))
    tint_p = float(np.clip(raw_preds["tint"], -1.0, 1.0))
    sat_p = float(np.clip(np.exp(raw_preds["sat"]), 0.55, 1.35))
    contrast_p = float(np.clip(np.exp(raw_preds["contrast"]), 0.65, 1.95))
    shoulder_p = float(1.0 / (1.0 + np.exp(-raw_preds["shoulder"])))
    toe_p = float(np.tanh(raw_preds["toe"]))

    params = RenderParamsT19(
        ev=round(ev_p, 4),
        temp=round(temp_p, 4),
        tint=round(tint_p, 4),
        sat=round(sat_p, 4),
        contrast=round(contrast_p, 4),
        shoulder=round(shoulder_p, 4),
        toe=round(toe_p, 4),
        matrix=mat_tup,
    )
    return dict(
        status="ok",
        params=params.to_dict(),
        v3_anchor=pv17,
        features=feats,
        branch=branch_name,
    )


def pool_roll_params_t21(
    params_list: list[dict | RenderParamsT19],
    *,
    alpha_wb: float = 0.35,
    alpha_tone: float = 0.45,
    k_shrink: float = 4.0,
    mad_wb_scale: float | None = 0.12,
) -> list[dict]:
    """T21b hierarchical roll pooling: shrink stock tone parameters toward roll medians, and shrink WB
    gated smoothly by within-roll illuminant dispersion (``mad_wb_scale``)."""
    n = len(params_list)
    dicts = [p.to_dict() if hasattr(p, "to_dict") else dict(p) for p in params_list]
    if n < 2:
        return dicts
    w_roll = float(n) / (float(n) + float(k_shrink))
    if mad_wb_scale is not None and mad_wb_scale > 0.0:
        tp_vals = np.array([float(d["temp"]) for d in dicts], dtype=np.float64)
        tn_vals = np.array([float(d["tint"]) for d in dicts], dtype=np.float64)
        mad_wb = float(np.median(np.hypot(tp_vals - np.median(tp_vals), tn_vals - np.median(tn_vals))))
        gate_wb = float(np.exp(-mad_wb / float(mad_wb_scale)))
    else:
        gate_wb = 1.0
    keys = [
        ("temp", alpha_wb * gate_wb),
        ("tint", alpha_wb * gate_wb),
        ("contrast", alpha_tone),
        ("shoulder", alpha_tone),
        ("toe", alpha_tone),
    ]
    medians = {k: float(np.median([float(d[k]) for d in dicts])) for k, _ in keys}
    out = []
    for d in dicts:
        nd = dict(d)
        for k, a in keys:
            lo, hi = T19_BOUNDS[k]
            val = (1.0 - a * w_roll) * float(d[k]) + (a * w_roll) * medians[k]
            nd[k] = round(float(np.clip(val, lo, hi)), 4)
        out.append(nd)
    return out

estimate_params_t22 = estimate_params_t20
pool_roll_params_t22 = pool_roll_params_t21
extract_t23_features = extract_t20_features
pool_roll_params_t23 = pool_roll_params_t21


def estimate_params_t23(
    P_t18: np.ndarray,
    mask: np.ndarray | None = None,
    cfg: dict = NOREF_T23,
    *,
    d_hi=(0.85, 0.85, 0.85),
    d_lo=(0.0, 0.0, 0.0),
    has_cam_matrix: bool = True,
    matrix=IDENTITY,
    P_v1: np.ndarray | None = None,
    cfg_v3: dict = NOREF_V3,
) -> dict:
    return estimate_params_t20(
        P_t18, mask, cfg,
        d_hi=d_hi, d_lo=d_lo, has_cam_matrix=has_cam_matrix,
        matrix=matrix, P_v1=P_v1, cfg_v3=cfg_v3,
    )


def _eval_et_trees(trees: list[dict], x32: np.ndarray) -> float:
    acc = 0.0
    for t in trees:
        cl, cr, feat, th, val = t["cl"], t["cr"], t["f"], t["th"], t["v"]
        nid = 0
        while cl[nid] != -1:
            nid = cl[nid] if float(x32[feat[nid]]) <= th[nid] else cr[nid]
        acc += val[nid]
    return acc / len(trees)


def _eval_xgb_trees(spec: dict, x32: np.ndarray) -> float:
    acc = float(spec["base_score"])
    for t in spec["trees"]:
        lc, rc, fi, sc, dl = t["lc"], t["rc"], t["fi"], t["sc"], t["dl"]
        nid = 0
        while lc[nid] != -1:
            v = np.float32(x32[fi[nid]])
            if np.isnan(v):
                nid = lc[nid] if dl[nid] else rc[nid]
            elif v < np.float32(sc[nid]):
                nid = lc[nid]
            else:
                nid = rc[nid]
        acc += sc[nid]
    return acc


def _t24_resize(arr: np.ndarray, side: int) -> np.ndarray:
    H, W = arr.shape[:2]
    ys = np.linspace(0, H - 1, side).astype(np.int32)
    xs = np.linspace(0, W - 1, side).astype(np.int32)
    return np.ascontiguousarray(arr[ys][:, xs], dtype=np.float32)


def _t24_params_to_raw7(p: dict) -> list[float]:
    sh = min(max(float(p["shoulder"]), 0.03), 0.97)
    return [
        float(p["ev"]),
        float(p["temp"]),
        float(p["tint"]),
        float(np.log(min(max(float(p["sat"]), 0.55), 1.35))),
        float(np.log(min(max(float(p["contrast"]), 0.65), 1.95))),
        float(np.log(sh / (1.0 - sh))),
        float(np.arctanh(min(max(float(p["toe"]), -0.95), 0.95))),
    ]


def _t24_raw7_to_params(r: list[float], matrix) -> dict:
    return {
        "ev": round(float(np.clip(r[0], -2.5, 2.5)), 4),
        "temp": round(float(np.clip(r[1], -1.0, 1.0)), 4),
        "tint": round(float(np.clip(r[2], -1.0, 1.0)), 4),
        "sat": round(float(np.clip(np.exp(r[3]), 0.55, 1.35)), 4),
        "contrast": round(float(np.clip(np.exp(r[4]), 0.65, 1.95)), 4),
        "shoulder": round(float(1.0 / (1.0 + np.exp(-r[5]))), 4),
        "toe": round(float(np.tanh(r[6])), 4),
        "matrix": matrix,
    }


def _t24_density_feats(P_v1_crop: np.ndarray, d_lo, d_hi, gamma: float) -> tuple[dict, float]:
    d_lo_a = np.asarray(d_lo, dtype=np.float64)
    d_hi_a = np.asarray(d_hi, dtype=np.float64)
    dg = max(float(d_hi_a[1] - d_lo_a[1]), 1e-4)
    Rv = dg / float(gamma)
    flr = 10.0 ** (-Rv)
    inner = np.maximum(np.asarray(P_v1_crop, dtype=np.float64) * (1.0 - flr) + flr, 1e-12)
    s_orig = 1.0 + np.log10(inner) / Rv
    D = d_lo_a + s_orig * (d_hi_a - d_lo_a)
    dg_vals = D[..., 1] - d_lo_a[1]
    return {
        "D_med_g": float(np.median(dg_vals)),
        "D_p10_g": float(np.percentile(dg_vals, 10)),
        "D_p90_g": float(np.percentile(dg_vals, 90)),
        "D_mid_frac": float(np.median(dg_vals)) / dg,
        "D_iqr_g": float(np.percentile(dg_vals, 75) - np.percentile(dg_vals, 25)),
        "thin_sig": float(1.0 / (1.0 + np.exp((dg - 0.58) / 0.10))),
    }, dg


def _t24_closed_loop_feats(P_crop128: np.ndarray, p_dict: dict, matrix) -> dict:
    disp, _ = render_t19(P_crop128, RenderParamsT19(matrix=matrix, **p_dict))
    px = disp.reshape(-1, 3).astype(np.float64)
    luma = np.clip((px * LUMA).sum(-1), 1e-5, None)
    lr = np.log2(np.maximum(px[:, 0], 1e-4) / np.maximum(px[:, 1], 1e-4))
    lb = np.log2(np.maximum(px[:, 2], 1e-4) / np.maximum(px[:, 1], 1e-4))
    sat = (px.max(-1) - px.min(-1)) / np.maximum(px.max(-1), 1e-3)
    ql = np.percentile(luma, [2, 10, 25, 50, 75, 90, 98])
    sh_m = luma <= ql[2]
    md_m = (luma > ql[2]) & (luma < ql[4])
    hi_m = luma >= ql[4]
    sh_lr = float(np.median(lr[sh_m])) if sh_m.any() else float(np.median(lr))
    sh_lb = float(np.median(lb[sh_m])) if sh_m.any() else float(np.median(lb))
    md_lr = float(np.median(lr[md_m])) if md_m.any() else float(np.median(lr))
    md_lb = float(np.median(lb[md_m])) if md_m.any() else float(np.median(lb))
    hi_lr = float(np.median(lr[hi_m])) if hi_m.any() else float(np.median(lr))
    hi_lb = float(np.median(lb[hi_m])) if hi_m.any() else float(np.median(lb))
    wn = np.exp(-(lr ** 2 + lb ** 2) / 0.25)
    wn /= max(float(wn.sum()), 1e-8)
    return {
        "cl_luma_p02": float(ql[0]),
        "cl_luma_p10": float(ql[1]),
        "cl_luma_p25": float(ql[2]),
        "cl_luma_p50": float(ql[3]),
        "cl_luma_p75": float(ql[4]),
        "cl_luma_p90": float(ql[5]),
        "cl_luma_p98": float(ql[6]),
        "cl_luma_std": float(luma.std()),
        "cl_clip_hi": float((px.max(-1) > 0.96).mean()),
        "cl_clip_lo": float((px.min(-1) < 0.02).mean()),
        "cl_sat_med": float(np.median(sat)),
        "cl_sat_p90": float(np.percentile(sat, 90)),
        "cl_sh_lr": sh_lr,
        "cl_sh_lb": sh_lb,
        "cl_md_lr": md_lr,
        "cl_md_lb": md_lb,
        "cl_hi_lr": hi_lr,
        "cl_hi_lb": hi_lb,
        "cl_cross_lr": hi_lr - sh_lr,
        "cl_cross_lb": hi_lb - sh_lb,
        "cl_nlr": float((wn * lr).sum()),
        "cl_nlb": float((wn * lb).sum()),
    }


pool_roll_params_t24 = pool_roll_params_t21


def estimate_params_t24(
    P_t23: np.ndarray,
    mask: np.ndarray | None = None,
    cfg: dict = NOREF_T24,
    *,
    d_hi=(0.85, 0.85, 0.85),
    d_lo=(0.0, 0.0, 0.0),
    gamma: float = 0.6,
    has_cam_matrix: bool = True,
    matrix=IDENTITY,
    P_v1: np.ndarray | None = None,
    cfg_v3: dict = NOREF_V3,
    use_ev_guard: bool = True,
) -> dict:
    """T24 ML residual estimator on top of locked T23 physics predictions (pure NumPy/JSON)."""
    bundle = _load_t20_weights(cfg.get("weights_name", "t24"))
    est23 = estimate_params_t20(
        P_t23, mask, cfg,
        d_hi=d_hi, d_lo=d_lo, has_cam_matrix=has_cam_matrix,
        matrix=matrix, P_v1=P_v1, cfg_v3=cfg_v3,
        _weights_dict=bundle["base_weights"],
    )
    branch_name = est23["branch"]
    p23 = dict(est23["params"])
    mat_tup = RenderParamsT19(matrix=p23["matrix"]).matrix
    p23_nomat = {k: v for k, v in p23.items() if k != "matrix"}

    P_crop128 = _t24_resize(_t20_interior_crop(P_t23, mask), 128)
    P_v1_crop = _t20_interior_crop(P_v1 if P_v1 is not None else P_t23, mask)
    P_v1_64 = _t24_resize(P_v1_crop, 64)

    df, dg = _t24_density_feats(P_v1_64, d_lo, d_hi, gamma)
    cl = _t24_closed_loop_feats(P_crop128, p23_nomat, mat_tup)

    d_ev_guard = 0.0
    narrow_span = 0.0
    if use_ev_guard and branch_name == "cal":
        eg = bundle["ev_guard"]
        narrow_span = float(np.clip((0.42 - dg) / 0.14, 0.0, 1.0))
        if cl["cl_luma_p50"] < eg["p50_thresh"] and p23_nomat["ev"] < 0.0:
            thin_g = float(1.0 / (1.0 + np.exp((dg - eg["span_mid"]) / eg["span_width"])))
            if thin_g > 0.05:
                lift = eg["scale"] * thin_g * float(np.log2(eg["p50_thresh"] / max(cl["cl_luma_p50"], 0.02)))
                d_ev_guard = min(eg["max_lift"], max(0.0, lift))
                if d_ev_guard > 1e-4:
                    p23_nomat["ev"] = float(np.clip(p23_nomat["ev"] + d_ev_guard, -2.5, 2.5))
                    cl = _t24_closed_loop_feats(P_crop128, p23_nomat, mat_tup)
        # Headroom-gated exposure guard for thin/sky-dominated negatives (e.g. 80% sky + dark foreground)
        if dg < 0.60 and p23_nomat["ev"] < -0.40 and cl["cl_luma_p50"] < 0.45 and cl["cl_luma_p90"] < 0.76 and cl["cl_clip_hi"] < 0.01:
            thin_soft = float(np.clip((0.60 - dg) / 0.10, 0.0, 1.0))
            headroom_cap = max(0.0, 2.4 * (0.85 - cl["cl_luma_p90"]))
            ev_pull = 0.85 * thin_soft * (-0.25 - p23_nomat["ev"])
            lift_soft = min(0.75, headroom_cap, max(0.0, ev_pull))
            if lift_soft > 1e-3:
                d_ev_guard += lift_soft
                p23_nomat["ev"] = float(np.clip(p23_nomat["ev"] + lift_soft, -2.5, 2.5))
                cl = _t24_closed_loop_feats(P_crop128, p23_nomat, mat_tup)
        # Ultra-narrow density-span flat/sky guard: avoid dragging open sky down to mid-grey
        if narrow_span > 0.0 and p23_nomat["ev"] < -0.55 and cl["cl_luma_p98"] < 0.92:
            lift_sky = min(0.85, 0.90 * narrow_span * (-0.45 - p23_nomat["ev"]))
            if lift_sky > 1e-3:
                d_ev_guard += lift_sky
                p23_nomat["ev"] = float(np.clip(p23_nomat["ev"] + lift_sky, -2.5, 2.5))
                cl = _t24_closed_loop_feats(P_crop128, p23_nomat, mat_tup)

    raw7 = _t24_params_to_raw7(p23_nomat)
    feats = dict(est23["features"])
    feats.update(df)
    feats.update(cl)
    feats.update({"b_ev": raw7[0], "b_tp": raw7[1], "b_tn": raw7[2], "b_sh": raw7[5], "b_to": raw7[6]})

    br_pkg = bundle["residual_branches"][branch_name]
    fnames = br_pkg["feature_names"]
    x32 = np.array([feats[k] for k in fnames], dtype=np.float32)
    wx = float(br_pkg["weights"]["xgb"])
    we = float(br_pkg["weights"]["et"])
    preds = {}
    for tname, hspec in br_pkg["heads"].items():
        vx = _eval_xgb_trees(hspec["xgb"], x32)
        ve = _eval_et_trees(hspec["et"], x32)
        preds[tname] = wx * vx + we * ve

    raw7_out = list(raw7)
    raw7_out[5] += float(np.clip(preds["d_shoulder"], -1.5, 1.5))
    raw7_out[6] += float(np.clip(preds["d_toe"], -1.5, 1.5))
    p_out = _t24_raw7_to_params(raw7_out, p23["matrix"])
    # Ultra-narrow density-span flat/sky guard: prevent contrast/toe from turning gentle sky vignetting black
    if narrow_span > 0.0:
        if p_out["contrast"] > 0.95:
            blend = 0.70 * narrow_span
            p_out["contrast"] = round(float(p_out["contrast"] * (1.0 - blend) + 0.95 * blend), 4)
        p_out["toe"] = round(float(max(p_out["toe"], 0.0) * (1.0 - 0.5 * narrow_span) + 0.55 * narrow_span), 4)
    p_out["slope_r"] = round(float(np.clip(preds["slope_r"], -0.40, 0.40)), 5)
    p_out["slope_b"] = round(float(np.clip(preds["slope_b"], -0.40, 0.40)), 5)
    if br_pkg["use_eps"]:
        esc = float(br_pkg.get("eps_scale", 1.0))
        p_out["eps_rgb"] = [round(float(esc * np.clip(preds[c], -0.04, 0.04)), 6) for c in ("eps_r", "eps_g", "eps_b")]
    else:
        p_out["eps_rgb"] = [0.0, 0.0, 0.0]
    # Physical floor restoration for narrow density-span negatives (e.g. open sky / low-DR scenes):
    # invert() subtracts 10^(-dg/gamma) assuming a true black exists in-frame; restore it smoothly when dg < 0.70.
    flr_phys = float(10.0 ** (-dg / max(float(gamma), 1e-4)))
    alpha_flr = float(np.clip((0.70 - dg) / 0.38, 0.0, 1.0))
    fr_val = round(float(np.clip(alpha_flr * flr_phys, 0.0, 0.45)), 5)
    if fr_val > 1e-4:
        p_out["floor_restore"] = fr_val
    if br_pkg["use_mat"]:
        v6 = [float(np.clip(preds[c], -0.20, 0.20)) for c in ("dm_01", "dm_02", "dm_10", "dm_12", "dm_20", "dm_21")]
        dm = [
            [-(v6[0] + v6[1]), v6[0], v6[1]],
            [v6[2], -(v6[2] + v6[3]), v6[3]],
            [v6[4], v6[5], -(v6[4] + v6[5])],
        ]
        p_out["residual_matrix"] = [[round(float(x), 6) for x in row] for row in dm]
    else:
        p_out["residual_matrix"] = [list(r) for r in ZERO_MATRIX]

    return dict(
        status="ok",
        params=p_out,
        v3_anchor=est23["v3_anchor"],
        features=feats,
        branch=branch_name,
        p23_base=p23,
        d_ev_guard=round(float(d_ev_guard), 4),
    )
