"""Film base (unexposed film + orange mask) estimation in linear camera RGB. Pure numpy.

The film base is the least dense part of the film, so it transmits the most light of any
area *covered by film*. Two kinds of area can be brighter and must not be taken as base:
saturated samples, and light reaching the sensor without passing through film (gaps at the
film edge). On the NLP sample set the latter differed clearly from base in chromaticity
(experiments/s1_diag), which the automatic estimator uses as a consensus check.

Regions are (top, left, height, width) in CFA pixel coordinates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .raw import CHANNELS, channel_stats, check_region, sample_stats, superpixel

# warning thresholds
MAX_REGION_CV = 0.05        # per-channel coefficient of variation inside a base region
MAX_REGION_SPREAD = 0.03    # relative disagreement between manual regions
MIN_SAMPLES = 200           # per channel


@dataclass
class BaseEstimate:
    rgb: tuple[float, float, float]           # linear camera RGB of film base, 1 = white level
    method: str                                 # "manual" | "auto"
    regions: list[tuple[int, int, int, int]]
    stats: dict                                 # per-channel statistics of the pooled samples
    params: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _pooled_stats(linear, colors, sat, regions):
    """Channel statistics over the union of regions (CFA-aware)."""
    mask = np.zeros(linear.shape, bool)
    for top, left, h, w in regions:
        check_region((top, left, h, w), linear.shape)
        mask[top : top + h, left : left + w] = True
    return sample_stats(linear[mask], colors[mask], sat[mask])


def _check_stats(stats, warnings, where):
    for name in CHANNELS:
        s = stats[name]
        if s["n"] < MIN_SAMPLES:
            warnings.append(f"{where}: only {s['n']} usable {name} samples")
        if s["sat_frac"] > 0:
            warnings.append(f"{where}: {s['sat_frac']:.2%} of {name} samples saturated (excluded)")
        if s["n"] and s["cv"] > MAX_REGION_CV:
            warnings.append(f"{where}: {name} not uniform (cv={s['cv']:.3f}); region may include image content")
        if s["n"] and s["median"] <= 0:
            warnings.append(f"{where}: {name} median at or below black level")


def estimate_base_manual(linear, colors, sat, regions) -> BaseEstimate:
    """Film base from user-specified regions. Uses the median of non-saturated samples per channel."""
    if not regions:
        raise ValueError("at least one region is required")
    regions = [tuple(int(x) for x in r) for r in regions]
    warnings: list[str] = []
    per_region = [channel_stats(linear, colors, sat, r) for r in regions]
    for i, s in enumerate(per_region):
        _check_stats(s, warnings, f"region {i}")
    pooled = _pooled_stats(linear, colors, sat, regions)
    if any(pooled[c]["n"] == 0 for c in CHANNELS):
        raise ValueError("regions contain no usable samples for at least one channel")
    rgb = tuple(pooled[c]["median"] for c in CHANNELS)
    spread = {}
    if len(regions) > 1:
        for c in CHANNELS:
            meds = [s[c]["median"] for s in per_region if s[c]["n"]]
            spread[c] = float((max(meds) - min(meds)) / np.median(meds))
            if spread[c] > MAX_REGION_SPREAD:
                warnings.append(f"regions disagree on {c} by {spread[c]:.1%}")
    return BaseEstimate(
        rgb=rgb, method="manual", regions=regions, stats=pooled,
        diagnostics=dict(per_region=per_region, region_spread=spread), warnings=warnings,
    )


def _neighbourhood_rel_std(sp):
    """Max over channels of the 3x3 neighbourhood std / mean, per block (edge blocks -> inf)."""
    h, w, _ = sp.shape
    out = np.full((h, w), np.inf, np.float32)
    if h < 3 or w < 3:
        return out
    win = np.stack([sp[dy : h - 2 + dy, dx : w - 2 + dx] for dy in range(3) for dx in range(3)])
    mean = win.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(mean > 0, win.std(0) / mean, np.inf)
    out[1:-1, 1:-1] = rel.max(-1)
    return out


def estimate_base_auto(
    linear, colors, sat, pattern_size: int, *,
    block: int = 24, top_frac: float = 0.01, max_rel_std: float = 0.03,
    chroma_tol: float = 0.05, crop: tuple[int, int, int, int] | None = None,
) -> BaseEstimate:
    """Automatic film base estimate.

    1. Downsample to CFA-aware blocks; drop blocks with any saturated sample and blocks outside
       ``crop`` (decoder-reported valid area).
    2. Keep spatially uniform blocks (3x3 neighbourhood relative std <= ``max_rel_std``).
    3. Candidates = brightest ``top_frac`` of the uniform blocks (at least 10).
    4. Chromaticity consensus: keep candidates whose log2(R/G) and log2(B/G) lie within
       ``chroma_tol`` of the candidates' median (iterated once around the new median).
    5. Base = per-channel median of the kept blocks.

    Known limitation: unsaturated light that bypassed the film and makes up the majority of the
    candidates would be taken as base. A wider candidate pool was tried and rejected because large
    bright image areas then dominated the consensus (_DSF9235). Use manual regions in that case.
    """
    sp, sp_sat = superpixel(linear, colors, sat, pattern_size, block)
    return estimate_base_blocks(sp, sp_sat, block=block, top_frac=top_frac, max_rel_std=max_rel_std,
                                chroma_tol=chroma_tol, crop=crop)


def block48_L_p01(
    sp, sp_sat, *, block: int = 24, crop: tuple[int, int, int, int] | None = None, pct: float = 0.1,
) -> list[float]:
    """Per-channel low percentile of ``L = -log10(sp48)`` over valid unsaturated 48x48 blocks inside ``crop``.

    When ``sp`` is at ``block=24`` and has at least 2x2 blocks, pools adjacent 2x2 blocks into 48x48 blocks
    (same scale as the base-estimator superpixels, suppresses single-pixel blue-channel shot noise); otherwise
    uses ``sp`` directly.
    """
    if block == 24 and sp.shape[0] >= 2 and sp.shape[1] >= 2:
        hb, wb = (sp.shape[0] // 2) * 2, (sp.shape[1] // 2) * 2
        sp48 = sp[:hb, :wb].reshape(hb // 2, 2, wb // 2, 2, 3).mean((1, 3), dtype=np.float64)
        sat48 = sp_sat[:hb, :wb].reshape(hb // 2, 2, wb // 2, 2).any((1, 3))
        bsize = 48
    else:
        sp48, sat48, bsize = np.asarray(sp, np.float64), sp_sat, block
    hb, wb, _ = sp48.shape
    valid = ~sat48 & np.all(sp48 > 0, -1)
    if crop is not None:
        t, l, h, w = crop
        rows, cols = np.arange(hb) * bsize, np.arange(wb) * bsize
        inside = ((rows >= t) & (rows + bsize <= t + h))[:, None] & ((cols >= l) & (cols + bsize <= l + w))[None, :]
        if (valid & inside).any():
            valid &= inside
    vals = sp48[valid]
    if len(vals) == 0:
        raise ValueError("no valid unsaturated blocks for quantile black point")
    L = -np.log10(vals)
    return [float(x) for x in np.percentile(L, pct, axis=0)]


def split_log_base(rgb) -> tuple[float, np.ndarray]:
    """Decompose ``log10(rgb)`` into ``(level, chroma)`` with ``sum(chroma) == 0``."""
    lb = np.log10(np.asarray(rgb, np.float64))
    lvl = float(lb.mean())
    return lvl, lb - lvl


def base_deviation(own_rgb, ref_rgb) -> dict:
    """Log-density deviation of ``own_rgb`` relative to ``ref_rgb``: ``dD = -log10(own / ref)`` (+ = own denser)."""
    d = np.log10(np.asarray(ref_rgb, np.float64) / np.asarray(own_rgb, np.float64))
    lvl = float(d.mean())
    ch = d - lvl
    return dict(
        dD=[float(x) for x in d],
        level_diff=lvl,
        chroma_diff=[float(x) for x in ch],
        level_part=float(abs(lvl)),
        chroma_part=float(np.max(np.abs(ch))),
    )


def decoupled_quantile_base(roll_rgb, block_L_p01) -> tuple[tuple[float, float, float], float]:
    """Effective base ``b_eff`` for DQ(c_roll) (T3b): chroma from ``roll_rgb``, shared black point ``d_lo`` from
    the minimum of ``block_L_p01 + c_roll`` across R, G, B. Returns ``(b_eff, d_lo)``."""
    _, c_roll = split_log_base(roll_rgb)
    p01 = np.asarray(block_L_p01, np.float64)
    d_lo = float(np.min(p01 + c_roll))
    b_eff = tuple(float(x) for x in (10.0 ** (c_roll - d_lo)))
    return b_eff, d_lo


def natural_roll_base(
    roll_rgb, block_L_p01, *, own_rgb=None, paper_black_d: float = 0.02,
    softness: float = 0.5, ramp_lo: float = 0.04, ramp_hi: float = 0.10,
) -> tuple[tuple[float, float, float], dict]:
    """Natural roll-base fusion (T4, inspired by Darktable negadoctor paper-black + DQ anti-truncation):

    - Uses the roll chromaticity ``c_roll`` to remove the orange mask.
    - Blends the black point smoothly between the roll's physical base density ``-l_roll`` (which preserves
      a restrained paper-black margin on normal / high-key frames) and the frame's anti-truncation quantile
      floor ``d_lo_min - paper_black_d`` according to the frame's base chroma error ``chroma_part`` and
      ``softness`` in [0, 1] (0 = crisp DQ black point, 1 = full B1 roll black point unless ``d_lo_min < -l_roll``).
    """
    l_roll, c_roll = split_log_base(roll_rgb)
    d_lo_b1 = float(-l_roll)
    p01 = np.asarray(block_L_p01, np.float64)
    d_lo_min = float(np.min(p01 + c_roll))
    d_lo_dq = d_lo_min - float(paper_black_d)
    if d_lo_min < d_lo_b1:
        # Frame is thinner than the roll base in at least one channel: must lower d_lo to avoid negative clipping
        w = 0.0
        d_lo = d_lo_dq
    else:
        if own_rgb is not None:
            dc = base_deviation(own_rgb, roll_rgb)["chroma_part"]
            alpha = float(np.clip((dc - ramp_lo) / max(ramp_hi - ramp_lo, 1e-6), 0.0, 1.0))
        else:
            dc = None
            alpha = 1.0
        # w is the weight on the restrained roll-floor black point d_lo_b1
        w = float(np.clip(1.0 - alpha * (1.0 - 0.35 * float(softness)), 0.0, 1.0))
        if softness >= 0.999 and alpha == 0.0:
            w = 1.0
        d_lo = w * d_lo_b1 + (1.0 - w) * d_lo_dq
    b_eff = tuple(float(x) for x in (10.0 ** (c_roll - d_lo)))
    return b_eff, dict(d_lo_used=float(d_lo), d_lo_b1=d_lo_b1, d_lo_min=d_lo_min,
                       paper_black_d=float(paper_black_d), b1_weight=w)


def estimate_base_blocks(
    sp, sp_sat, *, block: int, top_frac: float = 0.01, max_rel_std: float = 0.03,
    chroma_tol: float = 0.05, crop: tuple[int, int, int, int] | None = None,
) -> BaseEstimate:
    """Steps 2-5 of ``estimate_base_auto`` on precomputed (hb, wb, 3) block means ``sp`` and
    block saturation flags ``sp_sat``; ``block`` is the block size in input pixels. Shared by the
    camera path (CFA-aware blocks) and the scanner path (blocks of a full RGB image)."""
    params = dict(block=block, top_frac=top_frac, max_rel_std=max_rel_std,
                  chroma_tol=chroma_tol, crop=crop)
    hb, wb, _ = sp.shape
    valid = ~sp_sat & np.all(sp > 0, -1)
    if crop is not None:
        t, l, h, w = crop
        rows, cols = np.arange(hb) * block, np.arange(wb) * block
        inside = ((rows >= t) & (rows + block <= t + h))[:, None] & ((cols >= l) & (cols + block <= l + w))[None, :]
        valid &= inside
    uniform = valid & (_neighbourhood_rel_std(sp) <= max_rel_std)
    warnings: list[str] = []
    n_uniform = int(uniform.sum())
    if n_uniform == 0:
        raise ValueError("no uniform unsaturated blocks; specify the film base region manually")

    Y = sp.mean(-1)
    n_cand = min(n_uniform, max(10, int(round(top_frac * n_uniform))))
    thr = float(np.sort(Y[uniform])[-n_cand])
    cand = uniform & (Y >= thr)
    idx = np.argwhere(cand)
    vals = sp[cand]
    chroma = np.log2(vals[:, [0, 2]] / vals[:, [1]])
    consensus = np.ones(len(vals), bool)
    for _ in range(2):
        centre = np.median(chroma[consensus], 0)
        consensus = np.all(np.abs(chroma - centre) <= chroma_tol, -1)
        if not consensus.any():
            raise ValueError("no chromatically consistent base candidates; specify region manually")
    outlier_frac = float(1 - consensus.mean())

    keep = consensus
    if outlier_frac > 0.2:
        warnings.append(f"{outlier_frac:.0%} of the brightest candidates rejected by chromaticity check; "
                        "possible light bypassing the film - verify base region")

    kept = vals[keep]
    rgb = tuple(float(x) for x in np.median(kept, 0))
    regions = [(int(r * block), int(c * block), block, block) for r, c in idx[keep]]
    stats = {}
    for i, name in enumerate(CHANNELS):
        p05, p95 = np.percentile(kept[:, i], [5, 95])
        stats[name] = dict(n_blocks=int(len(kept)), median=rgb[i], p05=float(p05), p95=float(p95),
                           rel_spread=float((p95 - p05) / rgb[i]))
        if stats[name]["rel_spread"] > 0.1:
            warnings.append(f"{name} base blocks spread {stats[name]['rel_spread']:.1%} (p05-p95)")
    sat_blocks = int(sp_sat.sum())
    if sat_blocks:
        warnings.append(f"{sat_blocks} blocks contain saturated samples (excluded)")
    kr, kc = idx[keep][:, 0] * block, idx[keep][:, 1] * block
    diagnostics = dict(
        n_blocks=hb * wb, n_saturated_blocks=sat_blocks, n_uniform=n_uniform, n_candidates=int(len(vals)),
        n_kept=int(keep.sum()), chroma_outlier_frac=outlier_frac,
        chroma_centre_log2_RG_BG=[float(x) for x in centre],
        rejected_mean_rgb=[float(x) for x in vals[~keep].mean(0)] if (~keep).any() else None,
        kept_bbox=[int(kr.min()), int(kc.min()), int(kr.max() + block), int(kc.max() + block)],
        brightness_threshold=thr,
    )
    return BaseEstimate(rgb=rgb, method="auto", regions=regions, stats=stats, params=params,
                        diagnostics=diagnostics, warnings=warnings)


@dataclass
class RollBase:
    rgb_norm: tuple[float, float, float]       # base at the reference exposure time t_ref
    t_ref: float
    members: list[str]                          # frames whose interior minimum density forms the floor cluster
    used: list[str]                             # most transparent members, averaged into rgb_norm
    diagnostics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def for_exposure(self, t: float) -> tuple[float, float, float]:
        """Base in the raw signal of a frame captured with exposure time ``t``."""
        if not (np.isfinite(t) and t > 0):
            raise ValueError(f"exposure time must be positive, got {t}")
        return tuple(float(x * t / self.t_ref) for x in self.rgb_norm)

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_base_roll(candidates: dict, exposures: dict, *, t_ref: float | None = None,
                       level_window: float = 0.1, chroma_tol: float = 0.05, n_use: int = 3,
                       min_members: int = 3) -> RollBase:
    """Film base for a roll scanned without film border, from per-frame interior candidates.

    ``candidates``: frame -> linear RGB of the frame's most transparent uniform area (e.g.
    ``estimate_base_auto(...).rgb``); ``exposures``: frame -> exposure time of the scan. All frames
    must share ISO, aperture, lens and light source; only exposure time is normalised (to ``t_ref``,
    default the median exposure).

    A frame's interior minimum is base only where the scene left the film unexposed; elsewhere it is
    denser and usually coloured. Areas below the film's toe converge to the colour of base + fog, so:
    1. frames within ``level_window`` (density, G channel) of the most transparent frame, and
    2. within ``chroma_tol`` (log2 R/G, B/G) of their median chromaticity, form the floor cluster;
    3. base = per-channel median of the ``n_use`` most transparent cluster members.
    The result is the least dense film observed on the roll: an upper bound on the base density, equal
    to it only if some frame reached the floor. Fewer than ``min_members`` cluster frames -> warning.
    """
    names = sorted(candidates)
    if not names:
        raise ValueError("no frames")
    if set(names) != set(exposures):
        raise ValueError("candidates and exposures must cover the same frames")
    rgb = np.array([candidates[n] for n in names], np.float64)
    t = np.array([exposures[n] for n in names], np.float64)
    if rgb.shape[1:] != (3,) or not np.all(np.isfinite(rgb)) or np.any(rgb <= 0):
        raise ValueError("candidate RGB values must be finite and positive")
    if not np.all(np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("exposure times must be finite and positive")
    t_ref = float(np.median(t)) if t_ref is None else float(t_ref)
    norm = rgb * (t_ref / t)[:, None]
    dG = np.log10(norm[:, 1].max() / norm[:, 1])          # density above the most transparent frame
    chroma = np.log2(norm[:, [0, 2]] / norm[:, [1]])
    near = dG <= level_window
    centre = np.median(chroma[near], 0)
    member = near & np.all(np.abs(chroma - centre) <= chroma_tol, -1)
    if not member.any():
        raise ValueError("no frames form a consistent minimum-density cluster; specify the base manually")
    centre = np.median(chroma[member], 0)
    order = [i for i in np.argsort(dG) if member[i]]
    use = order[:n_use]
    base = tuple(float(x) for x in np.median(norm[use], 0))
    warnings = []
    if member.sum() < min_members:
        warnings.append(f"only {int(member.sum())} frame(s) reach the minimum-density cluster; the roll may "
                        "never reach the film base, so the estimate may be a scene shadow")
    diagnostics = dict(
        level_window=level_window, chroma_tol=chroma_tol, n_use=n_use,
        cluster_chroma_log2_RG_BG=[float(x) for x in centre],
        cluster_chroma_spread=[float(x) for x in chroma[member].max(0) - chroma[member].min(0)],
        cluster_density_spread_G=float(dG[member].max()),
        frames={n: dict(density_above_floor=[float(x) for x in np.log10(np.array(base) / norm[i])],
                        chroma_offset=[float(x) for x in chroma[i] - centre], member=bool(member[i]))
                for i, n in enumerate(names)},
    )
    return RollBase(rgb_norm=base, t_ref=t_ref, members=[names[i] for i in order],
                    used=[names[i] for i in use], diagnostics=diagnostics, warnings=warnings)
