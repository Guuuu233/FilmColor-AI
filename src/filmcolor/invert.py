"""Density computation, negative inversion and basic tone mapping. Pure numpy.

Signal chain (definitions in docs/ENGINE_MATH.md):

T   (H, W, 3) demosaiced linear camera RGB of the scan, 0 = black level, 1 = white level.
b   (3,) film base in the same signal (``base_from_regions``).
t   = T / b                       transmittance relative to base (1 at base).
D   = -log10(max(t, eps))         density above base (0 at base, > 0 where exposed).
s   = (D - D_lo) / (D_hi - D_lo)  per-channel normalised density (0 = black point, 1 = white point).
P   = (10**(R*(s-1)) - 10**-R) / (1 - 10**-R),  R = (D_hi_G - D_lo_G) / gamma
                                  linear positive, 0 at the black point, 1 at the white point.
O   = srgb_oetf(clip(exposure * P, 0, 1))   display encoding for viewing.

P and O are in uncalibrated camera RGB (scan camera + film dyes); no colour matrix is applied.
"""
from __future__ import annotations

import atexit
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field

import numpy as np

from .raw import CHANNELS, check_region

EPS = 1e-6  # floor for transmittance before log10 (render path clips below-black to 0)
_EPS_F32 = np.float32(EPS)
CHUNK_ROWS = 256  # rows per block when inverting a full image; density and inversion are per pixel
PIXEL_THREADS = max(1, min(6, os.cpu_count() or 1))  # row blocks processed concurrently (numpy releases the GIL)
_POOL: ThreadPoolExecutor | None = None
_POOL_WORKERS = 0
_POOL_LOCK = threading.Lock()
_IN_POOL = threading.local()


def get_pixel_pool(max_workers: int | None = None) -> ThreadPoolExecutor:
    """Shared persistent thread pool for row-chunked numpy/scipy/PIL ops (avoids per-call pthread churn)."""
    global _POOL, _POOL_WORKERS
    want = max(1, int(max_workers or PIXEL_THREADS))
    with _POOL_LOCK:
        if _POOL is None or _POOL_WORKERS != want:
            old = _POOL
            _POOL = ThreadPoolExecutor(max_workers=want)
            _POOL_WORKERS = want
            if old is not None:
                old.shutdown(wait=False)
        return _POOL


def _shutdown_pixel_pool() -> None:
    global _POOL
    if _POOL is not None:
        _POOL.shutdown(wait=False)
        _POOL = None


atexit.register(_shutdown_pixel_pool)


def map_row_blocks(n_rows: int, fn, chunk_rows: int) -> list:
    """``[fn(r0) for r0 in range(0, n_rows, chunk_rows)]`` on up to ``PIXEL_THREADS`` threads, results in block order.
    Only for per-pixel work whose blocks do not interact (each ``fn`` writes its own rows and returns its own
    statistics), so the values are those of the serial loop. ``chunk_rows`` is required and must be the stride
    ``fn`` slices with: read ``CHUNK_ROWS`` once at the call site (a default would freeze it at import time)."""
    starts = range(0, n_rows, chunk_rows)
    if PIXEL_THREADS > 1 and len(starts) > 1 and not getattr(_IN_POOL, "active", False):
        def _wrapped(r0: int):
            _IN_POOL.active = True
            try:
                return fn(r0)
            finally:
                _IN_POOL.active = False

        return list(get_pixel_pool(PIXEL_THREADS).map(_wrapped, starts))
    return [fn(r0) for r0 in starts]


def _all_finite(arr: np.ndarray) -> bool:
    if arr.size == 0:
        return True
    if np.isfinite(arr.sum(dtype=np.float64)):
        return True
    return bool(np.all(np.isfinite(arr)))


@dataclass
class InversionParams:
    base: tuple[float, float, float]
    d_lo: tuple[float, float, float] = (0.0, 0.0, 0.0)   # black point density per channel
    d_hi: tuple[float, float, float] = (1.0, 1.0, 1.0)   # white point density per channel
    gamma: float = 0.6                                    # film gamma used to set the log range R
    exposure: float = 1.0                                 # linear gain applied before display encoding
    spike_clamp: tuple[float, float] | None = None        # T4l: (mid_max, margin) for clamp_spikes; None = off

    def __post_init__(self):
        # plain floats so the parameters serialise to the sidecar whatever the caller passed
        for name in ("base", "d_lo", "d_hi"):
            setattr(self, name, tuple(float(x) for x in getattr(self, name)))
        if self.spike_clamp is not None:
            self.spike_clamp = tuple(float(x) for x in self.spike_clamp)
        self.gamma, self.exposure = float(self.gamma), float(self.exposure)
        if not np.all(np.isfinite([*self.base, *self.d_lo, *self.d_hi, self.gamma, self.exposure])):
            raise ValueError("inversion parameters must be finite")
        if np.any(np.asarray(self.base) <= 0):
            raise ValueError(f"film base must be positive, got {self.base}")
        if np.any(np.asarray(self.d_hi) <= np.asarray(self.d_lo)):
            raise ValueError(f"white point density must exceed black point: d_lo={self.d_lo} d_hi={self.d_hi}")
        if self.gamma <= 0 or self.exposure <= 0:
            raise ValueError("gamma and exposure must be positive")

    @property
    def log_range(self) -> float:
        return float((self.d_hi[1] - self.d_lo[1]) / self.gamma)

    def to_dict(self) -> dict:
        d = asdict(self) | {"log_range": self.log_range}
        if self.spike_clamp is None:  # sidecars without the clamp keep their previous field set
            del d["spike_clamp"]
        else:
            d["spike_clamp"] = list(self.spike_clamp)
        return d


def base_from_regions(rgb, regions) -> dict:
    """Film base in a 3-channel image: per-channel median over the union of regions.

    Use on the same signal that is inverted (the render path); same-colour CFA positions differ
    in sensitivity, so analysis-path and render-path channel means differ slightly
    (experiments/s1_diag/b_bias_notes.md).
    """
    if not regions:
        raise ValueError("at least one region is required")
    mask = np.zeros(rgb.shape[:2], bool)
    for r in regions:
        check_region(r, rgb.shape[:2])
        t, l, h, w = r
        mask[t : t + h, l : l + w] = True
    px = rgb[mask]
    if len(px) == 0:
        raise ValueError("base regions contain no pixels")
    med = np.median(px, 0)
    p05, p95 = np.quantile(px, [0.05, 0.95], axis=0)
    denom = np.where(np.abs(med) > EPS, med, EPS)
    return dict(rgb=tuple(float(x) for x in med), n=int(len(px)),
                rel_spread=[float(x) for x in (p95 - p05) / denom])


def central_region(shape, frac: float = 0.5) -> tuple[int, int, int, int]:
    """Default image region: centred rectangle covering ``frac`` of each dimension.

    Assumes the frame is roughly centred. Scans can include an opaque holder mask (very high
    density) and rebate/backlight around the frame; both must stay out of the white point.
    """
    h, w = int(shape[0] * frac), int(shape[1] * frac)
    return ((shape[0] - h) // 2, (shape[1] - w) // 2, h, w)


def region_mask(shape, region) -> np.ndarray:
    check_region(region, shape)
    t, l, h, w = region
    m = np.zeros(shape, bool)
    m[t : t + h, l : l + w] = True
    return m


def density(rgb, base) -> np.ndarray:
    b = np.asarray(base, np.float32)
    out = np.divide(rgb, b, dtype=np.float32) if np.asarray(rgb).dtype == np.float32 else (rgb / b)
    if out.dtype == np.float32:
        np.maximum(out, _EPS_F32, out=out)
        np.log10(out, out=out)
        np.negative(out, out=out)
        return out
    return -np.log10(np.maximum(out, EPS))


def neutral_transmission(rgb, base) -> np.ndarray:
    """Black-and-white negatives: one transmittance per pixel, the geometric mean of the
    base-normalised channel transmittances (so its density is the mean channel density), copied to
    all three channels. Invert it with base (1, 1, 1); the positive is then neutral by construction.
    Silver density is assumed spectrally flat enough that channel differences are noise and light
    or sensor imbalance, not image content."""
    b = np.asarray(base, np.float32)
    out = np.empty(rgb.shape, np.float32)
    for r0 in range(0, rgb.shape[0], CHUNK_ROWS):
        t = np.maximum(rgb[r0 : r0 + CHUNK_ROWS] / b, EPS)
        out[r0 : r0 + CHUNK_ROWS] = np.cbrt(t.prod(-1))[..., None]
    return out


def estimate_white_point(D, mask=None, pct: float = 99.9) -> tuple[float, float, float]:
    """Per-channel high percentile of density over ``mask`` (default all pixels).

    Assumes the densest image areas (scene highlights) are neutral, which balances channel contrast.
    """
    if not 0 < pct <= 100:
        raise ValueError(f"white point percentile must be in (0, 100], got {pct}")
    vals = D[mask] if mask is not None else D.reshape(-1, 3)
    if len(vals) == 0:
        raise ValueError("no pixels to estimate the white point from")
    return tuple(float(x) for x in np.percentile(vals, pct, 0))


MIN_WHITE_BLOCKS = 100  # fewer valid blocks than this: the block white point falls back to single pixels


def check_white_point(d_hi, d_lo=None) -> None:
    """Raise ``ValueError`` when the white point has no density above the black point in some channel.

    That happens on frames without an image: blank or unexposed film, a capture of the light source
    without film (the film base estimate then lands on the light itself), or a wrong image region or film
    base. The inversion log range would be zero there (``1 / (1 - 10**-R)`` divides by zero)."""
    lo = d_lo or (0.0, 0.0, 0.0)
    flat = [c for c, hi, l in zip("RGB", d_hi, lo) if hi <= l]
    if flat:
        raise ValueError(f"no density above the black point in channel(s) {''.join(flat)} of the image "
                         "region: blank or unexposed frame (or the light source without film), "
                         "or wrong image region / film base")


def block_white_point(rgb, base, mask, block: int, pct: float) -> tuple[tuple[float, float, float], dict]:
    """White point on ``block`` x ``block`` means of the linear signal (T4b): per-channel ``pct`` percentile of the
    density of block means, over blocks whose pixels are all in ``mask``. Averaging in the linear domain removes the
    read / shot noise that pushes single-pixel high-density quantiles up where the negative is densest and the
    signal is a few ADU above black (blue channel behind the orange mask). Falls back to single pixels when fewer
    than ``MIN_WHITE_BLOCKS`` blocks are valid (small images)."""
    if block < 1:
        raise ValueError(f"white point block must be >= 1, got {block}")
    h, w = (rgb.shape[0] // block) * block, (rgb.shape[1] // block) * block
    nb_y, nb_x = h // block, w // block
    sums = np.empty((nb_y, nb_x, 3), np.float64)
    ok = np.empty((nb_y, nb_x), dtype=bool)
    step = max(block, CHUNK_ROWS // block * block)  # whole blocks per row chunk keep memory small

    def _accum(r0: int) -> None:
        r1 = min(h, r0 + step)
        b0, b1 = r0 // block, r1 // block
        blk = rgb[r0:r1, :w].reshape(b1 - b0, block, nb_x, block, 3)
        sums[b0:b1] = blk.sum(axis=3, dtype=np.float32).sum(axis=1, dtype=np.float64)
        ok[b0:b1] = mask[r0:r1, :w].reshape(b1 - b0, block, nb_x, block).all(axis=3).all(axis=1)

    map_row_blocks(h, _accum, step)
    n = int(ok.sum())
    if n < MIN_WHITE_BLOCKS:
        return estimate_white_point(density(rgb[mask], base), None, pct), dict(method="pixel_fallback", block=block,
                                                                              pct=pct, n_blocks=n)
    means = (sums[ok] / (block * block)).astype(np.float32)
    d_hi = estimate_white_point(density(means, base), None, pct)
    info = dict(method="block_mean", block=block, pct=pct, n_blocks=n)

    # T4o/T4t: dense opaque holders or surrounding warm slide mounts can intrude into the central mask
    if d_hi[1] > 1.80 or (d_hi[1] > 1.15 and d_hi[0] > d_hi[1] + 0.25):
        grid_means = (sums / (block * block)).astype(np.float32)
        D_grid = density(grid_means, base)
        D_sort = np.sort(D_grid, axis=-1)
        if d_hi[1] > 1.80:
            hot = (D_sort[..., 1] > 1.85) & (D_sort[..., 2] > 2.10)
        else:
            cand = (D_sort[..., 1] > 1.05) & (D_sort[..., 2] > 1.45)
            edge_ring = np.concatenate([cand[0], cand[-1], cand[:, 0], cand[:, -1]])
            hot = cand if float(edge_ring.mean()) >= 0.75 else np.zeros_like(cand)
        if hot.any():
            from scipy import ndimage

            lab, n_features = ndimage.label(hot)
            if n_features > 0:
                edge_labels = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
                edge_labels = edge_labels[edge_labels > 0]
                if len(edge_labels) > 0:
                    edge_connected = np.isin(lab, edge_labels)
                    holder_blocks = ndimage.binary_dilation(edge_connected, iterations=2)
                    excluded = ok & holder_blocks
                    n_excluded = int(excluded.sum())
                    if n_excluded > 0:
                        if (ok & ~holder_blocks & hot).any():
                            # T4z: bridge a thin window mullion, retaining only the original dense blocks.
                            bridged, _ = ndimage.label(ndimage.binary_dilation(hot, iterations=1))
                            edge_ids = np.unique(np.concatenate([bridged[0], bridged[-1], bridged[:, 0], bridged[:, -1]]))
                            edge_ids = edge_ids[edge_ids > 0]
                            edge_hot = np.isin(bridged, edge_ids) & hot
                            holder_blocks = ndimage.binary_dilation(edge_hot, iterations=2)
                            n_excluded = int((ok & holder_blocks).sum())
                        ok_clean = ok & ~holder_blocks
                        n_clean = int(ok_clean.sum())
                        if n_clean >= MIN_WHITE_BLOCKS:
                            means_clean = (sums[ok_clean] / (block * block)).astype(np.float32)
                            d_hi = estimate_white_point(density(means_clean, base), None, pct)
                            info["n_blocks"] = n_clean
                            info["holder_blocks_excluded"] = n_excluded

    # T5f: fine sub-block highlights (small bright canopy sky gaps < 16x16 px, A623.tif) smoothed away by 16x16 means
    if block >= 16 and block % 4 == 0 and d_hi[1] >= 0.45 and not info.get("holder_blocks_excluded"):
        b4 = block // 4
        wb = w // block

        def _fine_block(r0: int):
            r1 = min(h, r0 + step)
            b0, b1 = r0 // block, r1 // block
            return (rgb[r0:r1, :w].reshape(b1 - b0, 4, b4, wb, 4, b4, 3)
                    .mean(axis=(2, 5), dtype=np.float64)
                    .transpose(0, 2, 1, 3, 4)[ok[b0:b1]].reshape(-1, 3).astype(np.float32))

        m4_parts = [sub for sub in map_row_blocks(h, _fine_block, step) if len(sub)]
        m4 = np.concatenate(m4_parts, axis=0) if m4_parts else np.empty((0, 3), np.float32)
        D4 = density(m4, base)
        d4_990 = estimate_white_point(D4, None, 99.0)
        if d4_990[1] - d_hi[1] >= 0.10 and d4_990[1] / d_hi[1] >= 1.15:
            d_hi_own = list(d_hi)
            d_hi = estimate_white_point(D4, None, 99.5)
            info["fine_highlight_rescue"] = dict(sub_block=b4, pct=99.5, d4_p990_g=float(d4_990[1]),
                                                 d_hi_own=d_hi_own)

    # T4t/T5e: a non-green white point far above G (or B > 1.95 * G when G >= 0.55 on 8-bit highlights, A831/A832)
    if d_hi[1] >= 0.20 and (any(d_hi[c] > max(2.5, 2.2 * d_hi[1]) for c in (0, 2))
                            or (d_hi[1] >= 0.55 and d_hi[2] > 1.95 * d_hi[1])):
        d_list = list(d_hi)
        capped = {}
        for c, name in ((0, "R"), (2, "B")):
            if d_list[c] > max(2.5, 2.2 * d_list[1]) or (c == 2 and d_list[1] >= 0.55 and d_list[2] > 1.95 * d_list[1]):
                capped[name] = float(d_list[c])
                d_list[c] = float(1.45 * d_list[1])
        d_hi = tuple(d_list)
        info["channel_cap"] = capped

    return d_hi, info


def neutral_ratio_white_point(d_hi, ratio, *, anchor: float = 0.7, scale: float = 0.94,
                              margin: tuple[float, float] = (0.0, 0.03)) -> tuple[tuple[float, float, float], dict]:
    """T4c: keep a blue sky from being taken as white. ``ratio`` = (D_R / D_G, D_B / D_G) of neutral greys of the roll
    (app: median over the roll of each frame's 0.25 <= D_G <= 0.75 density ratios). When the frame's white point has
    the signature of blue sky - R / G below ``ratio[0] - margin[0]`` and B / G above ``ratio[1] + margin[1]`` - its
    channel ratios are moved ``anchor`` of the way to ``ratio`` and the level set from the frame's own white point:
    d_hi_G = ``scale`` * max_c(d_hi_c / r_c). Otherwise ``d_hi`` is returned unchanged."""
    d = np.asarray(d_hi, np.float64)
    dg = max(float(d[1]), EPS)
    own = np.array([d[0] / dg, 1.0, d[2] / dg])
    roll = np.array([float(ratio[0]), 1.0, float(ratio[1])])
    info = dict(ratio_roll=[float(roll[0]), float(roll[2])], ratio_own=[float(own[0]), float(own[2])], anchor=anchor,
                scale=scale, margin=list(margin), applied=False)
    if not (own[0] < roll[0] - margin[0] and own[2] > roll[2] + margin[1]):
        return tuple(float(x) for x in d), info
    r_eff = (1 - anchor) * own + anchor * roll
    g = scale * float(np.max(d / r_eff))
    out = tuple(float(x) for x in r_eff * g)
    return out, dict(info, applied=True, ratio_used=[float(r_eff[0]), float(r_eff[2])], d_hi_own=[float(x) for x in d])


def blue_white_guard(d_hi, guard, base=None) -> tuple[tuple[float, float, float], dict]:
    """T4k: when d_hi_B / d_hi_G < min_ratio, d_hi_B moves ``weight`` of the way to
    max(d_hi_B, target_ratio x d_hi_G). ``guard`` = (min_ratio, target_ratio, weight), optionally followed by
    max_base_bg (T4m): only bases with B/G below that threshold pass the orange-mask gate. Colour film only."""
    values = [float(x) for x in guard]
    if len(values) not in (3, 4):
        raise ValueError("blue white guard needs 3 or 4 values")
    lo, target, weight = values[:3]
    d = [float(x) for x in d_hi]
    ratio = d[2] / max(d[1], EPS)
    info = dict(guard=values, ratio_bg=ratio, applied=False)
    if d[1] <= 0.0:
        return tuple(d), info
    if len(values) == 4 and base is not None:
        bg = float(base[2] / max(base[1], EPS))
        info["base_bg"] = bg
        if bg >= values[3]:
            return tuple(d), info
    if ratio < 0.55:  # T4x: retain low-R/G flare rejection; bounded rescue for collapsed narrow-blue spans
        if d[0] > 0.96 * d[1] and ratio < 0.28:
            d_b = d[2] + 0.85 * (0.28 * d[1] - d[2])
            return (d[0], d[1], d_b), dict(info, applied=True, narrow_blue_rescue=True,
                                          d_hi_b_own=d[2], ratio_bg_used=d_b / d[1])
        return tuple(d), dict(info, reason="below_valid_ratio")
    if ratio >= lo or (len(values) == 4 and base is not None and base[0] <= 0.70 * base[1] and d[0] <= 0.70 * d[1]):
        return tuple(d), info
    d_b = d[2] + weight * (max(d[2], target * d[1]) - d[2])
    return (d[0], d[1], d_b), dict(info, applied=True, d_hi_b_own=d[2], ratio_bg_used=d_b / d[1])


SELF_RATIO_BAND = (0.20, 0.75)  # T4l: midtones for self_ratio_white_point, as shares of the frame's own d_hi_G
SELF_RATIO_MIN_PX = 500


def self_ratio_white_point(d_hi, Dm, guard) -> tuple[tuple[float, float, float], dict]:
    """T4l: a channel's white point collapsed against the frame's own midtones. ``Dm`` (n, 3): densities of the image
    pixels; ``guard`` = (collapse, target). r_mid = median of D_c / D_G over the pixels with
    SELF_RATIO_BAND[0] x d_hi_G <= D_G <= SELF_RATIO_BAND[1] x d_hi_G (all channels above base), the device's and film's
    own channel ratio (camera scan of C-41 about 0.75 / 1.3, BlueNeg scanner DNG about 0.4-0.7 for B / G); a channel
    whose white point ratio d_hi_c / d_hi_G is below collapse x r_mid (no white object: a warm ridge lacks blue, a
    blue sky lacks red) is set to target x r_mid x d_hi_G. Fewer than SELF_RATIO_MIN_PX midtone pixels: unchanged."""
    collapse, target = (float(x) for x in guard)
    d = np.asarray(d_hi, np.float64)
    dg = max(float(d[1]), EPS)
    lo, hi = SELF_RATIO_BAND[0] * dg, SELF_RATIO_BAND[1] * dg
    sel = (Dm[:, 1] >= lo) & (Dm[:, 1] <= hi) & (Dm[:, 0] > 0) & (Dm[:, 2] > 0)
    info = dict(guard=[collapse, target], band=list(SELF_RATIO_BAND), n_mid=int(sel.sum()), applied=False)
    if int(sel.sum()) < SELF_RATIO_MIN_PX:
        return tuple(float(x) for x in d), info
    m = Dm[sel].astype(np.float64)
    r_mid = np.median(m / m[:, 1:2], axis=0)
    r_hi = d / dg
    low = r_hi < collapse * r_mid
    low[1] = False
    info.update(ratio_mid=[float(r_mid[0]), float(r_mid[2])], ratio_hi=[float(r_hi[0]), float(r_hi[2])])
    if not low.any():
        return tuple(float(x) for x in d), info
    out = np.where(low, target * r_mid * d[1], d)
    return tuple(float(x) for x in out), dict(info, applied=True, channels=[c for c, x in zip("RGB", low) if x],
                                              d_hi_own=[float(x) for x in d])


def clamp_spikes(s, clamp, *, zero_escape: bool = False) -> tuple[np.ndarray, int]:
    """T4l/T4r: single-channel spikes of the normalised density ``s`` (..., 3). Where the middle channel is at most
    ``clamp[0]`` (not an opaque holder, which is dense in two or more channels and stays for ``holder.holder_mask``),
    one channel is held to max(middle, 1) + ``clamp[1]``: a signal at or below the black level (blue behind the orange
    mask, A1106) gives the density floor 6 D and a positive tens of thousands of times past white.
    T4r/T4s: when escape pixels (mid > mid_max, min <= 1.6, max > 5) exceed 1% of the array or ``zero_escape`` is True,
    pixels with min <= 1.6 are also clamped to min(bound_z, bound).
    Returns the clamped array and the number of pixels changed."""
    mid_max, margin = (np.float32(x) for x in clamp)
    one_plus_m = np.float32(1.0) + margin
    if s.shape[-1] == 3:
        r, g, b = s[..., 0], s[..., 1], s[..., 2]
        # Exact 3-channel min/mid/max without np.sort allocation (~4-6x faster, bit-exact)
        rg_min = np.minimum(r, g)
        rg_max = np.maximum(r, g)
        s_max_2d = np.maximum(rg_max, b)
        # Fast path: no channel exceeds 1.0 + margin -> bound >= 1.0 + margin everywhere -> 0 clamped pixels
        if not bool((s_max_2d > one_plus_m).any()):
            return s, 0
        s_min_2d = np.minimum(rg_min, b)
        s_mid_2d = np.minimum(rg_max, np.maximum(rg_min, b))
        s_min = s_min_2d[..., None]
        s_mid = s_mid_2d[..., None]
        s_max = s_max_2d[..., None]
    else:
        s_sort = np.sort(s, axis=-1)
        s_min = s_sort[..., 0:1]
        s_mid = s_sort[..., 1:2]
        s_max = s_sort[..., 2:3]
        if not bool((s_max > one_plus_m).any()):
            return s, 0

    bound = np.where(s_mid > mid_max, np.float32(np.inf), np.maximum(s_mid, np.float32(1.0)) + margin)

    if not zero_escape:
        escape = (s_mid > mid_max) & (s_min <= np.float32(1.6)) & (s_max > np.float32(5.0))
        zero_escape = float(np.mean(escape)) > 0.01
    if zero_escape:
        bound_z = np.maximum(s_min, np.float32(1.0)) + margin
        bound = np.where(s_min <= np.float32(1.6), np.minimum(bound_z, bound), bound)

    over = s > bound
    if not over.any():
        return s, 0
    return np.minimum(s, bound), int(over.any(axis=-1).sum())


def invert(D, p: InversionParams, stats: dict | None = None, *, zero_escape: bool = False) -> np.ndarray:
    """P from density ``D``; with ``p.spike_clamp``, single-channel spikes are clamped first (``clamp_spikes``; the
    number of pixels changed is added to ``stats["spike_px"]`` when ``stats`` is given)."""
    lo, hi = np.asarray(p.d_lo, np.float32), np.asarray(p.d_hi, np.float32)
    s = (D - lo) / (hi - lo)
    if p.spike_clamp is not None and s.ndim >= 1 and s.shape[-1] == 3:
        s, n = clamp_spikes(s, p.spike_clamp, zero_escape=zero_escape)
        if stats is not None:
            stats["spike_px"] = stats.get("spike_px", 0) + n
    R = np.float32(p.log_range)
    floor = np.float32(10.0 ** -p.log_range)
    # T4n: a tiny channel white point can send holder densities past float32's exponent range.
    exp = np.minimum(R * (s - 1), np.float32(12.0))
    return (np.power(np.float32(10.0), exp) - floor) / (1 - floor)


def srgb_oetf(x):
    arr = np.asarray(x)
    if arr.ndim > 0 and arr.dtype == np.float32:
        xc = np.clip(arr, np.float32(0.0), np.float32(1.0))
        out = xc * np.float32(12.92)
        hi = xc > np.float32(0.0031308)
        if hi.any():
            out[hi] = np.float32(1.055) * np.power(xc[hi], np.float32(1.0 / 2.4)) - np.float32(0.055)
        return out
    x = np.clip(arr, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def tone_map(P, p: InversionParams) -> np.ndarray:
    return srgb_oetf(P * np.float32(p.exposure)).astype(np.float32)


def range_report(D, P, p: InversionParams, mask=None) -> dict:
    """Effective dynamic range diagnostics per channel over ``mask``."""
    Dm = D[mask] if mask is not None else D.reshape(-1, 3)
    Pm = P[mask] if mask is not None else P.reshape(-1, 3)
    if len(Dm) == 0:
        raise ValueError("no pixels for range report")
    q01, q999 = np.quantile(Dm, [0.001, 0.999], axis=0)
    clip_thr = np.float32(1.0 / p.exposure)
    clip_high = np.mean(Pm > clip_thr, axis=0)
    below_black = np.mean(Pm < np.float32(0.0), axis=0)
    brighter_base = np.mean(Dm < np.float32(0.0), axis=0)
    out = {}
    for c, name in enumerate(CHANNELS):
        out[name] = dict(
            clip_high=float(clip_high[c]),          # brighter than output white
            below_black=float(below_black[c]),      # less dense than the black point
            brighter_than_base=float(brighter_base[c]),
            density_p01=float(q01[c]),
            density_p999=float(q999[c]),
        )
    return out


@dataclass
class InversionResult:
    positive: np.ndarray        # P, linear
    params: InversionParams
    report: dict = field(default_factory=dict)

    @property
    def display(self) -> np.ndarray:
        """O, sRGB-encoded for viewing; computed on access (the pipeline renders P instead)."""
        return tone_map(self.positive, self.params)


def invert_negative(rgb, base, *, mask=None, white_pct: float = 99.9, gamma: float = 0.6,
                    exposure: float = 1.0, d_lo=None, d_hi=None, white_block: int | None = None,
                    white_ratio=None, white_b_guard=None, white_self_ratio=None,
                    spike_clamp=None) -> InversionResult:
    """Full chain with automatic white point unless ``d_hi`` is given. ``mask`` selects image pixels
    used for the white point and the report (exclude borders, rebate, holder and saturated areas).
    ``white_block``: estimate the automatic white point on block means of that size (``block_white_point``)
    instead of single pixels (None, the default: single pixels, unchanged). ``white_ratio``: roll neutral density
    ratios (D_R / D_G, D_B / D_G) checked against the automatic white point (``neutral_ratio_white_point``).
    ``white_b_guard``: (min_ratio, target_ratio, weight[, max_base_bg]) for ``blue_white_guard`` on the automatic white point.
    ``white_self_ratio``: (collapse, target) for ``self_ratio_white_point`` on the automatic white point (T4l).
    ``spike_clamp``: (mid_max, margin) for ``clamp_spikes`` (T4l), kept in the parameters (replayed on export)."""
    if not _all_finite(np.asarray(rgb)):
        raise ValueError("input image contains NaN or infinite values")
    if not np.all(np.isfinite(base)) or np.any(np.asarray(base) <= 0):
        raise ValueError(f"film base must be positive and finite, got {tuple(base)}")
    # global statistics only need the masked pixels; their density is computed once on that
    # selection (same values, same order as density(rgb)[mask]) and the full image in row blocks
    Dm = density(rgb[mask] if mask is not None else rgb.reshape(-1, 3), base)
    white_info = None
    if d_hi is None:
        if white_block is not None and rgb.ndim == 3:
            m = mask if mask is not None else np.ones(rgb.shape[:2], bool)
            d_hi, white_info = block_white_point(rgb, base, m, int(white_block), white_pct)
        else:
            d_hi = estimate_white_point(Dm, None, white_pct)
        if white_ratio is not None:
            d_hi, ratio_info = neutral_ratio_white_point(d_hi, white_ratio)
            white_info = dict(white_info or dict(method="pixel", pct=white_pct), neutral_ratio=ratio_info)
        if white_self_ratio is not None:
            d_hi, self_info = self_ratio_white_point(d_hi, Dm, white_self_ratio)
            white_info = dict(white_info or dict(method="pixel", pct=white_pct), self_ratio=self_info)
        if white_b_guard is not None:
            d_hi, guard_info = blue_white_guard(d_hi, white_b_guard, base=base)
            white_info = dict(white_info or dict(method="pixel", pct=white_pct), blue_guard=guard_info)
        check_white_point(d_hi, d_lo)
    p = InversionParams(base=tuple(float(x) for x in base), d_lo=tuple(d_lo or (0.0, 0.0, 0.0)),
                        d_hi=tuple(float(x) for x in d_hi), gamma=gamma, exposure=exposure, spike_clamp=spike_clamp)
    lo, hi = np.asarray(p.d_lo, np.float32), np.asarray(p.d_hi, np.float32)
    zero_escape = False
    if spike_clamp is not None and Dm.ndim >= 1 and Dm.shape[-1] == 3 and len(Dm) > 0:
        mid_max = np.float32(spike_clamp[0])
        max_thr = np.float32(5.0 if (white_info or {}).get("holder_blocks_excluded") else 2.0)
        inv_span = (hi - lo).astype(np.float32)
        n_dm = len(Dm)
        esc_count = 0
        dm_step = 262144
        for i0 in range(0, n_dm, dm_step):
            s_blk = (Dm[i0 : i0 + dm_step] - lo) / inv_span
            r, g, b = s_blk[..., 0], s_blk[..., 1], s_blk[..., 2]
            rg_min = np.minimum(r, g)
            rg_max = np.maximum(r, g)
            s_min = np.minimum(rg_min, b)
            s_max = np.maximum(rg_max, b)
            s_mid = np.minimum(rg_max, np.maximum(rg_min, b))
            esc_count += int(np.count_nonzero((s_mid > mid_max) & (s_min <= np.float32(1.6)) & (s_max > max_thr)))
        zero_escape = (esc_count / n_dm) > 0.001
    # share of the whole image denser than the white point: large values flag opaque areas
    # (holder mask) or highlights outside the region used for the white point
    clamp_stats = {}
    if rgb.ndim == 3:
        P = np.empty(rgb.shape, np.float32)
        step = CHUNK_ROWS
        clip_thr = np.float32(1.0 / p.exposure)
        zero_f32 = np.float32(0.0)

        def _block(r0: int):
            D = density(rgb[r0 : r0 + step], base)
            st: dict = {}
            p_blk = invert(D, p, st, zero_escape=zero_escape)
            P[r0 : r0 + step] = p_blk
            if mask is not None:
                pm_blk = p_blk[mask[r0 : r0 + step]]
                n_m = len(pm_blk)
                c_hi = np.count_nonzero(pm_blk > clip_thr, axis=0) if n_m else np.zeros(3, dtype=np.int64)
                c_lo = np.count_nonzero(pm_blk < zero_f32, axis=0) if n_m else np.zeros(3, dtype=np.int64)
            else:
                n_m = p_blk.shape[0] * p_blk.shape[1]
                c_hi = np.count_nonzero(p_blk > clip_thr, axis=(0, 1))
                c_lo = np.count_nonzero(p_blk < zero_f32, axis=(0, 1))
            return np.count_nonzero(D > hi, axis=(0, 1)), st.get("spike_px", 0), n_m, c_hi, c_lo

        blocks = map_row_blocks(rgb.shape[0], _block, step)
        n_above = np.sum([b[0] for b in blocks], axis=0, dtype=np.int64)
        if spike_clamp is not None:
            clamp_stats["spike_px"] = int(sum(b[1] for b in blocks))
        above_white = [float(k / (rgb.shape[0] * rgb.shape[1])) for k in n_above]
        if len(Dm) == 0:
            raise ValueError("no pixels for range report")
        total_m = sum(b[2] for b in blocks)
        clip_high = np.sum([b[3] for b in blocks], axis=0, dtype=np.int64) / total_m
        below_black = np.sum([b[4] for b in blocks], axis=0, dtype=np.int64) / total_m
        q01, q999 = np.quantile(Dm, [0.001, 0.999], axis=0)
        brighter_base = np.mean(Dm < zero_f32, axis=0)
        rng_rep = {
            name: dict(
                clip_high=float(clip_high[c]),
                below_black=float(below_black[c]),
                brighter_than_base=float(brighter_base[c]),
                density_p01=float(q01[c]),
                density_p999=float(q999[c]),
            )
            for c, name in enumerate(CHANNELS)
        }
    else:
        D = density(rgb, base)
        P = invert(D, p, clamp_stats, zero_escape=zero_escape)
        above_white = [float(np.mean(D[..., c] > p.d_hi[c])) for c in range(3)]
        Pm = P[mask] if mask is not None else P.reshape(-1, 3)
        rng_rep = range_report(Dm, Pm, p)
    report = dict(white_pct=white_pct, range=rng_rep,
                  full_image_above_white=above_white)
    if white_info is not None:  # reports of the default single-pixel white point keep their previous field set
        report["white_point"] = white_info
    if spike_clamp is not None:
        n_px = rgb.shape[0] * rgb.shape[1] if rgb.ndim == 3 else int(np.prod(rgb.shape[:-1]))
        report["spike_clamp"] = dict(clamp=list(p.spike_clamp), pixels=int(clamp_stats.get("spike_px", 0)),
                                     share=float(clamp_stats.get("spike_px", 0) / max(n_px, 1)))
    return InversionResult(positive=P, params=p, report=report)


def _t18_smoothstep(x: float, lo: float, hi: float) -> float:
    t = float(np.clip((x - lo) / max(hi - lo, 1e-8), 0.0, 1.0))
    return t * t * (3.0 - 2.0 * t)


def is_orange_mask_ratios(d_hi, d_lo=(0.0, 0.0, 0.0), *, has_cam_matrix: bool = True) -> bool:
    """Physical negative regime detection: camera-matrix RAWs or orange-masked scans (d_hi_b / d_hi_g > 0.80)."""
    if has_cam_matrix:
        return True
    dh = np.asarray(d_hi, np.float64) - np.asarray(d_lo, np.float64)
    dg = max(float(dh[1]), 1e-4)
    return float(dh[2]) / dg > 0.80


def t18_channel_ratios(d_hi, d_lo=(0.0, 0.0, 0.0), *, has_cam_matrix: bool = True,
                       version: str = "t18") -> tuple[float, float, float, float]:
    """T18/T23 decoupled channel slope ratios (rho_r, 1.0, rho_b) and midtone pivot_s."""
    dh = np.asarray(d_hi, np.float64) - np.asarray(d_lo, np.float64)
    dg = max(float(dh[1]), 1e-4)
    r_u = float(dh[0]) / dg
    b_u = float(dh[2]) / dg
    if has_cam_matrix or b_u > 0.80:
        w_r = _t18_smoothstep(0.60 - r_u, 0.0, 0.14)
        w_b = _t18_smoothstep(b_u - 1.35, 0.0, 0.18)
        rr = (1.0 - 0.85 * w_r) * r_u + (0.85 * w_r) * 0.725
        rb = (1.0 - 0.75 * w_b) * b_u + (0.75 * w_b) * 1.18
        pivot_s = 0.50
    else:
        rr = float(np.clip(0.96 * r_u, 0.72, 1.18))
        rb = float(np.clip(1.13 * b_u, 0.28, 0.75))
        pivot_s = 0.62
    return rr, 1.0, rb, pivot_s


UNCAL_T18_EPS = (0.012, -0.008, -0.008)
UNCAL_T23_EPS = (0.012, -0.008, -0.008)


_INV_LOG10_2 = 1.0 / math.log10(2.0)


def _t18_affine_params(d_lo, d_hi, gamma: float, *, has_cam_matrix: bool = True,
                       eps_rgb: tuple[float, float, float] | None = None,
                       version: str = "t18"):
    lo = np.asarray(d_lo, np.float64)
    hi = np.asarray(d_hi, np.float64)
    if float(gamma) <= 0 or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        raise ValueError(f"invalid T18 parameters: d_lo={d_lo}, d_hi={d_hi}, gamma={gamma}")
    if np.any(hi <= lo):
        raise ValueError(f"white point density must exceed black point: d_lo={d_lo} d_hi={d_hi}")
    rr, rg, rb, pivot_s = t18_channel_ratios(hi, lo, has_cam_matrix=has_cam_matrix, version=version)
    rho = np.array([rr, rg, rb], dtype=np.float64)
    dg = max(float(hi[1] - lo[1]), 1e-4)
    d_anchor = lo + pivot_s * (hi - lo) + (1.0 - pivot_s) * rho * dg
    if eps_rgb is None:
        if is_orange_mask_ratios(hi, lo, has_cam_matrix=has_cam_matrix):
            eps_arr = np.zeros(3, dtype=np.float64)
        else:
            eps_arr = np.asarray(UNCAL_T23_EPS if version == "t23" else UNCAL_T18_EPS, dtype=np.float64)
    else:
        eps_arr = np.asarray(eps_rgb, dtype=np.float64)
    R = dg / float(gamma)
    flr = 10.0 ** (-R)
    return lo, hi, rho, dg, d_anchor, eps_arr, R, flr


def _t18_chunk_from_s(s: np.ndarray, R: float, flr: float) -> np.ndarray:
    if not (R > 0.0 and flr < 1.0):
        raise ValueError(f"T18 inversion needs a positive log range, got R={R!r} (white point on the film base)")
    max_e2 = np.float32(12.0 * _INV_LOG10_2)
    r_log2 = np.float32(R * _INV_LOG10_2)
    flr32 = np.float32(flr)
    inv_omf = np.float32(1.0 / (1.0 - flr))
    out = np.asarray(s, np.float32) - np.float32(1.0)
    out *= r_log2
    np.minimum(out, max_e2, out=out)
    np.exp2(out, out=out)
    out -= flr32
    out *= inv_omf
    return out


def invert_t18(D, p: InversionParams, *, has_cam_matrix: bool = True,
               stats: dict | None = None, zero_escape: bool = False,
               eps_rgb: tuple[float, float, float] | None = None,
               version: str = "t18") -> np.ndarray:
    """T18/T23 decoupled-slope inversion from density ``D`` (midtone-pivoted slope ratio rho)."""
    D_arr = np.asarray(D, np.float32)
    if not _all_finite(D_arr):
        raise ValueError("density input contains NaN or infinite values")
    _, _, rho, dg, d_anchor, eps_arr, R, flr = _t18_affine_params(
        p.d_lo, p.d_hi, p.gamma, has_cam_matrix=has_cam_matrix, eps_rgb=eps_rgb, version=version,
    )
    inv_rd = (1.0 / (rho * dg)).astype(np.float32)
    anchor32 = d_anchor.astype(np.float32)
    shift32 = (np.float32(1.0) + eps_arr.astype(np.float32))

    def _run(chunk: np.ndarray) -> np.ndarray:
        s = (chunk - anchor32) * inv_rd
        s += shift32
        if p.spike_clamp is not None and s.ndim >= 1 and s.shape[-1] == 3:
            s, n = clamp_spikes(s, p.spike_clamp, zero_escape=zero_escape)
            if stats is not None:
                stats["spike_px"] = stats.get("spike_px", 0) + n
        return _t18_chunk_from_s(s, R, flr)

    if D_arr.ndim == 3 and D_arr.shape[0] > CHUNK_ROWS:
        out = np.empty(D_arr.shape, np.float32)
        for r0 in range(0, D_arr.shape[0], CHUNK_ROWS):
            out[r0 : r0 + CHUNK_ROWS] = _run(D_arr[r0 : r0 + CHUNK_ROWS])
        return out
    return _run(D_arr)


def invert_t18_from_positive(P, d_lo, d_hi, gamma: float = 0.6, *, has_cam_matrix: bool = True,
                             eps_rgb: tuple[float, float, float] | None = None,
                             version: str = "t18") -> np.ndarray:
    """Reconstruct density from legacy inverted positive ``P`` and re-invert with T18/T23 decoupled slope ratios."""
    P_arr = np.asarray(P, np.float32)
    if not _all_finite(P_arr):
        raise ValueError("positive input contains NaN or infinite values")
    lo, hi, rho, dg, d_anchor, eps_arr, R, flr = _t18_affine_params(
        d_lo, d_hi, gamma, has_cam_matrix=has_cam_matrix, eps_rgb=eps_rgb, version=version,
    )
    # Algebraically fuse log10 -> D_orig -> s -> 10^(R*(s-1)) into exp2(alpha * log2(inner) + beta_log2)
    alpha = ((hi - lo) / (rho * dg)).astype(np.float32)
    beta_log2 = ((R * ((hi - d_anchor) / (rho * dg) + eps_arr)) * _INV_LOG10_2).astype(np.float32)
    one_m_f = np.float32(1.0 - flr)
    flr32 = np.float32(flr)
    inv_omf = np.float32(1.0 / (1.0 - flr))
    max_e2 = np.float32(12.0 * _INV_LOG10_2)
    min_in = np.float32(1e-12)

    def _run(chunk: np.ndarray) -> np.ndarray:
        xc = chunk * one_m_f
        xc += flr32
        np.maximum(xc, min_in, out=xc)
        np.log2(xc, out=xc)
        xc *= alpha
        xc += beta_log2
        np.minimum(xc, max_e2, out=xc)
        np.exp2(xc, out=xc)
        xc -= flr32
        xc *= inv_omf
        return xc

    step = CHUNK_ROWS
    if P_arr.ndim == 3 and P_arr.shape[0] > step:
        out = np.empty(P_arr.shape, np.float32)

        def _block(r0: int) -> None:
            out[r0 : r0 + step] = _run(P_arr[r0 : r0 + step])

        map_row_blocks(P_arr.shape[0], _block, step)
        return out
    return _run(P_arr)

invert_t22 = invert_t18
invert_t22_from_positive = invert_t18_from_positive


def invert_t23(D, p: InversionParams, *, has_cam_matrix: bool = True,
               stats: dict | None = None, zero_escape: bool = False,
               eps_rgb: tuple[float, float, float] | None = None) -> np.ndarray:
    return invert_t18(D, p, has_cam_matrix=has_cam_matrix, stats=stats,
                      zero_escape=zero_escape, eps_rgb=eps_rgb, version="t23")


def invert_t23_from_positive(P, d_lo, d_hi, gamma: float = 0.6, *, has_cam_matrix: bool = True,
                             eps_rgb: tuple[float, float, float] | None = None) -> np.ndarray:
    return invert_t18_from_positive(P, d_lo, d_hi, gamma, has_cam_matrix=has_cam_matrix,
                                    eps_rgb=eps_rgb, version="t23")
