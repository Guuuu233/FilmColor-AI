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

from dataclasses import asdict, dataclass, field

import numpy as np

from .raw import CHANNELS, check_region

EPS = 1e-6  # floor for transmittance before log10 (render path clips below-black to 0)
CHUNK_ROWS = 256  # rows per block when inverting a full image; density and inversion are per pixel


@dataclass
class InversionParams:
    base: tuple[float, float, float]
    d_lo: tuple[float, float, float] = (0.0, 0.0, 0.0)   # black point density per channel
    d_hi: tuple[float, float, float] = (1.0, 1.0, 1.0)   # white point density per channel
    gamma: float = 0.6                                    # film gamma used to set the log range R
    exposure: float = 1.0                                 # linear gain applied before display encoding

    def __post_init__(self):
        # plain floats so the parameters serialise to the sidecar whatever the caller passed
        for name in ("base", "d_lo", "d_hi"):
            setattr(self, name, tuple(float(x) for x in getattr(self, name)))
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
        return asdict(self) | {"log_range": self.log_range}


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
    med = np.median(px, 0)
    p05, p95 = np.percentile(px, [5, 95], 0)
    return dict(rgb=tuple(float(x) for x in med), n=int(len(px)),
                rel_spread=[float(x) for x in (p95 - p05) / med])


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
    t = rgb / np.asarray(base, np.float32)
    return -np.log10(np.maximum(t, EPS))


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


def block_white_point(rgb, base, mask, block: int, pct: float) -> tuple[tuple[float, float, float], dict]:
    """White point on ``block`` x ``block`` means of the linear signal (T4b): per-channel ``pct`` percentile of the
    density of block means, over blocks whose pixels are all in ``mask``. Averaging in the linear domain removes the
    read / shot noise that pushes single-pixel high-density quantiles up where the negative is densest and the
    signal is a few ADU above black (blue channel behind the orange mask). Falls back to single pixels when fewer
    than ``MIN_WHITE_BLOCKS`` blocks are valid (small images)."""
    if block < 1:
        raise ValueError(f"white point block must be >= 1, got {block}")
    h, w = (rgb.shape[0] // block) * block, (rgb.shape[1] // block) * block
    sums = np.zeros((h // block, w // block, 3), np.float64)
    full = np.zeros((h // block, w // block), np.int64)
    step = max(block, CHUNK_ROWS // block * block)  # whole blocks per row chunk keep the float64 copy small
    for r0 in range(0, h, step):
        r1, b0, b1 = min(h, r0 + step), r0 // block, min(h, r0 + step) // block
        sums[b0:b1] = rgb[r0:r1, :w].reshape(b1 - b0, block, w // block, block, 3).sum(axis=(1, 3), dtype=np.float64)
        full[b0:b1] = mask[r0:r1, :w].reshape(b1 - b0, block, w // block, block).sum(axis=(1, 3))
    ok = full == block * block
    n = int(ok.sum())
    if n < MIN_WHITE_BLOCKS:
        return estimate_white_point(density(rgb[mask], base), None, pct), dict(method="pixel_fallback", block=block,
                                                                              pct=pct, n_blocks=n)
    means = (sums[ok] / (block * block)).astype(np.float32)
    return estimate_white_point(density(means, base), None, pct), dict(method="block_mean", block=block, pct=pct,
                                                                      n_blocks=n)


def neutral_ratio_white_point(d_hi, ratio, *, anchor: float = 0.7, scale: float = 0.94,
                              margin: tuple[float, float] = (0.0, 0.03)) -> tuple[tuple[float, float, float], dict]:
    """T4c: keep a blue sky from being taken as white. ``ratio`` = (D_R / D_G, D_B / D_G) of neutral greys of the roll
    (app: median over the roll of each frame's 0.25 <= D_G <= 0.75 density ratios). When the frame's white point has
    the signature of blue sky - R / G below ``ratio[0] - margin[0]`` and B / G above ``ratio[1] + margin[1]`` - its
    channel ratios are moved ``anchor`` of the way to ``ratio`` and the level set from the frame's own white point:
    d_hi_G = ``scale`` * max_c(d_hi_c / r_c). Otherwise ``d_hi`` is returned unchanged."""
    d = np.asarray(d_hi, np.float64)
    own = np.array([d[0] / d[1], 1.0, d[2] / d[1]])
    roll = np.array([float(ratio[0]), 1.0, float(ratio[1])])
    info = dict(ratio_roll=[float(roll[0]), float(roll[2])], ratio_own=[float(own[0]), float(own[2])], anchor=anchor,
                scale=scale, margin=list(margin), applied=False)
    if not (own[0] < roll[0] - margin[0] and own[2] > roll[2] + margin[1]):
        return tuple(float(x) for x in d), info
    r_eff = (1 - anchor) * own + anchor * roll
    g = scale * float(np.max(d / r_eff))
    out = tuple(float(x) for x in r_eff * g)
    return out, dict(info, applied=True, ratio_used=[float(r_eff[0]), float(r_eff[2])], d_hi_own=[float(x) for x in d])


def blue_white_guard(d_hi, guard) -> tuple[tuple[float, float, float], dict]:
    """T4k: a frame whose densest areas are warm (sunlit ridge, no white object) has a blue white point too close to
    green, so every blue midtone is stretched (DSC_3827: d_hi_B / d_hi_G 1.02, the roll's frames about 1.35).
    ``guard`` = (min_ratio, target_ratio, weight): when d_hi_B / d_hi_G < min_ratio, d_hi_B moves ``weight`` of the
    way to max(d_hi_B, target_ratio x d_hi_G). Colour film only (a B&W positive is neutral by construction)."""
    lo, target, weight = (float(x) for x in guard)
    d = [float(x) for x in d_hi]
    ratio = d[2] / d[1]
    info = dict(guard=[lo, target, weight], ratio_bg=ratio, applied=False)
    if ratio >= lo:
        return tuple(d), info
    d_b = d[2] + weight * (max(d[2], target * d[1]) - d[2])
    return (d[0], d[1], d_b), dict(info, applied=True, d_hi_b_own=d[2], ratio_bg_used=d_b / d[1])


def invert(D, p: InversionParams) -> np.ndarray:
    lo, hi = np.asarray(p.d_lo, np.float32), np.asarray(p.d_hi, np.float32)
    s = (D - lo) / (hi - lo)
    R = np.float32(p.log_range)
    floor = np.float32(10.0 ** -p.log_range)
    return (np.power(np.float32(10.0), R * (s - 1)) - floor) / (1 - floor)


def srgb_oetf(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def tone_map(P, p: InversionParams) -> np.ndarray:
    return srgb_oetf(P * np.float32(p.exposure)).astype(np.float32)


def range_report(D, P, p: InversionParams, mask=None) -> dict:
    """Effective dynamic range diagnostics per channel over ``mask``."""
    Dm = D[mask] if mask is not None else D.reshape(-1, 3)
    Pm = P[mask] if mask is not None else P.reshape(-1, 3)
    out = {}
    for c, name in enumerate(CHANNELS):
        out[name] = dict(
            clip_high=float(np.mean(Pm[:, c] * p.exposure > 1)),   # brighter than output white
            below_black=float(np.mean(Pm[:, c] < 0)),              # less dense than the black point
            brighter_than_base=float(np.mean(Dm[:, c] < 0)),
            density_p01=float(np.percentile(Dm[:, c], 0.1)),
            density_p999=float(np.percentile(Dm[:, c], 99.9)),
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
                    white_ratio=None, white_b_guard=None) -> InversionResult:
    """Full chain with automatic white point unless ``d_hi`` is given. ``mask`` selects image pixels
    used for the white point and the report (exclude borders, rebate, holder and saturated areas).
    ``white_block``: estimate the automatic white point on block means of that size (``block_white_point``)
    instead of single pixels (None, the default: single pixels, unchanged). ``white_ratio``: roll neutral density
    ratios (D_R / D_G, D_B / D_G) checked against the automatic white point (``neutral_ratio_white_point``).
    ``white_b_guard``: (min_ratio, target_ratio, weight) for ``blue_white_guard`` on the automatic white point."""
    if not np.all(np.isfinite(rgb)):
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
        if white_b_guard is not None:
            d_hi, guard_info = blue_white_guard(d_hi, white_b_guard)
            white_info = dict(white_info or dict(method="pixel", pct=white_pct), blue_guard=guard_info)
        lo = d_lo or (0.0, 0.0, 0.0)
        flat = [c for c, hi, l in zip("RGB", d_hi, lo) if hi <= l]
        if flat:
            raise ValueError(f"no density above the black point in channel(s) {''.join(flat)} of the image "
                             "region: blank or unexposed frame, or wrong image region / film base")
    p = InversionParams(base=tuple(float(x) for x in base), d_lo=tuple(d_lo or (0.0, 0.0, 0.0)),
                        d_hi=tuple(float(x) for x in d_hi), gamma=gamma, exposure=exposure)
    # share of the whole image denser than the white point: large values flag opaque areas
    # (holder mask) or highlights outside the region used for the white point
    hi = np.asarray(p.d_hi, np.float32)
    if rgb.ndim == 3:
        P = np.empty(rgb.shape, np.float32)
        n_above = np.zeros(3, np.int64)
        for r0 in range(0, rgb.shape[0], CHUNK_ROWS):
            D = density(rgb[r0 : r0 + CHUNK_ROWS], base)
            P[r0 : r0 + CHUNK_ROWS] = invert(D, p)
            n_above += np.count_nonzero(D > hi, axis=(0, 1))
        above_white = [float(k / (rgb.shape[0] * rgb.shape[1])) for k in n_above]
    else:
        D = density(rgb, base)
        P = invert(D, p)
        above_white = [float(np.mean(D[..., c] > p.d_hi[c])) for c in range(3)]
    report = dict(white_pct=white_pct, range=range_report(Dm, invert(Dm, p), p), full_image_above_white=above_white)
    if white_info is not None:  # reports of the default single-pixel white point keep their previous field set
        report["white_point"] = white_info
    return InversionResult(positive=P, params=p, report=report)
