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
                    exposure: float = 1.0, d_lo=None, d_hi=None) -> InversionResult:
    """Full chain with automatic white point unless ``d_hi`` is given. ``mask`` selects image pixels
    used for the white point and the report (exclude borders, rebate, holder and saturated areas)."""
    if not np.all(np.isfinite(rgb)):
        raise ValueError("input image contains NaN or infinite values")
    if not np.all(np.isfinite(base)) or np.any(np.asarray(base) <= 0):
        raise ValueError(f"film base must be positive and finite, got {tuple(base)}")
    # global statistics only need the masked pixels; their density is computed once on that
    # selection (same values, same order as density(rgb)[mask]) and the full image in row blocks
    Dm = density(rgb[mask] if mask is not None else rgb.reshape(-1, 3), base)
    if d_hi is None:
        d_hi = estimate_white_point(Dm, None, white_pct)
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
    return InversionResult(positive=P, params=p,
                           report=dict(white_pct=white_pct, range=range_report(Dm, invert(Dm, p), p),
                                       full_image_above_white=above_white))
