"""CFA-aware RAW linearisation and statistics (analysis path). Pure numpy, decoder-agnostic.

Signal definitions
------------------
cfa      raw sensor values in DN, visible area, shape (H, W).
pattern  (n, n) colour index per CFA position, tiling ``cfa`` from (0, 0).
         0 = R, 1 = G, 2 = B (Bayer's second green is folded into 1).
         Bayer: n = 2. X-Trans: n = 6.
black    (n, n) black level in DN per CFA position.
white    saturation level in DN.
linear   (cfa - black) / (white - black): camera-native linear RGB samples, 0 = black level,
         1 = white level. No white balance, no colour matrix. Values below 0 (read noise) are kept.

Channel values from different CFA colours are never averaged together.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CHANNELS = ("R", "G", "B")


@dataclass(frozen=True)
class RawFrame:
    cfa: np.ndarray
    pattern: np.ndarray
    black: np.ndarray
    white: float
    # decoder-reported valid area (top, left, height, width) in cfa coordinates, or None
    crop: tuple[int, int, int, int] | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        n = self.pattern.shape[0]
        if self.pattern.shape != (n, n) or self.black.shape != (n, n):
            raise ValueError("pattern and black must be square arrays of the same size")
        if self.cfa.ndim != 2:
            raise ValueError("cfa must be 2-D")
        if not set(np.unique(self.pattern)) <= {0, 1, 2}:
            raise ValueError("pattern must contain only colour indices 0, 1, 2")
        if np.any(self.black >= self.white):
            raise ValueError("black level must be below white level")

    @property
    def colors(self) -> np.ndarray:
        return tile(self.pattern, self.cfa.shape)


def tile(pattern: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    n = pattern.shape[0]
    reps = (-(-shape[0] // n), -(-shape[1] // n))
    return np.tile(pattern, reps)[: shape[0], : shape[1]]


def linearize(frame: RawFrame) -> np.ndarray:
    black = tile(frame.black, frame.cfa.shape).astype(np.float32)
    return (frame.cfa.astype(np.float32) - black) / (np.float32(frame.white) - black)


def saturation_mask(frame: RawFrame, threshold: float | None = None) -> np.ndarray:
    """True where the raw value reached the saturation threshold (default: white level)."""
    return frame.cfa >= (frame.white if threshold is None else threshold)


def _region_slice(region, shape):
    if region is None:
        return np.s_[:, :]
    check_region(region, shape)
    top, left, h, w = region
    return np.s_[top : top + h, left : left + w]


def check_region(region, shape):
    top, left, h, w = region
    if h <= 0 or w <= 0 or top < 0 or left < 0 or top + h > shape[0] or left + w > shape[1]:
        raise ValueError(f"region {region} outside image of shape {shape}")


def channel_stats(linear, colors, sat, region=None) -> dict:
    """Per-channel statistics over a region (top, left, height, width), CFA-aware.

    Saturated samples are counted in ``sat_frac`` and excluded from the other statistics.
    """
    sl = _region_slice(region, linear.shape)
    return sample_stats(linear[sl], colors[sl], sat[sl])


def sample_stats(lin, col, s) -> dict:
    """Per-channel statistics of arbitrary samples given their values, colours and saturation flags."""
    out = {}
    for c, name in enumerate(CHANNELS):
        m = col == c
        n_all = int(m.sum())
        v = lin[m & ~s]
        if v.size == 0:
            out[name] = dict(n=0, n_total=n_all, sat_frac=1.0 if n_all else 0.0)
            continue
        mean = float(v.mean())
        std = float(v.std())
        out[name] = dict(
            n=int(v.size), n_total=n_all, sat_frac=float(1 - v.size / n_all),
            mean=mean, median=float(np.median(v)), std=std,
            cv=float(std / mean) if mean > 0 else float("inf"),
            p01=float(np.percentile(v, 1)), p99=float(np.percentile(v, 99)),
        )
    return out


def superpixel(linear, colors, sat, pattern_size: int, block: int):
    """Downsample to (H//block, W//block, 3) by averaging each CFA colour within each block.

    ``block`` must be a multiple of the CFA pattern size so every block holds the same
    number of samples of each colour. Returns (rgb, sat_any) where sat_any marks blocks
    containing at least one saturated sample (their values include the clipped samples).
    """
    if block % pattern_size:
        raise ValueError(f"block {block} must be a multiple of pattern size {pattern_size}")
    h, w = (linear.shape[0] // block) * block, (linear.shape[1] // block) * block
    if h == 0 or w == 0:
        raise ValueError("image smaller than one block")
    hb, wb = h // block, w // block
    lin = linear[:h, :w].reshape(hb, block, wb, block)
    col = colors[:h, :w].reshape(hb, block, wb, block)
    rgb = np.empty((hb, wb, 3), np.float32)
    for c in range(3):
        m = col == c
        cnt = m.sum(axis=(1, 3))
        if np.any(cnt == 0):
            raise ValueError(f"channel {CHANNELS[c]} missing from some blocks")
        rgb[..., c] = (lin * m).sum(axis=(1, 3)) / cnt
    sat_any = sat[:h, :w].reshape(hb, block, wb, block).any(axis=(1, 3))
    return rgb, sat_any
