"""Parametric colour correction after inversion (Sprint 2). Pure numpy.

Fixed order, definitions and rationale in docs/SPRINT2_PARAMS.md:

  P (scene-linear positive) -> 1 exposure -> 2 white balance -> 3 colour matrix -> 4 saturation
  -> clip < 0 -> 5a shoulder -> 5b midtone -> 5c contrast (sRGB-encoded) -> output

Steps 1-4 work on scene-linear uncalibrated camera RGB read with sRGB primaries (Rec.709 luma
weights). With all defaults the output equals the Sprint 1 display encoding, srgb(clip(P, 0, 1)).
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields

import numpy as np

from .invert import srgb_oetf

LUMA = np.array([0.2126, 0.7152, 0.0722])
MID_GREY = 0.18
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
PER_IMAGE = ("ev", "temp", "tint", "sat", "contrast", "midtone", "shoulder")
BOUNDS = dict(ev=(-3.0, 3.0), temp=(-1.0, 1.0), tint=(-1.0, 1.0), sat=(0.0, 2.0),
              contrast=(0.5, 2.0), midtone=(0.05, 0.5), shoulder=(0.0, 1.0))


@contextmanager
def experimental_bounds(**overrides):
    """Temporarily change parameter bounds, for experiments only (results must record the change).
    The official bounds and defaults are untouched outside the ``with`` block."""
    unknown = set(overrides) - set(BOUNDS)
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")
    saved = dict(BOUNDS)
    BOUNDS.update({k: tuple(map(float, v)) for k, v in overrides.items()})
    try:
        yield
    finally:
        BOUNDS.clear()
        BOUNDS.update(saved)


MATRIX_DIAG = (0.5, 2.0)
MATRIX_OFF_DIAG = (-0.8, 0.8)
MATRIX_MAX_COND = 10.0


@dataclass
class RenderParams:
    ev: float = 0.0           # exposure, stops
    temp: float = 0.0         # relative warm (+) / cool (-), log2 gain units; not Kelvin
    tint: float = 0.0         # relative magenta (+) / green (-), log2 gain units
    sat: float = 1.0          # saturation about Rec.709 luma
    contrast: float = 1.0     # slope at the midtone pivot in the sRGB-encoded domain
    midtone: float = MID_GREY  # display-linear value given to scene 18 % grey (after exposure)
    shoulder: float = 0.0     # highlight compression: inputs up to 1 + 7*shoulder roll off to white
    matrix: tuple = IDENTITY  # calibration of scanner + light + film; rows sum to 1

    def __post_init__(self):
        for name in PER_IMAGE:
            v = float(getattr(self, name))
            lo, hi = BOUNDS[name]
            if not (np.isfinite(v) and lo <= v <= hi):
                raise ValueError(f"{name}={v} outside [{lo}, {hi}]")
            setattr(self, name, v)
        m = np.asarray(self.matrix, np.float64)
        if m.shape != (3, 3) or not np.all(np.isfinite(m)):
            raise ValueError("colour matrix must be a finite 3x3 array")
        if np.any(np.abs(m.sum(1) - 1) > 1e-6):
            raise ValueError(f"colour matrix rows must sum to 1 (neutral preserving), got {m.sum(1)}")
        diag, off = np.diag(m), m[~np.eye(3, dtype=bool)]
        if np.any((diag < MATRIX_DIAG[0]) | (diag > MATRIX_DIAG[1])):
            raise ValueError(f"colour matrix diagonal outside {MATRIX_DIAG}: {diag}")
        if np.any((off < MATRIX_OFF_DIAG[0]) | (off > MATRIX_OFF_DIAG[1])):
            raise ValueError(f"colour matrix off-diagonal outside {MATRIX_OFF_DIAG}")
        cond = float(np.linalg.cond(m))
        if cond > MATRIX_MAX_COND:
            raise ValueError(f"colour matrix condition number {cond:.1f} exceeds {MATRIX_MAX_COND}")
        self.matrix = tuple(tuple(float(x) for x in row) for row in m)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["matrix"] = [list(r) for r in self.matrix]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RenderParams:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown render parameters: {sorted(unknown)}")
        return cls(**d)


def wb_gains(temp: float, tint: float) -> np.ndarray:
    """Per-channel linear gains, normalised so a neutral's Rec.709 luma is unchanged."""
    log2g = np.array([temp / 2 + tint / 3, -2 * tint / 3, -temp / 2 + tint / 3])
    g = 2.0 ** log2g
    return g / (LUMA @ g)


def shoulder_curve(x, shoulder: float):
    """Extended Reinhard with white Lw = 1 + 7*shoulder (identity when shoulder = 0); not clipped."""
    if shoulder == 0:
        return x
    lw2 = np.float32((1 + 7 * shoulder) ** 2)
    return x * (1 + x / lw2) / (1 + x)


def srgb_eotf(e):
    e = np.clip(e, 0, 1)
    return np.where(e <= 0.04045, e / 12.92, np.power((e + 0.055) / 1.055, 2.4))


def render(P, p: RenderParams, mask=None):
    """Return (sRGB-encoded float32 image in [0, 1], diagnostics). Diagnostics are computed over
    ``mask`` (image-region pixels) when given; borders, rebate and holder would otherwise dominate."""
    x = np.asarray(P, np.float32)
    if not np.all(np.isfinite(x)):
        raise ValueError("input contains NaN or infinite values")
    if p.ev:
        x = x * np.float32(2.0 ** p.ev)
    if p.temp or p.tint:
        x = x * wb_gains(p.temp, p.tint).astype(np.float32)
    if p.matrix != IDENTITY:
        x = x @ np.asarray(p.matrix, np.float32).T
    if p.sat != 1:
        y = (x @ LUMA.astype(np.float32))[..., None]
        x = y + np.float32(p.sat) * (x - y)
    below_zero = float(np.mean((x < 0)[mask] if mask is not None else x < 0))
    x = np.maximum(x, 0)
    v = shoulder_curve(x, p.shoulder)
    above_white = float(np.mean((v > 1)[mask] if mask is not None else v > 1))
    v = np.minimum(v, 1)
    if p.shoulder or p.midtone != MID_GREY:
        g = np.log(p.midtone) / np.log(float(shoulder_curve(np.float32(MID_GREY), p.shoulder)))
        v = np.power(v, np.float32(g))
    e = srgb_oetf(v).astype(np.float32)
    if p.contrast != 1:
        pv, c = np.float32(srgb_oetf(p.midtone)), np.float32(p.contrast)
        e = np.where(e <= pv, pv * np.power(e / pv, c),
                     1 - (1 - pv) * np.power(np.maximum(1 - e, 0) / (1 - pv), c)).astype(np.float32)
    diagnostics = dict(below_zero_after_colour=below_zero, above_white=above_white)
    return e, diagnostics
