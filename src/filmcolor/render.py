"""Parametric colour correction after inversion (Sprint 2 + T4 darkroom extensions). Pure numpy.

Fixed order, definitions and rationale in docs/SPRINT2_PARAMS.md:

  P (scene-linear positive) -> 1 exposure -> 2 white balance -> 3 colour matrix -> 4 saturation & vibrance
  -> clip < 0 -> 4b channel contrast slopes (slope_r, slope_b) -> 5a shoulder & whites -> 5b midtone
  (per channel, or blended with the same curve applied to luminance: chroma_preserve, T4c)
  -> 5c contrast (sRGB-encoded) -> 5d zonal highlights / shadows / blacks -> 5e tone curve -> output

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
EXTENDED_SCALARS = ("highlights", "shadows", "whites", "blacks", "vibrance", "slope_r", "slope_b", "chroma_preserve")
BOUNDS = dict(
    ev=(-3.0, 3.0), temp=(-1.0, 1.0), tint=(-1.0, 1.0), sat=(0.0, 2.0),
    contrast=(0.5, 2.0), midtone=(0.05, 0.5), shoulder=(0.0, 1.0),
)
EXTENDED_BOUNDS = dict(
    highlights=(-1.0, 1.0), shadows=(-1.0, 1.0), whites=(-1.0, 1.0), blacks=(-1.0, 1.0),
    vibrance=(-1.0, 1.0), slope_r=(-0.5, 0.5), slope_b=(-0.5, 0.5), chroma_preserve=(0.0, 1.0),
)
# sRGB (D65) -> XYZ, the matrix dcraw / LibRaw use to turn a camera's XYZ -> camera matrix into camera -> sRGB
XYZ_FROM_SRGB_D65 = np.array([[0.412453, 0.357580, 0.180423],
                              [0.212671, 0.715160, 0.072169],
                              [0.019334, 0.119193, 0.950227]])
# chroma_preserve weight falls to 0 below / above these display-linear luminances (smoothstep edges)
CHROMA_PRESERVE_DARK = (0.02, 0.06)
CHROMA_PRESERVE_BRIGHT = (0.88, 0.94)
CURVE_CHANNELS = ("rgb", "r", "g", "b")
CURVE_PRESETS = {
    "linear": {"rgb": [[0.0, 0.0], [1.0, 1.0]]},
    "soft": {"rgb": [[0.0, 0.03], [0.25, 0.27], [0.5, 0.51], [0.75, 0.74], [1.0, 0.98]]},
    "standard": {"rgb": [[0.0, 0.0], [0.25, 0.22], [0.5, 0.50], [0.75, 0.78], [1.0, 1.0]]},
    "matte": {"rgb": [[0.0, 0.06], [0.22, 0.24], [0.5, 0.50], [0.78, 0.76], [1.0, 0.96]]},
    "classic": {"rgb": [[0.0, 0.0], [0.22, 0.17], [0.5, 0.50], [0.78, 0.83], [1.0, 1.0]]},
}


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


MATRIX_DIAG = (0.5, 2.4)
MATRIX_OFF_DIAG = (-1.6, 0.8)
MATRIX_MAX_COND = 10.0


def _validate_curve(tc) -> tuple:
    """Normalise ``tone_curve`` (None, list of [x, y] pairs for composite RGB, or dict of channel -> points)
    into a canonical tuple of ``(channel, ((x0, y0), ...))`` sorted by channel; empty tuple = identity."""
    if not tc:
        return ()
    if isinstance(tc, (list, tuple)) and tc and isinstance(tc[0], (list, tuple)) and len(tc[0]) == 2 \
            and isinstance(tc[0][0], (int, float)):
        tc = {"rgb": tc}
    elif isinstance(tc, (list, tuple)) and all(isinstance(it, (list, tuple)) and len(it) == 2
                                               and isinstance(it[0], str) for it in tc):
        tc = dict(tc)
    if not isinstance(tc, dict):
        raise ValueError("tone_curve must be a dict of channel -> [[x, y], ...] or a list of [x, y] points")
    out = []
    for ch, pts in sorted(tc.items()):
        ch_l = str(ch).lower()
        if ch_l not in CURVE_CHANNELS:
            raise ValueError(f"unknown tone_curve channel {ch!r}; must be one of {CURVE_CHANNELS}")
        if not pts:
            continue
        arr = np.asarray(pts, np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2 or not np.all(np.isfinite(arr)):
            raise ValueError(f"tone_curve[{ch!r}] must be at least two finite [x, y] points")
        if np.any((arr < 0.0) | (arr > 1.0)):
            raise ValueError(f"tone_curve[{ch!r}] coordinates must be in [0, 1]")
        order = np.argsort(arr[:, 0], kind="mergesort")
        arr = arr[order]
        if np.any(np.diff(arr[:, 0]) <= 1e-6):
            raise ValueError(f"tone_curve[{ch!r}] x coordinates must be strictly increasing")
        if len(arr) == 2 and np.allclose(arr, [[0.0, 0.0], [1.0, 1.0]], atol=1e-7):
            continue
        out.append((ch_l, tuple((float(x), float(y)) for x, y in arr)))
    return tuple(out)


def eval_monotone_spline(pts, x: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson monotone cubic Hermite interpolation of ``pts`` ((N, 2) in [0, 1]) on ``x``."""
    arr = np.asarray(pts, np.float64)
    px, py = arr[:, 0], arr[:, 1]
    n = len(px)
    if n == 2:
        t = np.clip((x - px[0]) / max(px[1] - px[0], 1e-6), 0.0, 1.0)
        return np.clip(py[0] + t * (py[1] - py[0]), 0.0, 1.0).astype(np.float32)
    dx = np.diff(px)
    dy = np.diff(py)
    delta = dy / dx
    m = np.zeros(n, np.float64)
    m[0], m[-1] = delta[0], delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            m[i] = 0.0
        else:
            w1 = 2 * dx[i] + dx[i - 1]
            w2 = dx[i] + 2 * dx[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    for i in range(n - 1):
        if abs(delta[i]) < 1e-12:
            m[i] = m[i + 1] = 0.0
        else:
            a, b = m[i] / delta[i], m[i + 1] / delta[i]
            h = np.hypot(a, b)
            if h > 3.0:
                s = 3.0 / h
                m[i], m[i + 1] = s * a * delta[i], s * b * delta[i]
    # Evaluate via a 4096-entry LUT for fast per-pixel application
    grid = np.linspace(0.0, 1.0, 4096, dtype=np.float64)
    idx = np.clip(np.searchsorted(px, grid, side="right") - 1, 0, n - 2)
    h = dx[idx]
    t = np.clip((grid - px[idx]) / h, 0.0, 1.0)
    t2, t3 = t * t, t * t * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    lut = np.clip(h00 * py[idx] + h10 * h * m[idx] + h01 * py[idx + 1] + h11 * h * m[idx + 1], 0.0, 1.0)
    lut[grid <= px[0]] = py[0]
    lut[grid >= px[-1]] = py[-1]
    pos = np.clip(x, 0.0, 1.0) * 4095.0
    i0 = np.minimum(pos.astype(np.int32), 4094)
    f = (pos - i0).astype(np.float32)
    lut32 = lut.astype(np.float32)
    return lut32[i0] + f * (lut32[i0 + 1] - lut32[i0])


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
    highlights: float = 0.0   # Lightroom-style highlight recovery (-) / boost (+) in [-1, 1]
    shadows: float = 0.0      # shadow lift (+) / deepen (-) in [-1, 1]
    whites: float = 0.0       # white point / upper-tone pull in [-1, 1]
    blacks: float = 0.0       # black point / paper-black lift (+) or crush (-) in [-1, 1]
    vibrance: float = 0.0     # saturation boost weighted towards low-chroma pixels in [-1, 1]
    slope_r: float = 0.0      # red-vs-green contrast gamma ratio offset in [-0.5, 0.5] (RawTherapee style)
    slope_b: float = 0.0      # blue-vs-green contrast gamma ratio offset in [-0.5, 0.5]
    chroma_preserve: float = 0.0  # 0 = shoulder & midtone per channel; 1 = on luminance, keeping R:G:B (T4c)
    tone_curve: tuple = ()    # optional monotone spline points per channel

    def __post_init__(self):
        for name in PER_IMAGE:
            v = float(getattr(self, name))
            lo, hi = BOUNDS[name]
            if not (np.isfinite(v) and lo <= v <= hi):
                raise ValueError(f"{name}={v} outside [{lo}, {hi}]")
            setattr(self, name, v)
        for name in EXTENDED_SCALARS:
            v = float(getattr(self, name))
            lo, hi = EXTENDED_BOUNDS[name]
            if not (np.isfinite(v) and lo <= v <= hi):
                raise ValueError(f"{name}={v} outside [{lo}, {hi}]")
            setattr(self, name, v)
        self.tone_curve = _validate_curve(self.tone_curve)
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

    def has_extended(self) -> bool:
        return bool(self.highlights or self.shadows or self.whites or self.blacks
                    or self.vibrance or self.slope_r or self.slope_b or self.chroma_preserve or self.tone_curve)

    def to_dict(self, *, full: bool = False) -> dict:
        d = {k: getattr(self, k) for k in PER_IMAGE}
        d["matrix"] = [list(r) for r in self.matrix]
        if full or self.has_extended():
            for k in EXTENDED_SCALARS:
                if full or getattr(self, k) != 0.0:
                    d[k] = getattr(self, k)
            if full or self.tone_curve:
                d["tone_curve"] = {ch: [list(pt) for pt in pts] for ch, pts in self.tone_curve}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RenderParams:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown render parameters: {sorted(unknown)}")
        return cls(**d)


def camera_color_matrix(rgb_xyz_matrix, mix: float = 1.0) -> tuple:
    """Camera RGB -> sRGB matrix from a RAW file's XYZ -> camera matrix (LibRaw ``rgb_xyz_matrix``, first three rows),
    as dcraw builds it: cam_rgb = cam_xyz @ XYZ_FROM_SRGB_D65 with rows normalised to sum 1, inverted (so rows sum
    to 1: neutrals stay neutral). ``mix`` blends it with the identity: mix * M + (1 - mix) * I (T4c; the app's
    default is 0.35). Applied to the inverted positive this is an approximation: the matrix describes the scanning
    camera's crosstalk for light reaching it, the positive is a per-channel inversion of that light. Raises
    ValueError when the result is not a valid ``RenderParams.matrix`` (bounds, condition number)."""
    if not 0.0 <= float(mix) <= 1.0:
        raise ValueError(f"camera matrix mix must be in [0, 1], got {mix}")
    cam_xyz = np.asarray(rgb_xyz_matrix, np.float64)[:3, :3]
    if not np.all(np.isfinite(cam_xyz)) or not np.any(cam_xyz):
        raise ValueError("no camera colour matrix in this file")
    cam_rgb = cam_xyz @ XYZ_FROM_SRGB_D65
    cam_rgb = cam_rgb / cam_rgb.sum(1, keepdims=True)
    m = float(mix) * np.linalg.inv(cam_rgb) + (1.0 - float(mix)) * np.eye(3)
    m = m - (m.sum(1, keepdims=True) - 1.0) / 3.0  # exact row sums of 1 after floating point
    return RenderParams(matrix=tuple(map(tuple, m))).matrix


def _smoothstep(x, lo: float, hi: float):
    t = np.clip((x - np.float32(lo)) / np.float32(hi - lo), 0.0, 1.0)
    return t * t * (np.float32(3.0) - np.float32(2.0) * t)


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


CHUNK_ROWS = 256  # rows per block in ``render``; every step is per pixel, so blocks do not interact


def render(P, p: RenderParams, mask=None, *, chunk_rows: int = CHUNK_ROWS):
    """Return (sRGB-encoded float32 image in [0, 1], diagnostics). Diagnostics are computed over
    ``mask`` (image-region pixels) when given; borders, rebate and holder would otherwise dominate.

    Images taller than ``chunk_rows`` are rendered in row blocks into one output array (same
    pixels as ``render_full``; peak memory is the output plus one block of temporaries)."""
    x = np.asarray(P, np.float32)
    if x.ndim != 3 or x.shape[0] <= chunk_rows:
        e, diagnostics, _ = render_full(x, p, mask)
        return e, diagnostics
    out = np.empty(x.shape, np.float32)
    n_neg = n_over = n = 0
    for r0 in range(0, x.shape[0], chunk_rows):
        xc = x[r0 : r0 + chunk_rows]
        if not np.all(np.isfinite(xc)):
            raise ValueError("input contains NaN or infinite values")
        e, neg, over, _ = _render_core(xc, p)
        out[r0 : r0 + chunk_rows] = e
        if mask is not None:
            m = mask[r0 : r0 + chunk_rows]
            neg, over = neg[m], over[m]
        n_neg += int(np.count_nonzero(neg))
        n_over += int(np.count_nonzero(over))
        n += neg.size
    frac = (lambda k: k / n) if n else (lambda k: float("nan"))
    return out, dict(below_zero_after_colour=frac(n_neg), above_white=frac(n_over))


def overexposure_stops(v):
    """Per-channel stops above output white of display-linear values before the white clip (0 below white)."""
    return np.log2(np.maximum(v, 1.0))


def render_full(P, p: RenderParams, mask=None):
    """``render`` plus the display-linear values after the shoulder and before the white clip (for
    measuring how far highlights are pushed past white; values > 1 are clipped in the output)."""
    x = np.asarray(P, np.float32)
    if not np.all(np.isfinite(x)):
        raise ValueError("input contains NaN or infinite values")
    e, neg, over, pre_clip = _render_core(x, p)
    below_zero = float(np.mean(neg[mask] if mask is not None else neg))
    above_white = float(np.mean(over[mask] if mask is not None else over))
    diagnostics = dict(below_zero_after_colour=below_zero, above_white=above_white)
    return e, diagnostics, pre_clip


def _apply_zonal_tones(e: np.ndarray, p: RenderParams) -> np.ndarray:
    """Lightroom-style zonal adjustments in sRGB-encoded space [0, 1]: blacks, whites, shadows, highlights."""
    u = e
    if p.blacks:
        b = np.float32(p.blacks)
        if b > 0:
            # Paper-black lift + soft toe (negadoctor style: lifts deep shadows without shifting white)
            lift = np.float32(0.14 * b)
            u = lift + (np.float32(1.0) - lift) * u
        else:
            # Smoothly deepen shadows near black
            u = u - np.float32(-0.22 * b) * u * (np.float32(1.0) - u) ** 2
    if p.whites:
        w = np.float32(p.whites)
        u = u + np.float32(0.28 * w) * (u ** 2) * (np.float32(1.0) - u)
    if p.shadows:
        sh = np.float32(p.shadows)
        # Basis concentrated in dark-to-lower-mid tones (peaks around u = 0.25)
        u = u + np.float32(0.42 * sh) * u * ((np.float32(1.0) - u) ** 2.2)
    if p.highlights:
        hl = np.float32(p.highlights)
        # Basis concentrated in upper-mid-to-bright tones (peaks around u = 0.75)
        u = u + np.float32(0.42 * hl) * (u ** 2.2) * (np.float32(1.0) - u)
    return np.clip(u, 0.0, 1.0).astype(np.float32)


def _apply_tone_curve(e: np.ndarray, tc: tuple) -> np.ndarray:
    out = e
    ch_map = dict(tc)
    if "rgb" in ch_map:
        out = eval_monotone_spline(ch_map["rgb"], out)
    if any(k in ch_map for k in ("r", "g", "b")):
        out = out.copy()
        for idx, k in enumerate(("r", "g", "b")):
            if k in ch_map:
                out[..., idx] = eval_monotone_spline(ch_map[k], out[..., idx])
    return out


def _render_core(x, p: RenderParams):
    """Per-pixel steps on float32 ``x`` (..., 3): (encoded output, below-zero flags after colour,
    above-white flags after the shoulder, display-linear values before the white clip)."""
    if p.ev:
        x = x * np.float32(2.0 ** p.ev)
    if p.temp or p.tint:
        x = x * wb_gains(p.temp, p.tint).astype(np.float32)
    if p.matrix != IDENTITY:
        x = x @ np.asarray(p.matrix, np.float32).T
    if p.sat != 1 or p.vibrance:
        y = (x @ LUMA.astype(np.float32))[..., None]
        if p.vibrance:
            denom = np.maximum(y[..., 0], np.float32(0.05))
            rel_c = np.clip((x.max(-1) - x.min(-1)) / denom, 0.0, 1.5)
            vib_w = (np.float32(1.0) - np.clip(rel_c / np.float32(0.85), 0.0, 1.0)) ** 1.5
            gain = np.float32(p.sat) * (np.float32(1.0) + np.float32(0.65 * p.vibrance) * vib_w)[..., None]
        else:
            gain = np.float32(p.sat)
        if p.chroma_preserve:  # T4c: a saturation boost fades out in deep shadows (no coloured noise in blacks)
            dark = _smoothstep(y, 0.0, 0.02)
            gain = np.where(gain > 1, 1 + (gain - 1) * dark, gain)
        x = y + np.maximum(gain, np.float32(0.0)) * (x - y)
    neg = x < 0
    x = np.maximum(x, 0)
    if p.slope_r or p.slope_b:
        # RawTherapee Film Negative style per-channel power ratio anchored at MID_GREY
        x = x.copy()
        piv = np.float32(MID_GREY)
        if p.slope_r:
            gr = np.float32(2.0 ** (-0.65 * p.slope_r))
            x[..., 0] = piv * np.power(np.maximum(x[..., 0] / piv, np.float32(1e-6)), gr)
        if p.slope_b:
            gb = np.float32(2.0 ** (-0.65 * p.slope_b))
            x[..., 2] = piv * np.power(np.maximum(x[..., 2] / piv, np.float32(1e-6)), gb)
    eff_shoulder = float(np.clip(p.shoulder + max(0.0, -0.45 * p.highlights) + max(0.0, -0.25 * p.whites), 0.0, 1.0))
    pre_clip = shoulder_curve(x, eff_shoulder)
    over = pre_clip > 1
    v = np.minimum(pre_clip, 1)
    if eff_shoulder or p.midtone != MID_GREY:
        g = np.log(p.midtone) / np.log(float(shoulder_curve(np.float32(MID_GREY), eff_shoulder)))
        v_ch = np.power(v, np.float32(g))
        if p.chroma_preserve:
            # T4c: the same scalar curve on luminance, applied as one gain to R, G, B (keeps their ratios, which the
            # per-channel power compresses by g); blended in by a weight that falls to 0 in deep shadows and near
            # white, where per-channel behaviour avoids coloured noise and channel overflow
            y_in = np.maximum(x @ LUMA.astype(np.float32), np.float32(1e-6))[..., None]
            v_y = np.power(np.minimum(shoulder_curve(y_in, eff_shoulder), 1), np.float32(g))
            v_lum = np.clip(x * (v_y / y_in), 0.0, 1.0)
            w = (np.float32(p.chroma_preserve) * _smoothstep(v_y, *CHROMA_PRESERVE_DARK)
                 * (np.float32(1.0) - _smoothstep(v_y, *CHROMA_PRESERVE_BRIGHT)))
            v = (np.float32(1.0) - w) * v_ch + w * v_lum
        else:
            v = v_ch
    e = srgb_oetf(v).astype(np.float32)
    if p.contrast != 1:
        pv, c = np.float32(srgb_oetf(p.midtone)), np.float32(p.contrast)
        e = np.where(e <= pv, pv * np.power(e / pv, c),
                     1 - (1 - pv) * np.power(np.maximum(1 - e, 0) / (1 - pv), c)).astype(np.float32)
    if p.blacks or p.whites or p.shadows or p.highlights:
        e = _apply_zonal_tones(e, p)
    if p.tone_curve:
        e = _apply_tone_curve(e, p.tone_curve)
    return e, neg, over, pre_clip
