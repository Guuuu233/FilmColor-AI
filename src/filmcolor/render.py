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
from dataclasses import dataclass, fields

import numpy as np

from .invert import _all_finite, map_row_blocks, srgb_oetf

LUMA = np.array([0.2126, 0.7152, 0.0722])
LUMA_F32 = LUMA.astype(np.float32)
MID_GREY = 0.18
IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
PER_IMAGE = ("ev", "temp", "tint", "sat", "contrast", "midtone", "shoulder")
EXTENDED_SCALARS = (
    "highlights", "shadows", "whites", "blacks", "vibrance", "slope_r", "slope_b",
    "chroma_preserve", "bpc", "soft_clip", "skin_protect", "hd_curve", "log_unmix",
)
BOUNDS = dict(
    ev=(-3.0, 3.0), temp=(-1.0, 1.0), tint=(-1.0, 1.0), sat=(0.0, 2.0),
    contrast=(0.5, 2.0), midtone=(0.05, 0.5), shoulder=(0.0, 1.0),
)
EXTENDED_BOUNDS = dict(
    highlights=(-1.0, 1.0), shadows=(-1.0, 1.0), whites=(-1.0, 1.0), blacks=(-1.0, 1.0),
    vibrance=(-1.0, 1.0), slope_r=(-0.5, 0.5), slope_b=(-0.5, 0.5), chroma_preserve=(0.0, 1.0),
    bpc=(0.0, 0.25), soft_clip=(0.0, 0.99), skin_protect=(0.0, 1.0), hd_curve=(0.0, 1.0),
    log_unmix=(0.0, 1.0),
)
# sRGB (D65) -> XYZ, the matrix dcraw / LibRaw use to turn a camera's XYZ -> camera matrix into camera -> sRGB
XYZ_FROM_SRGB_D65 = np.array([[0.412453, 0.357580, 0.180423],
                              [0.212671, 0.715160, 0.072169],
                              [0.019334, 0.119193, 0.950227]])
_SRGB_FROM_XYZ_F32 = np.linalg.inv(XYZ_FROM_SRGB_D65).astype(np.float32)
_EYE3_MASK_OFF = ~np.eye(3, dtype=bool)
_VALID_MATRICES: dict[tuple, tuple] = {IDENTITY: IDENTITY}
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


def _validate_matrix(matrix) -> tuple:
    if matrix is IDENTITY or (isinstance(matrix, tuple) and matrix == IDENTITY):
        return IDENTITY
    if isinstance(matrix, tuple) and len(matrix) == 3 and all(isinstance(r, tuple) and len(r) == 3 for r in matrix):
        cached = _VALID_MATRICES.get(matrix)
        if cached is not None:
            return cached
    m = np.asarray(matrix, np.float64)
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        raise ValueError("colour matrix must be a finite 3x3 array")
    if np.any(np.abs(m.sum(1) - 1) > 1e-6):
        raise ValueError(f"colour matrix rows must sum to 1 (neutral preserving), got {m.sum(1)}")
    diag, off = np.diag(m), m[_EYE3_MASK_OFF]
    if np.any((diag < MATRIX_DIAG[0]) | (diag > MATRIX_DIAG[1])):
        raise ValueError(f"colour matrix diagonal outside {MATRIX_DIAG}: {diag}")
    if np.any((off < MATRIX_OFF_DIAG[0]) | (off > MATRIX_OFF_DIAG[1])):
        raise ValueError(f"colour matrix off-diagonal outside {MATRIX_OFF_DIAG}")
    cond = float(np.linalg.cond(m))
    if cond > MATRIX_MAX_COND:
        raise ValueError(f"colour matrix condition number {cond:.1f} exceeds {MATRIX_MAX_COND}")
    canonical = tuple(tuple(float(x) for x in row) for row in m)
    if len(_VALID_MATRICES) >= 256:
        _VALID_MATRICES.clear()
        _VALID_MATRICES[IDENTITY] = IDENTITY
    _VALID_MATRICES[canonical] = canonical
    return canonical


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
    bpc: float = 0.0          # highlight-anchored linear-light black point compensation in [0, 0.25]
    soft_clip: float = 0.0    # smooth display-linear highlight roll-off threshold in [0, 0.99]
    skin_protect: float = 0.0  # NegPy LAB skin-tone chroma protection strength in [0, 1]
    hd_curve: float = 0.0     # orthogonal log-print-density Softplus H&D curve blend in [0, 1]
    log_unmix: float = 0.0    # log-density C-41 dye crosstalk unmixing strength in [0, 1]
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
        self.matrix = _validate_matrix(self.matrix)

    def has_extended(self) -> bool:
        return bool(self.highlights or self.shadows or self.whites or self.blacks
                    or self.vibrance or self.slope_r or self.slope_b or self.chroma_preserve
                    or self.bpc or self.soft_clip or self.skin_protect or self.hd_curve or self.log_unmix or self.tone_curve)

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
    row_sums = cam_rgb.sum(1, keepdims=True)
    if np.any(np.abs(row_sums) < 1e-12):
        raise ValueError("camera colour matrix is singular")
    cam_rgb = cam_rgb / row_sums
    try:
        inv_cam = np.linalg.inv(cam_rgb)
    except np.linalg.LinAlgError as exc:
        raise ValueError("camera colour matrix is singular") from exc
    m = float(mix) * inv_cam + (1.0 - float(mix)) * np.eye(3)
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
    arr = np.asarray(e)
    if arr.ndim > 0 and arr.dtype == np.float32:
        ec = np.clip(arr, np.float32(0.0), np.float32(1.0))
        out = ec * np.float32(1.0 / 12.92)
        hi = ec > np.float32(0.04045)
        if hi.any():
            out[hi] = np.power((ec[hi] + np.float32(0.055)) * np.float32(1.0 / 1.055), np.float32(2.4))
        return out
    ec = np.clip(arr, 0, 1)
    return np.where(ec <= 0.04045, ec / 12.92, np.power((ec + 0.055) / 1.055, 2.4))


CHUNK_ROWS = 512  # rows per block in ``render``; every step is per pixel, so blocks do not interact



def _render_chunked(x: np.ndarray, core, mask, chunk_rows: int) -> tuple[np.ndarray, dict]:
    """Run per-pixel ``core(block) -> (encoded, below_zero_flags, above_white_flags, ...)`` over row blocks of
    float32 ``x`` into one output array, and return it with mask-restricted diagnostics. Blocks are independent,
    so they run on ``invert.map_row_blocks`` threads without changing a single output value."""
    if x.size == 0 or x.ndim != 3 or x.shape[0] <= chunk_rows:
        if not _all_finite(x):
            raise ValueError("input contains NaN or infinite values")
        if x.size == 0:
            return x.copy(), dict(below_zero_after_colour=float("nan"), above_white=float("nan"))
        e, neg, over = core(x)[:3]
        if mask is not None:
            neg, over = neg[mask], over[mask]
        n = neg.size
        frac = (lambda k: k / n) if n else (lambda k: float("nan"))
        return e, dict(below_zero_after_colour=frac(int(np.count_nonzero(neg))),
                       above_white=frac(int(np.count_nonzero(over))))
    out = np.empty(x.shape, np.float32)

    def _block(r0: int) -> tuple[int, int, int]:
        xc = x[r0 : r0 + chunk_rows]
        if not _all_finite(xc):
            raise ValueError("input contains NaN or infinite values")
        e, neg, over = core(xc)[:3]
        out[r0 : r0 + chunk_rows] = e
        if mask is not None:
            m = mask[r0 : r0 + chunk_rows]
            neg, over = neg[m], over[m]
        return int(np.count_nonzero(neg)), int(np.count_nonzero(over)), neg.size

    counts = map_row_blocks(x.shape[0], _block, chunk_rows)
    n_neg, n_over, n = (sum(c[i] for c in counts) for i in range(3))
    frac = (lambda k: k / n) if n else (lambda k: float("nan"))
    return out, dict(below_zero_after_colour=frac(n_neg), above_white=frac(n_over))


def render(P, p: RenderParams, mask=None, *, chunk_rows: int = CHUNK_ROWS, fast: bool = False):
    """Return (sRGB-encoded float32 image in [0, 1], diagnostics). Diagnostics are computed over
    ``mask`` (image-region pixels) when given; borders, rebate and holder would otherwise dominate.

    Images taller than ``chunk_rows`` are rendered in row blocks into one output array (same
    pixels as ``render_full``; peak memory is the output plus one block of temporaries)."""
    x = np.asarray(P, np.float32)
    pre = _prepare_render_luts(p) if fast else None
    return _render_chunked(x, lambda xc: _render_core(xc, p, _pre=pre), mask, chunk_rows)


def overexposure_stops(v):
    """Per-channel stops above output white of display-linear values before the white clip (0 below white)."""
    return np.log2(np.maximum(v, 1.0))


def render_full(P, p: RenderParams, mask=None):
    """``render`` plus the display-linear values after the shoulder and before the white clip (for
    measuring how far highlights are pushed past white; values > 1 are clipped in the output)."""
    x = np.asarray(P, np.float32)
    if not _all_finite(x):
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
            om = np.float32(1.0) - u
            u = u - np.float32(-0.22 * b) * u * (om * om)
    if p.whites:
        w = np.float32(p.whites)
        u = u + np.float32(0.28 * w) * (u * u) * (np.float32(1.0) - u)
    if p.shadows:
        sh = np.float32(p.shadows)
        # Basis concentrated in dark-to-lower-mid tones (peaks around u = 0.25)
        u = u + np.float32(0.42 * sh) * u * np.power(np.maximum(np.float32(1.0) - u, np.float32(0.0)), np.float32(2.2))
    if p.highlights:
        hl = np.float32(p.highlights)
        # Basis concentrated in upper-mid-to-bright tones (peaks around u = 0.75)
        u = u + np.float32(0.42 * hl) * np.power(np.maximum(u, np.float32(0.0)), np.float32(2.2)) * (np.float32(1.0) - u)
    return np.clip(u, np.float32(0.0), np.float32(1.0))


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


def _apply_skin_protect(e: np.ndarray, strength: float) -> np.ndarray:
    """NegPy-style CIELAB skin-tone chroma protection (52 deg +/- 20 deg hue locus)."""
    if strength <= 0.0:
        return e
    lin = srgb_eotf(e).astype(np.float32)
    xyz = lin @ XYZ_FROM_SRGB_D65.T.astype(np.float32)
    # D65 white point: (0.95047, 1.0, 1.08883)
    wp = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    t = xyz / wp
    f = np.where(t > np.float32(0.008856), np.cbrt(np.maximum(t, 0.0)), np.float32(7.787) * t + np.float32(16.0 / 116.0))
    L = np.float32(116.0) * f[..., 1] - np.float32(16.0)
    a = np.float32(500.0) * (f[..., 0] - f[..., 1])
    b = np.float32(200.0) * (f[..., 1] - f[..., 2])
    C = np.hypot(a, b)
    h_deg = np.degrees(np.arctan2(b, a))
    dh = ((h_deg - np.float32(52.0) + np.float32(180.0)) % np.float32(360.0)) - np.float32(180.0)
    w_hue = np.exp(np.float32(-0.5) * (dh / np.float32(20.0)) ** 2)
    w_lum = _smoothstep(L, 22.0, 35.0) * (np.float32(1.0) - _smoothstep(L, 82.0, 94.0))
    c_ceil = np.float32(28.0)
    excess = np.maximum(C - c_ceil, np.float32(0.0))
    w_c = _smoothstep(C, 18.0, 30.0)
    w = np.float32(strength) * w_hue * w_lum * w_c
    # Soft compression of chroma above c_ceil: keeps hue angle & L* exact
    c_new = C - w * excess * np.float32(0.45)
    scale = np.where(C > np.float32(1e-4), c_new / np.maximum(C, np.float32(1e-4)), np.float32(1.0))
    a2 = a * scale
    b2 = b * scale
    fy = (L + np.float32(16.0)) / np.float32(116.0)
    fx = a2 / np.float32(500.0) + fy
    fz = fy - b2 / np.float32(200.0)
    f3 = np.stack([fx, fy, fz], axis=-1)
    t2 = np.where(f3 > np.float32(0.206897), f3 ** 3, (f3 - np.float32(16.0 / 116.0)) / np.float32(7.787))
    xyz2 = t2 * wp
    lin2 = np.clip(xyz2 @ _SRGB_FROM_XYZ_F32.T, 0.0, 1.0)
    return srgb_oetf(lin2).astype(np.float32)


def _hd_softplus_curve(val, midtone: float, eff_shoulder: float):
    """Orthogonal log-print-density H&D curve anchored at MID_GREY (0.18)."""
    e_log = np.log10(np.maximum(val, np.float32(1e-6)) / np.float32(MID_GREY))
    d_mid = np.float32(-np.log10(max(midtone, 1e-4)))
    gamma_p = np.float32(0.70)
    d_lin = d_mid - gamma_p * e_log
    d_min = np.float32(0.004 + 0.016 * eff_shoulder)
    a_hl = np.float32(4.5 / max(eff_shoulder, 0.18))
    d1 = d_min + np.logaddexp(np.float32(0.0), a_hl * (d_lin - d_min)) / a_hl
    d_max = np.float32(3.00)
    a_sh = np.float32(6.0)
    d_out = d_max - np.logaddexp(np.float32(0.0), a_sh * (d_max - d1)) / a_sh
    return np.clip(np.power(np.float32(10.0), -d_out), np.float32(0.0), np.float32(1.0))


_SQRT_LUT_N = 8192
_SQRT_GRID_V = (np.linspace(0.0, 1.0, _SQRT_LUT_N, dtype=np.float64) ** 2)


def _max3(arr: np.ndarray) -> np.ndarray:
    return np.maximum(np.maximum(arr[..., 0], arr[..., 1]), arr[..., 2])


def _min3(arr: np.ndarray) -> np.ndarray:
    return np.minimum(np.minimum(arr[..., 0], arr[..., 1]), arr[..., 2])


def _eval_sqrt_lut(v: np.ndarray, lut: np.ndarray, dlut: np.ndarray | None = None, *, preclipped: bool = False) -> np.ndarray:
    n1 = np.float32(_SQRT_LUT_N - 1)
    vc = v if preclipped else np.clip(v, np.float32(0.0), np.float32(1.0))
    pos = np.sqrt(vc)
    if np.ndim(pos) == 0:
        pos_f = float(pos) * float(n1)
        idx = min(int(pos_f), _SQRT_LUT_N - 2)
        step = float(dlut[idx]) if dlut is not None else float(lut[idx + 1] - lut[idx])
        return np.float32(float(lut[idx]) + (pos_f - idx) * step)
    pos *= n1
    i0 = pos.astype(np.int32)
    np.minimum(i0, _SQRT_LUT_N - 2, out=i0)
    pos -= i0
    out = lut[i0]
    if dlut is not None:
        pos *= dlut[i0]
        out += pos
        return out
    d = lut[i0 + 1] - out
    d *= pos
    out += d
    return out


def _build_post_lut(p: RenderParams, g: float = 1.0) -> tuple[np.ndarray, dict | None]:
    """8192-entry sqrt-spaced float32 LUT for v -> v^g -> srgb_oetf -> contrast -> zonal -> rgb tone_curve."""
    v = _SQRT_GRID_V
    vg = v if g == 1.0 else np.power(v, g)
    e = np.where(vg <= 0.0031308, 12.92 * vg, 1.055 * np.power(vg, 1.0 / 2.4) - 0.055)
    if p.contrast != 1.0:
        pv = float(srgb_oetf(p.midtone))
        e = np.where(e <= pv, pv * np.power(e / pv, p.contrast),
                     1.0 - (1.0 - pv) * np.power(np.maximum(1.0 - e, 0.0) / (1.0 - pv), p.contrast))
    e32 = e.astype(np.float32)
    if p.blacks or p.whites or p.shadows or p.highlights:
        e32 = _apply_zonal_tones(e32, p)
    ch_map = dict(p.tone_curve) if p.tone_curve else None
    if ch_map and "rgb" in ch_map:
        e32 = eval_monotone_spline(ch_map["rgb"], e32)
        ch_map = {k: v for k, v in ch_map.items() if k != "rgb"} or None
    return np.clip(e32, np.float32(0.0), np.float32(1.0)), ch_map


_LUT_CACHE: dict[tuple, tuple] = {}


def _prepare_render_luts(p: RenderParams):
    eff_shoulder = float(np.clip(p.shoulder + max(0.0, -0.45 * p.highlights) + max(0.0, -0.25 * p.whites), 0.0, 1.0))
    if p.hd_curve > 0.0:
        return None
    has_g = bool(eff_shoulder or p.midtone != MID_GREY)
    has_post = bool(has_g or p.contrast != 1.0 or p.blacks or p.whites or p.shadows or p.highlights or p.tone_curve)
    if not has_post:
        return None
    key = (
        round(eff_shoulder, 7), p.midtone, p.contrast,
        p.blacks, p.whites, p.shadows, p.highlights,
        bool(p.chroma_preserve and has_g), p.tone_curve,
    )
    cached = _LUT_CACHE.get(key)
    if cached is not None:
        return cached
    g = (np.log(p.midtone) / np.log(float(shoulder_curve(np.float32(MID_GREY), eff_shoulder)))) if has_g else 1.0
    use_cp = bool(p.chroma_preserve and has_g)
    post_lut, ch_map = _build_post_lut(p, 1.0 if use_cp else g)
    post_dlut = np.diff(post_lut, append=np.float32(0.0))
    if use_cp:
        pow_g_lut = np.power(_SQRT_GRID_V, g).astype(np.float32)
        pow_g_dlut = np.diff(pow_g_lut, append=np.float32(0.0))
    else:
        pow_g_lut = pow_g_dlut = None
    res = (eff_shoulder, g, post_lut, post_dlut, pow_g_lut, pow_g_dlut, ch_map, use_cp)
    if len(_LUT_CACHE) >= 256:
        _LUT_CACHE.clear()
    _LUT_CACHE[key] = res
    return res


def _fused_linear_matrix(ev: float, temp: float, tint: float, matrix: tuple, sat: float) -> np.ndarray | None:
    """Combine ev, wb_gains, colour matrix, and linear saturation into one 3x3 float32 matrix (or None if identity)."""
    if not (ev or temp or tint or matrix != IDENTITY or sat != 1.0):
        return None
    w = (wb_gains(temp, tint) if (temp or tint) else np.ones(3)) * (2.0 ** ev if ev else 1.0)
    m = (np.asarray(matrix, np.float64) if matrix != IDENTITY else np.eye(3)) * w[None, :]
    if sat != 1.0:
        s = float(sat) * np.eye(3) + (1.0 - float(sat)) * np.outer(np.ones(3), LUMA)
        m = s @ m
    return m.astype(np.float32)


def pre_shoulder_linear(x: np.ndarray, p: RenderParams, *, fast_pow: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Steps 1..4b on float32 ``x`` right before ``shoulder_curve``; returns ``(x_linear, below_zero_flags)``."""
    if fast_pow and p.log_unmix == 0.0 and p.vibrance == 0.0 and p.bpc == 0.0 and p.slope_r == 0.0 and p.slope_b == 0.0 and not (p.chroma_preserve and p.sat > 1.0):
        m_lin = _fused_linear_matrix(p.ev, p.temp, p.tint, p.matrix, p.sat)
        if m_lin is not None:
            x = (x.reshape(-1, 3) @ m_lin.T).reshape(x.shape) if x.ndim >= 2 else (x @ m_lin.T)
        neg = x < np.float32(0.0)
        return np.maximum(x, np.float32(0.0)), neg
    if p.log_unmix > 0.0:
        d = -np.log10(np.maximum(x, np.float32(1e-5)))
        m_xtalk = np.array(
            [[1.06, -0.04, -0.02], [-0.02, 1.05, -0.03], [-0.08, 0.14, 0.94]],
            dtype=np.float32,
        )
        m_blend = np.eye(3, dtype=np.float32) + np.float32(p.log_unmix) * (
            m_xtalk - np.eye(3, dtype=np.float32)
        )
        x = np.power(np.float32(10.0), -(d @ m_blend.T))
    if p.ev:
        x = x * np.float32(2.0 ** p.ev)
    if p.temp or p.tint:
        x = x * wb_gains(p.temp, p.tint).astype(np.float32)
    if p.matrix != IDENTITY:
        x = x @ np.asarray(p.matrix, np.float32).T
    if p.sat != 1 or p.vibrance:
        y = (x @ LUMA_F32)[..., None]
        if p.vibrance:
            denom = np.maximum(y[..., 0], np.float32(0.05))
            rel_c = np.clip((_max3(x) - _min3(x)) / denom, np.float32(0.0), np.float32(1.5))
            u_vib = np.float32(1.0) - np.clip(rel_c / np.float32(0.85), np.float32(0.0), np.float32(1.0))
            vib_w = u_vib * np.sqrt(u_vib)
            gain = np.float32(p.sat) * (np.float32(1.0) + np.float32(0.65 * p.vibrance) * vib_w)[..., None]
        else:
            gain = np.float32(p.sat)
        if p.chroma_preserve:  # T4c: a saturation boost fades out in deep shadows (no coloured noise in blacks)
            dark = _smoothstep(y, 0.0, 0.02)
            gain = np.where(gain > 1, 1 + (gain - 1) * dark, gain)
        x = y + np.maximum(gain, np.float32(0.0)) * (x - y)
    if p.bpc:
        bpc_v = np.float32(p.bpc)
        x = (x - bpc_v) / np.float32(1.0 - bpc_v)
    neg = x < 0
    x = np.maximum(x, np.float32(0.0))
    if p.slope_r or p.slope_b:
        # RawTherapee Film Negative style per-channel power ratio anchored at MID_GREY
        x = x.copy()
        piv = np.float32(MID_GREY)
        inv_piv = np.float32(1.0 / MID_GREY)
        if p.slope_r:
            gr = np.float32(2.0 ** (-0.65 * p.slope_r))
            r_norm = np.ascontiguousarray(x[..., 0])
            np.multiply(r_norm, inv_piv, out=r_norm)
            np.maximum(r_norm, np.float32(1e-6), out=r_norm)
            x[..., 0] = piv * (np.exp2(gr * np.log2(r_norm)) if fast_pow else np.power(r_norm, gr))
        if p.slope_b:
            gb = np.float32(2.0 ** (-0.65 * p.slope_b))
            b_norm = np.ascontiguousarray(x[..., 2])
            np.multiply(b_norm, inv_piv, out=b_norm)
            np.maximum(b_norm, np.float32(1e-6), out=b_norm)
            x[..., 2] = piv * (np.exp2(gb * np.log2(b_norm)) if fast_pow else np.power(b_norm, gb))
    return x, neg


def _soft_clip_arr(val: np.ndarray, soft_clip: float) -> np.ndarray:
    if not soft_clip:
        return np.minimum(val, np.float32(1.0))
    th = np.float32(soft_clip)
    inv_1mt = np.float32(1.0 / max(1.0 - float(th), 1e-4))
    u = np.maximum(val - th, np.float32(0.0)) * inv_1mt
    v = np.where(val > th, th + np.float32(1.0 - float(th)) * u / (np.float32(1.0) + u), val)
    return np.minimum(v, np.float32(1.0))


def _blend_chroma_preserve(x: np.ndarray, v_ch: np.ndarray, lum_curve, strength: float) -> np.ndarray:
    y_in = np.maximum(x @ LUMA_F32, np.float32(1e-6))[..., None]
    v_y = lum_curve(y_in)
    v_lum = np.clip(x * (v_y / y_in), np.float32(0.0), np.float32(1.0))
    w = (np.float32(strength) * _smoothstep(v_y, *CHROMA_PRESERVE_DARK)
         * (np.float32(1.0) - _smoothstep(v_y, *CHROMA_PRESERVE_BRIGHT)))
    return (np.float32(1.0) - w) * v_ch + w * v_lum


def _render_core(x, p: RenderParams, *, _pre=None):
    """Per-pixel steps on float32 ``x`` (..., 3): (encoded output, below-zero flags after colour,
    above-white flags after the shoulder, display-linear values before the white clip)."""
    x, neg = pre_shoulder_linear(x, p, fast_pow=(_pre is not None))
    eff_shoulder = float(np.clip(p.shoulder + max(0.0, -0.45 * p.highlights) + max(0.0, -0.25 * p.whites), 0.0, 1.0))
    pre_clip = shoulder_curve(x, eff_shoulder)
    over = pre_clip > 1
    if _pre is not None:
        _, _, post_lut, post_dlut, pow_g_lut, pow_g_dlut, ch_map, use_cp = _pre
        v = _soft_clip_arr(pre_clip, p.soft_clip)
        if use_cp:
            v_ch = _eval_sqrt_lut(v, pow_g_lut, pow_g_dlut, preclipped=True)
            v = _blend_chroma_preserve(
                x, v_ch,
                lambda y_in: _eval_sqrt_lut(_soft_clip_arr(shoulder_curve(y_in, eff_shoulder), p.soft_clip), pow_g_lut, pow_g_dlut, preclipped=True),
                p.chroma_preserve,
            )
        e = _eval_sqrt_lut(v, post_lut, post_dlut, preclipped=True)
        if ch_map is not None:
            for idx, k in enumerate(("r", "g", "b")):
                if k in ch_map:
                    e[..., idx] = eval_monotone_spline(ch_map[k], e[..., idx])
        if p.skin_protect:
            e = _apply_skin_protect(e, p.skin_protect)
        return e, neg, over, pre_clip

    if p.hd_curve >= 1.0:
        v = _hd_softplus_curve(x, p.midtone, eff_shoulder)
        if p.chroma_preserve:
            v = _blend_chroma_preserve(x, v, lambda y_in: _hd_softplus_curve(y_in, p.midtone, eff_shoulder), p.chroma_preserve)
    else:
        v = _soft_clip_arr(pre_clip, p.soft_clip) if p.soft_clip else np.minimum(pre_clip, 1)
        if eff_shoulder or p.midtone != MID_GREY:
            g = np.log(p.midtone) / np.log(float(shoulder_curve(np.float32(MID_GREY), eff_shoulder)))
            v_ch = np.power(v, np.float32(g))
            if p.chroma_preserve:
                v = _blend_chroma_preserve(
                    x, v_ch,
                    lambda y_in: np.power(_soft_clip_arr(shoulder_curve(y_in, eff_shoulder), p.soft_clip) if p.soft_clip else np.minimum(shoulder_curve(y_in, eff_shoulder), 1), np.float32(g)),
                    p.chroma_preserve,
                )
            else:
                v = v_ch
        if p.hd_curve > 0.0:
            v_hd = _hd_softplus_curve(x, p.midtone, eff_shoulder)
            if p.chroma_preserve:
                v_hd = _blend_chroma_preserve(x, v_hd, lambda y_in: _hd_softplus_curve(y_in, p.midtone, eff_shoulder), p.chroma_preserve)
            v = np.float32(1.0 - p.hd_curve) * v + np.float32(p.hd_curve) * v_hd
    e = srgb_oetf(v).astype(np.float32)
    if p.contrast != 1:
        pv, c = np.float32(srgb_oetf(p.midtone)), np.float32(p.contrast)
        lo_mask = e <= pv
        e_out = np.empty_like(e)
        if lo_mask.any():
            e_out[lo_mask] = pv * np.power(e[lo_mask] / pv, c)
        hi_mask = ~lo_mask
        if hi_mask.any():
            one = np.float32(1.0)
            e_out[hi_mask] = one - (one - pv) * np.power(np.maximum(one - e[hi_mask], np.float32(0.0)) / (one - pv), c)
        e = e_out
    if p.blacks or p.whites or p.shadows or p.highlights:
        e = _apply_zonal_tones(e, p)
    if p.tone_curve:
        e = _apply_tone_curve(e, p.tone_curve)
    if p.skin_protect:
        e = _apply_skin_protect(e, p.skin_protect)
    return e, neg, over, pre_clip


def render_fast_full(P, p: RenderParams):
    """Fast LUT-accelerated equivalent of ``render_full`` (<4e-7 max diff) for internal parameter estimation."""
    x = np.asarray(P, np.float32)
    if not _all_finite(x):
        raise ValueError("input contains NaN or infinite values")
    return _render_core(x, p, _pre=_prepare_render_luts(p))


T19_BOUNDS = dict(
    ev=(-3.0, 3.0),
    temp=(-1.5, 1.5),
    tint=(-1.5, 1.5),
    sat=(0.0, 2.0),
    contrast=(0.5, 2.5),
    shoulder=(0.0, 1.0),
    toe=(-1.0, 1.0),
)


@dataclass
class RenderParamsT19:
    """T19 orthogonal 7-parameter log-exposure paper curve (Jacobian cond ~ 4.3)."""
    ev: float = 0.0
    temp: float = 0.0
    tint: float = 0.0
    sat: float = 1.0
    contrast: float = 1.15
    shoulder: float = 0.5
    toe: float = 0.0
    matrix: tuple = IDENTITY

    def __post_init__(self):
        for name, (lo, hi) in T19_BOUNDS.items():
            v = float(getattr(self, name))
            if not (np.isfinite(v) and lo <= v <= hi):
                raise ValueError(f"{name}={v} outside [{lo}, {hi}]")
            setattr(self, name, v)
        self.matrix = _validate_matrix(self.matrix)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in T19_BOUNDS}
        d["matrix"] = [list(r) for r in self.matrix]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RenderParamsT19:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown T19 render parameters: {sorted(unknown)}")
        return cls(**d)


def _t19_curve_only(x: np.ndarray, contrast: float, shoulder: float, toe: float) -> np.ndarray:
    L_hi = np.float32(np.log2(1.0 / MID_GREY))
    L_lo = np.float32(5.5)
    p_sh = np.float32(8.0 - (8.0 - 1.25) * shoulder)
    toe_pos = float(np.log1p(np.exp(4.0 * toe)) / 4.0)
    toe_neg = float(np.log1p(np.exp(-4.0 * toe)) / 4.0)
    q_sh = np.float32(max(10.0 - 8.2 * toe_pos, 1.5))
    k_blk = np.float32(0.55 * toe_neg)
    c = np.float32(contrast)

    x_safe = np.maximum(x, np.float32(1e-6))
    u = c * np.log2(x_safe * np.float32(1.0 / MID_GREY))
    pos = u >= np.float32(0.0)
    tilde = np.empty_like(u)
    if pos.any():
        up = u[pos] + np.float32(1e-7)
        tilde[pos] = up / np.power(np.float32(1.0) + np.power(up / L_hi, p_sh), np.float32(1.0) / p_sh)
    nmask = ~pos
    if nmask.any():
        un = (-u[nmask]) + np.float32(1e-7)
        tilde[nmask] = (
            -un / np.power(np.float32(1.0) + np.power(un / L_lo, q_sh), np.float32(1.0) / q_sh)
            - k_blk * (un * un * un) / (np.float32(1.0) + un * un)
        )
    v = np.clip(np.float32(MID_GREY) * np.exp2(tilde), np.float32(0.0), np.float32(1.0))
    return srgb_oetf(v).astype(np.float32)


_T19_LUT_N = 16385  # odd size -> u=0.5 (x=0.18 mid-grey) hits exact grid index 8192
_T19_U_GRID = np.linspace(0.0, 1.0 - 1e-7, _T19_LUT_N, dtype=np.float64)
_T19_X_GRID = (MID_GREY * _T19_U_GRID / (1.0 - _T19_U_GRID)).astype(np.float32)
_T19_LUT_CACHE: dict[tuple, tuple] = {}


def _prepare_t19_lut(p: RenderParamsT19) -> tuple[np.ndarray, np.ndarray, np.float32]:
    key = (p.contrast, p.shoulder, p.toe)
    cached = _T19_LUT_CACHE.get(key)
    if cached is not None:
        return cached
    lut = _t19_curve_only(_T19_X_GRID, p.contrast, p.shoulder, p.toe)
    dlut = np.diff(lut, append=np.float32(0.0))
    # u = c * log2(x / 0.18) > log2(1 / 0.18)  <=>  x > 0.18 * (1 / 0.18)**(1/c)
    x_over = np.float32(MID_GREY * ((1.0 / MID_GREY) ** (1.0 / p.contrast)))
    res = (lut, dlut, x_over)
    if len(_T19_LUT_CACHE) >= 256:
        _T19_LUT_CACHE.clear()
    _T19_LUT_CACHE[key] = res
    return res


def _apply_t19_curve_or_lut(
    x: np.ndarray, p: RenderParamsT19 | RenderParamsT24, _t19_pre, *, inplace: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    if _t19_pre is not None:
        lut, dlut, x_over = _t19_pre
        over = x > x_over
        xc = np.maximum(x, np.float32(0.0), out=x) if inplace else np.maximum(x, np.float32(0.0))
        denom = xc + np.float32(MID_GREY)
        np.divide(xc, denom, out=xc)
        xc *= np.float32(_T19_LUT_N - 1)
        i0 = xc.astype(np.int32)
        np.minimum(i0, _T19_LUT_N - 2, out=i0)
        xc -= i0
        xc *= dlut[i0]
        xc += lut[i0]
        return xc, over
    x_pos = np.maximum(x, np.float32(1e-6), out=x) if inplace else np.maximum(x, np.float32(1e-6))
    x_over = np.float32(MID_GREY * ((1.0 / MID_GREY) ** (1.0 / p.contrast)))
    over = x_pos > x_over
    return _t19_curve_only(x_pos, p.contrast, p.shoulder, p.toe), over


def _render_t19_core(x: np.ndarray, p: RenderParamsT19, *, _t19_pre=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m_lin = _fused_linear_matrix(p.ev, p.temp, p.tint, p.matrix, p.sat)
    owned = False
    if m_lin is not None:
        x = (x.reshape(-1, 3) @ m_lin.T).reshape(x.shape) if x.ndim >= 2 else (x @ m_lin.T)
        owned = True
    neg = x < np.float32(0.0)
    e, over = _apply_t19_curve_or_lut(x, p, _t19_pre, inplace=owned)
    return e, neg, over


def render_t19(P, p: RenderParamsT19, mask=None, *, chunk_rows: int = CHUNK_ROWS, fast: bool = True):
    """Render T18-inverted linear positive ``P`` with the T19 orthogonal C2 log-exposure paper curve."""
    pre = _prepare_t19_lut(p) if fast else None
    return _render_chunked(np.asarray(P, np.float32), lambda xc: _render_t19_core(xc, p, _t19_pre=pre), mask, chunk_rows)


RenderParamsT22 = RenderParamsT19
render_t22 = render_t19

RenderParamsT23 = RenderParamsT19
render_t23 = render_t19


ZERO_MATRIX = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

T24_BOUNDS = dict(
    T19_BOUNDS,
    slope_r=(-0.6, 0.6),
    slope_b=(-0.6, 0.6),
)


def _validate_eps_rgb(eps) -> tuple[float, float, float]:
    arr = np.asarray(eps, dtype=np.float64)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError("eps_rgb must be a finite 3-element sequence")
    if np.any((arr < -0.1) | (arr > 0.1)):
        raise ValueError(f"eps_rgb outside [-0.1, 0.1]: {arr}")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _validate_residual_matrix(matrix) -> tuple:
    if matrix is ZERO_MATRIX or (isinstance(matrix, tuple) and matrix == ZERO_MATRIX):
        return ZERO_MATRIX
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        raise ValueError("residual_matrix must be a finite 3x3 array")
    if np.any(np.abs(m) > 0.35):
        raise ValueError("residual_matrix entries outside [-0.35, 0.35]")
    return tuple(tuple(float(x) for x in row) for row in m)


@dataclass
class RenderParamsT24:
    """T24 physical parameters: T19 curve + black-level pedestal eps_rgb, residual_matrix, and channel slopes."""
    ev: float = 0.0
    temp: float = 0.0
    tint: float = 0.0
    sat: float = 1.0
    contrast: float = 1.15
    shoulder: float = 0.5
    toe: float = 0.0
    slope_r: float = 0.0
    slope_b: float = 0.0
    eps_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    residual_matrix: tuple = ZERO_MATRIX
    matrix: tuple = IDENTITY
    floor_restore: float = 0.0

    def __post_init__(self):
        for name, (lo, hi) in T24_BOUNDS.items():
            v = float(getattr(self, name))
            if not (np.isfinite(v) and lo <= v <= hi):
                raise ValueError(f"{name}={v} outside [{lo}, {hi}]")
            setattr(self, name, v)
        self.eps_rgb = _validate_eps_rgb(self.eps_rgb)
        self.residual_matrix = _validate_residual_matrix(self.residual_matrix)
        self.matrix = _validate_matrix(self.matrix)
        fr = float(self.floor_restore)
        if not (np.isfinite(fr) and 0.0 <= fr <= 0.50):
            raise ValueError(f"floor_restore={fr} outside [0.0, 0.50]")
        self.floor_restore = fr

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in T24_BOUNDS}
        d["eps_rgb"] = list(self.eps_rgb)
        d["residual_matrix"] = [list(r) for r in self.residual_matrix]
        d["matrix"] = [list(r) for r in self.matrix]
        if self.floor_restore != 0.0:
            d["floor_restore"] = self.floor_restore
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RenderParamsT24:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown T24 render parameters: {sorted(unknown)}")
        return cls(**d)


def _t24_linear_matrix(p: RenderParamsT24) -> np.ndarray | None:
    if not (p.ev or p.temp or p.tint or p.sat != 1.0 or p.matrix != IDENTITY or p.residual_matrix != ZERO_MATRIX):
        return None
    mat_eff = np.asarray(p.matrix, dtype=np.float64) + np.asarray(p.residual_matrix, dtype=np.float64)
    w = (wb_gains(p.temp, p.tint) if (p.temp or p.tint) else np.ones(3)) * (2.0 ** p.ev if p.ev else 1.0)
    m = mat_eff * w[None, :]
    if p.sat != 1.0:
        s = float(p.sat) * np.eye(3) + (1.0 - float(p.sat)) * np.outer(np.ones(3), LUMA)
        m = s @ m
    return m.astype(np.float32)


def _render_t24_core(
    x: np.ndarray, p: RenderParamsT24, m_lin: np.ndarray | None, *, _t19_pre=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    owned = False
    if p.floor_restore > 0.0:
        fr = np.float32(p.floor_restore)
        x = np.maximum(x, np.float32(0.0)) * (np.float32(1.0) - fr) + fr
        owned = True
    if p.eps_rgb != (0.0, 0.0, 0.0):
        eps = np.asarray(p.eps_rgb, dtype=np.float32)
        x = x + eps
        np.maximum(x, np.float32(1e-6), out=x)
        owned = True
    if m_lin is not None:
        x = (x.reshape(-1, 3) @ m_lin.T).reshape(x.shape) if x.ndim >= 2 else (x @ m_lin.T)
        owned = True
    neg = x < np.float32(0.0)
    if p.slope_r != 0.0 or p.slope_b != 0.0:
        x = np.maximum(x, np.float32(1e-6), out=x) if owned else np.maximum(x, np.float32(1e-6))
        owned = True
        inv_mid = np.float32(1.0 / MID_GREY)
        mid32 = np.float32(MID_GREY)
        for ch, s in ((0, p.slope_r), (2, p.slope_b)):
            if s != 0.0:
                c = x[..., ch] * inv_mid
                np.log2(c, out=c)
                c *= np.float32(1.0 + s)
                np.exp2(c, out=c)
                c *= mid32
                x[..., ch] = c
    e, over = _apply_t19_curve_or_lut(x, p, _t19_pre, inplace=owned)
    return e, neg, over


def render_t24(P, p: RenderParamsT24, mask=None, *, chunk_rows: int = CHUNK_ROWS, fast: bool = True):
    """Render T23-inverted linear positive ``P`` with T24 black-level pedestal, residual matrix, channel slopes, and T19 curve."""
    m_lin = _t24_linear_matrix(p)
    pre = _prepare_t19_lut(p) if fast else None
    return _render_chunked(np.asarray(P, np.float32), lambda xc: _render_t24_core(xc, p, m_lin, _t19_pre=pre), mask, chunk_rows)
