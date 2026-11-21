"""Colour conversions for evaluation: ICC matrix/TRC profiles -> XYZ (D50) -> CIELAB, CIEDE2000.
Pure numpy. Only RGB matrix/TRC display profiles are supported (sRGB-, ProPhoto-, AdobeRGB-like).
"""
from __future__ import annotations

import struct

import numpy as np

D50 = np.array([0.9642, 1.0, 0.8249])
# XYZ(D50) -> linear sRGB (D65), Bradford-adapted; for previews only
XYZ_D50_TO_SRGB = np.array([[3.1338561, -1.6168667, -0.4906146],
                            [-0.9787684, 1.9161415, 0.0334540],
                            [0.0719453, -0.2289914, 1.4052427]])
_25_POW_7 = 25.0 ** 7
_RAD_30 = np.radians(30.0)
_RAD_6 = np.radians(6.0)
_RAD_63 = np.radians(63.0)


def _s15f16(b: bytes) -> float:
    return struct.unpack(">i", b)[0] / 65536


def _tags(icc: bytes) -> dict:
    if len(icc) < 132:
        raise ValueError("truncated ICC profile header")
    n = struct.unpack(">I", icc[128:132])[0]
    if n > 4096 or 132 + 12 * n > len(icc):
        raise ValueError("corrupt ICC tag table")
    out = {}
    for i in range(n):
        sig, off, size = struct.unpack(">4sII", icc[132 + 12 * i : 144 + 12 * i])
        if off + size > len(icc):
            raise ValueError(f"ICC tag {sig!r} exceeds profile length")
        out[sig.decode("latin-1")] = icc[off : off + size]
    return out


def _trc(data: bytes):
    """Return a function mapping encoded [0, 1] -> linear [0, 1]."""
    if len(data) < 12:
        raise ValueError("truncated ICC TRC tag")
    kind = data[:4]
    if kind == b"curv":
        n = struct.unpack(">I", data[8:12])[0]
        if 12 + 2 * n > len(data):
            raise ValueError("truncated ICC curv table")
        if n == 0:
            return lambda x: np.clip(x, 0, 1)
        if n == 1:
            g = struct.unpack(">H", data[12:14])[0] / 256
            return lambda x: np.power(np.clip(x, 0, 1), g)
        table = np.frombuffer(data[12 : 12 + 2 * n], ">u2").astype(np.float64) / 65535.0
        dtbl = np.diff(table, append=0.0)
        scale = float(n - 1)
        max_idx = n - 2

        def _eval_uniform_lut(x):
            pos = np.clip(x, 0.0, 1.0) * scale
            if np.ndim(pos) == 0:
                idx = min(int(pos), max_idx)
                return float(table[idx] + (float(pos) - idx) * dtbl[idx])
            i0 = pos.astype(np.intp)
            np.minimum(i0, max_idx, out=i0)
            return table[i0] + (pos - i0) * dtbl[i0]

        return _eval_uniform_lut
    if kind == b"para":
        ftype = struct.unpack(">H", data[8:10])[0]
        if ftype not in (0, 1, 2, 3, 4):
            raise ValueError(f"unsupported ICC parametric curve type {ftype}")
        nparams = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}[ftype]
        if 12 + 4 * nparams > len(data):
            raise ValueError("truncated ICC parametric curve parameters")
        p = [_s15f16(data[12 + 4 * i : 16 + 4 * i]) for i in range(nparams)]
        g, a, b, c, d, e, f = (p + [0.0] * 7)[:7]  # ICC order; unused ones stay 0
        if ftype == 0:
            return lambda x: np.power(np.clip(x, 0, 1), g)
        if ftype in (1, 2) and abs(a) < 1e-12:
            raise ValueError("degenerate ICC parametric curve (a == 0)")
        if ftype == 1:
            return lambda x: np.where(x >= -b / a, np.power(np.clip(a * x + b, 0, None), g), 0.0)
        if ftype == 2:
            return lambda x: np.where(x >= -b / a, np.power(np.clip(a * x + b, 0, None), g) + c, c)
        if ftype == 3:
            return lambda x: np.where(x >= d, np.power(np.clip(a * x + b, 0, None), g), c * x)
        return lambda x: np.where(x >= d, np.power(np.clip(a * x + b, 0, None), g) + e, c * x + f)
    raise ValueError(f"unsupported TRC type {kind!r}")


class MatrixTRCProfile:
    def __init__(self, icc: bytes):
        if len(icc) < 132 or icc[16:20] != b"RGB " or icc[36:40] != b"acsp":
            raise ValueError("not an RGB ICC profile")
        t = _tags(icc)
        missing = {"rXYZ", "gXYZ", "bXYZ", "rTRC", "gTRC", "bTRC"} - set(t)
        if missing:
            raise ValueError(f"not a matrix/TRC profile (missing {sorted(missing)})")
        for s in ("rXYZ", "gXYZ", "bXYZ"):
            if len(t[s]) < 20:
                raise ValueError(f"truncated {s} tag in ICC profile")
        cols = [[_s15f16(t[s][8 + 4 * i : 12 + 4 * i]) for i in range(3)] for s in ("rXYZ", "gXYZ", "bXYZ")]
        self.matrix = np.array(cols).T          # linear RGB -> XYZ(D50)
        trc_bytes = (t["rTRC"], t["gTRC"], t["bTRC"])
        self._shared_trc = (trc_bytes[0] == trc_bytes[1] == trc_bytes[2])
        if self._shared_trc:
            fn = _trc(trc_bytes[0])
            self.trc = [fn, fn, fn]
        else:
            self.trc = [_trc(b) for b in trc_bytes]

    def to_xyz(self, rgb01):
        if self._shared_trc:
            lin = self.trc[0](rgb01)
        else:
            lin = np.stack([self.trc[c](rgb01[..., c]) for c in range(3)], -1)
        return lin @ self.matrix.T


def xyz_to_lab(xyz, white=D50):
    r = xyz / white
    f = np.where(r > (6 / 29) ** 3, np.cbrt(r), r / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def lab_to_xyz(lab, white=D50):
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    r = np.where(f > 6 / 29, f ** 3, 3 * (6 / 29) ** 2 * (f - 4 / 29))
    return r * white


def xyz_to_srgb_preview(xyz):
    lin = np.clip(xyz @ XYZ_D50_TO_SRGB.T, 0, 1)
    if lin.ndim > 0 and lin.dtype == np.float32:
        out = lin * np.float32(12.92)
        hi = lin > np.float32(0.0031308)
        if hi.any():
            out[hi] = np.float32(1.055) * np.power(lin[hi], np.float32(1.0 / 2.4)) - np.float32(0.055)
        return out
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def delta_e2000(lab1, lab2):
    """CIEDE2000 (kL = kC = kH = 1), Sharma et al. 2005."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm7 = (0.5 * (C1 + C2)) ** 7
    one_plus_G = 1.5 - 0.5 * np.sqrt(Cm7 / (Cm7 + _25_POW_7))
    a1p, a2p = one_plus_G * a1, one_plus_G * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    c_prod = C1p * C2p
    zero_c = c_prod == 0
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(zero_c, 0, dh)
    dHp = 2 * np.sqrt(c_prod) * np.sin(np.radians(0.5 * dh))
    Lpm, Cpm = 0.5 * (L1 + L2), 0.5 * (C1p + C2p)
    hs = h1p + h2p
    hpm = np.where(zero_c, hs,
                   np.where(np.abs(h1p - h2p) <= 180, 0.5 * hs,
                            np.where(hs < 360, 0.5 * (hs + 360), 0.5 * (hs - 360))))
    hpm_rad = np.radians(hpm)
    T = (1 - 0.17 * np.cos(hpm_rad - _RAD_30) + 0.24 * np.cos(2 * hpm_rad)
         + 0.32 * np.cos(3 * hpm_rad + _RAD_6) - 0.20 * np.cos(4 * hpm_rad - _RAD_63))
    dtheta_rad = np.radians(30 * np.exp(-(((hpm - 275) * 0.04) ** 2)))
    Cpm7 = Cpm ** 7
    Rc = 2 * np.sqrt(Cpm7 / (Cpm7 + _25_POW_7))
    Lm50_sq = (Lpm - 50) ** 2
    Sl = 1 + 0.015 * Lm50_sq / np.sqrt(20 + Lm50_sq)
    Sc, Sh = 1 + 0.045 * Cpm, 1 + 0.015 * Cpm * T
    Rt = -np.sin(2 * dtheta_rad) * Rc
    tL, tC, tH = dLp / Sl, dCp / Sc, dHp / Sh
    return np.sqrt(tL * tL + tC * tC + tH * tH + Rt * tC * tH)
