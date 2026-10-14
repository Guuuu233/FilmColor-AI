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


def _s15f16(b: bytes) -> float:
    return struct.unpack(">i", b)[0] / 65536


def _tags(icc: bytes) -> dict:
    n = struct.unpack(">I", icc[128:132])[0]
    out = {}
    for i in range(n):
        sig, off, size = struct.unpack(">4sII", icc[132 + 12 * i : 144 + 12 * i])
        out[sig.decode("latin-1")] = icc[off : off + size]
    return out


def _trc(data: bytes):
    """Return a function mapping encoded [0, 1] -> linear [0, 1]."""
    kind = data[:4]
    if kind == b"curv":
        n = struct.unpack(">I", data[8:12])[0]
        if n == 0:
            return lambda x: x
        if n == 1:
            g = struct.unpack(">H", data[12:14])[0] / 256
            return lambda x: np.power(np.clip(x, 0, 1), g)
        table = np.frombuffer(data[12 : 12 + 2 * n], ">u2").astype(np.float64) / 65535
        grid = np.linspace(0, 1, n)
        return lambda x: np.interp(np.clip(x, 0, 1), grid, table)
    if kind == b"para":
        ftype = struct.unpack(">H", data[8:10])[0]
        nparams = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}[ftype]
        p = [_s15f16(data[12 + 4 * i : 16 + 4 * i]) for i in range(nparams)]
        g, a, b, c, d, e, f = (p + [0.0] * 7)[:7]  # ICC order; unused ones stay 0
        if ftype == 0:
            return lambda x: np.power(np.clip(x, 0, 1), g)
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
        if icc[16:20] != b"RGB " or icc[36:40] != b"acsp":
            raise ValueError("not an RGB ICC profile")
        t = _tags(icc)
        missing = {"rXYZ", "gXYZ", "bXYZ", "rTRC", "gTRC", "bTRC"} - set(t)
        if missing:
            raise ValueError(f"not a matrix/TRC profile (missing {sorted(missing)})")
        cols = [[_s15f16(t[s][8 + 4 * i : 12 + 4 * i]) for i in range(3)] for s in ("rXYZ", "gXYZ", "bXYZ")]
        self.matrix = np.array(cols).T          # linear RGB -> XYZ(D50)
        self.trc = [_trc(t[s]) for s in ("rTRC", "gTRC", "bTRC")]

    def to_xyz(self, rgb01):
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
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)


def delta_e2000(lab1, lab2):
    """CIEDE2000 (kL = kC = kH = 1), Sharma et al. 2005."""
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - np.sqrt(Cm7 / (Cm7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lpm, Cpm = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1p + h2p
    hpm = np.where(C1p * C2p == 0, hs,
                   np.where(np.abs(h1p - h2p) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)))
    T = (1 - 0.17 * np.cos(np.radians(hpm - 30)) + 0.24 * np.cos(np.radians(2 * hpm))
         + 0.32 * np.cos(np.radians(3 * hpm + 6)) - 0.20 * np.cos(np.radians(4 * hpm - 63)))
    dtheta = 30 * np.exp(-(((hpm - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cpm ** 7 / (Cpm ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lpm - 50) ** 2 / np.sqrt(20 + (Lpm - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cpm, 1 + 0.015 * Cpm * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))
