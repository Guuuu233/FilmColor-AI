"""16-bit TIFF export with an embedded ICC profile.

The engine output is uncalibrated camera RGB (docs/ENGINE_MATH.md). A colour-managed viewer still
needs primaries, so the embedded profile interprets the data with sRGB primaries, and says so in
its description. It is not a colorimetric claim.
"""
from __future__ import annotations

from functools import lru_cache
import struct

import numpy as np
import tifffile

# sRGB colorants adapted to the D50 PCS (Bradford), as in common sRGB ICC profiles
SRGB_D50 = ((0.436066, 0.222488, 0.013916), (0.385147, 0.716873, 0.097076), (0.143066, 0.060608, 0.714096))
D50 = (0.9642, 1.0, 0.8249)
ENCODINGS = ("srgb", "linear")
DESCRIPTION = {
    "srgb": "FilmColor uncalibrated camera RGB - sRGB primaries assumed, sRGB transfer",
    "linear": "FilmColor uncalibrated camera RGB - sRGB primaries assumed, linear transfer",
}


def _s15f16(v: float) -> bytes:
    return struct.pack(">i", int(round(v * 65536)))


def _xyz(v) -> bytes:
    return b"XYZ " + bytes(4) + b"".join(_s15f16(x) for x in v)


def _desc(text: str) -> bytes:
    a = text.encode("ascii") + b"\0"
    # textDescriptionType: ASCII part, empty Unicode part, empty ScriptCode part (67-byte field)
    return b"desc" + bytes(4) + struct.pack(">I", len(a)) + a + bytes(8) + bytes(3) + bytes(67)


def _text(text: str) -> bytes:
    return b"text" + bytes(4) + text.encode("ascii") + b"\0"


def _curv(encoding: str) -> bytes:
    if encoding == "linear":
        return b"curv" + bytes(4) + struct.pack(">I", 0)  # identity
    x = np.linspace(0, 1, 4096)
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)  # device value -> linear
    table = np.round(lin * 65535).astype(">u2").tobytes()
    return b"curv" + bytes(4) + struct.pack(">I", 4096) + table


@lru_cache(maxsize=4)
def icc_profile(encoding: str = "srgb") -> bytes:
    """Minimal ICC v2.1 RGB display profile (matrix/TRC). Deterministic: no date, no profile ID."""
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {ENCODINGS}")
    trc = _curv(encoding)
    tags = [
        (b"desc", _desc(DESCRIPTION[encoding])),
        (b"cprt", _text("No copyright, use freely")),
        (b"wtpt", _xyz(D50)),
        (b"rXYZ", _xyz(SRGB_D50[0])), (b"gXYZ", _xyz(SRGB_D50[1])), (b"bXYZ", _xyz(SRGB_D50[2])),
        (b"rTRC", trc), (b"gTRC", trc), (b"bTRC", trc),
    ]
    offset = 128 + 4 + 12 * len(tags)
    table, data, placed = b"", b"", {}
    for sig, body in tags:
        if body in placed:  # identical TRCs share one data block
            table += sig + struct.pack(">II", placed[body], len(body))
            continue
        pad = (-offset) % 4
        data += bytes(pad)
        offset += pad
        placed[body] = offset
        table += sig + struct.pack(">II", offset, len(body))
        data += body
        offset += len(body)
    body = struct.pack(">I", len(tags)) + table + data
    size = 128 + len(body)
    header = (struct.pack(">I", size) + bytes(4) + struct.pack(">I", 0x02100000) + b"mntrRGB XYZ "
              + bytes(12) + b"acsp" + bytes(4) + bytes(4) + bytes(4) + bytes(4) + bytes(8)
              + struct.pack(">I", 0) + b"".join(_s15f16(x) for x in D50) + bytes(4))
    header += bytes(128 - len(header))
    return header + body


def to_uint16(img) -> np.ndarray:
    return np.round(np.clip(img, 0, 1) * 65535).astype(np.uint16)


def write_tiff16(path, img, encoding: str = "srgb") -> None:
    """Write (H, W, 3) values in [0, 1], already encoded per ``encoding``, as 16-bit RGB TIFF."""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) image, got shape {img.shape}")
    tifffile.imwrite(path, to_uint16(img), photometric="rgb", iccprofile=icc_profile(encoding),
                     compression="zlib", software="filmcolor")
