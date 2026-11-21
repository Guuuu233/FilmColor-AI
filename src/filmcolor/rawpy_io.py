"""rawpy/LibRaw adapter. The only module that touches the RAW decoding library."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import rawpy

from .raw import RawFrame


class RawDecodeError(ValueError):
    """The decoder could not read the file (not a supported RAW, truncated or corrupted)."""


@contextmanager
def _open(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"RAW file not found: {path}")
    try:
        with rawpy.imread(str(path)) as r:
            yield r
    except rawpy.LibRawError as e:
        raise RawDecodeError(f"{path}: LibRaw could not decode the file ({type(e).__name__}: {e}); "
                             "not a supported RAW, truncated or corrupted") from e


def _colors_match_pattern(raw_colors: np.ndarray, pattern: np.ndarray) -> bool:
    n = pattern.shape[0]
    h, w = raw_colors.shape
    hb, wb = h // n, w // n
    if hb and wb:
        core = raw_colors[: hb * n, : wb * n].reshape(hb, n, wb, n)
        pat_raw = pattern[None, :, None, :]
        # Bayer G2 is 3 in raw_colors_visible, folded to 1 in pattern
        ok = (core == pat_raw) | ((core == 3) & (pat_raw == 1))
        if not bool(ok.all()):
            return False
    if h % n:
        rem = np.where(raw_colors[hb * n :, : wb * n] == 3, 1, raw_colors[hb * n :, : wb * n])
        if not np.array_equal(rem, np.tile(pattern[: h % n, :], (1, wb))):
            return False
    if w % n:
        rem = np.where(raw_colors[:, wb * n :] == 3, 1, raw_colors[:, wb * n :])
        reps_r = -(-h // n)
        if not np.array_equal(rem, np.tile(pattern[:, : w % n], (reps_r, 1))[:h, :]):
            return False
    return True


def _extract_frame(r, path: str | Path) -> RawFrame:
    if r.raw_type != rawpy.RawType.Flat or r.raw_pattern is None:
        raise ValueError(f"{path}: only single-plane CFA RAW files are supported")
    lib_pattern = np.asarray(r.raw_pattern)
    black4 = np.asarray(r.black_level_per_channel, dtype=np.float64)
    black = black4[lib_pattern]  # per CFA position, before folding G2 into G
    pattern = np.where(lib_pattern == 3, 1, lib_pattern).astype(np.uint8)
    n = pattern.shape[0]
    rc = r.raw_colors_visible
    lib_origin = np.where(rc[:n, :n] == 3, 1, rc[:n, :n])
    if not np.array_equal(lib_origin, pattern):
        raise ValueError(f"{path}: raw_pattern does not tile raw_colors_visible from origin")
    if not _colors_match_pattern(rc, pattern):
        raise ValueError(f"{path}: tiled pattern disagrees with decoder colour map")
    cfa = np.array(r.raw_image_visible, dtype=np.uint16, copy=True)
    s = r.sizes
    crop = None
    if s.crop_width and s.crop_height:
        crop = (int(s.crop_top_margin), int(s.crop_left_margin), int(s.crop_height), int(s.crop_width))
    meta = dict(
        source=str(path),
        color_desc=r.color_desc.decode(),
        camera_whitebalance=[float(x) for x in r.camera_whitebalance],
        rgb_xyz_matrix=np.asarray(r.rgb_xyz_matrix)[:3].tolist(),
        flip=int(s.flip),
        margins=(int(s.top_margin), int(s.left_margin)),
    )
    return RawFrame(cfa=cfa, pattern=pattern, black=black, white=float(r.white_level), crop=crop, meta=meta)


def _postprocess_linear(r, algorithm: rawpy.DemosaicAlgorithm | None = None) -> np.ndarray:
    img = r.postprocess(
        demosaic_algorithm=algorithm,
        output_color=rawpy.ColorSpace.raw,
        gamma=(1, 1),
        no_auto_bright=True,
        output_bps=16,
        use_camera_wb=False,
        use_auto_wb=False,
        user_wb=[1.0, 1.0, 1.0, 1.0],
        user_flip=0,
        # LibRaw by default raises its notion of "maximum" to the data max when the data
        # exceed 75 % of it, which would make scaling image-dependent. Disable it.
        adjust_maximum_thr=0.0,
        highlight_mode=rawpy.HighlightMode.Clip,
    )
    out = img.astype(np.float32)
    out /= np.float32(65535.0)
    return out


def load_raw(path: str | Path) -> RawFrame:
    with _open(path) as r:
        return _extract_frame(r, path)


def demosaic_linear(path: str | Path, algorithm: rawpy.DemosaicAlgorithm | None = None) -> np.ndarray:
    """Render path: full-resolution demosaiced linear camera RGB, float32 (H, W, 3).

    Scaled to match ``linearize``: 0 = black level, 1 = white level. No white balance, no colour
    matrix, no orientation flip, no auto-brightness. Values below black are clipped to 0 by LibRaw.
    """
    with _open(path) as r:
        return _postprocess_linear(r, algorithm)


@contextmanager
def open_raw_session(path: str | Path):
    """Open ``path`` once, yielding ``(RawFrame, demosaic_fn)`` so callers validate metadata/regions
    and estimate the CFA base before demosaicing without unpacking the RAW file twice."""
    with _open(path) as r:
        frame = _extract_frame(r, path)
        yield frame, lambda algorithm=None: _postprocess_linear(r, algorithm)
