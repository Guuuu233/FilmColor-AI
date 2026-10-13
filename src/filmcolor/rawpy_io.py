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


def load_raw(path: str | Path) -> RawFrame:
    with _open(path) as r:
        if r.raw_type != rawpy.RawType.Flat or r.raw_pattern is None:
            raise ValueError(f"{path}: only single-plane CFA RAW files are supported")
        lib_pattern = np.asarray(r.raw_pattern)
        black4 = np.asarray(r.black_level_per_channel, dtype=np.float64)
        black = black4[lib_pattern]  # per CFA position, before folding G2 into G
        pattern = np.where(lib_pattern == 3, 1, lib_pattern).astype(np.uint8)
        cfa = np.array(r.raw_image_visible, dtype=np.uint16, copy=True)
        lib_colors = np.where(r.raw_colors_visible == 3, 1, r.raw_colors_visible)
        n = pattern.shape[0]
        if not np.array_equal(lib_colors[:n, :n], pattern):
            raise ValueError(f"{path}: raw_pattern does not tile raw_colors_visible from origin")
        s = r.sizes
        # rawpy takes crop_* from LibRaw raw_inset_crops[0]; LibRaw does not document whether they
        # are relative to the visible area or the full raw image. Equivalent when both margins are 0
        # (verified for the X-T2 samples); unverified for cameras with non-zero margins.
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
        frame = RawFrame(cfa=cfa, pattern=pattern, black=black, white=float(r.white_level), crop=crop, meta=meta)
    if not np.array_equal(frame.colors, lib_colors):
        raise ValueError(f"{path}: tiled pattern disagrees with decoder colour map")
    return frame


def demosaic_linear(path: str | Path, algorithm: rawpy.DemosaicAlgorithm | None = None) -> np.ndarray:
    """Render path: full-resolution demosaiced linear camera RGB, float32 (H, W, 3).

    Scaled to match ``linearize``: 0 = black level, 1 = white level. No white balance, no colour
    matrix, no orientation flip, no auto-brightness. Values below black are clipped to 0 by LibRaw.
    """
    with _open(path) as r:
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
    return img.astype(np.float32) / 65535.0
