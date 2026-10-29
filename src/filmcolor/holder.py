"""Opaque film holder / mask around the frame (T4i), shared by the preview display and, since T4l, the estimator's
metering frame (``noref``)."""
from __future__ import annotations

import numpy as np

# T4i: an opaque film holder / mask around the frame is far denser than the image, so its positive is many stops past
# white with unequal channels (yellow or magenta after the white clip). Shown as one dark neutral instead (display
# only: the export and the 1:1 crop are unchanged; T4l: the estimator meters inside the film box, noref._metering_mask).
HOLDER_OVER = 2.0    # positive above this in two channels (one stop past the white point) ...
HOLDER_PEAK = 16.0   # ... and above this in the brightest one (four stops; _DSF8948's holder: R ~300, G ~36, B ~3.6)
HOLDER_GREY = 0.02   # sRGB-encoded value the holder is displayed with


def holder_mask(P, over: float = HOLDER_OVER, peak: float = HOLDER_PEAK) -> np.ndarray:
    """Pixels of positive ``P`` (h, w, 3) above ``over`` in two channels and ``peak`` in one that connect to the image
    edge (a highlight inside the frame does not)."""
    from scipy import ndimage

    s = np.sort(np.asarray(P), axis=-1)
    hot = (s[..., 1] > over) & (s[..., 2] > peak)
    if not hot.any():
        return hot
    lab, _ = ndimage.label(hot)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return np.isin(lab, edge[edge > 0])


def film_box(holder: np.ndarray) -> list[int] | None:
    """(top, left, height, width) of the pixels outside ``holder``, or None when the holder covers nothing."""
    if not holder.any():
        return None
    keep = ~holder
    rows, cols = np.where(keep.any(1))[0], np.where(keep.any(0))[0]
    return [int(rows[0]), int(cols[0]), int(rows[-1] - rows[0] + 1), int(cols[-1] - cols[0] + 1)]
