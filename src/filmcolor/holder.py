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
# T5a: a clipped scene highlight at the frame edge (B096's windows, dense past the scan range) passes holder_mask too;
# a holder band covers its whole box edge, the window only part of it but reaches deep into the frame (display only)
SCENE_HOLDER_MIN_DEPTH = 0.08      # share of the film box's short side
SCENE_HOLDER_MAX_EDGE_SPAN = 0.60  # share of each film box edge it touches
# T5c: a dense rebate strip inside film_box detached from the outer holder by a 1-3 px seam (A1298 after T5b):
# starts within REBATE_STRIP_MAX_GAP * short side of a box edge, stays within SCENE_HOLDER_MIN_DEPTH * short side of
# that edge, and spans >= REBATE_STRIP_MIN_SPAN of that edge (display only)
REBATE_STRIP_MAX_GAP = 0.01
REBATE_STRIP_MIN_SPAN = 0.35


def holder_mask(P, over: float = HOLDER_OVER, peak: float = HOLDER_PEAK) -> np.ndarray:
    """Pixels of positive ``P`` (h, w, 3) above ``over`` in two channels and ``peak`` in one that connect to the image
    edge (a highlight inside the frame does not). T4m: default dual threshold ((s1>2)&(s2>16))|((s1>1.25)&(s2>8))."""
    from scipy import ndimage

    s = np.sort(np.asarray(P), axis=-1)
    s1, s2 = s[..., 1], s[..., 2]
    if over == HOLDER_OVER and peak == HOLDER_PEAK:
        hot = ((s1 > 2.0) & (s2 > 16.0)) | ((s1 > 1.25) & (s2 > 8.0))
    else:
        hot = (s1 > over) & (s2 > peak)
    if not hot.any():
        return hot
    lab, _ = ndimage.label(hot)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return np.isin(lab, edge[edge > 0])


def film_box(holder: np.ndarray) -> list[int] | None:
    """(top, left, height, width) of the pixels outside ``holder``, or None when the holder covers nothing
    or no valid film area remains (T4m: row/col share > 0.60 when holder.mean >= 0.15)."""
    if not holder.any():
        return None
    keep = ~holder
    if not keep.any():
        return None
    if float(holder.mean()) >= 0.15:
        rows = np.where(keep.mean(axis=1) > 0.60)[0]
        cols = np.where(keep.mean(axis=0) > 0.60)[0]
        if not len(rows) or not len(cols):
            rows, cols = np.where(keep.any(1))[0], np.where(keep.any(0))[0]
    else:
        rows, cols = np.where(keep.any(1))[0], np.where(keep.any(0))[0]
    if not len(rows) or not len(cols):
        return None
    return [int(rows[0]), int(cols[0]), int(rows[-1] - rows[0] + 1), int(cols[-1] - cols[0] + 1)]


def scene_highlight_components(holder: np.ndarray, min_depth: float = SCENE_HOLDER_MIN_DEPTH,
                               max_edge_span: float = SCENE_HOLDER_MAX_EDGE_SPAN) -> np.ndarray:
    """T5a: holder components inside film_box(holder) that reach >= min_depth * short side into the box and cover
    < max_edge_span of every box edge they touch: a clipped scene highlight touching the frame edge (B096 windows),
    not a holder band. Only when holder.mean() >= 0.15 (film_box band mode); otherwise all False."""
    from scipy import ndimage

    keep = np.zeros_like(holder)
    if float(holder.mean()) < 0.15:
        return keep
    fb = film_box(holder)
    if fb is None:
        return keep
    t, l, h, w = fb
    sub = holder[t:t + h, l:l + w]
    lab, n = ndimage.label(sub)
    if not n:
        return keep
    dist = ndimage.distance_transform_edt(np.pad(np.ones((h, w), bool), 1, constant_values=False))[1:-1, 1:-1]
    depth = ndimage.maximum(dist, lab, np.arange(1, n + 1))
    for i in range(n):
        if depth[i] < min_depth * min(h, w):
            continue
        m = lab == i + 1
        rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
        spans = []
        if m[0].any() or m[-1].any():
            spans.append((cols[-1] - cols[0] + 1) / w)
        if m[:, 0].any() or m[:, -1].any():
            spans.append((rows[-1] - rows[0] + 1) / h)
        if spans and max(spans) >= max_edge_span:
            continue
        keep[t:t + h, l:l + w] |= m
    return keep


def rebate_strip_components(P, holder: np.ndarray, max_gap: float = REBATE_STRIP_MAX_GAP,
                            max_depth: float = SCENE_HOLDER_MIN_DEPTH,
                            min_span: float = REBATE_STRIP_MIN_SPAN) -> np.ndarray:
    """T5c: hot components inside film_box(holder) that run along a box edge (within max_gap * short side of the edge,
    at most max_depth * short side deep, spanning >= min_span of that edge): a dense rebate strip separated from the
    outer holder by a narrow seam (A1298), painted HOLDER_GREY in display. Only when holder.mean() >= 0.15."""
    from scipy import ndimage

    keep = np.zeros_like(holder)
    if float(holder.mean()) < 0.15:
        return keep
    fb = film_box(holder)
    if fb is None:
        return keep
    t, l, h, w = fb
    short = min(h, w)
    gap, dep = max_gap * short, max_depth * short
    s = np.sort(np.asarray(P), axis=-1)
    s1, s2 = s[..., 1], s[..., 2]
    hot = (((s1 > 2.0) & (s2 > 16.0)) | ((s1 > 1.25) & (s2 > 8.0))) & ~holder
    sub = hot[t:t + h, l:l + w]
    if not sub.any():
        return keep
    lab, n = ndimage.label(sub)
    for i in range(n):
        m = lab == i + 1
        rs, cs = np.where(m)
        r0, r1, c0, c1 = int(rs.min()), int(rs.max()), int(cs.min()), int(cs.max())
        if ((c0 <= gap and c1 + 1 <= dep and (r1 - r0 + 1) >= min_span * h)
                or (w - 1 - c1 <= gap and w - c0 <= dep and (r1 - r0 + 1) >= min_span * h)
                or (r0 <= gap and r1 + 1 <= dep and (c1 - c0 + 1) >= min_span * w)
                or (h - 1 - r1 <= gap and h - r0 <= dep and (c1 - c0 + 1) >= min_span * w)):
            keep[t:t + h, l:l + w] |= m
    return keep
