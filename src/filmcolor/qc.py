"""Basic automatic quality checks on an inverted image. Pure numpy.

The thresholds are provisional (Sprint 1). They flag images for a human look - grossly dark or
flat output, heavy clipping, strong casts - and say nothing about colour accuracy.
"""
from __future__ import annotations

import numpy as np

from .raw import CHANNELS

THRESHOLDS = dict(
    max_clip_high=0.005,          # share of image-region pixels above output white, per channel
    max_below_black=0.10,         # share crushed below the black point, per channel
    median_luma=(0.15, 0.85),     # sRGB-encoded luma median of the image region
    min_tonal_range=0.35,         # p99 - p1 of encoded luma
    channel_mean_ratio=1.5,       # max / min of encoded channel means
    max_base_inside_image=0.5,    # share of auto base blocks inside the image region
)
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)  # applied to encoded values: a luma proxy


def _check(name, value, ok, threshold, level="warn"):
    return dict(name=name, value=value, threshold=threshold, status="pass" if ok else level)


def quality_checks(display, mask, report, base_inside_image: float | None = None) -> dict:
    """``display``: (H, W, 3) sRGB-encoded output; ``mask``: image-region pixels; ``report``: from
    ``invert_negative``; ``base_inside_image``: share of automatically chosen base blocks lying
    inside the image region (None to skip). Film base normally sits on the rebate outside the
    image; a base found inside it suggests a scan without film edge, where the estimate is a
    scene shadow instead (denser than base, often coloured). Not all such cases are caught.
    Returns overall status (pass / warn / fail) and the individual checks."""
    t = THRESHOLDS
    px = display[mask] if mask is not None else display.reshape(-1, 3)
    checks = []
    finite = bool(np.all(np.isfinite(px)))
    checks.append(_check("finite", finite, finite, True, "fail"))
    if finite:
        lo, hi = float(px.min()), float(px.max())
        checks.append(_check("value_range", [lo, hi], lo >= 0 and hi <= 1, [0, 1], "fail"))
        y = px @ LUMA
        med = float(np.median(y))
        checks.append(_check("median_luma", med, t["median_luma"][0] <= med <= t["median_luma"][1], t["median_luma"]))
        p1, p99 = np.percentile(y, [1, 99])
        checks.append(_check("tonal_range", float(p99 - p1), p99 - p1 >= t["min_tonal_range"], t["min_tonal_range"]))
        means = px.mean(0)
        ratio = float(means.max() / max(means.min(), 1e-6))
        checks.append(_check("channel_mean_ratio", ratio, ratio <= t["channel_mean_ratio"], t["channel_mean_ratio"]))
    r = report["range"]
    clip = max(r[c]["clip_high"] for c in CHANNELS)
    checks.append(_check("clip_high", clip, clip <= t["max_clip_high"], t["max_clip_high"]))
    crush = max(r[c]["below_black"] for c in CHANNELS)
    checks.append(_check("below_black", crush, crush <= t["max_below_black"], t["max_below_black"]))
    if base_inside_image is not None:
        checks.append(_check("base_inside_image", base_inside_image,
                             base_inside_image <= t["max_base_inside_image"], t["max_base_inside_image"]))
    statuses = {c["status"] for c in checks}
    status = "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass"
    return dict(status=status, checks=checks, provisional=True,
                info=dict(full_image_above_white=report.get("full_image_above_white")))
