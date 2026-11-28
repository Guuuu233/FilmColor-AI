"""Frames with no image density (blank film, light source only) must fail with a clear message, not a crash."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from filmcolor import plugin_bridge
from filmcolor.invert import _t18_chunk_from_s, check_white_point

RAW = Path(__file__).resolve().parents[1] / "data/raw/public/nlp_official/_DSF8434.RAF"


def test_check_white_point_names_flat_channels() -> None:
    check_white_point((0.29, 0.51, 0.39))
    check_white_point((0.29, 0.51, 0.39), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match=r"no density above the black point in channel\(s\) G "):
        check_white_point((0.0519, 0.0, 0.0448))
    with pytest.raises(ValueError, match=r"channel\(s\) RGB "):
        check_white_point((0.1, 0.1, 0.1), (0.1, 0.2, 0.3))


def test_t18_chunk_from_s_rejects_zero_log_range() -> None:
    s = np.full((2, 2, 3), 0.5, np.float32)
    assert np.isfinite(_t18_chunk_from_s(s, 0.5, 10.0 ** -0.5)).all()
    for r in (0.0, -0.1, float("nan")):
        with pytest.raises(ValueError, match="positive log range"):
            _t18_chunk_from_s(s, r, 10.0 ** -r)


def test_fast_camera_preview_blank_frame_raises_clear_error(monkeypatch) -> None:
    """A capture of the light source alone puts the automatic white point on the film base (G density 0);
    the plugin fast path used to raise ZeroDivisionError there."""
    if not RAW.exists():
        pytest.skip(f"{RAW.name} not present")
    orig = plugin_bridge.block_white_point

    def white_on_base(*args, **kwargs):
        d_hi, info = orig(*args, **kwargs)
        return (float(d_hi[0]), 0.0, float(d_hi[2])), info

    monkeypatch.setattr(plugin_bridge, "block_white_point", white_on_base)
    with pytest.raises(ValueError, match=r"no density above the black point in channel\(s\) G "):
        plugin_bridge._fast_load_camera_preview(RAW, max_side=256, film="color", engine="t25", preset="standard",
                                                base_mode="auto", extra={}, lr_crop=None)
