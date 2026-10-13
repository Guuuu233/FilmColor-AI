import numpy as np
import pytest

from filmcolor.raw import RawFrame, channel_stats, linearize, saturation_mask, superpixel, tile
from synth import PATTERNS, black_for, mosaic

WHITE = 16383.0


@pytest.mark.parametrize("kind", PATTERNS)
def test_tile_handles_non_multiple_shapes(kind):
    p = PATTERNS[kind]
    t = tile(p, (13, 17))
    assert t.shape == (13, 17)
    n = p.shape[0]
    assert np.array_equal(t[:n, :n], p)
    assert np.array_equal(t[n : 2 * n, n : 2 * n], p)


@pytest.mark.parametrize("kind", PATTERNS)
def test_linearize_uses_per_position_black(kind):
    p = PATTERNS[kind]
    black = black_for(p)
    rgb = np.empty((12, 12, 3), np.float32)
    rgb[:] = (0.1, 0.5, 0.9)
    f = mosaic(rgb, p, black, WHITE)
    lin = linearize(f)
    col = f.colors
    for c, v in enumerate((0.1, 0.5, 0.9)):
        # rounding to integer DN bounds the error by 0.5 / (white - black)
        assert np.allclose(lin[col == c], v, atol=0.6 / (WHITE - black.max()))


def test_linearize_black_level_maps_to_zero_and_keeps_negative_noise():
    p = PATTERNS["bayer"]
    black = black_for(p, jitter=False)
    cfa = np.full((4, 4), 512, np.uint16)
    cfa[0, 0] = 500  # below black: read noise
    f = RawFrame(cfa=cfa, pattern=p, black=black, white=WHITE)
    lin = linearize(f)
    assert lin[1, 1] == 0
    assert lin[0, 0] < 0


def test_saturation_mask():
    p = PATTERNS["bayer"]
    cfa = np.array([[100, 16383], [16382, 16383]], np.uint16)
    f = RawFrame(cfa=cfa, pattern=p, black=black_for(p, jitter=False), white=WHITE)
    assert saturation_mask(f).tolist() == [[False, True], [False, True]]
    assert saturation_mask(f, threshold=16000).sum() == 3


@pytest.mark.parametrize("kind", PATTERNS)
def test_channel_stats_recovers_values_and_excludes_saturated(kind):
    p = PATTERNS[kind]
    rgb = np.empty((24, 24, 3), np.float32)
    rgb[:] = (0.2, 0.4, 0.05)
    rgb[:6, :6] = (1.5, 1.5, 1.5)  # over white -> clipped and saturated
    f = mosaic(rgb, p, black_for(p), WHITE)
    lin, col, sat = linearize(f), f.colors, saturation_mask(f)
    s = channel_stats(lin, col, sat)
    for name, v in zip("RGB", (0.2, 0.4, 0.05)):
        assert s[name]["median"] == pytest.approx(v, abs=1e-4)
        assert s[name]["sat_frac"] > 0
    region = channel_stats(lin, col, sat, (6, 6, 12, 12))
    assert all(region[c]["sat_frac"] == 0 for c in "RGB")
    with pytest.raises(ValueError):
        channel_stats(lin, col, sat, (20, 20, 12, 12))


@pytest.mark.parametrize("kind,block", [("bayer", 2), ("bayer", 8), ("xtrans", 6), ("xtrans", 24)])
def test_superpixel_never_mixes_channels(kind, block):
    p = PATTERNS[kind]
    rgb = np.empty((48, 72, 3), np.float32)
    rgb[:] = (0.1, 0.5, 0.9)  # very different per channel: any mixing would show
    f = mosaic(rgb, p, black_for(p), WHITE)
    sp, ss = superpixel(linearize(f), f.colors, saturation_mask(f), p.shape[0], block)
    assert sp.shape == (48 // block, 72 // block, 3)
    assert np.allclose(sp, (0.1, 0.5, 0.9), atol=1e-4)
    assert not ss.any()


def test_superpixel_rejects_block_not_multiple_of_pattern():
    p = PATTERNS["xtrans"]
    f = mosaic(np.full((24, 24, 3), 0.3, np.float32), p, black_for(p), WHITE)
    with pytest.raises(ValueError):
        superpixel(linearize(f), f.colors, saturation_mask(f), 6, 8)


def test_superpixel_flags_saturated_blocks():
    p = PATTERNS["xtrans"]
    rgb = np.full((24, 24, 3), 0.3, np.float32)
    rgb[0, 0] = 2.0
    f = mosaic(rgb, p, black_for(p), WHITE)
    _, ss = superpixel(linearize(f), f.colors, saturation_mask(f), 6, 12)
    assert ss.tolist() == [[True, False], [False, False]]


@pytest.mark.parametrize("bad", [
    dict(pattern=np.array([[0, 3], [1, 2]], np.uint8)),
    dict(black=np.full((2, 2), 20000.0)),
    dict(black=np.zeros((6, 6))),
])
def test_rawframe_validation(bad):
    args = dict(cfa=np.zeros((4, 4), np.uint16), pattern=PATTERNS["bayer"], black=np.zeros((2, 2)), white=WHITE)
    args.update(bad)
    with pytest.raises(ValueError):
        RawFrame(**args)
