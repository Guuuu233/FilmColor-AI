"""Row-block processing (S3-2 memory) must give exactly the whole-image result."""
import numpy as np
import pytest

from filmcolor import invert as inv
from filmcolor.render import RenderParams, render, render_full

ALL = RenderParams(ev=0.3, temp=-0.2, tint=0.15, sat=1.3, contrast=1.4, midtone=0.25, shoulder=0.4,
                   matrix=((1.1, -0.05, -0.05), (-0.05, 1.1, -0.05), (0.0, -0.1, 1.1)))


def positive(h=41, w=23, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w, 3)) * 1.6 - 0.1).astype(np.float32)  # includes < 0 and > 1


@pytest.mark.parametrize("p", [RenderParams(), ALL])
@pytest.mark.parametrize("rows", [1, 3, 40])
@pytest.mark.parametrize("with_mask", [False, True])
def test_render_blocks_equal_whole_image(p, rows, with_mask):
    P = positive()
    mask = None
    if with_mask:
        mask = np.zeros(P.shape[:2], bool)
        mask[5:30, 2:20] = True
    e0, d0, _ = render_full(P, p, mask)
    e1, d1 = render(P, p, mask, chunk_rows=rows)
    assert e1.dtype == np.float32 and np.array_equal(e0, e1)
    assert d0 == d1


def test_render_blocks_reject_non_finite():
    P = positive()
    P[37, 4, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        render(P, RenderParams(), chunk_rows=4)


def whole_image_inversion(rgb, base, mask, **kw):
    """The pre-S3-2 definition: density of the full image, then mask."""
    D = inv.density(rgb, base)
    d_hi = kw.get("d_hi") or inv.estimate_white_point(D, mask, kw.get("white_pct", 99.9))
    p = inv.InversionParams(base=tuple(map(float, base)), d_lo=(0.0, 0.0, 0.0), d_hi=tuple(map(float, d_hi)),
                            gamma=0.6, exposure=1.0)
    P = inv.invert(D, p)
    above = [float(np.mean(D[..., c] > p.d_hi[c])) for c in range(3)]
    return P, p, inv.range_report(D, P, p, mask), above


@pytest.mark.parametrize("rows", [1, 4, 100])
@pytest.mark.parametrize("with_mask", [False, True])
def test_inversion_blocks_equal_whole_image(monkeypatch, rows, with_mask):
    monkeypatch.setattr(inv, "CHUNK_ROWS", rows)
    base = (0.6, 0.35, 0.2)
    rng = np.random.default_rng(2)
    rgb = (np.asarray(base, np.float32) * 10 ** (-rng.random((37, 29, 3)) * 2)).astype(np.float32)
    mask = None
    if with_mask:
        mask = np.zeros(rgb.shape[:2], bool)
        mask[3:30, 5:25] = True
    P0, p0, rep0, above0 = whole_image_inversion(rgb, base, mask)
    res = inv.invert_negative(rgb, base, mask=mask)
    assert res.params == p0 and np.array_equal(res.positive, P0)
    assert res.report["range"] == rep0 and res.report["full_image_above_white"] == above0
    assert np.array_equal(res.display, inv.tone_map(P0, p0))
