"""Synthetic CFA frames with known ground truth, for Bayer and X-Trans."""
import numpy as np

from filmcolor.raw import RawFrame, tile

BAYER_RGGB = np.array([[0, 1], [1, 2]], np.uint8)
# as reported by LibRaw for a Fujifilm X-T2 (G2 folded into G)
XTRANS = np.array([
    [0, 2, 1, 2, 0, 1],
    [1, 1, 0, 1, 1, 2],
    [1, 1, 2, 1, 1, 0],
    [2, 0, 1, 0, 2, 1],
    [1, 1, 2, 1, 1, 0],
    [1, 1, 0, 1, 1, 2],
], np.uint8)
PATTERNS = {"bayer": BAYER_RGGB, "xtrans": XTRANS}


def black_for(pattern, base=512.0, jitter=True):
    """Per-position black levels; Bayer gets distinct G1/G2 values to exercise per-position handling."""
    b = np.full(pattern.shape, base)
    if jitter and pattern.shape == (2, 2):
        b[0, 1], b[1, 0] = base + 8, base + 4
    return b


def mosaic(rgb, pattern, black, white=16383.0, noise_dn=0.0, seed=0) -> RawFrame:
    """Sample linear RGB (1 = white level) through the CFA into DN, optionally with Gaussian noise."""
    h, w, _ = rgb.shape
    col = tile(pattern, (h, w))
    blk = tile(black, (h, w))
    lin = np.take_along_axis(rgb, col[..., None].astype(np.intp), -1)[..., 0]
    dn = blk + lin * (white - blk)
    if noise_dn:
        dn = dn + np.random.default_rng(seed).normal(0, noise_dn, dn.shape)
    dn = np.clip(np.round(dn), 0, white).astype(np.uint16)
    return RawFrame(cfa=dn, pattern=pattern, black=black, white=white)


def film_scene(h=960, w=1440, base=(0.6, 0.5, 0.15), seed=1):
    """Linear RGB 'scan': film base strip on the left (cols < w/6), image area elsewhere.

    The image area is base attenuated by a smooth density field (0.2..1.5 D per channel) plus a
    textured patch, so it is darker than base everywhere.
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(base, np.float32)
    img = np.empty((h, w, 3), np.float32)
    img[:] = base
    x0 = w // 6
    yy, xx = np.mgrid[0:h, 0 : w - x0].astype(np.float32)
    for c in range(3):
        d = 0.2 + 1.3 * (0.5 + 0.5 * np.sin(xx / (90 + 20 * c) + c) * np.cos(yy / (130 + 15 * c)))
        img[:, x0:, c] = base[c] * 10 ** (-d)
    tex = rng.uniform(0.3, 1.0, (h // 3, (w - x0) // 3, 3)).astype(np.float32)
    img[h // 3 : 2 * h // 3, x0 + (w - x0) // 3 : x0 + 2 * (w - x0) // 3] *= tex
    return img
