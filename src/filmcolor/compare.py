"""Compare two renderings of the same negative (e.g. ours vs NLP): load with their ICC profiles into
CIELAB, align (90-degree rotation, small scale, translation) and measure differences. Pure numpy.

Alignment works on block-averaged luminance, so the reference may be cropped, rotated by a multiple
of 90 degrees and slightly rescaled (e.g. by a raw converter's lens correction).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import tifffile

from .colorimetry import MatrixTRCProfile, delta_e2000, xyz_to_lab


def _read_rgb_icc(path):
    """(image array, embedded ICC bytes) from a TIFF, or from a JPEG (reference images only; Pillow is
    imported here so that the engine does not depend on it)."""
    if str(path).lower().endswith((".jpg", ".jpeg")):
        from PIL import Image

        with Image.open(path) as im:
            icc = im.info.get("icc_profile")
            img = np.asarray(im.convert("RGB")) if icc else None
    else:
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            tag = page.tags.get("InterColorProfile")
            icc = bytes(tag.value) if tag is not None else None
            img = page.asarray() if icc else None
    if icc is None:
        raise ValueError(f"{path}: no embedded ICC profile")
    return img, icc


def load_lab(path, factor: int = 1) -> np.ndarray:
    """CIELAB (D50) from an RGB TIFF or JPEG with an embedded matrix/TRC ICC profile, block-averaged by
    ``factor`` in linear XYZ (physically meaningful averaging) before the Lab transform."""
    img, icc = _read_rgb_icc(path)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"{path}: expected an RGB image, got shape {img.shape}")
    scale = float(np.iinfo(img.dtype).max) if img.dtype.kind == "u" else 1.0
    prof = MatrixTRCProfile(icc)
    h = (img.shape[0] // factor) * factor
    step = factor * max(1, 256 // factor)  # rows per chunk, a multiple of factor
    chunks = [block_mean(prof.to_xyz(img[y : y + step, :, :3].astype(np.float32) / scale), factor)
              for y in range(0, h, step)]
    return xyz_to_lab(np.concatenate(chunks))


def _box_mean(a, size: int):
    """Mean of the size x size box around each pixel (edges padded by replication)."""
    r = size // 2
    p = np.pad(np.asarray(a, np.float64), r, mode="edge")
    c = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return (c[size:, size:] - c[:-size, size:] - c[size:, :-size] + c[:-size, :-size]) / size ** 2


def _local_detail(L, size: int):
    """|L - mean of the size x size box around each pixel|."""
    return np.abs(L - _box_mean(L, size))


def highlight_texture(ours_lab, ref_lab, ref_clipped=None, l_min: float = 80.0, size: int = 5,
                      min_pixels: int = 50) -> dict:
    """Local tonal detail in the reference's highlights: mean |L* - local mean| of ours over that of the
    reference, on pixels with reference L* >= ``l_min`` and (if given, (H, W, 3) bool) not clipped in the
    reference. ~1 = detail kept; towards 0 = flattened, whether by clipping or by compression below white.
    ``ratio`` is None when fewer than ``min_pixels`` pixels qualify."""
    mask = np.asarray(ref_lab)[..., 0] >= l_min
    if ref_clipped is not None:
        mask &= ~np.asarray(ref_clipped).any(-1)
    # keep only pixels whose whole box lies in the mask, so edges against darker areas (sky / mountain) do
    # not dominate the measure
    mask = _box_mean(mask, size) > 1 - 1e-9
    r = size // 2
    mask[:r], mask[-r:], mask[:, :r], mask[:, -r:] = False, False, False, False
    n = int(mask.sum())
    if n < min_pixels:
        return dict(ratio=None, n=n, share=float(mask.mean()))
    do, dr = _local_detail(ours_lab[..., 0], size)[mask].mean(), _local_detail(ref_lab[..., 0], size)[mask].mean()
    return dict(ratio=float(do / dr) if dr > 0 else None, ours=float(do), ref=float(dr), n=n, share=float(mask.mean()))


def block_mean(img, f: int):
    h, w = (img.shape[0] // f) * f, (img.shape[1] // f) * f
    return img[:h, :w].reshape(h // f, f, w // f, f, *img.shape[2:]).mean(axis=(1, 3))


def resize_bilinear(img, shape):
    """Bilinear resample of a 2-D or 3-D array to ``shape`` (pixel-centre aligned)."""
    h, w = img.shape[:2]
    ys = np.clip((np.arange(shape[0]) + 0.5) * h / shape[0] - 0.5, 0, h - 1)
    xs = np.clip((np.arange(shape[1]) + 0.5) * w / shape[1] - 0.5, 0, w - 1)
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    wy, wx = ys - y0, xs - x0
    if img.ndim == 3:
        wy, wx = wy[:, None, None], wx[None, :, None]
    else:
        wy, wx = wy[:, None], wx[None, :]
    top = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
    bot = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def ncc_map(image, template):
    """Normalised cross-correlation of ``template`` at every valid position in ``image`` (FFT)."""
    H, W = image.shape
    h, w = template.shape
    if h > H or w > W:
        raise ValueError("template larger than image")
    t = template - template.mean()
    tn = np.sqrt((t ** 2).sum())
    shape = (H + h, W + w)
    corr = np.fft.irfft2(np.fft.rfft2(image, shape) * np.conj(np.fft.rfft2(t, shape)), shape)[: H - h + 1, : W - w + 1]
    ii = np.pad(image, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ii2 = np.pad(image ** 2, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def win(a):
        return a[h:, w:] - a[:-h, w:] - a[h:, :-w] + a[:-h, :-w]

    n = h * w
    var = win(ii2) - win(ii) ** 2 / n
    return corr / (np.sqrt(np.maximum(var, 1e-12)) * tn + 1e-12)


@dataclass
class Alignment:
    rot90: int          # reference rotated by k * 90 degrees counter-clockwise to match
    scale: float        # reference resized by this factor (at working resolution)
    offset: tuple       # (row, col) of the reference's top-left in the working-resolution image
    ncc: float
    factor: int         # working resolution = full resolution / factor

    def to_dict(self):
        return asdict(self)


def align(img_l, ref_l, factor: int, scales=np.arange(0.97, 1.0301, 0.005)) -> Alignment:
    """Find where the (cropped) reference sits in the image, on L* block-averaged by ``factor``."""
    a = block_mean(img_l, factor)
    best = None
    for k in range(4):
        r = block_mean(np.rot90(ref_l, k), factor)
        for s in scales:
            shape = (int(round(r.shape[0] * s)), int(round(r.shape[1] * s)))
            if shape[0] > a.shape[0] or shape[1] > a.shape[1]:
                continue
            m = ncc_map(a, resize_bilinear(r, shape))
            i = np.unravel_index(np.argmax(m), m.shape)
            if best is None or m[i] > best.ncc:
                best = Alignment(k, float(s), (int(i[0]), int(i[1])), float(m[i]), factor)
    if best is None:
        raise ValueError("reference does not fit inside the image at any tested rotation/scale")
    return best


def aligned_pair(img_lab, ref_lab, al: Alignment):
    """Both images on the working grid, cropped to the reference area."""
    a = block_mean(img_lab, al.factor)
    r = block_mean(np.rot90(ref_lab, al.rot90), al.factor)
    shape = (int(round(r.shape[0] * al.scale)), int(round(r.shape[1] * al.scale)))
    r = resize_bilinear(r, shape)
    y, x = al.offset
    return a[y : y + shape[0], x : x + shape[1]], r


def _crop_margin(lab, margin: float) -> np.ndarray:
    if not 0.0 <= float(margin) < 0.5:
        raise ValueError(f"margin must be in [0, 0.5), got {margin}")
    h, w = lab.shape[:2]
    my, mx = int(h * margin), int(w * margin)
    if h - 2 * my <= 0 or w - 2 * mx <= 0:
        raise ValueError(f"margin {margin} leaves an empty interior on shape {(h, w)}")
    return lab[my : h - my, mx : w - mx].reshape(-1, 3)


def metrics(lab, ref, margin: float = 0.03) -> dict:
    """Differences ``lab`` - ``ref`` on aligned grids, ignoring a border ``margin`` (crop edges)."""
    a, r = _crop_margin(lab, margin), _crop_margin(ref, margin)
    de = delta_e2000(a, r)
    ca, cr = np.hypot(a[:, 1], a[:, 2]), np.hypot(r[:, 1], r[:, 2])
    coloured = cr > 10
    pct = [5, 25, 50, 75, 95]
    return dict(
        n=int(len(a)),
        de2000_median=float(np.median(de)), de2000_p90=float(np.percentile(de, 90)),
        mean_dL=float((a[:, 0] - r[:, 0]).mean()),
        mean_da=float((a[:, 1] - r[:, 1]).mean()), mean_db=float((a[:, 2] - r[:, 2]).mean()),
        chroma_ratio=float(np.median(ca[coloured] / cr[coloured])) if coloured.any() else None,
        L_percentiles=dict(pct=pct, image=np.percentile(a[:, 0], pct).tolist(), reference=np.percentile(r[:, 0], pct).tolist()),
    )


def _core(lab, margin):
    return _crop_margin(lab, margin).astype(np.float64)


def match_tone(L, ref_L):
    """Monotone mapping of L onto the distribution of ref_L (histogram matching)."""
    out = np.empty_like(L)
    out[np.argsort(L, kind="stable")] = np.quantile(ref_L, (np.arange(len(L)) + 0.5) / len(L))
    return out


def tone_curve(L, ref_L, knots=np.linspace(0, 100, 101)):
    """The same mapping as a curve sampled at ``knots`` (for sharing across images)."""
    q = np.linspace(0, 1, 501)
    return np.interp(knots, np.quantile(L, q), np.quantile(ref_L, q))


def fit_ab_affine(ab, ref_ab):
    """Least-squares 2x2 matrix + offset mapping ab -> ref_ab (cast, hue rotation, saturation)."""
    X = np.c_[ab, np.ones(len(ab))]
    coef, *_ = np.linalg.lstsq(X, ref_ab, rcond=None)
    return coef  # (3, 2)


def apply_ab_affine(ab, coef):
    return np.c_[ab, np.ones(len(ab))] @ coef


def decompose(lab, ref, margin: float = 0.03) -> dict:
    """How much of the difference per-image global corrections explain, fitted in-sample to ``ref``
    (an upper bound for global per-image parameters, not a prediction): median dE2000 with no
    correction, after matching the L* distribution, and after also fitting an a*b* affine map."""
    a, r = _core(lab, margin), _core(ref, margin)
    out = dict(none=float(np.median(delta_e2000(a, r))))
    t = a.copy()
    t[:, 0] = match_tone(a[:, 0], r[:, 0])
    out["tone"] = float(np.median(delta_e2000(t, r)))
    t[:, 1:] = apply_ab_affine(t[:, 1:], fit_ab_affine(t[:, 1:], r[:, 1:]))
    out["tone_ab"] = float(np.median(delta_e2000(t, r)))
    return out
