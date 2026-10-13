"""Abnormal inputs (S1-8): every case must raise a clear error or be flagged, never silently pass."""
import numpy as np
import pytest

from filmcolor.base import estimate_base_auto
from filmcolor.invert import InversionParams, invert_negative, region_mask
from filmcolor.qc import quality_checks
from filmcolor.raw import linearize, saturation_mask
from filmcolor.rawpy_io import RawDecodeError, demosaic_linear, load_raw
from synth import PATTERNS, black_for, film_scene, mosaic

BASE = np.array([0.6, 0.5, 0.15], np.float32)
WHITE = 16383.0


def prepared(kind, rgb):
    p = PATTERNS[kind]
    f = mosaic(rgb, p, black_for(p), WHITE, noise_dn=3.0)
    return linearize(f), f.colors, saturation_mask(f), p.shape[0]


def negative(h=60, w=80):
    """Smooth synthetic negative: base attenuated by 0..1.2 D."""
    d = np.linspace(0, 1.2, w, dtype=np.float32)[None, :, None] * np.ones((h, 1, 3), np.float32)
    return BASE * 10 ** (-d)


# --- decoder ---------------------------------------------------------------------------------

def test_missing_file_is_reported_as_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.RAF")
    with pytest.raises(FileNotFoundError):
        demosaic_linear(tmp_path / "nope.RAF")


def test_non_raw_file_raises_decode_error(tmp_path):
    p = tmp_path / "text.RAF"
    p.write_text("not a raw file")
    with pytest.raises(RawDecodeError, match="text.RAF"):
        load_raw(p)
    with pytest.raises(RawDecodeError):
        demosaic_linear(p)


def test_directory_is_not_a_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path)


# --- base estimation -------------------------------------------------------------------------

@pytest.mark.parametrize("kind", PATTERNS)
def test_auto_base_fails_on_fully_saturated_frame(kind):
    lin, col, sat, n = prepared(kind, np.full((240, 240, 3), 2.0, np.float32))
    with pytest.raises(ValueError, match="manually"):
        estimate_base_auto(lin, col, sat, n)


@pytest.mark.parametrize("kind", PATTERNS)
def test_auto_base_fails_on_black_frame(kind):
    lin, col, sat, n = prepared(kind, np.zeros((240, 240, 3), np.float32))
    with pytest.raises(ValueError):
        estimate_base_auto(lin, col, sat, n)


def test_auto_base_fails_on_image_smaller_than_a_block():
    lin, col, sat, n = prepared("xtrans", np.full((12, 12, 3), 0.5, np.float32))
    with pytest.raises(ValueError):
        estimate_base_auto(lin, col, sat, n)


def test_frame_without_border_and_without_flat_areas_asks_for_manual_base():
    rgb = film_scene(base=tuple(BASE))[:, 1440 // 6 :]  # drop the base strip
    lin, col, sat, n = prepared("bayer", rgb)
    with pytest.raises(ValueError, match="manually"):
        estimate_base_auto(lin, col, sat, n)


def test_frame_without_border_silently_takes_least_dense_flat_area():
    """Known limitation: with no film edge, a flat deep shadow is taken as base (denser than the
    true base). The estimator cannot tell; pipeline QC checks where the base blocks lie."""
    rgb = film_scene(base=tuple(BASE))[:, 1440 // 6 :]
    rgb[300:600, 300:600] = BASE * 10 ** -0.1  # flat deep shadow, 0.1 D above base
    lin, col, sat, n = prepared("bayer", rgb)
    est = estimate_base_auto(lin, col, sat, n)
    np.testing.assert_allclose(est.rgb, BASE * 10 ** -0.1, rtol=0.01)
    assert est.warnings == []


# --- inversion -------------------------------------------------------------------------------

def test_inversion_rejects_non_finite_input():
    rgb = negative()
    rgb[3, 4, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        invert_negative(rgb, BASE)
    rgb[3, 4, 1] = np.inf
    with pytest.raises(ValueError):
        invert_negative(rgb, BASE)


def test_inversion_params_reject_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        InversionParams(base=tuple(BASE), d_hi=(1.0, float("nan"), 1.0))
    with pytest.raises(ValueError):
        InversionParams(base=(float("inf"), 0.5, 0.1))


def test_blank_frame_gives_a_clear_error():
    blank = np.broadcast_to(BASE, (60, 80, 3)).copy()
    with pytest.raises(ValueError, match="blank"):
        invert_negative(blank, BASE)


def test_bad_white_percentile_and_region():
    with pytest.raises(ValueError):
        invert_negative(negative(), BASE, white_pct=0)
    with pytest.raises(ValueError):
        invert_negative(negative(), BASE, white_pct=101)
    with pytest.raises(ValueError):
        region_mask((60, 80), (50, 70, 20, 20))
    with pytest.raises(ValueError):
        invert_negative(negative(), (0.6, 0.0, 0.15))  # non-positive base


# --- quality checks --------------------------------------------------------------------------

def report(clip=0.001, crush=0.0):
    return {"range": {c: {"clip_high": clip, "below_black": crush} for c in "RGB"},
            "full_image_above_white": [0.0, 0.0, 0.0]}


def ramp(lo=0.0, hi=1.0, gain=(1.0, 1.0, 1.0)):
    y = np.linspace(lo, hi, 200, dtype=np.float32)
    return np.clip(np.stack([y * g for g in gain], -1)[None].repeat(10, 0), 0, 1)


def by_name(qc):
    return {c["name"]: c["status"] for c in qc["checks"]}


def test_qc_passes_a_normal_image():
    res = invert_negative(negative(), BASE)
    qc = quality_checks(res.display, None, res.report)
    assert qc["status"] == "pass", qc


@pytest.mark.parametrize("img, rep, failing", [
    (ramp(0, 0.2), report(), "median_luma"),                       # far too dark
    (ramp(0.8, 1.0), report(), "median_luma"),                     # far too bright
    (ramp(0.4, 0.6), report(), "tonal_range"),                     # flat
    (ramp(gain=(1.0, 0.5, 0.4)), report(), "channel_mean_ratio"),  # strong cast
    (ramp(), report(clip=0.03), "clip_high"),
    (ramp(), report(crush=0.2), "below_black"),
])
def test_qc_flags_problems(img, rep, failing):
    qc = quality_checks(img, None, rep)
    assert qc["status"] == "warn"
    assert by_name(qc)[failing] == "warn"


def test_qc_fails_on_nan_and_out_of_range():
    img = ramp()
    img[0, 0, 0] = np.nan
    assert quality_checks(img, None, report())["status"] == "fail"
    img = ramp()
    img[0, 0, 0] = 1.5
    assert by_name(quality_checks(img, None, report()))["value_range"] == "fail"


def test_qc_uses_only_masked_pixels():
    img = np.zeros((10, 600, 3), np.float32)  # mostly black surround, e.g. rebate and holder
    img[:, 400:] = ramp()
    mask = np.zeros(img.shape[:2], bool)
    mask[:, 400:] = True
    assert by_name(quality_checks(img, None, report()))["median_luma"] == "warn"
    assert by_name(quality_checks(img, mask, report()))["median_luma"] == "pass"


def test_qc_flags_base_found_inside_image_region():
    img = ramp()
    assert by_name(quality_checks(img, None, report(), base_inside_image=0.0))["base_inside_image"] == "pass"
    qc = quality_checks(img, None, report(), base_inside_image=0.9)
    assert qc["status"] == "warn" and by_name(qc)["base_inside_image"] == "warn"
    assert "base_inside_image" not in by_name(quality_checks(img, None, report()))
