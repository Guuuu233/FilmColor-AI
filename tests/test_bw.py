"""Explicit black-and-white negative mode (S3-4)."""
import numpy as np
import pytest
import tifffile

from filmcolor.invert import EPS, neutral_transmission
from filmcolor.noref import NOREF_V2
from filmcolor.pipeline import convert, replay_kwargs

BASE = np.array([0.15, 0.48, 0.46])  # bluish B&W base seen through a camera without white balance


def bw_scan(h=240, w=360):
    """Base strip on the left, neutral silver densities elsewhere (same density in every channel)."""
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[:h, :w]
    dens = np.where(xx < 96, 0.0, 0.1 + 1.4 * ((yy // 30 + xx // 40) % 8) / 7)
    t = BASE * 10 ** (-dens[..., None]) * (1 + 0.002 * rng.standard_normal((h, w, 3)))
    return np.clip(np.round(t * 65535), 0, 65535).astype(np.uint16)


def write_dng(path, rgb):
    tifffile.imwrite(path, rgb, photometric=34892, planarconfig="contig", metadata=None)


def test_neutral_transmission_is_geometric_mean_of_normalised_channels():
    rgb = np.array([[[0.06, 0.24, 0.115], [0.15, 0.48, 0.46], [0.0, 0.1, 0.1]]], np.float32)
    t = neutral_transmission(rgb, BASE)
    assert t.shape == rgb.shape and t.dtype == np.float32
    assert np.all(t[..., 0] == t[..., 1]) and np.all(t[..., 1] == t[..., 2])
    norm = np.maximum(rgb / BASE.astype(np.float32), EPS)
    np.testing.assert_allclose(t[..., 0], np.cbrt(norm.prod(-1)), rtol=1e-6)
    assert t[0, 1, 0] == pytest.approx(1.0, rel=1e-6)  # the base itself
    assert t[0, 2, 0] > 0  # a non-positive channel is floored like density() does


def test_bw_mode_output_is_neutral_and_replays(tmp_path):
    p = tmp_path / "bw.dng"
    write_dng(p, bw_scan())
    region = (20, 120, 200, 220)
    out, sc = convert(p, image_region=region, film="bw")
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])
    assert sc["film"]["mode"] == "bw" and list(sc["inversion"]["params"]["base"]) == [1.0, 1.0, 1.0]
    np.testing.assert_allclose(sc["base"]["rgb_render"], BASE, rtol=0.01)
    out2, _ = convert(p, **replay_kwargs(sc))
    assert np.array_equal(out, out2)


def test_colour_mode_tints_the_same_negative(tmp_path):
    """Colour mode sets a white point per channel; here they differ only by noise, on real scans
    also by light and sensor balance, so its output is not exactly neutral."""
    p = tmp_path / "bw.dng"
    write_dng(p, bw_scan())
    out, sc = convert(p, image_region=(20, 120, 200, 220))
    assert "film" not in sc  # colour sidecars keep their field set
    assert not np.array_equal(out[..., 0], out[..., 2])


def test_bw_auto_keeps_tone_and_drops_colour_look(tmp_path):
    p = tmp_path / "bw.dng"
    write_dng(p, bw_scan())
    out, sc = convert(p, image_region=(20, 120, 200, 220), film="bw", auto=NOREF_V2)
    rp = sc["render"]["params"]
    assert (rp["temp"], rp["tint"], rp["sat"]) == (0.0, 0.0, 1.0)
    assert rp["matrix"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert sorted(sc["auto"]["bw_neutralised"]) == ["matrix", "sat", "temp", "tint"]
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])
    _, sc_c = convert(p, image_region=(20, 120, 200, 220), auto=NOREF_V2)
    for k in ("ev", "contrast", "midtone", "shoulder"):  # tonal parameters come from the same look
        assert rp[k] == sc_c["render"]["params"][k]


def test_film_mode_validated(tmp_path):
    p = tmp_path / "bw.dng"
    write_dng(p, bw_scan())
    with pytest.raises(ValueError, match="film"):
        convert(p, film="slide")
