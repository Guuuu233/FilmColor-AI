"""Tests on the NLP official sample RAFs (Fujifilm X-T2, X-Trans). Skipped when the files are absent.

Expected values are regression baselines observed on 2026-10-13 (experiments/s1_diag), not
physical ground truth.
"""
from pathlib import Path

import numpy as np
import pytest

from filmcolor.base import estimate_base_auto, estimate_base_manual
from filmcolor.invert import base_from_regions, central_region, invert_negative, region_mask
from filmcolor.raw import linearize, saturation_mask, superpixel
from filmcolor.rawpy_io import demosaic_linear, load_raw

SRC = Path(__file__).resolve().parents[1] / "data/raw/public/nlp_official"
FILES = sorted(SRC.glob("*.RAF")) if SRC.is_dir() else []
pytestmark = pytest.mark.skipif(len(FILES) != 12, reason="NLP sample RAFs not present")

# frames whose auto base chromaticity agreed within 0.04 (log2) when first measured
CONSISTENT_GROUP = ["_DSF9229", "_DSF9231", "_DSF9235", "_DSF9237", "_DSF9238", "_DSF9239"]


def _prepare(path):
    f = load_raw(path)
    return f, linearize(f), f.colors, saturation_mask(f)


def _auto(path):
    f, lin, col, sat = _prepare(path)
    return f, lin, col, sat, estimate_base_auto(lin, col, sat, f.pattern.shape[0], crop=f.crop)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_load_raw_xtrans(path):
    f = load_raw(path)
    assert f.pattern.shape == (6, 6)
    assert sorted(np.bincount(f.pattern.ravel())) == [8, 8, 20]  # X-Trans: 8 R, 8 B, 20 G
    assert f.cfa.shape == (4032, 6032)
    assert f.white == 16383
    assert np.all(f.black == 1022)
    assert f.meta["margins"] == (0, 0)  # crop basis is unambiguous only in this case
    t, l, h, w = f.crop
    assert 0 <= t and 0 <= l and t + h <= f.cfa.shape[0] and l + w <= f.cfa.shape[1]


def test_auto_base_regression_and_determinism():
    path = SRC / "_DSF9235.RAF"
    *_, a = _auto(path)
    *_, b = _auto(path)
    assert a.to_dict() == b.to_dict()
    rg, bg = a.diagnostics["chroma_centre_log2_RG_BG"]
    assert abs(rg - 0.432) < 0.01 and abs(bg - (-1.698)) < 0.01


def test_auto_base_chromaticity_consistent_in_group():
    chroma = np.array([_auto(SRC / f"{s}.RAF")[-1].diagnostics["chroma_centre_log2_RG_BG"]
                       for s in CONSISTENT_GROUP])
    assert np.all(chroma.max(0) - chroma.min(0) < 0.05), chroma


@pytest.mark.parametrize("stem", ["_DSF8434", "_DSF8951"])
def test_auto_base_excludes_saturated_blocks(stem):
    *_, est = _auto(SRC / f"{stem}.RAF")
    assert est.diagnostics["n_saturated_blocks"] > 0
    assert any("saturated" in w for w in est.warnings)
    # every kept block is free of saturated samples
    f, lin, col, sat = _prepare(SRC / f"{stem}.RAF")
    for t, l, h, w in est.regions:
        assert not sat[t : t + h, l : l + w].any()


@pytest.mark.parametrize("stem", ["_DSF8948", "_DSF9235"])
def test_manual_base_agrees_with_auto(stem):
    """Manual estimate on regions taken from the auto result: different statistic, same answer."""
    f, lin, col, sat, auto = _auto(SRC / f"{stem}.RAF")
    regions = auto.regions[:: max(1, len(auto.regions) // 5)][:5]
    manual = estimate_base_manual(lin, col, sat, regions)
    assert np.max(np.abs(np.array(manual.rgb) / np.array(auto.rgb) - 1)) < 0.01
    assert all(manual.stats[c]["sat_frac"] == 0 for c in "RGB")


@pytest.fixture(scope="module")
def rendered_9235():
    path = SRC / "_DSF9235.RAF"
    return path, demosaic_linear(path)


def test_analysis_and_render_paths_agree(rendered_9235):
    """CFA block means (analysis path) vs demosaiced block means (render path), same pixels."""
    path, dm = rendered_9235
    f, lin, col, sat = _prepare(path)
    block = 48
    sp, sp_sat = superpixel(lin, col, sat, f.pattern.shape[0], block)
    assert dm.shape[:2] == f.cfa.shape
    h, w = sp.shape[0] * block, sp.shape[1] * block
    bm = dm[:h, :w].reshape(sp.shape[0], block, sp.shape[1], block, 3).mean(axis=(1, 3))
    ok = ~sp_sat & np.all(sp > 0.02, -1)  # render path clips below-black samples to 0
    ratio = bm[ok] / sp[ok]
    med = np.median(ratio, 0)
    assert np.all(np.abs(med - 1) < 0.01), med
    assert np.all(np.percentile(ratio, 1, 0) > 0.97) and np.all(np.percentile(ratio, 99, 0) < 1.03)


def test_inversion_on_real_negative(rendered_9235):
    path, rgb = rendered_9235
    f, lin, col, sat, auto = _auto(path)
    base = base_from_regions(rgb, auto.regions)
    mask = ~sat & region_mask(sat.shape, central_region(sat.shape))
    res = invert_negative(rgb, base["rgb"], mask=mask)
    np.testing.assert_allclose(res.params.d_hi, (0.570, 0.668, 0.848), atol=0.02)
    for c in "RGB":
        assert abs(res.report["range"][c]["clip_high"] - 0.001) < 0.0005  # 99.9th percentile
    assert min(res.report["full_image_above_white"]) > 0.25  # holder mask flagged
    assert np.all(np.isfinite(res.positive))
    assert res.display.dtype == np.float32 and res.display.min() >= 0 and res.display.max() <= 1


def test_pipeline_output_reproducible_and_replayable(tmp_path):
    import tifffile

    from filmcolor.pipeline import replay_kwargs, write

    path = SRC / "_DSF8948.RAF"
    a = write(path, tmp_path / "a")
    b = write(path, tmp_path / "b")
    assert a["output"]["sha256"] == b["output"]["sha256"]  # same input, same version: identical bytes
    assert a["input"]["sha256"] == "644375ef0652998341a34ba0ddc904901a4b13ea5232aeca119c822201754326"
    for key in ("schema", "input", "software", "decode", "base", "image_region", "inversion", "report", "output"):
        assert key in a
    assert a["output"]["colorimetric_calibration"] == "none"
    assert a["qc"]["status"] in ("pass", "warn") and a["qc"]["provisional"]
    assert {c["name"]: c["status"] for c in a["qc"]["checks"]}["base_inside_image"] == "pass"
    assert a["render"]["params"]["ev"] == 0 and a["schema"].endswith("0.2")
    c = write(path, tmp_path / "c", **replay_kwargs(a))  # no estimation, parameters from the sidecar
    assert c["base"]["method"] == "given" and c["inversion"]["d_hi_source"] == "given"
    assert np.array_equal(tifffile.imread(tmp_path / "a/_DSF8948.tif"), tifffile.imread(tmp_path / "c/_DSF8948.tif"))


@pytest.mark.parametrize("keep", [1_000_000, 25_000_000, -100])
def test_truncated_raf_raises_decode_error(tmp_path, keep):
    from filmcolor.rawpy_io import RawDecodeError

    data = (SRC / "_DSF8948.RAF").read_bytes()
    p = tmp_path / "trunc.RAF"
    p.write_bytes(data[:keep])
    with pytest.raises(RawDecodeError, match="trunc.RAF"):
        load_raw(p)
    with pytest.raises(RawDecodeError):
        demosaic_linear(p)


def test_pipeline_rejects_bad_region_before_rendering():
    import time

    from filmcolor.pipeline import convert

    t0 = time.time()
    with pytest.raises(ValueError):
        convert(SRC / "_DSF8948.RAF", image_region=(4000, 0, 100, 100))
    with pytest.raises(ValueError):
        convert(SRC / "_DSF8948.RAF", image_region=(0, 0, 100))
    with pytest.raises(ValueError):
        convert(SRC / "_DSF8948.RAF", white_pct=0)
    assert time.time() - t0 < 5  # the 11 s demosaic was never reached



def test_pipeline_render_params_are_recorded_and_replayed(tmp_path):
    import tifffile

    from filmcolor.pipeline import replay_kwargs, write

    path = SRC / "_DSF9235.RAF"
    rp = dict(ev=0.3, temp=0.2, tint=-0.1, sat=1.3, contrast=1.2, midtone=0.22, shoulder=0.3)
    a = write(path, tmp_path / "a", render_params=rp)
    assert {k: a["render"]["params"][k] for k in rp} == rp
    b = write(path, tmp_path / "b", **replay_kwargs(a))
    assert np.array_equal(tifffile.imread(tmp_path / "a/_DSF9235.tif"), tifffile.imread(tmp_path / "b/_DSF9235.tif"))
    with pytest.raises(ValueError):
        write(path, tmp_path / "c", render_params=dict(sat=5))  # rejected before decoding


def test_pipeline_auto_records_estimate_and_replays(tmp_path):
    import tifffile

    from filmcolor.noref import NOREF_V2, load_look
    from filmcolor.pipeline import replay_kwargs, write

    path = SRC / "_DSF9235.RAF"
    a = write(path, tmp_path / "a", auto=NOREF_V2)
    au = a["auto"]
    assert (au["config"], au["look"], au["status"]) == ("noref_v2", "global_look_v1", "ok")
    assert len(au["look_file"]["sha256"]) == 64 and a["render"]["params"] == au["params"] == load_look()["params"]
    b = write(path, tmp_path / "b", **replay_kwargs(a))  # replay reads the recorded parameters, no estimation
    assert "auto" not in b
    assert np.array_equal(tifffile.imread(tmp_path / "a/_DSF9235.tif"), tifffile.imread(tmp_path / "b/_DSF9235.tif"))
    with pytest.raises(ValueError, match="mutually exclusive"):
        write(path, tmp_path / "c", auto=NOREF_V2, render_params=dict(ev=0.1))


def test_default_render_reproduces_sprint1_output_bytes(tmp_path):
    """Default render parameters must give the TIFF recorded by the Sprint 1 run (52a4acc)."""
    from filmcolor.pipeline import write

    a = write(SRC / "_DSF8948.RAF", tmp_path)
    assert a["output"]["sha256"] == "ed6516673a19abe372d2c53f2488527699226c3e49a451cbe6c0d0d92b30abae"
