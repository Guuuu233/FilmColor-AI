"""Regression and hardening tests for system bug fixes, edge cases, and fast paths."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import shutil
import struct
from pathlib import Path

import numpy as np
import pytest

from filmcolor import colorimetry, export, frozen, invert, noref, pipeline, preview, render
from test_scanner import scan, write_dng


def test_frozen_packaged_manifest_fallback_and_hash_cache_invalidation(tmp_path, monkeypatch):
    """Packaged wheel fallback (_manifests / FILMCOLOR_MANIFEST_DIR) works without repo tree,
    and sha256_file invalidates its stat-keyed cache immediately when file bytes change."""
    pkg_manifests = tmp_path / "_manifests"
    pkg_manifests.mkdir()
    sample = tmp_path / "sample.dat"
    sample.write_bytes(b"version-1")
    h1 = frozen.sha256_file(sample)

    manifest_file = pkg_manifests / "frozen_test_pkg.json"
    manifest_file.write_text(json.dumps({
        "id": "frozen_test_pkg",
        "files": [{"id": "F1", "sha256": h1}],
        "blocked": {"names": ["sealed_stem"], "roll_ids": ["sealed_roll"]},
    }))

    monkeypatch.setenv("FILMCOLOR_MANIFEST_DIR", str(pkg_manifests))
    monkeypatch.setattr(frozen, "MANIFESTS", [])
    monkeypatch.setattr(frozen, "_FROZEN_HASHES_CACHE", None)
    monkeypatch.setattr(frozen, "_FROZEN_BLOCKS_CACHE", None)

    fh = frozen.frozen_hashes()
    assert fh == {h1: "frozen_test_pkg:F1"}
    # Mutating returned dict must not corrupt internal cache
    fh["tamper"] = "x"
    assert "tamper" not in frozen.frozen_hashes()

    with pytest.raises(frozen.FrozenDataError, match="frozen_test_pkg:F1"):
        frozen.check_source(sample)
    with pytest.raises(frozen.FrozenDataError, match="sealed_stem"):
        frozen.check_name("sealed_stem.dng")
    with pytest.raises(frozen.FrozenDataError, match="sealed_roll"):
        frozen.check_roll("sealed_roll")

    # Modify file content -> sha256_file must recompute rather than returning stale hash
    sample.write_bytes(b"version-2-different-length")
    h2 = frozen.sha256_file(sample)
    assert h2 != h1
    assert frozen.check_source(sample) == h2


def test_software_info_handles_uninstalled_source_tree(monkeypatch):
    """software_info() falls back cleanly to filmcolor.__version__ when distribution metadata is absent."""
    def _missing(_name):
        raise importlib.metadata.PackageNotFoundError(_name)

    monkeypatch.setattr(importlib.metadata, "version", _missing)
    info = pipeline.software_info()
    assert info["filmcolor"] == "2.0.1"
    assert "numpy" in info and "rawpy" in info


def test_metering_mask_and_estimator_survive_all_holder_frame():
    """When holder_mask covers 100% of the frame, film_box returns None and estimate_params_v3
    falls back cleanly to manual_review_required instead of raising TypeError."""
    P_all_holder = np.full((64, 64, 3), 25.0, dtype=np.float32)
    m, diag = noref._metering_mask(P_all_holder, None, noref.NOREF_V3)
    assert diag is not None and diag["used"] is False
    res = noref.estimate_params_v3(P_all_holder, None, noref.NOREF_V3, span_d_g=0.90)
    assert res["status"] == "manual_review_required"


def test_frame_stats_rejects_empty_mask_cleanly():
    P = np.full((32, 32, 3), 0.25, dtype=np.float32)
    empty_mask = np.zeros((32, 32), dtype=bool)
    with pytest.raises(ValueError, match="too few pixels"):
        noref.frame_stats(P, empty_mask)


def test_preview_render_stats_matches_render_and_rejects_bad_crop(tmp_path):
    rgb, irl = scan()
    p = tmp_path / "s.dng"
    write_dng(p, rgb, irl)
    pv = preview.load_preview(p, max_side=120, keep_full=True)
    params = dict(ev=0.25, temp=0.1, tint=-0.05, sat=1.15, contrast=1.35, midtone=0.22,
                  shoulder=0.30, chroma_preserve=0.65, slope_r=-0.08, slope_b=0.12)
    disp_r = pv.render(params)
    disp_s, stats = pv.render_stats(params)
    np.testing.assert_array_equal(disp_r, disp_s)
    assert 0.0 <= stats["below_black"] <= 1.0 and 0.0 <= stats["above_white"] <= 1.0

    with pytest.raises(ValueError, match="positive"):
        pv.render_crop((10, 10, 0, 50))
    with pytest.raises(ValueError, match="positive"):
        pv.render_crop((10, 10, 50, -5))


def test_render_fast_full_rejects_non_finite_and_matches_render_full():
    rng = np.random.default_rng(123)
    P = rng.uniform(0.0, 2.0, (64, 80, 3)).astype(np.float32)
    rp = render.RenderParams(
        ev=0.2, temp=0.1, tint=0.05, sat=0.95, contrast=1.4, midtone=0.21, shoulder=0.3,
        highlights=-0.1, shadows=0.08, whites=0.05, blacks=-0.03, vibrance=0.1,
        slope_r=-0.09, slope_b=0.14, chroma_preserve=0.75, bpc=0.01, soft_clip=0.88,
        tone_curve=render.CURVE_PRESETS["standard"],
    )
    e_full, _, pre_full = render.render_full(P, rp)
    e_fast, _, _, pre_fast = render.render_fast_full(P, rp)
    np.testing.assert_allclose(e_fast, e_full, atol=2e-4)
    np.testing.assert_allclose(pre_fast, pre_full, atol=2e-4)

    P_bad = P.copy()
    P_bad[3, 4, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        render.render_fast_full(P_bad, rp)
    with pytest.raises(ValueError, match="NaN"):
        render.render(P_bad, rp, fast=True)


def test_colorimetry_scalar_1d_distinct_trc_and_corrupt_icc():
    icc = bytearray(export.icc_profile("srgb"))
    prof = colorimetry.MatrixTRCProfile(bytes(icc))

    # Scalar / 0-D / 1-D (3,) / 2-D / 3-D inputs on shared curv LUT
    assert isinstance(prof.trc[0](0.5), float)
    assert isinstance(prof.trc[0](np.float64(0.5)), float)
    v1 = np.array([0.2, 0.5, 0.8], dtype=np.float64)
    v2 = v1[None, :]
    np.testing.assert_allclose(prof.to_xyz(v1), prof.to_xyz(v2)[0], atol=1e-15)

    # Distinct per-channel curv LUTs on 1-D (3,) input
    prof_distinct = colorimetry.MatrixTRCProfile(bytes(icc))
    prof_distinct._shared_trc = False
    np.testing.assert_allclose(prof_distinct.to_xyz(v1), prof.to_xyz(v1), atol=1e-15)

    # Corrupt / truncated ICC profiles raise ValueError cleanly (never struct.error or ZeroDivisionError)
    with pytest.raises(ValueError):
        colorimetry.MatrixTRCProfile(b"short")
    bad_para = b"para" + b"\x00" * 4 + struct.pack(">H", 1) + b"\x00\x00" + struct.pack(">iii", 65536, 0, 0)
    with pytest.raises(ValueError, match="a == 0"):
        colorimetry._trc(bad_para)


def test_bw_mode_neutralises_extended_v3_colour_params_and_thumbnails(tmp_path, monkeypatch):
    monkeypatch.setenv("FILMCOLOR_DATA_ROOT", str(tmp_path / "cache"))
    root = tmp_path / "neg"
    root.mkdir()
    rgb, irl = scan()
    p = root / "bw_test.dng"
    write_dng(p, rgb, irl)

    # 1. neutralise_bw_params strips extended colour scalars and R/G/B splines while keeping RGB tonal spline
    dirty = dict(
        ev=0.2, temp=0.35, tint=0.04, sat=0.88, contrast=1.42, midtone=0.22, shoulder=0.25,
        vibrance=0.15, slope_r=-0.09, slope_b=0.20, skin_protect=0.35, log_unmix=0.50,
        matrix=[[1.05, -0.03, -0.02], [-0.01, 1.02, -0.01], [-0.04, 0.08, 0.96]],
        tone_curve={"rgb": [[0.0, 0.02], [0.5, 0.52], [1.0, 0.98]], "b": [[0.0, 0.05], [1.0, 0.95]]},
    )
    clean, dropped = pipeline.neutralise_bw_params(dirty)
    assert clean["slope_r"] == 0.0 and clean["slope_b"] == 0.0
    assert clean["vibrance"] == 0.0 and clean["skin_protect"] == 0.0 and clean["log_unmix"] == 0.0
    assert clean["tone_curve"] == {"rgb": [[0.0, 0.02], [0.5, 0.52], [1.0, 0.98]]}
    assert set(dropped) >= {"matrix", "sat", "temp", "tint", "vibrance", "slope_r", "slope_b", "skin_protect", "log_unmix", "tone_curve"}

    # 2. convert(..., film='bw', auto=portrait) produces strictly neutral R == G == B pixels
    out, sc = pipeline.convert(p, film="bw", auto=noref.preset_config("portrait"))
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])
    assert "slope_r" not in sc["render"]["params"] and "slope_b" not in sc["render"]["params"]

def test_qc_empty_mask_and_compare_margin_guards():
    from filmcolor import compare, qc
    disp = np.full((16, 16, 3), 0.5, dtype=np.float32)
    empty = np.zeros((16, 16), dtype=bool)
    rep = {"range": {c: {"clip_high": 0.0, "below_black": 0.0} for c in ("R", "G", "B")}}
    res = qc.quality_checks(disp, empty, rep)
    assert res["status"] == "fail"
    assert any(c["name"] == "non_empty_mask" and c["status"] == "fail" for c in res["checks"])

    lab = np.zeros((10, 10, 3), dtype=np.float32)
    lab[..., 0] = 50.0
    with pytest.raises(ValueError, match="margin"):
        compare.metrics(lab, lab, margin=0.5)
    with pytest.raises(ValueError, match="margin"):
        compare.metrics(lab, lab, margin=-0.05)
    tiny = np.zeros((0, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="empty interior"):
        compare.metrics(tiny, tiny, margin=0.03)


def test_holder_rebate_strip_and_convert_cli_preset(tmp_path):
    import sys
    from filmcolor import holder
    P = np.full((160, 200, 3), 0.25, dtype=np.float32)
    # Outer bright holder border (> HOLDER_OVER)
    P[:12, :, :] = 30.0
    P[-12:, :, :] = 30.0
    P[:, :12, :] = 30.0
    P[:, -12:, :] = 30.0
    # Dense hot strip right inside top edge of film_box separated by 1px seam
    P[13:17, 15:185, :] = 20.0
    hm = holder.holder_mask(P)
    strip = holder.rebate_strip_components(P, hm)
    assert strip.shape == hm.shape and strip.dtype == bool
    assert strip[14, 100] and not strip[80, 100]

    # CLI --preset support and mutual-exclusion check
    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import convert as convert_cli

    rgb, irl = scan()
    dng = tmp_path / "in.dng"
    write_dng(dng, rgb, irl)
    out_c = tmp_path / "out_color"
    rc = convert_cli.main([str(out_c), str(dng), "--preset", "portrait"])
    assert rc == 0
    sc_c = json.loads((out_c / "in.json").read_text())
    assert sc_c["auto"]["preset"] == "portrait" and sc_c["auto"]["config"] == "noref_v3"

    out_bw = tmp_path / "out_bw"
    rc_bw = convert_cli.main([str(out_bw), str(dng), "--preset", "portrait", "--film", "bw"])
    assert rc_bw == 0
    sc_bw = json.loads((out_bw / "in.json").read_text())
    assert sc_bw["film"]["mode"] == "bw" and "slope_r" not in sc_bw["render"]["params"]

    with pytest.raises(SystemExit):
        convert_cli.main([str(tmp_path / "bad"), str(dng), "--auto", "--preset", "standard"])


def test_estimate_from_features_v3_parity():
    rng = np.random.default_rng(77)
    P = rng.uniform(0.02, 0.95, (96, 128, 3)).astype(np.float32)
    mask = np.ones((96, 128), dtype=bool)
    cfg = noref.preset_config("standard")
    r_direct = noref.estimate_params_v3(P, mask, cfg, span_d_g=0.92)
    mask_m, metering = noref._metering_mask(P, mask, cfg)
    st = noref.frame_stats(P, mask_m, cfg)
    px = noref._sample(P, mask_m, cfg["n_sample"] // 4, seed=1)
    tc = noref._textured_cells(P, mask_m, cfg)
    r_feat = noref._estimate_from_features_v3(px, tc, st, metering, cfg, span_d_g=0.92)
    assert r_direct == r_feat

    # Low-information frame parity
    P_flat = np.full((64, 64, 3), 0.25, dtype=np.float32)
    m_flat, met_flat = noref._metering_mask(P_flat, None, cfg)
    assert noref.estimate_params_v3(P_flat, None, cfg, span_d_g=0.10) == noref._estimate_from_features_v3(
        np.empty((0, 3), np.float32), {}, noref.frame_stats(P_flat, m_flat, cfg), met_flat, cfg, span_d_g=0.10
    )


def test_t17_colour_refinements_and_bw_low_info_thumbnail(tmp_path, monkeypatch):
    # 1. Calibrated non-sky cyan rescue + C-41 cross-unmix weight (0.62)
    rng = np.random.default_rng(1717)
    base_y = rng.uniform(0.04, 0.72, (96, 128, 1)).astype(np.float32)
    # Construct non-sky positive with cyan-biased midtones (log2 R/G < -0.10, log2 B/G > +0.08)
    P_cal = np.concatenate([base_y * 0.86, base_y * 1.00, base_y * 1.10], axis=-1).astype(np.float32)
    eye_m = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    cfg_t17 = dict(noref.preset_config("standard"), matrix=eye_m)
    cfg_t16 = dict(cfg_t17, t17_cal_cross_weight=0.50, t17_cal_nonsky_cyan_check=False)
    r17 = noref.estimate_params_v3(P_cal, None, cfg_t17, span_d_g=0.95)
    r16 = noref.estimate_params_v3(P_cal, None, cfg_t16, span_d_g=0.95)
    assert r17["params"]["temp"] > r16["params"]["temp"]
    assert r17["params"]["matrix"][0][0] > r16["params"]["matrix"][0][0]

    # 2. Uncalibrated WB confidence gamma + warm de-overshoot damping
    P_unc = np.concatenate([base_y * 0.78, base_y * 1.00, base_y * 1.18], axis=-1).astype(np.float32)
    u17 = noref.estimate_params_v3(P_unc, None, noref.preset_config("standard"), span_d_g=0.95)
    assert "t17_uncal_wb_deov" in u17["diagnostics"].get("print_grade_v3", {})
