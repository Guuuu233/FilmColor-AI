"""Unit and integration tests for T24 physical+ML residual pipeline (pure NumPy/JSON)."""
import json
import sys
import numpy as np
import pytest

from filmcolor.noref import NOREF_T24, estimate_params_t24, pool_roll_params_t24
from filmcolor.pipeline import coerce_render_params, convert, neutralise_bw_params, replay_kwargs
from filmcolor.preview import load_preview
from filmcolor.render import (
    ZERO_MATRIX,
    RenderParams,
    RenderParamsT19,
    RenderParamsT24,
    render_t19,
    render_t24,
)
from test_scanner import scan, write_dng


def test_t24_params_roundtrip_and_coerce():
    rp = RenderParamsT24(
        ev=0.15, temp=-0.1, tint=0.05, sat=0.95, contrast=1.25, shoulder=0.6, toe=-0.2,
        slope_r=0.08, slope_b=-0.06, eps_rgb=(0.004, -0.002, 0.003),
        residual_matrix=((-0.02, 0.01, 0.01), (0.01, -0.02, 0.01), (0.00, 0.01, -0.01)),
    )
    rt = RenderParamsT24.from_dict(json.loads(json.dumps(rp.to_dict())))
    assert rt == rp
    assert isinstance(coerce_render_params(rt.to_dict()), RenderParamsT24)
    # Legacy RenderParams with slope_r must still coerce to RenderParams, not RenderParamsT24
    leg = RenderParams(slope_r=0.2, slope_b=-0.1).to_dict()
    assert isinstance(coerce_render_params(leg), RenderParams)
    assert isinstance(coerce_render_params(RenderParamsT19().to_dict()), RenderParamsT19)

    with pytest.raises(ValueError):
        RenderParamsT24(slope_r=1.2)
    with pytest.raises(ValueError):
        RenderParamsT24(eps_rgb=(0.5, 0.0, 0.0))
    with pytest.raises(ValueError):
        RenderParamsT24(residual_matrix=((0.9, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))


def test_render_t24_parity_and_guards():
    rng = np.random.default_rng(2401)
    P = np.clip(0.18 * (2.0 ** rng.normal(0.0, 1.3, size=(96, 128, 3))), 1e-5, 5.0).astype(np.float32)

    # Zero-residual T24 matches render_t19 identically
    rp19 = RenderParamsT19(ev=-0.1, temp=0.12, tint=-0.05, sat=0.92, contrast=1.3, shoulder=0.55, toe=-0.25)
    rp24_base = RenderParamsT24(ev=-0.1, temp=0.12, tint=-0.05, sat=0.92, contrast=1.3, shoulder=0.55, toe=-0.25)
    e19, d19 = render_t19(P, rp19, fast=True)
    e24, d24 = render_t24(P, rp24_base, fast=True)
    assert np.allclose(e19, e24, atol=1e-6)
    assert d19 == d24

    # Full T24 fast LUT vs exact math parity and chunk parity
    rp24 = RenderParamsT24(
        ev=0.12, temp=-0.08, tint=0.04, sat=0.95, contrast=1.28, shoulder=0.48, toe=0.35,
        slope_r=0.09, slope_b=-0.07, eps_rgb=(0.005, -0.002, 0.004),
        residual_matrix=((-0.02, 0.01, 0.01), (0.01, -0.02, 0.01), (0.0, 0.01, -0.01)),
    )
    ef, df = render_t24(P, rp24, fast=True, chunk_rows=1024)
    ex, dx = render_t24(P, rp24, fast=False, chunk_rows=1024)
    ec, dc = render_t24(P, rp24, fast=True, chunk_rows=32)
    assert np.allclose(ef, ex, atol=5e-6)
    assert np.allclose(ef, ec, atol=1e-7)
    assert df == dx == dc

    empty = np.zeros((0, 4, 3), dtype=np.float32)
    ee, de = render_t24(empty, rp24)
    assert ee.shape == empty.shape and np.isnan(de["above_white"])
    bad = P.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        render_t24(bad, rp24)


def test_estimate_params_t24_branches_and_pooling():
    rng = np.random.default_rng(2402)
    syn = np.clip(0.18 * (2.0 ** rng.normal(0.0, 1.0, size=(128, 160, 3))), 1e-4, 2.5).astype(np.float32)

    est_cal = estimate_params_t24(syn, d_hi=(0.85, 0.90, 1.05), has_cam_matrix=True)
    assert est_cal["status"] == "ok" and est_cal["branch"] == "cal"
    rp_cal = RenderParamsT24.from_dict(est_cal["params"])
    assert rp_cal.eps_rgb == (0.0, 0.0, 0.0)

    est_unc = estimate_params_t24(syn, d_hi=(0.85, 0.90, 0.40), has_cam_matrix=False)
    assert est_unc["status"] == "ok" and est_unc["branch"] == "uncal"
    rp_unc = RenderParamsT24.from_dict(est_unc["params"])
    assert rp_unc.residual_matrix == ZERO_MATRIX

    flat = np.full((64, 64, 3), 0.22, dtype=np.float32)
    for hc in (True, False):
        ef = estimate_params_t24(flat, d_hi=(0.85, 0.90, 0.45), has_cam_matrix=hc)
        assert np.isfinite(RenderParamsT24.from_dict(ef["params"]).ev)

    # Roll pooling preserves T24 extra controls while shrinking WB/tone
    p1 = dict(est_unc["params"], temp=0.10, contrast=1.10)
    p2 = dict(est_unc["params"], temp=0.12, contrast=1.12)
    p3 = dict(est_unc["params"], temp=0.11, contrast=1.11)
    p4 = dict(est_unc["params"], temp=0.60, contrast=1.60)
    pooled = pool_roll_params_t24([p1, p2, p3, p4])
    assert pooled[3]["temp"] < 0.60 and pooled[3]["contrast"] < 1.60
    assert pooled[3]["eps_rgb"] == p4["eps_rgb"]
    assert pooled[3]["slope_r"] == p4["slope_r"]


def test_t24_pipeline_preview_bw_and_cli(tmp_path):
    scripts_dir = str((__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import convert as convert_cli

    rgb, irl = scan()
    dng = tmp_path / "t24.dng"
    write_dng(dng, rgb, irl)

    out, sc = convert(dng, image_region=(20, 120, 200, 220), auto=NOREF_T24)
    assert out.shape == (rgb.shape[0], rgb.shape[1], 3)
    assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0
    assert sc["auto"]["config"] == "noref_t24" and sc["auto"]["version"] == "T24"
    assert sc["auto"]["branch"] == "cal"
    assert "eps_rgb" in sc["render"]["params"] and "residual_matrix" in sc["render"]["params"]

    out_u, sc_u = convert(dng, image_region=(20, 120, 200, 220), d_hi=(1.35, 1.49, 0.55), auto=NOREF_T24)
    assert out_u.shape == (rgb.shape[0], rgb.shape[1], 3) and sc_u["auto"]["branch"] == "uncal"

    out_bw, sc_bw = convert(dng, image_region=(20, 120, 200, 220), auto=NOREF_T24, film="bw")
    assert out_bw.shape == (rgb.shape[0], rgb.shape[1], 3)
    assert sc_bw["render"]["params"]["eps_rgb"] == [0.0, 0.0, 0.0]
    assert sc_bw["render"]["params"]["slope_r"] == 0.0
    assert sc_bw["render"]["params"]["residual_matrix"] == [list(r) for r in ZERO_MATRIX]

    bw_p, dropped = neutralise_bw_params(
        RenderParamsT24(slope_r=0.1, eps_rgb=(0.01, 0.0, 0.0), residual_matrix=((0.02, -0.01, -0.01), (0, 0, 0), (0, 0, 0))).to_dict()
    )
    assert "eps_rgb" in dropped and "residual_matrix" in dropped and "slope_r" in dropped

    pv = load_preview(dng, max_side=120, image_region=(20, 120, 200, 220), auto=NOREF_T24, keep_full=True)
    assert pv.auto["config"] == "noref_t24" and pv.auto["version"] == "T24"
    disp_pv = pv.render(pv.auto["params"])
    disp_st, stats = pv.render_stats(pv.auto["params"])
    np.testing.assert_array_equal(disp_pv, disp_st)
    assert 0.0 <= stats["below_black"] <= 1.0 and 0.0 <= stats["above_white"] <= 1.0
    crop = pv.render_crop((20, 120, 32, 32), pv.auto["params"])
    assert crop.shape == (32, 32, 3)

    rc = convert_cli.main([str(tmp_path / "cli_out_t24"), str(dng), "--t24"])
    assert rc == 0


@pytest.mark.parametrize("cfg_name,kw", [
    ("NOREF_T24", dict()),                                   # orange-mask scan -> CAL branch
    ("NOREF_T24", dict(d_hi=(1.35, 1.49, 0.55))),            # BlueNeg ratios -> UNCAL branch (eps_rgb, T23 slopes)
    ("NOREF_T24", dict(film="bw")),
    ("NOREF_T23", dict(d_hi=(1.35, 1.49, 0.55))),
    ("NOREF_T22", dict(d_hi=(1.35, 1.49, 0.55))),
])
def test_t19_family_sidecar_replay_is_exact(tmp_path, cfg_name, kw):
    """A T19-family render is made on the T18/T23 re-inverted positive; replaying its sidecar (which has no ``auto``)
    must re-invert the same way. Before render.t18_inversion was recorded, UNCAL replays differed by up to 0.09."""
    import filmcolor.noref as noref
    rgb, irl = scan()
    dng = tmp_path / "r.dng"
    write_dng(dng, rgb, irl)
    img, sc = convert(dng, image_region=(20, 120, 200, 220), auto=getattr(noref, cfg_name), **kw)
    assert set(sc["render"]["t18_inversion"]) == {"version", "has_cam_matrix"}
    img_r, sc_r = convert(dng, **replay_kwargs(sc))
    np.testing.assert_array_equal(img_r, img)
    assert sc_r["render"]["t18_inversion"] == sc["render"]["t18_inversion"]
    # sidecars written before the field existed: derived from the auto config
    old = json.loads(json.dumps(sc))
    del old["render"]["t18_inversion"]
    if kw.get("film") != "bw":  # a B&W sidecar's neutralised matrix no longer tells the camera regime
        np.testing.assert_array_equal(convert(dng, **replay_kwargs(old))[0], img)


def test_legacy_sidecars_and_bad_t18_inversion(tmp_path):
    rgb, irl = scan()
    dng = tmp_path / "l.dng"
    write_dng(dng, rgb, irl)
    _, sc = convert(dng, image_region=(20, 120, 200, 220))
    assert "t18_inversion" not in sc["render"] and "t18_inversion" not in replay_kwargs(sc)
    with pytest.raises(ValueError, match="t18_inversion"):
        convert(dng, image_region=(20, 120, 200, 220), render_params=RenderParamsT24().to_dict(),
                t18_inversion=dict(version="t99", has_cam_matrix=False))


def test_t24_preview_export_matches_previewed_pixels(tmp_path):
    import tifffile
    rgb, irl = scan()
    dng = tmp_path / "p.dng"
    write_dng(dng, rgb, irl)
    for kw in (dict(), dict(d_hi=(1.35, 1.49, 0.55))):
        pv = load_preview(dng, max_side=120, image_region=(20, 120, 200, 220), auto=NOREF_T24, keep_full=True, **kw)
        region = (40, 140, 64, 96)
        crop = pv.render_crop(region, pv.auto["params"])
        sc = pv.export(tmp_path / f"o{len(kw)}", pv.auto["params"])
        out = tifffile.imread(sc["output"]["path"]).astype(np.float64) / 65535.0
        t, l, h, w = region
        assert np.abs(out[t:t + h, l:l + w] - crop).max() <= 0.5 / 65535 + 1e-6
        assert sc["render"]["t18_inversion"] == pv.t18_inversion()


def test_t24_degenerate_frames_fail_cleanly():
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError, match="too small"):
        estimate_params_t24(np.full((2, 3, 3), 0.2, np.float32), d_hi=(0.85, 0.9, 1.05))
    with pytest.raises(ValueError, match="too few pixels"):
        estimate_params_t24(np.zeros((64, 64, 3), np.float32), d_hi=(0.85, 0.9, 1.05))
    bad = np.full((64, 64, 3), 0.2, np.float32)
    bad[3, 4, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        estimate_params_t24(bad, d_hi=(0.85, 0.9, 1.05))
    # empty mask: estimate on the whole frame, diagnostics undefined rather than wrong
    P = np.clip(0.18 * 2.0 ** rng.normal(0, 1, (64, 64, 3)), 1e-4, 3).astype(np.float32)
    est = estimate_params_t24(P, np.zeros((64, 64), bool), d_hi=(0.85, 0.9, 0.4), has_cam_matrix=False)
    e, d = render_t24(P, RenderParamsT24.from_dict(est["params"]), np.zeros((64, 64), bool))
    assert np.isfinite(e).all() and np.isnan(d["above_white"])


def test_t24_perceptual_v4_weights_and_non_flattening():
    from filmcolor import noref
    v4 = noref._load_t25_weights("t25_perceptual")
    assert v4 is not None and v4["version"] == "T25"
    cal_heads = v4["branches"]["cal"]
    assert "d_contrast" in cal_heads and cal_heads["d_contrast"]["bounds"][0] >= 0.0
    assert "d_sat" in cal_heads and cal_heads["d_sat"]["alpha"] > 0.0


def test_noref_audit_bugfixes():
    """Regression tests for B09-B13 and P07 in noref.py."""
    from filmcolor import noref

    # B09: _t24_density_feats handles gamma == 0 without ZeroDivisionError
    P = np.full((16, 16, 3), 0.25, dtype=np.float32)
    df, dg = noref._t24_density_feats(P, (0.0, 0.0, 0.0), (0.8, 0.8, 0.8), gamma=0.0)
    assert all(np.isfinite(v) for v in df.values()) and dg > 0.0

    # B10: estimate_params_v2 'sat' group handles neutral/B&W frames without ZeroDivisionError
    neutral = np.linspace(0.05, 0.85, 64 * 64, dtype=np.float32).reshape(64, 64)
    neutral_rgb = np.repeat(neutral[..., None], 3, axis=-1)
    res_v2 = noref.estimate_params_v2(neutral_rgb, cfg=noref.NOREF_V2)
    assert res_v2["status"] == "ok" and np.isfinite(res_v2["params"]["sat"])

    # B11: _t20_interior_crop handles thin 2px mask without returning empty array
    thin_mask = np.zeros((32, 160), dtype=bool)
    thin_mask[10:12, 10:150] = True
    crop_thin = noref._t20_interior_crop(np.full((32, 160, 3), 0.2, dtype=np.float32), thin_mask)
    assert crop_thin.shape[0] > 0 and crop_thin.shape[1] > 0

    # B12: estimate_params_t24 propagates v3_status and v3_flags
    const_frame = np.full((64, 64, 3), 0.2, dtype=np.float32)
    est_const = estimate_params_t24(const_frame, d_hi=(0.85, 0.9, 1.05))
    assert "v3_status" in est_const and "v3_flags" in est_const
    assert est_const["v3_status"] == "manual_review_required"

    # B13: _eval_et_trees and _weighted_median handle empty inputs safely
    assert noref._eval_et_trees([], np.zeros(5, dtype=np.float32)) == 0.0
    assert noref._weighted_median(np.array([], dtype=np.float64), np.array([], dtype=np.float64)) == 0.0

    # P07: load_look returns fresh copy so caller mutations do not pollute cache
    l1 = noref.load_look()
    l1["params"]["ev"] = 999.0
    l2 = noref.load_look()
    assert l2["params"]["ev"] != 999.0
