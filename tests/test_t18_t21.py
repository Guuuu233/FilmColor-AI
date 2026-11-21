"""Unit and integration tests for T18 decoupled inversion, T19 orthogonal C2 curve,
T20 smooth C1 physics estimator, and T21 hierarchical roll pooling."""
import numpy as np
import pytest

from filmcolor.invert import InversionParams, invert, invert_t18, invert_t18_from_positive, t18_channel_ratios
from filmcolor.noref import NOREF_T20, estimate_params_t20, extract_t20_features, pool_roll_params_t21
from filmcolor.pipeline import convert
from filmcolor.render import IDENTITY, MID_GREY, RenderParamsT19, render_t19, srgb_eotf
from test_scanner import scan, write_dng


def test_t18_decoupled_ratios_and_roundtrip():
    rr_cal, rg_cal, rb_cal, piv_cal = t18_channel_ratios((0.72, 1.0, 1.18), has_cam_matrix=True)
    assert rg_cal == 1.0 and piv_cal == 0.50
    assert np.isclose(rr_cal, 0.72, atol=1e-6) and np.isclose(rb_cal, 1.18, atol=1e-6)

    # CAL outlier guard smoothly pulls collapsed red slope toward 0.725
    rr_out, _, _, _ = t18_channel_ratios((0.35, 1.0, 1.18), has_cam_matrix=True)
    assert 0.55 < rr_out < 0.725

    # UNCAL continuous slope ratios
    rr_u, _, rb_u, piv_u = t18_channel_ratios((0.90, 1.0, 0.40), has_cam_matrix=False)
    assert piv_u == 0.62
    assert np.isclose(rr_u, 0.96 * 0.90, atol=1e-6)
    assert np.isclose(rb_u, 1.13 * 0.40, atol=1e-6)

    # Physical orange-mask TIFF without camera matrix routes to orange-mask T18 regime
    rr_ot, _, rb_ot, piv_ot = t18_channel_ratios((0.42, 0.90, 1.48), has_cam_matrix=False)
    assert piv_ot == 0.50 and rb_ot > 1.0 and rr_ot < 0.72

    # Exact equivalence: invert_t18(D) == invert_t18_from_positive(invert(D))
    rng = np.random.default_rng(1801)
    D = rng.uniform(0.05, 1.10, size=(32, 32, 3)).astype(np.float32)
    ip = InversionParams(base=(0.25, 0.25, 0.25), d_lo=(0.0, 0.0, 0.0), d_hi=(0.75, 0.95, 0.45), gamma=0.6)
    P_leg = invert(D, ip)
    for hc in (True, False):
        P_direct = invert_t18(D, ip, has_cam_matrix=hc)
        P_reinv = invert_t18_from_positive(P_leg, ip.d_lo, ip.d_hi, ip.gamma, has_cam_matrix=hc)
        assert np.allclose(P_direct, P_reinv, atol=2e-5, rtol=1e-4)


def test_t19_ortho_curve_properties_and_chunk_parity():
    # Midtone pivot invariance: 0.18 maps to srgb(0.18) regardless of contrast/shoulder/toe
    p_mid = np.full((4, 4, 3), MID_GREY, dtype=np.float32)
    for contrast, shoulder, toe in [(0.8, 0.1, -0.7), (1.3, 0.6, 0.0), (1.8, 0.9, 0.8)]:
        rp = RenderParamsT19(contrast=contrast, shoulder=shoulder, toe=toe)
        disp, _ = render_t19(p_mid, rp)
        assert np.allclose(srgb_eotf(disp), MID_GREY, atol=2e-5)

    # Strict monotonicity and chunked vs unchunked parity
    ramp = np.linspace(1e-4, 2.5, 600, dtype=np.float32)[:, None, None] * np.ones((1, 4, 3), dtype=np.float32)
    rp = RenderParamsT19(ev=0.25, temp=0.12, tint=-0.08, sat=0.92, contrast=1.35, shoulder=0.65, toe=-0.4)
    d_full, diag_full = render_t19(ramp, rp, chunk_rows=1024)
    d_chunk, diag_chunk = render_t19(ramp, rp, chunk_rows=64)
    assert np.allclose(d_full, d_chunk, atol=1e-7)
    assert diag_full == diag_chunk
    assert np.all(np.diff(d_full[:, 0, 1]) >= -1e-7)

    # Jacobian conditioning gate (< 15) on a synthetic scene
    rng = np.random.default_rng(1902)
    P = np.clip(0.18 * (2.0 ** rng.normal(0.0, 1.3, size=(300, 1, 3))), 1e-4, 4.0).astype(np.float32)
    names = ("ev", "temp", "tint", "sat", "contrast", "shoulder", "toe")
    scales = dict(ev=1.0, temp=0.25, tint=0.25, sat=0.25, contrast=0.25, shoulder=0.25, toe=0.50)
    p0 = dict(ev=0.1, temp=0.0, tint=0.0, sat=1.0, contrast=1.25, shoulder=0.35, toe=-0.2)
    base, _ = render_t19(P, RenderParamsT19(**p0))
    cols = []
    for k in names:
        h = 0.05 * scales[k]
        d = dict(p0)
        d[k] += h
        out, _ = render_t19(P, RenderParamsT19(**d))
        cols.append(((out - base) / h * scales[k]).ravel())
    J = np.stack(cols, 1)
    Jn = J / np.maximum(np.linalg.norm(J, axis=0), 1e-12)
    sv = np.linalg.svd(Jn, compute_uv=False)
    assert float(sv.max() / max(sv.min(), 1e-12)) < 15.0

    with pytest.raises(ValueError):
        RenderParamsT19(toe=1.5)


def test_t20_border_immunity_and_t21_roll_pooling():
    rng = np.random.default_rng(2001)
    img = np.clip(0.18 * (2.0 ** rng.normal(0.0, 1.1, size=(120, 160, 3))), 1e-4, 2.5).astype(np.float32)
    f_clean = extract_t20_features(img, mask=None, d_hi=(0.75, 0.90, 1.05))
    # Embed inside opaque black holder border; with mask selecting inner box, features match closely
    bordered = np.full((150, 200, 3), 1e-6, dtype=np.float32)
    bordered[15:135, 20:180] = img
    mask = np.zeros((150, 200), dtype=bool)
    mask[15:135, 20:180] = True
    f_masked = extract_t20_features(bordered, mask=mask, d_hi=(0.75, 0.90, 1.05))
    assert abs(f_clean["z50"] - f_masked["z50"]) < 0.05
    assert abs(f_clean["t_ge"] - f_masked["t_ge"]) < 0.05

    # Uniform flat frame safety + both branches
    flat = np.full((64, 64, 3), 0.22, dtype=np.float32)
    for hc in (True, False):
        est = estimate_params_t20(flat, d_hi=(0.75, 0.90, 0.45), has_cam_matrix=hc)
        assert est["status"] == "ok"
        assert est["branch"] == ("cal" if hc else "uncal")
        rp = RenderParamsT19.from_dict(est["params"])
        assert np.isfinite(rp.ev)
    est_ot = estimate_params_t20(flat, d_hi=(0.42, 0.90, 1.48), has_cam_matrix=False)
    assert est_ot["branch"] == "cal"

    # T21 roll pooling shrinks outlier toward roll consensus
    single = [{"ev": 0.1, "temp": 0.4, "tint": 0.0, "sat": 0.9, "contrast": 1.2, "shoulder": 0.6, "toe": 0.1, "matrix": IDENTITY}]
    assert pool_roll_params_t21(single) == single
    roll = [
        dict(single[0], temp=0.10, contrast=1.10),
        dict(single[0], temp=0.12, contrast=1.12),
        dict(single[0], temp=0.11, contrast=1.11),
        dict(single[0], temp=0.60, contrast=1.60),
    ]
    pooled = pool_roll_params_t21(roll)
    assert pooled[3]["temp"] < 0.60 and pooled[3]["contrast"] < 1.60


def test_pipeline_convert_with_noref_t20(tmp_path):
    rgb, irl = scan()
    dng = tmp_path / "s.dng"
    write_dng(dng, rgb, irl)
    # scan() produces an orange-masked synthetic negative (b_u ~ 1.09 > 0.80) -> physical "cal" regime
    out, sidecar = convert(dng, image_region=(20, 120, 200, 220), auto=NOREF_T20)
    assert out.shape == (rgb.shape[0], rgb.shape[1], 3)
    assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0
    assert sidecar["auto"]["branch"] == "cal"
    assert "toe" in sidecar["render"]["params"]

    # BlueNeg-ratio scanner DNG (d_hi_b / d_hi_g < 0.80) -> physical "uncal" regime
    out_u, sidecar_u = convert(dng, image_region=(20, 120, 200, 220), d_hi=(1.35, 1.49, 0.55), auto=NOREF_T20)
    assert out_u.shape == (rgb.shape[0], rgb.shape[1], 3)
    assert sidecar_u["auto"]["branch"] == "uncal"


def test_t19_fast_lut_parity_and_preview_t20_robustness(tmp_path):
    from filmcolor.preview import load_preview

    # 1. render_t19 fast LUT vs exact math parity (< 5e-6) and empty-input guard
    rng = np.random.default_rng(1921)
    P = np.clip(0.18 * (2.0 ** rng.normal(0.0, 1.4, size=(96, 128, 3))), 1e-5, 6.0).astype(np.float32)
    rp = RenderParamsT19(ev=-0.15, temp=0.14, tint=-0.09, sat=0.92, contrast=1.32, shoulder=0.55, toe=-0.35)
    e_fast, d_fast = render_t19(P, rp, fast=True)
    e_exact, d_exact = render_t19(P, rp, fast=False)
    assert np.allclose(e_fast, e_exact, atol=5e-6)
    assert d_fast == d_exact

    empty = np.zeros((0, 4, 3), dtype=np.float32)
    e_empty, d_empty = render_t19(empty, rp)
    assert e_empty.shape == empty.shape and np.isnan(d_empty["above_white"])

    # 2. invert_t18_from_positive rejects NaN/invalid gamma cleanly, pool_roll_params_t21([]) == []
    bad = P.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        invert_t18_from_positive(bad, (0.0, 0.0, 0.0), (0.8, 0.9, 1.1), 0.6)
    with pytest.raises(ValueError, match="invalid T18"):
        invert_t18_from_positive(P, (0.0, 0.0, 0.0), (0.8, 0.9, 1.1), -0.5)
    assert pool_roll_params_t21([]) == []

    # 3. pipeline.convert(..., auto=True/False) & preview.load_preview(..., auto=NOREF_T20)
    rgb, irl = scan()
    dng = tmp_path / "s.dng"
    write_dng(dng, rgb, irl)
    out_true, sc_true = convert(dng, image_region=(20, 120, 200, 220), auto=True)
    assert out_true.shape == (rgb.shape[0], rgb.shape[1], 3) and sc_true["auto"]["config"] == "noref_v2"
    with pytest.raises(TypeError, match="auto must be"):
        convert(dng, image_region=(20, 120, 200, 220), auto="bad")

    pv = load_preview(dng, max_side=160, image_region=(20, 120, 200, 220), auto=NOREF_T20, keep_full=True)
    assert pv.auto is not None and pv.auto["branch"] == "cal" and "toe" in pv.auto["params"]
    disp_pv = pv.render(pv.auto["params"])
    disp_st, stats = pv.render_stats(pv.auto["params"])
    np.testing.assert_array_equal(disp_pv, disp_st)
    assert 0.0 <= stats["below_black"] <= 1.0 and 0.0 <= stats["above_white"] <= 1.0
    crop = pv.render_crop((20, 120, 32, 32), pv.auto["params"])
    assert crop.shape == (32, 32, 3)

def test_t22_unified_entry_and_cli(tmp_path):
    import sys
    from filmcolor.noref import NOREF_T22
    from filmcolor.preview import load_preview

    scripts_dir = str((__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import convert as convert_cli

    rgb, irl = scan()
    dng = tmp_path / "t22.dng"
    write_dng(dng, rgb, irl)
    out, sc = convert(dng, image_region=(20, 120, 200, 220), auto=NOREF_T22)
    assert out.shape == (rgb.shape[0], rgb.shape[1], 3)
    assert sc["auto"]["config"] == "noref_t22" and sc["auto"]["version"] == "T22"
    pv = load_preview(dng, max_side=120, image_region=(20, 120, 200, 220), auto=NOREF_T22)
    assert pv.auto["config"] == "noref_t22" and pv.auto["version"] == "T22"
    rc = convert_cli.main([str(tmp_path / "cli_out"), str(dng), "--t22"])
    assert rc == 0


def test_t23_inversion_features_pipeline_and_cli(tmp_path):
    import sys
    from filmcolor.invert import invert_t23, invert_t23_from_positive
    from filmcolor.noref import NOREF_T23, estimate_params_t23, extract_t23_features, pool_roll_params_t23
    from filmcolor.preview import load_preview

    # 1. T23 UNCAL ratios and exact roundtrip invert_t23(D) == invert_t23_from_positive(invert(D))
    rr_u, rg_u, rb_u, piv_u = t18_channel_ratios((0.90, 1.0, 0.50), has_cam_matrix=False, version="t23")
    assert rg_u == 1.0 and piv_u == 0.62
    assert np.isclose(rr_u, 0.96 * 0.90, atol=1e-6)
    assert np.isclose(rb_u, 1.13 * 0.50, atol=1e-6)

    rng = np.random.default_rng(2301)
    D = rng.uniform(0.05, 1.10, size=(32, 32, 3)).astype(np.float32)
    ip = InversionParams(base=(0.25, 0.25, 0.25), d_lo=(0.0, 0.0, 0.0), d_hi=(0.75, 0.95, 0.45), gamma=0.6)
    P_leg = invert(D, ip)
    for hc in (True, False):
        assert np.allclose(invert_t23(D, ip, has_cam_matrix=hc),
                           invert_t23_from_positive(P_leg, ip.d_lo, ip.d_hi, ip.gamma, has_cam_matrix=hc),
                           atol=2e-5, rtol=1e-4)

    # 2. T23 feature extraction and estimator (CAL, UNCAL, orange-mask TIFF, flat frame)
    img = np.clip(0.18 * (2.0 ** rng.normal(0.0, 1.1, size=(120, 160, 3))), 1e-4, 2.5).astype(np.float32)
    feats = extract_t23_features(img, mask=None, d_hi=(0.75, 0.90, 1.05))
    for k in ("t_cneut", "n_cneut", "t_cbright", "t_sog6", "z_mean_log", "z_rms", "key_asym", "d_hi_g_sat", "log_span", "b_u"):
        assert np.isfinite(feats[k])
    flat = np.full((64, 64, 3), 0.22, dtype=np.float32)
    for hc, exp_br in ((True, "cal"), (False, "uncal")):
        est = estimate_params_t23(flat, d_hi=(0.75, 0.90, 0.45), has_cam_matrix=hc)
        assert est["status"] == "ok" and est["branch"] == exp_br
        assert np.isfinite(RenderParamsT19.from_dict(est["params"]).ev)

    # 3. Pipeline convert + preview + CLI --t23
    scripts_dir = str((__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import convert as convert_cli

    rgb, irl = scan()
    dng = tmp_path / "t23.dng"
    write_dng(dng, rgb, irl)
    out, sc = convert(dng, image_region=(20, 120, 200, 220), auto=NOREF_T23)
    assert out.shape == (rgb.shape[0], rgb.shape[1], 3)
    assert sc["auto"]["config"] == "noref_t23" and sc["auto"]["version"] == "T23"
    assert sc["auto"]["branch"] == "cal"

    out_u, sc_u = convert(dng, image_region=(20, 120, 200, 220), d_hi=(1.35, 1.49, 0.55), auto=NOREF_T23)
    assert out_u.shape == (rgb.shape[0], rgb.shape[1], 3) and sc_u["auto"]["branch"] == "uncal"

    pv = load_preview(dng, max_side=120, image_region=(20, 120, 200, 220), auto=NOREF_T23)
    assert pv.auto["config"] == "noref_t23" and pv.auto["version"] == "T23"
    assert pool_roll_params_t23([]) == []

    rc = convert_cli.main([str(tmp_path / "cli_out_t23"), str(dng), "--t23"])
    assert rc == 0
    # T23 UNCAL branch preserves exact T22 UNCAL parity while CAL uses upgraded T23 physics
    from filmcolor.noref import estimate_params_t22
    syn_u = np.clip(0.18 * (2.0 ** rng.normal(0.0, 1.0, size=(96, 128, 3))), 1e-4, 2.5).astype(np.float32)
    p22_u = estimate_params_t22(syn_u, d_hi=(0.85, 0.90, 0.40), has_cam_matrix=False)["params"]
    p23_u = estimate_params_t23(syn_u, d_hi=(0.85, 0.90, 0.40), has_cam_matrix=False)["params"]
    assert p22_u == p23_u
