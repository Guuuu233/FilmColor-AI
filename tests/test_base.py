import json

import numpy as np
import pytest

from filmcolor.base import estimate_base_auto, estimate_base_blocks, estimate_base_manual
from filmcolor.raw import linearize, saturation_mask
from synth import PATTERNS, black_for, film_scene, mosaic

BASE = (0.6, 0.5, 0.15)
WHITE = 16383.0


def prepared(kind, rgb, noise_dn=3.0):
    p = PATTERNS[kind]
    f = mosaic(rgb, p, black_for(p), WHITE, noise_dn=noise_dn)
    return linearize(f), f.colors, saturation_mask(f), p.shape[0]


def rel_err(a, b):
    return np.max(np.abs(np.asarray(a) / np.asarray(b) - 1))


@pytest.mark.parametrize("kind", PATTERNS)
def test_manual_recovers_base(kind):
    lin, col, sat, _ = prepared(kind, film_scene(base=BASE))
    est = estimate_base_manual(lin, col, sat, [(48, 24, 400, 180), (500, 24, 400, 180)])
    assert rel_err(est.rgb, BASE) < 0.002
    assert est.warnings == []
    assert set(est.diagnostics["region_spread"]) == {"R", "G", "B"}
    json.dumps(est.to_dict())  # sidecar-serialisable


def test_manual_warns_on_saturation_and_non_uniform_region():
    rgb = film_scene(base=BASE)
    rgb[100:110, 50:60] = 2.0
    lin, col, sat, _ = prepared("bayer", rgb)
    est = estimate_base_manual(lin, col, sat, [(90, 40, 40, 40)])
    assert any("saturated" in w for w in est.warnings)
    assert rel_err(est.rgb, BASE) < 0.002  # saturated samples excluded
    est2 = estimate_base_manual(lin, col, sat, [(300, 200, 200, 400)])  # straddles base and image
    assert any("not uniform" in w for w in est2.warnings)


def test_manual_warns_when_regions_disagree():
    rgb = film_scene(base=BASE)
    rgb[500:900, :240] *= 0.9  # e.g. uneven backlight
    lin, col, sat, _ = prepared("xtrans", rgb)
    est = estimate_base_manual(lin, col, sat, [(48, 24, 400, 180), (500, 24, 400, 180)])
    assert any("disagree" in w for w in est.warnings)


def test_manual_rejects_bad_regions():
    lin, col, sat, _ = prepared("bayer", film_scene(base=BASE))
    with pytest.raises(ValueError):
        estimate_base_manual(lin, col, sat, [])
    with pytest.raises(ValueError):
        estimate_base_manual(lin, col, sat, [(0, 0, 5000, 10)])


@pytest.mark.parametrize("kind", PATTERNS)
def test_auto_recovers_base(kind):
    lin, col, sat, n = prepared(kind, film_scene(base=BASE))
    est = estimate_base_auto(lin, col, sat, n)
    assert rel_err(est.rgb, BASE) < 0.005
    assert est.warnings == []
    # every kept block lies in the base strip
    assert all(l + w <= 1440 // 6 for _, l, _, w in est.regions)


@pytest.mark.parametrize("kind", PATTERNS)
def test_auto_ignores_saturated_light_leak(kind):
    rgb = film_scene(base=BASE)
    rgb[:96, 240:] = (1.4, 1.5, 1.2)  # direct backlight, clipped
    lin, col, sat, n = prepared(kind, rgb)
    est = estimate_base_auto(lin, col, sat, n)
    assert rel_err(est.rgb, BASE) < 0.005
    assert any("saturated" in w for w in est.warnings)


@pytest.mark.parametrize("kind", PATTERNS)
def test_auto_rejects_minority_unsaturated_leak_by_chromaticity(kind):
    rgb = film_scene(base=BASE)
    rgb[480:576, 720:816] = (0.8, 0.85, 0.6)  # brighter, bluer than base, not clipped
    lin, col, sat, n = prepared(kind, rgb)
    est = estimate_base_auto(lin, col, sat, n)
    assert rel_err(est.rgb, BASE) < 0.005
    assert est.diagnostics["chroma_outlier_frac"] > 0
    assert est.diagnostics["rejected_mean_rgb"][2] > 0.5


def test_auto_parameter_stability():
    lin, col, sat, n = prepared("xtrans", film_scene(base=BASE))
    ref = estimate_base_auto(lin, col, sat, n).rgb
    for kw in (dict(block=12), dict(block=48), dict(top_frac=0.005), dict(top_frac=0.02)):
        assert rel_err(estimate_base_auto(lin, col, sat, n, **kw).rgb, ref) < 0.005, kw


def test_auto_respects_crop():
    rgb = film_scene(base=BASE)
    dim = np.asarray(BASE, np.float32) * 0.5
    rgb[240:720, 480:960] = dim  # uniform but darker than base, inside the crop
    lin, col, sat, n = prepared("bayer", rgb)
    est = estimate_base_auto(lin, col, sat, n, crop=(0, 240, 960, 1200))  # excludes the base strip
    assert all(l >= 240 for _, l, _, _ in est.regions)
    assert rel_err(est.rgb, dim) < 0.005


def test_auto_raises_without_uniform_area():
    # per-pixel noise averages out inside a block, so vary the scene at block scale
    rng = np.random.default_rng(0)
    coarse = rng.uniform(0.05, 0.6, (10, 10, 3)).astype(np.float32)
    rgb = np.repeat(np.repeat(coarse, 24, 0), 24, 1)
    lin, col, sat, n = prepared("bayer", rgb)
    with pytest.raises(ValueError):
        estimate_base_auto(lin, col, sat, n)


def test_auto_is_deterministic():
    lin, col, sat, n = prepared("xtrans", film_scene(base=BASE))
    a = estimate_base_auto(lin, col, sat, n).to_dict()
    b = estimate_base_auto(lin, col, sat, n).to_dict()
    assert a == b


def test_auto_mode_seed_on_empty_bimodal_consensus():
    # Candidates with bimodal chromaticity: median falls between the modes where no points lie,
    # causing 2-round median consensus to be empty. mode_seed must trigger.
    hb, wb = 10, 10
    sp = np.zeros((hb, wb, 3), dtype=np.float32)
    # Cluster 1: upper half with G=0.5, log2(R/G)=+0.2, log2(B/G)=0.0
    rgb1 = np.array([0.5 * (2 ** 0.2), 0.5, 0.5], dtype=np.float32)
    # Cluster 2: lower half with G=0.5, log2(R/G)=-0.2, log2(B/G)=0.0
    rgb2 = np.array([0.5 * (2 ** -0.2), 0.5, 0.5], dtype=np.float32)
    sp[:5, :] = rgb1
    sp[5:, :] = rgb2
    sp_sat = np.zeros((hb, wb), dtype=bool)

    # With chroma_tol=0.05, median centre is [0, 0] which is 0.2 away from both clusters.
    est = estimate_base_blocks(sp, sp_sat, block=24, top_frac=1.0, max_rel_std=0.03, chroma_tol=0.05)
    assert est.diagnostics.get("mode_seed") is True
    assert "relaxed_uniformity" not in est.diagnostics
    # Exact argmax first tie selects cluster 1 (sp[:5, :])
    assert rel_err(est.rgb, rgb1) < 0.005
    assert est.diagnostics["n_kept"] == 24


def test_auto_relaxed_uniformity_on_dense_fake_base():
    # Fake base: dark shadow is very uniform (rel_std <= 0.03), real film base has mild variation
    # (rel_std ~ 0.08 > 0.03, <= 0.25). Relaxation triggers, picks real base.
    hb, wb = 20, 20
    sp = np.zeros((hb, wb, 3), dtype=np.float32)
    real_base = np.array(BASE, dtype=np.float32)
    for r in range(hb):
        for c in range(8):  # 40% of blocks
            sp[r, c] = real_base * (1.0 + 0.1 * (r % 3 - 1))

    dark_shadow = np.array([0.05, 0.04, 0.03], dtype=np.float32)
    sp[:, 8:] = dark_shadow
    sp_sat = np.zeros((hb, wb), dtype=bool)

    est = estimate_base_blocks(sp, sp_sat, block=24, top_frac=0.1, max_rel_std=0.03, chroma_tol=0.05)
    assert est.diagnostics.get("relaxed_uniformity") is True
    assert any("relaxed" in w for w in est.warnings)
    # Recovered RGB is from the real base, not the fake dark shadow
    assert rel_err(est.rgb, real_base) < 0.15
    assert all(c < 8 * 24 for _, c, _, _ in est.regions)
    assert "mode_seed" not in est.diagnostics


def test_auto_normal_path_preserves_metadata():
    lin, col, sat, n = prepared("bayer", film_scene(base=BASE))
    est = estimate_base_auto(lin, col, sat, n)
    assert rel_err(est.rgb, BASE) < 0.005
    assert "mode_seed" not in est.diagnostics
    assert "relaxed_uniformity" not in est.diagnostics
    assert not any("relaxed" in w for w in est.warnings)


def test_auto_relaxed_uniformity_respects_sat_and_crop():
    hb, wb = 20, 20
    sp = np.zeros((hb, wb, 3), dtype=np.float32)
    real_base = np.array(BASE, dtype=np.float32)
    for r in range(hb):
        for c in range(8):
            sp[r, c] = real_base * (1.0 + 0.1 * (r % 3 - 1))
    dark_shadow = np.array([0.05, 0.04, 0.03], dtype=np.float32)
    sp[:, 8:] = dark_shadow

    # 1. Saturation test: saturated real base blocks are excluded, even when relaxed
    sp_sat = np.zeros((hb, wb), dtype=bool)
    sp_sat[:, :8] = True
    est_sat = estimate_base_blocks(sp, sp_sat, block=24, top_frac=0.1, max_rel_std=0.03)
    assert all(c >= 8 * 24 for _, c, _, _ in est_sat.regions)
    assert "relaxed_uniformity" not in est_sat.diagnostics

    # 2. Crop test: real base blocks outside crop are excluded, even when relaxed
    sp_sat[:] = False
    crop = (0, 8 * 24, 20 * 24, 12 * 24)
    est_crop = estimate_base_blocks(sp, sp_sat, block=24, top_frac=0.1, max_rel_std=0.03, crop=crop)
    assert all(c >= 8 * 24 for _, c, _, _ in est_crop.regions)
    assert "relaxed_uniformity" not in est_crop.diagnostics
