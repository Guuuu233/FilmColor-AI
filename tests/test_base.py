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
    assert "bare_backlight_excluded" not in est.diagnostics
    assert not any("relaxed" in w for w in est.warnings)
    assert not any("bare backlight" in w for w in est.warnings)


def test_auto_bare_backlight_exclusion_triggers_on_bright_gap():
    """T4r: when bare backlight (Y >> film) is present, len(valid)>=100, p98Y > 5*p90Y,
    and mean(Y_valid <= 0.25*thr) > 0.88, bare backlight blocks are excluded and consensus is rerun."""
    hb, wb = 30, 30  # 900 blocks >= 100
    sp = np.zeros((hb, wb, 3), dtype=np.float32)
    sp_sat = np.zeros((hb, wb), dtype=bool)

    # Film base blocks: Y = 0.01
    film_base = np.array([0.012, 0.010, 0.008], dtype=np.float32)
    sp[:] = film_base

    # Bare backlight leak: 5x10 = 50 blocks (5.5% of area)
    # Interior 3x8 = 24 blocks are uniform (>= 10)
    backlight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    sp[10:15, 10:20] = backlight

    # Initial candidates will be top_frac of uniform blocks (backlight), thr ~ 1.0
    # 0.25 * thr = 0.25. Film blocks have Y ~ 0.01 <= 0.25 (94.4% > 88%).
    # p90 of Y_valid is ~0.01, p98 of Y_valid is 1.0 (> 5 * 0.01).
    # valid_film excludes backlight (Y <= 2.5 * p90Y = 0.025), reruns consensus.
    est = estimate_base_blocks(sp, sp_sat, block=24, top_frac=0.01, max_rel_std=0.03, chroma_tol=0.05)
    assert est.diagnostics.get("bare_backlight_excluded") is True
    assert any("bare backlight" in w for w in est.warnings)
    assert rel_err(est.rgb, film_base) < 0.01


def test_auto_bare_backlight_insufficient_samples_protection():
    """T4r: when len(valid) < 100, bare backlight check is bypassed."""
    hb, wb = 8, 8  # 64 blocks < 100
    sp = np.zeros((hb, wb, 3), dtype=np.float32)
    sp_sat = np.zeros((hb, wb), dtype=bool)

    film_base = np.array([0.06, 0.05, 0.04], dtype=np.float32)
    sp[:7, :] = film_base
    backlight = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    sp[7, :] = backlight

    est = estimate_base_blocks(sp, sp_sat, block=24, top_frac=0.05, max_rel_std=0.03, chroma_tol=0.05)
    assert "bare_backlight_excluded" not in est.diagnostics


def test_auto_bare_backlight_untriggered_on_normal_scene():
    """T4r: when p98Y <= 5*p90Y or mean(Y <= 0.25*thr) <= 0.88, bare backlight check does not trigger."""
    hb, wb = 20, 20
    sp = np.zeros((hb, wb, 3), dtype=np.float32)
    sp_sat = np.zeros((hb, wb), dtype=bool)

    # Gradient of film base from 0.4 to 0.6: p98 is never 5x p90
    base_grad = np.linspace(0.4, 0.6, hb * wb, dtype=np.float32).reshape(hb, wb, 1)
    sp[...] = base_grad * np.array([1.2, 1.0, 0.5], dtype=np.float32)

    est = estimate_base_blocks(sp, sp_sat, block=24, top_frac=0.05, max_rel_std=0.03, chroma_tol=0.05)
    assert "bare_backlight_excluded" not in est.diagnostics
    assert not any("bare backlight" in w for w in est.warnings)



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


def test_t4w_bare_backlight_preserves_true_orange_mask_border_on_dense_overexposed_frame():
    """T4w (fixes A1004.DNG): dense overexposed negative with transparent orange mask border (R > G and B/G < 0.43).
    When film border has true orange mask chromaticity (e.g. RGB = [0.102, 0.098, 0.039]), bare backlight exclusion
    must NOT trigger, preserving the true film base."""
    hb, wb = 30, 30  # 900 blocks
    sp = np.zeros((hb, wb, 3), dtype=np.float32)
    sp_sat = np.zeros((hb, wb), dtype=bool)

    # Synthetic overexposed negative:
    # 33% black frame / holder (Y ~ 0.0001)
    # ~61% dense overexposed film image ([0.015, 0.009, 0.003], Y ~ 0.01)
    # 5.5% true orange mask film border ([0.102, 0.098, 0.039], R > G and B/G = 0.398 < 0.43)
    dense_image = np.array([0.015, 0.009, 0.003], dtype=np.float32)
    black_holder = np.array([0.0001, 0.0001, 0.0001], dtype=np.float32)
    orange_base = np.array([0.102, 0.098, 0.039], dtype=np.float32)

    sp[:] = dense_image
    sp[:10, :] = black_holder  # 300 blocks (33%)
    sp[10:15, 10:20] = orange_base  # 50 blocks (5.5%)

    est = estimate_base_blocks(sp, sp_sat, block=24, top_frac=0.01, max_rel_std=0.03, chroma_tol=0.05)
    assert est.diagnostics.get("bare_backlight_excluded") is not True
    assert not any("bare backlight" in w for w in est.warnings)
    assert rel_err(est.rgb, orange_base) < 0.01


def test_t4w_bare_backlight_still_excludes_neutral_or_cool_light_leak():
    """T4w: same geometry as above, but the bright gap is a true neutral/cool bare backlight leak (R <= G or B/G >= 0.43).
    Verifies bare_backlight_excluded is True and film base falls back to the film area."""
    hb, wb = 30, 30  # 900 blocks
    sp = np.zeros((hb, wb, 3), dtype=np.float32)
    sp_sat = np.zeros((hb, wb), dtype=bool)

    dense_image = np.array([0.015, 0.009, 0.003], dtype=np.float32)
    black_holder = np.array([0.0001, 0.0001, 0.0001], dtype=np.float32)
    # Neutral/cool leak: RGB = [0.117, 0.119, 0.057], B/G = 0.479 >= 0.43 and R <= G
    backlight_leak = np.array([0.117, 0.119, 0.057], dtype=np.float32)

    sp[:] = dense_image
    sp[:10, :] = black_holder  # 300 blocks (33%)
    sp[10:15, 10:20] = backlight_leak  # 50 blocks (5.5%)

    est = estimate_base_blocks(sp, sp_sat, block=24, top_frac=0.01, max_rel_std=0.03, chroma_tol=0.05)
    assert est.diagnostics.get("bare_backlight_excluded") is True
    assert any("bare backlight" in w for w in est.warnings)
    assert rel_err(est.rgb, dense_image) < 0.01


def test_t4x_near_saturated_sony_craw_blocks_excluded_but_unit_white_preserved():
    """T4x (fixes A262.arw, A263.arw): Sony lossy cRAW clips highlight quantization just below white level (~0.9993).
    1) When near-saturated blocks (0.9985 < sp_max < 1.0) appear alongside real film base and remaining valid blocks >= 100:
       - estimate_base_blocks excludes the near-saturated blocks and logs near_saturated_blocks_excluded > 0.
       - Base is accurately estimated from the real film base.
    2) Unit white (sp == 1.0) blocks are preserved (sp_max < 1.0 condition), keeping exact synthetic fixtures intact.
    3) When remaining valid blocks < 100: near_saturated blocks are NOT excluded (failsafe protection)."""
    hb, wb = 30, 30  # 900 blocks
    block = 24

    # 1. Near-saturated Sony cRAW blocks (e.g. 5% = 45 blocks of [0.9993, 0.9991, 0.9993])
    # Real orange base blocks (e.g. 100 blocks of [0.14, 0.28, 0.13])
    # Dense film image (755 blocks of [0.02, 0.04, 0.02])
    sp1 = np.zeros((hb, wb, 3), dtype=np.float32)
    sp_sat1 = np.zeros((hb, wb), dtype=bool)

    dense_img = np.array([0.02, 0.04, 0.02], dtype=np.float32)
    real_base = np.array([0.14, 0.28, 0.13], dtype=np.float32)
    sony_craw_clip = np.array([0.9993, 0.9991, 0.9993], dtype=np.float32)

    sp1[:] = dense_img
    sp1[:3, :] = sony_craw_clip  # 90 blocks of near-sat (sp_max ~ 0.9993)
    sp1[5:10, :20] = real_base   # 100 blocks of true film base

    est1 = estimate_base_blocks(sp1, sp_sat1, block=block, top_frac=0.05, max_rel_std=0.03, chroma_tol=0.05)
    assert est1.diagnostics.get("near_saturated_blocks_excluded", 0) == 90
    assert rel_err(est1.rgb, real_base) < 0.02
    assert max(est1.rgb) < 0.50  # Must not pick the near-saturated ~0.9993 blocks

    # 2. Synthetic ideal unit-white fixture: sp == 1.0 is preserved (not in (0.9985, 1.0))
    # When all valid blocks are ideal 1.0 white base, it should not be excluded by near_sat.
    sp2 = np.ones((hb, wb, 3), dtype=np.float32)
    sp_sat2 = np.zeros((hb, wb), dtype=bool)

    est2 = estimate_base_blocks(sp2, sp_sat2, block=block, top_frac=0.10, max_rel_std=0.03, chroma_tol=0.05)
    assert "near_saturated_blocks_excluded" not in est2.diagnostics
    assert np.allclose(est2.rgb, [1.0, 1.0, 1.0], atol=1e-3)

    # 3. Failsafe protection: if remaining valid blocks < 100, near_sat exclusion does NOT trigger
    sp3 = np.zeros((hb, wb, 3), dtype=np.float32)
    sp_sat3 = np.zeros((hb, wb), dtype=bool)
    # 70 blocks of near-sat ([0.9993, 0.9991, 0.9993]) and 50 blocks of real base ([0.14, 0.28, 0.13])
    # Total valid blocks = 120, so valid & ~near_sat = 50 < 100 blocks
    sp3[10:14, 10:20] = sony_craw_clip   # 40 blocks of 3x3 uniform interior
    sp3[16:20, 10:20] = real_base        # 40 blocks of 3x3 uniform interior
    # Remaining blocks are 0 (invalid)

    est3 = estimate_base_blocks(sp3, sp_sat3, block=block, top_frac=0.20, max_rel_std=0.03, chroma_tol=0.20)
    assert "near_saturated_blocks_excluded" not in est3.diagnostics


def _t5b_clipped_light_blocks(near_sat=True):
    """T5b (A1298 / A1299 / A1300, Sony ILCE-7RM4): 50 x 50 blocks of a textured dense image, an orange film border
    (6 columns, [0.171, 0.212, 0.085]) and clipped backlight. With ``near_sat`` its top 8 rows sit at the 14-bit
    ceiling (max channel 0.9988, inside T4x's near-saturated band) and two rows just below it (0.9979) survive."""
    rng = np.random.default_rng(5)
    film_base = np.array([0.171, 0.212, 0.085], dtype=np.float32)
    sp = (film_base * rng.uniform(0.15, 0.6, (50, 50, 1))).astype(np.float32)
    sp[10:, :6] = film_base
    sp[:10, :20] = [0.456, 0.9979, 0.775]
    if near_sat:
        sp[:8] = [0.4575, 0.9988, 0.7765]
    return sp, np.zeros((50, 50), dtype=bool), film_base


def test_t5b_clipped_base_excluded():
    """T5b: near-saturated exclusion leaves clipped light (max channel 0.9979) as the brightest uniform blocks; a camera
    RAW base with a channel >= CLIPPED_BASE_MAX (0.99) is re-estimated from the blocks below 2.5x the 90th percentile.
    T5a returned the clipped base here (G 0.9979)."""
    from filmcolor.base import CLIPPED_BASE_MAX
    sp, sp_sat, film_base = _t5b_clipped_light_blocks()
    est = estimate_base_blocks(sp, sp_sat, block=24)
    assert CLIPPED_BASE_MAX == 0.99
    assert est.diagnostics.get("near_saturated_blocks_excluded") == 400
    assert est.diagnostics.get("clipped_base_excluded") is True
    assert "bare_backlight_excluded" not in est.diagnostics
    assert any("clipped light" in w for w in est.warnings)
    assert rel_err(est.rgb, film_base) < 1e-5 and max(est.rgb) < CLIPPED_BASE_MAX


def test_t5b_clipped_base_not_triggered_without_near_sat():
    """T5b: without near-saturated blocks the rule does not fire, even though the base is close to 1; the base is
    bit-identical to T5a's (the float32 clipped-light blocks themselves)."""
    sp, sp_sat, _ = _t5b_clipped_light_blocks(near_sat=False)
    est = estimate_base_blocks(sp, sp_sat, block=24)
    assert "near_saturated_blocks_excluded" not in est.diagnostics
    assert "clipped_base_excluded" not in est.diagnostics and "bare_backlight_excluded" not in est.diagnostics
    assert est.rgb == tuple(float(v) for v in np.float32([0.456, 0.9979, 0.775]))
    assert not any("clipped light" in w for w in est.warnings)


def test_t5b_scanner_path_unchanged():
    """T5b: the scanner path (exclude_near_saturated=False) never counts near-saturated blocks, so the rule cannot
    fire; the 0.9988 blocks stay the base bit-identically as in T5a, through estimate_base_scanner too."""
    from filmcolor.scanner import ScannerFrame, estimate_base_scanner
    sp, sp_sat, _ = _t5b_clipped_light_blocks()
    est = estimate_base_blocks(sp, sp_sat, block=24, exclude_near_saturated=False)
    assert "near_saturated_blocks_excluded" not in est.diagnostics and "clipped_base_excluded" not in est.diagnostics
    assert est.rgb == tuple(float(v) for v in np.float32([0.4575, 0.9988, 0.7765]))
    rgb = np.repeat(np.repeat(sp, 24, 0), 24, 1)
    fr = ScannerFrame(rgb=rgb, sat=np.zeros(rgb.shape[:2], dtype=bool), black=0.0, white=65535.0, meta={})
    assert estimate_base_scanner(fr, block=24).to_dict() == est.to_dict()


def test_t5c_scanner_white_ceiling_excluded():
    """T5c (A1086.TIF): under exclude_near_saturated=False (scanner path), when sp_sat.any() is True and the brightest
    uniform consensus blocks sit at >= 0.995 in all three channels (e.g. 65534/65535 = 0.999985 bare light in sprocket
    holes), clipped_base_excluded triggers and recovers the orange mask below, while an orange mask with R = 0.999 and
    B = 0.43 is untouched."""
    rng = np.random.default_rng(7)
    orange_base = np.array([0.936, 0.519, 0.318], dtype=np.float32)
    sp = (orange_base * rng.uniform(0.15, 0.65, (50, 50, 1))).astype(np.float32)
    sp[10:22, :12] = orange_base  # 144 uniform orange-mask blocks
    sp[:5, :20] = 65534.0 / 65535.0  # 100 uniform bare-white blocks one code below white in all 3 channels
    sp_sat = np.zeros((50, 50), dtype=bool)
    sp_sat[:2, 20:30] = True  # saturated sprocket-hole blocks

    est = estimate_base_blocks(sp, sp_sat, block=24, exclude_near_saturated=False)
    assert est.diagnostics.get("clipped_base_excluded") is True
    assert "bare_backlight_excluded" not in est.diagnostics
    assert rel_err(est.rgb, orange_base) < 1e-5

    # Single-channel near-saturated orange mask (R = 0.999, G = 0.53, B = 0.43) alongside sp_sat must NOT trigger
    sp_orange_hi = (orange_base * rng.uniform(0.15, 0.65, (50, 50, 1))).astype(np.float32)
    hi_mask = np.array([0.999, 0.53, 0.43], dtype=np.float32)
    sp_orange_hi[:6, :20] = hi_mask
    est_hi = estimate_base_blocks(sp_orange_hi, sp_sat, block=24, exclude_near_saturated=False)
    assert "clipped_base_excluded" not in est_hi.diagnostics
    assert rel_err(est_hi.rgb, hi_mask) < 1e-5


def test_t5g_natural_roll_base_level_diff_and_cyan_shadow_relaxation():
    """T5g:
    1) On monochromatic / B&W rolls (chroma_part < 0.02), natural_roll_base blends toward d_lo_dq according to
       level_diff in [ramp_lo, ramp_hi], capped at d_lo_b1 + max_black_pull_d (B003/B013/B052/B060).
    2) On C-41 colour negatives where the initial uniform candidates are cyan scene shadows (rgb_R >= 1.8 * rgb_G,
       rgb_B >= 1.10 * rgb_G) and p98 exceeds the initial estimate in R and G, estimate_base_blocks relaxes max_rel_std
       to 0.25 to recover the brighter orange-masked border (A1604.dng, A1606.dng, A1609.dng).
    """
    from filmcolor.base import estimate_base_blocks, natural_roll_base

    # 1. Monochromatic roll_nat level_diff ramp (chroma_part == 0.0, level_diff == 0.12 >= ramp_hi):
    roll_rgb = (0.16, 0.34, 0.30)
    own_rgb = tuple(x * (10.0 ** -0.12) for x in roll_rgb)
    p01 = [float(-np.log10(x)) for x in own_rgb]
    b_eff, info = natural_roll_base(roll_rgb, p01, own_rgb=own_rgb, softness=0.5)
    assert info["b1_weight"] == pytest.approx(0.175)
    assert info["d_lo_used"] == pytest.approx(info["d_lo_b1"] + 0.04)
    assert b_eff[1] < roll_rgb[1]

    # When level_diff < ramp_lo (0.02 < 0.04), b1_weight remains 1.0:
    own_near = tuple(x * (10.0 ** -0.02) for x in roll_rgb)
    _, info_near = natural_roll_base(roll_rgb, p01, own_rgb=own_near, softness=0.5)
    assert info_near["b1_weight"] == pytest.approx(1.0)
    assert info_near["d_lo_used"] == pytest.approx(info_near["d_lo_b1"])

    # 2. Cyan scene shadow relaxation in estimate_base_blocks:
    rng = np.random.default_rng(7)
    sp = np.empty((20, 20, 3), dtype=np.float32)
    # 95% uniform interior cyan shadow: R=0.28, G=0.10, B=0.133 (B/G = 1.33 >= 1.10, R/G = 2.80 >= 1.80)
    sp[:, :] = np.array([0.28, 0.10, 0.133], dtype=np.float32) * (1.0 + 0.005 * rng.standard_normal((20, 20, 1)).astype(np.float32))
    # 5% textured orange-masked border on row 1: R=0.38, G=0.14, B=0.14 (rel_std ~ 0.08 > 0.03, <= 0.25)
    checker = np.where((np.arange(20) % 2) == 0, 1.08, 0.92).astype(np.float32)[:, None]
    sp[1, :] = np.array([0.38, 0.14, 0.14], dtype=np.float32) * checker
    sp_sat = np.zeros((20, 20), dtype=bool)
    est = estimate_base_blocks(sp, sp_sat, block=24, exclude_near_saturated=False)
    assert est.diagnostics.get("relaxed_uniformity") is True
    assert est.rgb[0] > 0.34 and est.rgb[1] > 0.125




def test_auto_relaxed_uniformity_on_single_channel_underestimation():
    """Borderless negatives where smooth sky/clouds mask textured thin foreground (e.g. B channel >=1.55x brighter)
    must trigger relaxed_uniformity so the underestimated channel is not truncated to zero."""
    rng = np.random.default_rng(42)
    hb, wb = 20, 20
    sp = np.zeros((hb, wb, 3), dtype=np.float64)
    # Top half: smooth cloud-like region (low rel_std) with underestimated B channel
    sp[:10, :, :] = [0.10, 0.06, 0.022]
    # Bottom half: textured thin foreground (higher rel_std) with much brighter G/B
    fg = np.array([0.13, 0.11, 0.065])[None, None, :]
    noise = rng.uniform(0.88, 1.12, size=(10, wb, 3))
    sp[10:, :, :] = fg * noise
    sp_sat = np.zeros((hb, wb), dtype=bool)
    est = estimate_base_blocks(sp, sp_sat, block=24, top_frac=0.05, max_rel_std=0.03, chroma_tol=0.05)
    assert est.diagnostics.get("relaxed_uniformity") is True
    assert est.rgb[2] > 0.05  # blue channel recovered from textured thin foreground
