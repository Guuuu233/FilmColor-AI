import json

import numpy as np
import pytest

from filmcolor.base import estimate_base_auto, estimate_base_manual
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
