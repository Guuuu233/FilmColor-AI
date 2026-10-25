import numpy as np
import pytest

from filmcolor.base import estimate_base_roll

BASE = np.array([0.19, 0.19, 0.076])  # unmasked base at t_ref = 1/30; legacy contracts disable orange prior


def roll(floor_frames=4, shadow_frames=4, seed=0):
    """Frames whose interior minimum is either the base (up to small extra density) or a coloured shadow."""
    rng = np.random.default_rng(seed)
    cand, t = {}, {}
    for i in range(floor_frames + shadow_frames):
        ti = float(rng.choice([1 / 25, 1 / 30, 1 / 40, 1 / 50, 1 / 60]))
        if i < floor_frames:
            d = np.full(3, 0.01 * i)                      # neutral extra density: fog / near-floor shadow
        else:
            d = np.array([0.15, 0.25, 0.35]) + 0.05 * i   # coloured scene density
        cand[f"f{i}"] = tuple(BASE * 10 ** -d * ti * 30)
        t[f"f{i}"] = ti
    return cand, t


def test_recovers_base_and_normalises_exposure():
    cand, t = roll()
    r = estimate_base_roll(cand, t, t_ref=1 / 30, orange_mask_prior=False)
    assert sorted(r.members) == ["f0", "f1", "f2", "f3"]
    assert r.used == ["f0", "f1", "f2"]
    np.testing.assert_allclose(r.rgb_norm, BASE * 10 ** -0.01, rtol=1e-9)  # median of the 3 most transparent
    assert not r.warnings


def test_for_exposure_scales_linearly():
    cand, t = roll()
    r = estimate_base_roll(cand, t, t_ref=1 / 30, orange_mask_prior=False)
    np.testing.assert_allclose(r.for_exposure(1 / 60), np.array(r.rgb_norm) / 2)
    with pytest.raises(ValueError):
        r.for_exposure(0)


def test_coloured_shadow_close_in_level_is_rejected_by_chroma():
    cand, t = roll(floor_frames=3, shadow_frames=0)
    cand["warm"] = tuple(BASE * 10 ** -np.array([0.0, 0.05, 0.12]))  # within level window, wrong colour
    t["warm"] = 1 / 30
    r = estimate_base_roll(cand, t, t_ref=1 / 30, orange_mask_prior=False)
    assert "warm" not in r.members
    assert r.diagnostics["frames"]["warm"]["member"] is False


def test_warns_when_few_frames_reach_floor():
    cand, t = roll(floor_frames=1, shadow_frames=5)
    r = estimate_base_roll(cand, t, orange_mask_prior=False)
    assert r.members == ["f0"]
    assert any("minimum-density cluster" in w for w in r.warnings)


def test_default_t_ref_is_median_exposure():
    cand, t = roll()
    r = estimate_base_roll(cand, t, orange_mask_prior=False)
    assert r.t_ref == float(np.median(list(t.values())))


def test_orange_roll_consensus_rejects_brighter_colourless_flare():
    base = np.array([0.34, 0.20, 0.076])
    cand = {f"orange{i}": tuple(base * 10 ** (-0.01 * i)) for i in range(6)}
    cand.update({f"flare{i}": (0.28, 0.28, 0.18) for i in range(3)})
    r = estimate_base_roll(cand, dict.fromkeys(cand, 1.0))
    assert r.diagnostics["orange_consensus"]
    assert all(n.startswith("orange") for n in r.members)
    np.testing.assert_allclose(r.rgb_norm, base * 10 ** -0.01)


@pytest.mark.parametrize("n_plain", [3, 5])
def test_minority_orange_frames_do_not_override_an_unmasked_roll(n_plain):
    cand = {f"plain{i}": (0.20, 0.20, 0.15) for i in range(n_plain)}
    cand["orange"] = (0.25, 0.18, 0.06)
    r = estimate_base_roll(cand, dict.fromkeys(cand, 1.0))
    assert not r.diagnostics["orange_consensus"]
    assert r.members == [f"plain{i}" for i in range(n_plain)]


def test_natural_black_pull_is_limited_without_losing_anti_truncation():
    from filmcolor.base import natural_roll_base, split_log_base

    base = (0.34, 0.20, 0.076)
    level, chroma = split_log_base(base)
    b1 = -level
    for extra in (0.30, -0.10):
        # The channel quantiles have a common floor; a coloured own candidate requests DQ fusion.
        p01 = b1 + extra - chroma
        _, diag = natural_roll_base(base, p01, own_rgb=(0.10, 0.10, 0.10))
        if extra > 0:
            assert diag["d_lo_used"] == pytest.approx(b1 + 0.04)
        else:
            assert diag["d_lo_used"] == pytest.approx(b1 + extra - 0.02)
            assert diag["b1_weight"] == 0


@pytest.mark.parametrize("bad", ["nan", "zero_rgb", "zero_t", "mismatch", "empty"])
def test_rejects_invalid_input(bad):
    cand, t = roll()
    if bad == "nan":
        cand["f0"] = (np.nan, 0.1, 0.1)
    elif bad == "zero_rgb":
        cand["f0"] = (0.0, 0.1, 0.1)
    elif bad == "zero_t":
        t["f0"] = 0.0
    elif bad == "mismatch":
        t.pop("f0")
    else:
        cand, t = {}, {}
    with pytest.raises(ValueError):
        estimate_base_roll(cand, t, orange_mask_prior=False)
