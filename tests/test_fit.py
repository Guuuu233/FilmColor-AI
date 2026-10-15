import numpy as np
import pytest

from filmcolor.fit import DEFAULTS, MODES, SCALES, fit_reference, param_conditioning, render_lab
from filmcolor.render import PER_IMAGE, RenderParams


def scene(n=6000, seed=0):
    """Scene-linear pixels: grey ramp up to 2x white plus varied colours."""
    rng = np.random.default_rng(seed)
    grey = np.geomspace(2e-3, 2, n // 3)[:, None].repeat(3, 1)
    colour = rng.uniform(0, 1.2, (n - n // 3, 3)) * rng.uniform(0.1, 1, (n - n // 3, 1))
    return np.concatenate([grey, colour]).astype(np.float32)


TRUE6 = dict(ev=0.0, temp=0.2, tint=-0.1, sat=1.3, contrast=1.25, midtone=0.24, shoulder=0.35)


def test_six_parameter_fit_recovers_known_parameters():
    P = scene()
    ref, _ = render_lab(P[None], RenderParams(**TRUE6))
    r = fit_reference(P, ref[0], "6", n_starts=2)
    assert r["de2000_median"] < 0.1
    for k in MODES["6"]["free"]:
        assert abs(r["params"][k] - TRUE6[k]) < 0.1 * SCALES[k] + 0.01, (k, r["params"][k])
    assert r["params"]["ev"] == 0


def test_seven_parameter_fit_matches_reference_output():
    P = scene()
    true = TRUE6 | dict(ev=0.4)
    ref, _ = render_lab(P[None], RenderParams(**true))
    r = fit_reference(P, ref[0], "7", n_starts=2)
    assert r["de2000_median"] < 0.2  # the image is matched; ev/midtone individually may trade off


def test_regularisation_pulls_unneeded_parameters_to_defaults():
    P = scene()
    ref, _ = render_lab(P[None], RenderParams())
    r = fit_reference(P, ref[0], "7reg", n_starts=2)
    for k in PER_IMAGE:
        assert abs(r["params"][k] - DEFAULTS[k]) < 0.05 * SCALES[k], k
    heavy = fit_reference(P, render_lab(P[None], RenderParams(**TRUE6))[0][0], "7", n_starts=1, lam=50.0)
    assert all(abs(heavy["params"][k] - DEFAULTS[k]) < 0.2 * SCALES[k] for k in PER_IMAGE)


def test_fit_respects_bounds_and_reports_bound_hits():
    P = scene()
    ref, _ = render_lab(P[None], RenderParams(sat=2.0))
    r = fit_reference(P, ref[0], "6", n_starts=1)
    assert r["params"]["sat"] <= 2.0
    assert "sat" in r["at_bound"]


def test_fit_is_deterministic():
    P = scene(2000)
    ref, _ = render_lab(P[None], RenderParams(**TRUE6))
    a = fit_reference(P, ref[0], "6", n_starts=2)
    b = fit_reference(P, ref[0], "6", n_starts=2)
    assert a == b


def test_fit_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        fit_reference(scene(100), np.zeros((50, 3)), "6")
    with pytest.raises(KeyError):
        fit_reference(scene(100), np.zeros((100, 3)), "5")


def test_param_conditioning_flags_correlated_parameters():
    c = param_conditioning(scene(), RenderParams(**TRUE6))
    assert 0 < c["sv_ratio"] < 1
    assert c["cosine"]["ev/midtone"] > c["cosine"]["temp/sat"]


def test_experimental_bounds_are_temporary_and_used_by_the_fitter():
    from filmcolor.render import BOUNDS, experimental_bounds

    P = scene()
    ref, _ = render_lab(P[None], RenderParams(contrast=2.0))
    with experimental_bounds(contrast=(0.5, 3.0)):
        assert BOUNDS["contrast"] == (0.5, 3.0)
        RenderParams(contrast=2.8)
        r = fit_reference(P, ref[0], "6", n_starts=1)
        assert "contrast" not in r["at_bound"]
    assert BOUNDS["contrast"] == (0.5, 2.0)
    with pytest.raises(ValueError):
        RenderParams(contrast=2.8)
    with pytest.raises(ValueError):
        with experimental_bounds(exposure=(0, 1)):
            pass


def test_scale_override_changes_penalty_and_is_recorded():
    P = scene(3000)
    ref, _ = render_lab(P[None], RenderParams(**TRUE6))
    a = fit_reference(P, ref[0], "7reg", n_starts=1)
    b = fit_reference(P, ref[0], "7reg", n_starts=1, scales=dict(midtone=0.01))
    assert b["scales"]["midtone"] == 0.01 and a["scales"]["midtone"] == SCALES["midtone"]
    assert abs(b["params"]["midtone"] - 0.18) < abs(a["params"]["midtone"] - 0.18)  # tighter scale, stronger pull


def test_conditioning_reports_scaled_singular_values():
    c = param_conditioning(scene(), RenderParams(**TRUE6))
    assert len(c["sv_scaled"]) == 7 and all(v >= 0 for v in c["sv_scaled"])


def test_highlight_penalty_zero_below_white_and_masked_where_reference_white():
    from filmcolor.fit import highlight_penalty

    assert highlight_penalty(np.full((10, 3), 0.99)) == 0
    over = np.full((10, 3), 2.0)  # one stop past white in every channel
    p1 = highlight_penalty(over)
    assert p1 == pytest.approx(3 * (np.sqrt(1 + 0.05 ** 2) - 0.05))
    assert highlight_penalty(np.full((10, 3), 4.0)) > p1
    allowed = np.zeros((10, 3), bool)
    allowed[:5] = True
    assert highlight_penalty(over, allowed) == pytest.approx(p1 / 2)
    allowed[:, 0] = True  # one clipped channel allowed everywhere
    assert highlight_penalty(over, allowed) == pytest.approx(p1 / 2 * 2 / 3)


def test_highlight_constraint_trades_midtone_match_for_less_clipping():
    from filmcolor.render import experimental_bounds

    P = scene(3000)
    ref, _ = render_lab(P[None], RenderParams(ev=1.5, shoulder=1.0))  # bright, highlights compressed, not white
    # without shoulder and midtone the fitter can only brighten with exposure, which clips highlights
    with experimental_bounds(shoulder=(0.0, 0.0), midtone=(0.18, 0.18)):
        off = fit_reference(P, ref[0], "7", n_starts=2, highlight_weight=0.0)
        on = fit_reference(P, ref[0], "7", n_starts=2)
    assert on["highlight"]["weight"] == 10.0 and off["highlight"]["ref_clipped_frac"] == 0
    assert off["highlight"]["clipped_frac"] > 0.3
    assert on["highlight"]["clipped_frac"] < 0.75 * off["highlight"]["clipped_frac"]
    assert on["params"]["ev"] < off["params"]["ev"]
    assert on["de2000_median"] > off["de2000_median"]  # the trade is paid in average colour match


def test_clipping_allowed_where_reference_clips():
    from filmcolor.fit import highlight_penalty, reference_clipped

    P = scene(3000)
    # grey ramp above 0.5 white, and single channels of bright colours clipped, in the reference
    ref, _, pre = render_lab(P[None], RenderParams(ev=1.0), with_pre_clip=True)
    assert highlight_penalty(pre[0], reference_clipped(ref[0])) == 0  # the truth is not penalised
    r = fit_reference(P, ref[0], "7", n_starts=2)
    assert r["highlight"]["ref_clipped_frac"] > 0.05
    assert abs(r["params"]["ev"] - 1.0) < 0.05 and r["de2000_median"] < 0.1


def test_highlight_weight_must_be_non_negative():
    with pytest.raises(ValueError):
        fit_reference(scene(100), np.zeros((100, 3)), "7", highlight_weight=-1)


def test_clip_area_counts_shallow_wide_clipping_that_the_depth_penalty_misses():
    from filmcolor.fit import clip_area, highlight_penalty

    below = np.full((100, 3), 0.999)
    assert clip_area(below) == 0
    shallow = np.full((100, 3), 2 ** 0.06)  # every pixel just past the 0.05-stop ramp
    assert clip_area(shallow) == pytest.approx(1.0)
    assert highlight_penalty(shallow) < 0.1  # the depth penalty barely sees it
    half = shallow.copy()
    half[50:] = 0.5
    assert clip_area(half) == pytest.approx(0.5)
    allowed = np.zeros((100, 3), bool)
    allowed[:25] = True
    assert clip_area(half, allowed) == pytest.approx(0.25)
    one = np.full((100, 3), 0.5)
    one[:, 2] = 2 ** 0.025  # half way up the ramp in one channel: smoothstep(0.5) = 0.5
    assert clip_area(one) == pytest.approx(0.5)


def test_area_weight_must_be_non_negative():
    with pytest.raises(ValueError):
        fit_reference(scene(100), np.zeros((100, 3)), "7", area_weight=-1)


def test_fit_config_v1_is_frozen_and_guarded():
    from filmcolor.fit import FIT_CONFIG_V1, fit_reference_v1
    from filmcolor.render import experimental_bounds

    c = FIT_CONFIG_V1
    assert (c["mode"], c["lam"], c["highlight_weight"], c["area_weight"]) == ("7reg", 0.05, 30.0, 10.0)
    assert c["scales"] == SCALES
    P = scene(3000)
    ref, _ = render_lab(P[None], RenderParams(**TRUE6))
    r = fit_reference_v1(P, ref[0])
    assert r["fit_config"] == "fit_config_v1" and r["lam"] == 0.05
    assert r["highlight"]["weight"] == 30.0 and r["highlight"]["area_weight"] == 10.0
    assert r["de2000_median"] < 1.0
    with experimental_bounds(contrast=(0.5, 3.0)):
        with pytest.raises(RuntimeError, match="bounds"):
            fit_reference_v1(P, ref[0])
