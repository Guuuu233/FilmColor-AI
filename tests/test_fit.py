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
