import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from filmcolor import noref
from filmcolor.noref import NOREF_V2, estimate_params_v2, load_look
from filmcolor.render import RenderParams

LOOK = load_look()
ALL = dict(NOREF_V2, groups=("brightness", "wb", "tone", "sat"), spread_band=[0.55, 0.85])


def scene(n=200_000, median=0.18, stops=7.0, cast=(0.0, 0.0), chroma=0.15, seed=0):
    """Positive with log-uniform luminance spanning ``stops`` around ``median``, mild colour spread
    around a neutral shifted by ``cast`` = (log2 R/G, log2 B/G)."""
    rng = np.random.default_rng(seed)
    y = median * 2.0 ** rng.uniform(-stops / 2, stops / 2, n)
    c = np.array(cast) + rng.normal(0, chroma, (n, 2))
    P = np.stack([y * 2 ** c[:, 0], y, y * 2 ** c[:, 1]], -1)
    return P.reshape(400, -1, 3).astype(np.float32)


def look_params(**kw):
    return RenderParams(**dict(LOOK["params"], **kw)).to_dict()


def test_look_is_a_recorded_development_fit_not_an_engine_default():
    assert LOOK["id"] == "global_look_v1"
    assert "supervised" in LOOK["nature"] and LOOK["fit"]["fit_code_commit"]
    assert LOOK["fit"]["train"].startswith("nlp_official")
    assert RenderParams() != RenderParams(**LOOK["params"])  # the engine default is untouched


def test_no_groups_returns_the_look_exactly():
    e = estimate_params_v2(scene(), None, dict(NOREF_V2, groups=()), look=LOOK)
    assert e["status"] == "ok" and e["params"] == look_params()


def test_deterministic():
    P = scene(seed=3)
    assert estimate_params_v2(P, None, ALL, look=LOOK) == estimate_params_v2(P.copy(), None, ALL, look=LOOK)


def test_low_information_frame_needs_review_and_is_not_enhanced():
    e = estimate_params_v2(scene(stops=1.5), None, ALL, look=LOOK)
    assert e["status"] == "manual_review_required" and e["confidence"]["overall"] == 0
    assert e["params"] == look_params() and e["flags"][0].startswith("low_information")


def test_highlights_far_past_the_white_point_need_review():
    P = scene().reshape(-1, 3)
    P[: len(P) // 20] = 4.0  # 5 % of the frame two stops past the inversion white point
    e = estimate_params_v2(P.reshape(400, -1, 3), None, ALL, look=LOOK)
    assert e["status"] == "manual_review_required" and e["flags"][0].startswith("white_point_invalid")


def test_backlit_frame_is_brightened_within_the_clip_budget():
    P = scene(median=0.03, stops=5).reshape(-1, 3)
    P[: len(P) // 200] = 1.3  # a small window above white
    e = estimate_params_v2(P.reshape(400, -1, 3), None, ALL, look=LOOK)
    assert e["status"] == "ok" and e["params"]["ev"] > LOOK["params"]["ev"] + 1
    assert e["diagnostics"]["clip_share"] <= ALL["clip_budget"]
    assert e["confidence"]["exposure"] < 1


def test_exposure_confidence_falls_with_the_size_of_the_shift():
    small = estimate_params_v2(scene(median=0.10), None, ALL, look=LOOK)
    large = estimate_params_v2(scene(median=0.02, stops=5), None, ALL, look=LOOK)
    assert large["diagnostics"]["brightness_shift"] > small["diagnostics"]["brightness_shift"]
    assert large["confidence"]["exposure"] < small["confidence"]["exposure"]


def test_high_key_frame_left_alone_when_brighten_only():
    P = scene(median=0.5, stops=4)
    sym = estimate_params_v2(P, None, dict(ALL, groups=("brightness",)), look=LOOK)
    up = estimate_params_v2(P, None, dict(ALL, groups=("brightness",), brighten_only=True), look=LOOK)
    assert sym["params"]["ev"] < LOOK["params"]["ev"] and up["params"]["ev"] == pytest.approx(LOOK["params"]["ev"])


def test_strongly_coloured_light_is_not_forced_neutral():
    # sunset-like frame: every pixel far from neutral, no low-saturation candidates
    e = estimate_params_v2(scene(stops=5, cast=(0.8, -0.9), chroma=0.05), None, dict(ALL, groups=("wb",)), look=LOOK)
    assert e["status"] == "ok" and e["confidence"]["wb"] == 0
    assert (e["params"]["temp"], e["params"]["tint"]) == pytest.approx((LOOK["params"]["temp"], LOOK["params"]["tint"]))


def test_mild_cast_with_neutral_evidence_is_reduced():
    e = estimate_params_v2(scene(cast=(0.12, -0.08), chroma=0.04), None, dict(ALL, groups=("wb",)), look=LOOK)
    wb = e["diagnostics"]["wb"]
    assert e["confidence"]["wb"] > 0.5 and wb["temp"] < 0  # cools the warm (R up, B down) cast


def test_group_validation():
    with pytest.raises(ValueError, match="unknown parameter groups"):
        estimate_params_v2(scene(), None, dict(NOREF_V2, groups=("contrast",)), look=LOOK)
    with pytest.raises(ValueError, match="spread_band"):
        estimate_params_v2(scene(), None, dict(NOREF_V2, groups=("tone",)), look=LOOK)


def test_estimation_input_matches_the_development_runs():
    from filmcolor.compare import block_mean

    P = np.random.default_rng(1).uniform(0, 1, (103, 98, 3)).astype(np.float32)
    P4, m = noref.estimation_input(P, (8, 4, 40, 60))
    assert np.array_equal(P4, block_mean(P, 4).astype(np.float32))
    assert m.shape == P4.shape[:2] and m.sum() == 10 * 15 and m[2, 1] and not m[1, 1]


def test_information_isolation():
    """The estimator takes no reference argument and imports nothing that reads or compares references."""
    assert list(inspect.signature(estimate_params_v2).parameters) == ["P", "mask", "cfg", "roll", "look"]
    tree = ast.parse(Path(noref.__file__).read_text())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not mods & {"fit", "compare", "colorimetry", "filmcolor.fit", "filmcolor.compare"}
    assert mods <= {"__future__", "json", "pathlib", "numpy", "render"}


def test_noref_v3_adapts_exposure_and_protects_highlights_without_max_contrast():
    """T4 v2 (2026-10-23): NOREF_V3 (NegPy textured-cell Auto Density / Auto Grade / Highlight Hold / Shadow Hold)
    replaces fixed contrast=2.0 with adaptive contrast in [0.95, 1.38], lifts underexposed frames into the target
    encoded median band (0.45..0.52), and protects highlights within clip_budget."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3

    dark = scene(median=0.03, stops=5.0, seed=10)
    ed = estimate_params_v3(dark, None, NOREF_V3)
    assert ed["status"] == "ok"
    assert 0.95 <= ed["params"]["contrast"] <= 1.38
    assert ed["params"]["ev"] > 0.8
    assert 0.43 <= ed["diagnostics"]["encoded_median"] <= 0.55
    assert ed["diagnostics"]["clip_share"] <= NOREF_V3["clip_budget"]
    assert "textured_cells" in ed["diagnostics"]

    bright = scene(median=0.18, stops=4.0, seed=11).reshape(-1, 3)
    bright[: len(bright) // 120] = 3.5  # ~0.8% bright highlights well above white
    eb = estimate_params_v3(bright.reshape(400, -1, 3), None, NOREF_V3)
    assert eb["status"] == "ok"
    assert 0.95 <= eb["params"]["contrast"] <= 1.38
    assert eb["diagnostics"]["clip_share"] <= NOREF_V3["clip_budget"]
    assert eb["params"]["shoulder"] > NOREF_V3["base_shoulder"] or eb["params"].get("highlights", 0.0) < 0.0
