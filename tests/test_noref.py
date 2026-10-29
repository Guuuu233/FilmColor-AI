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


def backlit_scene(fg=0.05, sky=0.5, seed=31):
    """Half bright sky, half foreground of median ``fg`` spanning 4 stops (the T4c test scene with more sky, so the
    median sits in the sky and the foreground stays dark after exposure, as on A012)."""
    rng = np.random.default_rng(seed)
    n = 200_000
    y = np.where(rng.random(n) < sky, 0.9 * 2.0 ** rng.uniform(-0.5, 0.3, n), fg * 2.0 ** rng.uniform(-2.5, 1.5, n))
    c = rng.normal(0, 0.1, (n, 2))
    return np.stack([y * 2 ** c[:, 0], y, y * 2 ** c[:, 1]], -1).reshape(400, -1, 3).astype(np.float32)


# T4k white-balance changes off (cooling damp, tint boost, base tint, grey-world undamping, cool-cast temp cap): tests
# written for the earlier rules check those rules alone
T4K_WB_OFF = dict(wb_v3_cool_damp=1.0, wb_v3_tint_boost=1.0, base_tint=0.0, wb_v3_gw_cool_undamp=False,
                  wb_v3_cast_max_temp=None)


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
    """The estimator takes no reference argument and imports nothing that reads or compares references (T4k: plus the
    frame's own inversion G density range, keyword only)."""
    sig = inspect.signature(estimate_params_v2).parameters
    assert list(sig) == ["P", "mask", "cfg", "roll", "look", "span_d_g"]
    assert sig["span_d_g"].kind is inspect.Parameter.KEYWORD_ONLY
    tree = ast.parse(Path(noref.__file__).read_text())
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not mods & {"fit", "compare", "colorimetry", "filmcolor.fit", "filmcolor.compare"}
    assert mods <= {"__future__", "json", "pathlib", "numpy", "render"}


def test_noref_v3_adapts_exposure_and_protects_highlights_without_max_contrast():
    """T4 v2 (2026-10-23): NOREF_V3 (NegPy textured-cell Auto Density / Auto Grade / Highlight Hold / Shadow Hold)
    replaces fixed contrast=2.0 with adaptive contrast in [0.95, 1.38], lifts underexposed frames into the target
    encoded median band (0.45..0.52), and protects highlights within clip_budget. T4d / T4e: up to
    backlit_contrast_cap on frames with a backlit weight."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3

    dark = scene(median=0.03, stops=5.0, seed=10)
    ed = estimate_params_v3(dark, None, NOREF_V3)
    assert ed["status"] == "ok"
    assert 0.95 <= ed["params"]["contrast"] <= NOREF_V3["backlit_contrast_cap"]
    assert ed["params"]["ev"] > 0.8
    assert 0.43 <= ed["diagnostics"]["encoded_median"] <= 0.55
    assert ed["diagnostics"]["clip_share"] <= NOREF_V3["clip_budget"]
    assert "textured_cells" in ed["diagnostics"]

    bright = scene(median=0.18, stops=4.0, seed=11).reshape(-1, 3)
    bright[: len(bright) // 120] = 3.5  # ~0.8% bright highlights well above white
    eb = estimate_params_v3(bright.reshape(400, -1, 3), None, NOREF_V3)
    assert eb["status"] == "ok"
    assert 0.95 <= eb["params"]["contrast"] <= NOREF_V3["backlit_contrast_cap"]
    assert eb["diagnostics"]["clip_share"] <= NOREF_V3["clip_budget"]
    assert eb["params"]["shoulder"] > NOREF_V3["base_shoulder"] or eb["params"].get("highlights", 0.0) < 0.0


def test_noref_v3_t4b_keeps_print_contrast_and_never_sets_blacks():
    """T4b (2026-10-24): contrast and shadow decisions are measured on the encoded output. A normal frame keeps the
    print contrast centre 1.30 with no shadow lift and blacks 0 (T4 v2 gave contrast 0.95, shadows +0.15, blacks +0.10
    on such frames by comparing -log10 of the positive with print-density thresholds); a crushed frame gets a
    positive shadows lift and still blacks 0. T4d / T4e: the centre is base_contrast; the normal frame has no
    backlit weight (T4f: 3.5 stops; the wider T4f backlit weight gives a 4-stop log-uniform scene w = 0.06)."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3

    normal = estimate_params_v3(scene(median=0.18, stops=3.5, seed=21), None, NOREF_V3)
    assert normal["status"] == "ok" and normal["config_rev"] == NOREF_V3["rev"]
    assert "backlit_midtone" not in normal["flags"]  # T4c: a 7-stop log-uniform scene counts as backlit
    assert normal["params"]["contrast"] == pytest.approx(NOREF_V3["base_contrast"])
    assert normal["params"].get("blacks", 0.0) == 0.0 and normal["params"].get("shadows", 0.0) == 0.0
    assert normal["diagnostics"]["encoded_p25"] >= NOREF_V3["shadow_p25_floor"]

    crushed = scene(median=0.18, stops=7.0, seed=22).reshape(-1, 3)
    crushed[: len(crushed) * 2 // 5] *= 1e-3        # 40 % of the frame near black
    ec = estimate_params_v3(crushed.reshape(400, -1, 3), None, NOREF_V3)
    assert ec["diagnostics"]["encoded_p25_before_shadows"] < NOREF_V3["shadow_p25_floor"]
    assert 0.0 < ec["params"]["shadows"] <= NOREF_V3["shadow_max"]
    assert ec["params"].get("blacks", 0.0) == 0.0
    assert ec["params"]["contrast"] >= NOREF_V3["contrast_bounds_v3"][0]


def test_wb_v3_corrects_a_cast_when_top_and_bottom_differ():
    """T4b: a frame whose top and bottom halves have different scene colours (sky above, grass below) but share a
    green cast on their neutral greys: estimate_wb_v3 still measures the cast with non-zero confidence and moves
    tint towards magenta (the T4 v2 estimate_wb set confidence 0 on such frames). T4k: a tint towards magenta is
    multiplied by wb_v3_tint_boost."""
    from filmcolor.noref import NOREF_V3, estimate_wb_v3, wb_from_chroma

    rng = np.random.default_rng(5)
    n = 100_000
    y = 0.18 * 2.0 ** rng.uniform(-3, 3, n)
    cast = np.array([-0.12, -0.12])                  # greens: log2 R/G and B/G both low
    c = cast + rng.normal(0, 0.04, (n, 2))
    c[: n // 2][rng.random(n // 2) < 0.5] += [-0.5, 0.6]   # top: half the pixels are blue sky
    c[n // 2 :][rng.random(n - n // 2) < 0.5] += [0.1, -0.7]  # bottom: half are grass / earth
    px = np.stack([y * 2 ** c[:, 0], y, y * 2 ** c[:, 1]], -1).astype(np.float32)
    wb = estimate_wb_v3(px, NOREF_V3)
    assert wb["confidence"] > 0.3
    assert np.allclose(wb["chroma"], cast, atol=0.04)
    temp, tint = wb_from_chroma(cast)
    assert wb["tint"] == pytest.approx(NOREF_V3["wb_v3_tint_boost"] * NOREF_V3["wb_v3_strength"] * wb["confidence"]
                                       * tint, abs=0.02)
    assert wb["tint"] > 0.0


def test_noref_v3_t4c_backlit_frame_lifts_midtone_and_renders_the_given_matrix():
    """T4c: a frame with a bright background and a dark foreground gets a higher midtone, shoulder and contrast (T4e:
    each a step above the preset's own value at full weight); the estimate renders with cfg["matrix"] and
    chroma_preserve and returns them in its parameters. T4k: with the bright-share condition off (this scene's sky
    sits around 0.9 x 2^(-0.5..0.3)) and the tone pixel filter off (it leaves out the darkest foreground pixels, so
    p25 rises a little); both tested in test_t4k_*."""
    from filmcolor.noref import NOREF_V3 as V3, estimate_params_v3
    from filmcolor.render import IDENTITY

    NOREF_V3 = dict(V3, backlit_sky_check=False, tone_border_filter=False)

    P = backlit_scene()
    m = ((1.2, -0.1, -0.1), (-0.05, 1.2, -0.15), (0.0, -0.2, 1.2))
    e = estimate_params_v3(P, None, dict(NOREF_V3, matrix=m))
    assert "backlit_midtone" in e["flags"] and e["diagnostics"]["backlit_check"]["weight"] == 1.0
    m0 = NOREF_V3["base_midtone"]
    assert m0 < e["params"]["midtone"] <= m0 + NOREF_V3["backlit_midtone_step"] + 1e-4
    assert e["params"]["shoulder"] >= NOREF_V3["base_shoulder"] + NOREF_V3["backlit_shoulder_step"]
    assert e["params"]["contrast"] == pytest.approx(NOREF_V3["base_contrast"] + NOREF_V3["backlit_contrast_step"])
    assert e["params"]["matrix"] == [list(r) for r in m]
    assert e["params"]["chroma_preserve"] == NOREF_V3["chroma_preserve"]
    assert estimate_params_v3(P, None, NOREF_V3)["params"]["matrix"] == [list(r) for r in IDENTITY]


def test_t4d_style_presets_are_overrides_of_noref_v3_and_standard_is_noref_v3():
    """T4d: every style preset only overrides existing NOREF_V3 entries, standard is NOREF_V3 itself, unknown names
    are refused, and every preset gives a valid estimate whose blacks is the preset's constant."""
    from filmcolor.noref import NOREF_V3, STYLE_PRESETS, estimate_params_v3, preset_config

    assert list(STYLE_PRESETS) == ["standard", "natural", "portrait", "vivid", "print_classic"]
    assert preset_config("standard") == NOREF_V3
    with pytest.raises(ValueError, match="unknown style preset"):
        preset_config("frontier")
    P = scene(median=0.18, stops=4.0, seed=31)
    for name, spec in STYLE_PRESETS.items():
        assert set(spec["cfg"]) <= set(NOREF_V3) and spec["label"]
        cfg = preset_config(name)
        e = estimate_params_v3(P, None, cfg)
        assert e["status"] == "ok" and e["preset"] == name
        assert e["params"].get("blacks", 0.0) == cfg["base_blacks"] and e["params"]["chroma_preserve"] == cfg["chroma_preserve"]
        RenderParams.from_dict(e["params"])  # inside the slider bounds


def test_t4d_presets_order_brightness_contrast_and_colour():
    """T4d preset expectations on a plain 4-stop scene (no backlit weight): portrait brighter and print darker than
    standard; vivid more contrast and colour than standard, natural less; print and vivid crush blacks a little.
    (On a 4-stop scene the darker print preset already gets some backlit weight: encoded p25 < 0.36.)"""
    from filmcolor.noref import estimate_params_v3, preset_config

    P = scene(median=0.18, stops=3.5, chroma=0.12, seed=32)
    e = {n: estimate_params_v3(P, None, preset_config(n)) for n in ("standard", "natural", "portrait", "vivid",
                                                                     "print_classic")}
    assert all(x["diagnostics"]["backlit_check"]["weight"] == 0 for x in e.values())
    med = {n: x["diagnostics"]["encoded_median"] for n, x in e.items()}
    assert med["portrait"] > med["standard"] > med["print_classic"]
    c = {n: x["params"]["contrast"] for n, x in e.items()}
    assert c["vivid"] > c["print_classic"] > c["standard"] > c["natural"] > c["portrait"]
    for k in ("sat", "vibrance"):
        assert e["vivid"]["params"][k] > e["standard"]["params"][k] > e["natural"]["params"][k]
    b = {n: x["params"].get("blacks", 0.0) for n, x in e.items()}
    assert b["print_classic"] < b["vivid"] < b["standard"] == 0


def test_t4d_white_balance_falls_back_to_neutral_and_presets_add_their_offsets():
    """T4d: with no near-neutral evidence (wb confidence 0) NOREF_V3 stays at temp = tint = 0 instead of 0.55 x the
    global_look_v1 prior (T4b-T4c); portrait adds its offsets (T4f +0.02 / -0.02) to the same estimate. T4k: every
    preset's tint also gets base_tint."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3, preset_config

    P = scene(median=0.18, stops=4.0, cast=(0.9, -0.9), chroma=0.05, seed=33)  # every pixel strongly coloured
    # T4g adapts to strong casts. Disable the adaptive evidence here to keep testing the no-evidence fallback.
    cfg = dict(NOREF_V3, wb_v3_adaptive_share=0.0, wb_v3_gw_weight=0.0)
    e = estimate_params_v3(P, None, cfg)
    assert e["diagnostics"]["wb"]["confidence"] == 0
    assert NOREF_V3["base_tint"] > 0
    assert e["params"]["temp"] == 0 and e["params"]["tint"] == pytest.approx(NOREF_V3["base_tint"])
    old = estimate_params_v3(P, None, dict(cfg, wb_prior_weight=0.55))
    assert old["params"]["temp"] == pytest.approx(0.55 * LOOK["params"]["temp"], abs=1e-3)
    cp = preset_config("portrait", cfg)
    pt = estimate_params_v3(P, None, cp)
    assert pt["params"]["temp"] == pytest.approx(cp["temp_offset"])
    assert pt["params"]["tint"] == pytest.approx(cp["tint_offset"] + cp["base_tint"])


def test_t4d_backlit_weight_is_continuous():
    """T4d: the backlit weight grows continuously as the foreground gets darker under a bright sky (no switch);
    T4e: contrast is the preset's base_contrast + weight x backlit_contrast_step. T4k: bright-share condition and tone
    pixel filter off (they move the weight of this scene a little)."""
    from filmcolor.noref import NOREF_V3 as V3, estimate_params_v3

    NOREF_V3 = dict(V3, backlit_sky_check=False, tone_border_filter=False)
    ws = []
    for fg in (0.20, 0.15, 0.09, 0.08, 0.075, 0.07, 0.05):
        e = estimate_params_v3(backlit_scene(fg=fg), None, NOREF_V3)
        w = e["diagnostics"]["backlit_check"]["weight"]
        ws.append(w)
        assert 0.0 <= w <= 1.0
        if not (set(e["flags"]) & {"contrast_lowered_wide_frame"}):
            exp = NOREF_V3["base_contrast"] + w * NOREF_V3["backlit_contrast_step"]
            assert e["params"]["contrast"] == pytest.approx(exp, abs=1e-3)
    assert ws == sorted(ws) and ws[0] == 0 and ws[-1] == 1 and any(0 < w < 1 for w in ws)


def test_t4e_backlit_adjustment_is_relative_to_each_preset():
    """T4e: on a backlit frame the five presets keep distinct contrast (T4d pulled all of them to the same
    1.55 / 0.42 / 0.35 at full weight): contrast = min(cap, preset base + w x step), shoulder at least
    min(cap, preset base + w x step) (highlight hold may add more), midtone between the preset's base and
    min(cap, base + step). T4f: the midtone is not asserted distinct here: natural (lower contrast, so a higher p25)
    reaches backlit_p25_target 0.34 at 0.329, 0.001 from vivid's 0.33 at its search cap on this scene; the base
    midtones are distinct (test_t4f_presets_have_distinct_base_midtones)."""
    from filmcolor.noref import STYLE_PRESETS, estimate_params_v3, preset_config

    P = backlit_scene()
    e = {n: estimate_params_v3(P, None, preset_config(n)) for n in STYLE_PRESETS}
    for n, x in e.items():
        cfg, p = preset_config(n), x["params"]
        w = x["diagnostics"]["backlit_check"]["weight"]
        assert w > 0 and "backlit_midtone" in x["flags"]
        step = cfg["backlit_contrast_step"] if cfg["base_contrast"] >= 1.25 else 0.0
        exp_c = min(cfg["backlit_contrast_cap"], cfg["base_contrast"] + w * step)
        assert p["contrast"] == pytest.approx(exp_c, abs=1e-3)
        assert p["shoulder"] >= min(cfg["backlit_shoulder_cap"], cfg["base_shoulder"] + w * cfg["backlit_shoulder_step"]) - 1e-3
        hi = min(cfg["backlit_midtone_cap"], cfg["base_midtone"] + cfg["backlit_midtone_step"])
        assert cfg["base_midtone"] <= p["midtone"] <= hi + 1e-4
    v = sorted(x["params"]["contrast"] for x in e.values())
    assert all(b - a > 1e-3 for a, b in zip(v, v[1:])), v


def test_t4g_presets_keep_their_base_midtones():
    """T4g: standard and vivid share 0.23; the other base midtones are print 0.21, natural 0.24, portrait 0.26."""
    from filmcolor.noref import estimate_params_v3, preset_config

    P = scene(median=0.18, stops=3.5, chroma=0.12, seed=32)
    order = ("print_classic", "vivid", "standard", "natural", "portrait")
    e = {n: estimate_params_v3(P, None, preset_config(n)) for n in order}
    assert all(x["diagnostics"]["backlit_check"]["weight"] == 0 for x in e.values())
    mids = [e[n]["params"]["midtone"] for n in order]
    assert mids == [preset_config(n)["base_midtone"] for n in order] == sorted(mids)
    assert mids == [0.21, 0.23, 0.23, 0.24, 0.26]


def test_t4g_adaptive_wb_recovers_evidence_outside_the_fixed_radius():
    from filmcolor.noref import NOREF_V3, estimate_wb_v3

    rng = np.random.default_rng(47)
    y = 0.18 * 2 ** rng.uniform(-2, 2, 20000)
    c = rng.normal([-0.60, 0.12], 0.025, (len(y), 2))
    px = np.stack([y * 2 ** c[:, 0], y, y * 2 ** c[:, 1]], -1)
    fixed = estimate_wb_v3(px, dict(NOREF_V3, wb_v3_adaptive_share=0.0, wb_v3_gw_weight=0.0))
    adaptive = estimate_wb_v3(px)
    assert fixed["confidence"] == 0
    assert adaptive["confidence"] > 0.1 and adaptive["temp"] > 0.05
    assert all(z["radius_effective"] > z["radius"] for z in adaptive["diagnostics"]["zones"])
    assert adaptive["diagnostics"]["grey_world"]["weight"] > 0


def test_t4h_clear_sky_grey_world_damps_only_the_blue_axis():
    """T4h: a cyan cast in the clear-sky direction (red low, blue high; A008-like [-0.30, +0.26]) keeps its full
    grey-world weight; the estimate's red axis is kept at wb_v3_gw_sky_red and only its blue axis is damped by
    wb_v3_sky_weight (T4g damped the whole weight), so the frame is warmed. T4j: this scene is the same cast in every
    zone, a cool cast by wb_v3_cool_* (test_t4j_uniform_cool_cast_is_corrected_not_damped); checked here with the
    check off. T4k: also a strongly cool mid-grey zone and grey world without a top-bottom split; undamping off."""
    from filmcolor.noref import NOREF_V3 as V3, estimate_wb_v3

    NOREF_V3 = dict(V3, wb_v3_cool_cast_check=False, wb_v3_gw_cool_undamp=False)
    rng = np.random.default_rng(53)
    y = 0.18 * 2 ** rng.uniform(-2, 2, 20000)
    c = rng.normal([-0.30, 0.26], 0.03, (len(y), 2))
    px = np.stack([y * 2 ** c[:, 0], y, y * 2 ** c[:, 1]], -1)
    wb = estimate_wb_v3(px, NOREF_V3)
    gw = wb["diagnostics"]["grey_world"]
    assert gw["clear_sky"] and gw["weight"] == NOREF_V3["wb_v3_gw_weight"]
    raw, used = np.array(gw["chroma"]), np.array(gw["chroma_used"])
    np.testing.assert_allclose(used, raw * [NOREF_V3["wb_v3_gw_sky_red"], NOREF_V3["wb_v3_sky_weight"]], rtol=1e-6)
    assert wb["temp"] > 0
    full_red = estimate_wb_v3(px, dict(NOREF_V3, wb_v3_gw_sky_red=1.0))
    assert full_red["diagnostics"]["grey_world"]["chroma_used"][0] == pytest.approx(raw[0])


def test_t4g_soft_presets_keep_shadow_highlight_and_slope_defaults():
    from filmcolor.noref import estimate_params_v3, preset_config

    for name in ("natural", "portrait"):
        cfg = preset_config(name)
        for P in (scene(median=0.18, stops=3.5, seed=32), backlit_scene()):
            e = estimate_params_v3(P, None, cfg)
            p = e["params"]
            assert p["contrast"] == cfg["base_contrast"]
            assert p["shadows"] >= cfg["base_shadows"]
            assert p["highlights"] <= cfg["base_highlights"]
            for c in ("r", "b"):
                assert p.get(f"slope_{c}", 0.0) == cfg[f"base_slope_{c}"]
            RenderParams.from_dict(p)


def sky_over_grass(seed=41):
    """Top 55 %: smooth bright clear sky (log2 R/G -0.12, B/G +0.13; within-cell activity far below 0.05 D); bottom
    45 %: textured darker grass (median 0.12 of white, 3 stops). A013-like: the full-frame median sits in the sky."""
    rng = np.random.default_rng(seed)
    h, w = 400, 500
    ns = int(h * 0.55)
    P = np.empty((h, w, 3), np.float32)
    y_sky = 0.55 * 2.0 ** rng.normal(0, 0.01, (ns, w))
    P[:ns] = y_sky[..., None] * 2.0 ** np.array([-0.12, 0.0, 0.13])
    y_g = 0.12 * 2.0 ** rng.uniform(-1.5, 1.5, (h - ns, w))
    c = rng.normal(0, 0.08, (h - ns, w, 2)) + [0.05, -0.25]
    P[ns:] = np.stack([y_g * 2 ** c[..., 0], y_g, y_g * 2 ** c[..., 1]], -1)
    return P


def test_t4f_textured_trimean_anchor_and_ev_refinement_ignore_a_large_sky():
    """T4f Bug 1: the anchor is the Tukey trimean of the textured cell densities, and the EV refinement measures the
    textured cells' encoded median, lowering exposure by at most ev_refine_down_max (and not below -it). On a sky over
    grass frame whose full-frame encoded median is far above the band at the anchor exposure (T4c-T4e bisected that
    median down to the band: -1.08 EV on A013), the exposure stays within 0.20 EV of the anchor."""
    from filmcolor.noref import NOREF_V3, _encoded_quantiles, _sample, _textured_cells, estimate_params_v3

    P = sky_over_grass()
    tc = _textured_cells(P, None, NOREF_V3)
    assert 0.3 < tc["textured_share"] < 0.6  # the grass, not the sky
    assert tc["anchor_d"] == pytest.approx(0.5 * tc["p50_d"] + 0.25 * (tc["p25_d"] + tc["p75_d"]))
    e = estimate_params_v3(P, None, NOREF_V3)
    r, ev = e["diagnostics"]["ev_refine"], e["params"]["ev"]
    px = _sample(P, None, NOREF_V3["n_sample"] // 4, seed=1)
    full_med = _encoded_quantiles(px, RenderParams(**dict(e["params"], ev=r["ev_anchor"])), qs=(50,))[0]
    assert full_med > NOREF_V3["target_encoded_band"][1] + 0.2
    dmax = NOREF_V3["ev_refine_down_max"]
    assert r["ev_anchor"] > 0.3 and ev >= max(r["ev_anchor"] - dmax, -dmax) - 1e-3 and ev <= r["ev_anchor"]


def test_t4f_ev_refinement_brightens_under_the_p98_ceiling():
    """T4f Bug 1: a frame whose textured median is below the band is brightened only while the full-frame encoded p98
    stays <= ev_refine_p98_max (a dark frame with a 2.5 % bright window; anchor_target_density 1.4 makes the anchor
    step lift too little, so the refinement has to brighten)."""
    from filmcolor.noref import NOREF_V3, _encoded_quantiles, _sample, estimate_params_v3

    P = scene(median=0.03, stops=5.0, seed=10).reshape(-1, 3)
    P[: len(P) // 40] = 0.9
    P = P.reshape(400, -1, 3)
    cfg = dict(NOREF_V3, anchor_target_density=1.4, backlit_p90=2.0)  # backlit rule off: only the refinement moves ev
    e = estimate_params_v3(P, None, cfg)
    r = e["diagnostics"]["ev_refine"]
    assert r["med_tex_before"] < cfg["target_encoded_band"][0] and r.get("capped_by_p98")
    assert r["ev"] > r["ev_anchor"]
    px = _sample(P, None, cfg["n_sample"] // 4, seed=1)
    p = dict(e["params"], ev=r["ev"], shoulder=cfg["base_shoulder"], highlights=0.0)
    assert _encoded_quantiles(px, RenderParams(**p), qs=(98,))[0] <= cfg["ev_refine_p98_max"] + 1e-3


def test_t4f_wb_damps_a_clear_sky_bright_zone():
    """T4f Bug 3: a bright near-neutral zone with red low and blue high (clear sky) gets wb_v3_sky_weight x its
    weight, so a blue sky pulls the white balance less towards warm; strength 0.65 (T4b-T4e 0.85)."""
    from filmcolor.noref import NOREF_V3, _sample, estimate_wb_v3

    assert NOREF_V3["wb_v3_strength"] == 0.65
    px = _sample(sky_over_grass(), None, NOREF_V3["n_sample"] // 4, seed=1)
    wb = estimate_wb_v3(px, NOREF_V3)
    z0 = wb["diagnostics"]["zones"][0]
    assert z0["clear_sky"] and z0["weight"] == pytest.approx(NOREF_V3["wb_v3_zones"][0][2] * NOREF_V3["wb_v3_sky_weight"])
    undamped = estimate_wb_v3(px, dict(NOREF_V3, wb_v3_sky_weight=1.0))
    assert 0 < wb["temp"] < undamped["temp"]


def test_t4f_anchor_lift_is_held_to_the_p98_ceiling():
    """T4f: on a dark frame with a 2.5 % bright window the textured anchor alone asks for a large positive ev; with
    ev_anchor_p98_cap (default) that lift stops where full-frame encoded p98 reaches ev_refine_p98_max (at base shoulder,
    no highlights), as the task list wants for A012's sky; without it the anchor ev is kept and p98 passes the ceiling."""
    from filmcolor.noref import NOREF_V3, _encoded_quantiles, _sample, estimate_params_v3

    P = scene(median=0.03, stops=5.0, seed=10).reshape(-1, 3)
    P[: len(P) // 40] = 0.9
    P = P.reshape(400, -1, 3)
    px = _sample(P, None, NOREF_V3["n_sample"] // 4, seed=1)
    cfg = dict(NOREF_V3, backlit_p90=2.0)  # backlit rule off

    def p98(e, ev):
        return _encoded_quantiles(px, RenderParams(**dict(e["params"], ev=ev, shoulder=cfg["base_shoulder"],
                                                          highlights=0.0)), qs=(98,))[0]

    on, off = estimate_params_v3(P, None, cfg), estimate_params_v3(P, None, dict(cfg, ev_anchor_p98_cap=False))
    r_on, r_off = on["diagnostics"]["ev_refine"], off["diagnostics"]["ev_refine"]
    assert r_on["anchor_capped_by_p98"] and r_on["ev_anchor_raw"] == pytest.approx(r_off["ev_anchor"])
    assert 0.0 <= r_on["ev_anchor"] < r_off["ev_anchor"]
    assert p98(on, r_on["ev_anchor"]) <= cfg["ev_refine_p98_max"] + 1e-3 < p98(off, r_off["ev_anchor"])


def cool_cast_scene(mid_grey=(-0.15, 0.22), seed=61):
    """40 % near-neutral greys (cool, ``mid_grey`` below the 70th luminance percentile) and the rest strongly cast
    [-0.6, +0.8]; both halves of the frame alike."""
    rng = np.random.default_rng(seed)
    n = 40000
    y = 0.18 * 2 ** rng.uniform(-2, 2, n)
    grey = rng.random(n) < 0.4
    low = y < np.percentile(y, 70)
    g = np.where(low[:, None], rng.normal(mid_grey, 0.04, (n, 2)), rng.normal([-0.15, 0.22], 0.04, (n, 2)))
    c = np.where(grey[:, None], g, rng.normal([-0.6, 0.8], 0.1, (n, 2)))
    return np.stack([y * 2 ** c[:, 0], y, y * 2 ** c[:, 1]], -1)


def test_t4j_uniform_cool_cast_is_corrected_not_damped():
    """T4j (_DSF8948): bright zone, mid-grey zone and grey world all cool (red low, blue high past wb_v3_cool_*) is a
    cool cast, not a sky: neither the bright zone nor the grey world is damped, the grey world gets
    wb_v3_cast_gw_weight, confidence at least wb_v3_cast_conf and strength wb_v3_cast_strength, so the frame is warmed
    much more than by the T4h rule (cool_cast_check off); the global strength stays 0.65. T4k changes off here
    (test_t4k_cool_cast_temp_is_capped)."""
    from filmcolor.noref import NOREF_V3 as V3, estimate_wb_v3

    NOREF_V3 = dict(V3, **T4K_WB_OFF)

    assert NOREF_V3["wb_v3_strength"] == 0.65 and NOREF_V3["wb_v3_cast_strength"] == 0.80
    px = cool_cast_scene()
    wb = estimate_wb_v3(px, NOREF_V3)
    d = wb["diagnostics"]
    z0, z1, gw = d["zones"][0]["chroma"], d["zones"][1]["chroma"], d["grey_world"]["chroma"]
    assert z0[0] < -0.05 < 0.05 < z0[1] and z1[0] < -0.10 < 0.10 < z1[1] and gw[0] < -0.20 < 0.25 < gw[1]
    assert d["cool_cast"]
    assert not any(z.get("clear_sky") for z in d["zones"]) and "clear_sky" not in d["grey_world"]
    assert d["zones"][0]["weight"] == NOREF_V3["wb_v3_zones"][0][2]
    assert d["grey_world"]["weight"] == NOREF_V3["wb_v3_cast_gw_weight"] and wb["confidence"] >= NOREF_V3["wb_v3_cast_conf"]
    t4h = estimate_wb_v3(px, dict(NOREF_V3, wb_v3_cool_cast_check=False))
    assert not t4h["diagnostics"]["cool_cast"] and t4h["diagnostics"]["grey_world"]["clear_sky"]
    assert t4h["diagnostics"]["zones"][0]["clear_sky"] and wb["temp"] > t4h["temp"] + 0.15 > 0.15


def test_t4j_sky_over_a_warm_mid_grey_zone_is_still_damped():
    """T4j: the same cool bright zone and grey world over a mid-grey zone whose blue is not high (A008's mid-grey zone
    is [-0.12, -0.08]) is not a cool cast; the T4h clear-sky damping applies as before (same result as with the check
    off)."""
    from filmcolor.noref import NOREF_V3, estimate_wb_v3

    px = cool_cast_scene(mid_grey=(-0.12, -0.08))
    wb = estimate_wb_v3(px, NOREF_V3)
    d = wb["diagnostics"]
    z1 = d["zones"][1]["chroma"]
    assert z1[1] < NOREF_V3["wb_v3_cool_z1"][1] and not d["cool_cast"]
    assert d["zones"][0]["clear_sky"] and d["grey_world"]["clear_sky"]
    assert d["zones"][0]["weight"] == pytest.approx(NOREF_V3["wb_v3_zones"][0][2] * NOREF_V3["wb_v3_sky_weight"])
    off = estimate_wb_v3(px, dict(NOREF_V3, wb_v3_cool_cast_check=False))
    assert (wb["temp"], wb["tint"], wb["confidence"]) == (off["temp"], off["tint"], off["confidence"])


def test_t4i_auto_contrast_and_blacks_on_a_narrow_lifted_frame():
    """T4i: a narrow frame (2 stops, encoded p90 - p10 below auto_contrast_spread) gets contrast + min(max, gain x
    shortfall); its darkest 2 % sit above auto_blacks_p2, so blacks gets -clip((p2 - offset) x gain, 0, max), measured
    after the contrast step; both off with their switches. T4j: off by default, switched on here."""
    from filmcolor.noref import NOREF_V3 as V3, estimate_params_v3

    NOREF_V3 = dict(V3, auto_contrast=True, auto_blacks=True)
    assert V3["auto_contrast"] is False and V3["auto_blacks"] is False and V3["auto_slope"] is False

    P = scene(median=0.18, stops=2.0, chroma=0.05, seed=63)
    e = estimate_params_v3(P, None, NOREF_V3)
    lv, spread = e["diagnostics"]["auto_levels"], e["diagnostics"]["contrast_check"]["spread_10_90"]
    assert spread < NOREF_V3["auto_contrast_spread"]
    inc = min(NOREF_V3["auto_contrast_max"], NOREF_V3["auto_contrast_gain"] * (NOREF_V3["auto_contrast_spread"] - spread))
    assert lv["contrast"] == pytest.approx(inc) and e["params"]["contrast"] == pytest.approx(NOREF_V3["base_contrast"] + inc, abs=1e-4)
    p2 = lv["blacks"]["p2"]
    assert p2 > NOREF_V3["auto_blacks_p2"]
    exp_b = -min(NOREF_V3["auto_blacks_max"], (p2 - NOREF_V3["auto_blacks_offset"]) * NOREF_V3["auto_blacks_gain"])
    assert exp_b < 0 and e["params"]["blacks"] == pytest.approx(exp_b, abs=1e-4)
    off = estimate_params_v3(P, None, dict(NOREF_V3, auto_contrast=False, auto_blacks=False))
    assert off["params"]["contrast"] == NOREF_V3["base_contrast"] and off["params"].get("blacks", 0.0) == 0.0
    wide = estimate_params_v3(scene(median=0.18, stops=7.0, seed=64), None, NOREF_V3)
    assert "contrast" not in wide["diagnostics"]["auto_levels"]


def test_t4i_auto_slope_follows_green_between_shadows_and_highlights():
    """T4i: a positive whose blue channel has a lower gamma than green (log2 B = 0.85 log2 G: blue in the shadows,
    short of blue in the highlights) gets a negative slope_b, at most auto_slope_max, that brings B / G at the shadow
    and highlight percentiles closer together in the render; a channel that follows G gets none. T4j: off by default,
    switched on here."""
    from filmcolor.noref import NOREF_V3, _sample, estimate_params_v3
    from filmcolor.render import render_full

    rng = np.random.default_rng(65)
    g = 0.18 * 2.0 ** rng.uniform(-3, 3, 200_000)
    P = np.stack([g, g, 0.18 * (g / 0.18) ** 0.85], -1).reshape(400, -1, 3).astype(np.float32)
    cfg = dict(NOREF_V3, auto_slope=True, wb_v3_strength=0.0)  # white balance off: only the slope is under test
    e = estimate_params_v3(P, None, cfg)
    s = e["diagnostics"]["auto_levels"]["slope"]
    assert "slope_r" not in s and -NOREF_V3["auto_slope_max"] <= s["slope_b"] < 0
    assert e["params"]["slope_b"] == pytest.approx(s["slope_b"], abs=1e-4) and e["params"].get("slope_r", 0.0) == 0.0
    px = _sample(P, None, 20000, seed=2)

    def bg_gap(p):
        out, _, _ = render_full(px, RenderParams(**p))
        y = out[:, 1]
        lo, hi = y <= np.percentile(y, 5), y >= np.percentile(y, 95)
        ratio = np.log2(np.maximum(out[:, 2], 1e-3) / np.maximum(out[:, 1], 1e-3))
        return float(np.median(ratio[lo]) - np.median(ratio[hi]))

    assert abs(bg_gap(e["params"])) < abs(bg_gap(dict(e["params"], slope_b=0.0)))
    flat = estimate_params_v3(scene(median=0.18, stops=6.0, chroma=0.03, seed=66), None, cfg)
    assert "slope_r" not in flat["diagnostics"]["auto_levels"]["slope"]
    assert "slope_b" not in flat["diagnostics"]["auto_levels"]["slope"]


def test_t4k_cool_cast_temp_is_capped():
    """T4k: a cool-cast frame's temp is at most wb_v3_cast_max_temp (T4j pushed _DSF8948 and four other frames to the
    +0.50 limit); without the cap it is higher."""
    from filmcolor.noref import NOREF_V3, estimate_wb_v3

    px = cool_cast_scene()
    wb = estimate_wb_v3(px, NOREF_V3)
    assert wb["diagnostics"]["cool_cast"] and wb["temp"] == pytest.approx(NOREF_V3["wb_v3_cast_max_temp"])
    free = estimate_wb_v3(px, dict(NOREF_V3, wb_v3_cast_max_temp=None))
    assert free["temp"] > NOREF_V3["wb_v3_cast_max_temp"] + 0.05


def test_t4k_cooling_is_damped_and_a_green_cast_is_countered_more():
    """T4k: a warm cast (red high, blue low: warm wall, lamp light) is cooled by wb_v3_cool_damp x the measured
    correction; a green cast gets wb_v3_tint_boost x its magenta correction; with both factors at 1 the T4j values."""
    from filmcolor.noref import NOREF_V3, estimate_wb_v3

    rng = np.random.default_rng(7)
    y = 0.18 * 2 ** rng.uniform(-2, 2, 40000)
    for cast, k in (((0.25, -0.25), "temp"), ((-0.12, -0.12), "tint")):
        c = rng.normal(cast, 0.03, (len(y), 2))
        px = np.stack([y * 2 ** c[:, 0], y, y * 2 ** c[:, 1]], -1)
        wb = estimate_wb_v3(px, NOREF_V3)
        raw = dict(zip(("temp", "tint"), wb["diagnostics"]["temp_tint_raw"]))
        off = estimate_wb_v3(px, dict(NOREF_V3, wb_v3_cool_damp=1.0, wb_v3_tint_boost=1.0))
        assert off[k] == pytest.approx(raw[k])
        if k == "temp":
            assert raw["temp"] < -0.1 and wb["temp"] == pytest.approx(NOREF_V3["wb_v3_cool_damp"] * raw["temp"])
        else:
            assert raw["tint"] > 0.05 and wb["tint"] == pytest.approx(NOREF_V3["wb_v3_tint_boost"] * raw["tint"])


def test_t4k_grey_world_is_not_sky_damped_when_mid_greys_are_cool_everywhere():
    """T4k (_DSF8951): a neutral bright zone (so no cool cast) over strongly cool mid greys and grey world, the same in
    the top and bottom halves, keeps its grey-world estimate undamped and is warmed more than with the clear-sky
    damping of the grey world (wb_v3_gw_cool_undamp off)."""
    from filmcolor.noref import NOREF_V3, estimate_wb_v3

    rng = np.random.default_rng(71)
    n = 40000
    y = 0.18 * 2 ** rng.uniform(-2, 2, n)
    grey = rng.random(n) < 0.4
    low = y < np.percentile(y, 70)
    g = np.where(low[:, None], rng.normal([-0.25, 0.25], 0.04, (n, 2)), rng.normal([0.0, 0.0], 0.04, (n, 2)))
    c = np.where(grey[:, None], g, rng.normal([-0.6, 0.8], 0.1, (n, 2)))
    px = np.stack([y * 2 ** c[:, 0], y, y * 2 ** c[:, 1]], -1)
    wb = estimate_wb_v3(px, NOREF_V3)
    d = wb["diagnostics"]
    assert not d["cool_cast"] and d["gw_cool_undamped"] and "clear_sky" not in d["grey_world"]
    assert d["top_bottom_spread"] < NOREF_V3["wb_v3_undamp_tb_max"]
    off = estimate_wb_v3(px, dict(NOREF_V3, wb_v3_gw_cool_undamp=False))
    assert off["diagnostics"]["grey_world"]["clear_sky"] and wb["temp"] > off["temp"] + 0.02


def test_t4k_black_rebate_is_not_metered():
    """T4k (_DSF9239): grainy black rebate across 30 % of the frame makes textured cells at its noise and its edge
    with the image, which the T4j metering took as the frame's shadows (anchor 1.16 D instead of 0.74 D, +1.05 EV);
    with the border filter the anchor is the image's own and no extra exposure is given."""
    from filmcolor.noref import NOREF_V3, _textured_cells, estimate_params_v3

    rng = np.random.default_rng(3)
    P = scene(median=0.18, stops=4.0, seed=81).copy()
    P[:120] = 0.004 * np.exp(0.35 * rng.standard_normal(P[:120].shape))
    P = P.astype(np.float32)
    image_only = _textured_cells(P[120:], None, NOREF_V3)["anchor_d"]
    tc = _textured_cells(P, None, NOREF_V3)
    assert tc["border_cells"]["used"] and tc["border_cells"]["dropped"] > 0
    assert tc["anchor_d"] == pytest.approx(image_only, abs=0.02)
    t4j = _textured_cells(P, None, dict(NOREF_V3, cell_border_filter=False))["anchor_d"]
    assert t4j > image_only + 0.3
    on = estimate_params_v3(P, None, NOREF_V3)
    off = estimate_params_v3(P, None, dict(NOREF_V3, cell_border_filter=False, tone_border_filter=False))
    assert on["params"]["ev"] < off["params"]["ev"] - 0.8
    assert on["diagnostics"]["tone_pixels"]["n"] < on["diagnostics"]["tone_pixels"]["n_all"]


def test_t4k_backlit_lift_needs_a_bright_background():
    """T4k: a dark foreground before a background that is only mid-bright (linear 0.35, below bright_luma) gets no
    backlit lift (weight x ramp(bright share; 0.04, 0.12) = 0); the T4j weight would be about 0.5."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3

    rng = np.random.default_rng(37)
    n = 200_000
    y = np.where(rng.random(n) < 0.45, 0.35 * 2.0 ** rng.uniform(-0.3, 0.2, n), 0.02 * 2.0 ** rng.uniform(-2.0, 1.5, n))
    c = rng.normal(0, 0.1, (n, 2))
    P = np.stack([y * 2 ** c[:, 0], y, y * 2 ** c[:, 1]], -1).reshape(400, -1, 3).astype(np.float32)
    on = estimate_params_v3(P, None, NOREF_V3)
    off = estimate_params_v3(P, None, dict(NOREF_V3, backlit_sky_check=False))
    assert on["diagnostics"]["tone_pixels"]["bright_share"] < NOREF_V3["backlit_sky_share_lo"]
    assert on["diagnostics"]["backlit_check"]["weight"] == 0 and "backlit_midtone" not in on["flags"]
    assert off["diagnostics"]["backlit_check"]["weight"] > 0.3 and "backlit_midtone" in off["flags"]


def test_t4k_thin_negative_caps_positive_ev_and_lowers_blacks():
    """T4k (A224, night frames): with the inversion's G density range span_d_g below thin_span_max and few bright
    pixels, positive ev is at most max(0, (span - 0.42) / 0.20) x 0.60 and, below 0.52, blacks is lowered by
    clip((0.52 - span) x 0.8, 0, 0.12); a normal range or no span given leaves the estimate unchanged."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3

    P = scene(median=0.05, stops=4.0, seed=82)
    free = estimate_params_v3(P, None, NOREF_V3)
    thin = estimate_params_v3(P, None, NOREF_V3, span_d_g=0.45)
    wide = estimate_params_v3(P, None, NOREF_V3, span_d_g=0.70)
    cap = (0.45 - NOREF_V3["thin_span_zero"]) / NOREF_V3["thin_span_ramp"] * NOREF_V3["thin_ev_max"]
    assert free["params"]["ev"] > 1.0 and "thin_negative" not in free["diagnostics"]
    assert thin["diagnostics"]["thin_negative"]["applied"] and thin["params"]["ev"] == pytest.approx(cap, abs=1e-4)
    assert thin["params"]["blacks"] == pytest.approx(-(0.52 - 0.45) * NOREF_V3["thin_blacks_gain"], abs=1e-4)
    assert not wide["diagnostics"]["thin_negative"]["applied"] and wide["params"] == free["params"]
    bright = estimate_params_v3(scene(median=0.5, stops=3.0, seed=83), None, NOREF_V3, span_d_g=0.45)
    assert bright["diagnostics"]["tone_pixels"]["bright_share"] >= NOREF_V3["thin_bright_share_max"]
    assert not bright["diagnostics"]["thin_negative"]["applied"]
