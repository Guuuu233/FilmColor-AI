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
    assert mods <= {"__future__", "json", "pathlib", "numpy", "render", "holder"}  # T4l: holder (the frame's own pixels)


def test_noref_v3_adapts_exposure_and_protects_highlights_without_max_contrast():
    """T4 v2 (2026-10-23): NOREF_V3 (NegPy textured-cell Auto Density / Auto Grade / Highlight Hold / Shadow Hold)
    replaces fixed contrast=2.0 with adaptive contrast in [0.95, 1.38], lifts underexposed frames into the target
    encoded median band (0.45..0.52), and protects highlights within clip_budget. T4d / T4e: up to
    backlit_contrast_cap on frames with a backlit weight. T4l: with the adaptive ev lowering off (it lowers this frame,
    whose textured cells sit above the band after the anchor's lift, further: test_t4l_ev_lowering_*)."""
    from filmcolor.noref import NOREF_V3 as V3, estimate_params_v3

    NOREF_V3 = dict(V3, ev_down_adaptive=False)
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
    backlit weight (T4f: 3.5 stops; the wider T4f backlit weight gives a 4-stop log-uniform scene w = 0.06). T4l: the
    crushed frame's black top 40 % is a full black strip for the metering frame, so strip trimming is off here."""
    from filmcolor.noref import NOREF_V3 as V3, estimate_params_v3

    NOREF_V3 = dict(V3, meter_trim_strips=False)

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
    p25 rises a little); both tested in test_t4k_*. T4l: adaptive ev lowering off (this sky lowers the textured median)."""
    from filmcolor.noref import NOREF_V3 as V3, estimate_params_v3
    from filmcolor.render import IDENTITY

    NOREF_V3 = dict(V3, backlit_sky_check=False, tone_border_filter=False, ev_down_adaptive=False)

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
    pixel filter off (they move the weight of this scene a little); T4l: adaptive ev lowering off."""
    from filmcolor.noref import NOREF_V3 as V3, estimate_params_v3

    NOREF_V3 = dict(V3, backlit_sky_check=False, tone_border_filter=False, ev_down_adaptive=False)
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
    median down to the band: -1.08 EV on A013), the exposure stays within 0.20 EV of the anchor. T4l: adaptive ev
    lowering off (on this grass the textured p25 is 0.52, next to its median 0.535, so the T4l ramp widens the limit;
    it lowers 0.04 EV more here)."""
    from filmcolor.noref import NOREF_V3 as V3, _encoded_quantiles, _sample, _textured_cells, estimate_params_v3

    NOREF_V3 = dict(V3, ev_down_adaptive=False)

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
    """T4k: with wb_v3_cast_max_temp set, a cool-cast frame's temp is at most that (T4j pushed _DSF8948 and four other
    frames to the +0.50 limit); off by default (None: up to wb_max_shift, as T4j)."""
    from filmcolor.noref import NOREF_V3, estimate_wb_v3

    assert NOREF_V3["wb_v3_cast_max_temp"] is None
    px = cool_cast_scene()
    wb = estimate_wb_v3(px, dict(NOREF_V3, wb_v3_cast_max_temp=0.38))
    assert wb["diagnostics"]["cool_cast"] and wb["temp"] == pytest.approx(0.38)
    free = estimate_wb_v3(px, NOREF_V3)
    assert free["temp"] > 0.38 + 0.05


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
    with the border filter the anchor is the image's own and no extra exposure is given. Off by default (it also drops
    dark foregrounds: A012); switched on here."""
    from filmcolor.noref import NOREF_V3 as V3, _textured_cells, estimate_params_v3

    assert not V3["cell_border_filter"] and not V3["tone_border_filter"]
    # T4l: the metering frame would cut these rebate rows first (test_t4l_metering_mask_*): off here
    NOREF_V3 = dict(V3, cell_border_filter=True, tone_border_filter=True, meter_holder_box=False, meter_trim_strips=False)

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
    backlit lift (weight x ramp(bright share; 0.04, 0.12) = 0); the T4j weight would be about 0.5. Off by default (0
    better / 2 worse on the 185 frames); switched on here."""
    from filmcolor.noref import NOREF_V3 as V3, estimate_params_v3

    assert not V3["backlit_sky_check"]
    NOREF_V3 = dict(V3, backlit_sky_check=True)

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
    """Thin-negative cap and black lowering use the configured span thresholds; a normal range or no span given
    leaves the estimate unchanged. T4m permits negative EV and deeper blacks on very thin negatives."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3

    P = scene(median=0.05, stops=4.0, seed=82)
    free = estimate_params_v3(P, None, NOREF_V3)
    thin = estimate_params_v3(P, None, NOREF_V3, span_d_g=0.45)
    wide = estimate_params_v3(P, None, NOREF_V3, span_d_g=0.70)
    cap = (0.45 - NOREF_V3["thin_span_zero"]) / NOREF_V3["thin_span_ramp"] * NOREF_V3["thin_ev_max"]
    assert free["params"]["ev"] > 1.0 and "thin_negative" not in free["diagnostics"]
    assert thin["diagnostics"]["thin_negative"]["applied"] and thin["params"]["ev"] == pytest.approx(cap, abs=1e-4)
    assert thin["params"]["blacks"] == pytest.approx(-(NOREF_V3["thin_blacks_span"] - 0.45) * NOREF_V3["thin_blacks_gain"], abs=1e-4)
    assert not wide["diagnostics"]["thin_negative"]["applied"] and wide["params"] == free["params"]
    bright = estimate_params_v3(scene(median=0.5, stops=3.0, seed=83), None, NOREF_V3, span_d_g=0.45)
    assert bright["diagnostics"]["tone_pixels"]["bright_share"] >= NOREF_V3["thin_bright_share_max"]
    assert not bright["diagnostics"]["thin_negative"]["applied"]


def test_t4l_metering_mask_uses_the_inset_film_box_and_cuts_full_black_strips_only():
    """T4l: with an opaque holder (holder.holder_mask) around the frame the metering mask is the film box inset by
    meter_box_inset; full black strips at the mask's edges (rebate, gaps between frames: >= meter_strip_share of a
    row at max channel <= meter_strip_luma) are cut with meter_strip_margin more; a dark subject inside the frame
    (A012's foreground, under meter_strip_share of any row) is kept; both off give the mask unchanged."""
    from filmcolor.noref import NOREF_V3, _metering_mask

    P = scene(median=0.18, stops=4.0, seed=91).copy()  # 400 x 500
    P[:40] = [300.0, 36.0, 3.6]  # holder above (positive far past white, as _DSF8948)
    m, d = _metering_mask(P, None, NOREF_V3)
    t, l, h, w = d["film_box"]
    assert d["used"] and (t, h) == (40, 360) and d["box_inset"] == [round(0.08 * 360), round(0.08 * 500)]
    assert not m[: 40 + d["box_inset"][0]].any() and m[200, 250]
    Q = scene(median=0.18, stops=4.0, seed=92).copy()
    Q[:30] = 0.004  # black gap between frames across the top
    Q[300:, 100:250] = 0.006  # a dark subject: 30 % of those rows
    m2, d2 = _metering_mask(Q, None, NOREF_V3)
    assert d2["strips"]["top"] == 30 and not m2[: 30 + NOREF_V3["meter_strip_margin"]].any()
    assert m2[350, 150] and d2["strips"]["bottom"] == 0
    off = dict(NOREF_V3, meter_holder_box=False, meter_trim_strips=False)
    assert _metering_mask(Q, None, off) == (None, None)


def test_t4l_ev_lowering_widens_on_frames_bright_all_over():
    """T4l: when the textured cells' encoded p25 is high (a frame bright all over, 13 BlueNeg rolls / A222) and
    ev_down_adaptive is enabled, the ev refinement may lower by ev_refine_down_max + ev_down_extra x ramp(p25;
    ev_down_q25), instead of 0.20; off by default (False: 4 better / 10 worse on the 185 dev frames); with p25 below the
    ramp (a sky over dark ground, A013) the limit stays 0.20 and the result is the T4k one."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3

    assert not NOREF_V3["ev_down_adaptive"]
    P = scene(median=0.75, stops=2.0, seed=93)
    off = estimate_params_v3(P, None, NOREF_V3)
    on = estimate_params_v3(P, None, dict(NOREF_V3, ev_down_adaptive=True))
    r, r_off = on["diagnostics"]["ev_refine"], off["diagnostics"]["ev_refine"]
    assert r["q25_tex"] >= NOREF_V3["ev_down_q25"][0] and r["down_max"] > NOREF_V3["ev_refine_down_max"] + 0.1
    floor = min(r_off["ev_anchor"], max(r_off["ev_anchor"] - NOREF_V3["ev_refine_down_max"], -NOREF_V3["ev_refine_down_max"]))
    assert off["params"]["ev"] >= floor - 1e-6 and on["params"]["ev"] < floor - 0.1
    high = estimate_params_v3(P, None, dict(NOREF_V3, ev_down_adaptive=True, ev_down_q25=(0.95, 0.99)))  # p25 below ramp
    assert high["diagnostics"]["ev_refine"]["down_max"] == pytest.approx(NOREF_V3["ev_refine_down_max"])
    assert high["params"]["ev"] == off["params"]["ev"]


def test_t4m_very_thin_negative_darkens_without_a_backlit_lift():
    from filmcolor.noref import NOREF_V3, estimate_params_v3

    P = backlit_scene(fg=0.02, sky=0.30, seed=91)
    off = estimate_params_v3(P, None, dict(NOREF_V3, thin_neg_check=False), span_d_g=0.30)
    assert off["diagnostics"]["backlit_check"]["weight"] > 0
    on = estimate_params_v3(P, None, NOREF_V3, span_d_g=0.30)
    assert on["diagnostics"]["thin_negative"]["applied"]
    assert on["params"]["ev"] <= NOREF_V3["thin_ev_min"] + 1e-4
    assert on["params"]["blacks"] == pytest.approx(-NOREF_V3["thin_blacks_max"])
    assert on["diagnostics"]["backlit_check"]["thin_suppressed"]
    assert on["diagnostics"]["backlit_check"]["weight"] == 0 and "backlit_midtone" not in on["flags"]
    assert on["params"]["midtone"] == NOREF_V3["base_midtone"]
    wide = estimate_params_v3(P, None, NOREF_V3, span_d_g=0.70)
    assert wide["params"] == off["params"]  # ordinary dense negatives keep the backlit treatment


def test_t4m_strong_cyan_without_neutrals_gets_a_confidence_floor():
    from filmcolor.noref import NOREF_V3, estimate_wb_v3

    P = scene(cast=(-1.4, -0.3), chroma=0.25, seed=92).reshape(-1, 3)
    off_cfg = dict(NOREF_V3, wb_v3_cyan_conf_check=False)
    off = estimate_wb_v3(P, off_cfg)
    on = estimate_wb_v3(P, NOREF_V3)
    assert not on["diagnostics"]["cool_cast"]
    assert all(z["narrow_share"] < 0.04 for z in on["diagnostics"]["zones"])
    assert on["chroma"][0] < -0.85 and on["diagnostics"]["cyan_conf_boost"]
    assert off["confidence"] < on["confidence"]
    assert on["confidence"] == pytest.approx(0.72)
    assert on["temp"] > off["temp"]
    for cast in ((-0.2, -0.1), (-0.5, -0.3), (0.9, -0.3)):
        px = scene(cast=cast, seed=93).reshape(-1, 3)
        a, b = estimate_wb_v3(px, NOREF_V3), estimate_wb_v3(px, off_cfg)
        assert "cyan_conf_boost" not in a["diagnostics"] and a == b
    cool = estimate_wb_v3(cool_cast_scene(), NOREF_V3)
    assert cool["diagnostics"]["cool_cast"] and "cyan_conf_boost" not in cool["diagnostics"]


def test_t4o_cyan_confidence_checks_blue_axis_threshold():
    """T4o: cyan confidence boost requires est[1] <= wb_v3_cyan_b_max (0.20), replacing the old requirement that
    mid-grey zone 1 be strongly red-deficient (chroma[0] < -0.65)."""
    from filmcolor.noref import NOREF_V3, estimate_wb_v3

    rng = np.random.default_rng(31)
    y = 2 ** rng.uniform(-6, 0, 100_000)
    red = np.where(rng.random(len(y)) < 0.8, -2.0, -0.5)

    # Blue axis <= 0.20: boosts confidence even when z1 chroma[0] >= -0.65
    px_pass = np.stack([y * 2 ** red, y, y * 2 ** -0.4], -1).astype(np.float32)
    res_pass = estimate_wb_v3(px_pass, NOREF_V3)
    diag_pass = res_pass["diagnostics"]
    assert all(z["narrow_share"] < 0.04 for z in diag_pass["zones"])
    assert res_pass["chroma"][0] < -0.85
    assert diag_pass["zones"][1]["chroma"][0] >= -0.65  # old T4n requirement not met
    assert res_pass["chroma"][1] <= NOREF_V3["wb_v3_cyan_b_max"]  # new T4o blue-axis check passes
    assert diag_pass["cyan_conf_boost"] is True
    assert res_pass["confidence"] >= NOREF_V3["wb_v3_cyan_min_conf"]

    # Blue axis > 0.20: boost is refused
    px_fail = np.stack([y * 2 ** red, y, y * 2 ** 0.35], -1).astype(np.float32)
    res_fail = estimate_wb_v3(px_fail, NOREF_V3)
    diag_fail = res_fail["diagnostics"]
    assert res_fail["chroma"][0] < -0.85
    assert res_fail["chroma"][1] > NOREF_V3["wb_v3_cyan_b_max"]
    assert "cyan_conf_boost" not in diag_fail
    assert res_fail == estimate_wb_v3(px_fail, dict(NOREF_V3, wb_v3_cyan_conf_check=False))


def test_t4o_spatial_cool_cast_gate_requires_both_spread_and_disagreement():
    """T4o: cool cast detection is disabled (split_sky=True) only when BOTH top_bottom_spread and zone_disagreement
    exceed 0.22; uniform cool cast remains enabled when either or neither threshold is exceeded.
    Explicitly disable T4p bright_px_share gate to verify the original T4o dual-threshold contract alone."""
    from filmcolor.noref import NOREF_V3, estimate_wb_v3

    cfg_base = dict(NOREF_V3, wb_v3_cast_max_bright_share=np.inf)
    px = cool_cast_scene()
    # 1. Uniform scene: neither threshold exceeded -> cool_cast active, split_sky False
    res_uniform = estimate_wb_v3(px, cfg_base)
    d_u = res_uniform["diagnostics"]
    assert d_u["top_bottom_spread"] <= cfg_base["wb_v3_cast_max_tb_spread"]
    assert d_u["zone_disagreement"] <= cfg_base["wb_v3_cast_max_zone_dis"]
    assert not d_u["split_sky"] and d_u["cool_cast"]

    # 2. Only top_bottom_spread exceeded -> split_sky False, cool_cast still active
    cfg_tb = dict(cfg_base, wb_v3_cast_max_tb_spread=d_u["top_bottom_spread"] * 0.5,
                  wb_v3_cast_max_zone_dis=d_u["zone_disagreement"] * 2.0)
    res_tb = estimate_wb_v3(px, cfg_tb)
    assert not res_tb["diagnostics"]["split_sky"] and res_tb["diagnostics"]["cool_cast"]

    # 3. Only zone_disagreement exceeded -> split_sky False, cool_cast still active
    cfg_zd = dict(cfg_base, wb_v3_cast_max_tb_spread=d_u["top_bottom_spread"] * 2.0,
                  wb_v3_cast_max_zone_dis=d_u["zone_disagreement"] * 0.5)
    res_zd = estimate_wb_v3(px, cfg_zd)
    assert not res_zd["diagnostics"]["split_sky"] and res_zd["diagnostics"]["cool_cast"]

    # 4. BOTH exceeded -> split_sky True, cool_cast suppressed
    cfg_both = dict(cfg_base, wb_v3_cast_max_tb_spread=d_u["top_bottom_spread"] * 0.5,
                    wb_v3_cast_max_zone_dis=d_u["zone_disagreement"] * 0.5)
    res_both = estimate_wb_v3(px, cfg_both)
    assert res_both["diagnostics"]["split_sky"] and not res_both["diagnostics"]["cool_cast"]


def test_t4o_capped_backlit_rescue_c1_and_c2():
    """T4o: Step 5 backlit capped rescue lifts midtone/shoulder/shadows for c1 (sparse highlights capping EV)
    or c2 (near backlit), without triggering on night frames (med < 0.24), low-span negatives, or stacking on
    normal backlit frames; output clip share remains within clip_budget."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3, _clip_share, RenderParams

    # Synthetic scene for c1: dark foreground with sparse highlights capping EV
    rng = np.random.default_rng(42)
    n = 100_000
    y = np.zeros(n)
    y[:80000] = rng.uniform(0.02, 0.06, 80000)
    y[80000:97000] = rng.uniform(0.12, 0.22, 17000)
    y[97000:] = 1.8
    c = rng.normal(0, 0.05, (n, 2))
    px_c1 = np.stack([y * 2 ** c[:, 0], y, y * 2 ** c[:, 1]], -1).reshape(200, 500, 3).astype(np.float32)

    # 1. c1 trigger
    e1 = estimate_params_v3(px_c1, None, NOREF_V3, span_d_g=1.1)
    bc1 = e1["diagnostics"]["backlit_check"]
    assert bc1.get("rescue_c1") is True
    assert bc1.get("rescue_weight", 0) > 0
    assert e1["params"]["midtone"] > NOREF_V3["base_midtone"]
    assert e1["params"]["shoulder"] > NOREF_V3["base_shoulder"]
    assert e1["params"].get("shadows", 0) > 0
    assert "backlit_midtone" in e1["flags"]
    assert _clip_share(px_c1, RenderParams(**e1["params"])) <= NOREF_V3["clip_budget"]

    # 2. c2 trigger: near backlit conditions
    cfg_c2 = dict(NOREF_V3, backlit_capped_rescue=True,
                  backlit_capped_med_band=(0.0, 0.01),  # disable c1
                  backlit_near_min_span=1.0,
                  backlit_near_q90_band=(bc1["rescue_q90"] - 0.01, bc1["rescue_q90"] + 0.01),
                  backlit_near_q25_band=(bc1["rescue_q25"] - 0.01, bc1["rescue_q25"] + 0.01))
    e2 = estimate_params_v3(px_c1, None, cfg_c2, span_d_g=1.1)
    bc2 = e2["diagnostics"]["backlit_check"]
    assert bc2.get("rescue_c2") is True
    assert bc2.get("rescue_weight", 0) > 0
    assert e2["params"]["midtone"] > NOREF_V3["base_midtone"]
    assert e2["params"]["shoulder"] > NOREF_V3["base_shoulder"]
    assert e2["params"].get("shadows", 0) > 0
    assert "backlit_midtone" in e2["flags"]
    assert _clip_share(px_c1, RenderParams(**e2["params"])) <= NOREF_V3["clip_budget"]

    # 3. Low span does not trigger rescue
    e_low = estimate_params_v3(px_c1, None, NOREF_V3, span_d_g=0.8)
    assert "rescue_weight" not in e_low["diagnostics"]["backlit_check"]

    # 4. Night scene (med < 0.24) does not trigger rescue
    y_night = np.zeros(n)
    y_night[:80000] = rng.uniform(0.005, 0.02, 80000)
    y_night[80000:97000] = rng.uniform(0.05, 0.10, 17000)
    y_night[97000:] = 1.8
    px_night = np.stack([y_night * 2 ** c[:, 0], y_night, y_night * 2 ** c[:, 1]], -1).reshape(200, 500, 3).astype(np.float32)
    e_night = estimate_params_v3(px_night, None, NOREF_V3, span_d_g=1.1)
    assert "rescue_weight" not in e_night["diagnostics"]["backlit_check"]

    # 5. Regular backlit does not double-stack rescue
    P_bl = backlit_scene(fg=0.05, sky=0.5, seed=31)
    e_bl = estimate_params_v3(P_bl, None, NOREF_V3, span_d_g=1.2)
    bc_bl = e_bl["diagnostics"]["backlit_check"]
    assert bc_bl["weight"] > 0
    assert "rescue_weight" not in bc_bl


def test_t4o_rescue_rolls_back_when_shoulder_cannot_meet_clip_budget(monkeypatch):
    from filmcolor import noref

    rng = np.random.default_rng(42)
    y = np.concatenate([rng.uniform(0.02, 0.06, 80_000), rng.uniform(0.12, 0.22, 17_000), np.full(3_000, 1.8)])
    c = rng.normal(0, 0.05, (len(y), 2))
    P = np.stack([y * 2 ** c[:, 0], y, y * 2 ** c[:, 1]], -1).reshape(200, 500, 3).astype(np.float32)
    original = noref._clip_share

    def reject_lift(px, params):
        if params.midtone > noref.NOREF_V3["base_midtone"]:
            return 0.01
        return original(px, params)

    monkeypatch.setattr(noref, "_clip_share", reject_lift)
    off = noref.estimate_params_v3(P, None, dict(noref.NOREF_V3, backlit_capped_rescue=False), span_d_g=1.1)
    on = noref.estimate_params_v3(P, None, noref.NOREF_V3, span_d_g=1.1)
    assert on["diagnostics"]["backlit_check"]["rescue_c1"]
    assert on["diagnostics"]["backlit_check"]["rescue_weight"] == 0
    assert on["params"] == off["params"] and "backlit_midtone" not in on["flags"]


def test_t4l_too_short_a_density_range_is_left_for_manual_review():
    """T4l (A1141, out of focus: 0.145 D): an inversion G density range below min_span_d_g gives
    manual_review_required with neutral parameters; None turns the check off, and without span_d_g nothing changes."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3

    P = scene(median=0.18, stops=4.0, seed=94)
    e = estimate_params_v3(P, None, NOREF_V3, span_d_g=0.15)
    assert e["status"] == "manual_review_required" and any("density range" in f for f in e["flags"])
    assert e["params"]["ev"] == 0 and e["params"].get("temp", 0) == 0
    assert estimate_params_v3(P, None, dict(NOREF_V3, min_span_d_g=None), span_d_g=0.15)["status"] == "ok"
    assert estimate_params_v3(P, None, NOREF_V3)["status"] == "ok"


def test_t4p_highlight_hold_ev_max_drop_and_bounds():
    """T4p: Step 5 highlight hold EV drop is bounded by ev_highlight_max_drop (default 0.60 EV),
    supports fractional maxdrop (e.g. 0.25), respects ev_limits lower floor, and setting 3.0 restores
    the uncapped drop path."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3

    # Scene with positive anchor EV and sparse bright highlights requiring large EV drop to clear clip budget
    P = scene(median=0.03, stops=5.0, seed=12)
    flat = P.reshape(-1, 3)
    flat[:int(0.005 * len(flat))] = 2.7

    # 1. Default T4p bounds drop to 0.60 EV
    e_def = estimate_params_v3(P, None, NOREF_V3)
    hh_def = e_def["diagnostics"]["highlight_hold"]
    assert hh_def["ev_lowered"] == pytest.approx(0.60)
    assert hh_def["max_drop"] == 0.60
    assert "ev_lowered_for_highlights:0.6" in e_def["flags"]

    # 2. Config max_drop=3.0 restores old deeper drop behavior (> 0.60 EV)
    e_old = estimate_params_v3(P, None, dict(NOREF_V3, ev_highlight_max_drop=3.0))
    hh_old = e_old["diagnostics"]["highlight_hold"]
    assert hh_old["ev_lowered"] == pytest.approx(0.80)
    assert e_old["params"]["ev"] < e_def["params"]["ev"] - 0.15

    # 3. Fractional maxdrop (e.g. 0.25)
    e_frac = estimate_params_v3(P, None, dict(NOREF_V3, ev_highlight_max_drop=0.25))
    hh_frac = e_frac["diagnostics"]["highlight_hold"]
    assert hh_frac["ev_lowered"] == pytest.approx(0.25)
    assert e_frac["params"]["ev"] == pytest.approx(e_def["params"]["ev"] + 0.35)

    # 4. ev_limits lower bound halts drop before max_drop is reached
    e_floor = estimate_params_v3(P, None, dict(NOREF_V3, ev_limits=(1.80, 2.5), ev_highlight_max_drop=1.5))
    hh_floor = e_floor["diagnostics"]["highlight_hold"]
    assert e_floor["params"]["ev"] == pytest.approx(1.80)
    assert hh_floor["ev_lowered"] == pytest.approx(0.50)


def test_t4p_rescue_pre_hold_evidence_when_only_highlights_compressed():
    """T4p: when Step 5 clears the clip budget by adjusting shoulder and highlights alone (ev_lowered == 0),
    backlit capped rescue uses pre-hold Step 4b evidence for median and q25, with q90 = max(pre_q90, cur_q90),
    so highlight compression alone does not pull q90 below threshold and rescue triggers on pre-hold evidence."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3, RenderParams, _encoded_quantiles

    rng = np.random.default_rng(42)
    n = 100_000
    y = np.zeros(n)
    y[:80000] = rng.uniform(0.02, 0.06, 80000)
    y[80000:97000] = rng.uniform(0.10, 0.155, 17000)
    y[97000:] = 1.8
    c = rng.normal(0, 0.05, (n, 2))
    P = np.stack([y * 2 ** c[:, 0], y, y * 2 ** c[:, 1]], -1).reshape(200, 500, 3).astype(np.float32)

    # 1. Run without rescue to capture the actual Step 5 parameters before rescue
    e_off = estimate_params_v3(P, None, dict(NOREF_V3, backlit_capped_rescue=False), span_d_g=1.1)
    p_step5 = RenderParams(**e_off["params"])
    _, q90_held = _encoded_quantiles(P.reshape(-1, 3), p_step5, qs=(25, 90))

    # 2. Run with rescue active
    e_on = estimate_params_v3(P, None, NOREF_V3, span_d_g=1.1)
    hh = e_on["diagnostics"]["highlight_hold"]
    bc = e_on["diagnostics"]["backlit_check"]
    r = e_on["diagnostics"]["ev_refine"]

    assert hh["ev_lowered"] == 0.0
    # Pre-hold Step 4b evidence is preserved for median and q25:
    assert bc["rescue_med_before"] == r["med_tex_after"]
    assert bc["rescue_q25"] == bc["p25"]

    # True Step 5 highlight compression pulled post-hold q90 below capped_q90_min (0.45):
    assert q90_held < NOREF_V3["backlit_capped_q90_min"]
    # But rescue kept pre-hold Step 4b evidence (bc["p90"] >= 0.45) via max(q90_pre, q90_cur), triggering c1 rescue:
    assert bc["rescue_q90"] == max(float(bc["p90"]), q90_held)
    assert bc["rescue_q90"] >= NOREF_V3["backlit_capped_q90_min"]
    assert bc["rescue_c1"] is True
    assert bc["rescue_weight"] > 0


def test_t4p_bright_share_gate_and_quads_fixed_radius():
    """T4p/T4q: split_sky triggers when zone_disagreement > 0.22 and (top_bottom_spread > 0.22 or
    bright_px_share >= 0.45). A high-brightness frame with zone disagreement (bright_share >= 0.45)
    is identified as split_sky and suppresses false cool cast; a low-brightness frame with the same
    zone disagreement (bright_share < 0.45) preserves cool_cast.
    T4q: quads restored to fixed radius max(r for _, r, _ in cfg['wb_v3_zones']) (0.35), so quads
    do not expand to adaptive zone radii."""
    from filmcolor.noref import NOREF_V3, estimate_wb_v3

    def cool_scene(median_y=0.18, seed=61):
        rng = np.random.default_rng(seed)
        n = 40000
        y = median_y * 2 ** rng.uniform(-2, 2, n)
        grey = rng.random(n) < 0.4
        low = y < np.percentile(y, 70)
        # Zone 0 (>= 70th percentile): cool with z0 = [-0.28, 0.06]
        # Zone 1 (< 70th percentile): cool with z1 = [-0.11, 0.22]
        # Disagreement between z0 and z1: norm([-0.17, -0.16]) = 0.2335 > 0.22
        g = np.where(low[:, None], rng.normal([-0.11, 0.22], 0.02, (n, 2)), rng.normal([-0.28, 0.06], 0.02, (n, 2)))
        c = np.where(grey[:, None], g, rng.normal([-0.6, 0.8], 0.1, (n, 2)))
        return np.stack([y * 2 ** c[:, 0], y, y * 2 ** c[:, 1]], -1).astype(np.float32)

    # 1. Low brightness (bright_px_share < 0.45): cool cast is preserved
    dark_p = cool_scene(median_y=0.18)
    wb_dark = estimate_wb_v3(dark_p, NOREF_V3)
    d_dark = wb_dark["diagnostics"]
    assert d_dark["zone_disagreement"] > NOREF_V3["wb_v3_cast_max_zone_dis"]
    assert d_dark["top_bottom_spread"] <= NOREF_V3["wb_v3_cast_max_tb_spread"]
    assert d_dark["bright_px_share"] < NOREF_V3["wb_v3_cast_max_bright_share"]
    assert not d_dark["split_sky"]
    assert d_dark["cool_cast"] is True

    # 2. High brightness (bright_px_share >= 0.45): false cool cast inhibited via split_sky
    bright_p = cool_scene(median_y=0.75)
    wb_bright = estimate_wb_v3(bright_p, NOREF_V3)
    d_bright = wb_bright["diagnostics"]
    assert d_bright["zone_disagreement"] > NOREF_V3["wb_v3_cast_max_zone_dis"]
    assert d_bright["top_bottom_spread"] <= NOREF_V3["wb_v3_cast_max_tb_spread"]
    assert d_bright["bright_px_share"] >= NOREF_V3["wb_v3_cast_max_bright_share"]
    assert d_bright["split_sky"] is True
    assert not d_bright["cool_cast"]

    # 3. Quads fixed radius: quads use fixed radius max(r for _, r, _ in cfg['wb_v3_zones'])
    # When all pixels have chroma norm > 0.35, quads finds no pixels even though adaptive zones expand
    rng_fixed = np.random.default_rng(88)
    n = 40000
    y_f = 0.18 * 2 ** rng_fixed.uniform(-1, 1, n)
    theta = rng_fixed.uniform(0, 2 * np.pi, n)
    radius = rng_fixed.uniform(0.36, 0.40, n)
    c_f = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    px_f = np.stack([y_f * 2 ** c_f[:, 0], y_f, y_f * 2 ** c_f[:, 1]], -1).astype(np.float32)

    r_quad_fixed = max(r for _, r, _ in NOREF_V3["wb_v3_zones"])
    assert r_quad_fixed == 0.35
    wb_f = estimate_wb_v3(px_f, NOREF_V3)
    d_f = wb_f["diagnostics"]
    # Zones adaptively expand radius_effective beyond 0.35:
    assert any(z["radius_effective"] > r_quad_fixed for z in d_f["zones"])
    # But quads stays strictly bounded by fixed r_quad_fixed (0.35), so no quadrant meets min_px:
    assert "top_bottom_spread" not in d_f

    # When pixels with chroma norm <= 0.35 are present in top and bottom halves:
    c_f_pass = c_f.copy()
    half = n // 2
    c_f_pass[:1200] = [0.1, 0.1]
    c_f_pass[half:half + 1200] = [-0.1, -0.1]
    px_f_pass = np.stack([y_f * 2 ** c_f_pass[:, 0], y_f, y_f * 2 ** c_f_pass[:, 1]], -1).astype(np.float32)
    wb_f_pass = estimate_wb_v3(px_f_pass, NOREF_V3)
    assert "top_bottom_spread" in wb_f_pass["diagnostics"]


def test_t4q_overlift_damp_end_to_end_and_diagnostics():
    """T4q: Step 5 rescue-to-Step 6 overlift_damp check:
    - Triggers when base_p['ev'] >= overlift_ev_min (1.50) and Step 4b q25 >= overlift_q25_min (0.35).
    - Computes ramp_ev, pull, and blacks drop according to formula and clamps pull <= max_pull (0.65).
    - Lowers ev and blacks; sets diagnostics overlift_damp.
    - Does NOT trigger when EV < 1.50 or q25 < 0.35, or when overlift_damp_check=False.
    - Validates boundary thresholds (= ev_min, = q25_min) and max_pull capping.
    - Verifies Step 4b backlit and Step 5 highlight hold / rescue diagnostics remain invariant ON vs OFF."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3

    rng = np.random.default_rng(42)
    n = 100_000
    y = rng.uniform(0.02, 0.08, n)
    P = np.stack([y, y, y], -1).reshape(200, 500, 3).astype(np.float32)

    # Isolate the legacy high-EV branch from T4w's independently enabled moderate branch.
    cfg = dict(NOREF_V3, overlift_mod_check=False)
    res_on = estimate_params_v3(P, None, cfg, span_d_g=1.1)
    res_off = estimate_params_v3(P, None, dict(cfg, overlift_damp_check=False), span_d_g=1.1)

    assert "overlift_damp" in res_on["diagnostics"]
    od = res_on["diagnostics"]["overlift_damp"]
    assert od["applied"] is True
    assert "overlift_damp" not in res_off["diagnostics"]

    # Verify mathematical formula for pull and blacks drop:
    ev0 = od["ev_before"]
    q25 = od["q25_before"]
    assert ev0 >= NOREF_V3["overlift_ev_min"]
    assert q25 >= NOREF_V3["overlift_q25_min"]
    assert ev0 == pytest.approx(res_off["params"]["ev"], abs=1e-4)

    expected_ramp = float(np.clip((ev0 - 1.30) / 0.50, 0.0, 1.0))
    expected_pull = float(np.clip(3.2 * (q25 - 0.33) * expected_ramp, 0.0, 0.65))
    expected_blk_drop = float(min(0.06, expected_pull * 0.10))
    expected_ev_after = ev0 - expected_pull

    assert od["pull"] == pytest.approx(expected_pull, abs=1e-6)
    assert od["ev_after"] == pytest.approx(expected_ev_after, abs=1e-6)
    assert od["blacks_lowered"] == pytest.approx(expected_blk_drop, abs=1e-6)
    assert res_on["params"]["ev"] == round(expected_ev_after, 4)
    assert res_on["params"]["blacks"] == round(-expected_blk_drop, 4)
    assert res_on["params"]["ev"] < res_off["params"]["ev"]

    # Invariance of Step 4b backlit & Step 5 highlight hold diagnostics between ON and OFF:
    assert res_on["diagnostics"]["backlit_check"] == res_off["diagnostics"]["backlit_check"]
    assert res_on["diagnostics"]["highlight_hold"] == res_off["diagnostics"]["highlight_hold"]

    # 2. Invariance on backlit rescue frame (Step 5 highlight / rescue diagnostics match):
    y_res = np.zeros(n)
    y_res[:80000] = rng.uniform(0.02, 0.06, 80000)
    y_res[80000:97000] = rng.uniform(0.12, 0.22, 17000)
    y_res[97000:] = 1.8
    c_res = rng.normal(0, 0.05, (n, 2))
    P_res = np.stack([y_res * 2 ** c_res[:, 0], y_res, y_res * 2 ** c_res[:, 1]], -1).reshape(200, 500, 3).astype(np.float32)

    res_res_on = estimate_params_v3(P_res, None, NOREF_V3, span_d_g=1.1)
    res_res_off = estimate_params_v3(P_res, None, dict(NOREF_V3, overlift_damp_check=False), span_d_g=1.1)
    assert res_res_on["diagnostics"]["backlit_check"] == res_res_off["diagnostics"]["backlit_check"]
    assert res_res_on["diagnostics"]["highlight_hold"] == res_res_off["diagnostics"]["highlight_hold"]

    # 3. Non-trigger case: current EV < 1.50 (high median scene)
    y_med = rng.uniform(0.10, 0.25, n)
    P_med = np.stack([y_med, y_med, y_med], -1).reshape(200, 500, 3).astype(np.float32)
    res_med = estimate_params_v3(P_med, None, NOREF_V3)
    assert res_med["params"]["ev"] < 1.50
    assert "overlift_damp" not in res_med["diagnostics"]

    # 4. Non-trigger case: Step 4b q25 < 0.35 (wide dynamic range scene with dark shadows)
    y_wide = np.concatenate([rng.uniform(0.005, 0.02, 35000), rng.uniform(0.02, 0.15, 65000)])
    P_wide = np.stack([y_wide, y_wide, y_wide], -1).reshape(200, 500, 3).astype(np.float32)
    res_wide = estimate_params_v3(P_wide, None, NOREF_V3)
    assert res_wide["params"]["ev"] >= 1.50
    assert res_wide["diagnostics"]["backlit_check"]["p25"] < 0.35
    assert "overlift_damp" not in res_wide["diagnostics"]

    # 5. Boundary equality tests (= ev_min, = q25_min)
    # Exactly at ev_min: triggers
    res_ev_bnd = estimate_params_v3(P, None, dict(NOREF_V3, overlift_ev_min=ev0))
    assert res_ev_bnd["diagnostics"].get("overlift_damp", {}).get("applied") is True
    # Strictly above ev_min: does not trigger
    res_ev_bnd_fail = estimate_params_v3(P, None, dict(NOREF_V3, overlift_ev_min=ev0 + 1e-4))
    assert "overlift_damp" not in res_ev_bnd_fail["diagnostics"]

    # Exactly at q25_min: triggers
    res_q25_bnd = estimate_params_v3(P, None, dict(NOREF_V3, overlift_q25_min=q25))
    assert res_q25_bnd["diagnostics"].get("overlift_damp", {}).get("applied") is True
    # Strictly above q25_min: does not trigger
    res_q25_bnd_fail = estimate_params_v3(P, None, dict(NOREF_V3, overlift_q25_min=q25 + 1e-4))
    assert "overlift_damp" not in res_q25_bnd_fail["diagnostics"]

    # 6. Max pull limit clamping
    res_clamped = estimate_params_v3(P, None, dict(NOREF_V3, overlift_max_pull=0.20))
    od_clamped = res_clamped["diagnostics"]["overlift_damp"]
    assert od_clamped["pull"] == 0.20
    assert od_clamped["ev_after"] == pytest.approx(ev0 - 0.20, abs=1e-6)
    assert od_clamped["blacks_lowered"] == pytest.approx(0.02, abs=1e-6)


def test_t4r_overlift_wide_bright_exemption_and_damping_conditions():
    """T4r: scenes with wide negative density span (span_d_g >= overlift_max_span = 1.15) AND bright content
    (bright_share >= overlift_max_bright_share = 0.05) are exempt from overlift damping even when base EV >= 1.50
    and q25 >= 0.35.
    - Both met (span >= 1.15 and bright_share >= 0.05): overlift_damp is bypassed.
    - Only wide (span >= 1.15, bright_share < 0.05): damping continues.
    - Only bright (span < 1.15, bright_share >= 0.05): damping continues.
    - Neither (span < 1.15, bright_share < 0.05): damping continues.
    - span_d_g=None (unknown span): continues legacy damping behavior regardless of bright_share.
    - Boundary checks: exactly at span=1.15 (exempt) vs span < 1.15 (damped); exactly at bright_share=0.05."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3

    rng = np.random.default_rng(42)
    n = 100_000
    y_base = rng.uniform(0.02, 0.08, n)

    # 1. Wide and bright scene: bright_share >= 0.05 (6000 px = 6.0% at 0.46)
    y_wb = y_base.copy()
    y_wb[:6000] = 0.46
    P_wb = np.stack([y_wb, y_wb, y_wb], -1).reshape(200, 500, 3).astype(np.float32)

    # Verify scene properties when undamped
    res_exempt = estimate_params_v3(P_wb, None, NOREF_V3, span_d_g=1.20)
    assert res_exempt["diagnostics"]["tone_pixels"]["bright_share"] >= NOREF_V3["overlift_max_bright_share"]
    assert res_exempt["diagnostics"]["backlit_check"]["p25"] >= NOREF_V3["overlift_q25_min"]
    assert res_exempt["params"]["ev"] >= NOREF_V3["overlift_ev_min"]
    # Damping exempt:
    assert "overlift_damp" not in res_exempt["diagnostics"]

    # 2. Only wide (span=1.20 >= 1.15, but bright_share < 0.05: 2000 px = 2.0%)
    y_ow = y_base.copy()
    y_ow[:2000] = 0.46
    P_ow = np.stack([y_ow, y_ow, y_ow], -1).reshape(200, 500, 3).astype(np.float32)
    res_ow = estimate_params_v3(P_ow, None, NOREF_V3, span_d_g=1.20)
    assert res_ow["diagnostics"]["tone_pixels"]["bright_share"] < NOREF_V3["overlift_max_bright_share"]
    assert "overlift_damp" in res_ow["diagnostics"]
    assert res_ow["diagnostics"]["overlift_damp"]["applied"] is True

    # 3. Only bright (bright_share >= 0.05, but span=1.10 < 1.15)
    res_ob = estimate_params_v3(P_wb, None, NOREF_V3, span_d_g=1.10)
    assert "overlift_damp" in res_ob["diagnostics"]
    assert res_ob["diagnostics"]["overlift_damp"]["applied"] is True

    # 4. Neither wide nor bright (span=1.10 < 1.15, bright_share < 0.05)
    res_neither = estimate_params_v3(P_ow, None, NOREF_V3, span_d_g=1.10)
    assert "overlift_damp" in res_neither["diagnostics"]
    assert res_neither["diagnostics"]["overlift_damp"]["applied"] is True

    # 5. span_d_g=None: legacy behavior continues (damped even when bright_share >= 0.05)
    res_none = estimate_params_v3(P_wb, None, NOREF_V3, span_d_g=None)
    assert "overlift_damp" in res_none["diagnostics"]
    assert res_none["diagnostics"]["overlift_damp"]["applied"] is True

    # 6. Boundary conditions for span_d_g threshold (1.15)
    res_span_exact = estimate_params_v3(P_wb, None, NOREF_V3, span_d_g=1.15)
    assert "overlift_damp" not in res_span_exact["diagnostics"]

    res_span_below = estimate_params_v3(P_wb, None, NOREF_V3, span_d_g=1.15 - 1e-4)
    assert "overlift_damp" in res_span_below["diagnostics"]
    assert res_span_below["diagnostics"]["overlift_damp"]["applied"] is True

    # 7. Boundary conditions for bright_share threshold (0.05)
    bs_actual = res_exempt["diagnostics"]["tone_pixels"]["bright_share"]
    # Exact threshold: exempt
    res_bs_exact = estimate_params_v3(P_wb, None, dict(NOREF_V3, overlift_max_bright_share=bs_actual), span_d_g=1.20)
    assert "overlift_damp" not in res_bs_exact["diagnostics"]
    # Strictly above actual bright_share: damped
    res_bs_above = estimate_params_v3(P_wb, None, dict(NOREF_V3, overlift_max_bright_share=bs_actual + 1e-4), span_d_g=1.20)
    assert "overlift_damp" in res_bs_above["diagnostics"]
    assert res_bs_above["diagnostics"]["overlift_damp"]["applied"] is True


def test_t4r_split_sky_no_tb_spread_and_threshold_boundaries():
    """T4r: when top_bottom_spread cannot be measured (e.g. sparse neutral evidence in quads), split_sky uses a lower
    bright_px_share threshold (wb_v3_cast_max_bright_share_no_tb = 0.12) instead of the standard 0.45.
    - Without top_bottom_spread:
      * bright_px_share >= 0.12 triggers split_sky=True and suppresses false cool cast.
      * bright_px_share < 0.12 keeps split_sky=False and preserves cool cast.
    - With top_bottom_spread present:
      * bright_px_share between 0.12 and 0.45 (with tb_spread <= 0.22) does NOT trigger split_sky (uses 0.45).
    - Boundary checks around 0.12 threshold without top_bottom_spread."""
    from filmcolor.noref import NOREF_V3, estimate_wb_v3

    def make_cool_scene(median_y, suppress_quads=True, seed=61):
        rng = np.random.default_rng(seed)
        n = 40000
        y = median_y * 2 ** rng.uniform(-2, 2, n)
        grey = rng.random(n) < 0.4
        low = y < np.percentile(y, 70)
        if suppress_quads:
            # Chroma norm in [0.36, 0.40], strictly above fixed quads radius 0.35 -> no quads pixels
            # Zone 0 vs Zone 1 diff: [-0.08 - (-0.30), 0.38 - 0.22] = [0.22, 0.16] -> norm 0.272 > 0.22
            g = np.where(low[:, None], rng.normal([-0.08, 0.38], 0.01, (n, 2)), rng.normal([-0.30, 0.22], 0.01, (n, 2)))
        else:
            # Chroma within quads radius 0.35 -> top_bottom_spread is measured
            g = np.where(low[:, None], rng.normal([-0.11, 0.22], 0.02, (n, 2)), rng.normal([-0.28, 0.06], 0.02, (n, 2)))
        c = np.where(grey[:, None], g, rng.normal([-0.6, 0.8], 0.1, (n, 2)))
        return np.stack([y * 2 ** c[:, 0], y, y * 2 ** c[:, 1]], -1).astype(np.float32)

    # 1. No top_bottom_spread, bright_px_share >= 0.12 (median_y=0.18 gives ~0.160 > 0.12):
    # Under T4r no-tb threshold (0.12), split_sky triggers and suppresses cool_cast.
    # Note: under old 0.45 threshold, 0.160 would NOT trigger split_sky.
    px_no_tb_bright = make_cool_scene(0.18, suppress_quads=True)
    res_no_tb_bright = estimate_wb_v3(px_no_tb_bright, NOREF_V3)
    d_ntb_b = res_no_tb_bright["diagnostics"]
    assert "top_bottom_spread" not in d_ntb_b
    assert d_ntb_b["zone_disagreement"] > NOREF_V3["wb_v3_cast_max_zone_dis"]
    assert d_ntb_b["bright_px_share"] >= NOREF_V3["wb_v3_cast_max_bright_share_no_tb"]
    assert d_ntb_b["bright_px_share"] < NOREF_V3["wb_v3_cast_max_bright_share"]
    assert d_ntb_b["split_sky"] is True
    assert not d_ntb_b["cool_cast"]

    # Under explicit old config (setting no_tb threshold to 0.45): split_sky is False
    res_old_cfg = estimate_wb_v3(px_no_tb_bright, dict(NOREF_V3, wb_v3_cast_max_bright_share_no_tb=0.45))
    assert res_old_cfg["diagnostics"]["split_sky"] is False
    assert res_old_cfg["diagnostics"]["cool_cast"] is True

    # 2. No top_bottom_spread, dark scene (median_y=0.10 gives bright_share=0.0 < 0.12):
    # split_sky is False, cool_cast preserved.
    px_no_tb_dark = make_cool_scene(0.10, suppress_quads=True)
    res_no_tb_dark = estimate_wb_v3(px_no_tb_dark, NOREF_V3)
    d_ntb_d = res_no_tb_dark["diagnostics"]
    assert "top_bottom_spread" not in d_ntb_d
    assert d_ntb_d["zone_disagreement"] > NOREF_V3["wb_v3_cast_max_zone_dis"]
    assert d_ntb_d["bright_px_share"] < NOREF_V3["wb_v3_cast_max_bright_share_no_tb"]
    assert not d_ntb_d["split_sky"]
    assert d_ntb_d["cool_cast"] is True

    # 3. With top_bottom_spread present and <= 0.22:
    # bright_px_share ~0.159 is compared against 0.45 (not 0.12) -> split_sky is False!
    px_with_tb = make_cool_scene(0.18, suppress_quads=False)
    res_with_tb = estimate_wb_v3(px_with_tb, NOREF_V3)
    d_wtb = res_with_tb["diagnostics"]
    assert "top_bottom_spread" in d_wtb
    assert d_wtb["top_bottom_spread"] <= NOREF_V3["wb_v3_cast_max_tb_spread"]
    assert d_wtb["zone_disagreement"] > NOREF_V3["wb_v3_cast_max_zone_dis"]
    assert d_wtb["bright_px_share"] >= NOREF_V3["wb_v3_cast_max_bright_share_no_tb"]
    assert d_wtb["bright_px_share"] < NOREF_V3["wb_v3_cast_max_bright_share"]
    assert not d_wtb["split_sky"]
    assert d_wtb["cool_cast"] is True

    # 4. Threshold boundary checks on no_tb threshold:
    br_share = d_ntb_b["bright_px_share"]
    # Exactly at threshold: triggers
    r_exact = estimate_wb_v3(px_no_tb_bright, dict(NOREF_V3, wb_v3_cast_max_bright_share_no_tb=br_share))
    assert r_exact["diagnostics"]["split_sky"] is True
    # Strictly above threshold: does not trigger
    r_above = estimate_wb_v3(px_no_tb_bright, dict(NOREF_V3, wb_v3_cast_max_bright_share_no_tb=br_share + 1e-4))
    assert r_above["diagnostics"]["split_sky"] is False
    assert r_above["diagnostics"]["cool_cast"] is True


def test_t4r_cyan_confidence_boost_zone_disagreement_gate():
    """T4r: cyan confidence boost (wb_v3_cyan_conf_check) requires zone_disagreement <= wb_v3_cyan_max_zone_dis (0.60)
    in addition to est[0] < -0.85, est[1] <= 0.20, and all zones narrow_share < 0.04.
    - When zone_disagreement <= 0.60: cyan confidence boost is applied (confidence floor >= 0.72).
    - When zone_disagreement > 0.60: conflicting scene colours suppress the boost.
    - Boundary checks around 0.60 threshold."""
    from filmcolor.noref import NOREF_V3, estimate_wb_v3

    rng = np.random.default_rng(42)
    n = 100_000
    y = 2 ** rng.uniform(-6, 0, n)
    p25, p70, p75, p96 = np.percentile(y, [25, 70, 75, 96])

    def make_cyan_scene(dis):
        c = np.zeros((n, 2), dtype=np.float32)
        z0 = (y >= p70) & (y <= p96)
        z1_only = (y >= p25) & (y < p70)
        rest = ~(z0 | z1_only)
        c[z0] = [-1.0 - dis / 2, -0.1]
        c[z1_only] = [-1.0 + dis / 2, -0.1]
        c[rest] = [-1.0, -0.1]
        return np.stack([y * 2 ** c[:, 0], y, y * 2 ** c[:, 1]], -1).astype(np.float32)

    # 1. Consistent cyan scene (dis = 0.40 -> zone_disagreement ~0.39 <= 0.60):
    px_pass = make_cyan_scene(0.40)
    res_pass = estimate_wb_v3(px_pass, NOREF_V3)
    d_pass = res_pass["diagnostics"]
    assert all(z["narrow_share"] < NOREF_V3["wb_v3_cyan_narrow_max"] for z in d_pass["zones"])
    assert res_pass["chroma"][0] < NOREF_V3["wb_v3_cyan_r_max"]
    assert res_pass["chroma"][1] <= NOREF_V3["wb_v3_cyan_b_max"]
    assert d_pass["zone_disagreement"] <= NOREF_V3["wb_v3_cyan_max_zone_dis"]
    assert d_pass.get("cyan_conf_boost") is True
    assert res_pass["confidence"] >= NOREF_V3["wb_v3_cyan_min_conf"]

    # 2. Conflicting cyan scene (dis = 0.70 -> zone_disagreement ~0.69 > 0.60):
    px_fail = make_cyan_scene(0.70)
    res_fail = estimate_wb_v3(px_fail, NOREF_V3)
    d_fail = res_fail["diagnostics"]
    assert all(z["narrow_share"] < NOREF_V3["wb_v3_cyan_narrow_max"] for z in d_fail["zones"])
    assert res_fail["chroma"][0] < NOREF_V3["wb_v3_cyan_r_max"]
    assert res_fail["chroma"][1] <= NOREF_V3["wb_v3_cyan_b_max"]
    assert d_fail["zone_disagreement"] > NOREF_V3["wb_v3_cyan_max_zone_dis"]
    # Boost is refused due to zone disagreement:
    assert "cyan_conf_boost" not in d_fail
    assert res_fail == estimate_wb_v3(px_fail, dict(NOREF_V3, wb_v3_cyan_conf_check=False))

    # 3. Boundary check on zone_disagreement threshold
    zd_pass = d_pass["zone_disagreement"]
    # Exact threshold: boost triggers
    res_bnd_pass = estimate_wb_v3(px_pass, dict(NOREF_V3, wb_v3_cyan_max_zone_dis=zd_pass))
    assert res_bnd_pass["diagnostics"].get("cyan_conf_boost") is True
    # Strictly below actual zone_disagreement: boost refused
    res_bnd_fail = estimate_wb_v3(px_pass, dict(NOREF_V3, wb_v3_cyan_max_zone_dis=zd_pass - 1e-4))
    assert "cyan_conf_boost" not in res_bnd_fail["diagnostics"]


def test_t4t_wb_v3_cyan_conf_requires_min_bright_share():
    """T4t: cyan confidence boost (wb_v3_cyan_conf_check) requires bright_px_share >= wb_v3_cyan_min_bright_share (0.10).
    Dark monochrome foliage (e.g. A239 with bright_px_share ~0.04 < 0.10) is intercepted and does NOT trigger cyan_conf_boost,
    preventing green foliage from being pushed to purple/magenta tint.
    Bright cyan scenes (e.g. bright_px_share ~0.25 >= 0.10) trigger cyan_conf_boost normally."""
    from filmcolor.noref import LUMA, NOREF_V3, estimate_wb_v3

    rng = np.random.default_rng(55)
    n = 100_000

    # Helper to generate scene with controlled true luma bright_px_share (px @ LUMA > bright_luma=0.45)
    # and consistent cyan chroma (est[0] < -0.85, est[1] <= 0.20, low zone_disagreement, narrow_share < 0.04)
    def make_scene_with_bright_share(bright_target):
        n_bright = int(n * bright_target)
        n_dark = n - n_bright
        y_bright = rng.uniform(0.46, 0.95, n_bright)
        y_dark = rng.uniform(0.02, 0.44, n_dark)
        y_target = np.concatenate([y_bright, y_dark]).astype(np.float32)

        # Base chromaticity ratios: log2(R/G) = -1.1, log2(B/G) = -0.1
        ratios = np.array([2.0 ** -1.1, 1.0, 2.0 ** -0.1], dtype=np.float32)
        unit_luma = float(np.dot(ratios, LUMA))
        unit_px = ratios / unit_luma

        # Add small jitter so norm is > radius (narrow_share < 0.04)
        c_jitter = rng.normal(0, 0.02, (n, 3)).astype(np.float32)
        c_jitter[:, 1] = 0.0
        px = y_target[:, None] * (unit_px[None, :] * (2.0 ** c_jitter))
        return px.astype(np.float32)

    # 1. Dark foliage scene: bright_px_share = 0.04 < 0.10 -> cyan_conf_boost must NOT trigger
    px_dark = make_scene_with_bright_share(0.04)
    res_dark = estimate_wb_v3(px_dark, NOREF_V3)
    d_dark = res_dark["diagnostics"]
    assert all(z["narrow_share"] < NOREF_V3["wb_v3_cyan_narrow_max"] for z in d_dark["zones"])
    assert res_dark["chroma"][0] < NOREF_V3["wb_v3_cyan_r_max"]
    assert res_dark["chroma"][1] <= NOREF_V3["wb_v3_cyan_b_max"]
    assert d_dark.get("zone_disagreement", 0.0) <= NOREF_V3["wb_v3_cyan_max_zone_dis"]
    # Key check: bright_px_share < 0.10 suppresses cyan boost
    assert "cyan_conf_boost" not in d_dark
    assert res_dark["confidence"] < NOREF_V3["wb_v3_cyan_min_conf"]

    # 2. Bright cyan scene: bright_px_share = 0.25 >= 0.10 -> cyan_conf_boost triggers
    px_bright = make_scene_with_bright_share(0.25)
    res_bright = estimate_wb_v3(px_bright, NOREF_V3)
    d_bright = res_bright["diagnostics"]
    assert all(z["narrow_share"] < NOREF_V3["wb_v3_cyan_narrow_max"] for z in d_bright["zones"])
    assert res_bright["chroma"][0] < NOREF_V3["wb_v3_cyan_r_max"]
    assert res_bright["chroma"][1] <= NOREF_V3["wb_v3_cyan_b_max"]
    assert d_bright.get("zone_disagreement", 0.0) <= NOREF_V3["wb_v3_cyan_max_zone_dis"]
    assert d_bright.get("cyan_conf_boost") is True
    assert res_bright["confidence"] >= NOREF_V3["wb_v3_cyan_min_conf"]


def test_t4t_wb_v3_extreme_zone_disagreement_damps_positive_tint():
    """T4t: when zone_disagreement > wb_v3_tint_max_zone_dis (1.25), extreme two-tone split scenes
    (e.g. A281 golden dome vs cyan shadows) damp positive tint by multiplying by
    wb_v3_tint_conflict_damp (0.55) and setting diag['tint_conflict_damped'] = True,
    instead of boosting by wb_v3_tint_boost (1.20).
    Negative tint and boundary condition (zone_disagreement <= 1.25) remain unaffected."""
    from filmcolor.noref import LUMA, NOREF_V3, estimate_wb_v3

    rng = np.random.default_rng(77)
    n = 100_000
    y_target = (2.0 ** rng.uniform(-5, 0, n)).astype(np.float32)
    p25, p70, p75, p96 = np.percentile(y_target, [25, 70, 75, 96])

    # Helper to create split-tone scene with specified zone disagreement and tint direction,
    # strictly normalizing by LUMA so true luma percentiles match p25/p70/p75/p96
    def make_split_scene(c_z0, c_z1):
        z0 = (y_target >= p70) & (y_target <= p96)
        z1 = (y_target >= p25) & (y_target < p70)
        c = np.zeros((n, 2), dtype=np.float32)
        c[z0] = c_z0
        c[z1] = c_z1
        rest = ~(z0 | z1)
        c[rest] = 0.5 * (np.array(c_z0) + np.array(c_z1))
        rgb_ratios = np.stack([2.0 ** c[:, 0], np.ones(n, dtype=np.float32), 2.0 ** c[:, 1]], axis=-1)
        unit_luma = rgb_ratios @ LUMA.astype(np.float32)
        px = (y_target / unit_luma)[:, None] * rgb_ratios
        return px.astype(np.float32)

    # 1. Extreme two-tone split: zone_disagreement ~ 1.50 > 1.25, positive tint raw
    c0 = [-0.5, -1.6]
    c1 = [-1.0, -0.2]
    px_extreme = make_split_scene(c0, c1)

    res_extreme = estimate_wb_v3(px_extreme, NOREF_V3)
    d_extreme = res_extreme["diagnostics"]
    assert d_extreme.get("zone_disagreement", 0.0) > 1.25
    # Raw tint must be positive
    assert d_extreme["temp_tint_raw"][1] > 0
    # Must be damped by 0.55 instead of boosted by 1.20:
    assert d_extreme.get("tint_conflict_damped") is True
    expected_damped_tint = d_extreme["temp_tint_raw"][1] * 0.55
    assert res_extreme["tint"] == pytest.approx(expected_damped_tint, abs=1e-5)

    # 2. Moderate disagreement: zone_disagreement <= 1.25 (e.g. diff norm ~ 0.8)
    # Positive tint gets normal 1.20 boost when neutral evidence is present in at least one zone
    c0_mod = [-0.5, -1.0]
    c1_mod = [-0.9, -0.4]
    px_mod = make_split_scene(c0_mod, c1_mod)
    # Provide 5% neutral evidence in z0 so no_narrow_tint_capped does not apply
    idx0 = np.where((y_target >= p70) & (y_target <= p96))[0]
    px_mod[idx0[:int(0.06 * len(idx0))]] = y_target[idx0[:int(0.06 * len(idx0)), None]]
    res_mod = estimate_wb_v3(px_mod, NOREF_V3)
    d_mod = res_mod["diagnostics"]
    assert d_mod.get("zone_disagreement", 0.0) <= 1.25
    assert d_mod.get("tint_conflict_damped") is not True
    assert "no_narrow_tint_capped" not in d_mod
    expected_boosted_tint = d_mod["temp_tint_raw"][1] * NOREF_V3["wb_v3_tint_boost"]
    assert res_mod["tint"] == pytest.approx(expected_boosted_tint, abs=1e-5)

    # 3. Negative tint raw with extreme disagreement: must NOT damp (tint <= 0 untouched)
    c0_neg = [0.0, 1.5]
    c1_neg = [0.0, 3.0]
    px_neg = make_split_scene(c0_neg, c1_neg)
    res_neg = estimate_wb_v3(px_neg, NOREF_V3)
    d_neg = res_neg["diagnostics"]
    assert d_neg.get("zone_disagreement", 0.0) > 1.25
    assert d_neg["temp_tint_raw"][1] < 0
    assert d_neg.get("tint_conflict_damped") is not True
    assert res_neg["tint"] == pytest.approx(d_neg["temp_tint_raw"][1], abs=1e-5)


def test_t4u_wb_v3_caps_positive_tint_when_both_zones_lack_narrow_neutrals():
    from filmcolor.noref import LUMA, NOREF_V3, estimate_wb_v3

    rng = np.random.default_rng(239)
    n = 100_000
    y_dark = rng.uniform(0.04, 0.35, n).astype(np.float32)

    c = np.full((n, 2), [-1.20, 0.00], dtype=np.float32)
    c += rng.normal(0, 0.01, (n, 2)).astype(np.float32)

    rgb_ratios = np.stack([2.0 ** c[:, 0], np.ones(n, dtype=np.float32), 2.0 ** c[:, 1]], axis=-1)
    unit_luma = rgb_ratios @ LUMA.astype(np.float32)
    px = (y_dark / unit_luma)[:, None] * rgb_ratios

    wb = estimate_wb_v3(px.astype(np.float32), NOREF_V3)
    d = wb["diagnostics"]

    assert not d.get("cool_cast")
    assert not d.get("cyan_conf_boost")
    assert len(d["zones"]) == 2
    assert all(z["narrow_share"] < NOREF_V3["wb_v3_cyan_narrow_max"] for z in d["zones"])
    assert d.get("zone_disagreement", 0.0) <= NOREF_V3.get("wb_v3_tint_max_zone_dis", np.inf)
    assert d["temp_tint_raw"][1] > 0.25
    assert d.get("no_narrow_tint_capped") is True
    assert wb["tint"] == pytest.approx(0.18)


def test_t4u_wb_v3_preserves_tint_when_narrow_neutrals_present_or_cyan_boosted():
    from filmcolor.noref import LUMA, NOREF_V3, estimate_wb_v3

    rng = np.random.default_rng(240)
    n = 100_000
    y_dark = rng.uniform(0.04, 0.35, n).astype(np.float32)

    # 1. When narrow-band neutral evidence is present in a zone (narrow_share >= 0.04)
    c_neutral = np.full((n, 2), [-0.85, 0.00], dtype=np.float32)
    p70, p96 = np.percentile(y_dark, [70, 96])
    z0 = (y_dark >= p70) & (y_dark <= p96)
    z0_indices = np.where(z0)[0]
    c_neutral[z0_indices[:int(0.06 * len(z0_indices))]] = [0.0, 0.0]

    rgb_ratios = np.stack([2.0 ** c_neutral[:, 0], np.ones(n, dtype=np.float32), 2.0 ** c_neutral[:, 1]], axis=-1)
    unit_luma = rgb_ratios @ LUMA.astype(np.float32)
    px_neutral = (y_dark / unit_luma)[:, None] * rgb_ratios

    wb_neutral = estimate_wb_v3(px_neutral.astype(np.float32), NOREF_V3)
    d_n = wb_neutral["diagnostics"]
    assert any(z["narrow_share"] >= NOREF_V3["wb_v3_cyan_narrow_max"] for z in d_n["zones"])
    assert "no_narrow_tint_capped" not in d_n
    assert d_n["temp_tint_raw"][1] > 0.25
    assert wb_neutral["tint"] == pytest.approx(d_n["temp_tint_raw"][1] * NOREF_V3["wb_v3_tint_boost"], abs=1e-5)

    # 2. When cyan_conf_boost is True (bright cyan scene)
    n_bright = int(0.25 * n)
    y_bright = np.concatenate([rng.uniform(0.46, 0.95, n_bright), rng.uniform(0.02, 0.44, n - n_bright)]).astype(np.float32)
    c_cyan = np.full((n, 2), [-1.10, -0.10], dtype=np.float32)
    c_cyan += rng.normal(0, 0.01, (n, 2)).astype(np.float32)
    rgb_ratios_c = np.stack([2.0 ** c_cyan[:, 0], np.ones(n, dtype=np.float32), 2.0 ** c_cyan[:, 1]], axis=-1)
    unit_luma_c = rgb_ratios_c @ LUMA.astype(np.float32)
    px_cyan = (y_bright / unit_luma_c)[:, None] * rgb_ratios_c

    wb_cyan = estimate_wb_v3(px_cyan.astype(np.float32), NOREF_V3)
    d_c = wb_cyan["diagnostics"]
    assert d_c.get("cyan_conf_boost") is True
    assert "no_narrow_tint_capped" not in d_c
    assert d_c["temp_tint_raw"][1] > 0.25
    assert wb_cyan["tint"] == pytest.approx(d_c["temp_tint_raw"][1] * NOREF_V3["wb_v3_tint_boost"], abs=1e-5)
    assert "no_narrow_tint_capped" not in d_c


def test_t4w_overlift_moderate_damps_bright_shadow_mid_ev_frames():
    """T4w: fills [0.85, 1.50) EV moderate overlift blind spot on non-wide frames with bright Step 4b shadows.
    Triggers diagnostics['overlift_damp']['mode'] == 'moderate' and lowers EV and blacks."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3

    rng = np.random.default_rng(42)
    n = 100_000
    y = rng.uniform(0.07 * 0.5, 0.07 * 1.5, n)
    P = np.stack([y, y, y], -1).reshape(200, 500, 3).astype(np.float32)

    res_on = estimate_params_v3(P, None, NOREF_V3, span_d_g=0.85)
    res_off = estimate_params_v3(P, None, dict(NOREF_V3, overlift_mod_check=False), span_d_g=0.85)

    assert "overlift_damp" in res_on["diagnostics"]
    od = res_on["diagnostics"]["overlift_damp"]
    assert od["applied"] is True
    assert od.get("mode") == "moderate"
    assert "overlift_damp" not in res_off["diagnostics"]

    ev0 = od["ev_before"]
    q25 = od["q25_before"]
    assert NOREF_V3["overlift_mod_ev_min"] <= ev0 < NOREF_V3["overlift_ev_min"]
    assert q25 >= NOREF_V3["overlift_mod_q25_min"]
    assert ev0 == pytest.approx(res_off["params"]["ev"], abs=1e-4)

    assert 0.0 < od["pull"] <= NOREF_V3["overlift_mod_max_pull"]
    assert od["ev_after"] < ev0
    assert od["blacks_lowered"] > 0.0
    assert res_on["params"]["ev"] < res_off["params"]["ev"]
    assert res_on["params"]["blacks"] < res_off["params"].get("blacks", 0.0)


def test_t4w_overlift_moderate_respects_bright_share_wide_span_and_preset_shadow_comp():
    """T4w: verifies moderate overlift damping respects bright_share, span_d_g, and preset shadow compensation."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3, preset_config

    rng = np.random.default_rng(42)
    n = 100_000
    y = rng.uniform(0.07 * 0.5, 0.07 * 1.5, n)
    P = np.stack([y, y, y], -1).reshape(200, 500, 3).astype(np.float32)

    # 1. Wide negative density span (span_d_g >= 1.15): no moderate damping
    res_wide = estimate_params_v3(P, None, NOREF_V3, span_d_g=1.15)
    assert "overlift_damp" not in res_wide["diagnostics"]

    # 2. Bright share >= 0.20 (25% pixels at 0.46): no moderate damping
    y_bright = y.copy()
    y_bright[:25000] = 0.46
    P_bright = np.stack([y_bright, y_bright, y_bright], -1).reshape(200, 500, 3).astype(np.float32)
    res_bright = estimate_params_v3(P_bright, None, NOREF_V3, span_d_g=0.85)
    assert res_bright["diagnostics"]["tone_pixels"]["bright_share"] >= 0.20
    assert "overlift_damp" not in res_bright["diagnostics"]

    # 3. Portrait preset shadow compensation protection (base_shadows=0.12 -> threshold 0.442):
    # Scene with q25 ~ 0.428 (< 0.442) has moderate damping bypassed due to shadow compensation.
    rng_p = np.random.default_rng(123)
    y_p = 0.06 * 2.0 ** rng_p.uniform(-1.5, 1.5, n)
    P_p = np.stack([y_p, y_p, y_p], -1).reshape(200, 500, 3).astype(np.float32)

    cfg_portrait = preset_config("portrait")
    assert cfg_portrait.get("base_shadows") == 0.12

    res_portrait = estimate_params_v3(P_p, None, cfg_portrait, span_d_g=0.85)
    q25_portrait = res_portrait["diagnostics"]["backlit_check"]["p25"]
    assert 0.40 <= q25_portrait < 0.40 + 0.35 * 0.12
    assert "overlift_damp" not in res_portrait["diagnostics"]

    # Without shadow compensation, the same scene triggers moderate damping:
    cfg_uncomp = dict(cfg_portrait, overlift_mod_shadow_comp=0.0)
    res_uncomp = estimate_params_v3(P_p, None, cfg_uncomp, span_d_g=0.85)
    assert res_uncomp["diagnostics"].get("overlift_damp", {}).get("mode") == "moderate"


def test_t4x_thin_flat_check_darkens_washed_out_short_span_frame():
    """T4x (fixes 19960812D-22-las-vegas): flat high-q25 thin negatives where stretched floor
    bypassed thin_bright_share_max (bright_share >= 0.42).
    - When span_d_g < 0.60, base_p['ev'] <= 0.0, q25 >= 0.62, and (q90 - q25) <= 0.145:
      triggers thin_negative with mode='flat_bright', pulling down EV and blacks."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3

    rng = np.random.default_rng(42)
    n = 100_000
    # Scene with range_stops >= 1.0 (so not low info), bright_share > 0.45,
    # and tightly clustered high-q25 / narrow IQR:
    y = np.concatenate([
        rng.uniform(0.12, 0.18, 2000),
        rng.uniform(0.48, 0.62, 96000),
        rng.uniform(0.80, 0.90, 2000),
    ]).astype(np.float32)
    rng.shuffle(y)
    P = np.stack([y, y, y], -1).reshape(200, 500, 3)

    span = 0.51  # < 0.60
    res = estimate_params_v3(P, None, NOREF_V3, span_d_g=span)

    assert res["status"] == "ok"
    tn = res["diagnostics"].get("thin_negative")
    assert tn is not None
    assert tn["applied"] is True
    assert tn.get("mode") == "flat_bright"
    assert tn["span_d_g"] == pytest.approx(span)
    assert tn["bright_share"] > 0.45
    assert tn["q25"] >= 0.64
    assert tn["iqr_25_90"] <= 0.10
    assert tn["pull"] > 0.0
    assert tn["blacks_lowered"] > 0.0
    assert res["params"]["ev"] < tn["ev_before"]
    assert res["params"]["blacks"] < 0.0


def test_t4x_thin_flat_check_skips_normal_iqr_highkey_frames():
    """T4x: thin_flat_check does NOT trigger on:
    1) High-key frames with normal dynamic gradation (iqr_25_90 > 0.145), even with short span.
    2) Dense or normal negatives where span_d_g >= 0.60."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3

    rng = np.random.default_rng(42)
    n = 100_000

    # 1. High-key frame with normal dynamic gradation (q25 >= 0.62 but iqr > 0.145):
    # e.g. q25 ~ 0.65, q90 ~ 0.80 -> iqr ~ 0.156 > 0.145
    y_normal_iqr = np.concatenate([
        rng.uniform(0.35, 0.40, 30000),
        rng.uniform(0.40, 0.70, 40000),
        rng.uniform(0.70, 0.90, 30000),
    ]).astype(np.float32)
    P_normal_iqr = np.stack([y_normal_iqr, y_normal_iqr, y_normal_iqr], -1).reshape(200, 500, 3)

    res_iqr = estimate_params_v3(P_normal_iqr, None, NOREF_V3, span_d_g=0.51)
    tn_iqr = res_iqr["diagnostics"].get("thin_negative")
    bc_iqr = res_iqr["diagnostics"]["backlit_check"]
    assert bc_iqr["p25"] >= 0.62
    assert (bc_iqr["p90"] - bc_iqr["p25"]) > 0.145
    assert tn_iqr is None or not tn_iqr.get("applied") or tn_iqr.get("mode") != "flat_bright"

    # 2. Scene with span_d_g = 0.65 >= 0.60:
    y_flat = np.concatenate([
        rng.uniform(0.12, 0.18, 2000),
        rng.uniform(0.48, 0.62, 96000),
        rng.uniform(0.80, 0.90, 2000),
    ]).astype(np.float32)
    rng.shuffle(y_flat)
    P_flat = np.stack([y_flat, y_flat, y_flat], -1).reshape(200, 500, 3)

    res_span = estimate_params_v3(P_flat, None, NOREF_V3, span_d_g=0.65)
    tn_span = res_span["diagnostics"].get("thin_negative")
    assert tn_span is None or not tn_span.get("applied") or tn_span.get("mode") != "flat_bright"


def test_t4x_ultra_flat_bright_recovers_contrast_on_overexposed_plateau_but_skips_narrow_cloud_frame():
    """T4x (fixes A1004.DNG, A1033.CR2):
    1) Overexposed plateau negative: all pixels sit on dense shoulder, range_stops >= 1.20,
       q25 >= 0.72, and iqr_25_90 <= 0.068:
       - Triggers ultra_flat_bright: applied=True
       - Pulls down EV and blacks, while boosting contrast.
    2) Narrow dynamic range cloud frame (e.g. A135, range_stops < 1.20):
       - Strictly skips ultra_flat_bright, preserving soft flat clouds."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3

    rng = np.random.default_rng(42)
    n = 100_000

    # 1. Overexposed plateau with ground/subject range_stops >= 1.20 (e.g. ~2.5 stops)
    y_uf = np.concatenate([
        rng.uniform(0.12, 0.18, 2000),
        rng.uniform(0.68, 0.76, 96000),
        rng.uniform(0.85, 0.95, 2000),
    ]).astype(np.float32)
    rng.shuffle(y_uf)
    P_uf = np.stack([y_uf, y_uf, y_uf], -1).reshape(200, 500, 3)

    # Use span_d_g = 1.0 (>= 0.60) to isolate from thin_flat_check
    res_uf = estimate_params_v3(P_uf, None, NOREF_V3, span_d_g=1.0)
    assert res_uf["stats"]["range_stops"] >= 1.20
    uf = res_uf["diagnostics"].get("ultra_flat_bright")
    assert uf is not None
    assert uf["applied"] is True
    assert uf["pull"] > 0.0
    assert uf["blacks_lowered"] > 0.0
    assert uf["contrast_after"] > NOREF_V3["base_contrast"]
    assert res_uf["params"]["ev"] < uf["ev_before"]
    assert res_uf["params"]["blacks"] < 0.0
    assert res_uf["params"]["contrast"] == pytest.approx(uf["contrast_after"], abs=5e-5)

    # 2. Narrow cloud frame: range_stops < 1.20 (e.g. ~0.95 stops, modeled after A135)
    y_cloud = np.concatenate([
        rng.uniform(0.38, 0.42, 2000),
        rng.uniform(0.68, 0.76, 96000),
        rng.uniform(0.78, 0.82, 2000),
    ]).astype(np.float32)
    rng.shuffle(y_cloud)
    P_cloud = np.stack([y_cloud, y_cloud, y_cloud], -1).reshape(200, 500, 3)

    res_cloud = estimate_params_v3(P_cloud, None, NOREF_V3, span_d_g=1.0)
    assert 0.85 <= res_cloud["stats"]["range_stops"] < 1.20
    assert "ultra_flat_bright" not in res_cloud["diagnostics"]


def test_t4x_overlift_moderate_covers_070_to_085_ev_band():
    """T4x: overlift_mod_check threshold lowered from 0.85 to 0.70 EV (ev_pivot: 0.58, ev_ramp: 0.55).
    Verifies scenes in the [0.70, 0.85) EV band with span_d_g < 1.15, bright_share < 0.20,
    and q25 >= 0.40 trigger moderate overlift damping."""
    from filmcolor.noref import NOREF_V3, estimate_params_v3

    rng = np.random.default_rng(42)
    n = 100_000
    # y centered around 0.095..0.100 produces ev_before in [0.72, 0.84]
    y = rng.uniform(0.098 * 0.7, 0.098 * 1.3, n).astype(np.float32)
    P = np.stack([y, y, y], -1).reshape(200, 500, 3)

    span = 0.82
    res_on = estimate_params_v3(P, None, NOREF_V3, span_d_g=span)
    res_off = estimate_params_v3(P, None, dict(NOREF_V3, overlift_mod_check=False), span_d_g=span)

    assert "overlift_damp" in res_on["diagnostics"]
    od = res_on["diagnostics"]["overlift_damp"]
    assert od["applied"] is True
    assert od.get("mode") == "moderate"
    assert "overlift_damp" not in res_off["diagnostics"]

    ev0 = od["ev_before"]
    q25 = od["q25_before"]
    # Key check: EV falls strictly in the newly covered [0.70, 0.85) EV range
    assert 0.70 <= ev0 < 0.85
    assert q25 >= NOREF_V3["overlift_mod_q25_min"]
    assert od["pull"] > 0.0
    assert od["ev_after"] < ev0
    assert od["blacks_lowered"] > 0.0
    assert res_on["params"]["ev"] < res_off["params"]["ev"]
    assert res_on["params"]["blacks"] < res_off["params"].get("blacks", 0.0)
