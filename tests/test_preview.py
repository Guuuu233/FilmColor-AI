"""Preview layer (filmcolor.preview): the export equals the conversion it previews, byte for byte, and the preview
is close to the downscaled full-resolution render."""
import numpy as np
import pytest

from filmcolor.compare import block_mean
from filmcolor.noref import NOREF_V2
from filmcolor.pipeline import convert
from filmcolor.preview import HOLDER_GREY, load_preview

from test_bw import bw_scan, write_dng as write_bw_dng
from test_pins_and_cli import PARAMS, RAF, SCAN_REGION, needs_raf
from test_scanner import scan, write_dng

PIN_SCANNER = "9148a756cf980a476a8c2294608e4b3886cdb9ab1df38665d41d2ff86966c83f"
PIN_SCANNER_BW = "1e219285d1824c07ca34be08049b854db71f0d82a7429a836a570d6b007e00d0"
PIN_RAF = "ed6516673a19abe372d2c53f2488527699226c3e49a451cbe6c0d0d92b30abae"
PIN_RAF_AUTO = "f9b26d3c02b08c33d031d96944247ca83f131cafd986d432bf2918433966a60d"
PIN_RAF_PARAMS = "972ce161c3acabc232b3a9a1182804d86d3fbd750b8f48844cec690d3d470958"


def test_scanner_preview_exports_the_previewed_conversion(tmp_path):
    rgb, irl = scan()
    p = tmp_path / "s.dng"
    write_dng(p, rgb, irl)
    pv = load_preview(p, max_side=100, image_region=SCAN_REGION)
    assert pv.factor == 4 and pv.positive.shape[:2] == (60, 90)
    assert pv.export(tmp_path / "o")["output"]["sha256"] == PIN_SCANNER
    full, _ = convert(p, image_region=SCAN_REGION)
    d = np.abs(pv.render() - block_mean(full, pv.factor))
    assert d.mean() < 0.01 and np.percentile(d, 99) < 0.05


def test_scanner_bw_preview_exports_the_previewed_conversion(tmp_path):
    p = tmp_path / "bw.dng"
    write_bw_dng(p, bw_scan())
    pv = load_preview(p, max_side=100, image_region=SCAN_REGION, film="bw")
    assert pv.export(tmp_path / "o")["output"]["sha256"] == PIN_SCANNER_BW


def test_region_coordinates(tmp_path):
    rgb, irl = scan()
    p = tmp_path / "s.dng"
    write_dng(p, rgb, irl)
    pv = load_preview(p, max_side=100)
    assert pv.to_sensor((5, 10, 15, 20)) == (20, 40, 60, 80)
    assert pv.to_preview(pv.to_sensor((5, 10, 15, 20))) == (5, 10, 15, 20)


def test_reload_redoes_base_and_inversion(tmp_path):
    rgb, irl = scan()
    p = tmp_path / "s.dng"
    write_dng(p, rgb, irl)
    pv = load_preview(p, max_side=100, image_region=SCAN_REGION)
    assert pv.base_info["method"] == "auto"
    manual = [pv.to_sensor((0, 0, 60, 6))]
    pv2 = pv.reload(base_regions=manual)
    direct = load_preview(p, max_side=100, image_region=SCAN_REGION, base_regions=manual)
    assert pv2.base_info["method"] == "manual" and pv2.load_kwargs["base_regions"] == manual
    assert np.array_equal(pv2.positive, direct.positive) and pv2.inversion == direct.inversion
    out = pv2.export(tmp_path / "o")
    full, sc = convert(p, image_region=SCAN_REGION, base_regions=manual)
    assert out["inversion"]["params"] == sc["inversion"]["params"]


@needs_raf
def test_camera_preview_default_auto_and_params(tmp_path):
    pv = load_preview(RAF, auto=NOREF_V2)
    assert max(pv.positive.shape[:2]) <= 1600
    assert pv.export(tmp_path / "d")["output"]["sha256"] == PIN_RAF
    assert pv.export(tmp_path / "p", PARAMS)["output"]["sha256"] == PIN_RAF_PARAMS
    _, sc = convert(RAF, auto=NOREF_V2)
    assert pv.auto["params"] == sc["auto"]["params"]
    assert pv.export(tmp_path / "a", pv.auto["params"])["output"]["sha256"] == PIN_RAF_AUTO
    # T4i: the film holder around this 6x6 frame is shown as HOLDER_GREY in the preview only; elsewhere the preview
    # is the downscaled export
    assert 0.2 < pv.holder.mean() < 0.5
    for params in (None, PARAMS):
        full, _ = convert(RAF, render_params=params)
        ref = block_mean(full, pv.factor)[:pv.positive.shape[0], :pv.positive.shape[1]]
        disp = pv.render(params)
        assert np.all(disp[pv.holder] == np.float32(HOLDER_GREY))
        d = np.abs(disp - ref)[~pv.holder]
        assert d.mean() < 0.01 and np.percentile(d, 99) < 0.05, (params, d.mean(), np.percentile(d, 99))


def test_t4e_with_auto_reuses_the_kept_estimator_input(tmp_path):
    """T4e: a preview loaded with an auto estimate keeps its estimator input (noref.estimation_input of the
    full-resolution positive), so with_auto re-estimates another style preset without the full-resolution positive,
    with exactly the parameters a fresh load with that preset gives."""
    from filmcolor.noref import estimation_input, preset_config

    rgb, irl = scan()
    p = tmp_path / "s.dng"
    write_dng(p, rgb, irl)
    pv = load_preview(p, max_side=100, image_region=SCAN_REGION, auto=preset_config("standard"))
    assert pv.full_positive is None and pv.est_input is not None
    full = load_preview(p, max_side=100, image_region=SCAN_REGION, keep_full=True)
    ref_input = estimation_input(full.full_positive, full.image_region)
    assert all(np.array_equal(a, b) for a, b in zip(pv.est_input, ref_input))
    vivid = pv.with_auto(preset_config("vivid"))
    fresh = load_preview(p, max_side=100, image_region=SCAN_REGION, auto=preset_config("vivid"))
    assert vivid.auto["params"] == fresh.auto["params"] and vivid.auto["preset"] == "vivid"
    assert vivid.est_input is pv.est_input and vivid.load_kwargs["auto"]["preset"] == "vivid"
    assert full.with_auto(preset_config("vivid")).auto["params"] == fresh.auto["params"]  # computed when not kept


def test_t4i_holder_mask_is_the_edge_connected_far_past_white_region():
    """T4i Bug 3: an opaque holder at the frame edge (positive hundreds of times white in one channel, past 2 in two;
    _DSF8948's is R ~300, G ~36, B ~3.6) is the holder; a highlight of the same values inside the frame, the dark
    rebate and a frame without a holder are not. film_box is the rest of the frame."""
    from filmcolor.preview import film_box, holder_mask

    P = np.full((60, 90, 3), 0.3, np.float32)
    P[:, :15] = (300.0, 36.0, 3.6)       # holder on the left
    P[:, 80:] = (40.0, 2.5, 30.0)        # magenta holder on the right
    P[:, 15:18] = 0.0                    # rebate between holder and image
    P[20:25, 40:45] = (300.0, 36.0, 3.6) # a highlight inside the frame
    P[:, 10:12, 2] = 0.9                 # holder pixels with one channel below white stay holder
    m = holder_mask(P)
    assert m[:, :15].all() and m[:, 80:].all()
    assert not m[:, 15:80].any()
    assert film_box(m) == [0, 15, 60, 65]
    assert not holder_mask(np.full((20, 20, 3), 0.5, np.float32)).any() and film_box(np.zeros((4, 4), bool)) is None


def test_t4m_holder_dual_threshold_explicit_threshold_and_film_box():
    """T4m: default dual threshold ((s1>2)&(s2>16))|((s1>1.25)&(s2>8)), explicit non-default thresholds,
    film_box row/col ratio selection when holder >= 15%, fallback on full holder / empty rows, and
    preview _paint_holder paints outer frame."""
    from filmcolor.holder import HOLDER_GREY, film_box, holder_mask

    # 1. Dual threshold default: second channel > 1.25 and peak > 8 is recognized if edge-connected
    P = np.full((100, 100, 3), 0.3, np.float32)
    P[:, :20] = (10.0, 1.5, 0.5)  # sorted: s1=1.5 > 1.25, s2=10.0 > 8.0, edge connected
    m_default = holder_mask(P)
    assert m_default[:, :20].all()

    # With explicit non-default threshold (peak=17.0), this region (s1=1.5, s2=10.0) is NOT detected
    m_explicit = holder_mask(P, peak=17.0)
    assert not m_explicit.any()

    # 2. Internal highlight not mistaken even if matching dual threshold or peak threshold
    P_high = np.full((100, 100, 3), 0.3, np.float32)
    P_high[40:60, 40:60] = (50.0, 20.0, 5.0)  # inside highlight, not edge-connected
    assert not holder_mask(P_high).any()

    # 3. Sprocket holes / holder frame and film_box selection with >60% keep ratio
    # Simulate a 100x100 frame with holder on left/right (columns 0..15 and 85..100, image keep ratio 70% > 60%)
    # plus isolated sprocket hole intrusions into holder (columns 13..14 have keep in only 20 rows -> keep ratio 20% <= 60%)
    holder = np.zeros((100, 100), dtype=bool)
    holder[:, :15] = True
    holder[:, 85:] = True
    holder[:, 13:15] = True
    holder[40:60, 13:15] = False  # 20% keep in cols 13..14
    assert holder.mean() >= 0.15
    fb = film_box(holder)
    assert fb == [0, 15, 100, 70]  # cols 13..14 excluded because keep col ratio = 0.20 <= 0.60

    # 4. Fallback boundaries: all holder -> None
    all_holder = np.ones((50, 50), dtype=bool)
    assert film_box(all_holder) is None

    # Fallback when keep exists but no row or col exceeds 0.60 (e.g. sparse keep)
    sparse = np.ones((100, 100), dtype=bool)
    sparse[::2, ::2] = False  # keep is 25% across all rows/cols, mean in each row/col is 0.5 <= 0.60
    fb_sparse = film_box(sparse)
    assert fb_sparse is not None  # falls back without IndexError
    assert fb_sparse == [0, 0, 99, 99]

    # 5. Preview._paint_holder: outer frame painted with HOLDER_GREY
    from pathlib import Path
    from filmcolor.preview import Preview
    dummy_pos = np.full((100, 100, 3), 0.5, np.float32)
    dummy_mask = np.ones((100, 100), dtype=bool)
    pv = Preview(path=Path("dummy"), sha256="abc", factor=1, positive=dummy_pos, mask=dummy_mask,
                 image_region=(0, 0, 100, 100), region_source="test", film="color", base_info={},
                 inversion={}, white_pct=99.9, report={}, base_inside_image=None, full_shape=(100, 100),
                 holder=holder)
    e = np.full((100, 100, 3), 0.5, np.float32)
    painted = pv._paint_holder(e)
    # Both holder and outer film_box (e.g. cols 13..14) should be HOLDER_GREY
    assert np.all(painted[:, :15] == np.float32(HOLDER_GREY))
    assert np.all(painted[:, 85:] == np.float32(HOLDER_GREY))
    # Inside film box stays 0.5
    assert np.all(painted[:, 15:85] == 0.5)


def _t5a_frame_holder():
    """Holder (bool, 1000 x 400) of a film box [300, 0, 300, 200]: dense mask above, below and right of the frame."""
    holder = np.zeros((1000, 400), bool)
    holder[:300, 200:] = True
    holder[600:, 200:] = True
    return holder


def _t5a_preview(holder):
    from pathlib import Path
    from filmcolor.preview import Preview
    h, w = holder.shape
    return Preview(path=Path("dummy"), sha256="abc", factor=1, positive=np.full((h, w, 3), 0.5, np.float32),
                   mask=np.ones((h, w), bool), image_region=(0, 0, h, w), region_source="test", film="color",
                   base_info={}, inversion={}, white_pct=99.9, report={}, base_inside_image=None, full_shape=(h, w),
                   holder=holder)


def test_t5a_holder_paint_keeps_deep_short_contact_component():
    """T5a (B096): a holder component inside the film box that reaches 0.30 of its short side into it and covers 0.30
    of the edge it touches is a clipped scene highlight (an over-exposed window), not a holder band: it is not painted
    HOLDER_GREY. The holder mask itself (white point, metering) is unchanged."""
    from filmcolor.holder import film_box, scene_highlight_components
    holder = _t5a_frame_holder()
    holder[300:360, 70:130] = True               # window: touches the box top over 60 / 200, 60 px deep
    assert holder.mean() >= 0.15 and film_box(holder) == [300, 0, 300, 200]
    scene = scene_highlight_components(holder)
    assert scene[300:360, 70:130].all() and scene.sum() == 3600
    pv = _t5a_preview(holder)
    painted = pv._paint_holder(np.full((1000, 400, 3), 0.5, np.float32))
    assert np.all(painted[300:360, 70:130] == 0.5)                   # the window keeps its render
    assert np.all(painted[:300] == np.float32(HOLDER_GREY)) and np.all(painted[600:] == np.float32(HOLDER_GREY))
    assert pv.holder[300:360, 70:130].all()                          # holder mask not changed


def test_t5a_holder_paint_keeps_full_edge_band():
    """T5a (B029's left holder strip): a band 0.125 of the short side deep covering the whole edge it touches is a
    holder and stays painted; a depth-only rule would have released it."""
    from filmcolor.holder import scene_highlight_components
    holder = _t5a_frame_holder()
    holder[300:600, 0:25] = True                 # 25 / 200 = 0.125 deep, along the full 300 px left edge
    assert not scene_highlight_components(holder).any()
    assert scene_highlight_components(holder, max_edge_span=1.01)[300:600, 0:25].all()
    painted = _t5a_preview(holder)._paint_holder(np.full((1000, 400, 3), 0.5, np.float32))
    assert np.all(painted[holder] == np.float32(HOLDER_GREY))
    assert np.all(painted[300:600, 25:200] == 0.5)


def test_t5a_holder_paint_small_share_unchanged():
    """T5a: with holder_share < 0.15 (no film-box band mode) nothing is released; the paint equals T4z's
    ``e[holder] = HOLDER_GREY`` exactly, also for a deep short-contact component."""
    from filmcolor.holder import scene_highlight_components
    holder = np.zeros((300, 300), bool)
    holder[0:60, 100:160] = True                 # deep, short contact; share 0.04
    holder[:, :8] = True
    assert holder.mean() < 0.15 and not scene_highlight_components(holder).any()
    e = np.random.default_rng(1).random((300, 300, 3)).astype(np.float32)
    t4z = e.copy()
    t4z[holder] = np.float32(HOLDER_GREY)
    assert np.array_equal(_t5a_preview(holder)._paint_holder(e.copy()), t4z)


def test_t5a_input_density_warning_params_unchanged(tmp_path, monkeypatch):
    """T5a: a white point d_hi_G = 2.5 D (> INPUT_DHI_G_MAX 2.40) on a colour negative adds the
    ``input_density_implausible`` hint to report["input_warnings"]; status, parameters and inversion are exactly those
    of the same load without the hint."""
    import filmcolor.preview as prv
    from filmcolor.noref import NOREF_V3
    rgb, irl = scan()
    p = tmp_path / "s.dng"
    write_dng(p, rgb, irl)
    kw = dict(max_side=100, image_region=SCAN_REGION, d_hi=(2.5, 2.5, 2.5), auto=NOREF_V3)
    pv = load_preview(p, **kw)
    codes = [w["code"] for w in pv.report["input_warnings"]]
    assert codes == ["input_density_implausible"] and pv.report["input_warnings"][0]["value"] == 2.5
    monkeypatch.setattr(prv, "INPUT_DHI_G_MAX", 10.0)
    quiet = load_preview(p, **kw)
    assert quiet.report["input_warnings"] == []
    assert pv.auto["status"] == quiet.auto["status"] and pv.auto["params"] == quiet.auto["params"]
    assert pv.inversion == quiet.inversion and np.array_equal(pv.positive, quiet.positive)
    assert {k: v for k, v in pv.report.items() if k != "input_warnings"} == \
           {k: v for k, v in quiet.report.items() if k != "input_warnings"}
    assert load_preview(p, max_side=100, image_region=SCAN_REGION).report["input_warnings"] == []


def test_t5a_input_empty_channel_warning(tmp_path):
    """T5a (A229: centre p99 [0.831, 0.738, 0.002]): a channel whose centre linear input p99 is below 0.02 while
    another reaches 0.30 gives ``input_channel_empty``. INPUT_EMPTY_CHANNEL = (0.02, 0.30): p99 0.03 (above the
    threshold) or a dim frame (no channel >= 0.30) gives none."""
    from filmcolor.preview import input_warnings
    hit = input_warnings([0.8, 1.0, 0.9], [0.83, 1.0, 0.015])
    assert [(w["code"], w["channel"]) for w in hit] == [("input_channel_empty", "B")]
    assert input_warnings([0.8, 1.0, 0.9], [0.83, 1.0, 0.03]) == []
    assert input_warnings([0.8, 1.0, 0.9], [0.25, 0.28, 0.01]) == []
    rgb, irl = scan()
    rgb = rgb.astype(np.float64)
    rgb[..., :2] *= 2.0                          # base R 0.90, G 0.40 of white
    rgb[..., 2] = np.minimum(rgb[..., 2], 650)   # blue ~ 0.01 of white everywhere
    rgb = np.clip(np.round(rgb), 0, 65535).astype(np.uint16)
    p = tmp_path / "empty_b.dng"
    write_dng(p, rgb, irl)
    pv = load_preview(p, max_side=100, image_region=SCAN_REGION)
    c = pv.report["input_center_p99"]
    assert c[2] < 0.02 and c[1] >= 0.30
    assert [w["channel"] for w in pv.report["input_warnings"] if w["code"] == "input_channel_empty"] == ["B"]


def test_t5c_rebate_strip_components_and_paint():
    """T5c (A1298): a hot component inside film_box that starts within 0.01 * short of a box edge, stays within
    0.08 * short of that edge, and spans >= 0.35 of that edge is painted HOLDER_GREY in Preview._paint_holder without
    changing pv.holder, whereas components reaching > 0.08 * short deep or spanning < 0.35 of the edge are untouched."""
    from filmcolor.holder import film_box, rebate_strip_components

    holder = _t5a_frame_holder()  # film_box is [300, 0, 300, 200], short = 200, gap = 2.0, dep = 16.0
    assert holder.mean() >= 0.15 and film_box(holder) == [300, 0, 300, 200]
    pv = _t5a_preview(holder)
    # 1. Valid detached rebate strip along left edge of film_box: c in [1..12] (c0=1 <= 2, c1+1=13 <= 16), r in [350..470] (height 120 >= 0.35*300=105)
    pv.positive[350:470, 1:13] = [18.0, 24.0, 50000.0]
    # 2. Deep highlight component reaching c1+1 = 30 > 16 (0.08 * 200): must NOT be selected
    pv.positive[480:590, 1:30] = [18.0, 24.0, 50000.0]
    # 3. Short edge component with height 60 < 105 (0.35 * 300): must NOT be selected
    pv.positive[305:365, 188:199] = [18.0, 24.0, 50000.0]

    rebate = rebate_strip_components(pv.positive, holder)
    assert rebate[350:470, 1:13].all()
    assert not rebate[480:590, 1:30].any()
    assert not rebate[305:365, 188:199].any()
    assert int(rebate.sum()) == 120 * 12

    e = np.full(pv.positive.shape, 0.5, np.float32)
    painted = pv._paint_holder(e)
    assert np.all(painted[350:470, 1:13] == np.float32(HOLDER_GREY))
    assert np.all(painted[480:590, 1:30] == 0.5)
    assert np.all(painted[305:365, 188:199] == 0.5)
    assert not pv.holder[350:470, 1:13].any()

