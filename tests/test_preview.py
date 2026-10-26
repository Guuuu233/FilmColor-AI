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
