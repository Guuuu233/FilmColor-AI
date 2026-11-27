"""Unit and integration tests for FilmColor Lightroom Classic & Photoshop UXP plugin bridge and manifests."""
from __future__ import annotations

import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import numpy as np
from PIL import Image
import pytest

from filmcolor import plugin_bridge
from test_scanner import scan, write_dng

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def bridge_server(tmp_path, monkeypatch):
    monkeypatch.setenv("FILMCOLOR_DATA_ROOT", str(tmp_path / "cache"))
    neg = tmp_path / "neg"
    neg.mkdir()
    rgb, irl = scan()
    f1 = neg / "R01_01.dng"
    f2 = neg / "R01_02.dng"
    write_dng(f1, rgb, irl)
    write_dng(f2, (rgb * np.array([0.85, 0.90, 0.95])).astype(rgb.dtype), irl)

    srv = plugin_bridge.serve_bridge(port=0, idle_timeout=0)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    def call(path: str, body=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(body).encode("utf-8") if body is not None else None,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read()), r.status
        except urllib.error.HTTPError as e:
            return json.loads(e.read()), e.code

    try:
        yield call, [f1, f2], tmp_path
    finally:
        srv.shutdown()
        srv.server_close()


def test_bridge_health_preview_and_views(bridge_server):
    call, files, _ = bridge_server
    h, code = call("/health")
    assert code == 200 and h["ok"] is True and "t24" in h["engines"] and len(h["presets"]) >= 6

    # Preview after (T24 + cinema preset + user delta sliders + base64 for PS UXP)
    p1, code = call("/plugin/preview", dict(
        path=str(files[0]), film="color", engine="t24", preset="cinema",
        view="after", return_base64=True,
        adjustments=dict(ev=0.15, temp=10, shadows=15, skin_protect=30),
    ))
    assert code == 200 and Path(p1["preview_path"]).is_file()
    assert p1["image_data_url"].startswith("data:image/jpeg;base64,")

    # Switching slider produces a NEW preview_path so Lightroom f:picture refreshes reliably
    p2, code = call("/plugin/preview", dict(
        path=str(files[0]), film="color", engine="t24", preset="portrait",
        view="after", adjustments=dict(ev=-0.1),
    ))
    assert code == 200 and p2["preview_path"] != p1["preview_path"] and Path(p2["preview_path"]).is_file()

    # T24 auto baseline view & raw negative view
    pa, code = call("/plugin/preview", dict(path=str(files[0]), engine="t24", view="auto"))
    assert code == 200 and Path(pa["preview_path"]).is_file()
    pn, code = call("/plugin/preview", dict(path=str(files[0]), engine="t24", view="negative"))
    assert code == 200 and Path(pn["preview_path"]).is_file()

    # Engine hot-switch to V3 and B&W
    pv3, code = call("/plugin/preview", dict(path=str(files[0]), film="bw", engine="v3", preset="standard"))
    assert code == 200 and pv3["film"] == "bw" and pv3["engine"] == "v3"


def test_bridge_roll_analyze_and_convert_single_and_batch(bridge_server):
    call, files, tmp = bridge_server
    rinfo, code = call("/plugin/roll_analyze", dict(
        paths=[str(f) for f in files], film="color", engine="t24", roll_pool=True,
    ))
    assert code == 200 and rinfo["roll_id"] and rinfo["n_files"] == 2
    assert rinfo["has_roll_base"] is True and rinfo["n_pooled"] == 2

    # Convert frame 1 -> 16-bit TIFF + sidecar next to source (non-overwriting)
    c1, code = call("/plugin/convert", dict(
        path=str(files[0]), film="color", engine="t24", preset="natural",
        base_mode="roll_nat", roll_id=rinfo["roll_id"], roll_pool=True,
        format="tiff16", encoding="srgb", write_sidecar=True, copy_exif=False,
        adjustments=dict(ev=0.1, highlights=-15, clarity=10),
    ))
    assert code == 200 and c1["output_path"].endswith("_FilmColor.tif")
    assert Path(c1["output_path"]).is_file() and Path(c1["sidecar_path"]).is_file()

    # Second conversion never overwrites first
    c2, code = call("/plugin/convert", dict(
        path=str(files[0]), film="color", engine="t24", preset="standard",
        format="tiff16", write_sidecar=False, copy_exif=False,
    ))
    assert code == 200 and c2["output_path"] != c1["output_path"] and Path(c2["output_path"]).is_file()

    # Subfolder JPEG conversion with auto_crop
    c3, code = call("/plugin/convert", dict(
        path=str(files[1]), film="color", engine="t24", preset="vivid",
        format="jpeg", quality=92, max_edge=240, subfolder="FilmColor_Positive",
        auto_crop=True, write_sidecar=False, copy_exif=False,
    ))
    assert code == 200 and "FilmColor_Positive" in c3["output_path"] and c3["output_path"].endswith(".jpg")
    with Image.open(c3["output_path"]) as im:
        assert max(im.size) == 240


def test_bridge_graceful_shutdown():
    srv = plugin_bridge.serve_bridge(port=0, idle_timeout=0)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/shutdown", data=b"{}", headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        data = json.loads(r.read())
    t.join(timeout=5)
    srv.server_close()
    assert data["ok"] is True and not t.is_alive()


def test_plugin_bundles_structure_and_manifests():
    lr_dir = REPO / "plugins" / "lightroom" / "FilmColor.lrplugin"
    for fname in ("Info.lua", "json.lua", "Config.lua", "Bridge.lua",
                  "ConvertDialog.lua", "QuickConvert.lua", "StopBridge.lua",
                  "Shutdown.lua", "PluginInfoProvider.lua"):
        assert (lr_dir / fname).is_file(), f"missing Lightroom plugin file: {fname}"
    info_txt = (lr_dir / "Info.lua").read_text(encoding="utf-8")
    assert "com.filmcolor.lightroom" in info_txt and "ConvertDialog.lua" in info_txt

    ps_dir = REPO / "plugins" / "photoshop-uxp"
    for fname in ("manifest.json", "index.html", "styles.css", "main.js"):
        assert (ps_dir / fname).is_file(), f"missing Photoshop UXP file: {fname}"
    mf = json.loads((ps_dir / "manifest.json").read_text(encoding="utf-8"))
    assert mf["manifestVersion"] == 5 and mf["host"][0]["app"] == "PS"
    assert "http://127.0.0.1:18765" in mf["requiredPermissions"]["network"]["domains"]


def test_bridge_frontier_noritsu_and_lr_crop_orientation(bridge_server):
    call, files, _ = bridge_server
    # Preview with Frontier preset + Lightroom sensor-space crop & BC (90 CW) orientation
    p_fro, code = call("/plugin/preview", dict(
        path=str(files[0]), film="color", engine="t24", preset="frontier",
        use_lr_crop=True, lr_crop=[0.1, 0.1, 0.9, 0.9], lr_orientation="BC",
        max_width=520, max_height=390,
    ))
    assert code == 200 and p_fro["preset"] == "frontier" and Path(p_fro["preview_path"]).is_file()
    # Original synthetic DNG is 240x360 (landscape); after 0.8 crop (192x288) and 90 CW ("BC"), output is portrait (w < h)
    assert p_fro["width"] < p_fro["height"]

    # Convert with Noritsu preset + Lightroom crop + orientation -> exact portrait cropped TIFF
    c_nor, code = call("/plugin/convert", dict(
        path=str(files[0]), film="color", engine="t24", preset="noritsu",
        use_lr_crop=True, lr_crop=[0.1, 0.1, 0.9, 0.9], lr_orientation="BC",
        format="tiff16", write_sidecar=True, copy_exif=False,
    ))
    assert code == 200 and Path(c_nor["output_path"]).is_file()
    h_out, w_out = c_nor["shape"]
    assert h_out > w_out  # rotated 90 CW to portrait


def test_bridge_input_validation_and_jpeg_encoding_safety(bridge_server):
    call, files, _ = bridge_server
    # Missing file -> HTTP 404
    err, code = call("/plugin/convert", dict(path=str(files[0].with_name("nonexistent_frame.dng")), film="color"))
    assert code == 404 and "FileNotFoundError" in err.get("error", "")

    # Missing path -> HTTP 400 with clean message
    err, code = call("/plugin/preview", dict(film="color"))
    assert code == 400 and "path" in err.get("error", "")

    # Invalid film or format -> HTTP 400 upfront
    err, code = call("/plugin/convert", dict(path=str(files[0]), film="slide"))
    assert code == 400
    err, code = call("/plugin/convert", dict(path=str(files[0]), film="color", format="bmp"))
    assert code == 400

    # JPEG export with encoding="linear" must safely coerce to sRGB encoding (no linear ICC on sRGB JPEG)
    res, code = call("/plugin/convert", dict(
        path=str(files[0]), film="color", engine="t24", preset="frontier",
        format="jpeg", encoding="linear", write_sidecar=True, copy_exif=False,
    ))
    assert code == 200 and Path(res["output_path"]).is_file()
    sc = json.loads(Path(res["sidecar_path"]).read_text(encoding="utf-8"))
    assert sc["output"]["encoding"] == "srgb"


def test_bridge_lr_crop_negative_view_mirror_and_roll_crops(bridge_server):
    call, files, _ = bridge_server
    # 1. Negative view respects lr_crop + mirrored orientation BA
    p_full, c1 = call("/plugin/preview", dict(
        path=str(files[0]), film="color", engine="t24", view="negative",
        use_lr_crop=False, max_width=520, max_height=390,
    ))
    p_crop, c2 = call("/plugin/preview", dict(
        path=str(files[0]), film="color", engine="t24", view="negative",
        use_lr_crop=True, lr_crop=[0.1, 0.25, 0.9, 0.75], lr_orientation="BA",
        max_width=520, max_height=390,
    ))
    assert c1 == 200 and c2 == 200
    from PIL import Image
    im_full = Image.open(p_full["preview_path"])
    im_crop = Image.open(p_crop["preview_path"])
    # Cropped width ratio (~0.5/0.8) must be noticeably narrower aspect than full landscape (3:2)
    assert (im_crop.width / im_crop.height) < (im_full.width / im_full.height) * 0.85

    # 2. Mirrored orientation BA flips pixels horizontally in convert_one
    c_ab, code_ab = call("/plugin/convert", dict(
        path=str(files[0]), film="color", engine="t24", preset="standard",
        use_lr_crop=True, lr_orientation="AB", format="png", copy_exif=False,
    ))
    c_ba, code_ba = call("/plugin/convert", dict(
        path=str(files[0]), film="color", engine="t24", preset="standard",
        use_lr_crop=True, lr_orientation="BA", format="png", copy_exif=False,
    ))
    assert code_ab == 200 and code_ba == 200
    import numpy as np
    arr_ab = np.asarray(Image.open(c_ab["output_path"]))
    arr_ba = np.asarray(Image.open(c_ba["output_path"]))
    assert arr_ab.shape == arr_ba.shape
    assert np.array_equal(arr_ab[:, ::-1], arr_ba)

    # 3. roll_analyze accepts per-frame lr_crops cleanly
    r_res, code_r = call("/plugin/roll_analyze", dict(
        paths=[str(files[0]), str(files[1])],
        lr_crops={str(files[0]): [0.1, 0.1, 0.9, 0.9], str(files[1]): [0.1, 0.1, 0.9, 0.9]},
        film="color", engine="t24", roll_pool=True,
    ))
    assert code_r == 200 and r_res["n_pooled"] == 2


def test_lightroom_lua_syntax_and_no_raw_yielding_pcall() -> None:
    """Guard against unclosed string literals and Lua 5.1 raw pcall() across LR SDK yields."""
    lr_dir = Path(__file__).resolve().parents[1] / "plugins" / "lightroom" / "FilmColor.lrplugin"
    lua_files = sorted(lr_dir.glob("*.lua"))
    assert len(lua_files) >= 8

    for f in lua_files:
        for lineno, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            in_sq = in_dq = esc = False
            i = 0
            while i < len(line):
                ch = line[i]
                if esc:
                    esc = False
                    i += 1
                    continue
                if ch == "\\" and (in_sq or in_dq):
                    esc = True
                    i += 1
                    continue
                if not in_sq and not in_dq and line[i:i + 2] == "--":
                    break
                if ch == '"' and not in_sq:
                    in_dq = not in_dq
                elif ch == "'" and not in_dq:
                    in_sq = not in_sq
                i += 1
            assert not in_sq and not in_dq, f"Unclosed string literal in {f.name}:{lineno}: {line!r}"

    # Bridge.lua must route yielding SDK calls through safeCall (LrTasks.pcall when canYield)
    bridge_lines = (lr_dir / "Bridge.lua").read_text(encoding="utf-8").splitlines()
    assert any("LrTasks.pcall" in l for l in bridge_lines)
    raw_pcall_lines = [
        (i, l) for i, l in enumerate(bridge_lines, 1)
        if "pcall(" in l and "LrTasks.pcall(" not in l and "return pcall(fn" not in l and not l.strip().startswith("--")
    ]
    assert not raw_pcall_lines, f"Raw pcall() in Bridge.lua breaks yielding LR SDK calls: {raw_pcall_lines}"

    # Shutdown.lua calls Bridge.shutdown inside LrTasks.startAsyncTask -> must use LrTasks.pcall
    shut_txt = (lr_dir / "Shutdown.lua").read_text(encoding="utf-8")
    assert "LrTasks.pcall(Bridge.shutdown)" in shut_txt


def test_apply_geometry_fast_preview_handles_angle_with_1ch_mask() -> None:
    """Straighten angle != 0 in fast_preview must support RGB, (H,W,1) mask, and (H,W) mask."""
    from filmcolor.darkroom import Geometry, apply_geometry

    geom = Geometry(rot90=0, angle=1.34, crop=(0.05, 0.08, 0.95, 0.95))
    rgb = np.full((120, 160, 3), 0.5, dtype=np.float32)
    m3 = np.ones((120, 160, 1), dtype=np.float32)
    m2 = np.ones((120, 160), dtype=np.float32)

    out_rgb = apply_geometry(rgb, geom, apply_rot90=False, apply_crop=True, fast_preview=True)
    out_m3 = apply_geometry(m3, geom, apply_rot90=False, apply_crop=True, fast_preview=True)
    out_m2 = apply_geometry(m2, geom, apply_rot90=False, apply_crop=True, fast_preview=True)

    assert out_rgb.shape[:2] == out_m3.shape[:2] == out_m2.shape
    assert out_rgb.shape == (108, 139, 3)
    assert out_m3.shape == (108, 139, 1)
    assert out_m2.shape == (108, 139)


def test_fused_invert_and_render_matches_preview_with_asymmetric_dhi_and_spike_clamp() -> None:
    """Preview (invert_t18_from_positive) and full export (_fused_invert_and_render) must agree
    to <1e-4 even with strongly asymmetric R/G/B density spans and active spike_clamp."""
    from filmcolor.invert import _t18_chunk_from_s, clamp_spikes, invert_t18_from_positive
    from filmcolor.plugin_bridge import SPIKE_CLAMP, _fused_invert_and_render
    from filmcolor.render import RenderParamsT24, render_t24

    rng = np.random.default_rng(2026)
    base_rgb = (0.068, 0.086, 0.038)
    d_lo = (0.0, 0.0, 0.0)
    d_hi = (0.1155, 0.3144, 0.4705)
    gamma = 0.6
    # Include both normal tones and extreme highlight spikes in R/B
    rgb = np.clip(rng.uniform(0.002, 0.09, size=(64, 64, 3)).astype(np.float32), 1e-4, 1.0)
    rgb[:8, :8, 0] = 0.0005  # triggers single-channel R spike clamp
    rp = RenderParamsT24(ev=0.3, temp=0.4, tint=0.1, contrast=1.2, shoulder=0.7, toe=0.3, sat=0.85, floor_restore=0.05)

    # Preview path: green-channel R_leg + legacy spike_clamp + invert_t18_from_positive + render_t24
    D = -np.log10(np.maximum(rgb / np.asarray(base_rgb, np.float32), np.float32(1e-7)))
    s_leg, _ = clamp_spikes(D / np.asarray(d_hi, np.float32), SPIKE_CLAMP, zero_escape=False)
    R_leg = float(d_hi[1] - d_lo[1]) / gamma
    pos_leg = _t18_chunk_from_s(s_leg, R_leg, float(10.0 ** (-R_leg)))
    im_preview = render_t24(invert_t18_from_positive(pos_leg, d_lo, d_hi, gamma, has_cam_matrix=True, version="t23"), rp, None)[0]

    # Full-res export path
    im_export = _fused_invert_and_render(
        rgb, base_rgb=base_rgb, d_lo=d_lo, d_hi=d_hi, gamma=gamma,
        spike_clamp=SPIKE_CLAMP, rp=rp, film="color",
        t18_inv={"version": "t23", "has_cam_matrix": True},
    )
    assert float(np.abs(im_preview - im_export).max()) < 2e-4


def test_preset_matrix_boost_capped_for_steep_camera_matrices() -> None:
    """Steep camera XYZ matrices (e.g. Panasonic/Sony) must not exceed 1.28 diagonal boost under style presets."""
    from filmcolor.plugin_bridge import _apply_t24_preset

    # Steep camera XYZ matrix resembling Panasonic RW2
    steep_raw_m = [
        [0.82, -0.25, -0.08],
        [-0.42, 1.22, 0.22],
        [-0.08, 0.16, 0.65],
    ]
    base_p = dict(ev=0.0, temp=0.0, tint=0.0, contrast=1.0, shoulder=0.5, toe=0.0, sat=0.9)
    p_out, _ = _apply_t24_preset(base_p, "frontier", "color", "t24", raw_matrix=steep_raw_m, scene_metrics=None)
    m = p_out["matrix"]
    assert max(m[0][0], m[1][1], m[2][2]) <= 1.281


def test_convert_dialog_live_roll_preview_wired() -> None:
    cd = (Path(__file__).resolve().parents[1] / "plugins" / "lightroom" / "FilmColor.lrplugin" / "ConvertDialog.lua").read_text(encoding="utf-8")
    assert "ensureRollAnalyzed" in cd
    assert '"rollSync"' in cd
    assert "roll_id = activeRollId" in cd


def test_t25_api_and_coerce_render_params() -> None:
    from filmcolor.noref import NOREF_T25, _load_t25_weights, estimate_params_t25, pool_roll_params_t25
    from filmcolor.pipeline import coerce_render_params
    from filmcolor.render import RenderParamsT25, T25_BOUNDS, render_t25
    from filmcolor.darkroom import T25_STYLE_PRESETS

    pkg = _load_t25_weights("t25_perceptual")
    assert pkg is not None and pkg.get("version") == "T25"
    assert NOREF_T25["version"] == "T25" and NOREF_T25["id"] == "noref_t25"
    assert callable(estimate_params_t25) and callable(pool_roll_params_t25) and callable(render_t25)
    assert "frontier" in T25_STYLE_PRESETS and "slope_r" in T25_BOUNDS and "toe" in T25_BOUNDS
    # Partial T25 dict without eps_rgb/residual_matrix must still coerce to RenderParamsT25
    rp = coerce_render_params({"ev": 0.2, "floor_restore": 0.08, "slope_r": 0.01, "slope_b": -0.01})
    assert isinstance(rp, RenderParamsT25)
    assert abs(rp.floor_restore - 0.08) < 1e-6


def test_clamp_spikes_3ch_fast_matches_sort_bit_exact() -> None:
    from filmcolor.invert import clamp_spikes

    rng = np.random.default_rng(7)
    for case in [
        rng.uniform(0.0, 1.05, size=(64, 64, 3)).astype(np.float32),      # zero spikes fast path
        rng.uniform(0.0, 2.2, size=(64, 64, 3)).astype(np.float32),       # mixed spikes
        np.concatenate([rng.uniform(0.0, 1.5, (64, 64, 2)), np.full((64, 64, 1), 8.0)], axis=-1).astype(np.float32),  # zero_escape
    ]:
        out, n = clamp_spikes(case, (1.15, 0.12), zero_escape=False)
        # Reference np.sort implementation
        mid_max, margin = np.float32(1.15), np.float32(0.12)
        ss = np.sort(case, axis=-1)
        smin, smid, smax = ss[..., 0:1], ss[..., 1:2], ss[..., 2:3]
        bound = np.where(smid > mid_max, np.float32(np.inf), np.maximum(smid, np.float32(1.0)) + margin)
        if float(np.mean((smid > mid_max) & (smin <= np.float32(1.6)) & (smax > np.float32(5.0)))) > 0.01:
            bz = np.maximum(smin, np.float32(1.0)) + margin
            bound = np.where(smin <= np.float32(1.6), np.minimum(bz, bound), bound)
        over = case > bound
        ref = np.minimum(case, bound) if over.any() else case
        ref_n = int(over.any(axis=-1).sum())
        assert n == ref_n
        assert np.array_equal(out, ref)


def test_bridge_security_cors_and_path_hardening(tmp_path: Path) -> None:
    import urllib.error, urllib.request, json, threading
    from filmcolor.plugin_bridge import serve_bridge

    srv = serve_bridge(0, idle_timeout=0)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        base = f"http://127.0.0.1:{port}"
        # 1. Health advertises T25 as default
        with urllib.request.urlopen(f"{base}/health", timeout=3) as resp:
            h = json.loads(resp.read().decode())
            assert h["ok"] is True and h["algorithm"] == "T25" and h["default_engine"] == "t25"
            assert "t25" in h["engines"] and "t24" in h["engines"]

        # 2. Third-party web origin blocked with 403
        req_bad = urllib.request.Request(f"{base}/health", headers={"Origin": "https://evil.example"})
        try:
            urllib.request.urlopen(req_bad, timeout=3)
            raise AssertionError("expected 403 for third-party origin")
        except urllib.error.HTTPError as exc:
            assert exc.code == 403

        # 3. Loopback & UXP null origins allowed
        for orig in ("http://127.0.0.1:8000", "http://localhost:3000", "null"):
            req_ok = urllib.request.Request(f"{base}/health", headers={"Origin": orig})
            with urllib.request.urlopen(req_ok, timeout=3) as resp:
                assert resp.status == 200
                assert resp.headers.get("Access-Control-Allow-Origin") == orig

        # 4. Path traversal in subfolder / suffix / bad encoding rejected with 400
        dummy = tmp_path / "dummy.tif"
        dummy.write_bytes(b"not-a-real-tiff")
        for bad_payload in [
            {"path": str(dummy), "subfolder": "../escape"},
            {"path": str(dummy), "suffix": "../evil"},
            {"path": str(dummy), "encoding": "bogus"},
            {"path": str(dummy), "max_edge": 999999},
        ]:
            req_c = urllib.request.Request(
                f"{base}/plugin/convert",
                data=json.dumps(bad_payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                urllib.request.urlopen(req_c, timeout=3)
                raise AssertionError(f"expected 400 for {bad_payload}")
            except urllib.error.HTTPError as exc:
                assert exc.code == 400
    finally:
        srv.shutdown()
        srv.server_close()


def test_apply_lr_crop_angle_exact_adobe_geometry() -> None:
    """Verify Lightroom rotated-crop dimensions, sign, and PIL/SciPy parity."""
    import math
    from filmcolor.darkroom import apply_lr_crop_angle

    # 1. Exact dimension match against real Lightroom catalog entry (_DSF8434.RAF 副本_1):
    #    raw=6000x4000, CropAngle=-0.95, (T, L, B, R)=(0.087519, 0.172074, 0.954972, 0.77472) -> 3558x3529
    #    Scaled 1/10 -> 600x400 -> 356x353
    img = np.zeros((400, 600, 3), dtype=np.float32)
    crop = (0.087519, 0.172074, 0.954972, 0.77472)
    out_fast = apply_lr_crop_angle(img, crop, -0.95, fast_preview=True)
    out_full = apply_lr_crop_angle(img, crop, -0.95, fast_preview=False)
    assert out_fast.shape == (353, 356, 3)
    assert out_full.shape == (353, 356, 3)

    # 2. Construct a synthetic white rectangle rotated by angle = -5.0 deg inside a black frame.
    #    LR crop box matching that exact tilted white rectangle must extract a pure-white image
    #    with zero black corners (proving both center and rotation sign).
    H, W = 400, 600
    ang = -5.0
    th = math.radians(ang)
    ct, st = math.cos(th), math.sin(th)
    cx, cy = 310.0, 205.0
    cw, ch = 220.0, 180.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    # Local coords inside the rotated rectangle:
    u_loc = (xx - cx) * ct + (yy - cy) * st
    v_loc = -(xx - cx) * st + (yy - cy) * ct
    inside = (np.abs(u_loc) <= cw * 0.5 + 1.5) & (np.abs(v_loc) <= ch * 0.5 + 1.5)
    synth = np.zeros((H, W, 3), dtype=np.float32)
    synth[inside] = 1.0

    # Top-left and bottom-right corners of that rotated rectangle in source coords:
    dx = cw * ct - ch * st
    dy = cw * st + ch * ct
    L = (cx + 0.5 - 0.5 * dx) / W
    R = (cx + 0.5 + 0.5 * dx) / W
    T = (cy + 0.5 - 0.5 * dy) / H
    B = (cy + 0.5 + 0.5 * dy) / H

    for fast in (True, False):
        res = apply_lr_crop_angle(synth, (T, L, B, R), ang, fast_preview=fast)
        assert res.shape == (int(round(ch)), int(round(cw)), 3)
        # Interior (margin 2px from bilinear edge) must be pure white (no black wedge!)
        assert float(res[2:-2, 2:-2].min()) > 0.99, f"fast={fast} min={res[2:-2, 2:-2].min()}"

    # 3. 1-ch and 2D masks work identically in both fast_preview and full precision
    m3 = synth[..., :1]
    m2 = synth[..., 0]
    for fast in (True, False):
        r3 = apply_lr_crop_angle(m3, (T, L, B, R), ang, fast_preview=fast)
        r2 = apply_lr_crop_angle(m2, (T, L, B, R), ang, fast_preview=fast)
        assert r3.shape == (int(round(ch)), int(round(cw)), 1)
        assert r2.shape == (int(round(ch)), int(round(cw)))
        assert float(r3[2:-2, 2:-2].min()) > 0.99
        assert float(r2[2:-2, 2:-2].min()) > 0.99


def test_json_nan_sanitization_and_strict_serialization() -> None:
    """B03: Non-finite floats must be sanitized before strict allow_nan=False JSON encoding."""
    import json
    from filmcolor.plugin_bridge import _sanitize_json_obj, _parse_max_dim

    raw = {
        "below_black": float("nan"),
        "above_white": float("inf"),
        "neg_inf": float("-inf"),
        "nested": [1.5, float("nan"), {"ok": True, "x": np.float32("nan")}],
    }
    clean = _sanitize_json_obj(raw)
    dumped = json.dumps(clean, ensure_ascii=False, allow_nan=False)
    parsed = json.loads(dumped)
    assert parsed["below_black"] is None
    assert parsed["above_white"] is None
    assert parsed["neg_inf"] is None
    assert parsed["nested"][0] == 1.5 and parsed["nested"][1] is None and parsed["nested"][2]["x"] is None

    # B10: _parse_max_dim clamps positive dims and rejects <=0/NaN
    assert _parse_max_dim(0) is None
    assert _parse_max_dim(-100) is None
    assert _parse_max_dim(float("nan")) is None
    assert _parse_max_dim(4) == 16
    assert _parse_max_dim(99999) == 4096
    assert _parse_max_dim(640) == 640


def test_exiftool_daemon_path_sanitization_and_timeout(tmp_path: Path) -> None:
    """B02: Reject newline/control paths and recover from hung exiftool stdout without deadlock."""
    import sys, time
    from filmcolor.plugin_bridge import _ExifToolDaemon, _safe_exif_path

    assert _safe_exif_path(Path("/tmp/bad\npath.tif")) is None
    assert _safe_exif_path(Path("/tmp/bad\rpath.tif")) is None

    # Stub script that mimics a hung exiftool (never prints {ready1})
    hang_script = tmp_path / "fake_exiftool.py"
    hang_script.write_text("import sys, time\ntime.sleep(30)\n", encoding="utf-8")
    src_f = tmp_path / "src.tif"
    dst_f = tmp_path / "dst.tif"
    src_f.write_bytes(b"a")
    dst_f.write_bytes(b"b")

    daemon = _ExifToolDaemon()
    try:
        # Use a tiny wrapper executable on POSIX
        if sys.platform != "win32":
            sh_stub = tmp_path / "hang_exiftool.sh"
            sh_stub.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
            sh_stub.chmod(0o755)
            t0 = time.monotonic()
            ok = daemon.copy_tags(str(sh_stub), src_f, dst_f, timeout_s=0.35)
            elapsed = time.monotonic() - t0
            assert ok is False
            assert elapsed < 3.0
    finally:
        daemon.close()


def test_bridge_concurrency_base64_skip_disk_and_active_tracking(bridge_server, tmp_path: Path) -> None:
    """P01/P03/B09/B11/B12/B13/B21/P05: Fine-grained concurrency, skip disk PNG, negative JPEG b64, active tracking."""
    import concurrent.futures, os, time
    from filmcolor.plugin_bridge import BridgeHandler, PluginBridgeState

    assert BridgeHandler.protocol_version == "HTTP/1.1"
    assert BridgeHandler.timeout is not None and BridgeHandler.timeout > 0

    call, files, _ = bridge_server

    # P03: return_base64=True with write_preview_file=False skips disk PNG write
    p_mem, code = call("/plugin/preview", dict(
        path=str(files[0]), film="color", engine="t25", preset="frontier",
        view="after", max_width=320, max_height=240,
        return_base64=True, write_preview_file=False,
    ))
    assert code == 200
    assert p_mem["preview_path"] is None
    assert p_mem["image_data_url"].startswith("data:image/jpeg;base64,")

    # P03: negative view with return_base64=True returns JPEG data URL
    p_neg, code_n = call("/plugin/preview", dict(
        path=str(files[0]), film="color", engine="t25", view="negative",
        max_width=320, max_height=240, return_base64=True, write_preview_file=False,
    ))
    assert code_n == 200
    assert p_neg["preview_path"] is None
    assert p_neg["image_data_url"].startswith("data:image/jpeg;base64,")

    # P01: Concurrent preview requests succeed cleanly without lock contention errors
    def _req(ev_val: float):
        return call("/plugin/preview", dict(
            path=str(files[0]), film="color", engine="t25", preset="frontier",
            view="after", max_width=320, max_height=240,
            return_base64=True, write_preview_file=False,
            adjustments=dict(ev=ev_val),
        ))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(_req, [-0.2, 0.0, 0.1, 0.25]))
    assert all(c == 200 and r["image_data_url"].startswith("data:image/jpeg;base64,") for r, c in results)

    # B21: convert_one accepts quality=None and softness=NaN safely
    c_res, c_code = call("/plugin/convert", dict(
        path=str(files[0]), film="color", engine="t25", preset="frontier",
        format="jpeg", quality=None, write_sidecar=False, copy_exif=False,
    ))
    assert c_code == 200 and Path(c_res["output_path"]).is_file()

    # B11 & B12: stale .tmp purge and track_request/wait_idle
    st = PluginBridgeState()
    stale_tmp = st.preview_dir / ".prev_test_stale.png.tmp"
    stale_tmp.write_bytes(b"stale")
    old_ts = time.time() - 7200
    os.utime(stale_tmp, (old_ts, old_ts))
    st._purge_stale_previews(max_age_s=3600.0)
    assert not stale_tmp.exists()

    assert st.active_requests == 0
    with st.track_request():
        assert st.active_requests == 1
        assert st.wait_idle(timeout=0.05) is False
    assert st.active_requests == 0
    assert st.wait_idle(timeout=0.5) is True


def test_plugin_hardening_lua_and_uxp_contracts() -> None:
    """Verify B04/B05/B06/B07/B14/B15/B17/B20/P02/P06/P07/P08 contracts in plugin sources."""
    root = Path(__file__).resolve().parents[1]
    lr_dir = root / "plugins" / "lightroom" / "FilmColor.lrplugin"
    uxp_dir = root / "plugins" / "photoshop-uxp"

    # B05: Info.lua registers both LrShutdownPlugin and LrShutdownApp
    info_txt = (lr_dir / "Info.lua").read_text(encoding="utf-8")
    assert 'LrShutdownPlugin = "Shutdown.lua"' in info_txt
    assert 'LrShutdownApp = "Shutdown.lua"' in info_txt

    # B15: Config.lua preserves valid custom Python paths without requiring '.venv'
    cfg_txt = (lr_dir / "Config.lua").read_text(encoding="utf-8")
    assert 'string.find(p.pythonPath, "%.venv")' not in cfg_txt

    # P07: I18n.lua reads LrPrefs directly and caches host locale
    i18n_lua = (lr_dir / "I18n.lua").read_text(encoding="utf-8")
    assert "Config.prefs()" not in i18n_lua
    assert "cachedHostLang" in i18n_lua

    # B04/B14/P08: Bridge.lua uses PowerShell quoting on Windows, table checks, and batch catalog import
    bridge_lua = (lr_dir / "Bridge.lua").read_text(encoding="utf-8")
    assert "Start-Process" in bridge_lua and "-RedirectStandardOutput" in bridge_lua
    assert 'type(data) == "table"' in bridge_lua and 'type(data) ~= "table"' in bridge_lua
    assert "function Bridge.importAndStackBatch" in bridge_lua

    # B06/B23: ConvertDialog.lua guards ensureRollAnalyzed with LrTasks.pcall and resets rotDelta
    cd_lua = (lr_dir / "ConvertDialog.lua").read_text(encoding="utf-8")
    assert "LrTasks.pcall" in cd_lua
    assert "props.rotDelta = 0" in cd_lua

    # B17: json.lua normalizes decimal comma and validates \uXXXX length
    json_lua = (lr_dir / "json.lua").read_text(encoding="utf-8")
    assert 'gsub(",", ".")' in json_lua
    assert "#hex == 4" in json_lua

    # B07/B08/B18/B20/P02/P06: UXP main.js & i18n.js checks
    main_js = (uxp_dir / "main.js").read_text(encoding="utf-8")
    assert "max_width: 640" in main_js and "max_height: 400" in main_js
    assert "write_preview_file: false" in main_js
    assert "lastPreviewPayload" in main_js
    assert 'item._obj === "error"' in main_js
    assert "synchronousExecution" not in main_js
    assert "safeReadJson" in main_js
    assert "cleanupTempEntry" in main_js
