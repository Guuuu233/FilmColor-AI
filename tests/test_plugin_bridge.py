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
