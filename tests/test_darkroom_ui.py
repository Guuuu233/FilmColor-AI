"""Tests for FilmColor Darkroom post-grading, geometry, T24 app integration, presets, and multi-format/batch export."""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from filmcolor import app
from filmcolor.darkroom import (
    DarkroomGrade,
    Geometry,
    T24_STYLE_PRESETS,
    apply_darkroom_grade,
    apply_geometry,
    apply_vignette,
    export_image,
)
from test_scanner import scan, write_dng


@pytest.fixture
def darkroom_server(tmp_path, monkeypatch):
    monkeypatch.setenv("FILMCOLOR_DATA_ROOT", str(tmp_path / "cache"))
    root = tmp_path / "neg"
    root.mkdir()
    rgb, irl = scan()
    write_dng(root / "frame01.dng", rgb, irl)
    write_dng(root / "frame02.dng", (rgb * np.array([0.85, 0.90, 0.95])).astype(rgb.dtype), irl)

    srv = app.serve(root, tmp_path / "out", port=0, engine="t24")
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    def call(path: str, body=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"} if body is not None else {},
        )
        try:
            with urllib.request.urlopen(req) as r:
                raw, ct, code, hd = r.read(), r.headers.get("Content-Type", ""), r.status, dict(r.headers)
        except urllib.error.HTTPError as e:
            raw, ct, code, hd = e.read(), e.headers.get("Content-Type", ""), e.code, dict(e.headers)
        return (json.loads(raw) if ct.startswith("application/json") else raw), code, hd

    try:
        yield call, root, tmp_path
    finally:
        srv.shutdown()
        srv.server_close()


def test_darkroom_grade_and_geometry_core():
    rng = np.random.default_rng(7)
    img = rng.uniform(0.1, 0.9, (120, 160, 3)).astype(np.float32)

    # Identity is exact no-op
    assert apply_darkroom_grade(img, DarkroomGrade()) is img
    assert apply_geometry(img, Geometry()) is img

    # Full grade produces valid [0, 1] float32 in both fast and exact modes
    g = DarkroomGrade(
        highlights=-0.2, shadows=0.25, whites=0.1, blacks=-0.05, vibrance=0.2, skin_protect=0.4,
        tone_curve={"rgb": [[0.0, 0.02], [0.5, 0.52], [1.0, 0.98]]},
        cg_shadows=(210.0, 0.25, -0.05), cg_midtones=(30.0, 0.1, 0.02), cg_highlights=(42.0, 0.2, 0.04),
        hsl={"orange": (0.1, 0.15, 0.1), "blue": (-0.1, 0.2, -0.15)},
        dehaze=0.15, clarity=0.2, sharpen=0.35, vignette=-0.3, grain=0.15,
    )
    out_fast = apply_darkroom_grade(img, g, fast=True)
    out_exact = apply_darkroom_grade(img, g, fast=False)
    assert out_fast.shape == img.shape and np.all(np.isfinite(out_fast))
    assert out_exact.shape == img.shape and np.all(np.isfinite(out_exact))
    assert float(np.abs(out_fast - out_exact).mean()) < 0.02

    # B&W skips color-only ops (vibrance, hsl, skin_protect) while allowing tonal + split-tone
    gray = np.repeat(img[..., 1:2], 3, axis=-1)
    g_col = DarkroomGrade(vibrance=0.6, skin_protect=0.8, hsl={"orange": (0.4, 0.5, 0.3)})
    out_bw = apply_darkroom_grade(gray, g_col, film="bw", fast=False)
    assert np.allclose(out_bw[..., 0], out_bw[..., 1]) and np.allclose(out_bw[..., 1], out_bw[..., 2])

    # Geometry rot90 + flip + straighten + crop + vignette
    gm = Geometry(rot90=1, angle=4.0, flip_h=True, crop=(0.1, 0.1, 0.9, 0.9))
    g_out = apply_vignette(apply_geometry(img, gm, fast_preview=False), -0.3)
    assert g_out.shape == (int(round(0.8 * 160)), int(round(0.8 * 120)), 3)
    assert np.all(np.isfinite(g_out)) and g_out.min() >= 0.0 and g_out.max() <= 1.0


def test_static_assets_and_traversal_guard(darkroom_server):
    call, _, _ = darkroom_server
    html, code, _ = call("/")
    assert code == 200 and b"/static/styles.css" in html and b"/static/app.js" in html
    css, code, _ = call("/static/styles.css")
    assert code == 200 and b"--bg-canvas" in css
    js, code, _ = call("/static/app.js")
    assert code == 200 and b"FilmColor Darkroom Studio" in js
    _, code, _ = call("/static/../app.py")
    assert code in (403, 404)


def test_t24_engine_presets_wb_render_and_user_presets(darkroom_server):
    call, _, _ = darkroom_server
    fj, code, _ = call("/api/files")
    assert code == 200 and fj["default_engine"] == "t24" and len(fj["files"]) == 2

    # Open with T24 default
    m, code, _ = call("/api/open", dict(path="frame01.dng", film="color", engine="t24"))
    assert code == 200 and m["engine"] == "t24" and m["auto"]["config"] == "noref_t24"
    assert [p["id"] for p in m["presets"]] == list(T24_STYLE_PRESETS)

    # Switch preset under T24 (including cinema with 3-way color grading)
    m_cin, code, _ = call("/api/preset", dict(token=m["token"], preset="cinema"))
    assert code == 200 and m_cin["preset"] == "cinema" and "cg_shadows" in m_cin["auto_grade"]

    # Switch engine T24 -> T23 -> V3 -> T24 without re-decoding
    m23, code, _ = call("/api/engine", dict(token=m["token"], engine="t23"))
    assert code == 200 and m23["engine"] == "t23" and m23["auto"]["config"] == "noref_t23"
    mv3, code, _ = call("/api/engine", dict(token=m["token"], engine="v3"))
    assert code == 200 and mv3["engine"] == "v3" and mv3["auto"]["config"] == "noref_v3"
    m24, code, _ = call("/api/engine", dict(token=m["token"], engine="t24"))
    assert code == 200 and m24["engine"] == "t24"

    # WB pick works natively under T24
    wb, code, _ = call("/api/wb_pick", dict(token=m24["token"], x=160, y=120, params=m24["auto"]["params"]))
    assert code == 200 and -1.0 <= wb["temp"] <= 1.0 and -1.0 <= wb["tint"] <= 1.0

    # Render with T24 params + DarkroomGrade + Geometry + JPEG streaming
    p_q = urllib.request.quote(json.dumps(m24["auto"]["params"]))
    g_q = urllib.request.quote(json.dumps(dict(clarity=0.2, hsl={"blue": [0.1, 0.2, -0.1]}, vignette=-0.2)))
    gm_q = urllib.request.quote(json.dumps(dict(rot90=1, angle=2.5, crop=[0.05, 0.05, 0.95, 0.95])))
    jpg, code, hd = call(f"/api/render?token={m24['token']}&fmt=jpeg&p={p_q}&g={g_q}&geom={gm_q}")
    assert code == 200 and jpg[:2] == b"\xff\xd8"

    # User presets CRUD
    res, code, _ = call("/api/user_presets", dict(name="Warm Cine", engine="t24", params=dict(ev=0.1), grade=dict(vignette=-0.2)))
    assert code == 200 and any(x["name"] == "Warm Cine" for x in res["presets"])
    got, code, _ = call("/api/user_presets")
    assert code == 200 and len(got["presets"]) == 1
    after, code, _ = call("/api/user_presets/delete", dict(name="Warm Cine"))
    assert code == 200 and len(after["presets"]) == 0


def test_multiformat_and_batch_export(darkroom_server):
    call, _, tmp = darkroom_server
    m, code, _ = call("/api/open", dict(path="frame01.dng", film="color", engine="t24"))
    assert code == 200
    tok = m["token"]
    grade = dict(shadows=0.15, cg_shadows=[200.0, 0.2, 0.0], vignette=-0.25)
    geom = dict(rot90=0, angle=2.0, flip_h=True, crop=[0.05, 0.05, 0.95, 0.95])

    # JPEG export with resize + grade + geometry
    exp_jpg, code, _ = call("/api/export", dict(
        token=tok, params=m["auto"]["params"], grade=grade, geometry=geom,
        format="jpeg", quality=92, max_edge=200, write_sidecar=True,
    ))
    assert code == 200 and exp_jpg["path"].endswith(".jpg") and Path(exp_jpg["path"]).exists()
    with Image.open(exp_jpg["path"]) as im:
        assert max(im.size) == 200
    sc = json.loads(Path(exp_jpg["sidecar"]).read_text())
    assert "darkroom_grade" in sc["render"] and "geometry" in sc["output"]

    # PNG export without sidecar
    exp_png, code, _ = call("/api/export", dict(
        token=tok, params=m["auto"]["params"], grade=grade, format="png", write_sidecar=False,
    ))
    assert code == 200 and exp_png["path"].endswith(".png") and exp_png["sidecar"] is None

    # Batch export both frames sequentially
    job, code, _ = call("/api/batch_export", dict(
        items=[
            dict(path="frame01.dng", film="color", engine="t24", params=m["auto"]["params"], grade=grade),
            dict(path="frame02.dng", film="color", engine="t24", preset="natural"),
        ],
        format="jpeg", quality=90, max_edge=220, write_sidecar=False,
    ))
    assert code == 200 and job["total"] == 2
    deadline = time.time() + 30
    st = None
    while time.time() < deadline:
        st, code, _ = call(f"/api/batch_export?job_id={job['job_id']}")
        assert code == 200
        if st["status"] == "done":
            break
        time.sleep(0.1)
    assert st is not None and st["status"] == "done" and len(st["results"]) == 2 and not st["errors"]


def test_adaptive_presets_respond_to_scene_metrics():
    import numpy as np
    from filmcolor.darkroom import analyze_scene_metrics, resolve_adaptive_preset

    # Flat, muted, hazy-black image vs high-contrast, vivid, clipping-highlight image
    flat_img = np.full((64, 64, 3), [0.38, 0.44, 0.37], dtype=np.float32)
    flat_img[:32, :, :] = [0.58, 0.64, 0.56]
    vivid_img = np.zeros((64, 64, 3), dtype=np.float32)
    vivid_img[:32, :, :] = [0.99, 0.96, 0.92]
    vivid_img[32:56, :, :] = [0.02, 0.25, 0.85]
    vivid_img[56:, :, :] = [0.01, 0.02, 0.04]

    m_flat = analyze_scene_metrics(flat_img)
    m_vivid = analyze_scene_metrics(vivid_img, above_white=0.15)

    # Standard must stay strictly unchanged
    std = resolve_adaptive_preset("standard", "color", m_flat)
    assert std["offsets"] == {} and std["grade"] == {}

    # Frontier should boost contrast/sat/vibrance more on flat+muted than on contrasty+vivid,
    # and pull highlights harder when highlight pressure is high
    f_flat = resolve_adaptive_preset("frontier", "color", m_flat)
    f_vivid = resolve_adaptive_preset("frontier", "color", m_vivid)
    assert f_flat["offsets"]["contrast"] > f_vivid["offsets"]["contrast"]
    assert f_flat["offsets"]["sat"] > f_vivid["offsets"]["sat"]
    assert f_flat["grade"]["vibrance"] > f_vivid["grade"]["vibrance"]
    assert f_vivid["grade"]["highlights"] < f_flat["grade"]["highlights"]
    assert f_flat["grade"]["blacks"] < f_vivid["grade"]["blacks"]  # anchors milky blacks

    # B&W strips color-only grading keys
    f_bw = resolve_adaptive_preset("frontier", "bw", m_flat)
    assert f_bw["camera_matrix_mix"] is None
    assert not any(k in f_bw["grade"] for k in ("vibrance", "skin_protect", "hsl"))


def test_linear_and_leica_presets():
    import numpy as np
    from filmcolor.darkroom import T24_STYLE_PRESETS, analyze_scene_metrics, resolve_adaptive_preset

    assert "linear" in T24_STYLE_PRESETS and "leica" in T24_STYLE_PRESETS
    img = np.linspace(0.05, 0.95, 64 * 64 * 3, dtype=np.float32).reshape(64, 64, 3)
    m = analyze_scene_metrics(img, above_white=0.08)

    lin = resolve_adaptive_preset("linear", "color", m)
    assert lin["offsets"]["contrast"] < -0.10
    assert lin["grade"]["highlights"] < -0.10
    assert lin["grade"]["shadows"] > 0.10
    assert not any(k in lin["grade"] for k in ("hsl", "cg_shadows", "cg_highlights", "vignette", "grain", "sharpen"))

    lei = resolve_adaptive_preset("leica", "color", m)
    assert lei["grade"]["clarity"] >= 0.08
    assert lei["grade"]["vignette"] < 0.0
    assert "red" in lei["grade"]["hsl"] and "blue" in lei["grade"]["hsl"]


def test_export_sidecar_shape_matches_max_edge_resize(darkroom_server):
    """When max_edge resizes an exported image, sidecar output.shape must match the resized output dimensions."""
    call, root, out = darkroom_server
    fl, code, _ = call("/api/files")
    assert code == 200
    meta, code, _ = call("/api/open", dict(path=fl["files"][0], film="color", engine="t24", preset="frontier"))
    assert code == 200
    res, code, _ = call("/api/export", dict(
        token=meta["token"],
        params=meta["auto"]["params"],
        grade=meta["auto_grade"],
        format="jpeg",
        max_edge=96,
        write_sidecar=True,
    ))
    out_im = Image.open(res["path"])
    sc = json.loads(Path(res["sidecar"]).read_text(encoding="utf-8"))
    assert sc["output"]["shape"] == [out_im.height, out_im.width]
    assert max(sc["output"]["shape"]) == 96
    assert sc["render"]["source"]["preset"] == "frontier"
