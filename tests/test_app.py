"""Local interface (filmcolor.app, SW-M2): the HTTP API on a synthetic scanner DNG, no browser. The export must equal
the direct conversion byte for byte, the 1:1 view must equal the export's pixels, and paths outside the root,
unsupported files and frozen files must be refused."""
import json
import struct
import threading
import urllib.error
import urllib.request
import zlib

import numpy as np
import pytest

import filmcolor.app as app
from filmcolor.frozen import FrozenDataError
from filmcolor.pipeline import convert, write
from filmcolor.preview import load_preview

from test_scanner import scan, write_dng

# the app's white point (T4b; T4k: blue white point guard)
APP_WP = dict(white_block=app.WHITE_BLOCK, white_pct=app.WHITE_PCT, white_b_guard=app.WHITE_B_GUARD)


def decode_png(b):
    assert b[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, w, h = 8, b"", None, None
    while pos < len(b):
        n, kind = struct.unpack(">I4s", b[pos:pos + 8])
        data = b[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, h = struct.unpack(">II", data[:8])
        elif kind == b"IDAT":
            idat += data
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, w * 3 + 1)
    assert np.all(raw[:, 0] == 0)
    return raw[:, 1:].reshape(h, w, 3)


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("FILMCOLOR_DATA_ROOT", str(tmp_path))  # app cache (scans, thumbnails, orientation) per test
    root = tmp_path / "neg"
    root.mkdir()
    rgb, irl = scan()
    write_dng(root / "s.dng", rgb, irl)
    (root / "notes.txt").write_text("x")
    (tmp_path / "outside.dng").write_bytes(b"x")
    srv = app.serve(root, tmp_path / "out", port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                data, ct, hd = r.read(), r.headers.get("Content-Type"), dict(r.headers)
                return (json.loads(data) if ct.startswith("application/json") else data), r.status, hd
        except urllib.error.HTTPError as e:
            return json.loads(e.read()), e.code, {}

    yield call, root, tmp_path
    srv.shutdown()


def test_open_render_crop_export(server):
    call, root, tmp = server
    files, code, _ = call("/api/files")
    assert code == 200 and files["files"] == ["s.dng"]
    meta, code, _ = call("/api/open", dict(path="s.dng", film="color"))
    assert code == 200 and meta["base"]["method"] == "auto" and meta["factor"] == 1
    p = dict(ev=0.2, sat=1.3)
    tok = meta["token"]
    png, code, hd = call(f"/api/render?token={tok}&p=" + urllib.request.quote(json.dumps(p)))
    pv = load_preview(root / "s.dng", **APP_WP)
    assert np.array_equal(decode_png(png), app.to8(pv.render(p)))
    assert 0 <= float(hd["X-Below-Black"]) <= 1 and 0 <= float(hd["X-Clip-High"]) <= 1
    full, _ = convert(root / "s.dng", render_params=p, **APP_WP)
    crop, code, _ = call(f"/api/crop?token={tok}&t=10&l=120&size=50&p=" + urllib.request.quote(json.dumps(p)))
    assert np.array_equal(decode_png(crop), app.to8(full[10:60, 120:170]))
    out, code, _ = call("/api/export", dict(params=p, token=tok))
    ref = write(root / "s.dng", tmp / "direct", render_params=p, **APP_WP)
    assert code == 200 and out["sha256"] == ref["output"]["sha256"]
    neg, code, _ = call(f"/api/negative?token={tok}")
    assert code == 200 and decode_png(neg).shape[:2] == (240, 360)


def test_t4h_orange_consensus_does_not_change_the_automatic_base_choice(server):
    """T4h: the T4g routing of orange-consensus rolls to roll_nat is removed; a frame consistent with the roll keeps
    its own B0, and roll_nat is still available on request."""
    _, root, tmp = server
    st = app.State(root, tmp / "out", allow_frozen=False)
    info = st.roll_info()
    info["roll"]["diagnostics"]["orange_consensus"] = True
    assert st.open("s.dng", "color")["base"]["method"] == "auto"
    assert st.open("s.dng", "color", base_mode="roll_nat")["base"]["method"] == "roll_nat"


def test_t4i_exiftool_is_found_outside_path(monkeypatch, tmp_path):
    """T4i: an app started without Homebrew on PATH still finds exiftool at the usual locations."""
    tool = tmp_path / "exiftool"
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    monkeypatch.setattr(app.shutil, "which", lambda name: None)
    monkeypatch.setattr(app, "EXIFTOOL_CANDIDATES", (str(tmp_path / "missing"), str(tool)))
    assert app._exiftool() == str(tool)
    monkeypatch.setattr(app, "EXIFTOOL_CANDIDATES", (str(tmp_path / "missing"),))
    assert app._exiftool() is None


def test_t4i_folder_of_several_rolls_is_not_routed_to_the_roll_base(server, monkeypatch):
    """T4i Bug 4: files from different capture dates (nlp_official: four rolls) are a collection: roll_info reports
    mixed_rolls and flags no frame. T4k: each (body, date) group is its own sub-roll, so a frame alone on its date
    keeps its own base (also in the roll modes), while in a single-date folder of three consistent frames the
    deviating frame (dC 0.23) is flagged and routed to roll_nat (T4k: only in a sub-roll of >= 3 frames without
    warnings)."""
    _, root, tmp = server
    rgb, irl = scan()
    for name in ("u.dng", "v.dng"):
        write_dng(root / name, rgb, irl)
    write_dng(root / "t.dng", (rgb * np.array([0.35, 0.65, 0.95])).astype(rgb.dtype), irl)  # dC 0.23 vs s.dng
    dates = {"s.dng": "2018:06:13", "u.dng": "2018:06:13", "v.dng": "2018:06:13", "t.dng": "2018:07:29"}

    def fake_exif(paths):
        return {p: dict(exposure=1.0, model="X-T2", date=dates[p.name]) for p in paths}

    monkeypatch.setattr(app, "_exif_info", fake_exif)
    st = app.State(root, tmp / "out", allow_frozen=False)
    info = st.roll_info()
    assert info["mixed_rolls"]["n_dates"] == 2 and not any(f["flag"] for f in info["frames"].values())
    assert info["roll"] is None and sorted(info["sub_rolls"]) == ["X-T2 2018:06:13", "X-T2 2018:07:29"]
    t = info["frames"]["t.dng"]
    assert t["sub_roll_single"] and t["roll_rgb"] == t["own_rgb"] == t["nat_rgb"] and t["chroma_part"] == 0
    assert st.open("t.dng", "color")["base"]["method"] == "auto"
    m = st.open("t.dng", "color", base_mode="roll_nat")
    assert m["base"]["method"] == "roll_nat" and m["base"]["rgb"] == pytest.approx(t["own_rgb"])
    assert st.open("s.dng", "color")["base"]["method"] == "auto"
    dates["t.dng"] = "2018:06:13"
    one = app.State(root, tmp / "out", allow_frozen=False)
    info1 = one.roll_info()
    assert info1["mixed_rolls"] is None and list(info1["sub_rolls"]) == [""]
    assert info1["frames"]["t.dng"]["flag"] and one.open("t.dng", "color")["base"]["method"] == "roll_nat"


def test_t4k_black_and_white_frame_of_a_collection_does_not_set_the_colour_roll_base(server, monkeypatch):
    """T4k (nlp_official: _DSF9228 B&W with log2 R/G -0.92 among colour frames at +0.44 on the same date): inside one
    (body, date) group a frame whose own-base chromaticity is more than SUBROLL_CHROMA_TOL from the others is its own
    sub-roll; the colour frames' roll base is theirs alone (T4j took the B&W base for every frame: green rolls)."""
    _, root, tmp = server
    rgb, irl = scan()
    for name in ("u.dng", "v.dng"):
        write_dng(root / name, rgb, irl)
    write_dng(root / "bw.dng", (rgb * np.array([0.3, 1.6, 2.4])).astype(rgb.dtype), irl)  # no orange mask, brighter G
    dates = {"s.dng": "2018:07:28", "u.dng": "2018:07:28", "v.dng": "2018:07:28", "bw.dng": "2018:07:28",
             "w.dng": "2018:07:29"}
    write_dng(root / "w.dng", rgb, irl)

    def fake_exif(paths):
        return {p: dict(exposure=1.0, model="X-T2", date=dates[p.name]) for p in paths}

    monkeypatch.setattr(app, "_exif_info", fake_exif)
    st = app.State(root, tmp / "out", allow_frozen=False)
    info = st.roll_info()
    subs = info["sub_rolls"]
    assert sorted(subs) == ["X-T2 2018:07:28 #1", "X-T2 2018:07:28 #2", "X-T2 2018:07:29"]
    colour = next(v for v in subs.values() if "s.dng" in v["frames"])
    assert sorted(colour["frames"]) == ["s.dng", "u.dng", "v.dng"] and "bw.dng" not in colour["roll"]["members"]
    own = np.array(info["frames"]["s.dng"]["own_rgb"])
    for r in ("s.dng", "u.dng", "v.dng"):
        assert np.allclose(info["frames"][r]["roll_rgb"], own, rtol=0.01)
    assert info["frames"]["bw.dng"]["sub_roll_single"]
    whole = app.estimate_base_roll({r: info["frames"][r]["own_rgb"] for r in ("s.dng", "u.dng", "v.dng", "bw.dng")},
                                   {r: 1.0 for r in ("s.dng", "u.dng", "v.dng", "bw.dng")})
    assert whole.members == ["bw.dng"]  # one base for the whole date group: the B&W frame's


def test_t4k_frame_much_denser_than_its_roll_base_is_routed_to_roll_nat(server, monkeypatch):
    """T4k (A005 / A006 / A011 / A196: B0 found a scene shadow, not the base): in a sub-roll of >= 3 frames without
    warnings a frame whose own base is ROLL_NAT_LEVEL_DIFF denser than the roll's, with the same chromaticity
    (dC < 0.10), opens with roll_nat; with ROLL_NAT_LEVEL_DIFF None (T4j, the default: 0.22 measured 9 better / 14
    worse) it keeps its own base."""
    _, root, tmp = server
    assert app.ROLL_NAT_LEVEL_DIFF is None
    monkeypatch.setattr(app, "ROLL_NAT_LEVEL_DIFF", 0.22)
    rgb, irl = scan()
    for name in ("u.dng", "v.dng"):
        write_dng(root / name, rgb, irl)
    write_dng(root / "d.dng", (rgb * 10 ** -0.3).astype(rgb.dtype), irl)  # 0.30 D denser, same colour

    st = app.State(root, tmp / "out", allow_frozen=False)
    info = st.roll_info()
    d = info["frames"]["d.dng"]
    assert info["mixed_rolls"] is None and d["chroma_part"] < 0.1 and d["level_diff"] >= app.ROLL_NAT_LEVEL_DIFF
    assert not info["roll"]["warnings"]
    assert st.open("d.dng", "color")["base"]["method"] == "roll_nat"
    assert st.open("s.dng", "color")["base"]["method"] == "auto"
    monkeypatch.setattr(app, "ROLL_NAT_LEVEL_DIFF", None)
    assert st.open("d.dng", "color")["base"]["method"] == "auto"


def test_manual_base_region(server):
    call, root, _ = server
    tok = call("/api/open", dict(path="s.dng"))[0]["token"]
    meta, code, _ = call("/api/base", dict(region=[0, 0, 240, 24], token=tok))
    assert code == 200 and meta["base"]["method"] == "manual" and meta["base"]["regions_preview"] == [[0, 0, 240, 24]]
    out, _, _ = call("/api/export", dict(params={}, token=meta["token"]))
    _, sc = convert(root / "s.dng", base_regions=[(0, 0, 240, 24)], **APP_WP)
    exported = json.loads(open(out["sidecar"]).read())
    assert exported["inversion"]["params"] == json.loads(json.dumps(sc["inversion"]["params"]))
    meta, _, _ = call("/api/base", dict(auto=True, token=meta["token"]))
    assert meta["base"]["method"] == "auto"


def test_refuses_outside_unsupported_and_frozen(server, monkeypatch):
    call, root, _ = server
    assert call("/api/open", dict(path="../outside.dng"))[1] == 403
    assert call("/api/open", dict(path="notes.txt"))[1] == 400
    assert call("/api/render")[1] == 400  # nothing open

    def frozen(path, **kw):
        raise FrozenDataError(f"{path} is frozen")
    monkeypatch.setattr(app, "check_source", frozen)
    assert call("/api/open", dict(path="s.dng"))[1] == 403


def q(d):
    return urllib.request.quote(json.dumps(d))


def sha_of(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_export_never_overwrites(server):
    """Review 2026-10-19 #1: a second export of the same photo, or of a same-name photo in another sub-folder,
    must not replace an earlier TIFF / sidecar pair."""
    call, root, tmp = server
    m = call("/api/open", dict(path="s.dng"))[0]
    first, code, _ = call("/api/export", dict(params=dict(ev=0.2), token=m.get("token")))
    assert code == 200
    second, code, _ = call("/api/export", dict(params=dict(ev=-0.5), token=m.get("token")))
    assert code == 200 and second["path"] != first["path"] and second["sidecar"] != first["sidecar"]
    assert sha_of(first["path"]) == first["sha256"]
    assert json.loads(open(first["sidecar"]).read())["render"]["params"]["ev"] == 0.2
    rgb, irl = scan()
    for sub in ("a", "b"):
        (root / sub).mkdir()
        write_dng(root / sub / "s.dng", rgb, irl)
    paths = []
    for sub in ("a", "b"):
        m = call("/api/open", dict(path=f"{sub}/s.dng"))[0]
        out, code, _ = call("/api/export", dict(params={}, token=m.get("token")))
        assert code == 200
        paths.append(out["path"])
    assert len(set(paths + [first["path"], second["path"]])) == 4


def test_stale_window_is_refused(server):
    """Review #2: window A opened s.dng, window B then opened t.dng; A's requests carry A's token and must be
    refused instead of acting on t.dng."""
    call, root, _ = server
    rgb, irl = scan()
    write_dng(root / "t.dng", (rgb * 0.8).astype(rgb.dtype), irl)
    a = call("/api/open", dict(path="s.dng"))[0]
    b = call("/api/open", dict(path="t.dng"))[0]
    assert a.get("token") != b.get("token")
    assert call("/api/export", dict(params={}, token=a.get("token")))[1] == 409
    assert call("/api/base", dict(auto=True, token=a.get("token")))[1] == 409
    assert call(f"/api/render?token={a.get('token')}&p={q({})}")[1] == 409
    assert call(f"/api/crop?token={a.get('token')}&t=0&l=0&size=10&p={q({})}")[1] == 409
    out, code, _ = call("/api/export", dict(params={}, token=b.get("token")))
    assert code == 200 and out["path"].endswith("t.tif")


def test_clip_share_counts_values_before_the_white_clip(server):
    """Review #3: the encoded output saturates at 0.99999994, so '>= 1' on it reports 0 %; the share must come
    from the display-linear values before the white clip, over the image-region mask, any channel."""
    from filmcolor.render import render_full
    call, root, _ = server
    m = call("/api/open", dict(path="s.dng"))[0]
    p = dict(ev=3.0)
    _, code, hd = call(f"/api/render?token={m.get('token')}&p={q(p)}")
    pv = load_preview(root / "s.dng")
    _, _, pre = render_full(pv.positive, app.RenderParams(**p))
    expected = float((pre > 1).any(-1)[pv.mask].mean())
    assert expected > 0.1
    assert code == 200 and float(hd["X-Clip-High"]) == pytest.approx(expected, abs=1e-5)


def test_export_refuses_a_changed_source(server):
    """Review #4: the export replays the previewed base; if the source file changed since it was opened, that
    base belongs to another file and the export must be refused."""
    call, root, tmp = server
    m = call("/api/open", dict(path="s.dng"))[0]
    rgb, irl = scan()
    write_dng(root / "s.dng", (rgb * 0.7).astype(rgb.dtype), irl)
    out, code, _ = call("/api/export", dict(params={}, token=m.get("token")))
    assert code == 409 and not list((tmp / "out").glob("*.tif"))


def test_sidecar_keeps_the_previewed_base_source(server):
    """Review #5: the export passes the previewed base as a given RGB; the sidecar must still say how it was
    found (auto estimate with its diagnostics, or manual regions) and keep the base-inside-image QC check."""
    call, root, _ = server
    m = call("/api/open", dict(path="s.dng"))[0]
    out, _, _ = call("/api/export", dict(params={}, token=m.get("token")))
    sc = json.loads(open(out["sidecar"]).read())
    src = sc["base"].get("source", {})
    assert src.get("method") == "auto" and "estimate" in src
    assert src.get("base_inside_image") == m["base"]["inside_image"]
    if m["base"]["inside_image"] is not None:
        assert "base_inside_image" in [c["name"] for c in sc["qc"]["checks"]]
    m = call("/api/base", dict(region=[0, 0, 240, 24], token=m.get("token")))[0]
    out, _, _ = call("/api/export", dict(params={}, token=m.get("token")))
    src = json.loads(open(out["sidecar"]).read())["base"].get("source", {})
    assert src.get("method") == "manual" and src.get("regions") == [[0, 0, 240, 24]]


def test_roll_base_and_decoupled_dq_modes_export_and_replay(server):
    """T3b / F5 (2026-10-23): /api/roll computes the roll base and per-frame DQ(c_roll) on 48x48 full-frame
    blocks; switching to roll_dq or roll_b1 records the full base.source in the sidecar, keeps base_inside_image
    in QC, and exports bytes identical to direct conversion and sidecar replay."""
    from filmcolor.pipeline import replay_kwargs

    call, root, tmp = server
    rgb, irl = scan()
    # Second frame in the same folder with a coloured scene shadow so its own auto base differs in chromaticity
    rgb2 = (rgb * np.array([0.55, 0.65, 0.75])).astype(rgb.dtype)
    write_dng(root / "t.dng", rgb2, irl)

    roll, code, _ = call("/api/roll")
    assert code == 200 and roll["n_files"] == 2 and roll["roll"] is not None
    assert set(roll["frames"]) == {"s.dng", "t.dng"}
    ft = roll["frames"]["t.dng"]
    assert ft["chroma_part"] is not None and ft["d_lo_shared"] is not None
    assert len(ft["dq_rgb"]) == 3 and len(ft["roll_rgb"]) == 3

    m = call("/api/open", dict(path="t.dng"))[0]
    assert m["base"]["method"] == "auto" and m["base"]["chroma_part"] == pytest.approx(ft["chroma_part"])

    # Switch to ③ DQ(c_roll)
    m_dq, code, _ = call("/api/base", dict(mode="roll_dq", token=m["token"]))
    assert code == 200 and m_dq["base"]["method"] == "roll_dq"
    assert m_dq["base"]["d_lo_shared"] == pytest.approx(ft["d_lo_shared"])
    out_dq, code, _ = call("/api/export", dict(params=dict(ev=0.1), token=m_dq["token"]))
    assert code == 200
    sc_dq = json.loads(open(out_dq["sidecar"]).read())
    src_dq = sc_dq["base"]["source"]
    assert src_dq["method"] == "roll_dq" and src_dq["d_lo_shared"] == pytest.approx(ft["d_lo_shared"])
    assert len(src_dq["block48_L_p01"]) == 3 and src_dq["base_inside_image"] == m["base"]["inside_image"]
    assert "base_inside_image" in [c["name"] for c in sc_dq["qc"]["checks"]]
    ref_dq = write(root / "t.dng", tmp / "direct_dq", **replay_kwargs(sc_dq))
    assert out_dq["sha256"] == ref_dq["output"]["sha256"]

    # Switch to ④ B1 (roll_b1)
    m_b1, code, _ = call("/api/base", dict(mode="roll_b1", token=m_dq["token"]))
    assert code == 200 and m_b1["base"]["method"] == "roll_b1"
    out_b1, code, _ = call("/api/export", dict(params={}, token=m_b1["token"]))
    sc_b1 = json.loads(open(out_b1["sidecar"]).read())
    assert sc_b1["base"]["source"]["method"] == "roll_b1"
    assert sc_b1["base"]["source"]["base_inside_image"] == m["base"]["inside_image"]

    # Switch back to ① B0 (auto)
    m_a, code, _ = call("/api/base", dict(mode="auto", token=m_b1["token"]))
    assert code == 200 and m_a["base"]["method"] == "auto"


def test_roll_nat_wb_pick_thumbnails_and_extended_export(server):
    """T4 (2026-10-23): /api/thumb returns a cached PNG thumbnail, /api/wb_pick computes neutralising temp/tint,
    roll_nat blends paper-black with DQ according to softness, and exporting with extended parameters + tone_curve
    replays byte-for-byte via replay_kwargs."""
    from filmcolor.pipeline import replay_kwargs

    call, root, tmp = server
    rgb, irl = scan()
    rgb2 = (rgb * np.array([0.55, 0.65, 0.75])).astype(rgb.dtype)
    write_dng(root / "t.dng", rgb2, irl)

    roll, code, _ = call("/api/roll")
    assert code == 200 and "nat_rgb" in roll["frames"]["t.dng"]

    # Thumbnail endpoint
    th, code, _ = call("/api/thumb?path=t.dng&film=color")
    assert code == 200 and decode_png(th).shape[2] == 3

    # Open t.dng in roll_nat mode with softness=0.6
    m = call("/api/open", dict(path="t.dng", base_mode="roll_nat", softness=0.6))[0]
    assert m["base"]["method"] == "roll_nat" and m["base"]["softness"] == pytest.approx(0.6)
    assert m["auto"]["config"] == "noref_v3" and 0.95 <= m["auto"]["params"]["contrast"] <= app.NOREF_V3["backlit_contrast_cap"]

    # White balance eyedropper
    wb, code, _ = call("/api/wb_pick", dict(token=m["token"], x=180, y=120))
    assert code == 200 and -1.0 <= wb["temp"] <= 1.0 and -1.0 <= wb["tint"] <= 1.0

    # Export with extended darkroom parameters and tone_curve, and verify byte-identical replay
    ext_params = dict(
        ev=0.15, temp=wb["temp"], tint=wb["tint"], contrast=1.2, sat=1.05,
        highlights=-0.25, shadows=0.2, whites=0.1, blacks=0.08, vibrance=0.15,
        slope_r=0.05, slope_b=-0.05,
        tone_curve={"rgb": [[0.0, 0.02], [0.5, 0.51], [1.0, 0.99]]},
    )
    out, code, _ = call("/api/export", dict(params=ext_params, token=m["token"]))
    assert code == 200
    sc = json.loads(open(out["sidecar"]).read())
    assert sc["base"]["source"]["method"] == "roll_nat"
    assert sc["render"]["params"]["highlights"] == pytest.approx(-0.25)
    assert "tone_curve" in sc["render"]["params"]
    ref = write(root / "t.dng", tmp / "replay_ext", **replay_kwargs(sc))
    assert out["sha256"] == ref["output"]["sha256"]


def test_png_roundtrip():
    img = (np.arange(4 * 5 * 3) % 256).astype(np.uint8).reshape(4, 5, 3)
    assert np.array_equal(decode_png(app.png_bytes(img)), img)


def test_t4b_block_white_point_orientation_and_rotated_export(server):
    """T4b (2026-10-24): the app opens photos with the block-mean white point (recorded in meta and the export
    sidecar), remembers a per-photo orientation set with /api/orient, and exports the turned image: output.rot90 in
    the sidecar, pixels = the unturned conversion turned with numpy.rot90, and replay_kwargs reproduce it."""
    import tifffile
    from filmcolor.pipeline import replay_kwargs

    call, root, tmp = server
    m, code, _ = call("/api/open", dict(path="s.dng"))
    assert code == 200 and m["rot90"] == 0 and m["rot90_source"] == "none"
    assert m["white_point"]["pct"] == app.WHITE_PCT and m["white_point"]["block"] == app.WHITE_BLOCK
    o, code, _ = call("/api/orient", dict(path="s.dng", rot90=-1))
    assert code == 200 and o["rot90"] == 3
    m = call("/api/open", dict(path="s.dng"))[0]
    assert m["rot90"] == 3 and m["rot90_source"] == "saved"
    assert call("/api/orient", dict(path="../outside.dng", rot90=1))[1] == 403

    p = dict(ev=0.1, contrast=1.3)
    out, code, _ = call("/api/export", dict(params=p, token=m["token"], rot90=m["rot90"]))
    assert code == 200 and out["rot90"] == 3
    sc = json.loads(open(out["sidecar"]).read())
    assert sc["output"]["rot90"] == 3 and "counter-clockwise" in sc["output"]["orientation"]
    assert sc["inversion"]["white_point"]["pct"] == app.WHITE_PCT
    plain = write(root / "s.dng", tmp / "plain", render_params=p, **APP_WP)
    assert "rot90" not in plain["output"]
    assert np.array_equal(tifffile.imread(out["path"]), np.rot90(tifffile.imread(plain["output"]["path"]), 3))
    ref = write(root / "s.dng", tmp / "replay_rot", **replay_kwargs(sc))
    assert out["sha256"] == ref["output"]["sha256"]


def test_t4c_scanner_dng_has_no_camera_matrix_and_roll_reports_neutral_ratio(server):
    """T4c: a scanner DNG opens with the identity matrix (no camera matrix to offer), NOREF_V3's chroma_preserve in
    its automatic parameters, and /api/roll reports the neutral-ratio statistic without the app applying it."""
    call, root, tmp = server
    rgb, irl = scan()
    write_dng(root / "t.dng", (rgb * np.array([0.55, 0.65, 0.75])).astype(rgb.dtype), irl)
    roll, code, _ = call("/api/roll")
    assert code == 200 and "neutral_ratio" in roll
    m, code, _ = call("/api/open", dict(path="s.dng"))
    assert code == 200 and m["camera_matrix"] is None and m["camera_matrix_mix"] == app.CAMERA_MATRIX_MIX
    assert m["auto"]["params"]["matrix"] == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert m["auto"]["params"]["chroma_preserve"] == app.NOREF_V3["chroma_preserve"]
    assert "neutral_ratio" not in (m["white_point"] or {})


def test_t4d_style_preset_re_estimates_the_open_photo_and_is_recorded_in_the_export(server):
    """T4d: /api/open starts from the standard preset; /api/preset re-estimates the open photo for another preset
    exactly as opening with it would (same token and positive, no reload), keeps it for the next photo, refuses
    unknown names, and the export sidecar records the preset in render.source."""
    call, root, tmp = server
    m, code, _ = call("/api/open", dict(path="s.dng"))
    assert code == 200 and m["preset"] == "standard" and [p["id"] for p in m["presets"]] == list(app.STYLE_PRESETS)
    std = m["auto"]["params"]
    v, code, _ = call("/api/preset", dict(preset="vivid", token=m["token"]))
    assert code == 200 and v["preset"] == "vivid" and v["token"] == m["token"] and v["white_point"] == m["white_point"]
    assert v["auto"]["params"]["chroma_preserve"] > std["chroma_preserve"] and v["auto"]["params"]["sat"] > std["sat"]
    assert v["auto"]["params"]["blacks"] == app.STYLE_PRESETS["vivid"]["cfg"]["base_blacks"]
    direct = load_preview(root / "s.dng", max_side=1600, auto=app.preset_config("vivid"), **APP_WP)
    assert v["auto"]["params"] == direct.auto["params"]
    bad, code, _ = call("/api/preset", dict(preset="noritsu", token=m["token"]))
    assert code == 400 and "unknown style preset" in bad["error"]
    out, code, _ = call("/api/export", dict(params=v["auto"]["params"], token=v["token"]))
    sc = json.loads(open(out["sidecar"]).read())
    assert code == 200 and sc["render"]["source"]["preset"] == "vivid" and sc["render"]["source"]["auto_rev"] == app.NOREF_V3["rev"]
    again = call("/api/open", dict(path="s.dng"))[0]
    assert again["preset"] == "vivid" and again["auto"]["params"] == v["auto"]["params"]
    p, code, _ = call("/api/open", dict(path="s.dng", preset="portrait"))
    assert code == 200 and p["preset"] == "portrait"
