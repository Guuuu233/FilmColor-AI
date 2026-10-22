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
def server(tmp_path):
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
    pv = load_preview(root / "s.dng")
    assert np.array_equal(decode_png(png), app.to8(pv.render(p)))
    assert 0 <= float(hd["X-Below-Black"]) <= 1 and 0 <= float(hd["X-Clip-High"]) <= 1
    full, _ = convert(root / "s.dng", render_params=p)
    crop, code, _ = call(f"/api/crop?token={tok}&t=10&l=120&size=50&p=" + urllib.request.quote(json.dumps(p)))
    assert np.array_equal(decode_png(crop), app.to8(full[10:60, 120:170]))
    out, code, _ = call("/api/export", dict(params=p, token=tok))
    ref = write(root / "s.dng", tmp / "direct", render_params=p)
    assert code == 200 and out["sha256"] == ref["output"]["sha256"]
    neg, code, _ = call(f"/api/negative?token={tok}")
    assert code == 200 and decode_png(neg).shape[:2] == (240, 360)


def test_manual_base_region(server):
    call, root, _ = server
    tok = call("/api/open", dict(path="s.dng"))[0]["token"]
    meta, code, _ = call("/api/base", dict(region=[0, 0, 240, 24], token=tok))
    assert code == 200 and meta["base"]["method"] == "manual" and meta["base"]["regions_preview"] == [[0, 0, 240, 24]]
    out, _, _ = call("/api/export", dict(params={}, token=meta["token"]))
    _, sc = convert(root / "s.dng", base_regions=[(0, 0, 240, 24)])
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


def test_png_roundtrip():
    img = (np.arange(4 * 5 * 3) % 256).astype(np.uint8).reshape(4, 5, 3)
    assert np.array_equal(decode_png(app.png_bytes(img)), img)
