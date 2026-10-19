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
    png, code, hd = call("/api/render?p=" + urllib.request.quote(json.dumps(p)))
    pv = load_preview(root / "s.dng")
    assert np.array_equal(decode_png(png), app.to8(pv.render(p)))
    assert 0 <= float(hd["X-Below-Black"]) <= 1 and 0 <= float(hd["X-Clip-High"]) <= 1
    full, _ = convert(root / "s.dng", render_params=p)
    crop, code, _ = call(f"/api/crop?t=10&l=120&size=50&p=" + urllib.request.quote(json.dumps(p)))
    assert np.array_equal(decode_png(crop), app.to8(full[10:60, 120:170]))
    out, code, _ = call("/api/export", dict(params=p))
    ref = write(root / "s.dng", tmp / "direct", render_params=p)
    assert code == 200 and out["sha256"] == ref["output"]["sha256"]
    neg, code, _ = call("/api/negative")
    assert code == 200 and decode_png(neg).shape[:2] == (240, 360)


def test_manual_base_region(server):
    call, root, _ = server
    call("/api/open", dict(path="s.dng"))
    meta, code, _ = call("/api/base", dict(region=[0, 0, 240, 24]))
    assert code == 200 and meta["base"]["method"] == "manual" and meta["base"]["regions_preview"] == [[0, 0, 240, 24]]
    out, _, _ = call("/api/export", dict(params={}))
    _, sc = convert(root / "s.dng", base_regions=[(0, 0, 240, 24)])
    exported = json.loads(open(out["sidecar"]).read())
    assert exported["inversion"]["params"] == json.loads(json.dumps(sc["inversion"]["params"]))
    meta, _, _ = call("/api/base", dict(auto=True))
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


def test_png_roundtrip():
    img = (np.arange(4 * 5 * 3) % 256).astype(np.uint8).reshape(4, 5, 3)
    assert np.array_equal(decode_png(app.png_bytes(img)), img)
