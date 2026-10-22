"""Minimal local user interface (docs/SOFTWARE_PLAN_v1.md SW-M2): a small HTTP server on 127.0.0.1 and one HTML
page, Python standard library only (http.server; previews are PNG written with zlib).

  python -m filmcolor.app --root FOLDER [--out EXPORT_FOLDER] [--port 8765] [--allow-frozen]

Only files under ROOT with a supported extension can be opened (camera RAW read by LibRaw, scanner LinearRaw DNG);
paths outside ROOT are refused. Files of the sealed test sets (filmcolor.frozen) are refused unless --allow-frozen.
One photo is open at a time: its full-resolution positive is kept for the 1:1 view (about 0.5 GB for 45 MP).
Every open and base change issues a new token; requests about the open photo must carry it (query ``token`` or body
field ``token``) and are refused with 409 when another window has opened something else since. Exports go to
OUT/<sub-folder of the source>/<stem>.tif + .json and never replace an existing pair (<stem>-2, -3, ...); the export
is refused (409) when the source file changed since it was opened.

API (JSON unless noted): GET /api/files; POST /api/open {path, film} (-> meta with token); GET
/api/render?token=&p=PARAMS (PNG, headers X-Below-Black, X-Clip-High: shares of the image region, at least one
channel, clip measured before the white clip); GET /api/negative?token= (JPEG or PNG); GET
/api/crop?token=&t=&l=&size=&p= (PNG, sensor pixels); POST /api/base {token, region: [top, left, height, width] in
preview pixels} or {token, auto: true}; POST /api/export {token, params, encoding}. PARAMS is a JSON object of
render parameters (render.RenderParams).
"""
from __future__ import annotations

import argparse
import json
import secrets
import struct
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import rawpy

from .compare import block_mean
from .frozen import FrozenDataError, check_source
from .noref import NOREF_V2
from .pipeline import FILM_MODES, free_stem
from .preview import Preview, SourceChangedError, load_preview
from .render import BOUNDS, RenderParams
from .scanner import is_scanner_dng, load_scanner

STATIC = Path(__file__).parent / "app_static"
EXTENSIONS = {".nef", ".nrw", ".raf", ".cr2", ".cr3", ".arw", ".rw2", ".orf", ".pef", ".srw", ".dng"}
SLIDERS = ("ev", "temp", "tint", "sat", "contrast", "midtone", "shoulder")


def png_bytes(img: np.ndarray) -> bytes:
    """8-bit RGB PNG (no filter, zlib level 1)."""
    h, w = img.shape[:2]
    raw = np.empty((h, w * 3 + 1), np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = np.ascontiguousarray(img[..., :3], np.uint8).reshape(h, -1)
    chunk = lambda kind, data: (struct.pack(">I", len(data)) + kind + data
                                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), 1)) + chunk(b"IEND", b""))


def to8(display: np.ndarray) -> np.ndarray:
    return np.round(np.clip(display, 0, 1) * 255).astype(np.uint8)


def negative_view(path: Path, max_side: int = 1600) -> tuple[bytes, str]:
    """The negative as captured, for side-by-side viewing: the camera's embedded JPEG preview, or the scanner
    DNG block-averaged and gamma-encoded for display. Not a colorimetric view."""
    if not is_scanner_dng(path):
        with rawpy.imread(str(path)) as r:
            t = r.extract_thumb()
        if t.format == rawpy.ThumbFormat.JPEG:
            return bytes(t.data), "image/jpeg"
        return png_bytes(np.asarray(t.data)), "image/png"
    rgb = load_scanner(path).rgb
    f = max(1, -(-max(rgb.shape[:2]) // max_side))
    small = block_mean(rgb, f) if f > 1 else rgb
    return png_bytes(to8((small / max(np.percentile(small, 99.9), 1e-6)) ** (1 / 2.2))), "image/png"


class StaleError(RuntimeError):
    """The request names a photo / base that is no longer the open one (another window changed it)."""


class State:
    def __init__(self, root: Path, out: Path, allow_frozen: bool, max_side: int = 1600):
        self.root, self.out, self.allow_frozen, self.max_side = root.resolve(), out, allow_frozen, max_side
        self.preview: Preview | None = None
        self.token: str | None = None
        self.negative: tuple[bytes, str] | None = None
        self.lock = threading.Lock()

    def resolve(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if self.root not in p.parents or not p.is_file():
            raise PermissionError(f"not a file under the root folder: {rel}")
        if p.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"unsupported file type {p.suffix} (camera RAW or scanner DNG only)")
        if not self.allow_frozen:
            check_source(p)  # sealed test sets: refused unless --allow-frozen
        return p

    def files(self) -> list[str]:
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")
                      if p.is_file() and p.suffix.lower() in EXTENSIONS and not p.name.startswith("."))

    def meta(self) -> dict:
        pv = self.preview
        est = pv.base_info.get("estimate", {})
        return dict(token=self.token, path=str(pv.path.relative_to(self.root)), film=pv.film, factor=pv.factor,
                    preview_shape=list(pv.positive.shape[:2]), full_shape=list(pv.full_shape),
                    base=dict(method=pv.base_info.get("method"), rgb=pv.base_info.get("rgb_render"),
                              warnings=est.get("warnings", []), inside_image=pv.base_inside_image,
                              regions_preview=[pv.to_preview(r) for r in (pv.load_kwargs.get("base_regions") or [])]),
                    auto=dict(params=pv.auto["params"], status=pv.auto.get("status"), flags=pv.auto.get("flags"))
                    if pv.auto else None,
                    image_region=list(pv.image_region), region_source=pv.region_source,
                    sliders={k: list(BOUNDS[k]) for k in SLIDERS}, defaults=RenderParams().to_dict())

    def open(self, rel: str, film: str) -> dict:
        if film not in FILM_MODES:
            raise ValueError(f"film must be one of {FILM_MODES}")
        p = self.resolve(rel)
        self.preview = None  # free the previous full-resolution positive first
        self.token = None
        self.preview = load_preview(p, max_side=self.max_side, auto=NOREF_V2, film=film, keep_full=True)
        self.token = secrets.token_hex(8)
        self.negative = None
        return self.meta()

    def need(self, token) -> Preview:
        if self.preview is None:
            raise ValueError("no photo open")
        if token != self.token:
            raise StaleError("another window opened a different photo or changed the base; open it again here")
        return self.preview

    def set_base(self, body: dict) -> dict:
        pv = self.need(body.get("token"))
        if body.get("auto"):
            self.preview = pv.reload(base_regions=None, base_rgb=None)
        else:
            t, l, h, w = (int(v) for v in body["region"])
            if h < 1 or w < 1:
                raise ValueError("empty base region")
            self.preview = pv.reload(base_regions=[pv.to_sensor((t, l, h, w))], base_rgb=None)
        self.token = secrets.token_hex(8)
        return self.meta()

    def export(self, body: dict) -> dict:
        pv = self.need(body.get("token"))
        rel = pv.path.relative_to(self.root)
        out_dir = self.out / rel.parent
        sc = pv.export(out_dir, RenderParams.from_dict(body.get("params") or {}), body.get("encoding", "srgb"),
                       stem=free_stem(out_dir, pv.path.stem), overwrite=False)
        return dict(path=sc["output"]["path"], sha256=sc["output"]["sha256"],
                    sidecar=str(Path(sc["output"]["path"]).with_suffix(".json")),
                    qc=dict(status=sc["qc"]["status"], failing=[c for c in sc["qc"]["checks"] if c["status"] != "pass"]))


def params_of(q: dict) -> RenderParams:
    return RenderParams.from_dict(json.loads(q["p"][0])) if "p" in q else RenderParams()


class Handler(BaseHTTPRequestHandler):
    state: State  # set by serve()

    def log_message(self, fmt, *args):  # quiet
        pass

    def send(self, code: int, body: bytes, ctype: str, headers: dict | None = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def json(self, obj, code: int = 200):
        self.send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def fail(self, e: Exception):
        code = (403 if isinstance(e, (PermissionError, FrozenDataError))
                else 409 if isinstance(e, (StaleError, SourceChangedError, FileExistsError)) else 400)
        self.json(dict(error=f"{type(e).__name__}: {e}"), code)

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        st = self.state
        try:
            if u.path in ("/", "/index.html"):
                return self.send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/files":
                return self.json(dict(root=str(st.root), files=st.files()))
            with st.lock:
                token = q.get("token", [None])[0]
                if u.path == "/api/render":
                    pv = st.need(token)
                    disp, stats = pv.render_stats(params_of(q))
                    return self.send(200, png_bytes(to8(disp)), "image/png",
                                     {"X-Below-Black": f"{stats['below_black']:.6f}",
                                      "X-Clip-High": f"{stats['above_white']:.6f}", "X-Token": st.token})
                if u.path == "/api/negative":
                    pv = st.need(token)
                    if st.negative is None:
                        st.negative = negative_view(pv.path, st.max_side)
                    return self.send(200, st.negative[0], st.negative[1])
                if u.path == "/api/crop":
                    pv = st.need(token)
                    size = min(int(q.get("size", ["600"])[0]), 2000)
                    region = (int(q["t"][0]), int(q["l"][0]), size, size)
                    return self.send(200, png_bytes(to8(pv.render_crop(region, params_of(q)))), "image/png")
            self.json(dict(error="not found"), 404)
        except Exception as e:  # noqa: BLE001 - reported to the page
            self.fail(e)

    def do_POST(self):
        u = urlparse(self.path)
        st = self.state
        try:
            b = self.body()
            with st.lock:
                if u.path == "/api/open":
                    return self.json(st.open(b["path"], b.get("film", "color")))
                if u.path == "/api/base":
                    return self.json(st.set_base(b))
                if u.path == "/api/export":
                    return self.json(st.export(b))
            self.json(dict(error="not found"), 404)
        except Exception as e:  # noqa: BLE001
            self.fail(e)


def serve(root, out, port: int = 8765, allow_frozen: bool = False) -> ThreadingHTTPServer:
    """Bind on 127.0.0.1:port (0 = any free port) and return the server; call serve_forever() on it."""
    handler = type("BoundHandler", (Handler,), dict(state=State(Path(root), Path(out), allow_frozen)))
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def main(argv=None):
    ap = argparse.ArgumentParser(description="FilmColor local interface")
    ap.add_argument("--root", required=True, help="folder with the negatives to open")
    ap.add_argument("--out", default="filmcolor_export", help="export folder (default ./filmcolor_export)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--allow-frozen", action="store_true", help="also open files of the sealed test sets")
    a = ap.parse_args(argv)
    srv = serve(a.root, a.out, a.port, a.allow_frozen)
    print(f"FilmColor: http://127.0.0.1:{srv.server_address[1]}/  (root {Path(a.root).resolve()}, export to "
          f"{Path(a.out).resolve()})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
