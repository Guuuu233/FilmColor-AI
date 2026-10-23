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

Base modes (POST /api/base):
  auto      per-frame automatic base (B0, the default)
  manual    user-drawn rectangle on the preview (region: [top, left, height, width])
  roll_dq   roll base chromaticity c_roll + decoupled 0.1 % quantile black point on full-frame 48x48 blocks (DQ(c_roll))
  roll_b1   full roll base vector (B1)

API (JSON unless noted): GET /api/files; GET /api/roll[?folder=]; POST /api/open {path, film[, base_mode]} (-> meta
with token); GET /api/render?token=&p=PARAMS (PNG, headers X-Below-Black, X-Clip-High: shares of the image region, at
least one channel, clip measured before the white clip); GET /api/negative?token= (JPEG or PNG); GET
/api/crop?token=&t=&l=&size=&p= (PNG, sensor pixels); POST /api/base {token, region: [top, left, height, width] in
preview pixels} or {token, auto: true} or {token, mode: "auto"|"roll_dq"|"roll_b1"}; POST /api/export {token, params,
encoding}. PARAMS is a JSON object of render parameters (render.RenderParams).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
import os
import secrets
import struct
import subprocess
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import rawpy

from .base import (base_deviation, block48_L_p01, decoupled_quantile_base,
                   estimate_base_blocks, estimate_base_roll)
from .compare import block_mean
from .frozen import FrozenDataError, check_source
from .invert import central_region
from .noref import NOREF_V2
from .pipeline import FILM_MODES, free_stem
from .preview import Preview, SourceChangedError, load_preview
from .raw import linearize, saturation_mask, superpixel
from .rawpy_io import load_raw
from .render import BOUNDS, RenderParams
from .scanner import block_means, is_scanner_dng, load_scanner

STATIC = Path(__file__).parent / "app_static"
EXTENSIONS = {".nef", ".nrw", ".raf", ".cr2", ".cr3", ".arw", ".rw2", ".orf", ".pef", ".srw", ".dng"}
SLIDERS = ("ev", "temp", "tint", "sat", "contrast", "midtone", "shoulder")
BASE_MODES = ("auto", "roll_dq", "roll_b1")


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


def _exposures(paths: list[Path]) -> dict[Path, float]:
    """EXIF ExposureTime per file (defaults to 1.0 when absent, e.g. scanner DNGs)."""
    out: dict[Path, float] = {}
    need_tool: list[Path] = []
    try:
        import exifread
    except ImportError:
        exifread = None
    for p in paths:
        val = None
        if exifread is not None:
            try:
                with open(p, "rb") as fh:
                    tags = exifread.process_file(fh, details=False, stop_tag="ExposureTime")
                r = tags.get("EXIF ExposureTime")
                if r and r.values:
                    v = float(Fraction(r.values[0].num, r.values[0].den))
                    if np.isfinite(v) and v > 0:
                        val = v
            except Exception:  # noqa: BLE001
                val = None
        if val is not None:
            out[p] = val
        else:
            need_tool.append(p)
    if need_tool:
        try:
            proc = subprocess.run(
                ["exiftool", "-j", "-n", "-q", "-q", "-ExposureTime", *(str(p) for p in need_tool)],
                capture_output=True, text=True, check=False,
            )
            by_src = {Path(d["SourceFile"]): d.get("ExposureTime") for d in json.loads(proc.stdout or "[]")}
        except Exception:  # noqa: BLE001
            by_src = {}
        for p in need_tool:
            v = by_src.get(p)
            out[p] = float(v) if isinstance(v, (int, float)) and np.isfinite(v) and v > 0 else 1.0
    return out


class StaleError(RuntimeError):
    """The request names a photo / base that is no longer the open one (another window changed it)."""


class State:
    def __init__(self, root: Path, out: Path, allow_frozen: bool, max_side: int = 1600):
        self.root, self.out, self.allow_frozen, self.max_side = root.resolve(), out, allow_frozen, max_side
        self.preview: Preview | None = None
        self.token: str | None = None
        self.negative: tuple[bytes, str] | None = None
        self.lock = threading.Lock()
        self.roll_lock = threading.Lock()
        self.frame_scans: dict[str, dict] = {}
        self.roll_cache: dict[str, dict] = {}

    def resolve(self, rel: str) -> Path:
        cand = Path(os.path.abspath(self.root / rel))
        if self.root not in cand.parents or not cand.is_file():
            raise PermissionError(f"not a file under the root folder: {rel}")
        if cand.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"unsupported file type {cand.suffix} (camera RAW or scanner DNG only)")
        if not self.allow_frozen:
            check_source(cand)  # sealed test sets: refused unless --allow-frozen
        return cand

    def files(self) -> list[str]:
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")
                      if (p.is_file() or p.is_symlink()) and p.suffix.lower() in EXTENSIONS
                      and not p.name.startswith(".") and p.exists())

    def _scan_one(self, rel: str) -> dict:
        p = self.resolve(rel)
        mtime = p.stat().st_mtime_ns
        cached = self.frame_scans.get(rel)
        if cached is not None and cached.get("mtime_ns") == mtime:
            return cached
        if is_scanner_dng(p):
            fr = load_scanner(p, allow_frozen=self.allow_frozen)
            sp, sp_sat = block_means(fr.rgb, fr.sat, 24)
            crop, shape = None, fr.sat.shape
            del fr
        else:
            f = load_raw(p)
            sp, sp_sat = superpixel(linearize(f), f.colors, saturation_mask(f), f.pattern.shape[0], 24)
            crop, shape = f.crop, f.cfa.shape
            del f
        p01 = block48_L_p01(sp, sp_sat, block=24, crop=crop)
        try:
            est = estimate_base_blocks(sp, sp_sat, block=24, crop=crop)
            t, l, h, w = central_region(shape)
            centres = [(r + rh / 2, c + rw / 2) for r, c, rh, rw in est.regions]
            inside = float(np.mean([t <= y < t + h and l <= x < l + w for y, x in centres]))
            res = dict(rel=rel, mtime_ns=mtime, own_rgb=list(est.rgb), estimate=est.to_dict(),
                       warnings=list(est.warnings), inside_image=inside, block48_L_p01=p01, error=None)
        except ValueError as e:
            res = dict(rel=rel, mtime_ns=mtime, own_rgb=None, estimate=None,
                       warnings=[str(e)], inside_image=None, block48_L_p01=p01, error=str(e))
        self.frame_scans[rel] = res
        return res

    def roll_info(self, folder: str = ".") -> dict:
        """Compute (and cache) the roll base B1 and per-frame DQ(c_roll) effective bases for ``folder``."""
        folder = "." if not folder or folder == "." else str(Path(folder))
        with self.roll_lock:
            all_rels = [r for r in self.files() if str(Path(r).parent) == folder]
            if not all_rels:
                raise ValueError(f"no supported files in folder {folder!r}")
            cached = self.roll_cache.get(folder)
            if cached is not None and cached.get("file_list") == all_rels:
                return cached
            paths = [self.resolve(r) for r in all_rels]
            exps_by_path = _exposures(paths)
            with ThreadPoolExecutor(min(4, len(all_rels))) as ex:
                scans = dict(zip(all_rels, ex.map(self._scan_one, all_rels)))
            cand = {r: scans[r]["own_rgb"] for r in all_rels if scans[r]["own_rgb"] is not None}
            exp_map = {r: exps_by_path[p] for r, p in zip(all_rels, paths) if r in cand}
            roll_err = None
            rb = None
            if cand:
                try:
                    rb = estimate_base_roll(cand, exp_map)
                except ValueError as e:
                    roll_err = str(e)
            else:
                roll_err = "no frame in the folder produced a valid automatic base"
            frames_out = {}
            for r, p in zip(all_rels, paths):
                sc = scans[r]
                t = exps_by_path[p]
                entry = dict(
                    exposure=t,
                    own_rgb=sc["own_rgb"],
                    inside_image=sc["inside_image"],
                    own_warnings=sc["warnings"],
                    own_error=sc["error"],
                    block48_L_p01=sc["block48_L_p01"],
                )
                if rb is not None:
                    b_roll = list(rb.for_exposure(t))
                    b_dq, d_lo_dq = decoupled_quantile_base(b_roll, sc["block48_L_p01"])
                    dev = base_deviation(sc["own_rgb"], b_roll) if sc["own_rgb"] is not None else None
                    warn = bool(
                        sc["error"]
                        or (dev is not None and dev["chroma_part"] >= 0.1)
                        or (sc["inside_image"] is not None and sc["inside_image"] > 0.5)
                    )
                    entry.update(
                        roll_rgb=b_roll,
                        dq_rgb=list(b_dq),
                        d_lo_shared=d_lo_dq,
                        deviation=dev,
                        chroma_part=dev["chroma_part"] if dev else None,
                        level_diff=dev["level_diff"] if dev else None,
                        flag=warn,
                    )
                else:
                    entry.update(roll_rgb=None, dq_rgb=None, d_lo_shared=None, deviation=None,
                                 chroma_part=None, level_diff=None, flag=bool(sc["error"]))
                frames_out[r] = entry
            info = dict(
                folder=folder,
                file_list=all_rels,
                n_files=len(all_rels),
                roll=rb.to_dict() if rb is not None else None,
                error=roll_err,
                frames=frames_out,
            )
            self.roll_cache[folder] = info
            return info

    def _roll_base_kwargs(self, rel: str, mode: str) -> dict:
        folder = str(Path(rel).parent)
        rinfo = self.roll_info(folder)
        if rinfo.get("roll") is None:
            raise ValueError(f"roll base unavailable for {folder}: {rinfo.get('error')}")
        fe = rinfo["frames"][rel]
        rb = rinfo["roll"]
        dev = fe["deviation"] or {}
        common = dict(
            method=mode,
            roll_rgb=fe["roll_rgb"],
            own_rgb=fe["own_rgb"],
            chroma_part=fe["chroma_part"],
            level_diff=fe["level_diff"],
            chroma_diff=dev.get("chroma_diff"),
            roll_used=rb["used"],
            roll_members=rb["members"],
            roll_warnings=rb["warnings"],
            base_inside_image=fe["inside_image"],
        )
        if mode == "roll_dq":
            src = dict(common, effective_rgb=fe["dq_rgb"], d_lo_shared=fe["d_lo_shared"],
                       block48_L_p01=fe["block48_L_p01"])
            return dict(base_regions=None, base_rgb=fe["dq_rgb"], base_source=src)
        if mode == "roll_b1":
            return dict(base_regions=None, base_rgb=fe["roll_rgb"], base_source=common)
        raise ValueError(f"unknown roll base mode {mode!r}")

    def meta(self) -> dict:
        pv = self.preview
        rel = str(pv.path.relative_to(self.root))
        src = pv._base_source or {}
        est = pv.base_info.get("estimate", {})
        method = src.get("method") or pv.base_info.get("method")
        folder = str(Path(rel).parent)
        f_roll = self.roll_cache.get(folder, {}).get("frames", {}).get(rel) or {}
        return dict(
            token=self.token, path=rel, film=pv.film, factor=pv.factor,
            preview_shape=list(pv.positive.shape[:2]), full_shape=list(pv.full_shape),
            base=dict(
                method=method,
                rgb=pv.base_info.get("rgb_render"),
                warnings=est.get("warnings", []) or src.get("roll_warnings", []),
                inside_image=pv.base_inside_image,
                regions_preview=[pv.to_preview(r) for r in (pv.load_kwargs.get("base_regions") or [])],
                chroma_part=src.get("chroma_part", f_roll.get("chroma_part")),
                level_diff=src.get("level_diff", f_roll.get("level_diff")),
                d_lo_shared=src.get("d_lo_shared"),
                roll_rgb=src.get("roll_rgb", f_roll.get("roll_rgb")),
                own_rgb=src.get("own_rgb", f_roll.get("own_rgb")),
            ),
            auto=dict(params=pv.auto["params"], status=pv.auto.get("status"), flags=pv.auto.get("flags"))
            if pv.auto else None,
            image_region=list(pv.image_region), region_source=pv.region_source,
            sliders={k: list(BOUNDS[k]) for k in SLIDERS}, defaults=RenderParams().to_dict(),
        )

    def open(self, rel: str, film: str, base_mode: str = "auto") -> dict:
        if film not in FILM_MODES:
            raise ValueError(f"film must be one of {FILM_MODES}")
        if base_mode not in BASE_MODES:
            raise ValueError(f"base_mode must be one of {BASE_MODES}")
        p = self.resolve(rel)
        self.preview = None  # free the previous full-resolution positive first
        self.token = None
        extra = self._roll_base_kwargs(rel, base_mode) if base_mode != "auto" else {}
        self.preview = load_preview(p, max_side=self.max_side, auto=NOREF_V2, film=film, keep_full=True, **extra)
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
        rel = str(pv.path.relative_to(self.root))
        mode = body.get("mode") or ("auto" if body.get("auto") else None)
        if mode == "auto":
            self.preview = pv.reload(base_regions=None, base_rgb=None, base_source=None)
        elif mode in ("roll_dq", "roll_b1"):
            self.preview = pv.reload(**self._roll_base_kwargs(rel, mode))
        elif "region" in body:
            t, l, h, w = (int(v) for v in body["region"])
            if h < 1 or w < 1:
                raise ValueError("empty base region")
            self.preview = pv.reload(base_regions=[pv.to_sensor((t, l, h, w))], base_rgb=None, base_source=None)
        else:
            raise ValueError("set_base requires mode ('auto'|'roll_dq'|'roll_b1') or region")
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
            if u.path == "/api/roll":
                folder = q.get("folder", ["."])[0]
                return self.json(st.roll_info(folder))
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
                    return self.json(st.open(b["path"], b.get("film", "color"), b.get("base_mode", "auto")))
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
    ap.add_argument("--port", type=int, default=None, help="port on 127.0.0.1 (default: try 8765, else any free port)")
    ap.add_argument("--allow-frozen", action="store_true", help="also open files of the sealed test sets")
    a = ap.parse_args(argv)
    if a.port is not None:
        srv = serve(a.root, a.out, a.port, a.allow_frozen)
    else:
        try:
            srv = serve(a.root, a.out, 8765, a.allow_frozen)
        except OSError:
            srv = serve(a.root, a.out, 0, a.allow_frozen)
    print(f"FilmColor: http://127.0.0.1:{srv.server_address[1]}/  (root {Path(a.root).resolve()}, export to "
          f"{Path(a.out).resolve()})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
