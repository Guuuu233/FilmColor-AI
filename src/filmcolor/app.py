"""Minimal local darkroom interface (docs/SOFTWARE_PLAN_v1.md SW-M2 + T4 upgrade): a small HTTP server on
127.0.0.1 and one HTML page, Python standard library only (http.server; previews are PNG written with zlib).

  python -m filmcolor.app --root FOLDER [--out EXPORT_FOLDER] [--port 8765] [--allow-frozen]

Only files under ROOT with a supported extension can be opened (camera RAW read by LibRaw, scanner LinearRaw DNG);
paths outside ROOT are refused. Files of the sealed test sets (filmcolor.frozen) are refused unless --allow-frozen.
One photo is open at a time: its full-resolution positive is kept for the 1:1 view (about 0.5 GB for 45 MP).
Every open and base change issues a new token; requests about the open photo must carry it (query ``token`` or body
field ``token``) and are refused with 409 when another window has opened something else since. Exports go to
OUT/<sub-folder of the source>/<stem>.tif + .json and never replace an existing pair (<stem>-2, -3, ...); the export
is refused (409) when the source file changed since it was opened.

Base modes (POST /api/base):
  auto      per-frame automatic base (B0, the default on normal frames)
  manual    user-drawn rectangle on the preview (region: [top, left, height, width])
  roll_nat  natural roll base fusion (negadoctor paper-black + DQ anti-truncation, softness in [0, 1])
  roll_dq   roll base chromaticity c_roll + decoupled 0.1 % quantile black point on full-frame 48x48 blocks (DQ(c_roll))
  roll_b1   full roll base vector (B1)

When the roll base is ready for a folder and the user opens a frame whose own B0 failed or whose base chromaticity
deviates from the roll by >= 0.10 D (and no explicit base_mode was requested), the server automatically opens it
in ``roll_nat`` mode so the user starts from a clean, well-balanced positive.

Disk cache: per-frame analysis-path scan statistics and 240 px positive thumbnails are cached under
``$FILMCOLOR_DATA_ROOT/app_cache`` (or ``/tmp/filmcolor_app_cache`` when unset) keyed by path, size and mtime_ns,
so restarting the app reopens rolls in < 1 second. Thumbnail names carry ``THUMB_REV``: changing the automatic
estimate or the white point needs a new revision so stale thumbnails are not shown.

White point (T4b): photos are opened with the automatic white point on ``WHITE_BLOCK`` x ``WHITE_BLOCK`` block means
at ``WHITE_PCT`` % (``invert.block_white_point``), not on single pixels, whose high-density tail in the blue channel is
read / shot noise (experiments/t4_auto_ui/t4b_whitepoint.json). The export replays the previewed white point.

Colour (T4c): camera RAW opens with the camera's colour matrix (``render.camera_color_matrix`` of the file's
``rgb_xyz_matrix``) mixed with the identity at the style preset's ``camera_matrix_mix`` (standard:
``CAMERA_MATRIX_MIX``), and the preset's ``chroma_preserve``. Scanner DNGs keep
the identity. ``/api/roll`` also reports each frame's and the roll's 0.25 <= D_G <= 0.75 density ratios (roll base B1,
cached thumbnail-resolution blocks), the statistic T4c proposed for ``invert.neutral_ratio_white_point``. It is not
applied: on dsc6161 the roll median is pulled to blue sky (0.678 : 1 : 1.214) and on dsc6233 to foliage
(0.748 : 1 : 0.904, which would move every frame's white point towards blue); experiments/t4_auto_ui/T4C_REPORT.md.

Style presets (T4d): the automatic starting point is estimated for the selected ``noref.STYLE_PRESETS`` entry
(default ``standard`` = NOREF_V3; POST /api/open with ``preset``, or POST /api/preset to re-estimate the open photo on
its kept estimator input, without decoding, inverting or averaging the positive again). The camera matrix share is the preset's
``camera_matrix_mix``. The selection stays for the next photos; thumbnails always use ``standard``. The export
records the preset in the sidecar's ``render.source``.

Orientation (T4b): each photo opens turned by the orientation last set for it here (POST /api/orient, kept in
``orientation.json`` of the cache folder), else by the RAW file's orientation flag. Display rotation is done by the
page; the export writes the turned image and records ``output.rot90`` in the sidecar.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
import os
import secrets
import struct
import subprocess
import tempfile
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import rawpy

from .base import (base_deviation, block48_L_p01, decoupled_quantile_base,
                   estimate_base_blocks, estimate_base_roll, natural_roll_base)
from .compare import block_mean
from .frozen import FrozenDataError, check_source
from .invert import central_region, invert_negative, neutral_transmission, region_mask
from .noref import NOREF_V3, STYLE_PRESETS, estimate_params_v3, preset_config, wb_from_chroma
from .pipeline import FILM_MODES, free_stem
from .preview import Preview, SourceChangedError, load_preview
from .raw import linearize, saturation_mask, superpixel
from .rawpy_io import load_raw
from .render import (BOUNDS, CURVE_PRESETS, EXTENDED_BOUNDS, EXTENDED_SCALARS, RenderParams, camera_color_matrix,
                     render as render_image)
from .scanner import block_means, is_scanner_dng, load_scanner

STATIC = Path(__file__).parent / "app_static"
EXTENSIONS = {".nef", ".nrw", ".raf", ".cr2", ".cr3", ".arw", ".rw2", ".orf", ".pef", ".srw", ".dng"}
SLIDERS = ("ev", "temp", "tint", "sat", "contrast", "midtone", "shoulder")
BASE_MODES = ("auto", "roll_nat", "roll_dq", "roll_b1")
THUMB_SIDE = 240
THUMB_REV = "t4f"  # thumbnails rendered with the T4f NOREF_V3 (standard preset), camera matrix and white point
WHITE_BLOCK, WHITE_PCT = 16, 99.8
FLIP_TO_ROT90 = {3: 2, 5: 1, 6: 3}  # LibRaw sizes.flip -> numpy.rot90 quarter turns (counter-clockwise) to view
CAMERA_MATRIX_MIX = NOREF_V3["camera_matrix_mix"]  # standard preset's share of the camera matrix (UI slider 0..100)
RATIO_D_G = (0.25, 0.75)  # density band (G, above the roll base) of the per-frame neutral-ratio statistic
RATIO_MIN_PX = 200        # blocks a frame needs in that band to contribute


def _cache_root() -> Path:
    env = os.environ.get("FILMCOLOR_DATA_ROOT")
    if env and Path(env).is_dir():
        d = Path(env) / "app_cache"
    else:
        d = Path(tempfile.gettempdir()) / "filmcolor_app_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


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
        self.cache_dir = _cache_root()
        self.orient_file = self.cache_dir / "orientation.json"
        self.raw_meta: dict[str, dict] = {}
        self.preset = "standard"  # style preset of the automatic starting point (T4d)

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

    def _disk_key(self, p: Path) -> str:
        rp = p.resolve()
        st = rp.stat()
        raw = f"v1|{rp}|{st.st_size}|{st.st_mtime_ns}".encode()
        return hashlib.sha256(raw).hexdigest()[:24]

    def _scan_one(self, rel: str) -> dict:
        p = self.resolve(rel)
        st = p.resolve().stat()
        mtime = st.st_mtime_ns
        cached = self.frame_scans.get(rel)
        if cached is not None and cached.get("mtime_ns") == mtime:
            return cached
        dkey = self._disk_key(p)
        scan_file = self.cache_dir / f"scan_{dkey}.json"
        sp_file = self.cache_dir / f"sp_{dkey}.npy"
        if scan_file.exists():
            try:
                res = json.loads(scan_file.read_text())
                res["rel"] = rel
                self.frame_scans[rel] = res
                return res
            except Exception:  # noqa: BLE001
                pass
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
        # Save a tiny thumbnail-resolution copy of sp (max side <= THUMB_SIDE) for instant positive thumbnails
        tf = max(1, -(-max(sp.shape[:2]) // THUMB_SIDE))
        sp_thumb = block_mean(sp, tf).astype(np.float32) if tf > 1 else sp.astype(np.float32)
        try:
            np.save(sp_file, sp_thumb)
        except Exception:  # noqa: BLE001
            pass
        p01 = block48_L_p01(sp, sp_sat, block=24, crop=crop)
        try:
            est = estimate_base_blocks(sp, sp_sat, block=24, crop=crop)
            t, l, h, w = central_region(shape)
            centres = [(r + rh / 2, c + rw / 2) for r, c, rh, rw in est.regions]
            inside = float(np.mean([t <= y < t + h and l <= x < l + w for y, x in centres]))
            res = dict(rel=rel, mtime_ns=mtime, disk_key=dkey, own_rgb=list(est.rgb), estimate=est.to_dict(),
                       warnings=list(est.warnings), inside_image=inside, block48_L_p01=p01, error=None)
        except ValueError as e:
            res = dict(rel=rel, mtime_ns=mtime, disk_key=dkey, own_rgb=None, estimate=None,
                       warnings=[str(e)], inside_image=None, block48_L_p01=p01, error=str(e))
        try:
            scan_file.write_text(json.dumps(res))
        except Exception:  # noqa: BLE001
            pass
        self.frame_scans[rel] = res
        return res

    def roll_info(self, folder: str = ".") -> dict:
        """Compute (and cache) the roll base B1, DQ(c_roll), and natural_roll_base for ``folder``."""
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
                    disk_key=sc.get("disk_key"),
                    own_rgb=sc["own_rgb"],
                    inside_image=sc["inside_image"],
                    own_warnings=sc["warnings"],
                    own_error=sc["error"],
                    block48_L_p01=sc["block48_L_p01"],
                )
                if rb is not None:
                    b_roll = list(rb.for_exposure(t))
                    entry["density_ratio"] = self._density_ratio(sc, b_roll)
                    b_dq, d_lo_dq = decoupled_quantile_base(b_roll, sc["block48_L_p01"])
                    b_nat, nat_diag = natural_roll_base(b_roll, sc["block48_L_p01"], own_rgb=sc["own_rgb"], softness=0.5)
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
                        nat_rgb=list(b_nat),
                        nat_info=nat_diag,
                        deviation=dev,
                        chroma_part=dev["chroma_part"] if dev else None,
                        level_diff=dev["level_diff"] if dev else None,
                        flag=warn,
                    )
                else:
                    entry.update(roll_rgb=None, dq_rgb=None, d_lo_shared=None, nat_rgb=None, nat_info=None,
                                 deviation=None, chroma_part=None, level_diff=None, flag=bool(sc["error"]))
                frames_out[r] = entry
            ratios = [e["density_ratio"] for e in frames_out.values() if e.get("density_ratio")]
            neutral = dict(ratio=[float(np.median([r["r"] for r in ratios])), float(np.median([r["b"] for r in ratios]))],
                           n_frames=len(ratios), d_g=list(RATIO_D_G)) if ratios else None
            info = dict(
                folder=folder,
                file_list=all_rels,
                n_files=len(all_rels),
                roll=rb.to_dict() if rb is not None else None,
                neutral_ratio=neutral,
                error=roll_err,
                frames=frames_out,
            )
            self.roll_cache[folder] = info
            return info

    def _density_ratio(self, sc: dict, b_roll) -> dict | None:
        """Medians of D_R / D_G and D_B / D_G over the frame's cached thumbnail-resolution blocks with
        RATIO_D_G[0] <= D_G <= RATIO_D_G[1], densities above the roll base ``b_roll`` (same analysis-path signal)."""
        sp_file = self.cache_dir / f"sp_{sc.get('disk_key')}.npy"
        if not sp_file.exists():
            return None
        sp = np.load(sp_file).reshape(-1, 3).astype(np.float64)
        D = -np.log10(np.maximum(sp / np.asarray(b_roll, np.float64), 1e-6))
        sel = (D[:, 1] >= RATIO_D_G[0]) & (D[:, 1] <= RATIO_D_G[1]) & (D[:, 0] > 0) & (D[:, 2] > 0)
        if sel.sum() < RATIO_MIN_PX:
            return None
        d = D[sel]
        return dict(r=float(np.median(d[:, 0] / d[:, 1])), b=float(np.median(d[:, 2] / d[:, 1])), n=int(sel.sum()))

    def _raw_meta(self, p: Path) -> dict:
        """Orientation flag and camera colour matrix of a camera RAW (empty for scanner DNGs), cached per path."""
        key = str(p.resolve())
        if key not in self.raw_meta:
            meta = {}
            if not is_scanner_dng(p):
                try:
                    with rawpy.imread(str(p)) as r:
                        meta = dict(flip=int(r.sizes.flip), rgb_xyz_matrix=np.asarray(r.rgb_xyz_matrix)[:3].tolist())
                except Exception:  # noqa: BLE001
                    meta = {}
            self.raw_meta[key] = meta
        return self.raw_meta[key]

    def camera_matrix(self, p: Path) -> tuple | None:
        """Full camera -> sRGB matrix of a camera RAW, or None (scanner DNG, no or unusable matrix)."""
        m = self._raw_meta(p).get("rgb_xyz_matrix")
        try:
            return camera_color_matrix(m, 1.0) if m is not None else None
        except ValueError:
            return None

    def _colour_kwargs(self, p: Path, film: str, preset: str = "standard") -> dict:
        """NOREF_V3 config of style preset ``preset`` for opening ``p``: colour film from a camera RAW renders with
        its camera matrix at the preset's ``camera_matrix_mix``."""
        cfg = preset_config(preset)
        if film != "color" or self.camera_matrix(p) is None:
            return dict(auto=cfg)
        m = camera_color_matrix(self._raw_meta(p)["rgb_xyz_matrix"], cfg["camera_matrix_mix"])
        return dict(auto=dict(cfg, matrix=m))

    def thumbnail_png(self, rel: str, film: str = "color") -> bytes:
        """Fast 240 px positive thumbnail of ``rel`` (cached on disk)."""
        p = self.resolve(rel)
        dkey = self._disk_key(p)
        thumb_file = self.cache_dir / f"thumb_{film}_{THUMB_REV}_{dkey}.png"
        if thumb_file.exists():
            return thumb_file.read_bytes()
        sc = self._scan_one(rel)
        sp_file = self.cache_dir / f"sp_{sc['disk_key']}.npy"
        if sp_file.exists():
            sp = np.load(sp_file).astype(np.float32)
        else:
            if is_scanner_dng(p):
                fr = load_scanner(p, allow_frozen=self.allow_frozen)
                sp, _ = block_means(fr.rgb, fr.sat, 24)
            else:
                f = load_raw(p)
                sp, _ = superpixel(linearize(f), f.colors, saturation_mask(f), f.pattern.shape[0], 24)
            tf = max(1, -(-max(sp.shape[:2]) // THUMB_SIDE))
            sp = block_mean(sp, tf).astype(np.float32) if tf > 1 else sp.astype(np.float32)
        folder = str(Path(rel).parent)
        fe = self.roll_cache.get(folder, {}).get("frames", {}).get(rel) or {}
        base = fe.get("nat_rgb") or fe.get("roll_rgb") or sc.get("own_rgb")
        if base is None:
            base = [float(x) for x in np.percentile(sp.reshape(-1, 3), 99.0, axis=0)]
        mask = region_mask(sp.shape[:2], central_region(sp.shape[:2]))
        if film == "bw":
            sp_in, inv_base = neutral_transmission(sp, base), (1.0, 1.0, 1.0)
        else:
            sp_in, inv_base = sp, tuple(base)
        ck = self._colour_kwargs(p, film)
        try:  # sp is already 24 px block means
            inv = invert_negative(sp_in, inv_base, mask=mask, white_pct=WHITE_PCT)
            pos = inv.positive
            est = estimate_params_v3(pos, mask, ck["auto"])
            rp = RenderParams.from_dict(est["params"])
            disp, _ = render_image(pos, rp)
        except Exception:  # noqa: BLE001
            disp = np.clip(sp / max(float(np.percentile(sp, 99.5)), 1e-6), 0, 1) ** (1 / 2.2)
        data = png_bytes(to8(disp))
        try:
            thumb_file.write_bytes(data)
        except Exception:  # noqa: BLE001
            pass
        return data

    def _roll_base_kwargs(self, rel: str, mode: str, softness: float = 0.5) -> dict:
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
        if mode == "roll_nat":
            b_nat, nat_diag = natural_roll_base(
                fe["roll_rgb"], fe["block48_L_p01"], own_rgb=fe["own_rgb"], softness=float(softness))
            src = dict(common, effective_rgb=list(b_nat), d_lo_shared=nat_diag["d_lo_used"],
                       softness=float(softness), natural_info=nat_diag, block48_L_p01=fe["block48_L_p01"])
            return dict(base_regions=None, base_rgb=list(b_nat), base_source=src)
        if mode == "roll_dq":
            src = dict(common, effective_rgb=fe["dq_rgb"], d_lo_shared=fe["d_lo_shared"],
                       block48_L_p01=fe["block48_L_p01"])
            return dict(base_regions=None, base_rgb=fe["dq_rgb"], base_source=src)
        if mode == "roll_b1":
            return dict(base_regions=None, base_rgb=fe["roll_rgb"], base_source=common)
        raise ValueError(f"unknown roll base mode {mode!r}")

    def _orientations(self) -> dict:
        try:
            return json.loads(self.orient_file.read_text())
        except (OSError, ValueError):
            return {}

    def orientation(self, p: Path) -> tuple[int, str]:
        """Quarter turns (counter-clockwise) to show ``p`` upright and where that came from."""
        saved = self._orientations().get(str(p.resolve()))
        if saved is not None:
            return int(saved) % 4, "saved"
        flip = self._raw_meta(p).get("flip")
        if flip is not None:
            k = FLIP_TO_ROT90.get(flip, 0)
            return k, "raw_flag" if k else "none"
        return 0, "none"

    def set_orientation(self, body: dict) -> dict:
        p = self.resolve(body["path"])
        k = int(body["rot90"]) % 4
        d = self._orientations()
        d[str(p.resolve())] = k
        self.orient_file.write_text(json.dumps(d, indent=1, sort_keys=True))
        return dict(path=body["path"], rot90=k, source="saved")

    def meta(self) -> dict:
        pv = self.preview
        rel = str(pv.path.relative_to(self.root))
        src = pv._base_source or {}
        est = pv.base_info.get("estimate", {})
        method = src.get("method") or pv.base_info.get("method")
        folder = str(Path(rel).parent)
        f_roll = self.roll_cache.get(folder, {}).get("frames", {}).get(rel) or {}
        all_bounds = {k: list(BOUNDS[k]) for k in SLIDERS} | {k: list(EXTENDED_BOUNDS[k]) for k in EXTENDED_SCALARS}
        rot, rot_src = self.orientation(pv.path)
        return dict(
            token=self.token, path=rel, film=pv.film, factor=pv.factor, rot90=rot, rot90_source=rot_src,
            camera_matrix=[list(r) for r in self.camera_matrix(pv.path)] if self.camera_matrix(pv.path) else None,
            camera_matrix_mix=(pv.load_kwargs.get("auto") or NOREF_V3).get("camera_matrix_mix", CAMERA_MATRIX_MIX),
            preset=pv.auto.get("preset") if pv.auto else None,
            presets=[dict(id=k, label=v["label"]) for k, v in STYLE_PRESETS.items()],
            white_point=dict(pv.report.get("white_point") or dict(method="pixel", pct=pv.white_pct),
                             d_hi=list(pv.inversion["d_hi"])),
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
                softness=src.get("softness", 0.5),
                roll_rgb=src.get("roll_rgb", f_roll.get("roll_rgb")),
                own_rgb=src.get("own_rgb", f_roll.get("own_rgb")),
            ),
            auto=dict(params=pv.auto["params"], status=pv.auto.get("status"), flags=pv.auto.get("flags"),
                      config=pv.auto.get("config"), diagnostics=pv.auto.get("diagnostics"))
            if pv.auto else None,
            image_region=list(pv.image_region), region_source=pv.region_source,
            sliders=all_bounds, curve_presets=CURVE_PRESETS,
            defaults=RenderParams().to_dict(full=True),
        )

    def open(self, rel: str, film: str, base_mode: str | None = None, softness: float = 0.5,
             preset: str | None = None) -> dict:
        if film not in FILM_MODES:
            raise ValueError(f"film must be one of {FILM_MODES}")
        if preset is not None:
            preset_config(preset)  # ValueError for an unknown name
            self.preset = preset
        p = self.resolve(rel)
        folder = str(Path(rel).parent)
        # Smart default: when no explicit base_mode was given and the roll cache is ready, auto-select
        # roll_nat if this frame's own B0 failed or deviates from the roll by >= 0.10 D.
        if base_mode is None:
            rinfo = self.roll_cache.get(folder)
            fe = (rinfo or {}).get("frames", {}).get(rel)
            if rinfo and rinfo.get("roll") and fe and (
                fe.get("own_error") or (fe.get("chroma_part") is not None and fe["chroma_part"] >= 0.1)
            ):
                base_mode = "roll_nat"
            else:
                base_mode = "auto"
        if base_mode not in BASE_MODES:
            raise ValueError(f"base_mode must be one of {BASE_MODES}")
        self.preview = None  # free the previous full-resolution positive first
        self.token = None
        extra = self._roll_base_kwargs(rel, base_mode, softness) if base_mode != "auto" else {}
        ck = self._colour_kwargs(p, film, self.preset)
        try:
            self.preview = load_preview(p, max_side=self.max_side, film=film, keep_full=True,
                                        white_block=WHITE_BLOCK, white_pct=WHITE_PCT, **ck, **extra)
        except ValueError:
            # If auto B0 failed and roll base can be computed for this folder, fall back to roll_nat automatically
            if base_mode == "auto":
                extra = self._roll_base_kwargs(rel, "roll_nat", softness)
                self.preview = load_preview(p, max_side=self.max_side, film=film, keep_full=True,
                                            white_block=WHITE_BLOCK, white_pct=WHITE_PCT, **ck, **extra)
            else:
                raise
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
        softness = float(body.get("softness", 0.5))
        if mode == "auto":
            self.preview = pv.reload(base_regions=None, base_rgb=None, base_source=None)
        elif mode in ("roll_nat", "roll_dq", "roll_b1"):
            self.preview = pv.reload(**self._roll_base_kwargs(rel, mode, softness))
        elif "region" in body:
            t, l, h, w = (int(v) for v in body["region"])
            if h < 1 or w < 1:
                raise ValueError("empty base region")
            self.preview = pv.reload(base_regions=[pv.to_sensor((t, l, h, w))], base_rgb=None, base_source=None)
        else:
            raise ValueError("set_base requires mode ('auto'|'roll_nat'|'roll_dq'|'roll_b1') or region")
        self.token = secrets.token_hex(8)
        return self.meta()

    def set_preset(self, body: dict) -> dict:
        """Re-estimate the open photo's automatic starting point for style preset ``body["preset"]`` on its kept
        full-resolution positive (no decode, base or inversion); the preset stays selected for the next photos."""
        pv = self.need(body.get("token"))
        name = body.get("preset")
        ck = self._colour_kwargs(pv.path, pv.film, name)  # ValueError for an unknown name
        self.preview = pv.with_auto(ck["auto"])
        self.preset = name
        return self.meta()

    def wb_pick(self, body: dict) -> dict:
        """Compute (temp, tint) that neutralises a 5x5 patch around preview pixel (x, y) on the open positive."""
        pv = self.need(body.get("token"))
        x, y = int(round(float(body["x"]))), int(round(float(body["y"])))
        h, w = pv.positive.shape[:2]
        x, y = max(0, min(x, w - 1)), max(0, min(y, h - 1))
        r0, r1 = max(0, y - 2), min(h, y + 3)
        c0, c1 = max(0, x - 2), min(w, x + 3)
        patch = pv.positive[r0:r1, c0:c1].reshape(-1, 3)
        valid = patch[np.all(patch > 1e-4, axis=-1)]
        if len(valid) == 0:
            raise ValueError("selected area is too dark or clipped to sample white balance")
        med = np.median(valid, axis=0)
        chroma = np.array([np.log2(med[0] / med[1]), np.log2(med[2] / med[1])])
        temp, tint = wb_from_chroma(chroma)
        lo_t, hi_t = BOUNDS["temp"]
        lo_i, hi_i = BOUNDS["tint"]
        return dict(
            temp=float(np.clip(temp, lo_t, hi_t)),
            tint=float(np.clip(tint, lo_i, hi_i)),
            sampled_rgb=[float(v) for v in med],
        )

    def export(self, body: dict) -> dict:
        pv = self.need(body.get("token"))
        rel = pv.path.relative_to(self.root)
        out_dir = self.out / rel.parent
        src = dict(preset=pv.auto.get("preset"), auto_config=pv.auto.get("config"), auto_rev=pv.auto.get("config_rev"),
                   note="style preset of the automatic starting point; the exported parameters are the page's")
        sc = pv.export(out_dir, RenderParams.from_dict(body.get("params") or {}), body.get("encoding", "srgb"),
                       stem=free_stem(out_dir, pv.path.stem), overwrite=False, rot90=int(body.get("rot90", 0)),
                       render_source=src if pv.auto else None)
        return dict(path=sc["output"]["path"], sha256=sc["output"]["sha256"], rot90=sc["output"].get("rot90", 0),
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
            if u.path == "/api/thumb":
                rel = q["path"][0]
                film = q.get("film", ["color"])[0]
                return self.send(200, st.thumbnail_png(rel, film), "image/png")
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
                    return self.json(st.open(b["path"], b.get("film", "color"), b.get("base_mode"),
                                             float(b.get("softness", 0.5)), b.get("preset")))
                if u.path == "/api/preset":
                    return self.json(st.set_preset(b))
                if u.path == "/api/base":
                    return self.json(st.set_base(b))
                if u.path == "/api/wb_pick":
                    return self.json(st.wb_pick(b))
                if u.path == "/api/orient":
                    return self.json(st.set_orientation(b))
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
