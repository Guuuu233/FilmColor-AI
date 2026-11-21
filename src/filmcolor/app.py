"""Local darkroom HTTP interface (python -m filmcolor.app --root FOLDER [--out EXPORT_FOLDER] [--port 8765])."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
import hashlib
import io
import json
import os
import secrets
import shutil
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
from .darkroom import (DarkroomGrade, Geometry, HSL_BANDS, T24_STYLE_PRESETS,
                       analyze_scene_metrics, apply_darkroom_grade, apply_geometry, apply_vignette,
                       export_image, free_stem_ext, resolve_adaptive_preset)
from .frozen import FrozenDataError, check_source
from .invert import central_region, invert_negative, neutral_transmission, region_mask
from .noref import (NOREF_T23, NOREF_T24, NOREF_V3, STYLE_PRESETS,
                    estimate_params_v3, preset_config, wb_from_chroma)
from .pipeline import (FILM_MODES, T19_FAMILY, coerce_render_params, convert,
                       free_stem, neutralise_bw_params, sha256)
from .preview import HOLDER_GREY, Preview, SourceChangedError, film_box, holder_mask, load_preview
from .holder import rebate_strip_components, scene_highlight_components
from .raw import linearize, saturation_mask, superpixel
from .rawpy_io import load_raw
from .render import (BOUNDS, CURVE_PRESETS, EXTENDED_BOUNDS, EXTENDED_SCALARS,
                     MID_GREY, RenderParams, RenderParamsT19, RenderParamsT24,
                     T19_BOUNDS, T24_BOUNDS, camera_color_matrix, pre_shoulder_linear,
                     render as render_image, srgb_eotf, _t24_linear_matrix)
from .scanner import block_means, is_scanner_dng, load_scanner

STATIC = Path(__file__).parent / "app_static"
STATIC_MIME = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}
EXTENSIONS = {".nef", ".nrw", ".raf", ".cr2", ".cr3", ".arw", ".rw2", ".orf", ".pef", ".srw", ".dng", ".tif", ".tiff"}
SLIDERS = ("ev", "temp", "tint", "sat", "contrast", "midtone", "shoulder")
BASE_MODES = ("auto", "roll_nat", "roll_dq", "roll_b1")
ENGINES = ("t24", "t23", "v3")
ENGINE_LABELS = (
    dict(id="t24", label="T24 AI 残差（推荐）", desc="最新物理+残差树联合模型，精度最高"),
    dict(id="t23", label="T23 物理基准", desc="纯物理参数估计 + T19 正交相纸曲线"),
    dict(id="v3", label="V3 经典引擎", desc="经典暗房规则引擎（兼容旧预设）"),
)
T24_CAM_MATRIX_MIX = 0.35
THUMB_SIDE = 240
THUMB_REV = "t5b"  # T5b: re-estimates camera-RAW bases on clipped light, so scans and thumbnails get new keys
WHITE_BLOCK, WHITE_PCT = 16, 99.8
SPIKE_CLAMP = (1.15, 0.12)  # T4l: invert.clamp_spikes (mid_max, margin); None = off
WHITE_SELF_RATIO = None
WHITE_B_GUARD = (1.06, 1.18, 0.65, 0.50)  # T4m/T5h: C-41 orange-mask gate (base B/G < 0.50, d_hi B/G < 1.06)
SUBROLL_CHROMA_TOL = 0.15
ROLL_NAT_MIN_FRAMES = 3
ROLL_NAT_LEVEL_DIFF = None
ROLL_NAT_BW_LEVEL_DIFF = 0.06
ROLL_BASE_MAX = 1.25
FLIP_TO_ROT90 = {3: 2, 5: 1, 6: 3}
CAMERA_MATRIX_MIX = NOREF_V3["camera_matrix_mix"]
RATIO_D_G = (0.25, 0.75)
RATIO_MIN_PX = 200


def _cache_root() -> Path:
    env = os.environ.get("FILMCOLOR_DATA_ROOT")
    if env and Path(env).is_dir():
        d = Path(env) / "app_cache"
    else:
        d = Path(tempfile.gettempdir()) / "filmcolor_app_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp.npy")
    try:
        np.save(tmp, arr)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


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


def jpeg_bytes(img: np.ndarray, quality: int = 92) -> bytes:
    """Fast 8-bit RGB JPEG for interactive preview streaming (~15x smaller than PNG)."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img[..., :3], np.uint8), mode="RGB").save(
        buf, format="JPEG", quality=quality, subsampling=0 if quality >= 92 else 2
    )
    return buf.getvalue()


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


EXIFTOOL_CANDIDATES = ("/opt/homebrew/bin/exiftool", "/usr/local/bin/exiftool")


def _exiftool() -> str | None:
    found = shutil.which("exiftool")
    return found or next((c for c in EXIFTOOL_CANDIDATES if os.access(c, os.X_OK)), None)


def _exif_info(paths: list[Path]) -> dict[Path, dict]:
    """EXIF ExposureTime (1.0 when absent, e.g. scanner DNGs), camera Model and capture date per file."""
    out: dict[Path, dict] = {}
    need_tool: list[Path] = []
    try:
        import exifread
    except ImportError:
        exifread = None
    for p in paths:
        info = dict(exposure=None, model=None, date=None)
        if exifread is not None:
            try:
                with open(p, "rb") as fh:
                    tags = exifread.process_file(fh, details=False, stop_tag="DateTimeOriginal")
                r = tags.get("EXIF ExposureTime")
                if r and r.values:
                    v = float(Fraction(r.values[0].num, r.values[0].den))
                    if np.isfinite(v) and v > 0:
                        info["exposure"] = v
                model, date = tags.get("Image Model"), tags.get("EXIF DateTimeOriginal")
                info["model"] = str(model).strip() or None if model else None
                info["date"] = str(date)[:10] or None if date else None
            except Exception:  # noqa: BLE001
                pass
        out[p] = info
        if info["exposure"] is None:
            need_tool.append(p)
    tool = _exiftool() if need_tool else None
    if tool:
        try:
            proc = subprocess.run(
                [tool, "-j", "-n", "-q", "-q", "-ExposureTime", "-Model", "-DateTimeOriginal", "-CreateDate",
                 *(str(p) for p in need_tool)], capture_output=True, text=True, check=False,
            )
            by_src = {Path(d["SourceFile"]): d for d in json.loads(proc.stdout or "[]")}
        except Exception:  # noqa: BLE001
            by_src = {}
        for p in need_tool:
            d = by_src.get(p, {})
            v = d.get("ExposureTime")
            if isinstance(v, (int, float)) and np.isfinite(v) and v > 0:
                out[p]["exposure"] = float(v)
            if d.get("Model") is not None:
                out[p]["model"] = out[p]["model"] or str(d["Model"]).strip() or None
            date = d.get("DateTimeOriginal") or d.get("CreateDate")
            if date is not None:
                out[p]["date"] = out[p]["date"] or str(date)[:10] or None
    for info in out.values():
        if info["exposure"] is None:
            info["exposure"] = 1.0
    return out


def mixed_rolls(exif: dict) -> dict | None:
    models = {v["model"] for v in exif.values() if v["model"]}
    dates = {v["date"] for v in exif.values() if v["date"]}
    if len(models) <= 1 and len(dates) <= 1:
        return None
    groups: dict[str, list] = {}
    for p, v in exif.items():
        groups.setdefault(f"{v['model']} {v['date']}", []).append(Path(p).name)
    return dict(n_models=len(models), n_dates=len(dates), groups=groups)


def sub_rolls(rels: list[str], exif: dict, own: dict, mixed: dict | None) -> dict[str, list[str]]:
    if mixed is None:
        return {"": list(rels)}
    groups: dict[tuple, list[str]] = {}
    for r in rels:
        groups.setdefault((exif[r]["model"], exif[r]["date"]), []).append(r)
    out: dict[str, list[str]] = {}
    for (model, date), rs in groups.items():
        c = {r: np.log2(np.asarray(own[r], np.float64)[[0, 2]] / own[r][1]) for r in rs if own.get(r) is not None}
        left, clusters = [r for r in rs if r in c], []

        def near(centre, pool):
            return [r for r in pool if np.max(np.abs(c[r] - centre)) <= SUBROLL_CHROMA_TOL]

        while left:
            seed = np.median([c[r] for r in left], 0)
            mem = near(seed, left) or [min(left, key=lambda r: float(np.max(np.abs(c[r] - seed))))]
            mem = near(np.median([c[r] for r in mem], 0), left) or mem
            clusters.append(mem)
            left = [r for r in left if r not in mem]
        rest = [r for r in rs if r not in c]
        if rest:
            if clusters:
                max(clusters, key=len).extend(rest)
            else:
                clusters.append(rest)
        name = f"{model or '?'} {date or '?'}"
        for i, mem in enumerate(clusters):
            out[name if len(clusters) == 1 else f"{name} #{i + 1}"] = [r for r in rs if r in mem]
    return out


def _scene_metrics_for_preview(pv: Preview) -> dict[str, float] | None:
    """Compute & cache lightweight scene metrics once per Preview from its locked auto render (~4 ms)."""
    cached = getattr(pv, "_scene_metrics", None)
    if cached is not None:
        return cached
    try:
        raw_p = (pv.auto or {}).get("params")
        if not raw_p:
            return None
        rp = coerce_render_params(dict(raw_p))
        disp, stats = pv.render_stats(rp)
        metrics = analyze_scene_metrics(disp, pv.mask, above_white=stats.get("above_white"))
        pv._scene_metrics = metrics
        return metrics
    except Exception:  # noqa: BLE001
        return None


def _apply_t24_preset(
    base_params: dict, preset: str, film: str, engine: str, *, raw_matrix=None, scene_metrics: dict | None = None
) -> tuple[dict, dict]:
    """Apply an adaptive darkroom style preset cleanly on top of locked T24 / T23 auto params."""
    spec = resolve_adaptive_preset(preset, film=film, scene_metrics=scene_metrics)
    bounds = T24_BOUNDS if engine == "t24" else T19_BOUNDS
    p = dict(base_params)
    mix = spec.get("camera_matrix_mix")
    if mix is not None and film == "color" and raw_matrix is not None:
        try:
            m = camera_color_matrix(raw_matrix, float(mix))
            p["matrix"] = [list(r) for r in m]
        except ValueError:
            pass
    color_keys = {"temp", "tint", "sat", "slope_r", "slope_b"}
    for k, dv in spec["offsets"].items():
        if k not in bounds:
            continue
        if film == "bw" and k in color_keys:
            continue
        lo, hi = bounds[k]
        p[k] = float(np.clip(float(p.get(k, 0.0)) + float(dv), lo, hi))
    grade = dict(spec["grade"])
    if film == "bw":
        for ck in ("vibrance", "skin_protect", "hsl"):
            grade.pop(ck, None)
    return p, grade


class StaleError(RuntimeError):
    """The request names a photo / base that is no longer the open one (another window changed it)."""


class State:
    def __init__(self, root: Path, out: Path, allow_frozen: bool, max_side: int = 1600, engine: str = "v3"):
        if engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
        self.root, self.out, self.allow_frozen, self.max_side = root.resolve(), out, allow_frozen, max_side
        self.preview: Preview | None = None
        self.token: str | None = None
        self.negative: tuple[bytes, str] | None = None
        self.lock = threading.Lock()
        self.roll_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._batch_lock = threading.Lock()
        self.frame_scans: dict[str, dict] = {}
        self.roll_cache: dict[str, dict] = {}
        self.batch_jobs: dict[str, dict] = {}
        self.cache_dir = _cache_root()
        self.orient_file = self.cache_dir / "orientation.json"
        self.user_presets_file = self.cache_dir / "user_presets.json"
        self.raw_meta: dict[str, dict] = {}
        self.preset = "standard"
        self.engine = engine

    def resolve(self, rel: str) -> Path:
        cand = Path(os.path.abspath(self.root / rel))
        if self.root not in cand.parents or not cand.is_file():
            raise PermissionError(f"not a file under the root folder: {rel}")
        if cand.suffix.lower() not in EXTENSIONS:
            raise ValueError(f"unsupported file type {cand.suffix} (camera RAW or scanner DNG only)")
        if not self.allow_frozen:
            check_source(cand)
        if cand.suffix.lower() in (".tif", ".tiff") and not is_scanner_dng(cand, allow_frozen=self.allow_frozen):
            raise ValueError(f"unsupported TIFF file {cand.name} (16-bit scanner negative TIFF only)")
        return cand

    def _is_valid_candidate(self, p: Path) -> bool:
        suf = p.suffix.lower()
        if suf not in (".tif", ".tiff"):
            return True
        if not self.allow_frozen:
            try:
                check_source(p)
            except FrozenDataError:
                return False
        return is_scanner_dng(p, allow_frozen=self.allow_frozen)

    def files(self) -> list[str]:
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")
                      if (p.is_file() or p.is_symlink()) and p.suffix.lower() in EXTENSIONS
                      and not p.name.startswith(".") and p.exists()
                      and self._is_valid_candidate(p))

    def _disk_key(self, p: Path) -> str:
        rp = p.resolve()
        st = rp.stat()
        raw = f"t5b|{rp}|{st.st_size}|{st.st_mtime_ns}".encode()
        return hashlib.sha256(raw).hexdigest()[:24]

    def _load_sp(self, p: Path):
        is_scan = is_scanner_dng(p, allow_frozen=self.allow_frozen)
        if is_scan:
            fr = load_scanner(p, allow_frozen=self.allow_frozen)
            return (*block_means(fr.rgb, fr.sat, 24), None, fr.sat.shape, True)
        f = load_raw(p)
        return (*superpixel(linearize(f), f.colors, saturation_mask(f), f.pattern.shape[0], 24), f.crop, f.cfa.shape, False)

    def _scan_one(self, rel: str) -> dict:
        p = self.resolve(rel)
        st = p.resolve().stat()
        mtime = st.st_mtime_ns
        with self._cache_lock:
            cached = self.frame_scans.get(rel)
        if cached is not None and cached.get("mtime_ns") == mtime:
            return cached
        dkey = self._disk_key(p)
        scan_file = self.cache_dir / f"scan_{dkey}.json"
        sp_file = self.cache_dir / f"sp_{dkey}.npy"
        if scan_file.exists():
            try:
                res = dict(json.loads(scan_file.read_text()), rel=rel)
                with self._cache_lock:
                    self.frame_scans[rel] = res
                return res
            except Exception:  # noqa: BLE001
                pass
        sp, sp_sat, crop, shape, is_scan = self._load_sp(p)
        tf = max(1, -(-max(sp.shape[:2]) // THUMB_SIDE))
        sp_thumb = block_mean(sp, tf).astype(np.float32) if tf > 1 else sp.astype(np.float32)
        try:
            _atomic_save_npy(sp_file, sp_thumb)
        except Exception:  # noqa: BLE001
            pass
        p01 = block48_L_p01(sp, sp_sat, block=24, crop=crop)
        try:
            est = estimate_base_blocks(sp, sp_sat, block=24, crop=crop, exclude_near_saturated=not is_scan)
            t, l, h, w = central_region(shape)
            centres = [(r + rh / 2, c + rw / 2) for r, c, rh, rw in est.regions]
            inside = float(np.mean([t <= y < t + h and l <= x < l + w for y, x in centres]))
            res = dict(rel=rel, mtime_ns=mtime, disk_key=dkey, own_rgb=list(est.rgb), estimate=est.to_dict(),
                       warnings=list(est.warnings), inside_image=inside, block48_L_p01=p01, error=None)
        except ValueError as e:
            res = dict(rel=rel, mtime_ns=mtime, disk_key=dkey, own_rgb=None, estimate=None,
                       warnings=[str(e)], inside_image=None, block48_L_p01=p01, error=str(e))
        try:
            _atomic_write_bytes(scan_file, json.dumps(res).encode("utf-8"))
        except Exception:  # noqa: BLE001
            pass
        with self._cache_lock:
            self.frame_scans[rel] = res
        return res

    def roll_info(self, folder: str = ".") -> dict:
        folder = "." if not folder or folder == "." else str(Path(folder))
        with self.roll_lock:
            all_rels = [r for r in self.files() if str(Path(r).parent) == folder]
            if not all_rels:
                raise ValueError(f"no supported files in folder {folder!r}")
            cached = self.roll_cache.get(folder)
            if cached is not None and cached.get("file_list") == all_rels:
                return cached
            paths = [self.resolve(r) for r in all_rels]
            exif = _exif_info(paths)
            exps_by_path = {p: v["exposure"] for p, v in exif.items()}
            mixed = mixed_rolls(exif)
            with ThreadPoolExecutor(min(4, len(all_rels))) as ex:
                scans = dict(zip(all_rels, ex.map(self._scan_one, all_rels)))
            cand = {r: scans[r]["own_rgb"] for r in all_rels if scans[r]["own_rgb"] is not None}
            exp_map = {r: exps_by_path[p] for r, p in zip(all_rels, paths) if r in cand}
            groups = sub_rolls(all_rels, {r: exif[p] for r, p in zip(all_rels, paths)}, cand, mixed)
            subs, sub_of = {}, {}
            for name, rs in groups.items():
                c_sub = {r: cand[r] for r in rs if r in cand}
                rb_sub, err_sub = None, None
                if c_sub:
                    try:
                        rb_sub = estimate_base_roll(c_sub, {r: exp_map[r] for r in c_sub})
                    except ValueError as e:
                        err_sub = str(e)
                else:
                    err_sub = "no frame in the sub-roll produced a valid automatic base"
                subs[name] = dict(frames=rs, n_base=len(c_sub), rb=rb_sub, error=err_sub)
                sub_of.update({r: name for r in rs})
            if mixed is None:
                rb, roll_err = subs[""]["rb"], subs[""]["error"]
                if not cand:
                    roll_err = "no frame in the folder produced a valid automatic base"
            else:
                rb, roll_err = None, None
            frames_out = {}
            for r, p in zip(all_rels, paths):
                sc = scans[r]
                t = exps_by_path[p]
                sub = subs[sub_of[r]]
                rb_f = sub["rb"]
                entry = dict(
                    exposure=t,
                    disk_key=sc.get("disk_key"),
                    own_rgb=sc["own_rgb"],
                    inside_image=sc["inside_image"],
                    own_warnings=sc["warnings"],
                    own_error=sc["error"],
                    estimate=sc.get("estimate"),
                    block48_L_p01=sc["block48_L_p01"],
                    sub_roll=sub_of[r],
                    sub_roll_single=bool(mixed is not None and sub["n_base"] == 1 and sc["own_rgb"] is not None),
                )
                if rb_f is not None and entry["sub_roll_single"]:
                    own = list(sc["own_rgb"])
                    entry.update(roll_rgb=own, dq_rgb=own, d_lo_shared=None, nat_rgb=own, nat_info=None,
                                 deviation=base_deviation(own, own), chroma_part=0.0, level_diff=0.0, flag=False,
                                 density_ratio=self._density_ratio(sc, own))
                elif rb_f is not None:
                    b_roll = list(rb_f.for_exposure(t))
                    entry["roll_implausible"] = bool(max(b_roll) > ROLL_BASE_MAX)
                    entry["density_ratio"] = self._density_ratio(sc, b_roll)
                    b_dq, d_lo_dq = decoupled_quantile_base(b_roll, sc["block48_L_p01"])
                    b_nat, nat_diag = natural_roll_base(b_roll, sc["block48_L_p01"], own_rgb=sc["own_rgb"], softness=0.5)
                    dev = base_deviation(sc["own_rgb"], b_roll) if sc["own_rgb"] is not None else None
                    warn = bool(
                        sc["error"]
                        or (dev is not None and dev["chroma_part"] >= 0.1)
                        or (sc["inside_image"] is not None and sc["inside_image"] > 0.5)
                    ) and mixed is None
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
                mixed_rolls=mixed,
                sub_rolls={k: dict(frames=v["frames"], n_base=v["n_base"], roll=v["rb"].to_dict() if v["rb"] else None,
                                   error=v["error"]) for k, v in subs.items()},
                neutral_ratio=neutral,
                error=roll_err,
                frames=frames_out,
            )
            self.roll_cache[folder] = info
            return info

    def _density_ratio(self, sc: dict, b_roll) -> dict | None:
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
        key = str(p.resolve())
        with self._cache_lock:
            cached = self.raw_meta.get(key)
        if cached is not None:
            return cached
        meta = {}
        if not is_scanner_dng(p, allow_frozen=self.allow_frozen):
            try:
                with rawpy.imread(str(p)) as r:
                    meta = dict(flip=int(r.sizes.flip), rgb_xyz_matrix=np.asarray(r.rgb_xyz_matrix)[:3].tolist())
            except Exception:  # noqa: BLE001
                meta = {}
        with self._cache_lock:
            self.raw_meta[key] = meta
        return meta

    def camera_matrix(self, p: Path) -> tuple | None:
        m = self._raw_meta(p).get("rgb_xyz_matrix")
        try:
            return camera_color_matrix(m, 1.0) if m is not None else None
        except ValueError:
            return None

    def _colour_kwargs(self, p: Path, film: str, preset: str = "standard", engine: str | None = None) -> dict:
        eng = engine or self.engine
        if eng in ("t24", "t23"):
            if preset not in T24_STYLE_PRESETS:
                raise ValueError(f"unknown style preset {preset!r}; one of {sorted(T24_STYLE_PRESETS)}")
            base_cfg = NOREF_T24 if eng == "t24" else NOREF_T23
            cfg = dict(base_cfg, preset=preset, camera_matrix_mix=T24_CAM_MATRIX_MIX)
            if film != "color":
                return dict(auto=cfg)
            raw_m = self._raw_meta(p).get("rgb_xyz_matrix")
            if raw_m is None:
                return dict(auto=cfg)
            try:
                m = camera_color_matrix(raw_m, T24_CAM_MATRIX_MIX)
            except ValueError:
                return dict(auto=cfg)
            return dict(auto=dict(cfg, matrix=m))

        cfg = preset_config(preset)
        if film != "color":
            return dict(auto=cfg)
        raw_m = self._raw_meta(p).get("rgb_xyz_matrix")
        if raw_m is None:
            return dict(auto=cfg)
        try:
            m = camera_color_matrix(raw_m, cfg["camera_matrix_mix"])
        except ValueError:
            return dict(auto=cfg)
        return dict(auto=dict(cfg, matrix=m))

    def thumbnail_png(self, rel: str, film: str = "color") -> bytes:
        if film not in FILM_MODES:
            raise ValueError(f"film must be one of {FILM_MODES}")
        p = self.resolve(rel)
        dkey = self._disk_key(p)
        thumb_prefix = "thumb_bw" if film == "bw" else "thumb"
        thumb_file = self.cache_dir / f"{thumb_prefix}_{THUMB_REV}_{dkey}.png"
        if thumb_file.exists():
            return thumb_file.read_bytes()
        sc = self._scan_one(rel)
        sp_file = self.cache_dir / f"sp_{sc['disk_key']}.npy"
        if sp_file.exists():
            sp = np.load(sp_file).astype(np.float32)
        else:
            sp = self._load_sp(p)[0]
            tf = max(1, -(-max(sp.shape[:2]) // THUMB_SIDE))
            sp = block_mean(sp, tf).astype(np.float32) if tf > 1 else sp.astype(np.float32)
        folder = str(Path(rel).parent)
        rinfo = self.roll_cache.get(folder, {})
        fe = rinfo.get("frames", {}).get(rel) or {}
        roll_base = None if rinfo.get("mixed_rolls") else (fe.get("nat_rgb") or fe.get("roll_rgb"))
        base = roll_base or sc.get("own_rgb")
        if base is None:
            base = [float(x) for x in np.percentile(sp.reshape(-1, 3), 99.0, axis=0)]
        mask = region_mask(sp.shape[:2], central_region(sp.shape[:2]))
        if film == "bw":
            sp_in, inv_base = neutral_transmission(sp, base), (1.0, 1.0, 1.0)
        else:
            sp_in, inv_base = sp, tuple(base)
        ck = self._colour_kwargs(p, film, preset="standard", engine="v3")
        try:
            inv = invert_negative(sp_in, inv_base, mask=mask, white_pct=WHITE_PCT,
                                  white_b_guard=WHITE_B_GUARD if film != "bw" else None,
                                  white_self_ratio=WHITE_SELF_RATIO if film != "bw" else None, spike_clamp=SPIKE_CLAMP)
            pos = inv.positive
            est = estimate_params_v3(pos, mask, ck["auto"], span_d_g=inv.params.d_hi[1] - inv.params.d_lo[1])
            raw_params = est.get("params") or {}
            est_params = neutralise_bw_params(raw_params)[0] if film == "bw" else raw_params
            rp = RenderParams.from_dict(est_params)
            disp, _ = render_image(pos, rp, fast=True)
            hm = holder_mask(pos)
            disp[(hm & ~scene_highlight_components(hm)) | rebate_strip_components(pos, hm)] = HOLDER_GREY
        except Exception:  # noqa: BLE001
            disp = np.clip(sp_in / max(float(np.percentile(sp_in, 99.5)), 1e-6), 0, 1) ** (1 / 2.2)
        data = png_bytes(to8(disp))
        try:
            _atomic_write_bytes(thumb_file, data)
        except Exception:  # noqa: BLE001
            pass
        return data

    def _roll_base_kwargs(self, rel: str, mode: str, softness: float = 0.5) -> dict:
        folder = str(Path(rel).parent)
        rinfo = self.roll_info(folder)
        fe = rinfo["frames"][rel]
        sub = rinfo["sub_rolls"][fe["sub_roll"]]
        rb = sub["roll"]
        if rb is None or fe["roll_rgb"] is None:
            raise ValueError(f"roll base unavailable for {rel}: {sub.get('error') or rinfo.get('error')}")
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
            sub_roll=fe["sub_roll"],
        )
        if fe.get("roll_implausible"):
            common["warnings"] = [f"整卷片基换算到本帧后高于传感器白点（{max(fe['roll_rgb']):.2f}），本卷的曝光归一化不成立"
                                  "（可能光圈或光源有变化），建议使用单帧片基。"]
        if fe.get("sub_roll_single"):
            return dict(base_regions=None, base_rgb=list(fe["own_rgb"]),
                        base_source=dict(common, effective_rgb=list(fe["own_rgb"]), single_frame_sub_roll=True))
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
        _atomic_write_bytes(self.orient_file, json.dumps(d, indent=1, sort_keys=True).encode("utf-8"))
        return dict(path=body["path"], rot90=k, source="saved")

    def user_presets(self) -> list[dict]:
        try:
            data = json.loads(self.user_presets_file.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, ValueError):
            return []

    def save_user_preset(self, body: dict) -> list[dict]:
        name = str(body.get("name", "")).strip()
        if not name or len(name) > 48:
            raise ValueError("preset name must be 1..48 characters")
        entry = dict(
            name=name,
            engine=body.get("engine") or self.engine,
            params=body.get("params") or {},
            grade=DarkroomGrade.from_dict(body.get("grade")).to_dict(),
        )
        items = [x for x in self.user_presets() if x.get("name") != name]
        items.append(entry)
        _atomic_write_bytes(self.user_presets_file, json.dumps(items, ensure_ascii=False, indent=2).encode("utf-8"))
        return items

    def delete_user_preset(self, body: dict) -> list[dict]:
        name = str(body.get("name", "")).strip()
        items = [x for x in self.user_presets() if x.get("name") != name]
        _atomic_write_bytes(self.user_presets_file, json.dumps(items, ensure_ascii=False, indent=2).encode("utf-8"))
        return items

    def _active_engine(self, pv: Preview) -> str:
        cfg_id = (pv.auto or {}).get("config", "")
        if cfg_id == NOREF_T24["id"]:
            return "t24"
        if cfg_id == NOREF_T23["id"]:
            return "t23"
        return "v3"

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
        cam_m = self.camera_matrix(pv.path)
        eng = self._active_engine(pv)

        if eng in ("t24", "t23"):
            preset_name = self.preset if self.preset in T24_STYLE_PRESETS else "standard"
            presets_list = [dict(id=k, label=v["label"], desc=v["desc"]) for k, v in T24_STYLE_PRESETS.items()]
            raw_auto_params = pv.auto["params"] if pv.auto else (RenderParamsT24().to_dict() if eng == "t24" else RenderParamsT19().to_dict())
            eff_params, auto_grade = _apply_t24_preset(raw_auto_params, preset_name, pv.film, eng, raw_matrix=self._raw_meta(pv.path).get("rgb_xyz_matrix"), scene_metrics=_scene_metrics_for_preview(pv))
            auto_block = dict(
                params=eff_params,
                base_params=raw_auto_params,
                status=pv.auto.get("status"),
                flags=pv.auto.get("flags"),
                config=pv.auto.get("config"),
                version=pv.auto.get("version"),
                branch=pv.auto.get("branch"),
                config_rev=pv.auto.get("config_rev"),
                diagnostics=pv.auto.get("diagnostics"),
            ) if pv.auto else None
            defaults_dict = (RenderParamsT24() if eng == "t24" else RenderParamsT19()).to_dict()
        else:
            preset_name = pv.auto.get("preset") if pv.auto else None
            presets_list = [dict(id=k, label=v["label"]) for k, v in STYLE_PRESETS.items()]
            auto_grade = {}
            auto_block = dict(
                params=pv.auto["params"], status=pv.auto.get("status"), flags=pv.auto.get("flags"),
                config=pv.auto.get("config"), config_rev=pv.auto.get("config_rev"),
                diagnostics=pv.auto.get("diagnostics"),
            ) if pv.auto else None
            defaults_dict = RenderParams().to_dict(full=True)

        return dict(
            token=self.token, path=rel, film=pv.film, factor=pv.factor, rot90=rot, rot90_source=rot_src,
            engine=eng, engines=list(ENGINE_LABELS),
            camera_matrix=[list(r) for r in cam_m] if cam_m else None,
            camera_matrix_mix=(pv.load_kwargs.get("auto") or NOREF_V3).get("camera_matrix_mix", CAMERA_MATRIX_MIX),
            preset=preset_name,
            presets=presets_list,
            auto_grade=auto_grade,
            hsl_bands=list(HSL_BANDS),
            white_point=dict(pv.report.get("white_point") or dict(method="pixel", pct=pv.white_pct),
                             d_hi=list(pv.inversion["d_hi"])),
            preview_shape=list(pv.positive.shape[:2]), full_shape=list(pv.full_shape),
            film_box=film_box(pv.holder) if pv.holder is not None else None,
            input_warnings=pv.report.get("input_warnings", []),
            base=dict(
                method=method,
                rgb=pv.base_info.get("rgb_render"),
                warnings=est.get("warnings", []) or src.get("roll_warnings", []) + src.get("warnings", []),
                inside_image=pv.base_inside_image,
                regions_preview=[pv.to_preview(r) for r in (pv.load_kwargs.get("base_regions") or [])],
                chroma_part=src.get("chroma_part", f_roll.get("chroma_part")),
                level_diff=src.get("level_diff", f_roll.get("level_diff")),
                d_lo_shared=src.get("d_lo_shared"),
                softness=src.get("softness", 0.5),
                roll_rgb=src.get("roll_rgb", f_roll.get("roll_rgb")),
                own_rgb=src.get("own_rgb", f_roll.get("own_rgb")),
            ),
            auto=auto_block,
            image_region=list(pv.image_region), region_source=pv.region_source,
            sliders=all_bounds, t24_bounds={k: list(v) for k, v in T24_BOUNDS.items()},
            curve_presets=CURVE_PRESETS,
            defaults=defaults_dict,
        )

    def open(self, rel: str, film: str, base_mode: str | None = None, softness: float = 0.5,
             preset: str | None = None, engine: str | None = None) -> dict:
        if film not in FILM_MODES:
            raise ValueError(f"film must be one of {FILM_MODES}")
        if engine is not None:
            if engine not in ENGINES:
                raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
            self.engine = engine
        eng = self.engine
        if preset is not None:
            if eng in ("t24", "t23"):
                if preset not in T24_STYLE_PRESETS:
                    raise ValueError(f"unknown style preset {preset!r}; one of {sorted(T24_STYLE_PRESETS)}")
            else:
                preset_config(preset)
            self.preset = preset
        elif eng == "v3" and self.preset not in STYLE_PRESETS:
            self.preset = "standard"

        p = self.resolve(rel)
        folder = str(Path(rel).parent)
        if base_mode is None:
            base_mode = "roll_nat" if self._route_roll_nat(folder, rel, film=film) else "auto"
        if base_mode not in BASE_MODES:
            raise ValueError(f"base_mode must be one of {BASE_MODES}")
        self.preview = None
        self.token = None
        extra = self._roll_base_kwargs(rel, base_mode, softness) if base_mode != "auto" else {}
        ck = self._colour_kwargs(p, film, self.preset, eng)
        wk = dict(white_block=WHITE_BLOCK, white_pct=WHITE_PCT, white_b_guard=WHITE_B_GUARD,
                  white_self_ratio=WHITE_SELF_RATIO, spike_clamp=SPIKE_CLAMP)
        try:
            self.preview = load_preview(p, max_side=self.max_side, film=film, keep_full=True, **wk, **ck, **extra)
        except ValueError:
            if base_mode == "auto":
                extra = self._roll_base_kwargs(rel, "roll_nat", softness)
                self.preview = load_preview(p, max_side=self.max_side, film=film, keep_full=True, **wk, **ck, **extra)
            else:
                raise
        self.token = secrets.token_hex(8)
        self.negative = None
        return self.meta()

    def _route_roll_nat(self, folder: str, rel: str, film: str = "color") -> bool:
        rinfo = self.roll_cache.get(folder) or {}
        fe = rinfo.get("frames", {}).get(rel)
        sub = rinfo.get("sub_rolls", {}).get((fe or {}).get("sub_roll"))
        if not fe or not sub or sub["roll"] is None or fe.get("roll_rgb") is None:
            return False
        if sub["n_base"] < ROLL_NAT_MIN_FRAMES or sub["roll"]["warnings"]:
            return False
        if fe.get("roll_implausible"):
            return False
        diag = ((fe.get("estimate") or {}).get("diagnostics")) or {}
        fallback = bool(diag.get("mode_seed") or diag.get("relaxed_uniformity") or diag.get("bare_backlight_excluded"))
        dc, lv = fe.get("chroma_part"), fe.get("level_diff")
        lvl_thr = ROLL_NAT_BW_LEVEL_DIFF if film == "bw" else ROLL_NAT_LEVEL_DIFF
        return bool(fe.get("own_error") or fallback or (dc is not None and dc >= 0.1)
                    or (lvl_thr is not None and lv is not None and lv >= lvl_thr))

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

    def set_engine(self, body: dict) -> dict:
        """Switch auto-estimation engine (t24 / t23 / v3) on the kept full-resolution positive (no RAW decode)."""
        pv = self.need(body.get("token"))
        eng = body.get("engine")
        if eng not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}, got {eng!r}")
        preset = body.get("preset") or self.preset
        if eng == "v3" and preset not in STYLE_PRESETS:
            preset = "standard"
        elif eng in ("t24", "t23") and preset not in T24_STYLE_PRESETS:
            preset = "standard"
        ck = self._colour_kwargs(pv.path, pv.film, preset, eng)
        self.preview = pv.with_auto(ck["auto"])
        self.engine = eng
        self.preset = preset
        return self.meta()

    def set_preset(self, body: dict) -> dict:
        pv = self.need(body.get("token"))
        name = body.get("preset")
        eng = self._active_engine(pv)
        if eng in ("t24", "t23"):
            if name not in T24_STYLE_PRESETS:
                raise ValueError(f"unknown style preset {name!r}; one of {sorted(T24_STYLE_PRESETS)}")
            self.preset = name
            return self.meta()
        ck = self._colour_kwargs(pv.path, pv.film, name, "v3")
        self.preview = pv.with_auto(ck["auto"])
        self.preset = name
        return self.meta()

    def wb_pick(self, body: dict) -> dict:
        """Compute (temp, tint) that neutralises a 5x5 patch around preview pixel (x, y) on the open positive."""
        pv = self.need(body.get("token"))
        eng = self._active_engine(pv)
        rp = coerce_render_params(body.get("params")) if body.get("params") else None
        is_t19 = isinstance(rp, T19_FAMILY) or (rp is None and eng in ("t24", "t23"))

        pos = pv._t18_pos(pv.positive) if is_t19 else pv.positive
        if body.get("geom"):
            gm = Geometry.from_dict(body["geom"])
            if not gm.is_identity():
                pos = apply_geometry(pos, gm, apply_rot90=True, apply_crop=bool(body.get("crop_apply", True)),
                                     fast_preview=False)

        x, y = int(round(float(body["x"]))), int(round(float(body["y"])))
        h, w = pos.shape[:2]
        x, y = max(0, min(x, w - 1)), max(0, min(y, h - 1))
        patch = pos[max(0, y - 2):min(h, y + 3), max(0, x - 2):min(w, x + 3)].reshape(-1, 3)
        valid = patch[np.all(patch > 1e-4, axis=-1)]
        if len(valid) == 0:
            raise ValueError("selected area is too dark or clipped to sample white balance")
        med = np.median(valid, axis=0)
        if isinstance(rp, RenderParamsT24) and rp.eps_rgb != (0.0, 0.0, 0.0):
            med = np.maximum(med + np.asarray(rp.eps_rgb, dtype=np.float64), 1e-5)

        temp, tint = wb_from_chroma(np.array([np.log2(med[0] / med[1]), np.log2(med[2] / med[1])]))
        bounds = T24_BOUNDS if is_t19 else BOUNDS
        lo_t, hi_t = bounds["temp"]
        lo_i, hi_i = bounds["tint"]

        if rp is not None and pv.film != "bw":
            if isinstance(rp, T19_FAMILY):
                if rp.sat > 0.0:
                    temp, tint = _solve_wb_newton_t19(med.astype(np.float32), rp, temp, tint, bounds)
            elif rp.sat > 0.0:
                temp, tint = _solve_wb_newton(med.astype(np.float32), rp, temp, tint)
        return dict(
            temp=float(np.clip(temp, lo_t, hi_t)),
            tint=float(np.clip(tint, lo_i, hi_i)),
            sampled_rgb=[float(v) for v in med],
        )

    def _export_with_preview(self, pv: Preview, body: dict) -> dict:
        rel = pv.path.relative_to(self.root)
        out_dir = self.out / rel.parent
        fmt = str(body.get("format", "tiff16")).lower()
        encoding = str(body.get("encoding", "srgb")).lower()
        quality = int(body.get("quality", 95))
        max_edge = body.get("max_edge")
        max_edge = int(max_edge) if max_edge not in (None, "", 0, "0") else None
        write_sidecar = bool(body.get("write_sidecar", True))

        grade = DarkroomGrade.from_dict(body.get("grade"))
        raw_geom = body.get("geometry")
        if raw_geom:
            geom = Geometry.from_dict(raw_geom)
            if "rot90" not in raw_geom and "rot90" in body:
                geom = replace(geom, rot90=int(body["rot90"]) % 4)
        else:
            geom = Geometry(rot90=int(body.get("rot90", 0)))

        raw_params = body.get("params") or {}
        rp = coerce_render_params(raw_params)

        src = dict(
            preset=(body.get("preset") or self.preset) if self._active_engine(pv) in ("t24", "t23") else (pv.auto.get("preset") if pv.auto else None),
            engine=self._active_engine(pv),
            auto_config=pv.auto.get("config") if pv.auto else None,
            auto_rev=pv.auto.get("config_rev") if pv.auto else None,
            note="style preset of the automatic starting point; the exported parameters are the page's",
        )

        # Preserve exact legacy fast-path for standard 16-bit TIFF without darkroom post-grade/crop/resize
        is_legacy_tiff = (
            fmt == "tiff16"
            and grade.is_identity()
            and not geom.has_non_rot90()
            and max_edge is None
            and write_sidecar
        )
        if is_legacy_tiff:
            sc = pv.export(
                out_dir, rp, encoding,
                stem=free_stem(out_dir, pv.path.stem), overwrite=False, rot90=geom.rot90,
                render_source=src if pv.auto else None,
            )
            return dict(
                path=sc["output"]["path"], sha256=sc["output"]["sha256"], rot90=sc["output"].get("rot90", 0),
                format="tiff16",
                sidecar=str(Path(sc["output"]["path"]).with_suffix(".json")),
                qc=dict(status=sc["qc"]["status"], failing=[c for c in sc["qc"]["checks"] if c["status"] != "pass"]),
            )

        # Enhanced full-res export: convert (always in sRGB for grading) -> grade -> geometry -> vignette -> encode/write
        if sha256(pv.path) != pv.sha256:
            raise SourceChangedError(f"{pv.path} changed since it was opened; open it again")
        base = pv.base_info["rgb_render"] if pv.film == "bw" else pv.inversion["base"]
        rp_dict = rp.to_dict() if isinstance(rp, (RenderParams, RenderParamsT19, RenderParamsT24)) else raw_params
        disp, sc = convert(
            pv.path, base_rgb=base, d_hi=pv.inversion["d_hi"], image_region=pv.image_region,
            white_pct=pv.white_pct, gamma=pv.inversion["gamma"], exposure=pv.inversion["exposure"],
            encoding="srgb", render_params=rp_dict, film=pv.film, base_source=pv.base_source(),
            white_point_source=pv.report.get("white_point"), rot90=0,
            render_source=src if pv.auto else None, spike_clamp=pv.inversion.get("spike_clamp"),
            t18_inversion=pv.t18_inversion() if isinstance(rp, T19_FAMILY) else None,
        )
        if not grade.is_identity():
            disp = apply_darkroom_grade(disp, grade, film=pv.film, apply_vignette_here=False, fast=False)
        if not geom.is_identity():
            disp = apply_geometry(disp, geom, apply_rot90=True, apply_crop=True, fast_preview=False)
        if grade.vignette:
            disp = apply_vignette(disp, grade.vignette)

        out_img = disp if (fmt != "tiff16" or encoding == "srgb") else srgb_eotf(disp).astype(np.float32)
        ext = ".tif" if fmt == "tiff16" else ".jpg" if fmt == "jpeg" else ".png"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = free_stem_ext(out_dir, pv.path.stem, ext, check_json=write_sidecar)
        out_path = out_dir / f"{stem}{ext}"
        export_image(out_path, out_img, fmt=fmt, encoding=encoding, quality=quality, max_edge=max_edge)

        file_hash = sha256(out_path)
        sc["output"].update(
            path=str(out_path), sha256=file_hash, format=fmt,
            encoding=encoding if fmt == "tiff16" else "srgb",
            bit_depth=16 if fmt == "tiff16" else 8,
            shape=([max(1, int(round(out_img.shape[0] * (max_edge / max(out_img.shape[:2]))))), max(1, int(round(out_img.shape[1] * (max_edge / max(out_img.shape[:2])))))] if (max_edge and max(out_img.shape[:2]) > max_edge) else list(out_img.shape[:2])),
        )
        if geom.rot90:
            sc["output"]["rot90"] = geom.rot90
        if not grade.is_identity():
            sc["render"]["darkroom_grade"] = grade.to_dict()
        if not geom.is_identity():
            sc["output"]["geometry"] = geom.to_dict()
        if max_edge:
            sc["output"]["max_edge"] = max_edge
        sidecar_path = None
        if write_sidecar:
            js = out_dir / f"{stem}.json"
            js.write_text(json.dumps(sc, indent=1, ensure_ascii=False), encoding="utf-8")
            sidecar_path = str(js)
        return dict(
            path=str(out_path), sha256=file_hash, rot90=geom.rot90, format=fmt,
            sidecar=sidecar_path,
            qc=dict(status=sc["qc"]["status"], failing=[c for c in sc["qc"]["checks"] if c["status"] != "pass"]),
        )

    def export(self, body: dict) -> dict:
        pv = self.need(body.get("token"))
        return self._export_with_preview(pv, body)

    def start_batch_export(self, body: dict) -> dict:
        items = body.get("items") or []
        if not items:
            raise ValueError("batch export requires a non-empty 'items' list")
        # Validate all paths upfront before spawning worker
        for it in items:
            self.resolve(it["path"])
        job_id = secrets.token_hex(6)
        fmt = str(body.get("format", "jpeg")).lower()
        encoding = str(body.get("encoding", "srgb")).lower()
        quality = int(body.get("quality", 95))
        max_edge = body.get("max_edge")
        write_sidecar = bool(body.get("write_sidecar", fmt == "tiff16"))

        job = dict(job_id=job_id, status="running", total=len(items), done=0, current=None, results=[], errors=[])
        with self._batch_lock:
            self.batch_jobs[job_id] = job

        def worker():
            # Strictly sequential (1 full-res frame at a time) to respect local 18 GiB RAM ceiling
            for it in items:
                rel = it["path"]
                with self._batch_lock:
                    job["current"] = rel
                try:
                    p = self.resolve(rel)
                    film = it.get("film", "color")
                    eng = it.get("engine") or self.engine
                    preset = it.get("preset", "standard")
                    bm = it.get("base_mode")
                    softness = float(it.get("softness", 0.5))
                    folder = str(Path(rel).parent)
                    if bm is None:
                        bm = "roll_nat" if self._route_roll_nat(folder, rel, film=film) else "auto"
                    extra = self._roll_base_kwargs(rel, bm, softness) if bm != "auto" else {}
                    ck = self._colour_kwargs(p, film, preset, eng)
                    wk = dict(white_block=WHITE_BLOCK, white_pct=WHITE_PCT, white_b_guard=WHITE_B_GUARD,
                              white_self_ratio=WHITE_SELF_RATIO, spike_clamp=SPIKE_CLAMP)
                    pv = load_preview(p, max_side=256, film=film, keep_full=False, **wk, **ck, **extra)
                    params = it.get("params")
                    grade = it.get("grade")
                    if params is None:
                        if eng in ("t24", "t23"):
                            params, p_grade = _apply_t24_preset(pv.auto["params"], preset, film, eng, raw_matrix=self._raw_meta(p).get("rgb_xyz_matrix"), scene_metrics=_scene_metrics_for_preview(pv))
                            if grade is None:
                                grade = p_grade
                        else:
                            params = pv.auto["params"]
                    rot_k = it.get("rot90")
                    if rot_k is None:
                        rot_k = self.orientation(p)[0]
                    res = self._export_with_preview(pv, dict(
                        params=params, grade=grade, geometry=it.get("geometry"), rot90=rot_k,
                        preset=preset, format=fmt, encoding=encoding, quality=quality,
                        max_edge=max_edge, write_sidecar=write_sidecar,
                    ))
                    del pv
                    with self._batch_lock:
                        job["results"].append(dict(source=rel, path=res["path"], format=res["format"]))
                except Exception as e:  # noqa: BLE001
                    with self._batch_lock:
                        job["errors"].append(dict(source=rel, error=f"{type(e).__name__}: {e}"))
                finally:
                    with self._batch_lock:
                        job["done"] += 1
            with self._batch_lock:
                job["current"] = None
                job["status"] = "done"

        threading.Thread(target=worker, daemon=True).start()
        return dict(job_id=job_id, total=len(items), status="running")

    def batch_status(self, job_id: str) -> dict:
        with self._batch_lock:
            job = self.batch_jobs.get(job_id)
            if job is None:
                raise ValueError(f"unknown batch job {job_id!r}")
            return json.loads(json.dumps(job))


def params_of(q: dict):
    return coerce_render_params(json.loads(q["p"][0])) if "p" in q else RenderParams()


def grade_of(q: dict) -> DarkroomGrade:
    return DarkroomGrade.from_dict(json.loads(q["g"][0])) if "g" in q else DarkroomGrade()


def geom_of(q: dict) -> Geometry | None:
    return Geometry.from_dict(json.loads(q["geom"][0])) if "geom" in q else None


class Handler(BaseHTTPRequestHandler):
    state: State

    def log_message(self, fmt, *args):
        pass

    def send(self, code: int, body: bytes, ctype: str, headers: dict | None = None):
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def json(self, obj, code: int = 200):
        self.send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def fail(self, e: Exception):
        code = (403 if isinstance(e, (PermissionError, FrozenDataError))
                else 409 if isinstance(e, (StaleError, SourceChangedError, FileExistsError)) else 400)
        self.json(dict(error=f"{type(e).__name__}: {e}"), code)

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        data = json.loads(self.rfile.read(n) or b"{}")
        if not isinstance(data, dict):
            raise ValueError("JSON request body must be an object")
        return data

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        st = self.state
        try:
            if u.path in ("/", "/index.html"):
                return self.send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path.startswith("/static/"):
                name = u.path[len("/static/"):]
                if "/" in name or "\\" in name or ".." in name:
                    return self.json(dict(error="forbidden"), 403)
                fp = STATIC / name
                if fp.is_file() and fp.suffix in STATIC_MIME:
                    return self.send(200, fp.read_bytes(), STATIC_MIME[fp.suffix])
                return self.json(dict(error="not found"), 404)
            if u.path == "/api/files":
                return self.json(dict(root=str(st.root), out=str(st.out), default_engine=st.engine, files=st.files()))
            if u.path == "/api/roll":
                folder = q.get("folder", ["."])[0]
                return self.json(st.roll_info(folder))
            if u.path == "/api/thumb":
                rel = q["path"][0]
                film = q.get("film", ["color"])[0]
                return self.send(200, st.thumbnail_png(rel, film), "image/png")
            if u.path == "/api/user_presets":
                return self.json(dict(presets=st.user_presets()))
            if u.path == "/api/batch_export":
                return self.json(st.batch_status(q["job_id"][0]))
            with st.lock:
                token = q.get("token", [None])[0]
                if u.path == "/api/render":
                    pv = st.need(token)
                    rp = params_of(q)
                    gr = grade_of(q)
                    gm = geom_of(q)
                    apply_crop = q.get("crop_apply", ["1"])[0] != "0"
                    img_fmt = q.get("fmt", ["png"])[0]
                    disp, stats = pv.render_stats(rp)
                    if not gr.is_identity():
                        disp = apply_darkroom_grade(disp, gr, film=pv.film, apply_vignette_here=False, fast=True)
                    if gm is not None and not gm.is_identity():
                        disp = apply_geometry(disp, gm, apply_rot90=True, apply_crop=apply_crop, fast_preview=True)
                    if gr.vignette:
                        disp = apply_vignette(disp, gr.vignette)
                    u8 = to8(disp)
                    if img_fmt == "jpeg":
                        payload, ctype = jpeg_bytes(u8, quality=92), "image/jpeg"
                    else:
                        payload, ctype = png_bytes(u8), "image/png"
                    return self.send(200, payload, ctype,
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
                    crop_img = pv.render_crop(region, params_of(q))
                    gr = grade_of(q)
                    if not gr.is_identity():
                        crop_img = apply_darkroom_grade(crop_img, gr, film=pv.film, apply_vignette_here=False, fast=True)
                    return self.send(200, png_bytes(to8(crop_img)), "image/png")
            self.json(dict(error="not found"), 404)
        except Exception as e:  # noqa: BLE001
            self.fail(e)

    def do_POST(self):
        u = urlparse(self.path)
        st = self.state
        try:
            b = self.body()
            if u.path == "/api/user_presets":
                return self.json(dict(presets=st.save_user_preset(b)))
            if u.path == "/api/user_presets/delete":
                return self.json(dict(presets=st.delete_user_preset(b)))
            if u.path == "/api/batch_export":
                return self.json(st.start_batch_export(b))
            if u.path == "/api/export":
                with st.lock:
                    pv_exp = st.need(b.get("token"))
                    b_exp = dict(b)
                    b_exp.setdefault("preset", st.preset)
                return self.json(st._export_with_preview(pv_exp, b_exp))
            with st.lock:
                if u.path == "/api/open":
                    return self.json(st.open(b["path"], b.get("film", "color"), b.get("base_mode"),
                                             float(b.get("softness", 0.5)), b.get("preset"), b.get("engine")))
                if u.path == "/api/engine":
                    return self.json(st.set_engine(b))
                if u.path == "/api/preset":
                    return self.json(st.set_preset(b))
                if u.path == "/api/base":
                    return self.json(st.set_base(b))
                if u.path == "/api/wb_pick":
                    return self.json(st.wb_pick(b))
                if u.path == "/api/orient":
                    return self.json(st.set_orientation(b))
            self.json(dict(error="not found"), 404)
        except Exception as e:  # noqa: BLE001
            self.fail(e)


def _t19_pre_curve_pixel(med32: np.ndarray, rp: RenderParamsT19 | RenderParamsT24, temp: float, tint: float) -> np.ndarray:
    p = replace(rp, temp=temp, tint=tint)
    if isinstance(p, RenderParamsT24):
        m_lin = _t24_linear_matrix(p)
        out = (med32 @ m_lin.T) if m_lin is not None else med32.copy()
        if p.slope_r or p.slope_b:
            out = np.maximum(out, np.float32(1e-6))
            for ch, s in ((0, p.slope_r), (2, p.slope_b)):
                if s:
                    out[ch] = np.float32(MID_GREY) * np.exp2(np.float32(1.0 + s) * np.log2(out[ch] / np.float32(MID_GREY)))
        return out
    from .render import _fused_linear_matrix
    m_lin = _fused_linear_matrix(p.ev, p.temp, p.tint, p.matrix, p.sat)
    return (med32 @ m_lin.T) if m_lin is not None else med32.copy()


def _solve_wb_newton_t19(med32: np.ndarray, rp: RenderParamsT19 | RenderParamsT24,
                         temp: float, tint: float, bounds: dict) -> tuple[float, float]:
    lo_t, hi_t = bounds["temp"]
    lo_i, hi_i = bounds["tint"]
    temp = float(np.clip(temp, lo_t, hi_t))
    tint = float(np.clip(tint, lo_i, hi_i))
    for _ in range(6):
        out0 = _t19_pre_curve_pixel(med32, rp, temp, tint)
        if np.any(out0 <= 1e-6):
            break
        f0 = np.array([np.log2(float(out0[0]) / float(out0[1])), np.log2(float(out0[2]) / float(out0[1]))])
        if float(np.max(np.abs(f0))) < 1e-5:
            break
        eps = 1e-3
        t_p = min(hi_t, temp + eps) if temp + eps <= hi_t else temp - eps
        i_p = min(hi_i, tint + eps) if tint + eps <= hi_i else tint - eps
        out_t = _t19_pre_curve_pixel(med32, rp, t_p, tint)
        out_i = _t19_pre_curve_pixel(med32, rp, temp, i_p)
        if np.any(out_t <= 1e-6) or np.any(out_i <= 1e-6):
            break
        ft = np.array([np.log2(float(out_t[0]) / float(out_t[1])), np.log2(float(out_t[2]) / float(out_t[1]))])
        fi = np.array([np.log2(float(out_i[0]) / float(out_i[1])), np.log2(float(out_i[2]) / float(out_i[1]))])
        jac = np.column_stack([(ft - f0) / (t_p - temp), (fi - f0) / (i_p - tint)])
        if abs(float(np.linalg.det(jac))) < 1e-6:
            break
        step = np.linalg.solve(jac, -f0)
        nt = float(np.clip(temp + step[0], lo_t, hi_t))
        ni = float(np.clip(tint + step[1], lo_i, hi_i))
        if abs(nt - temp) < 1e-6 and abs(ni - tint) < 1e-6:
            return nt, ni
        temp, tint = nt, ni
    return temp, tint


def _solve_wb_newton(med32: np.ndarray, rp: RenderParams, temp: float, tint: float) -> tuple[float, float]:
    lo_t, hi_t = BOUNDS["temp"]
    lo_i, hi_i = BOUNDS["tint"]
    x0, _ = pre_shoulder_linear(med32, RenderParams(ev=rp.ev, log_unmix=rp.log_unmix))
    if np.all(x0 > 1e-6):
        vg = float(x0[1])
        vr_req = MID_GREY * ((vg / MID_GREY) ** (2.0 ** (0.65 * rp.slope_r))) if rp.slope_r else vg
        vb_req = MID_GREY * ((vg / MID_GREY) ** (2.0 ** (0.65 * rp.slope_b))) if rp.slope_b else vg
        temp, tint = wb_from_chroma(np.array([np.log2(x0[0] / max(vr_req, 1e-6)), np.log2(x0[2] / max(vb_req, 1e-6))]))
    temp = float(np.clip(temp, lo_t, hi_t))
    tint = float(np.clip(tint, lo_i, hi_i))
    for _ in range(5):
        out0, _ = pre_shoulder_linear(med32, replace(rp, temp=temp, tint=tint))
        if np.any(out0 <= 1e-6):
            break
        f0 = np.array([np.log2(float(out0[0]) / float(out0[1])), np.log2(float(out0[2]) / float(out0[1]))])
        if float(np.max(np.abs(f0))) < 1e-5:
            break
        eps = 1e-3
        t_p = min(hi_t, temp + eps) if temp + eps <= hi_t else temp - eps
        i_p = min(hi_i, tint + eps) if tint + eps <= hi_i else tint - eps
        out_t, _ = pre_shoulder_linear(med32, replace(rp, temp=t_p, tint=tint))
        out_i, _ = pre_shoulder_linear(med32, replace(rp, temp=temp, tint=i_p))
        if np.any(out_t <= 1e-6) or np.any(out_i <= 1e-6):
            break
        ft = np.array([np.log2(float(out_t[0]) / float(out_t[1])), np.log2(float(out_t[2]) / float(out_t[1]))])
        fi = np.array([np.log2(float(out_i[0]) / float(out_i[1])), np.log2(float(out_i[2]) / float(out_i[1]))])
        jac = np.column_stack([(ft - f0) / (t_p - temp), (fi - f0) / (i_p - tint)])
        if abs(float(np.linalg.det(jac))) < 1e-6:
            break
        step = np.linalg.solve(jac, -f0)
        nt = float(np.clip(temp + step[0], lo_t, hi_t))
        ni = float(np.clip(tint + step[1], lo_i, hi_i))
        if abs(nt - temp) < 1e-6 and abs(ni - tint) < 1e-6:
            return nt, ni
        temp, tint = nt, ni
    return temp, tint


class _ReuseThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(root, out, port: int = 8765, allow_frozen: bool = False, engine: str = "v3") -> ThreadingHTTPServer:
    """Bind on 127.0.0.1:port (0 = any free port) and return the server; call serve_forever() on it."""
    handler = type("BoundHandler", (Handler,), dict(state=State(Path(root), Path(out), allow_frozen, engine=engine)))
    return _ReuseThreadingHTTPServer(("127.0.0.1", port), handler)


def main(argv=None):
    ap = argparse.ArgumentParser(description="FilmColor Darkroom local studio")
    ap.add_argument("--root", required=True, help="folder with the negatives to open")
    ap.add_argument("--out", default="filmcolor_export", help="export folder (default ./filmcolor_export)")
    ap.add_argument("--port", type=int, default=None, help="port on 127.0.0.1 (default: try 8765, else any free port)")
    ap.add_argument("--engine", default="t24", choices=ENGINES, help="default auto engine (default: t24)")
    ap.add_argument("--allow-frozen", action="store_true", help="also open files of the sealed test sets")
    a = ap.parse_args(argv)
    if a.port is not None:
        srv = serve(a.root, a.out, a.port, a.allow_frozen, engine=a.engine)
    else:
        try:
            srv = serve(a.root, a.out, 8765, a.allow_frozen, engine=a.engine)
        except OSError:
            srv = serve(a.root, a.out, 0, a.allow_frozen, engine=a.engine)
    print(f"FilmColor Darkroom: http://127.0.0.1:{srv.server_address[1]}/  (engine {a.engine}, root {Path(a.root).resolve()}, export to "
          f"{Path(a.out).resolve()})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
