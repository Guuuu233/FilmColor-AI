"""End-to-end conversion of one RAW negative, with a sidecar that records everything needed to
reproduce the output (inputs, software versions, every estimated and given parameter, diagnostics).

Replay: pass ``base_rgb`` and ``d_hi`` from a sidecar (plus its image region, gamma, exposure and
encoding) to skip estimation and get the same pixels.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import rawpy
import tifffile

from . import __version__
from .base import estimate_base_auto, estimate_base_manual
from .scanner import estimate_base_scanner, is_scanner_dng, load_scanner
from .export import DESCRIPTION, write_tiff16
from .invert import base_from_regions, central_region, invert_negative, invert_t18_from_positive, neutral_transmission, region_mask
from .qc import quality_checks
from .raw import check_region, linearize, saturation_mask
from .noref import (CONFIG_DIR, NOREF_T20, NOREF_T22, NOREF_T23, NOREF_T24, NOREF_V2, estimate_params_t20,
                    estimate_params_t24, estimate_params_v2, estimation_input)
from .render import IDENTITY, RenderParams, RenderParamsT19, RenderParamsT24, ZERO_MATRIX, render as render_image, render_t19, render_t24, srgb_eotf
from .rawpy_io import open_raw_session

SCHEMA = "filmcolor.sidecar/0.2"  # 0.2: render section (Sprint 2 parameters)
FILM_MODES = ("color", "bw")
BW_NEUTRAL = dict(temp=0.0, tint=0.0, sat=1.0, matrix=IDENTITY)  # look parts not applied to B&W film
BW_EXTENDED_ZERO = ("vibrance", "slope_r", "slope_b", "skin_protect", "log_unmix")
T19_FAMILY = (RenderParamsT19, RenderParamsT24)  # parameters defined on the T18/T23 re-inverted positive
_T18_VERSION_BY_CONFIG = {c["id"]: c["t18_version"] for c in (NOREF_T20, NOREF_T22, NOREF_T23, NOREF_T24)}
T18_VERSIONS = ("t18", "t23")


def t18_spec(auto) -> dict:
    """How the positive is re-inverted before a T19-family render: slope-ratio version and camera-matrix regime of
    noref config ``auto`` (no / legacy config: T18 slopes, no camera matrix)."""
    auto = auto if isinstance(auto, dict) else {}
    mat = auto.get("matrix", IDENTITY)
    return dict(version=str(auto.get("t18_version", "t18")),
                has_cam_matrix=bool(not np.allclose(np.asarray(mat, dtype=np.float64), np.eye(3))))


def _t18_spec_for_params(rp) -> dict:
    """Fallback for explicit T19-family parameters without a recorded spec (``--render`` files, pre-T24 sidecars)."""
    return dict(version="t23" if isinstance(rp, RenderParamsT24) else "t18", has_cam_matrix=bool(rp.matrix != IDENTITY))


def neutralise_bw_params(params: dict) -> tuple[dict, list[str]]:
    """Strip colour-only look parameters for ``film='bw'``, keeping tonal parameters intact."""
    out = dict(params, **BW_NEUTRAL)
    dropped = set(BW_NEUTRAL)
    for k in BW_EXTENDED_ZERO:
        if k in out:
            if float(out[k]) != 0.0:
                dropped.add(k)
            out[k] = 0.0
    tc = out.get("tone_curve")
    if isinstance(tc, dict) and any(ch in tc for ch in ("r", "g", "b")):
        out["tone_curve"] = {"rgb": tc["rgb"]} if "rgb" in tc else {}
        dropped.add("tone_curve")
    elif isinstance(tc, (list, tuple)) and tc and isinstance(tc[0], (list, tuple)) and len(tc[0]) == 2 \
            and isinstance(tc[0][0], str) and any(ch in ("r", "g", "b") for ch, _ in tc):
        out["tone_curve"] = tuple((ch, pts) for ch, pts in tc if ch == "rgb")
        dropped.add("tone_curve")
    if "eps_rgb" in out:
        if any(float(x) != 0.0 for x in out["eps_rgb"]):
            dropped.add("eps_rgb")
        out["eps_rgb"] = [0.0, 0.0, 0.0]
    if "residual_matrix" in out:
        if any(float(x) != 0.0 for row in out["residual_matrix"] for x in row):
            dropped.add("residual_matrix")
        out["residual_matrix"] = [list(r) for r in ZERO_MATRIX]
    return out, sorted(dropped)


def coerce_render_params(params) -> RenderParams | RenderParamsT19 | RenderParamsT24:
    if isinstance(params, (RenderParams, RenderParamsT19, RenderParamsT24)):
        return params
    if not params:
        return RenderParams()
    if isinstance(params, dict) and ("eps_rgb" in params or "residual_matrix" in params):
        return RenderParamsT24.from_dict(params)
    if isinstance(params, dict) and "toe" in params:
        return RenderParamsT19.from_dict(params)
    return RenderParams.from_dict(params)


def normalize_auto_cfg(auto) -> dict | None:
    if isinstance(auto, bool):
        return NOREF_V2 if auto else None
    if auto is not None and not isinstance(auto, dict):
        raise TypeError(f"auto must be a noref config dict, bool, or None, got {type(auto).__name__}")
    return auto


def run_auto_estimate(est_input, auto: dict, *, d_lo, d_hi, gamma: float, film: str = "color"):
    """Unified noref_v2 / noref_v3 / noref_t20 estimator returning ``(auto_info, render_params, is_t19)``."""
    pos4, mask4 = est_input
    if auto.get("id") in _T18_VERSION_BY_CONFIG:
        mat = auto.get("matrix", IDENTITY)
        spec = t18_spec(auto)
        pos4_t18 = invert_t18_from_positive(pos4, d_lo, d_hi, gamma, has_cam_matrix=spec["has_cam_matrix"],
                                            version=spec["version"])
        is_t24 = auto["id"] == NOREF_T24["id"]
        kw = dict(d_hi=d_hi, d_lo=d_lo, has_cam_matrix=spec["has_cam_matrix"], matrix=mat, P_v1=pos4)
        est = (estimate_params_t24(pos4_t18, mask4, auto, gamma=gamma, **kw) if is_t24
               else estimate_params_t20(pos4_t18, mask4, auto, **kw))
        if film == "bw":
            est_p, bw_dropped = neutralise_bw_params(est["params"])
            est = dict(est, params=est_p, bw_neutralised=bw_dropped)
        rp = (RenderParamsT24 if is_t24 else RenderParamsT19).from_dict(est["params"])
        w_file = CONFIG_DIR / f"{auto.get('weights_name', 't20_v3')}.json"
        ver_lbl = auto.get("version", "T22")
        info = dict(
            config=auto["id"],
            **({"version": auto["version"]} if "version" in auto else {}),
            status=est["status"], params=est["params"], branch=est["branch"], flags=est.get("flags", []),
            weights_file=dict(name=w_file.name, sha256=_cached_config_sha256(w_file)),
            input=f"{ver_lbl} decoupled positive, 4x4 block mean, image region",
        )
        return info, rp, True
    span_d_g = float(d_hi[1] - d_lo[1])
    est = estimate_params_v2(pos4, mask4, auto, span_d_g=span_d_g)
    if film == "bw":
        bw_params, bw_dropped = neutralise_bw_params(est["params"])
        est = dict(est, params=bw_params, bw_neutralised=bw_dropped)
    rp = RenderParams.from_dict(est["params"])
    look_file = CONFIG_DIR / f"{est['look']}.json"
    info = dict(est, look_file=dict(name=look_file.name, sha256=_cached_config_sha256(look_file)),
                input="positive x inversion exposure, 4x4 block mean, image region")
    return info, rp, False


def sha256(path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


@functools.lru_cache(maxsize=8)
def _cached_config_sha256(path: Path) -> str:
    return sha256(path)


@functools.lru_cache(maxsize=1)
def _git_repo_info() -> tuple[str | None, bool | None]:
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "src"],
                                    capture_output=True, text=True, check=True, timeout=5).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def software_info() -> dict:
    commit, dirty = _git_repo_info()
    try:
        pkg_ver = importlib.metadata.version("filmcolor")
    except importlib.metadata.PackageNotFoundError:
        pkg_ver = __version__
    return dict(filmcolor=pkg_ver, git_commit=commit, src_dirty=dirty,
                rawpy=rawpy.__version__, libraw=".".join(map(str, rawpy.libraw_version)), numpy=np.__version__)


def _base_info_from(est):
    return dict(method=est.method, estimate=est.to_dict())


def _camera_input(path, *, base_regions, base_rgb, image_region):
    """Camera RAW: CFA-aware base estimate on the analysis path, demosaiced linear render path."""
    with open_raw_session(path) as (f, demosaic):
        if image_region is not None:  # fail before the slow render step
            check_region(tuple(int(x) for x in image_region), f.cfa.shape)
        sat = saturation_mask(f)

        if base_rgb is not None:
            base_info = dict(method="given", rgb_render=[float(x) for x in base_rgb])
            regions = None
        else:
            lin, col = linearize(f), f.colors
            if base_regions:
                est = estimate_base_manual(lin, col, sat, base_regions)
            else:
                est = estimate_base_auto(lin, col, sat, f.pattern.shape[0], crop=f.crop)
            regions = est.regions
            base_info = _base_info_from(est)
            del lin, col

        rgb = demosaic()
    if base_rgb is None:
        rb = base_from_regions(rgb, regions)  # base on the same signal that is inverted
        base_rgb = rb["rgb"]
        base_info["rgb_render"] = list(rb["rgb"])
        base_info["render_stats"] = rb
    input_info = dict(shape=list(f.cfa.shape), cfa_pattern=f.pattern.tolist(), black=f.black.tolist(),
                      white=f.white, crop=list(f.crop) if f.crop else None, meta=f.meta)
    decode_info = dict(analysis="CFA-aware linearisation (filmcolor.raw)",
                       render="rawpy postprocess, default demosaic, raw colour, linear, no WB, "
                              "no auto-bright, adjust_maximum_thr=0, highlight clip, no flip")
    return rgb, sat, base_rgb, base_info, regions, input_info, decode_info


def _scanner_input(path, *, base_regions, base_rgb, image_region):
    """Scanner DNG / 16-bit TIFF: analysis and render use the same signal (no CFA)."""
    fr = load_scanner(path)
    if image_region is not None:
        check_region(tuple(int(x) for x in image_region), fr.sat.shape)
    rgb, sat = fr.rgb, fr.sat
    if base_rgb is not None:
        base_info = dict(method="given", rgb_render=[float(x) for x in base_rgb])
        regions = None
    else:
        if base_regions:
            regions = [tuple(int(x) for x in r) for r in base_regions]
            rb = base_from_regions(rgb, regions)
            sat_frac = float(np.mean([sat[t:t + h, l:l + w].mean() for t, l, h, w in regions]))
            base_info = dict(method="manual", estimate=dict(regions=regions, sat_frac=sat_frac))
        else:
            est = estimate_base_scanner(fr)
            regions = est.regions
            base_info = _base_info_from(est)
            rb = base_from_regions(rgb, regions)
        base_rgb = rb["rgb"]
        base_info["rgb_render"] = list(rb["rgb"])
        base_info["render_stats"] = rb
    is_tiff = Path(path).suffix.lower() in (".tif", ".tiff")
    input_info = dict(kind=("scanner_tiff8" if fr.meta["bits"] == 8 else "scanner_tiff16") if is_tiff else "scanner_linear_dng",
                      shape=list(sat.shape), black=fr.black, white=fr.white, meta=fr.meta)
    decode_info = dict(analysis="same as render (no CFA)",
                       render="tifffile %s, LinearRaw main image, (v - black) / (white - black), "
                              "infrared layer excluded" % tifffile.__version__)
    if is_tiff:
        signal = ("cross-channel R-tail and pedestal reconstruction, gamma 2.2 assumed"
                  if fr.meta.get("prestretched_reconstruction") is not None
                  else f"TRC linearisation {fr.meta['tiff_eotf']}" if "tiff_eotf" in fr.meta
                  else "signal assumed linear; no TRC applied")
        decode_info["render"] = (f"tifffile {tifffile.__version__} with native LZW / PackBits / Deflate strip fallback, "
                                 f"{fr.meta['bits']}-bit scanner image, (v - black) / (white - black), infrared excluded; "
                                 f"{signal}; no ICC colour matrix transform")
    return rgb, sat, base_rgb, base_info, regions, input_info, decode_info


def positive(path, *, base_regions=None, base_rgb=None, image_region=None, d_hi=None,
             white_pct: float = 99.9, gamma: float = 0.6, exposure: float = 1.0, film: str = "color",
             white_block: int | None = None, white_ratio=None, white_b_guard=None, white_self_ratio=None,
             spike_clamp=None) -> dict:
    """The scene-linear positive of one negative before any render parameter: decode, film base, inversion,
    times the inversion exposure (``convert`` renders exactly this). Returns ``positive`` (H, W, 3 float32),
    ``mask`` (unsaturated pixels in the image region), ``image_region``, ``region_source``, ``inversion``
    (InversionResult without its positive array), ``base_info``, ``input_info``, ``decode_info`` and
    ``base_inside_image`` (share of auto base blocks inside the image region, or None). ``white_block``: automatic
    white point on block means of that size (``invert.block_white_point``); None = single pixels. ``white_ratio``: roll
    neutral density ratios for ``invert.neutral_ratio_white_point`` (None = not checked). ``white_b_guard``: colour
    film only, ``invert.blue_white_guard`` on the automatic white point (None = not checked). ``white_self_ratio``:
    colour film only, ``invert.self_ratio_white_point`` (T4l). ``spike_clamp``: ``invert.clamp_spikes`` (T4l).
    ``input_center_p99`` (T5a): per-channel p99 of the linear input (before base and inversion) in the central region,
    for ``preview.input_warnings``."""
    if film not in FILM_MODES:
        raise ValueError(f"film must be one of {FILM_MODES}, got {film!r}")
    if not 0 < white_pct <= 100 or gamma <= 0 or exposure <= 0:
        raise ValueError(f"invalid parameters: white_pct={white_pct} gamma={gamma} exposure={exposure}")
    path = Path(path)
    if image_region is not None and len(image_region) != 4:
        raise ValueError(f"image region must be (top, left, height, width), got {image_region}")
    load = _scanner_input if is_scanner_dng(path) else _camera_input
    rgb, sat, base_rgb, base_info, regions, input_info, decode_info = load(
        path, base_regions=base_regions, base_rgb=base_rgb, image_region=image_region)

    region_source = "manual" if image_region else "central_default"
    image_region = tuple(int(x) for x in (image_region or central_region(sat.shape)))
    mask = ~sat & region_mask(sat.shape, image_region)
    ct, cl, ch, cw = central_region(sat.shape)  # T5a: linear input of the central region, for preview.input_warnings
    input_p99 = [float(x) for x in np.percentile(rgb[ct:ct + ch, cl:cl + cw].reshape(-1, 3), 99, axis=0)]
    inv_base = base_rgb
    if film == "bw":
        rgb, inv_base = neutral_transmission(rgb, base_rgb), (1.0, 1.0, 1.0)
    res = invert_negative(rgb, inv_base, mask=mask, white_pct=white_pct, gamma=gamma,
                          exposure=exposure, d_hi=d_hi, white_block=white_block, white_ratio=white_ratio,
                          white_b_guard=white_b_guard if film == "color" else None,
                          white_self_ratio=white_self_ratio if film == "color" else None, spike_clamp=spike_clamp)
    del rgb
    inside = None
    if regions is not None and base_info["method"] == "auto":
        t, l, h, w = image_region
        centres = [(r + rh / 2, c + rw / 2) for r, c, rh, rw in regions]
        inside = float(np.mean([t <= y < t + h and l <= x < l + w for y, x in centres]))
    # positive x inversion exposure, once (x 1.0 is exact, so the default reuses the array)
    pos = res.positive if res.params.exposure == 1 else res.positive * np.float32(res.params.exposure)
    res.positive = None
    return dict(positive=pos, mask=mask, image_region=image_region, region_source=region_source, inversion=res,
                base_info=base_info, input_info=input_info, decode_info=decode_info, base_inside_image=inside,
                input_center_p99=input_p99)


def convert(path, *, base_regions=None, base_rgb=None, image_region=None, d_hi=None,
            white_pct: float = 99.9, gamma: float = 0.6, exposure: float = 1.0, encoding: str = "srgb",
            render_params=None, auto=None, film: str = "color", base_source: dict | None = None,
            white_block: int | None = None, white_point_source: dict | None = None, rot90: int = 0,
            render_source: dict | None = None, white_b_guard=None, white_self_ratio=None, spike_clamp=None,
            t18_inversion: dict | None = None):
    """Return (image in [0, 1] encoded per ``encoding``, sidecar dict without output fields).

    ``base_source``: for a ``base_rgb`` replayed from an earlier estimate (preview export), how that base was
    found (method, estimate / regions, base_inside_image); stored as ``base.source``, and its base_inside_image
    feeds the QC check that a given base would otherwise skip. Pixels do not depend on it.
    ``white_block``: automatic white point on block means (``positive``); ``white_b_guard``: colour film's
    ``invert.blue_white_guard`` on the automatic white point (``positive``; the app's); ``white_self_ratio`` and
    ``spike_clamp``: T4l, as ``positive`` (the clamp is kept in the inversion parameters and replayed). ``white_point_source``: for a ``d_hi``
    replayed from an earlier estimate (preview export), how it was estimated; stored as ``inversion.white_point``.
    ``rot90``: output turned by that many quarter turns counter-clockwise (numpy.rot90) after rendering; QC and all
    statistics are measured before the turn. 0 (default) keeps the sensor orientation and the previous sidecar.
    ``render_source``: where the given ``render_params`` started from (the app: style preset and estimator config of
    the automatic starting point); stored as ``render.source``. Pixels do not depend on it.

    ``render_params``: RenderParams or its dict (Sprint 2 colour parameters); None = defaults,
    which reproduce the Sprint 1 output. ``auto``: a reference-free estimator config (``noref.NOREF_V2``)
    that sets the render parameters from the frame itself; excludes ``render_params``.
    ``film``: "color" (default, unchanged behaviour) or "bw", given explicitly (no automatic
    detection): neutral inversion (``invert.neutral_transmission``), and with ``auto`` only the tonal
    parameters are taken; temp, tint, saturation and matrix stay neutral (the look was fitted on
    colour negatives). Explicit ``render_params`` are applied as given in either mode.

    T19-family parameters (``RenderParamsT19`` / ``RenderParamsT24``) are defined on the T18/T23 re-inverted
    positive, which is made with ``t18_inversion`` = ``{version, has_cam_matrix}`` (from the auto config, else the
    given dict, else a default from the parameters) and recorded as ``render.t18_inversion`` for replay."""
    if film not in FILM_MODES:
        raise ValueError(f"film must be one of {FILM_MODES}, got {film!r}")
    auto = normalize_auto_cfg(auto)
    if auto is not None and render_params is not None:
        raise ValueError("auto and render_params are mutually exclusive")
    rp = coerce_render_params(render_params)
    if encoding not in DESCRIPTION:
        raise ValueError(f"encoding must be one of {tuple(DESCRIPTION)}")
    if not 0 < white_pct <= 100 or gamma <= 0 or exposure <= 0:
        raise ValueError(f"invalid parameters: white_pct={white_pct} gamma={gamma} exposure={exposure}")
    path = Path(path)
    p = positive(path, base_regions=base_regions, base_rgb=base_rgb, image_region=image_region, d_hi=d_hi,
                 white_pct=white_pct, gamma=gamma, exposure=exposure, film=film, white_block=white_block,
                 white_b_guard=white_b_guard, white_self_ratio=white_self_ratio, spike_clamp=spike_clamp)
    pos, mask, image_region, res = p["positive"], p["mask"], p["image_region"], p["inversion"]
    base_info, input_info, decode_info = p["base_info"], p["input_info"], p["decode_info"]
    region_source, inside = p["region_source"], p["base_inside_image"]
    del p
    if base_source is not None:
        base_info = dict(base_info, source=base_source)
        if inside is None:
            inside = base_source.get("base_inside_image")
    auto_info, t18 = None, None
    if auto is not None:
        auto_info, rp, is_t19 = run_auto_estimate(
            estimation_input(pos, image_region), auto,
            d_lo=res.params.d_lo, d_hi=res.params.d_hi, gamma=res.params.gamma, film=film,
        )
        if is_t19:
            t18 = t18_spec(auto)
    elif isinstance(rp, T19_FAMILY):
        t18 = dict(t18_inversion) if t18_inversion else _t18_spec_for_params(rp)
        if t18.get("version") not in T18_VERSIONS or not isinstance(t18.get("has_cam_matrix"), bool):
            raise ValueError(f"invalid t18_inversion {t18_inversion!r}: need version in {T18_VERSIONS} and a bool has_cam_matrix")
    if t18 is not None:
        pos = invert_t18_from_positive(pos, res.params.d_lo, res.params.d_hi, res.params.gamma,
                                       has_cam_matrix=t18["has_cam_matrix"], version=t18["version"])
    render_fn = render_t24 if isinstance(rp, RenderParamsT24) else render_t19 if isinstance(rp, RenderParamsT19) else render_image
    display, render_diag = render_fn(pos, rp, mask)
    del pos
    qc = quality_checks(display, mask, res.report, base_inside_image=inside)
    out = display if encoding == "srgb" else srgb_eotf(display).astype(np.float32)
    k = int(rot90) % 4
    if k:
        out = np.ascontiguousarray(np.rot90(out, k))

    sidecar = dict(
        schema=SCHEMA,
        input=dict(path=str(path), sha256=sha256(path), **input_info),
        software=software_info(),
        decode=decode_info,
        base=base_info,
        image_region=dict(region=list(image_region), source=region_source),
        inversion=dict(params=res.params.to_dict(), white_pct=white_pct, d_hi_source="given" if d_hi else "auto",
                       **({"white_point": white_point_source} if white_point_source is not None else {})),
        render=dict(params=rp.to_dict(), diagnostics=render_diag, definition="docs/SPRINT2_PARAMS.md"),
        report=res.report,
        qc=qc,
        output=dict(encoding=encoding, colorimetric_calibration="none",
                    icc_description=DESCRIPTION[encoding], bit_depth=16,
                    orientation="sensor (meta.flip not applied)"),
    )
    if k:  # sidecars of unrotated outputs keep their previous field set
        sidecar["output"].update(rot90=k, orientation=f"sensor turned {k} x 90 degrees counter-clockwise "
                                                      "(numpy.rot90; meta.flip not applied)")
    if t18 is not None:  # only T19-family renders carry it; legacy sidecars keep their field set
        sidecar["render"]["t18_inversion"] = t18
    if auto_info is not None:
        sidecar["auto"] = auto_info
    if render_source is not None:
        sidecar["render"]["source"] = render_source
    if film == "bw":  # colour sidecars keep their previous field set
        sidecar["film"] = dict(mode="bw", source="given",
                               inversion="neutral_transmission: geometric mean of base-normalised channel "
                                         "transmittances, inverted with base (1, 1, 1); film base in base.rgb_render")
    return out, sidecar


def free_stem(out_dir, stem: str) -> str:
    """``stem``, or ``stem-2``, ``stem-3``, ... : the first name whose .tif and .json are both unused."""
    out_dir, name, n = Path(out_dir), stem, 1
    while (out_dir / f"{name}.tif").exists() or (out_dir / f"{name}.json").exists():
        n += 1
        name = f"{stem}-{n}"
    return name


def write(path, out_dir, *, stem: str | None = None, overwrite: bool = True, **kwargs) -> dict:
    """Convert and write <stem>.tif plus <stem>.json (default stem: the input's); return the sidecar.
    ``overwrite=False`` refuses (FileExistsError) when either file of the pair already exists."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or Path(path).stem
    tif, js = out_dir / f"{stem}.tif", out_dir / f"{stem}.json"
    if not overwrite and (tif.exists() or js.exists()):
        raise FileExistsError(f"{tif} or {js.name} already exists")
    img, sidecar = convert(path, **kwargs)
    write_tiff16(tif, img, sidecar["output"]["encoding"])
    sidecar["output"].update(path=str(tif), sha256=sha256(tif))
    js.write_text(json.dumps(sidecar, indent=1))
    return sidecar


def replay_kwargs(sidecar: dict) -> dict:
    """Arguments that reproduce a sidecar's output without re-estimating anything."""
    p = sidecar["inversion"]["params"]
    kw = dict(base_rgb=p["base"], d_hi=p["d_hi"], image_region=sidecar["image_region"]["region"],
              white_pct=sidecar["inversion"]["white_pct"], gamma=p["gamma"], exposure=p["exposure"],
              encoding=sidecar["output"]["encoding"], render_params=sidecar.get("render", {}).get("params"))
    if p.get("spike_clamp") is not None:  # T4l
        kw["spike_clamp"] = tuple(p["spike_clamp"])
    if sidecar.get("film", {}).get("mode") == "bw":
        kw.update(film="bw", base_rgb=sidecar["base"]["rgb_render"])
    if sidecar["output"].get("rot90"):
        kw["rot90"] = sidecar["output"]["rot90"]
    rs = sidecar.get("render", {})
    if rs.get("t18_inversion"):
        kw["t18_inversion"] = dict(rs["t18_inversion"])
    elif kw["render_params"] and isinstance(coerce_render_params(kw["render_params"]), T19_FAMILY):
        cfg = sidecar.get("auto", {}).get("config")
        if cfg in _T18_VERSION_BY_CONFIG:  # sidecars written before render.t18_inversion was recorded
            rp = coerce_render_params(kw["render_params"])
            kw["t18_inversion"] = dict(version=_T18_VERSION_BY_CONFIG[cfg], has_cam_matrix=bool(rp.matrix != IDENTITY))
    return kw
