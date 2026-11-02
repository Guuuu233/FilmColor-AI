"""End-to-end conversion of one RAW negative, with a sidecar that records everything needed to
reproduce the output (inputs, software versions, every estimated and given parameter, diagnostics).

Replay: pass ``base_rgb`` and ``d_hi`` from a sidecar (plus its image region, gamma, exposure and
encoding) to skip estimation and get the same pixels.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import rawpy
import tifffile

from .base import estimate_base_auto, estimate_base_manual
from .scanner import estimate_base_scanner, is_scanner_dng, load_scanner
from .export import DESCRIPTION, write_tiff16
from .invert import base_from_regions, central_region, invert_negative, neutral_transmission, region_mask
from .qc import quality_checks
from .raw import check_region, linearize, saturation_mask
from .noref import CONFIG_DIR, estimate_params_v2, estimation_input
from .render import IDENTITY, RenderParams, render as render_image, srgb_eotf
from .rawpy_io import demosaic_linear, load_raw

SCHEMA = "filmcolor.sidecar/0.2"  # 0.2: render section (Sprint 2 parameters)
FILM_MODES = ("color", "bw")
BW_NEUTRAL = dict(temp=0.0, tint=0.0, sat=1.0, matrix=IDENTITY)  # look parts not applied to B&W film


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def software_info() -> dict:
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "src"],
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return dict(filmcolor=importlib.metadata.version("filmcolor"), git_commit=commit, src_dirty=dirty,
                rawpy=rawpy.__version__, libraw=".".join(map(str, rawpy.libraw_version)), numpy=np.__version__)


def _base_info_from(est):
    return dict(method=est.method, estimate=est.to_dict())


def _camera_input(path, *, base_regions, base_rgb, image_region):
    """Camera RAW: CFA-aware base estimate on the analysis path, demosaiced linear render path."""
    f = load_raw(path)
    if image_region is not None:  # fail before the slow render step
        check_region(tuple(int(x) for x in image_region), f.cfa.shape)
    lin, col, sat = linearize(f), f.colors, saturation_mask(f)

    if base_rgb is not None:
        base_info = dict(method="given", rgb_render=[float(x) for x in base_rgb])
        regions = None
    else:
        if base_regions:
            est = estimate_base_manual(lin, col, sat, base_regions)
        else:
            est = estimate_base_auto(lin, col, sat, f.pattern.shape[0], crop=f.crop)
        regions = est.regions
        base_info = _base_info_from(est)
    del lin, col

    rgb = demosaic_linear(path)
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
    input_info = dict(kind="scanner_tiff16" if is_tiff else "scanner_linear_dng",
                      shape=list(sat.shape), black=fr.black, white=fr.white, meta=fr.meta)
    decode_info = dict(analysis="same as render (no CFA)",
                       render="tifffile %s, LinearRaw main image, (v - black) / (white - black), "
                              "infrared layer excluded" % tifffile.__version__)
    if is_tiff:
        decode_info["render"] = ("tifffile %s with native LZW / PackBits strip fallback, 16-bit scanner image, "
                                 "(v - black) / (white - black), infrared excluded; signal assumed linear, "
                                 "ICC / TRC not applied" % tifffile.__version__)
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
    colour film only, ``invert.self_ratio_white_point`` (T4l). ``spike_clamp``: ``invert.clamp_spikes`` (T4l)."""
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
                base_info=base_info, input_info=input_info, decode_info=decode_info, base_inside_image=inside)


def convert(path, *, base_regions=None, base_rgb=None, image_region=None, d_hi=None,
            white_pct: float = 99.9, gamma: float = 0.6, exposure: float = 1.0, encoding: str = "srgb",
            render_params=None, auto=None, film: str = "color", base_source: dict | None = None,
            white_block: int | None = None, white_point_source: dict | None = None, rot90: int = 0,
            render_source: dict | None = None, white_b_guard=None, white_self_ratio=None, spike_clamp=None):
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
    colour negatives). Explicit ``render_params`` are applied as given in either mode."""
    if film not in FILM_MODES:
        raise ValueError(f"film must be one of {FILM_MODES}, got {film!r}")
    if auto is not None and render_params is not None:
        raise ValueError("auto and render_params are mutually exclusive")
    rp = render_params if isinstance(render_params, RenderParams) else RenderParams.from_dict(render_params or {})
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
    auto_info = None
    if auto is not None:  # estimated from this frame's positive only (no reference, no roll context)
        est = estimate_params_v2(*estimation_input(pos, image_region), auto,
                                 span_d_g=float(res.params.d_hi[1] - res.params.d_lo[1]))
        if film == "bw":  # colour parts of the look were fitted on colour negatives
            est = dict(est, params=dict(est["params"], **BW_NEUTRAL), bw_neutralised=sorted(BW_NEUTRAL))
        rp = RenderParams.from_dict(est["params"])
        look_file = CONFIG_DIR / f"{est['look']}.json"
        auto_info = dict(est, look_file=dict(name=look_file.name, sha256=sha256(look_file)),
                         input="positive x inversion exposure, 4x4 block mean, image region")
    display, render_diag = render_image(pos, rp, mask)
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
    return kw
