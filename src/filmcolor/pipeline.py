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

from .base import estimate_base_auto, estimate_base_manual
from .export import DESCRIPTION, write_tiff16
from .invert import base_from_regions, central_region, invert_negative, region_mask
from .qc import quality_checks
from .raw import check_region, linearize, saturation_mask
from .rawpy_io import demosaic_linear, load_raw

SCHEMA = "filmcolor.sidecar/0.1"


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


def convert(path, *, base_regions=None, base_rgb=None, image_region=None, d_hi=None,
            white_pct: float = 99.9, gamma: float = 0.6, exposure: float = 1.0, encoding: str = "srgb"):
    """Return (image in [0, 1] encoded per ``encoding``, sidecar dict without output fields)."""
    if encoding not in DESCRIPTION:
        raise ValueError(f"encoding must be one of {tuple(DESCRIPTION)}")
    if not 0 < white_pct <= 100 or gamma <= 0 or exposure <= 0:
        raise ValueError(f"invalid parameters: white_pct={white_pct} gamma={gamma} exposure={exposure}")
    path = Path(path)
    f = load_raw(path)
    if image_region is not None:  # fail before the slow render step
        if len(image_region) != 4:
            raise ValueError(f"image region must be (top, left, height, width), got {image_region}")
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
        base_info = dict(method=est.method, estimate=est.to_dict())
    del lin, col

    rgb = demosaic_linear(path)
    if base_rgb is None:
        rb = base_from_regions(rgb, regions)  # base on the same signal that is inverted
        base_rgb = rb["rgb"]
        base_info["rgb_render"] = list(rb["rgb"])
        base_info["render_stats"] = rb

    region_source = "manual" if image_region else "central_default"
    image_region = tuple(int(x) for x in (image_region or central_region(sat.shape)))
    mask = ~sat & region_mask(sat.shape, image_region)
    res = invert_negative(rgb, base_rgb, mask=mask, white_pct=white_pct, gamma=gamma,
                          exposure=exposure, d_hi=d_hi)
    del rgb
    inside = None
    if regions is not None and base_info["method"] == "auto":
        t, l, h, w = image_region
        centres = [(r + rh / 2, c + rw / 2) for r, c, rh, rw in regions]
        inside = float(np.mean([t <= y < t + h and l <= x < l + w for y, x in centres]))
    qc = quality_checks(res.display, mask, res.report, base_inside_image=inside)
    if encoding == "srgb":
        out = res.display
    else:
        out = np.clip(res.positive * np.float32(res.params.exposure), 0, 1)

    sidecar = dict(
        schema=SCHEMA,
        input=dict(path=str(path), sha256=sha256(path), shape=list(f.cfa.shape),
                   cfa_pattern=f.pattern.tolist(), black=f.black.tolist(), white=f.white,
                   crop=list(f.crop) if f.crop else None, meta=f.meta),
        software=software_info(),
        decode=dict(analysis="CFA-aware linearisation (filmcolor.raw)",
                    render="rawpy postprocess, default demosaic, raw colour, linear, no WB, "
                           "no auto-bright, adjust_maximum_thr=0, highlight clip, no flip"),
        base=base_info,
        image_region=dict(region=list(image_region), source=region_source),
        inversion=dict(params=res.params.to_dict(), white_pct=white_pct, d_hi_source="given" if d_hi else "auto"),
        report=res.report,
        qc=qc,
        output=dict(encoding=encoding, colorimetric_calibration="none",
                    icc_description=DESCRIPTION[encoding], bit_depth=16,
                    orientation="sensor (meta.flip not applied)"),
    )
    return out, sidecar


def write(path, out_dir, **kwargs) -> dict:
    """Convert and write <stem>.tif plus <stem>.json; return the sidecar."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img, sidecar = convert(path, **kwargs)
    tif = out_dir / f"{Path(path).stem}.tif"
    write_tiff16(tif, img, sidecar["output"]["encoding"])
    sidecar["output"].update(path=str(tif), sha256=sha256(tif))
    (out_dir / f"{Path(path).stem}.json").write_text(json.dumps(sidecar, indent=1))
    return sidecar


def replay_kwargs(sidecar: dict) -> dict:
    """Arguments that reproduce a sidecar's output without re-estimating anything."""
    p = sidecar["inversion"]["params"]
    return dict(base_rgb=p["base"], d_hi=p["d_hi"], image_region=sidecar["image_region"]["region"],
                white_pct=sidecar["inversion"]["white_pct"], gamma=p["gamma"], exposure=p["exposure"],
                encoding=sidecar["output"]["encoding"])
