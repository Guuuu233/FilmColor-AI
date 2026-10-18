"""Fast preview for interactive use (docs/SOFTWARE_PLAN_v1.md M1).

Decode, film base and inversion run once at full resolution (``pipeline.positive``); the scene-linear positive is
then block-averaged to at most ``max_side`` pixels and kept. Changing render parameters only re-renders that small
copy with the same ``render`` the export uses (per-pixel, so the preview differs from a downscaled export only by
the order of averaging and the curve). The export replays the preview's base and white point at full resolution
(the ``--replay`` semantics of pipeline.replay_kwargs): nothing is re-estimated, so what was previewed is what is
written.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .compare import block_mean
from .noref import CONFIG_DIR, estimate_params_v2, estimation_input
from .pipeline import BW_NEUTRAL, positive, sha256, write
from .render import RenderParams, render


@dataclass
class Preview:
    path: Path
    sha256: str
    factor: int                      # sensor pixels per preview pixel (per side)
    positive: np.ndarray             # (h, w, 3) float32, block mean of the full-resolution positive
    mask: np.ndarray                 # (h, w) bool, block mostly unsaturated and inside the image region
    image_region: tuple
    region_source: str
    film: str
    base_info: dict
    inversion: dict                  # InversionParams.to_dict(): base, d_hi, gamma, exposure, ...
    white_pct: float
    report: dict
    base_inside_image: float | None
    full_shape: tuple
    auto: dict | None = field(default=None)   # noref estimate on the full-resolution positive, if requested

    def render(self, params=None) -> np.ndarray:
        """sRGB-encoded display image (h, w, 3) float32 for render parameters (RenderParams or dict)."""
        rp = params if isinstance(params, RenderParams) else RenderParams.from_dict(params or {})
        return render(self.positive, rp, self.mask)[0]

    def to_sensor(self, region):
        """(top, left, height, width) in preview pixels -> sensor pixels."""
        return tuple(int(v) * self.factor for v in region)

    def to_preview(self, region):
        """(top, left, height, width) in sensor pixels -> preview pixels (rounded down)."""
        return tuple(int(v) // self.factor for v in region)

    def export(self, out_dir, params=None, encoding: str = "srgb") -> dict:
        """Write the full-resolution TIFF + sidecar with this preview's base and white point."""
        base = self.base_info["rgb_render"] if self.film == "bw" else self.inversion["base"]
        rp = params.to_dict() if isinstance(params, RenderParams) else params
        return write(self.path, out_dir, base_rgb=base, d_hi=self.inversion["d_hi"], image_region=self.image_region,
                     white_pct=self.white_pct, gamma=self.inversion["gamma"], exposure=self.inversion["exposure"],
                     encoding=encoding, render_params=rp, film=self.film)


def load_preview(path, *, max_side: int = 1600, auto=None, base_regions=None, base_rgb=None, image_region=None,
                 d_hi=None, white_pct: float = 99.9, gamma: float = 0.6, exposure: float = 1.0,
                 film: str = "color") -> Preview:
    """Full-resolution decode / base / inversion once; keep a block-mean copy of at most ``max_side`` pixels.
    ``auto``: a noref config (noref.NOREF_V2) estimated here on the full-resolution positive, exactly as
    ``convert(auto=...)`` does; the result is in ``Preview.auto`` (render with ``Preview.auto['params']``)."""
    path = Path(path)
    p = positive(path, base_regions=base_regions, base_rgb=base_rgb, image_region=image_region, d_hi=d_hi,
                 white_pct=white_pct, gamma=gamma, exposure=exposure, film=film)
    pos, mask = p["positive"], p["mask"]
    full_shape = pos.shape[:2]
    auto_info = None
    if auto is not None:
        est = estimate_params_v2(*estimation_input(pos, p["image_region"]), auto)
        if film == "bw":
            est = dict(est, params=dict(est["params"], **BW_NEUTRAL), bw_neutralised=sorted(BW_NEUTRAL))
        auto_info = dict(est, look_file=dict(name=f"{est['look']}.json", sha256=sha256(CONFIG_DIR / f"{est['look']}.json")))
    f = max(1, math.ceil(max(full_shape) / max_side))
    small = block_mean(pos, f).astype(np.float32) if f > 1 else pos
    msk = block_mean(mask.astype(np.float32), f) > 0.5 if f > 1 else mask
    del pos
    res = p["inversion"]
    return Preview(path=path, sha256=sha256(path), factor=f, positive=small, mask=msk,
                   image_region=p["image_region"], region_source=p["region_source"], film=film,
                   base_info=p["base_info"], inversion=res.params.to_dict(), white_pct=white_pct, report=res.report,
                   base_inside_image=p["base_inside_image"], full_shape=tuple(full_shape), auto=auto_info)
