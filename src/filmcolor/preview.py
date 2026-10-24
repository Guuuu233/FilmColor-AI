"""Fast preview for interactive use (docs/SOFTWARE_PLAN_v1.md M1).

Decode, film base and inversion run once at full resolution (``pipeline.positive``); the scene-linear positive is
then block-averaged to at most ``max_side`` pixels and kept. Changing render parameters only re-renders that small
copy with the same ``render`` the export uses (per-pixel, so the preview differs from a downscaled export only by
the order of averaging and the curve). The export replays the preview's base and white point at full resolution
(the ``--replay`` semantics of pipeline.replay_kwargs): nothing is re-estimated, so what was previewed is what is
written.

What the cached copy covers: the seven render parameters (``RenderParams``) only. Everything ``load_preview``
takes - base regions or base RGB, image region (it sets the white point and the QC statistics), white point,
gamma, inversion exposure, film mode, auto estimate - is fixed in the cache; changing any of them needs a new load
(``Preview.reload(**changes)``), which redoes decode, base and inversion. ``Preview.load_kwargs`` records them.

Downscaling: block mean (area average) of the scene-linear positive by an integer factor, remainder rows / columns
dropped. The render after it is per pixel but non-linear (tone curve, shoulder, sRGB encoding), so the preview of
an averaged block differs from the average of the rendered block, most at strong edges and clipped highlights.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .compare import block_mean
from .noref import CONFIG_DIR, estimate_params_v2, estimation_input
from .pipeline import BW_NEUTRAL, positive, sha256, write
from .render import RenderParams, render, render_full


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
    load_kwargs: dict = field(default_factory=dict)  # what load_preview was called with (besides path)
    full_positive: np.ndarray | None = field(default=None, repr=False)  # kept with keep_full=True (1:1 view)
    _base_source: dict | None = field(default=None, repr=False)

    def render_crop(self, region, params=None) -> np.ndarray:
        """1:1 render of a sensor-pixel region (top, left, height, width) of the full-resolution positive,
        exactly the export's pixels there (needs keep_full=True)."""
        if self.full_positive is None:
            raise ValueError("load_preview(..., keep_full=True) is needed for a 1:1 view")
        t, l, h, w = (int(v) for v in region)
        H, W = self.full_positive.shape[:2]
        t, l = max(0, min(t, H - 1)), max(0, min(l, W - 1))
        crop = self.full_positive[t:t + h, l:l + w]
        rp = params if isinstance(params, RenderParams) else RenderParams.from_dict(params or {})
        return render(crop, rp, None)[0]

    def reload(self, **changes) -> "Preview":
        """A new preview with some load arguments changed (base, region, film, ...): decode and invert again."""
        return load_preview(self.path, **(self.load_kwargs | changes))

    def render(self, params=None) -> np.ndarray:
        """sRGB-encoded display image (h, w, 3) float32 for render parameters (RenderParams or dict)."""
        rp = params if isinstance(params, RenderParams) else RenderParams.from_dict(params or {})
        return render(self.positive, rp, self.mask)[0]

    def render_stats(self, params=None) -> tuple[np.ndarray, dict]:
        """``render`` plus per-pixel shares over ``mask`` (image region, unsaturated) of the preview: below_black
        (positive < 0 in at least one channel, i.e. denser than the base) and above_white (display-linear value
        before the white clip > 1 in at least one channel). Measured on the block-mean copy, so isolated clipped
        pixels averaged into a block are not counted (use render_crop to inspect)."""
        rp = params if isinstance(params, RenderParams) else RenderParams.from_dict(params or {})
        e, _, pre = render_full(self.positive, rp)
        m = self.mask
        n = int(m.sum())
        share = (lambda flags: float(flags[m].sum() / n)) if n else (lambda flags: float("nan"))
        return e, dict(below_black=share((self.positive < 0).any(-1)), above_white=share((pre > 1).any(-1)))

    def to_sensor(self, region):
        """(top, left, height, width) in preview pixels -> sensor pixels."""
        return tuple(int(v) * self.factor for v in region)

    def to_preview(self, region):
        """(top, left, height, width) in sensor pixels -> preview pixels (rounded down)."""
        return tuple(int(v) // self.factor for v in region)

    def base_source(self) -> dict:
        """How this preview's base was found, for the export sidecar (``base.source``)."""
        if self._base_source is not None:
            return dict(self._base_source, base_inside_image=self.base_inside_image,
                        previewed_input_sha256=self.sha256)
        regions = self.load_kwargs.get("base_regions")
        return dict(method=self.base_info.get("method"), estimate=self.base_info.get("estimate"),
                    regions=[list(map(int, r)) for r in regions] if regions else None,
                    base_inside_image=self.base_inside_image, previewed_input_sha256=self.sha256)

    def export(self, out_dir, params=None, encoding: str = "srgb", *, stem: str | None = None,
               overwrite: bool = True, rot90: int = 0) -> dict:
        """Write the full-resolution TIFF + sidecar with this preview's base and white point, turned ``rot90``
        quarter turns counter-clockwise. Refuses (SourceChangedError) when the source file no longer has the SHA256
        it had when previewed."""
        if sha256(self.path) != self.sha256:
            raise SourceChangedError(f"{self.path} changed since it was opened; open it again")
        base = self.base_info["rgb_render"] if self.film == "bw" else self.inversion["base"]
        rp = params.to_dict() if isinstance(params, RenderParams) else params
        return write(self.path, out_dir, stem=stem, overwrite=overwrite, base_rgb=base, d_hi=self.inversion["d_hi"],
                     image_region=self.image_region, white_pct=self.white_pct, gamma=self.inversion["gamma"],
                     exposure=self.inversion["exposure"], encoding=encoding, render_params=rp, film=self.film,
                     base_source=self.base_source(), white_point_source=self.report.get("white_point"), rot90=rot90)


class SourceChangedError(RuntimeError):
    """The previewed source file changed on disk; its previewed base no longer applies."""


def load_preview(path, *, max_side: int = 1600, auto=None, base_regions=None, base_rgb=None, image_region=None,
                 d_hi=None, white_pct: float = 99.9, gamma: float = 0.6, exposure: float = 1.0,
                 film: str = "color", keep_full: bool = False, base_source: dict | None = None,
                 white_block: int | None = None) -> Preview:
    """Full-resolution decode / base / inversion once; keep a block-mean copy of at most ``max_side`` pixels.
    ``auto``: a noref config (noref.NOREF_V2) estimated here on the full-resolution positive, exactly as
    ``convert(auto=...)`` does; the result is in ``Preview.auto`` (render with ``Preview.auto['params']``).
    ``white_block``: automatic white point on block means of that size (``invert.block_white_point``; the app uses
    it); None = single pixels as ``convert``. The export replays the white point either way."""
    path = Path(path)
    p = positive(path, base_regions=base_regions, base_rgb=base_rgb, image_region=image_region, d_hi=d_hi,
                 white_pct=white_pct, gamma=gamma, exposure=exposure, film=film, white_block=white_block)
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
    full = pos if keep_full else None
    del pos
    res = p["inversion"]
    inside = p["base_inside_image"]
    if inside is None and base_source is not None:
        inside = base_source.get("base_inside_image")
    return Preview(path=path, sha256=sha256(path), factor=f, positive=small, mask=msk,
                   image_region=p["image_region"], region_source=p["region_source"], film=film,
                   base_info=p["base_info"], inversion=res.params.to_dict(), white_pct=white_pct, report=res.report,
                   base_inside_image=inside, full_shape=tuple(full_shape), auto=auto_info,
                   load_kwargs=dict(max_side=max_side, auto=auto, base_regions=base_regions, base_rgb=base_rgb,
                                    image_region=image_region, d_hi=d_hi, white_pct=white_pct, gamma=gamma,
                                    exposure=exposure, film=film, keep_full=keep_full, base_source=base_source,
                                    white_block=white_block),
                   full_positive=full, _base_source=base_source)
