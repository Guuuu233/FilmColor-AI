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
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .compare import block_mean
from .holder import HOLDER_GREY, HOLDER_OVER, HOLDER_PEAK, film_box, holder_mask  # noqa: F401 (T4i names, moved in T4l)
from .holder import rebate_strip_components, scene_highlight_components
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
    # (P4, mask4) = noref.estimation_input of the full-resolution positive, kept when an auto estimate was made, so a
    # style preset is re-estimated without averaging the full-resolution positive again (T4e)
    est_input: tuple | None = field(default=None, repr=False)
    holder: np.ndarray | None = field(default=None, repr=False)  # holder_mask of ``positive`` (T4i), display only
    _scene: np.ndarray | None = field(default=None, repr=False)   # T5a: scene_highlight_components(holder), cached
    _rebate: np.ndarray | None = field(default=None, repr=False)  # T5c: rebate_strip_components(positive, holder)

    def _paint_holder(self, e: np.ndarray) -> np.ndarray:
        if self.holder is not None and self.holder.any():
            if self._scene is None:
                self._scene = scene_highlight_components(self.holder)
            if self._rebate is None:
                self._rebate = rebate_strip_components(self.positive, self.holder)
            e[(self.holder & ~self._scene) | self._rebate] = np.float32(HOLDER_GREY)
            if float(self.holder.mean()) >= 0.15:
                fb = film_box(self.holder)
                if fb is not None:
                    t, l, h, w = fb
                    outer = np.ones(e.shape[:2], dtype=bool)
                    outer[t:t + h, l:l + w] = False
                    if outer.any():
                        e[outer] = np.float32(HOLDER_GREY)
        return e

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
        """sRGB-encoded display image (h, w, 3) float32 for render parameters (RenderParams or dict); the film holder
        (``holder``) is shown as ``HOLDER_GREY``, except clipped scene highlights at the frame edge
        (``holder.scene_highlight_components``, T5a)."""
        rp = params if isinstance(params, RenderParams) else RenderParams.from_dict(params or {})
        return self._paint_holder(render(self.positive, rp, self.mask)[0])

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
        return self._paint_holder(e), dict(below_black=share((self.positive < 0).any(-1)),
                                           above_white=share((pre > 1).any(-1)))

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

    def with_auto(self, auto) -> "Preview":
        """The same preview with the automatic estimate redone for noref config ``auto`` (a style preset), on the
        kept estimator input (or the kept full-resolution positive) exactly as ``load_preview`` does; no decode or
        inversion."""
        ei = self.est_input
        if ei is None:
            if self.full_positive is None:
                raise ValueError("load_preview(..., keep_full=True) or an auto estimate is needed to re-estimate")
            ei = estimation_input(self.full_positive, self.image_region)
        return replace(self, auto=auto_estimate(ei, auto, self.film, span_d_g=green_span(self.inversion)),
                       est_input=ei, load_kwargs=dict(self.load_kwargs, auto=auto))

    def export(self, out_dir, params=None, encoding: str = "srgb", *, stem: str | None = None,
               overwrite: bool = True, rot90: int = 0, render_source: dict | None = None) -> dict:
        """Write the full-resolution TIFF + sidecar with this preview's base and white point, turned ``rot90``
        quarter turns counter-clockwise. Refuses (SourceChangedError) when the source file no longer has the SHA256
        it had when previewed. ``render_source``: recorded as the sidecar's ``render.source``."""
        if sha256(self.path) != self.sha256:
            raise SourceChangedError(f"{self.path} changed since it was opened; open it again")
        base = self.base_info["rgb_render"] if self.film == "bw" else self.inversion["base"]
        rp = params.to_dict() if isinstance(params, RenderParams) else params
        return write(self.path, out_dir, stem=stem, overwrite=overwrite, base_rgb=base, d_hi=self.inversion["d_hi"],
                     image_region=self.image_region, white_pct=self.white_pct, gamma=self.inversion["gamma"],
                     exposure=self.inversion["exposure"], encoding=encoding, render_params=rp, film=self.film,
                     base_source=self.base_source(), white_point_source=self.report.get("white_point"), rot90=rot90,
                     render_source=render_source, spike_clamp=self.inversion.get("spike_clamp"))


class SourceChangedError(RuntimeError):
    """The previewed source file changed on disk; its previewed base no longer applies."""


def green_span(inversion: dict) -> float:
    """G density range of the inversion (white point minus black point, ``InversionParams.to_dict()``), T4k."""
    return float(inversion["d_hi"][1] - inversion["d_lo"][1])


# T5a input plausibility hints (report["input_warnings"]; status, parameters and inversion are not changed). Benchmark
# white points reach at most 1.59 D (G); A850-A852 (already positive images) 3.05-3.20. A229's blue channel is empty
# (centre p99 [0.831, 0.738, 0.002]); the next lowest channel p99 of the measured scanner TIFFs is 0.070.
INPUT_DHI_G_MAX = 2.40              # G white point density (D) above which the input is unlikely to be a negative
INPUT_EMPTY_CHANNEL = (0.02, 0.30)  # centre linear input: a channel p99 below [0] while another reaches [1]
INPUT_FLAT_CHANNEL_DHI = (0.04, 0.30)  # T5d: a channel white point below [0] D while another reaches [1] D


def input_warnings(d_hi, center_p99) -> list[dict]:
    """T5a: hints that the input may not be a usable negative scan, from the inversion white point ``d_hi`` and the
    per-channel p99 of the linear input in the central region (``pipeline.positive``'s ``input_center_p99``)."""
    out = []
    if float(d_hi[1]) > INPUT_DHI_G_MAX:
        out.append(dict(code="input_density_implausible", value=float(d_hi[1]),
                        message=f"密度范围 {float(d_hi[1]):.2f} D 超出负片常见范围：可能是已反相的正片、已编辑的图片或扫描裁切。"))
    empty_ch = set()
    if center_p99 is not None:
        p = [float(x) for x in center_p99]
        lo, hi = INPUT_EMPTY_CHANNEL
        for i, name in enumerate("RGB"):
            if p[i] < lo and max(p[j] for j in range(3) if j != i) >= hi:
                empty_ch.add(name)
                out.append(dict(code="input_channel_empty", channel=name, value=p[i],
                                message=f"{name} 通道几乎没有信号（中心 p99 = {p[i]:.3f}），扫描或导出可能丢失了该通道。"))
    d = [float(x) for x in d_hi]
    d_lo_thr, d_hi_thr = INPUT_FLAT_CHANNEL_DHI
    for i, name in enumerate("RGB"):
        if name not in empty_ch and d[i] < d_lo_thr and max(d[j] for j in range(3) if j != i) >= d_hi_thr:
            out.append(dict(code="input_channel_flat", channel=name, value=d[i],
                            message=f"{name} 通道密度跨度极小（{d[i]:.2f} D），可能存在通道底座抬升、漏光雾化或转码失真。"))
    return out


def auto_estimate(est_input, auto, film: str = "color", span_d_g: float | None = None) -> dict:
    """noref estimate (``Preview.auto``) for config ``auto`` on ``est_input`` = ``noref.estimation_input`` of the
    full-resolution positive, as ``convert``; ``span_d_g``: the inversion's G density range (``green_span``)."""
    est = estimate_params_v2(*est_input, auto, span_d_g=span_d_g)
    if film == "bw":
        est = dict(est, params=dict(est["params"], **BW_NEUTRAL), bw_neutralised=sorted(BW_NEUTRAL))
    return dict(est, look_file=dict(name=f"{est['look']}.json", sha256=sha256(CONFIG_DIR / f"{est['look']}.json")))


def load_preview(path, *, max_side: int = 1600, auto=None, base_regions=None, base_rgb=None, image_region=None,
                 d_hi=None, white_pct: float = 99.9, gamma: float = 0.6, exposure: float = 1.0,
                 film: str = "color", keep_full: bool = False, base_source: dict | None = None,
                 white_block: int | None = None, white_ratio=None, white_b_guard=None, white_self_ratio=None,
                 spike_clamp=None) -> Preview:
    """Full-resolution decode / base / inversion once; keep a block-mean copy of at most ``max_side`` pixels.
    ``auto``: a noref config (noref.NOREF_V2) estimated here on the full-resolution positive, exactly as
    ``convert(auto=...)`` does; the result is in ``Preview.auto`` (render with ``Preview.auto['params']``).
    ``white_block``: automatic white point on block means of that size (``invert.block_white_point``; the app uses
    it); None = single pixels as ``convert``. ``white_ratio``: roll neutral density ratios checked against the automatic
    white point (``invert.neutral_ratio_white_point``); ``white_b_guard``: colour film's ``invert.blue_white_guard``;
    ``white_self_ratio`` / ``spike_clamp``: T4l (``pipeline.positive``). The export replays the white point and the
    clamp either way."""
    path = Path(path)
    p = positive(path, base_regions=base_regions, base_rgb=base_rgb, image_region=image_region, d_hi=d_hi,
                 white_pct=white_pct, gamma=gamma, exposure=exposure, film=film, white_block=white_block,
                 white_ratio=white_ratio, white_b_guard=white_b_guard, white_self_ratio=white_self_ratio,
                 spike_clamp=spike_clamp)
    pos, mask = p["positive"], p["mask"]
    full_shape = pos.shape[:2]
    ei = estimation_input(pos, p["image_region"]) if auto is not None else None
    auto_info = auto_estimate(ei, auto, film, span_d_g=green_span(p["inversion"].params.to_dict())) if auto is not None \
        else None
    f = max(1, math.ceil(max(full_shape) / max_side))
    small = block_mean(pos, f).astype(np.float32) if f > 1 else pos
    msk = block_mean(mask.astype(np.float32), f) > 0.5 if f > 1 else mask
    full = pos if keep_full else None
    del pos
    res = p["inversion"]
    inside = p["base_inside_image"]
    if inside is None and base_source is not None:
        inside = base_source.get("base_inside_image")
    inv = res.params.to_dict()
    report = dict(res.report, input_center_p99=p["input_center_p99"],
                  input_warnings=input_warnings(inv["d_hi"], p["input_center_p99"]))
    return Preview(path=path, sha256=sha256(path), factor=f, positive=small, mask=msk,
                   image_region=p["image_region"], region_source=p["region_source"], film=film,
                   base_info=p["base_info"], inversion=inv, white_pct=white_pct, report=report,
                   base_inside_image=inside, full_shape=tuple(full_shape), auto=auto_info,
                   load_kwargs=dict(max_side=max_side, auto=auto, base_regions=base_regions, base_rgb=base_rgb,
                                    image_region=image_region, d_hi=d_hi, white_pct=white_pct, gamma=gamma,
                                    exposure=exposure, film=film, keep_full=keep_full, base_source=base_source,
                                    white_block=white_block, white_ratio=white_ratio, white_b_guard=white_b_guard,
                                    white_self_ratio=white_self_ratio, spike_clamp=spike_clamp),
                   full_positive=full, _base_source=base_source, est_input=ei, holder=holder_mask(small))
