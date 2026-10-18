"""How close is the preview (filmcolor.preview) to the downscaled full-resolution render, by region and parameters.

Error definition: |preview render - block mean (same factor) of the full-resolution render|, per pixel and
channel, sRGB-encoded display values in [0, 1] (the values written to the 16-bit TIFF before quantisation). The
full-resolution render is the export's own path (pipeline.positive + render.render with the same parameters).
Regions, defined on the downscaled reference: dark (luma < 0.1), bright (luma > 0.9), saturated (max - min
channel > 0.4), edges (top 5 % of the luma gradient magnitude), clipped (block contains a full-resolution pixel
at output white in any channel). Development files only; check_source guards the frozen sets.

  preview_check.py OUT_JSON RAW [RAW ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from filmcolor.compare import block_mean
from filmcolor.frozen import check_source
from filmcolor.pipeline import positive
from filmcolor.preview import load_preview
from filmcolor.render import RenderParams, render

PARAM_SETS = {
    "default": {},
    "moderate": dict(ev=0.3, temp=-0.2, tint=0.1, sat=1.2, contrast=1.3, midtone=0.3, shoulder=0.2),
    "extreme_bright_hard": dict(ev=2.0, contrast=2.0, sat=2.0, shoulder=0.0, midtone=0.5),
    "extreme_dark_soft": dict(ev=-2.0, contrast=0.5, sat=0.0, shoulder=1.0, midtone=0.05),
    "extreme_cast": dict(temp=1.0, tint=-1.0, sat=1.8),
}
LUMA = np.array([0.2126, 0.7152, 0.0722])


def stats(d, m):
    x = d[m] if m.any() else np.zeros((1, 3))
    return dict(share=round(float(m.mean()), 4), mean=round(float(x.mean()), 5),
                p99=round(float(np.percentile(x, 99)), 4), max=round(float(x.max()), 4))


def main(out_json, *raws):
    out = {}
    for raw in raws:
        check_source(raw)
        t = time.time()
        pv = load_preview(raw)
        t_load = time.time() - t
        pos = positive(raw)["positive"]
        f, (h, w) = pv.factor, pv.positive.shape[:2]
        res = dict(factor=f, preview_shape=[h, w], load_s=round(t_load, 1), params={})
        for name, p in PARAM_SETS.items():
            rp = RenderParams.from_dict(p)
            t = time.time()
            prev = pv.render(rp)
            t_render = time.time() - t
            full = render(pos, rp, None)[0]
            ref = block_mean(full, f)[:h, :w]
            clip = block_mean((full >= 1).any(-1).astype(np.float32), f)[:h, :w] > 0
            del full
            d = np.abs(prev - ref)
            y = ref @ LUMA
            gy, gx = np.gradient(y)
            g = np.hypot(gx, gy)
            regions = dict(all=np.ones_like(y, bool), dark=y < 0.1, bright=y > 0.9,
                           saturated=(ref.max(-1) - ref.min(-1)) > 0.4, edges=g >= np.percentile(g, 95), clipped=clip)
            res["params"][name] = dict(render_s=round(t_render, 3), **{k: stats(d, m) for k, m in regions.items()})
            print(Path(raw).name, name, json.dumps(res["params"][name]["all"]), "edges p99",
                  res["params"][name]["edges"]["p99"], "clipped p99", res["params"][name]["clipped"]["p99"], flush=True)
        del pos
        out[Path(raw).name] = res
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
