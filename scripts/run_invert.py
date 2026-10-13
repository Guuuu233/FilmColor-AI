"""Run the deterministic inversion on RAW files; write per-file parameter/diagnostic JSON and a preview sheet."""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

from filmcolor.base import estimate_base_auto
from filmcolor.invert import base_from_regions, central_region, invert_negative, region_mask
from filmcolor.raw import linearize, saturation_mask
from filmcolor.rawpy_io import demosaic_linear, load_raw

PREVIEW_STEP = 8


def run(path, image_region=None):
    f = load_raw(path)
    lin, col, sat = linearize(f), f.colors, saturation_mask(f)
    auto = estimate_base_auto(lin, col, sat, f.pattern.shape[0], crop=f.crop)
    rgb = demosaic_linear(path)
    base = base_from_regions(rgb, auto.regions)  # same signal as the inverted image
    region_source = "manual" if image_region else "central_default"
    image_region = image_region or central_region(sat.shape)
    mask = ~sat & region_mask(sat.shape, image_region)
    res = invert_negative(rgb, base["rgb"], mask=mask)
    h, w = (rgb.shape[0] // PREVIEW_STEP) * PREVIEW_STEP, (rgb.shape[1] // PREVIEW_STEP) * PREVIEW_STEP
    prev = res.display[:h, :w].reshape(h // PREVIEW_STEP, PREVIEW_STEP, w // PREVIEW_STEP, PREVIEW_STEP, 3).mean(axis=(1, 3))
    info = dict(file=str(path), base_analysis=auto.rgb, base_render=base, params=res.params.to_dict(),
                image_region=image_region, image_region_source=region_source,
                report=res.report, base_warnings=auto.warnings)
    return info, (prev * 255 + 0.5).astype(np.uint8)


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    infos, tiles = [], []
    for p in sys.argv[2:]:
        info, prev = run(p)
        infos.append(info); tiles.append(prev)
        r = info["report"]["range"]
        print(Path(p).name, "d_hi", [round(x, 3) for x in info["params"]["d_hi"]], "R", round(info["params"]["log_range"], 2),
              "clip_high", [round(r[c]["clip_high"] * 100, 2) for c in "RGB"], "below_black%", [round(r[c]["below_black"] * 100, 2) for c in "RGB"],
              "full_above_white%", [round(x * 100, 1) for x in info["report"]["full_image_above_white"]],
              flush=True)
        tifffile.imwrite(out / f"{Path(p).stem}_preview.tif", prev)
    (out / "invert.json").write_text(json.dumps(infos, indent=1))
