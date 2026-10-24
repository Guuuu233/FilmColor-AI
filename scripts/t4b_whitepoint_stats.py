"""T4b white-point noise check: per-pixel vs block-mean high density quantiles on the app's inversion input.

  t4b_whitepoint_stats.py OUT_JSON FOLDER [FOLDER ...]

For every frame of each roll folder (data/raw/own/<roll>), the base is the one the app opens it with (State.open
without base_mode: roll_nat at softness 0.5 when the frame's own B0 failed or its chroma deviates >= 0.10 D from
the roll, else per-frame B0). On the render-path linear RGB (demosaic_linear) inside the default image region and
unsaturated, it records the per-channel density quantiles 99 / 99.5 / 99.9 % of single pixels (99.9 % = the current
white point) and of b x b block means of the linear signal (b = 8, 16, 24; a block counts when all its pixels are
valid) at 99.5 / 99.8 / 99.9 %. Reads only the dev roll folders; frozen sets are refused by State.resolve.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from filmcolor.app import State
from filmcolor.compare import block_mean
from filmcolor.invert import central_region, density, region_mask
from filmcolor.pipeline import _camera_input

BLOCKS = (8, 16, 24)
BLOCK_PCTS = (99.5, 99.8, 99.9)
PIXEL_PCTS = (99.0, 99.5, 99.9)


def frame_stats(path: str, base_rgb, mode: str) -> dict:
    rgb, sat, base, base_info, _, _, _ = _camera_input(Path(path), base_regions=None, base_rgb=base_rgb,
                                                       image_region=None)
    mask = ~sat & region_mask(sat.shape, central_region(sat.shape))
    Dm = density(rgb[mask], base)
    out = dict(mode=mode, base=[float(x) for x in base], n_pixels=int(len(Dm)),
               pixel={str(q): [float(x) for x in np.percentile(Dm, q, 0)] for q in PIXEL_PCTS})
    del Dm
    out["block"] = {}
    for b in BLOCKS:
        bm = block_mean(rgb, b)
        ok = block_mean(mask.astype(np.float32), b) >= 1.0
        Db = density(bm[ok], base)
        out["block"][str(b)] = dict(n=int(ok.sum()),
                                    q={str(q): [float(x) for x in np.percentile(Db, q, 0)] for q in BLOCK_PCTS})
    return out


def main(out_json: str, folders: list[str]):
    jobs = []
    for folder in folders:
        st = State(Path(folder), Path("/tmp/fc_t4b_out"), allow_frozen=False)
        info = st.roll_info()
        for rel, fe in info["frames"].items():
            flagged = info.get("roll") and (fe.get("own_error") or (fe.get("chroma_part") or 0) >= 0.1)
            if flagged:
                base_rgb, mode = st._roll_base_kwargs(rel, "roll_nat", 0.5)["base_rgb"], "roll_nat"
            elif fe.get("own_error"):
                continue  # no base the app could open it with
            else:
                base_rgb, mode = None, "auto"
            jobs.append((Path(folder).name, rel, str(st.resolve(rel)), base_rgb, mode))
    res = {}
    with ProcessPoolExecutor(4) as ex:
        futs = {(roll, rel): ex.submit(frame_stats, p, b, m) for roll, rel, p, b, m in jobs}
        for k, f in futs.items():
            try:
                res[f"{k[0]}/{k[1]}"] = f.result()
            except ValueError as e:
                res[f"{k[0]}/{k[1]}"] = dict(error=str(e))
            print(k, "done", flush=True)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(res, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
