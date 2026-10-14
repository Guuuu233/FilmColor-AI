"""Convert RAW negatives to 16-bit TIFF + sidecar JSON.

  convert.py OUT_DIR RAW... [--encoding srgb|linear] [--image-region T,L,H,W] [--base-region T,L,H,W ...]
                            [--render PARAMS.json]
  convert.py OUT_DIR RAW --replay SIDECAR.json
"""
import argparse
import json

from filmcolor.pipeline import replay_kwargs, write


def region(s):
    return tuple(int(x) for x in s.split(","))


ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("raw", nargs="+")
ap.add_argument("--encoding", default="srgb", choices=["srgb", "linear"])
ap.add_argument("--image-region", type=region)
ap.add_argument("--base-region", type=region, action="append")
ap.add_argument("--render", help="JSON file with render parameters (docs/SPRINT2_PARAMS.md)")
ap.add_argument("--replay")
a = ap.parse_args()
if a.replay:
    if len(a.raw) != 1:
        ap.error("--replay takes exactly one RAW")
    kw = replay_kwargs(json.load(open(a.replay)))
else:
    kw = dict(encoding=a.encoding, image_region=a.image_region, base_regions=a.base_region,
              render_params=json.load(open(a.render)) if a.render else None)
for p in a.raw:
    sc = write(p, a.out_dir, **kw)
    r = sc["report"]["range"]
    print(p, "->", sc["output"]["path"], "d_hi", [round(x, 3) for x in sc["inversion"]["params"]["d_hi"]],
          "clip_high%", [round(r[c]["clip_high"] * 100, 2) for c in "RGB"], flush=True)
