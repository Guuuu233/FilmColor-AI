"""Look at conversions next to the finished print (development frames only, from the S3 caches): reference print,
engine defaults, fixed look (global_look_v1, what the reference-free auto mode starts from), the P4 label (fitted
to the print: what a perfect P4 prediction would give) and the M1 prediction where one exists. Rendered on the
aligned 1/8 grid with our sRGB output encoding; the reference is converted from its Lab.

  s3_preview.py LABEL_DIR RESULTS_JSON OUT_DIR ID [ID ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from filmcolor.colorimetry import lab_to_xyz, xyz_to_srgb_preview
from filmcolor.frozen import load_cache
from filmcolor.render import RenderParams, render

LOOK = json.loads(Path("src/filmcolor/configs/global_look_v1.json").read_text())["params"]
W, LABEL = 560, 20


def tile(rgb, text):
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8))
    im = im.resize((W, round(W * im.height / im.width)), Image.LANCZOS)
    out = Image.new("RGB", (W, im.height + LABEL), "white")
    out.paste(im, (0, LABEL))
    ImageDraw.Draw(out).text((4, 4), text, fill="black")
    return out


def main(label_dir, results_json, out_dir, *ids):
    res = json.load(open(results_json))
    m1 = {r["id"]: r["M1"] for k in ("A_a:p4", "B_dev_a:p4") for r in res[k]["frames"]}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for fid in ids:
        domain, stem = fid.split(":", 1)
        domain = {"ext": "ext_own"}.get(domain, domain)
        lab = json.loads((Path(label_dir) / domain / f"{stem}.json").read_text())
        cache = Path(lab["cache"]["dir"])
        P, R = load_cache(cache / f"{stem}_P8.npy"), load_cache(cache / f"{stem}_R8.npy")
        cols = [tile(xyz_to_srgb_preview(lab_to_xyz(R.astype(np.float64))), f"{fid}  shop / print (reference)")]
        for name, params, de in (("default", RenderParams().to_dict(), None), ("fixed look", LOOK, lab["fixed_look"]["de2000_median"]),
                                 ("P4 label (fitted to print)", lab["p4"]["params"], lab["p4"]["de2000_median"])):
            cols.append(tile(render(P, RenderParams(**params))[0], name + ("" if de is None else f"  dE {de:.1f}")))
        if fid in m1:
            cols.append(tile(render(P, RenderParams(**m1[fid]["params"]))[0], f"M1 prediction  dE {m1[fid]['de_median']:.1f}"))
        row = Image.new("RGB", (W * 5, max(c.height for c in cols)), "white")
        for i, c in enumerate(cols):
            row.paste(c, (i * W, 0))
        row.save(out_dir / f"{fid.replace(':', '_')}.jpg", quality=90)
        rows.append(row)
    sheet = Image.new("RGB", (W * 5, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(out_dir / "overview.jpg", quality=88)
    print(out_dir / "overview.jpg")


if __name__ == "__main__":
    main(*sys.argv[1:])
