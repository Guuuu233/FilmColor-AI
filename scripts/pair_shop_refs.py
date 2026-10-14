"""Pair reference positives (e.g. shop JPEGs) with negative RAWs by image content.

Every reference is aligned against every RAW (rotation, scale, translation; normalised cross-
correlation of negative density vs positive L*), so the pairing does not rely on file order.

  pair_shop_refs.py RAW_DIR REF_DIR OUT_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image

from filmcolor.colorimetry import xyz_to_lab
from filmcolor.compare import align, block_mean

WORK = 16  # working resolution = full resolution / WORK
SCALES = np.arange(0.96, 1.0401, 0.01)


def raw_density(path):
    with rawpy.imread(str(path)) as r:
        rgb = r.postprocess(half_size=True, output_color=rawpy.ColorSpace.raw, gamma=(1, 1), no_auto_bright=True,
                            output_bps=16, user_wb=[1, 1, 1, 1], user_flip=0)
    lum = block_mean(rgb.astype(np.float64) / 65535, WORK // 2).mean(-1)
    return -np.log10(np.maximum(lum, 1e-4))


def ref_lightness(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), np.float64) / 255  # sRGB-encoded (embedded profile is sRGB)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    y = block_mean(lin @ np.array([0.2126, 0.7152, 0.0722]), WORK)
    return xyz_to_lab(np.stack([y * 0.9642, y, y * 0.8249], -1))[..., 0]


def main(raw_dir, ref_dir, out_json):
    raws = sorted(p for p in Path(raw_dir).iterdir() if p.suffix.lower() == ".nef")
    refs = sorted(Path(ref_dir).glob("*.jpg"), key=lambda p: int(p.stem.split("(")[1].rstrip(")")))
    D = {p.name: raw_density(p) for p in raws}
    out = {}
    for ref in refs:
        L = ref_lightness(ref)
        scores = {}
        for name, d in D.items():
            al = align(d, L, 1, SCALES)
            scores[name] = al
        ranked = sorted(scores, key=lambda n: scores[n].ncc, reverse=True)
        n = int(ref.stem.split("(")[1].rstrip(")"))
        hyp = raws[n - 1].name
        best = scores[ranked[0]]
        out[ref.name] = dict(best=ranked[0], ncc=best.ncc, second=ranked[1], ncc_second=scores[ranked[1]].ncc,
                             hypothesis_by_order=hyp, ncc_hypothesis=scores[hyp].ncc, agrees=ranked[0] == hyp,
                             rot90=best.rot90, scale=best.scale, offset_work=list(best.offset), work_factor=WORK)
        print(ref.name, "->", ranked[0], f"ncc {best.ncc:.3f}", f"2nd {ranked[1]} {scores[ranked[1]].ncc:.3f}",
              "order-hypothesis", hyp, "agrees" if ranked[0] == hyp else "DIFFERS", f"rot {best.rot90} scale {best.scale:.2f}", flush=True)
    Path(out_json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
