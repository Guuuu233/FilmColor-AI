"""Cross-device evaluation on BlueNeg (S3-3): our frozen outputs vs the scanned printed photos.

The prints carry no ICC profile; they are read as sRGB-encoded 16-bit (D50-adapted sRGB matrix,
the same PCS as our ICC-tagged outputs). That is an assumption, and the prints are old and may be
faded, so differences are not colour accuracy. Registration is our own (transformations.pkl is not
loaded): L* normalised cross-correlation over 4 rotations x mirror x scales at low resolution, refined
at higher resolution; perspective is not modelled, so low NCC frames are reported and excluded from
the summary (threshold fixed below before any result was seen).

usage: eval_blueneg.py selection.json DATA_ROOT CONV_DIR out.json
  CONV_DIR/<variant>/<stem>.tif for variant in VARIANTS
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

from filmcolor.colorimetry import D50, xyz_to_lab
from filmcolor.compare import block_mean, decompose, load_lab, metrics, ncc_map, resize_bilinear

VARIANTS = ("default", "auto")
NCC_MIN = 0.7
SRGB_D50 = np.array([[0.4360747, 0.3850649, 0.1430804],
                     [0.2225045, 0.7168786, 0.0606169],
                     [0.0139322, 0.0971045, 0.7141733]])


def print_lab(path):
    a = tifffile.imread(path).astype(np.float64)[..., :3] / 65535
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return xyz_to_lab(lin @ SRGB_D50.T, white=D50).astype(np.float32)


def orient(img, k, flip):
    return np.rot90(img[:, ::-1] if flip else img, k)


def search(a_l, r_l, f, cands):
    """Best (k, flip, scale, offset, ncc) over candidate (k, flip, scale) at block factor f."""
    a = block_mean(a_l, f)
    best = None
    for k, flip, s in cands:
        r = block_mean(orient(r_l, k, flip), f)
        shape = (int(round(r.shape[0] * s)), int(round(r.shape[1] * s)))
        if min(shape) < 8 or shape[0] > a.shape[0] or shape[1] > a.shape[1]:
            continue
        m = ncc_map(a, resize_bilinear(r, shape))
        i = np.unravel_index(np.argmax(m), m.shape)
        if best is None or m[i] > best[4]:
            best = (k, flip, float(s), (int(i[0]), int(i[1])), float(m[i]))
    return best


def register(ours_l, ref_l):
    coarse = [(k, fl, s) for k in range(4) for fl in (False, True) for s in np.arange(0.30, 1.21, 0.02)]
    k, fl, s, _, _ = search(ours_l, ref_l, 8, coarse)
    fine = [(k, fl, x) for x in np.arange(s - 0.03, s + 0.0301, 0.004)]
    k, fl, s, off, ncc = search(ours_l, ref_l, 2, fine)
    return dict(rot90=k, mirror=fl, scale=s, offset=off, ncc=ncc, factor=2)


def pair(ours_lab, ref_lab, reg):
    f = reg["factor"]
    a = block_mean(ours_lab, f)
    r = block_mean(orient(ref_lab, reg["rot90"], reg["mirror"]), f)
    shape = (int(round(r.shape[0] * reg["scale"])), int(round(r.shape[1] * reg["scale"])))
    r = resize_bilinear(r, shape)
    y, x = reg["offset"]
    return a[y : y + shape[0], x : x + shape[1]], r


def main():
    sel, root, conv, out = json.load(open(sys.argv[1])), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    variants = sys.argv[5].split(",") if len(sys.argv) > 5 else VARIANTS  # first one is used to register
    rows = []
    for fr in sel["frames"]:
        stem = fr["filename"]
        ref = print_lab(root / fr["printed_path"])
        row = dict(frame=stem, roll=fr["roll_id"], film=fr["film_type"], scene=fr["scene_property"])
        reg = None
        for v in variants:
            tif = conv / v / f"{stem}.tif"
            ours = load_lab(tif, factor=4)
            if reg is None:  # register once, on the default output (same geometry for every variant)
                reg = register(ours[..., 0], ref[..., 0])
                row["registration"] = reg
            a, r = pair(ours, ref, reg)
            sc = json.load(open(tif.with_suffix(".json")))
            # the in-sample per-frame upper bound on a 4x coarser grid (about 50k pixels; full grid ~2 min)
            row[v] = dict(metrics=metrics(a, r), decompose=decompose(block_mean(a, 4), block_mean(r, 4)),
                          qc=sc["qc"]["status"],
                          auto=sc.get("auto", {}).get("status"),
                          base_warnings=sc["base"].get("estimate", {}).get("warnings"))
        print(stem, "ncc %.3f" % reg["ncc"], "rot", reg["rot90"], "mirror", reg["mirror"], "scale %.3f" % reg["scale"],
              " ".join(f"{v} dE {row[v]['metrics']['de2000_median']:.2f}" for v in variants), flush=True)
        rows.append(row)
    json.dump(dict(assumptions=__doc__.split("usage:")[0].strip(), ncc_min=NCC_MIN, frames=rows), open(out, "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
