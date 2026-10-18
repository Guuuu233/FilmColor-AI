"""Base-inversion diagnosis, step 2: linear scan caches for the diagnosis frames (experiments/s3_diag).

Per frame, under $FILMCOLOR_DATA_ROOT/s3_diag/T4/:
  <stem>_T4.npy  demosaiced linear camera RGB of the scan (rawpy_io.demosaic_linear, the signal pipeline.positive
                 inverts), 4x4 block mean, full frame, float32
  <stem>.json    image region and mask as in pipeline.positive (central default, unsaturated), full-resolution
                 percentiles of L = -log10(max(T, EPS)) per channel over that mask, and a replay check: the
                 cached S3 white point must equal the 99.9th percentile of L plus log10(cached base), because a
                 film base change is a per-channel constant in density (d_hi(b) = pct(L) + log10 b)
Every variant of the diagnosis is then computed from T4 with the same block-mean grid, so variants are compared
with each other on equal terms (T is averaged before inversion; the S3 P8 cache averages after it, the
difference is reported by diag_variants.py).

  diag_build_T.py FRAMES_JSON [--t8] (ID [ID ...] | --roll ROLL)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import rawpy

sys.path.insert(0, str(Path(__file__).parent))
from diag_common import camera_to_srgb  # noqa: E402

from filmcolor.compare import block_mean  # noqa: E402
from filmcolor.frozen import check_source, save_cache
from filmcolor.invert import EPS, central_region, region_mask
from filmcolor.paths import data_dir
from filmcolor.raw import saturation_mask
from filmcolor.rawpy_io import demosaic_linear, load_raw
from filmcolor.scanner import load_scanner

ROLL_BASE = Path(os.environ.get("DIAG_ROLL_BASE", "experiments/s3_diag/roll_base.json"))  # later rounds: own file
PCTS = (0.01, 0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 99.99)


def build(f, out_dir, store="T4"):
    t0 = time.time()
    raw_sha = check_source(f["raw"])
    if raw_sha != f["raw_sha256"]:
        raise SystemExit(f"{f['id']}: RAW differs from the frame list")
    rec = json.loads((data_dir("s3_ml", "cache", f["domain"]) / f"{f['stem']}.json").read_text())
    scanner = f["domain"] == "blueneg"  # linear scanner DNG: the decoded image is the signal (pipeline._scanner_input)
    if scanner:
        sfr = load_scanner(f["raw"])
        sat = sfr.sat
    else:
        fr = load_raw(f["raw"])
        sat = saturation_mask(fr)
        del fr
    region = central_region(sat.shape)
    if list(region) != rec["positive"]["image_region"]:
        raise SystemExit(f"{f['id']}: image region {region} differs from the S3 cache")
    in_region = region_mask(sat.shape, region)
    mask = ~sat & in_region
    rgb = sfr.rgb if scanner else demosaic_linear(f["raw"])
    Tm = rgb[mask]
    L = -np.log10(np.maximum(Tm, EPS))
    pct = {str(p): [float(x) for x in np.percentile(L, p, axis=0)] for p in PCTS}
    del L
    # white points of the camera-matrix variant (matrix on T / b before the density), which is not a per-channel
    # shift of L, at full resolution for the frame's own base and the roll base (camera RAW only)
    C, mct = None, {}
    if not scanner:
        with rawpy.imread(f["raw"]) as r:
            C = camera_to_srgb(np.asarray(r.rgb_xyz_matrix)[:3])
    b_roll = json.loads(ROLL_BASE.read_text())["rolls"][f["roll"]]["frames"][f["stem"]]["base_roll"]
    for name, b in (("B0", rec["positive"]["inversion"]["base"]), ("B1c", b_roll)) if not scanner else ():
        x = (Tm / np.asarray(b, np.float32)) @ C.T.astype(np.float32)
        mct[name] = dict(base=list(b), d_hi=[float(v) for v in np.percentile(-np.log10(np.maximum(x, EPS)), 99.9, 0)],
                         nonpositive_frac=float((x <= 0).any(-1).mean()))
        del x
    del Tm
    base = np.array(rec["positive"]["inversion"]["base"])
    d_hi_replay = np.array(pct["99.9"]) + np.log10(base)
    d_hi_cache = np.array(rec["positive"]["inversion"]["d_hi"])
    T4 = block_mean(rgb, 4).astype(np.float32)
    sat4 = block_mean(sat.astype(np.float32), 4)
    del rgb
    if scanner:
        del sfr
    # the w4 white point (diag_variants.white_T4): 4x4 block means, central region of the 1/4 grid, no saturated sample
    t, l, h, w = central_region(T4.shape[:2])
    L4 = -np.log10(np.maximum(T4[t : t + h, l : l + w][sat4[t : t + h, l : l + w] == 0], EPS))
    pct_b4 = {str(p): [float(x) for x in np.percentile(L4, p, axis=0)] for p in (0.1, 1, 50, 99, 99.9)}
    if store == "T4":
        save_cache(out_dir / f"{f['stem']}_T4.npy", T4, [raw_sha])
        save_cache(out_dir / f"{f['stem']}_sat4.npy", sat4.astype(np.float32), [raw_sha])
    else:
        save_cache(out_dir / f"{f['stem']}_T8.npy", block_mean(T4, 2).astype(np.float32), [raw_sha])
    out = dict(id=f["id"], roll=f["roll"], raw=f["raw"], raw_sha256=raw_sha, shape=list(sat.shape),
               image_region=list(region), mask_pixels=int(mask.sum()), saturated_in_region=float(sat[in_region].mean()),
               L_percentiles=pct, base_cache=base.tolist(), d_hi_cache=d_hi_cache.tolist(),
               d_hi_replay=d_hi_replay.tolist(), d_hi_replay_max_abs_err=float(np.abs(d_hi_replay - d_hi_cache).max()),
               camera_to_srgb=None if C is None else C.tolist(), mct_white=mct, L_percentiles_b4=pct_b4,
               T4_shape=list(T4.shape), stored=store, seconds=round(time.time() - t0, 1))
    (out_dir / f"{f['stem']}.json").write_text(json.dumps(out, indent=1))
    return out


def main(frames_json, *ids):
    """``--t8`` first: store the 1/8 full frame only (roll-wide checks) under s3_diag/T8; ``--roll ROLL`` instead
    of IDs: every development frame of that roll with an S3 cache record."""
    store = "T4"
    if ids and ids[0] == "--t8":
        store, ids = "T8", ids[1:]
    frames = {f["id"]: f for f in json.load(open(frames_json))["frames"]}
    if ids and ids[0] == "--roll":
        ids = [i for i, f in frames.items() if f["roll"] == ids[1]
               and (data_dir("s3_ml", "cache", f["domain"]) / f"{f['stem']}.json").exists()]
    out_dir = data_dir("s3_diag", store)
    for i in ids:
        r = build(frames[i], out_dir, store)
        print(i, "T4", r["T4_shape"], "d_hi replay err %.2e" % r["d_hi_replay_max_abs_err"], f"{r['seconds']} s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
