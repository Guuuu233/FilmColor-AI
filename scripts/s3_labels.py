"""S3 constrained labels (docs/S3_ML_DESIGN.md 2.3) for development frames, from the P8 / R8 caches.

Per frame: fit_config_s3_p4 and fit_config_s3_p3 with theta_fixed from global_look_v1 (kept by the colour-only
check, experiments/s3_ml/GLOBAL_LOOK_C1.md), the 7-parameter fit_config_v1 label, the ceiling of the
7-parameter space (ub7: all free, no regulariser, same highlight constraints; fit_config_v1's lam = 0.05 pulls
all seven towards the engine defaults, so v1 is not an upper bound for P4, whose fixed contrast 2.0 is far from
the default), and the fixed look itself (no fit) as reference point. Every start of the multi-start search is kept for the stability analysis.
Caches are read through filmcolor.frozen (a cache derived from a frozen file is refused).

  s3_labels.py FRAMES_JSON OUT_DIR [WORKERS] [ID ...]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from filmcolor.colorimetry import delta_e2000  # noqa: E402
from filmcolor.fit import (FIT_CONFIG_V1, clip_area, fit_reference, fit_reference_s3, fit_reference_v1,  # noqa: E402
                           reference_clipped, render_lab)
from filmcolor.frozen import load_cache  # noqa: E402
from filmcolor.paths import data_dir  # noqa: E402
from filmcolor.render import RenderParams  # noqa: E402

LOOK_FILE = Path("src/filmcolor/configs/global_look_v1.json")


def fixed_look(P, R, params):
    lab, diag, pre = render_lab(P[None], RenderParams(**params), with_pre_clip=True)
    de = delta_e2000(lab[0], R)
    return dict(params=params, de2000_median=float(np.median(de)), de2000_mean=float(de.mean()),
                clip_area=clip_area(pre[0], reference_clipped(R)), diagnostics=diag)


def label(args):
    f, out_dir = args
    t0 = time.time()
    cache = data_dir("s3_ml", "cache", f["domain"])
    P = load_cache(cache / f"{f['stem']}_P8.npy").reshape(-1, 3)
    R = load_cache(cache / f"{f['stem']}_R8.npy").reshape(-1, 3).astype(np.float64)
    look = json.loads(LOOK_FILE.read_text())
    c = FIT_CONFIG_V1
    res = dict(id=f["id"], domain=f["domain"], roll=f["roll"], experiments=f["experiments"],
               look=dict(id=look["id"], sha256=hashlib.sha256(LOOK_FILE.read_bytes()).hexdigest()),
               cache=json.loads((cache / f"{f['stem']}.json").read_text()) | dict(dir=str(cache)),
               fixed_look=fixed_look(P, R, look["params"]),
               p4=fit_reference_s3(P, R, "p4", look), p3=fit_reference_s3(P, R, "p3", look), v1=fit_reference_v1(P, R),
               ub7=fit_reference(P, R, "7", lam=0.0, highlight_weight=c["highlight_weight"],
                                 area_weight=c["area_weight"]))
    res["seconds"] = round(time.time() - t0, 1)
    out = Path(out_dir) / f["domain"] / f"{f['stem']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    return f["id"], {k: round(res[k]["de2000_median"], 2) for k in ("fixed_look", "p4", "p3", "v1", "ub7")}, res["seconds"]


def main(frames_json, out_dir, workers="2", *ids):
    frames = [f for f in json.load(open(frames_json))["frames"] if not ids or f["id"] in ids]
    with ProcessPoolExecutor(int(workers)) as ex:
        for fid, de, sec in ex.map(label, [(f, out_dir) for f in frames]):
            print(fid, "median dE", de, f"{sec} s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
