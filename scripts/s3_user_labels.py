"""Exploratory (not in the S3 design): the same negatives graded by two people - the shop's colourist (ext_own
prints) and the user (own Lightroom edits in the photo folder) - fitted with the same P4 labels.

Questions, written before any result: (1) per parameter, is user minus shop a stable offset (|mean| larger than
the spread over frames) or frame-specific; (2) how far is a frame rendered with the shop-fitted P4 parameters
from the user's print (what a model trained on the shop's style would miss); (3) can P4 express the user's
grades (gap to the 7-parameter ceiling ub7).

Per frame: the user's print aligned to the cached full-frame X8 on the 1/8 grid (L* of the default rendering,
as s3_build_cache.py, but aligned at 1/8 because only X8 is cached), 3 % margin; P4 label, ub7 ceiling, fixed
look; and the shop P4 parameters rendered against the user's print. Every read goes through filmcolor.frozen.

  s3_user_labels.py FRAMES_JSON SHOP_LABEL_DIR PHOTO_ROOT OUT_DIR [WORKERS]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from filmcolor.colorimetry import delta_e2000  # noqa: E402
from filmcolor.compare import align, aligned_pair, load_lab  # noqa: E402
from filmcolor.fit import FIT_CONFIG_V1, fit_reference, fit_reference_s3, render_lab  # noqa: E402
from filmcolor.frozen import check_source, load_cache, save_cache  # noqa: E402
from filmcolor.paths import data_dir  # noqa: E402
from filmcolor.render import RenderParams  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from s3_build_cache import srgb_assumed_lab  # noqa: E402
from sensitivity import margin_crop  # noqa: E402

FOLDERS = ("新疆/#012 5207", "新疆/#007 金200")  # the user's own edits of development rolls (photo audit)
LOOK = json.loads(Path("src/filmcolor/configs/global_look_v1.json").read_text())


def de_of(P, R, params):
    lab, _ = render_lab(P[None], RenderParams(**params))
    de = delta_e2000(lab[0], R)
    return dict(de2000_median=float(np.median(de)), de2000_mean=float(de.mean()))


def label(args):
    f, ref, shop_label, out_dir = args
    ref_sha = check_source(ref)
    cache = data_dir("s3_ml", "cache", f["domain"])
    X8 = load_cache(cache / f"{f['stem']}_X8.npy")
    ours, _ = render_lab(X8, RenderParams())
    try:
        ref8 = load_lab(ref, 8)
        kind = "icc"
    except ValueError:  # no embedded ICC
        ref8, kind = srgb_assumed_lab(ref, 8), "srgb_assumed"
    al = align(ours[..., 0], ref8[..., 0], 1)
    P8, R8 = aligned_pair(X8, ref8, al)
    P8, R8 = margin_crop(P8).astype(np.float32), margin_crop(R8).astype(np.float32)
    ucache = data_dir("s3_ml", "cache_user", f["domain"])
    save_cache(ucache / f"{f['stem']}_P8.npy", P8, [f["raw_sha256"], ref_sha])
    save_cache(ucache / f"{f['stem']}_R8.npy", R8, [f["raw_sha256"], ref_sha])
    P, R = P8.reshape(-1, 3), R8.reshape(-1, 3).astype(np.float64)
    c = FIT_CONFIG_V1
    res = dict(id=f["id"], roll=f["roll"], user_ref=ref, user_ref_sha256=ref_sha, ref_kind=kind, alignment=al.to_dict(),
               fixed_look=de_of(P, R, LOOK["params"]),
               p4=fit_reference_s3(P, R, "p4", LOOK),
               ub7=fit_reference(P, R, "7", lam=0.0, highlight_weight=c["highlight_weight"], area_weight=c["area_weight"]),
               shop_p4_on_user=de_of(P, R, shop_label["p4"]["params"]),
               shop_p4_params=shop_label["p4"]["params"], shop_p4_de=shop_label["p4"]["de2000_median"])
    for k in ("p4", "ub7"):
        res[k].pop("runs")
    out = Path(out_dir) / f"{f['stem']}.json"
    out.write_text(json.dumps(res, indent=1, default=float))
    return f["id"], al.ncc, res["p4"]["de2000_median"], res["shop_p4_on_user"]["de2000_median"]


def main(frames_json, shop_label_dir, photo_root, out_dir, workers="6"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    byname = {f["orig_name"].rsplit(".", 1)[0]: f for f in json.load(open(frames_json))["frames"] if f["domain"] == "ext_own"}
    jobs = []
    for folder in FOLDERS:
        for p in sorted(Path(photo_root, folder).iterdir()):
            m = re.match(r"(DSC_\d+)", p.name)
            f = byname.get(m.group(1)) if m else None
            lab = Path(shop_label_dir) / "ext_own" / f"{f['stem']}.json" if f else None
            if f is None or not lab.exists():  # no development negative, or a pipeline failure (no P)
                print("skip", p.name, flush=True)
                continue
            jobs.append((f, str(p), json.loads(lab.read_text()), out_dir))
    with ProcessPoolExecutor(int(workers)) as ex:
        for fid, ncc, de_user, de_cross in ex.map(label, jobs):
            print(fid, "ncc %.3f" % ncc, "P4 on user %.2f" % de_user, "shop P4 on user %.2f" % de_cross, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
