"""Base-inversion diagnosis, step 4: do the candidate changes hold on whole development rolls (experiments/s3_diag)?

Every development frame of the three diagnosis rolls with an S3 cache (dsc6161, dsc8242, Gold 200), against the
shop print and, where it exists, the user's edit. Inversions (no reference is read to build any of them; the roll
base uses only the negatives of the roll):
  B0     the engine (per-frame automatic base, black point = base, full-resolution white point)
  B0w4   white point on 4x4 block means (diag_build_T.py L_percentiles_b4)
  B1     roll base (diag_roll_base.py), black point = base
  B1w4   both changes
Per inversion: fixed look (global_look_v1), P4 label (fit_config_s3_p4), ub7 ceiling; metrics as diag_variants.py.
On B1 also the diagnostic ub7 + per-channel contrast fit (diag_common.fit_extra "chan_gamma"; sees the reference),
to see whether the contrast factors the references ask for are stable within a roll.
Reads the T8 caches (diag_build_T.py --t8); the shop grid is the full-frame 1/8 grid cropped as the S3 cache.

  diag_rollcheck.py FRAMES_JSON OUT_DIR [WORKERS] ROLL [ROLL ...]
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from diag_common import LOOK, evaluate, fit_extra, grid_T, positive_T  # noqa: E402

import numpy as np  # noqa: E402

from filmcolor.fit import FIT_CONFIG_V1, fit_reference, fit_reference_s3  # noqa: E402
from filmcolor.frozen import check_source, load_cache  # noqa: E402
from filmcolor.invert import EPS  # noqa: E402
from filmcolor.paths import data_dir  # noqa: E402

ROLL_BASE = Path("experiments/s3_diag/roll_base.json")


def task(args):
    f, ref_kind, out_dir = args
    t0 = time.time()
    stem = f["stem"]
    check_source(f["raw"])
    trec = json.loads((data_dir("s3_diag", "T8") / f"{stem}.json").read_text())
    T8 = load_cache(data_dir("s3_diag", "T8") / f"{stem}_T8.npy")
    crec = json.loads((data_dir("s3_ml", "cache", f["domain"]) / f"{stem}.json").read_text())
    if ref_kind == "shop":
        cdir, al = data_dir("s3_ml", "cache", f["domain"]), dict(crec["alignment"])
        if al["factor"] != 2:
            raise SystemExit(f"{f['id']}: expected the S3 alignment factor 2 (P4 grid)")
        al["factor"] = 1  # T8 is the 1/8 grid the factor-2 alignment works on
    else:
        cdir = data_dir("s3_ml", "cache_user", f["domain"])
        al = json.loads(Path(f"experiments/s3_ml/user_labels/{stem}.json").read_text())["alignment"]
    R = load_cache(cdir / f"{stem}_R8.npy").astype(np.float64)
    P_cached = load_cache(cdir / f"{stem}_P8.npy")
    inv0 = crec["positive"]["inversion"]
    Tg, grid_info = grid_T(T8, al, R.shape, P_cached, lambda T: positive_T(T, inv0["base"], inv0["d_lo"], inv0["d_hi"]))
    T, R = Tg.reshape(-1, 3).astype(np.float64), R.reshape(-1, 3)
    LG = -np.log10(np.maximum(T[:, 1], EPS))
    b0 = np.array(trec["base_cache"])
    broll = np.array(json.loads(ROLL_BASE.read_text())["rolls"][f["roll"]]["frames"][stem]["base_roll"])
    full, b4 = np.array(trec["L_percentiles"]["99.9"]), np.array(trec["L_percentiles_b4"]["99.9"])
    inv = {"B0": (b0, full), "B0w4": (b0, b4), "B1": (broll, full), "B1w4": (broll, b4)}
    out = dict(id=f["id"], roll=f["roll"], ref_kind=ref_kind, grid=grid_info, variants={},
               own_minus_roll_density=np.log10(broll / b0).tolist())
    cfg = FIT_CONFIG_V1
    for name, (b, pct) in inv.items():
        d_hi = pct + np.log10(b)
        P = positive_T(T, b, (0, 0, 0), d_hi)
        res = dict(base=b.tolist(), d_hi=d_hi.tolist(),
                   below_black=[float(x) for x in (P < 0).mean(0)])
        res["fixed_look"], _ = evaluate(P, R, LOOK["params"], LG)
        p4 = fit_reference_s3(P, R, "p4", LOOK)
        ub7 = fit_reference(P, R, "7", lam=0.0, highlight_weight=cfg["highlight_weight"], area_weight=cfg["area_weight"])
        res["p4"], _ = evaluate(P, R, p4["params"], LG)
        res["ub7"], _ = evaluate(P, R, ub7["params"], LG)
        res["p4"]["params"], res["ub7"]["params"] = p4["params"], ub7["params"]
        if name == "B1":  # diagnostic: per-channel contrast the reference asks for (stable per roll or per frame?)
            fx, Px = fit_extra("chan_gamma", T, dict(base=b, d_lo=(0, 0, 0), d_hi=d_hi), R, ub7["params"])
            m, _ = evaluate(Px, R, fx["params"], LG)
            res["ub7_chan_gamma"] = m | {k: v for k, v in fx.items() if k != "params"} | dict(params=fx["params"])
        out["variants"][name] = res
    out["seconds"] = round(time.time() - t0, 1)
    (Path(out_dir) / f"{stem}_{ref_kind}.json").write_text(json.dumps(out, indent=1, default=float))
    v = out["variants"]
    return f["id"], ref_kind, " ".join("%s p4 %.2f ub7 %.2f" % (k, v[k]["p4"]["de2000_median"], v[k]["ub7"]["de2000_median"])
                                       for k in v), out["seconds"]


def main(frames_json, out_dir, workers, *rolls):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    jobs = []
    for f in json.load(open(frames_json))["frames"]:
        if f["roll"] not in rolls or not (data_dir("s3_diag", "T8") / f"{f['stem']}.json").exists():
            continue
        kinds = ["shop"] + (["user"] if Path(f"experiments/s3_ml/user_labels/{f['stem']}.json").exists() else [])
        jobs += [(f, k, out_dir) for k in kinds if not (Path(out_dir) / f"{f['stem']}_{k}.json").exists()]  # resumable
    print(len(jobs), "tasks to do", flush=True)
    with ProcessPoolExecutor(int(workers)) as ex:
        for r in ex.map(task, jobs):
            print(*r, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
