"""Post-hoc diagnostic (uses the references; not part of any estimator): how much of the gap between the
fixed look (global_look_v1) and the per-frame reference fits is closed if only one parameter group is
set perfectly per frame, the others staying at the look's values.

  noref_group_oracle.py OUT_JSON [WORKERS]

Objective as fit_config_v1 without the label regulariser (mean pseudo-Huber dE2000 + 30 * highlight depth
+ 10 * clipped area), L-BFGS-B from the look plus two random starts, on the reference-aligned P8 / R8 grids.
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from noref_eval import SETS, measure  # noqa: E402

from filmcolor.colorimetry import delta_e2000  # noqa: E402
from filmcolor.fit import FD_STEP, FIT_CONFIG_V1, HUBER_DELTA, MAXITER, SCALES, clip_area, highlight_penalty, \
    reference_clipped, render_lab  # noqa: E402
from filmcolor.noref import load_look  # noqa: E402
from filmcolor.render import BOUNDS, RenderParams  # noqa: E402

GROUPS = dict(brightness=("ev", "midtone"), wb=("temp", "tint"), tone=("contrast", "shoulder"), sat=("sat",),
              all=("ev", "temp", "tint", "sat", "contrast", "midtone", "shoulder"))
N_SAMPLE, N_STARTS = 20000, 3


def fit_group(P, R, base, free, seed=0):
    P2, R2 = P.reshape(-1, 3).astype(np.float32), R.reshape(-1, 3)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(P2), min(N_SAMPLE, len(P2)), replace=False))
    Ps, Rs, allowed = P2[idx][None], R2[idx], reference_clipped(R2)[idx]
    d0, sc = np.array([base[k] for k in free]), np.array([SCALES[k] for k in free])
    blo, bhi = np.array([BOUNDS[k][0] for k in free]), np.array([BOUNDS[k][1] for k in free])
    lo, hi = (blo - d0) / sc, (bhi - d0) / sc

    def params_of(u):
        return RenderParams(**dict(base, **{k: float(v) for k, v in zip(free, np.clip(d0 + sc * u, blo, bhi))}))

    def objective(u):
        lab, _, pre = render_lab(Ps, params_of(u), with_pre_clip=True)
        de = delta_e2000(lab[0], Rs)
        return float((np.sqrt(de ** 2 + HUBER_DELTA ** 2) - HUBER_DELTA).mean()
                     + FIT_CONFIG_V1["highlight_weight"] * highlight_penalty(pre[0], allowed)
                     + FIT_CONFIG_V1["area_weight"] * clip_area(pre[0], allowed))

    best = None
    for u0 in [np.zeros(len(free))] + [rng.uniform(lo, hi) for _ in range(N_STARTS - 1)]:
        r = minimize(objective, u0, method="L-BFGS-B", bounds=list(zip(lo, hi)), options=dict(eps=FD_STEP, maxiter=MAXITER))
        if best is None or r.fun < best.fun:
            best = r
    return params_of(best.x).to_dict()


def job(args):
    which, stem, group = args
    cache = SETS[which]["eval_cache"]
    P, R = np.load(f"{cache}/{stem}_P8.npy"), np.load(f"{cache}/{stem}_R8.npy")
    look = load_look()["params"]
    params = look if group == "look" else fit_group(P, R, look, GROUPS[group])
    return dict(set=which, stem=stem, group=group, params=params, eval=measure(P, R, params))


def main(out, workers=8):
    stems = [(w, p.name[: -len("_R8.npy")]) for w in SETS for p in sorted(Path(SETS[w]["eval_cache"]).glob("*_R8.npy"))]
    jobs = [(w, s, g) for w, s in stems for g in ("look", *GROUPS)]
    with Pool(int(workers)) as pool:
        res = pool.map(job, jobs, chunksize=1)
    json.dump(dict(look=load_look()["id"], groups=GROUPS, n_sample=N_SAMPLE, n_starts=N_STARTS,
                   objective="fit_config_v1 without lam", results=res), open(out, "w"), indent=1)
    print("done", len(res))


if __name__ == "__main__":
    main(*sys.argv[1:])
