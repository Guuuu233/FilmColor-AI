"""Global base look for the reference-free estimator: one set of the 7 render parameters fitted to the pooled
development frames (the parameter-space version of the fixed "shared correction" baseline), with a cross-set
check (fit on one reference set, evaluate on the other).

  noref_global.py OUT_JSON

Uses the reference-aligned P8 / R8 caches; equal pixel count per frame; anomalous frames left out.
Fitting uses fit_reference in mode 7 without the label regulariser (lam = 0) but with the fit_config_v1
highlight constraints. This is development-set tuning of global constants, not per-image solving.
"""
import json
import statistics as st
import sys

import numpy as np

from filmcolor.colorimetry import delta_e2000
from filmcolor.fit import FIT_CONFIG_V1, fit_reference, render_lab
from filmcolor.render import RenderParams

SETS = dict(nlp=("experiments/s2_fit/cache", ()), shop=("experiments/s2_shop/fit/cache", ("DSC_3822", "DSC_3858")))
STEMS = dict(nlp=[f"_DSF{n}" for n in (8434, 8948, 8951, 8983, 9228, 9229, 9231, 9235, 9237, 9238, 9239, 9243)],
             shop=[f"DSC_{n}" for n in (3823, 3825, 3827, 3828, 3829, 3831, 3833, 3835, 3837, 3838, 3839, 3840, 3843, 3844,
                                        3846, 3847, 3849, 3853, 3855, 3856, 3859)])
PER_FRAME = 3000


def load(name):
    cache = SETS[name][0]
    return {s: (np.load(f"{cache}/{s}_P8.npy"), np.load(f"{cache}/{s}_R8.npy")) for s in STEMS[name]}


def pooled(frames, seed=0):
    rng = np.random.default_rng(seed)
    P, R = [], []
    for p8, r8 in frames.values():
        p, r = p8.reshape(-1, 3), r8.reshape(-1, 3)
        i = rng.choice(len(p), PER_FRAME, replace=False)
        P.append(p[i]), R.append(r[i])
    return np.concatenate(P), np.concatenate(R)


def evaluate(frames, params):
    return {s: float(np.median(delta_e2000(render_lab(p8, RenderParams(**params))[0].reshape(-1, 3), r8.reshape(-1, 3))))
            for s, (p8, r8) in frames.items()}


def main(out):
    data = {n: load(n) for n in SETS}
    c = FIT_CONFIG_V1
    fits = {}
    for train in ("nlp", "shop", "both"):
        frames = {**data["nlp"], **data["shop"]} if train == "both" else data[train]
        P, R = pooled(frames)
        fits[train] = fit_reference(P, R, "7", lam=0.0, n_sample=len(P), highlight_weight=c["highlight_weight"],
                                    area_weight=c["area_weight"])["params"]
    res = dict(fits=fits, eval={})
    for train, params in [("default", RenderParams().to_dict()), *fits.items()]:
        res["eval"][train] = {n: evaluate(data[n], params) for n in SETS}
        print(f"{train:8s}", " ".join(f"{k}={v:.2f}" for k, v in params.items() if k != "matrix"), "| median dE",
              " ".join(f"{n} {st.median(res['eval'][train][n].values()):.2f}" for n in SETS), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
