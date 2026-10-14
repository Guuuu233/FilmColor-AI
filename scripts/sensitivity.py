"""Sensitivity of the reference fitter (research tool) on the 12 NLP samples. Needs the P4 caches written
by fit_nlp.py.

  sensitivity.py FIT_DIR COMPARE_JSON OUT_DIR [SET]

SET s2-3 (default): penalty strength, parameter scales, contrast bound and sub-windows, without the
highlight constraint (as run in S2-3). SET s2-6: penalty strength with the highlight constraint, and the
highlight weight itself.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from filmcolor.compare import Alignment, aligned_pair, load_lab  # noqa: E402
from filmcolor.fit import SCALES, fit_reference, param_conditioning  # noqa: E402
from filmcolor.render import RenderParams, experimental_bounds  # noqa: E402

MARGIN = 0.03
LAMBDAS = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
BASE_LAM = 0.05
WINDOW = 0.85
HIGHLIGHT_WEIGHTS = (0.0, 3.0, 10.0, 30.0)


def configs(which="s2-3"):
    if which == "s2-6":
        c = {f"lam={lam} hw=10.0": dict(lam=lam, highlight_weight=10.0) for lam in LAMBDAS}
        c.update({f"lam={BASE_LAM} hw={hw}": dict(lam=BASE_LAM, highlight_weight=hw) for hw in HIGHLIGHT_WEIGHTS
                  if hw != 10.0})
        return c
    if which != "s2-3":
        raise ValueError(f"unknown config set {which!r}")
    return {name: cfg | dict(highlight_weight=0.0) for name, cfg in _configs_s2_3().items()}


def _configs_s2_3():
    c = {f"lam={lam}": dict(lam=lam) for lam in LAMBDAS}
    for k in ("ev", "midtone", "contrast"):
        for f in (0.5, 2.0):
            c[f"scale {k}x{f}"] = dict(lam=BASE_LAM, scales={k: SCALES[k] * f})
    for hi in (2.5, 3.0):
        for lam in (0.0, BASE_LAM):
            c[f"contrast<= {hi} lam={lam}"] = dict(lam=lam, bounds=dict(contrast=(0.5, hi)))
    for w in range(3):
        c[f"window{w}"] = dict(lam=BASE_LAM, window=w, n_starts=3)
    return c


def margin_crop(a):
    h, w = a.shape[:2]
    my, mx = int(h * MARGIN), int(w * MARGIN)
    return a[my : h - my, mx : w - mx]


def task(args):
    stem, name, cfg, cache = args
    P = np.load(cache / f"{stem}_P8.npy")
    R = np.load(cache / f"{stem}_R8.npy")
    window = None
    if "window" in cfg:
        rng = np.random.default_rng(1000 + cfg["window"])
        h, w = P.shape[:2]
        wh, ww = int(h * WINDOW), int(w * WINDOW)
        y, x = int(rng.integers(0, h - wh + 1)), int(rng.integers(0, w - ww + 1))
        P, R = P[y : y + wh, x : x + ww], R[y : y + wh, x : x + ww]
        window = [y, x, wh, ww]
    with experimental_bounds(**cfg.get("bounds", {})):
        r = fit_reference(P.reshape(-1, 3), R.reshape(-1, 3), "7", lam=cfg["lam"], scales=cfg.get("scales"),
                          n_starts=cfg.get("n_starts", 5), highlight_weight=cfg["highlight_weight"])
        cond = param_conditioning(P.reshape(-1, 3), RenderParams(**r["params"])) if name.startswith("lam=") else None
    r.pop("runs")
    return dict(stem=stem, config=name, cfg={k: v for k, v in cfg.items()}, window=window, fit=r, conditioning=cond)


def main(fit_dir, compare_json, out_dir, which="s2-3"):
    fit_dir, out_dir = Path(fit_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = fit_dir / "cache"
    comp = json.loads(Path(compare_json).read_text())
    stems = sorted(comp)
    for stem in stems:  # aligned, margin-cropped 2-D inputs, written once
        if not (cache / f"{stem}_R8.npy").exists():
            P8, R8 = aligned_pair(np.load(cache / f"{stem}_P4.npy"), load_lab(comp[stem]["reference"], 4),
                                  Alignment(**comp[stem]["alignment"]))
            np.save(cache / f"{stem}_P8.npy", margin_crop(P8).astype(np.float32))
            np.save(cache / f"{stem}_R8.npy", margin_crop(R8))
    jobs = [(stem, name, cfg, cache) for name, cfg in configs(which).items() for stem in stems]
    workers = max(1, min(8, (os.cpu_count() or 2) - 2))
    print(len(jobs), "fits on", workers, "workers", flush=True)
    results = []
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(task, jobs)):
            results.append(r)
            if (i + 1) % 24 == 0:
                print(i + 1, "done", flush=True)
    (out_dir / "sensitivity.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
