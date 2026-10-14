"""Fit render parameters to NLP references in three modes (6, 7, 7reg); research tool, sees the reference.

  fit_nlp.py SIDECAR_DIR REF_DIR COMPARE_JSON OUT_DIR [STEM ...]

SIDECAR_DIR: Sprint 1 outputs (<stem>.json, for base / white point); COMPARE_JSON: S1-9 alignments.
"""
import json
import sys
from pathlib import Path

import numpy as np

from filmcolor.colorimetry import delta_e2000
from filmcolor.compare import Alignment, aligned_pair, block_mean, load_lab
from filmcolor.fit import MODES, fit_reference, param_conditioning, render_lab
from filmcolor.invert import InversionParams, density, invert
from filmcolor.render import RenderParams
from filmcolor.rawpy_io import demosaic_linear

MARGIN = 0.03  # as in compare.metrics


def core(a):
    h, w = a.shape[:2]
    my, mx = int(h * MARGIN), int(w * MARGIN)
    return a[my : h - my, mx : w - mx].reshape(-1, 3)


def aligned_inputs(stem, sidecar, ref_path, al, cache_dir):
    """Scene-linear positive (Sprint 1 inversion) and reference Lab on the S1-9 aligned 1/8 grid."""
    cache = cache_dir / f"{stem}_P4.npy"
    if cache.exists():
        P4 = np.load(cache)
    else:
        p = sidecar["inversion"]["params"]
        rgb = demosaic_linear(sidecar["input"]["path"])
        P = invert(density(rgb, p["base"]), InversionParams(base=p["base"], d_lo=p["d_lo"], d_hi=p["d_hi"],
                                                            gamma=p["gamma"], exposure=p["exposure"]))
        P4 = block_mean(P * np.float32(p["exposure"]), 4).astype(np.float32)
        np.save(cache, P4)
    P8, R8 = aligned_pair(P4, load_lab(ref_path, 4), Alignment(**al))
    return core(P8).astype(np.float32), core(R8)


def main(sidecar_dir, ref_dir, compare_json, out_dir, *stems):
    sidecar_dir, ref_dir, out_dir = Path(sidecar_dir), Path(ref_dir), Path(out_dir)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    comp = json.loads(Path(compare_json).read_text())
    for stem in stems or sorted(comp):
        sc = json.loads((sidecar_dir / f"{stem}.json").read_text())
        P, R = aligned_inputs(stem, sc, Path(comp[stem]["reference"]), comp[stem]["alignment"], cache_dir)
        lab0, diag0 = render_lab(P[None], RenderParams())
        de0 = delta_e2000(lab0[0], R)
        res = dict(stem=stem, n_pixels=len(P), reference=comp[stem]["reference"],
                   default=dict(de2000_median=float(np.median(de0)), de2000_p90=float(np.percentile(de0, 90)), diagnostics=diag0),
                   conditioning_default=param_conditioning(P, RenderParams()), fits={})
        for mode in MODES:
            res["fits"][mode] = fit_reference(P, R, mode)
        res["conditioning_fit7"] = param_conditioning(P, RenderParams(**res["fits"]["7"]["params"]))
        (out_dir / f"{stem}.json").write_text(json.dumps(res, indent=1))
        f = res["fits"]
        print(stem, f"default {res['default']['de2000_median']:.1f}",
              " ".join(f"{m}:{f[m]['de2000_median']:.2f}/near{f[m]['n_near_optimal']}/bound{len(f[m]['at_bound'])}" for m in MODES),
              f"cos(ev,mid) default {res['conditioning_default']['cosine']['ev/midtone']:.2f} fit7 {res['conditioning_fit7']['cosine']['ev/midtone']:.2f}",
              flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
