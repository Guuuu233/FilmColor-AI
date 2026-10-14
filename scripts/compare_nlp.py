"""Compare our outputs with NLP reference conversions.

  compare_nlp.py OURS_DIR REF_DIR OUT_DIR   (ours: <stem>.tif, reference: <stem>-positive.tif or <stem>.tif)
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

from filmcolor.colorimetry import delta_e2000, lab_to_xyz, xyz_to_srgb_preview
from filmcolor.compare import (_core, align, aligned_pair, apply_ab_affine, decompose, fit_ab_affine,
                              load_lab, metrics, tone_curve)

LOAD_FACTOR, ALIGN_FACTOR = 4, 2  # metrics at 1/8 of full resolution


def preview(lab):
    return xyz_to_srgb_preview(lab_to_xyz(lab))


def main(ours_dir, ref_dir, out_dir):
    ours_dir, ref_dir, out_dir = Path(ours_dir), Path(ref_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results, pairs = {}, {}
    for ours in sorted(ours_dir.glob("*.tif")):
        ref = next((p for p in (ref_dir / f"{ours.stem}-positive.tif", ref_dir / f"{ours.stem}.tif") if p.exists()), None)
        if ref is None:
            continue
        a_lab, r_lab = load_lab(ours, LOAD_FACTOR), load_lab(ref, LOAD_FACTOR)
        al = align(a_lab[..., 0], r_lab[..., 0], ALIGN_FACTOR)
        a, r = aligned_pair(a_lab, r_lab, al)
        m = metrics(a, r)
        d = decompose(a, r)
        pairs[ours.stem] = (a, r)
        results[ours.stem] = dict(ours=str(ours), reference=str(ref), alignment=al.to_dict(), metrics=m, decompose=d)
        print(ours.stem, f"rot {al.rot90} scale {al.scale:.3f} ncc {al.ncc:.3f}",
              f"dE00 med {m['de2000_median']:.1f} p90 {m['de2000_p90']:.1f}",
              f"dL {m['mean_dL']:+.1f} da {m['mean_da']:+.1f} db {m['mean_db']:+.1f} chroma x{m['chroma_ratio'] or 0:.2f}",
              "L50 ours/ref", round(m["L_percentiles"]["image"][2], 1), round(m["L_percentiles"]["reference"][2], 1),
              "| per-image dE after tone/+ab", round(d["tone"], 1), round(d["tone_ab"], 1), flush=True)
        de = np.clip(delta_e2000(a, r) / 20, 0, 1)[..., None] * np.array([1.0, 0.2, 0.2])
        sheet = np.concatenate([preview(a), np.ones((a.shape[0], 6, 3)), preview(r), np.ones((a.shape[0], 6, 3)), de], 1)
        tifffile.imwrite(out_dir / f"{ours.stem}_compare.tif", (sheet * 255 + 0.5).astype(np.uint8))
    shared_loo(pairs, results)
    (out_dir / "compare.json").write_text(json.dumps(results, indent=1))


def shared_loo(pairs, results, n=20000, seed=0):
    """Fixed corrections shared across images, leave-one-out: tone curve and a*b* affine fitted on
    the other colour images, applied to the held-out one. B&W frames (reference chroma ~0) skipped."""
    rng = np.random.default_rng(seed)
    knots = np.linspace(0, 100, 101)
    data = {}
    for stem, (a, r) in pairs.items():
        a, r = _core(a, 0.03), _core(r, 0.03)
        if np.median(np.hypot(r[:, 1], r[:, 2])) < 2:
            continue
        i = rng.choice(len(a), min(n, len(a)), replace=False)
        data[stem] = (a[i], r[i])
    for stem, (a, r) in data.items():
        others = [v for k, v in data.items() if k != stem]
        curve = np.mean([tone_curve(oa[:, 0], orr[:, 0], knots) for oa, orr in others], 0)
        oa = np.concatenate([o[0] for o in others])
        orr = np.concatenate([o[1] for o in others])
        oa_t = oa.copy()
        oa_t[:, 0] = np.interp(oa[:, 0], knots, curve)
        coef = fit_ab_affine(oa_t[:, 1:], orr[:, 1:])
        t = a.copy()
        t[:, 0] = np.interp(a[:, 0], knots, curve)
        de_tone = float(np.median(delta_e2000(t, r)))
        t[:, 1:] = apply_ab_affine(t[:, 1:], coef)
        de_all = float(np.median(delta_e2000(t, r)))
        results[stem]["shared_loo"] = dict(tone=de_tone, tone_ab=de_all, ab_coef=coef.tolist())
        print(stem, f"shared LOO dE after tone {de_tone:.1f} +ab {de_all:.1f}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
