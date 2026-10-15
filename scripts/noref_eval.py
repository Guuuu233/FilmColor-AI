"""Run the reference-free estimator (filmcolor.noref) on a development set, then evaluate against references.

  noref_eval.py SET OUT_DIR [key=value ...]     SET: nlp | shop; key=value overrides NOREF_V1 entries

Estimation sees only the frame's positive at 1/4 resolution (computed from the sidecar, cached) and, for the
shop roll, the frame_stats of the other frames of the roll. NLP scans include the holder, so statistics come
from the sidecar's image region; the shop roll is borderless (SOURCES.md), so the whole frame is used.
Evaluation afterwards uses the reference-aligned, margin-cropped grids cached by sensitivity.py (P8 / R8)
and the fit_config_v1 labels (experiments/s2_hl_area, area weight 10) as the seen-the-answer upper bound.
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import tifffile

from filmcolor.colorimetry import delta_e2000, lab_to_xyz, xyz_to_srgb_preview
from filmcolor.compare import block_mean, highlight_texture
from filmcolor.fit import reference_clipped, render_lab
from filmcolor.invert import InversionParams, density, invert
from filmcolor.noref import NOREF_V1, estimate_params, frame_stats
from filmcolor.rawpy_io import demosaic_linear
from filmcolor.render import RenderParams, render

SETS = dict(
    nlp=dict(sidecars="experiments/s1_out", p4="experiments/s2_fit/cache", eval_cache="experiments/s2_fit/cache",
             v1="experiments/s2_hl_area/nlp/sensitivity.json", borderless=False, roll=False),
    shop=dict(sidecars="experiments/s2_shop/out", p4="experiments/s2_shop/noref_p4", eval_cache="experiments/s2_shop/fit/cache",
              v1="experiments/s2_hl_area/shop/sensitivity.json", borderless=True, roll=True),
)
V1_CONFIG = "lam=0.05 hw=30.0 area=10.0"


def positive4(sidecar, stem, *caches):
    """exposure * P block-averaged by 4 (as fit_nlp.py), from the first cache that has it, else computed."""
    for c in caches:
        if (Path(c) / f"{stem}_P4.npy").exists():
            return np.load(Path(c) / f"{stem}_P4.npy")
    p = sidecar["inversion"]["params"]
    P = invert(density(demosaic_linear(sidecar["input"]["path"]), p["base"]),
               InversionParams(base=p["base"], d_lo=p["d_lo"], d_hi=p["d_hi"], gamma=p["gamma"], exposure=p["exposure"]))
    P4 = block_mean(P * np.float32(p["exposure"]), 4).astype(np.float32)
    np.save(Path(caches[-1]) / f"{stem}_P4.npy", P4)
    return P4


def region_mask4(sidecar, shape):
    t, l, h, w = (int(x) // 4 for x in sidecar["image_region"]["region"])
    m = np.zeros(shape, bool)
    m[t : t + h, l : l + w] = True
    return m


def measure(P8, R8, params):
    lab, _, pre = render_lab(P8, RenderParams(**params), with_pre_clip=True)
    de = delta_e2000(lab.reshape(-1, 3), R8.reshape(-1, 3))
    ref_clip = reference_clipped(R8).reshape(R8.shape)
    tex = highlight_texture(lab, R8, ref_clip)
    return dict(de2000_median=float(np.median(de)), de2000_p90=float(np.percentile(de, 90)),
                clip_not_ref=float(((pre > 1) & ~ref_clip).any(-1).mean()), texture=tex["ratio"])


def main(which, out_dir, *overrides):
    cfg = dict(NOREF_V1)
    for o in overrides:
        k, v = o.split("=", 1)
        cfg[k] = json.loads(v)
    if overrides:
        cfg["id"] = NOREF_V1["id"] + "+" + ",".join(overrides)
    s = SETS[which]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(s["p4"]).mkdir(parents=True, exist_ok=True)
    sidecars = {p.stem: json.loads(p.read_text()) for p in sorted(Path(s["sidecars"]).glob("*.json"))}
    pos, masks, stats = {}, {}, {}
    for stem, sc in sidecars.items():  # --- inputs for estimation: positives only
        pos[stem] = positive4(sc, stem, s["eval_cache"], s["p4"])
        masks[stem] = None if s["borderless"] else region_mask4(sc, pos[stem].shape[:2])
        stats[stem] = frame_stats(pos[stem], masks[stem], cfg)
    est = {}
    for stem in sidecars:  # --- estimation (no reference anywhere in this loop)
        roll = [stats[o] for o in sidecars if o != stem] if s["roll"] else None
        est[stem] = estimate_params(pos[stem], masks[stem], cfg, roll)
    v1 = {r["stem"]: r["fit"]["params"] for r in json.load(open(s["v1"])) if r["config"] == V1_CONFIG}
    results = {}
    for stem, e in est.items():  # --- evaluation against references, where they exist
        r = dict(estimate=e)
        p8 = Path(s["eval_cache"]) / f"{stem}_P8.npy"
        if p8.exists():
            P8, R8 = np.load(p8), np.load(Path(s["eval_cache"]) / f"{stem}_R8.npy")
            r["eval"] = dict(default=measure(P8, R8, RenderParams().to_dict()), noref=measure(P8, R8, e["params"]),
                             v1=measure(P8, R8, v1[stem]))
            cols = [render(P8, RenderParams())[0], render(P8, RenderParams(**e["params"]))[0],
                    render(P8, RenderParams(**v1[stem]))[0], xyz_to_srgb_preview(lab_to_xyz(R8))]
        else:
            cols = [render(pos[stem][::2, ::2], RenderParams())[0], render(pos[stem][::2, ::2], RenderParams(**e["params"]))[0]]
        gap = np.ones((cols[0].shape[0], 6, 3))
        sheet = np.concatenate(sum([[c, gap] for c in cols], [])[:-1], 1)
        tifffile.imwrite(out_dir / f"{stem}_preview.tif", (np.clip(sheet, 0, 1) * 255 + 0.5).astype(np.uint8))
        results[stem] = r
        ev = r.get("eval")
        print(stem, " ".join(f"{k}={v:.2f}" for k, v in e["params"].items() if k != "matrix"), e["flags"],
              f"wb_conf={e['confidence'].get('wb', 0):.2f}",
              f"| dE default {ev['default']['de2000_median']:.2f} noref {ev['noref']['de2000_median']:.2f} v1 {ev['v1']['de2000_median']:.2f}"
              if ev else "| no reference", flush=True)
    (out_dir / "noref.json").write_text(json.dumps(dict(config=cfg, set=which, results=results), indent=1))
    ev = [r["eval"] for r in results.values() if "eval" in r]
    if ev:
        print("median dE default / noref / v1:", *(round(st.median(x[k]["de2000_median"] for x in ev), 2) for k in ("default", "noref", "v1")))


if __name__ == "__main__":
    main(*sys.argv[1:])
