"""Tables of T4C_REPORT.md from the scripts/t4c_ref_eval.py outputs in experiments/t4_auto_ui/t4c/.

  t4c_summary.py [EVAL_DIR]

Prints (1) per variant: median / mean of the per-frame median dE2000 over the 9 reference frames, median chroma ratio,
mean dL*, and per frame dE; (2) the 37-frame dsc6161 comparison T4b vs T4c (frames better / worse, largest changes,
backlit flags); (3) density ratios of reference-neutral and reference-blue-sky pixels per frame (T4c run).
"""
import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "experiments/t4_auto_ui/t4c")
VARIANTS = ("t4b", "t4c", "v_task_065", "v_cam065_cp0", "v_cam0_cp065", "v_bright", "v_ratio_roll", "v_ratio_frame")


def frames(tag: str) -> dict:
    out = {}
    for roll in ("dsc6161", "dsc6233"):
        d = json.loads((D / f"eval_{tag}_{roll}.json").read_text())
        out.update({k: dict(v, _commit=d["git_commit"][:7], _dirty=d["src_dirty"]) for k, v in d["frames"].items()})
    return out


def short(rel: str) -> str:
    return rel.split("_")[0]


def main():
    print("## variants (9 frames)")
    print("| variant | commit | dirty | dE median | dE mean | chroma ratio median | mean dL | " +
          " | ".join(short(r) for r in frames("t4b")) + " |")
    for tag in VARIANTS:
        fr = frames(tag)
        de = np.array([v["metrics"]["de2000_median"] for v in fr.values()])
        cr = np.array([v["metrics"]["chroma_ratio"] for v in fr.values()], float)
        dl = np.array([v["metrics"]["mean_dL"] for v in fr.values()])
        c = {v["_commit"] for v in fr.values()}
        dirty = any(v["_dirty"] for v in fr.values())
        print(f"| {tag} | {','.join(sorted(c))} | {dirty} | {np.median(de):.2f} | {de.mean():.2f} | {np.nanmedian(cr):.2f} | "
              f"{dl.mean():+.1f} | " + " | ".join(f"{x:.2f}" for x in de) + " |")
    print("\n## per-roll medians")
    for tag in VARIANTS:
        fr = frames(tag)
        a = [v["metrics"]["de2000_median"] for k, v in fr.items() if k < "A050"]
        b = [v["metrics"]["de2000_median"] for k, v in fr.items() if k >= "A050"]
        print(f"{tag}: user edits (dsc6161, {len(a)}) {np.median(a):.2f} | shop prints (dsc6233, {len(b)}) {np.median(b):.2f}")
    print("\n## dL / chroma per frame (t4b -> t4c)")
    b, c = frames("t4b"), frames("t4c")
    for k in b:
        mb, mc = b[k]["metrics"], c[k]["metrics"]
        print(f"{short(k)}: dE {mb['de2000_median']:.2f} -> {mc['de2000_median']:.2f}  dL {mb['mean_dL']:+.1f} -> "
              f"{mc['mean_dL']:+.1f}  chroma {mb['chroma_ratio']:.2f} -> {mc['chroma_ratio']:.2f}  flags {c[k].get('flags')}")

    print("\n## 37 frames dsc6161 (user edits)")
    e = {t: json.loads((D / f"eval37_{t}_dsc6161.json").read_text()) for t in ("t4b", "t4c")}
    for t, d in e.items():
        print(t, d["git_commit"][:7], "dirty", d["src_dirty"], "frames", len(d["frames"]))
    ks = sorted(e["t4b"]["frames"])
    xb = np.array([e["t4b"]["frames"][k]["metrics"]["de2000_median"] for k in ks])
    xc = np.array([e["t4c"]["frames"][k]["metrics"]["de2000_median"] for k in ks])
    for name, f in (("dE", "de2000_median"), ("chroma", "chroma_ratio"), ("dL", "mean_dL")):
        vb = np.array([e["t4b"]["frames"][k]["metrics"][f] for k in ks], float)
        vc = np.array([e["t4c"]["frames"][k]["metrics"][f] for k in ks], float)
        print(f"{name}: median {np.nanmedian(vb):.2f} -> {np.nanmedian(vc):.2f}, mean {np.nanmean(vb):.2f} -> {np.nanmean(vc):.2f}")
    d = xc - xb
    print(f"better {int((d < 0).sum())} worse {int((d > 0).sum())} | |change|>0.25: better {int((d < -0.25).sum())} "
          f"worse {int((d > 0.25).sum())}")
    order = np.argsort(d)
    print("largest improvements:", ", ".join(f"{short(ks[i])} {xb[i]:.2f}->{xc[i]:.2f}" for i in order[:5]))
    print("largest worsenings:", ", ".join(f"{short(ks[i])} {xb[i]:.2f}->{xc[i]:.2f}" for i in order[::-1][:5]))
    bl = [k for k in ks if "backlit_midtone" in (e["t4c"]["frames"][k].get("flags") or [])]
    print("backlit_midtone:", ", ".join(f"{short(k)} {e['t4b']['frames'][k]['metrics']['de2000_median']:.2f}->"
                                        f"{e['t4c']['frames'][k]['metrics']['de2000_median']:.2f}" for k in bl))

    print("\n## density ratios R/G, B/G (t4c run; n = pixels)")
    for k, v in c.items():
        dr = v["density_ratios"]
        cells = []
        for s in ("ref_neutral", "ref_blue_sky", "mid_density", "white_point"):
            x = dr[s]
            cells.append(f"{s} {x['r']:.3f}:1:{x['b']:.3f} (n={x.get('n', '-')})" if "r" in x else f"{s} n={x.get('n')}")
        print(short(k), " | ".join(cells))
        for band in dr["ref_neutral"].get("by_d_g", []):
            print(f"    neutral D_G {band['d_g'][0]:.2f}-{band['d_g'][1]:.2f} n={band['n']}: {band['r']:.3f}:1:{band['b']:.3f}")


if __name__ == "__main__":
    main()
