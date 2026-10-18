"""BlueNeg roll-base study, step 2: is the LinearRaw signal on one scale across the frames of a roll? (read-only)

Per downloaded development negative (sealed frames refused by name): the engine's automatic film base for scanner
DNGs (estimate_base_scanner + base_from_regions, as pipeline._scanner_input), per-channel percentiles of 24 px
block means (0.1 ... 99.99; on a negative the high end is the least dense film) and the SilverFast job values
that could have been applied to the pixels (NeutralizeFactor*, IsAutoMode, IsNeutralizeMode, OrangeMask*).
Per-frame records are cached in OUT_DIR/frames/ (resumable).

Tests, within real rolls (<yyyymm><letter>), frame values centred on their roll mean:
  T1 neutralize: slope of the frame's base chroma (log2 R/G, B/G) on log2 of the matching NeutralizeFactor
     ratio. ~0: factors only recorded; ~+-1: factors applied to the RGB.
  T2 auto exposure: spread of the clearest-film level (G 99.99th percentile, in D) across frames vs the spread of
     the frames' base candidates; per-frame exposure normalisation would pin the clearest level.
  T3 SilverFast's own mask (OrangeMaskWhite*) against our base candidate: same scale or not (recorded only).

  bn_comparability.py META_JSON DATA_ROOT SCAN_SETTINGS_JSON OUT_DIR
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from filmcolor.frozen import check_name
from filmcolor.invert import base_from_regions
from filmcolor.scanner import block_means, estimate_base_scanner, load_scanner

PCTS = (0.1, 1, 5, 50, 95, 99, 99.9, 99.99)


def roll_of(x):
    rid = x["roll_id"]
    return "ds-" + (x["date"][:6] + rid[-1] if rid[-1].isalpha() else rid)


def frame_record(path):
    fr = load_scanner(path)
    est = estimate_base_scanner(fr)
    rb = base_from_regions(fr.rgb, est.regions)
    sp, sp_sat = block_means(fr.rgb, fr.sat, 24)
    v = sp[~sp_sat]
    pct = {str(p): np.percentile(v, p, axis=0).tolist() for p in PCTS}
    return dict(base=list(rb["rgb"]), base_warnings=list(getattr(est, "warnings", []) or []), pct=pct,
                sat_frac=float(fr.sat.mean()), shape=list(fr.sat.shape))


def f(x, k, default=None):
    try:
        return float(x[k])
    except (KeyError, TypeError, ValueError):
        return default


def main(meta_json, root, settings_json, out_dir):
    root, out = Path(root), Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    xmp = {fr["name"]: fr["xmp"] for fr in json.loads(Path(settings_json).read_text())["frames"]}
    rows = []
    for x in json.loads(Path(meta_json).read_text()):
        p = root / x["negative_path"]
        if not p.exists() or x["filename"] not in xmp:
            continue
        check_name(p)
        cache = out / "frames" / f"{x['filename']}.json"
        if not cache.exists():
            cache.write_text(json.dumps(frame_record(p)))
            print("done", x["filename"], flush=True)
        r = json.loads(cache.read_text())
        s = xmp[x["filename"]]
        r.update(name=x["filename"], roll=roll_of(x), partition=x["partition"], film=x["film_type"],
                 nf=[f(s, f"NeutralizeFactor{c}", 1.0) for c in ("Red", "Green", "Blue")],
                 auto=s.get("IsAutoMode"), neutralize=s.get("IsNeutralizeMode"),
                 mask_white=[f(s, f"OrangeMaskWhite{c}") for c in ("Red", "Green", "Blue")])
        rows.append(r)

    by = defaultdict(list)
    for r in rows:
        by[r["roll"]].append(r)
    # T1: centred within roll
    xs, ys = defaultdict(list), defaultdict(list)
    for fs in by.values():
        if len(fs) < 2:
            continue
        b = np.log2(np.array([r["base"] for r in fs]))
        nf = np.log2(np.array([r["nf"] for r in fs]))
        for i, (a, c) in enumerate(((0, 1), (2, 1))):
            bc, nc = b[:, a] - b[:, c], nf[:, a] - nf[:, c]
            xs[i] += list(nc - nc.mean())
            ys[i] += list(bc - bc.mean())
    t1 = {}
    for i, name in enumerate(("R/G", "B/G")):
        x_, y_ = np.array(xs[i]), np.array(ys[i])
        slope = float((x_ * y_).sum() / (x_ * x_).sum()) if (x_ * x_).sum() > 0 else None
        t1[name] = dict(n=len(x_), slope=slope, nf_spread_log2=float(x_.std()) if len(x_) else None,
                        base_chroma_spread_log2=float(y_.std()) if len(y_) else None,
                        corr=float(np.corrcoef(x_, y_)[0, 1]) if len(x_) > 2 and x_.std() > 0 else None)
    rolls = {}
    for roll, fs in sorted(by.items()):
        clear = np.array([np.log10(r["pct"]["99.99"][1]) for r in fs])
        base = np.array([np.log10(r["base"][1]) for r in fs])
        mw = [r["mask_white"] for r in fs if None not in r["mask_white"]]
        rolls[roll] = dict(
            frames=len(fs), film=sorted({r["film"] for r in fs}), auto_modes=sorted({str(r["auto"]) for r in fs}),
            clear_G_range_D=float(clear.max() - clear.min()), base_G_range_D=float(base.max() - base.min()),
            base_chroma_log2=[[round(float(np.log2(r["base"][0] / r["base"][1])), 3),
                               round(float(np.log2(r["base"][2] / r["base"][1])), 3)] for r in fs],
            clear_max_vs_base_D=[round(float(np.log10(r["pct"]["99.99"][1] / r["base"][1])), 3) for r in fs],
            silverfast_mask_white_chroma_log2=[[round(float(np.log2(m[0] / m[1])), 3),
                                                round(float(np.log2(m[2] / m[1])), 3)] for m in mw],
            names=[r["name"] for r in fs])
    res = dict(n=len(rows), t1_neutralize=t1, rolls=rolls)
    (out / "comparability.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(t1, indent=1))
    for k, v in rolls.items():
        print(k, v["frames"], v["auto_modes"], "clear G range %.3f D, base G range %.3f D" % (v["clear_G_range_D"], v["base_G_range_D"]))


if __name__ == "__main__":
    main(*sys.argv[1:5])
