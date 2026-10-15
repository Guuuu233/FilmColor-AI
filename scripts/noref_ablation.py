"""S2-7 noref_v2 ablation: fixed look first, then one parameter group at a time.

  noref_ablation.py OUT_DIR [WORKERS]

Stages (each builds on the last *kept* stage):
  A  global_look_v1 only
  B  + brightness: B1 symmetric median band, B2 brighten only (high-key frames left alone); of the
     variants passing the rule, the one with the lower mean of the two set medians is kept
  C  + white balance
  D  + tone (shoulder / contrast); its encoded-spread band is the p10..p90 of the spreads of all
     development frames (both sets, references not used) under the last kept stage
  E  + saturation

Keep rule, fixed before running (S2-7 decision): versus the last kept stage,
  1. the median dE2000 (anomalous frames DSC_3822 / DSC_3858 excluded) is lower on both sets;
  2. over both sets, more frames improve than worsen by more than 0.5;
  3. the largest share of pixels clipped where the reference is not (clip_not_ref) stays <= 2 %.
A group that fails stays at the look's value.

Estimation sees only the positives (P4) and, for the shop roll, the other frames' statistics; references
are read only afterwards, for measurement. All scores are development-set scores.
"""
import json
import os
import statistics as st
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from noref_eval import SETS, V1_CONFIG, measure, positive4, region_mask4  # noqa: E402

from filmcolor.noref import NOREF_V2, encoded_spread, estimate_params_v2, frame_stats, load_look  # noqa: E402

ANOMALOUS = {"DSC_3822", "DSC_3858"}
MARGIN, MAX_CLIP = 0.5, 0.02

_DATA = {}


def load_set(which):
    s = SETS[which]
    sidecars = {p.stem: json.loads(p.read_text()) for p in sorted(Path(s["sidecars"]).glob("*.json"))}
    pos, masks, stats = {}, {}, {}
    for stem, sc in sidecars.items():
        pos[stem] = positive4(sc, stem, s["eval_cache"], s["p4"])
        masks[stem] = None if s["borderless"] else region_mask4(sc, pos[stem].shape[:2])
        stats[stem] = frame_stats(pos[stem], masks[stem], NOREF_V2)
    v1 = {r["stem"]: r["fit"]["params"] for r in json.load(open(s["v1"])) if r["config"] == V1_CONFIG}
    return dict(pos=pos, masks=masks, stats=stats, roll=s["roll"], cache=s["eval_cache"], v1=v1)


def _init():
    _DATA.update({w: load_set(w) for w in SETS})


def _job(args):
    which, stem, cfg = args
    d = _DATA[which]
    roll = [d["stats"][o] for o in d["pos"] if o != stem] if d["roll"] else None
    e = estimate_params_v2(d["pos"][stem], d["masks"][stem], cfg, roll)
    r = dict(set=which, stem=stem, estimate=e)
    p8 = Path(d["cache"]) / f"{stem}_P8.npy"
    if p8.exists():
        r["eval"] = measure(np.load(p8), np.load(Path(d["cache"]) / f"{stem}_R8.npy"), e["params"])
    return r


def _spread_job(args):
    which, stem, params = args
    d = _DATA[which]
    return encoded_spread(d["pos"][stem], d["masks"][stem], params)


def run_stage(pool, cfg, frames):
    return {(r["set"], r["stem"]): r for r in pool.map(_job, [(w, s, cfg) for w, s in frames])}


def summary(res):
    out = {}
    for w in SETS:
        de = [r["eval"]["de2000_median"] for (s, stem), r in res.items() if s == w and "eval" in r and stem not in ANOMALOUS]
        out[w] = st.median(de)
    clip = [r["eval"]["clip_not_ref"] for r in res.values() if "eval" in r and r["stem"] not in ANOMALOUS]
    out["max_clip_not_ref"] = max(clip)
    return out


def decide(new, old):
    sn, so = summary(new), summary(old)
    better = worse = 0
    for k, r in new.items():
        if "eval" in r and r["stem"] not in ANOMALOUS:
            d = r["eval"]["de2000_median"] - old[k]["eval"]["de2000_median"]
            better += d < -MARGIN
            worse += d > MARGIN
    checks = dict(median_lower_both=all(sn[w] < so[w] for w in SETS), more_better_than_worse=better > worse,
                  clip_ok=sn["max_clip_not_ref"] <= MAX_CLIP)
    return dict(keep=all(checks.values()), checks=checks, better=better, worse=worse, summary=sn, previous=so)


def main(out_dir, workers=8):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    look = load_look()
    _init()
    frames = [(w, s) for w in SETS for s in _DATA[w]["pos"]]
    log = dict(look=look["id"], look_params=look["params"], rule=dict(margin=MARGIN, max_clip=MAX_CLIP,
               anomalous=sorted(ANOMALOUS)), stages={})
    with Pool(int(workers), initializer=_init) as pool:
        def stage(name, cfg):
            res = run_stage(pool, cfg, frames)
            log["stages"][name] = dict(groups=list(cfg["groups"]), brighten_only=cfg["brighten_only"],
                                       spread_band=cfg["spread_band"], results=list(res.values()))
            print(name, cfg["groups"], {k: round(v, 3) for k, v in summary(res).items()}, flush=True)
            return res

        kept_cfg = dict(NOREF_V2, groups=())
        kept = stage("A", kept_cfg)
        log["kept"] = ["A"]
        cands = {}
        for name, bo in (("B1", False), ("B2", True)):
            cfg = dict(NOREF_V2, groups=("brightness",), brighten_only=bo)
            res = stage(name, cfg)
            dec = decide(res, kept)
            log["stages"][name]["decision"] = dec
            print("  ", name, dec["checks"], f"better {dec['better']} worse {dec['worse']}", flush=True)
            if dec["keep"]:
                cands[name] = (cfg, res, (dec["summary"]["nlp"] + dec["summary"]["shop"]) / 2)
        if cands:
            name = min(cands, key=lambda k: cands[k][2])
            kept_cfg, kept = cands[name][0], cands[name][1]
            log["kept"].append(name)
        for name, group in (("C", "wb"), ("D", "tone"), ("E", "sat")):
            cfg = dict(kept_cfg, groups=tuple(kept_cfg["groups"]) + (group,))
            if group == "tone":
                params = {(r["set"], r["stem"]): r["estimate"]["params"] for r in kept.values()
                          if r["estimate"]["status"] == "ok"}
                spreads = pool.map(_spread_job, [(w, s, p) for (w, s), p in params.items()])
                cfg["spread_band"] = [float(x) for x in np.percentile(spreads, [10, 90])]
                log["spread_band_source"] = dict(stage=log["kept"][-1], n_frames=len(spreads),
                                                 percentiles=[10, 90], band=cfg["spread_band"])
            res = stage(name, cfg)
            dec = decide(res, kept)
            log["stages"][name]["decision"] = dec
            print("  ", name, dec["checks"], f"better {dec['better']} worse {dec['worse']}", flush=True)
            if dec["keep"]:
                kept_cfg, kept = cfg, res
                log["kept"].append(name)
    log["final"] = dict(stages=log["kept"], groups=list(kept_cfg["groups"]), brighten_only=kept_cfg["brighten_only"],
                        spread_band=kept_cfg["spread_band"])
    log["v1_params"] = {w: _DATA[w]["v1"] for w in SETS}
    (out_dir / "ablation.json").write_text(json.dumps(log, indent=1))
    print("kept:", log["kept"], flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
