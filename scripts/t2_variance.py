"""T2 parameter variance decomposition (task list 2026-10-21 v2), no training: how much of the P4 labels
(ev, temp, tint, midtone) is shared by a roll and how much changes frame to frame.

  t2_variance.py T1_FRAMES_JSON OUT_DIR

Inputs: experiments/s3_ml/labels/<domain>/<stem>.json (field p4.params, p4.at_bound; the shop / print / NLP grade
the label was fitted to), experiments/s3_ml/user_labels/<stem>.json (same fit to the user's Lightroom grade),
T1 frames.json (roll id, roll level, frame order), experiments/s3_separate/check/*_shop.json (per-frame base vs
roll base: chroma part = max channel |c0 - c1|, level part = |l0 - l1|, in D). Numbers only; reading rules are in
the task list.

  2a  one-way random-effects ANOVA per parameter: ICC = s2_b / (s2_b + s2_w), s2_b = max(0, (MSB - MSW) / n0),
      plus SSB / SST; 95 % percentile intervals from 2000 roll bootstraps (rolls drawn with replacement)
  2b  56 frames with shop and user grades: d = user - shop, split into the roll mean of d (offset) and the rest
  2c  predictors in parameter space on frames of rolls with >= 4 labelled frames: mean of the other rolls;
      mean of k = 1, 2, 3 other frames of the same roll (all combinations, at most 300 drawn); mean of all other
      frames of the same roll (optimistic bound, not a model)
  2d  semivariance 0.5 E[(x_i - x_j)^2] of frame pairs of one roll by frame distance, against pairs of two rolls
"""
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PARAMS = ("ev", "temp", "tint", "midtone")
SCALE = dict(ev=1.0, temp=0.25, tint=0.25, midtone=0.05)  # P4 fit scales (fit_config_s3_p4)
BOUND = dict(ev=(-3, 3), temp=(-1, 1), tint=(-1, 1), midtone=(0.05, 0.5))
SEED, B = 20261022, 2000
BINS = ((1, 1), (2, 3), (4, 7), (8, 15), (16, 10 ** 6))


def load(t1_frames):
    t1 = json.loads(Path(t1_frames).read_text())
    by_file = {Path(f).name: r for r in t1 for f in r["files"]}
    by_id = {r["id"]: r for r in t1}
    dev = {(c["id"]): c for p in (REPO / "experiments/s3_separate/check").glob("*_shop.json")
           for c in [json.loads(p.read_text())]}
    rows = []
    for p in sorted((REPO / "experiments/s3_ml/labels").glob("*/*.json")):
        d = json.loads(p.read_text())
        dom, stem = p.parent.name, p.stem
        if dom == "ext_own":
            t = by_file[Path(d["cache"]["raw"]).name]
        elif dom == "blueneg":
            t = by_id[f"blueneg:{stem}"]
        elif dom == "shop":
            t = by_id[f"shop_r01:{stem}"]
        else:
            t = by_id[f"nlp:{stem}"]
        assert t["partition"] == "dev", t["id"]
        c = dev.get(d["id"])
        rows.append(dict(id=d["id"], domain=dom, stem=stem, roll=t["roll"], roll_level=t["roll_level"],
                         order=t["frame_no"], label_roll=d["roll"],
                         x={k: d["p4"]["params"][k] for k in PARAMS}, at_bound=d["p4"]["at_bound"],
                         chroma_part=None if c is None else float(max(abs(v) for v in c["chroma_diff"])),
                         level_part=None if c is None else abs(c["level_diff"])))
    return rows


def anova(x, g):
    x, g = np.asarray(x, float), np.asarray(g)
    rolls = sorted(set(g))
    k, n = len(rolls), len(x)
    m = x.mean()
    ni = np.array([np.sum(g == r) for r in rolls])
    mi = np.array([x[g == r].mean() for r in rolls])
    ssb = float(np.sum(ni * (mi - m) ** 2))
    ssw = float(sum(np.sum((x[g == r] - mi[i]) ** 2) for i, r in enumerate(rolls)))
    if k < 2 or n - k < 1:
        return None
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - np.sum(ni ** 2) / n) / (k - 1)
    s2b = max(0.0, (msb - msw) / n0)
    return dict(icc=s2b / (s2b + msw) if s2b + msw > 0 else 0.0, ssb_share=ssb / (ssb + ssw), sd_between=s2b ** 0.5,
                sd_within=msw ** 0.5, n0=float(n0))


def boot(rows, fn, rng):
    """Roll bootstrap: rolls drawn with replacement, a roll drawn twice enters as two rolls."""
    by = defaultdict(list)
    for r in rows:
        by[r["roll"]].append(r)
    keys = sorted(by)
    out = []
    for _ in range(B):
        draw = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        rs = [dict(r, roll=f"{k}#{j}", orig=k) for j, k in enumerate(draw) for r in by[k]]
        v = fn(rs)
        if v is not None:
            out.append(v)
    if not out:
        return None
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), len(out)]


def r4(v):
    return None if v is None else round(float(v), 4)


def t2a(rows, rng):
    res = dict(frames=len(rows), rolls=len({r["roll"] for r in rows}),
               roll_sizes=sorted((sum(x["roll"] == k for x in rows) for k in {r["roll"] for r in rows}), reverse=True),
               roll_levels=sorted({r["roll_level"] for r in rows}), params={})
    for p in PARAMS:
        a = anova([r["x"][p] for r in rows], [r["roll"] for r in rows])
        icc_ci = boot(rows, lambda rs: (lambda z: z and z["icc"])(anova([r["x"][p] for r in rs], [r["roll"] for r in rs])), rng)
        share_ci = boot(rows, lambda rs: (lambda z: z and z["ssb_share"])(anova([r["x"][p] for r in rs], [r["roll"] for r in rs])), rng)
        res["params"][p] = dict({k: r4(v) for k, v in a.items()}, icc_ci95=[r4(v) for v in icc_ci[:2]],
                                ssb_share_ci95=[r4(v) for v in share_ci[:2]], boot_ok=icc_ci[2],
                                at_bound=sum(p in r["at_bound"] for r in rows), sd_total=r4(np.std([r["x"][p] for r in rows], ddof=1)))
    return res


def t2b(rows, aset):
    shop = {r["stem"]: r for r in rows if r["domain"] == "ext_own"}
    us = []
    for q in sorted((REPO / "experiments/s3_ml/user_labels").glob("*.json")):
        d = json.loads(q.read_text())
        s = shop[q.stem]
        us.append(dict(stem=q.stem, roll=s["roll"], d={p: d["p4"]["params"][p] - s["x"][p] for p in PARAMS},
                       user_at_bound=d["p4"]["at_bound"]))
    res = dict(frames=len(us), rolls={k: sum(u["roll"] == k for u in us) for k in sorted({u["roll"] for u in us})},
               params={})
    for p in PARAMS:
        d = np.array([u["d"][p] for u in us])
        g = np.array([u["roll"] for u in us])
        off = {k: float(d[g == k].mean()) for k in sorted(set(g))}
        resid = np.array([u["d"][p] - off[u["roll"]] for u in us])
        ss = float(np.sum(d ** 2))
        ss_off = float(sum(np.sum(g == k) * v ** 2 for k, v in off.items()))
        res["params"][p] = dict(
            rms_difference=r4(np.sqrt(np.mean(d ** 2))), roll_offset={k: r4(v) for k, v in off.items()},
            sd_frame_part=r4(np.sqrt(np.mean(resid ** 2))), offset_share_of_sum_sq=r4(ss_off / ss if ss else None),
            sd_between_rolls_own=aset["own"]["params"][p]["sd_between"],
            sd_within_rolls_own=aset["own"]["params"][p]["sd_within"],
            user_at_bound=sum(p in u["user_at_bound"] for u in us),
            shop_at_bound=sum(p in shop[u["stem"]]["at_bound"] for u in us))
    return res


def t2c(rows, rng_py):
    by = defaultdict(list)
    for r in rows:
        by[r["roll"]].append(r)
    ev = [r for r in rows if len(by[r["roll"]]) >= 4]
    res = dict(frames=len(ev), rolls=len({r["roll"] for r in ev}), params={})
    for p in PARAMS:
        sq = defaultdict(list)
        for r in ev:
            y = r["x"][p]
            other = [q["x"][p] for q in rows if q["roll"] != r["roll"]]
            sq["global_mean_other_rolls"].append((np.mean(other) - y) ** 2)
            same = [q["x"][p] for q in by[r["roll"]] if q is not r]
            for k in (1, 2, 3):
                combos = list(itertools.combinations(range(len(same)), k))
                if len(combos) > 300:
                    combos = rng_py.sample(combos, 300)
                sq[f"same_roll_mean_of_{k}"].append(np.mean([(np.mean([same[i] for i in c]) - y) ** 2 for c in combos]))
            sq["same_roll_mean_of_all_others"].append((np.mean(same) - y) ** 2)
        res["params"][p] = {k: dict(rmse=r4(np.sqrt(np.mean(v))), rmse_in_scale_units=r4(np.sqrt(np.mean(v)) / SCALE[p]))
                            for k, v in sq.items()}
    return res


def t2d(rows, rng):
    def gam(rs, p):
        by = defaultdict(list)
        for r in rs:
            by[r["roll"]].append(r)
        acc = defaultdict(list)
        for k, v in by.items():
            for a, b in itertools.combinations([r for r in v if r["order"] is not None], 2):
                dist = abs(a["order"] - b["order"])
                lab = next(f"{lo}-{hi}" if hi < 10 ** 6 else f">={lo}" for lo, hi in BINS if lo <= dist <= hi)
                acc[lab].append(0.5 * (a["x"][p] - b["x"][p]) ** 2)
                acc["within_all"].append(0.5 * (a["x"][p] - b["x"][p]) ** 2)
        keys = sorted(by)
        for i, k in enumerate(keys):
            for k2 in keys[i + 1:]:
                if by[k][0].get("orig", k) == by[k2][0].get("orig", k2):
                    continue
                xa = np.array([r["x"][p] for r in by[k]])
                xb = np.array([r["x"][p] for r in by[k2]])
                acc["between_rolls"].append(float(np.mean(0.5 * (xa[:, None] - xb[None, :]) ** 2)))
        return acc

    labels = [f"{lo}-{hi}" if hi < 10 ** 6 else f">={lo}" for lo, hi in BINS] + ["within_all", "between_rolls"]
    by = defaultdict(list)
    for r in rows:
        by[r["roll"]].append(r)
    keys = sorted(by)
    draws = []  # one roll bootstrap draw -> all parameters and distance classes at once
    for _ in range(B):
        draw = [keys[i] for i in rng.integers(0, len(keys), len(keys))]
        rs = [dict(r, roll=f"{k}#{j}", orig=k) for j, k in enumerate(draw) for r in by[k]]
        draws.append({p: {lab: float(np.mean(v)) for lab, v in gam(rs, p).items() if v} for p in PARAMS})
    res = dict(frames=len(rows), rolls=len(keys), params={})
    for p in PARAMS:
        acc = gam(rows, p)
        out = {}
        for lab in labels:
            v = acc.get(lab, [])
            bs = [d[p][lab] for d in draws if lab in d[p]]
            out[lab] = dict(n=len(v), semivariance=r4(np.mean(v)) if v else None,
                            ci95=[r4(np.percentile(bs, 2.5)), r4(np.percentile(bs, 97.5))] if v and bs else None)
        res["params"][p] = out
    res["n_counts"] = "pairs of frames for distance classes and within_all; pairs of rolls for between_rolls"
    return res


def base_groups(rows):
    res = {}
    for name, sel in (("chroma_or_level_part>=0.1", lambda r: r["chroma_part"] is not None and max(r["chroma_part"], r["level_part"]) >= 0.1),
                      ("both<0.1", lambda r: r["chroma_part"] is not None and max(r["chroma_part"], r["level_part"]) < 0.1),
                      ("no_record", lambda r: r["chroma_part"] is None)):
        rs = [r for r in rows if sel(r)]
        g = dict(frames=len(rs), rolls=len({r["roll"] for r in rs}), params={})
        for p in PARAMS:
            x = np.array([r["x"][p] for r in rs])
            by = defaultdict(list)
            for r in rs:
                by[r["roll"]].append(r["x"][p])
            dfw = sum(len(v) - 1 for v in by.values())
            sw = np.sqrt(sum(np.sum((np.array(v) - np.mean(v)) ** 2) for v in by.values()) / dfw) if dfw else None
            g["params"][p] = dict(mean=r4(x.mean()) if len(x) else None, sd=r4(x.std(ddof=1)) if len(x) > 1 else None,
                                  pooled_sd_within_roll=r4(sw), at_bound=sum(p in r["at_bound"] for r in rs))
        res[name] = g
    return res


def main(t1_frames, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load(t1_frames)
    sets = dict(all=rows, own=[r for r in rows if r["domain"] in ("ext_own", "shop")],
                blueneg=[r for r in rows if r["domain"] == "blueneg"])
    rng = np.random.default_rng(SEED)
    res = dict(seed=SEED, bootstrap=B, label_field="p4.params (fit_config_s3_p4, look global_look_v1)",
               frames_by_domain={d: sum(r["domain"] == d for r in rows) for d in sorted({r["domain"] for r in rows})})
    res["2a"] = {k: t2a(v, rng) for k, v in sets.items()}
    res["2b"] = t2b(rows, res["2a"])
    rp = random.Random(SEED)
    res["2c"] = {k: t2c(v, rp) for k, v in sets.items()}
    res["2d"] = {k: t2d(v, rng) for k, v in sets.items()}
    res["base_deviation_groups"] = {k: base_groups(v) for k, v in sets.items()}
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
