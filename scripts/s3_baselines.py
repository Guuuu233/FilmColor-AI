"""S3 step 4: baselines M0-M2 for experiment A (BlueNeg leave-one-roll-out) and the development stage of
experiment B (4 folds, one ext_own roll out, shop_r01 always in training); docs/S3_ML_DESIGN.md v2.2.

  s3_baselines.py FRAMES_JSON LABEL_DIR OUT_DIR [WORKERS]

Stage 1 (in-fold): features from the full-frame X8 caches only; standardisation, the ridge regulariser (inner
leave-one-roll-out) and the M0 fold medians from the training frames of each fold. Stage 2: every prediction is
rendered on the aligned P8 grid against R8 (dE2000, mean dLab, new clipping) and on X8 for the quality checks.
Stage 3: experiment A tiers with the roll bootstrap; experiment B development numbers per roll (descriptive).
Frozen test files and their caches are never read (filmcolor.frozen guards every cache read).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json  # noqa: E402
import statistics as st  # noqa: E402
import sys  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from filmcolor.colorimetry import delta_e2000  # noqa: E402
from filmcolor.fit import FIT_CONFIG_S3, SCALES, reference_clipped, render_lab  # noqa: E402
from filmcolor.frozen import load_cache  # noqa: E402
from filmcolor.learn import Ridge, features_glob, features_grid, roll_bootstrap, select_alpha  # noqa: E402
from filmcolor.paths import data_dir  # noqa: E402
from filmcolor.qc import quality_checks  # noqa: E402
from filmcolor.render import BOUNDS, RenderParams, overexposure_stops, render  # noqa: E402

LOOK = json.loads(Path("src/filmcolor/configs/global_look_v1.json").read_text())
VARIANTS = {v: tuple(FIT_CONFIG_S3[v]["free"]) for v in ("p4", "p3")}
METHODS = ("M0_fixed", "M0_median", "M1", "M2")
NEW_CLIP_FLAG = 0.02


def load_frames(frames_json, label_dir):
    frames = []
    for f in json.load(open(frames_json))["frames"]:
        lp = Path(label_dir) / f["domain"] / f"{f['stem']}.json"
        if not lp.exists():  # pipeline failure (processing_status.json): no label this round
            continue
        lab = json.loads(lp.read_text())
        X8 = load_cache(data_dir("s3_ml", "cache", f["domain"]) / f"{f['stem']}_X8.npy")
        frames.append(dict(id=f["id"], domain=f["domain"], stem=f["stem"], roll=f["roll"],
                           assumed=f.get("color_space_assumed"), glob=features_glob(X8), grid=features_grid(X8),
                           labels={v: lab[v]["params"] for v in VARIANTS},
                           label_de={k: lab[k]["de2000_median"] for k in ("p4", "p3", "ub7", "v1", "fixed_look")}))
    return frames


def experiments(frames):
    dom = lambda *d: [f for f in frames if f["domain"] in d]  # noqa: E731
    bl, ext, shop, nlp = dom("blueneg"), dom("ext_own"), dom("shop"), dom("nlp")
    return {
        "A_a": dict(test=bl, extra=[], validate="test_rolls"),
        "A_b": dict(test=bl, extra=nlp + shop, validate="test_rolls"),
        "B_dev_a": dict(test=ext, extra=shop, validate="test_rolls"),
        "B_dev_b": dict(test=ext, extra=shop + bl + nlp, validate="test_rolls"),
    }


def design(fs, method):
    return np.array([f["glob"] if method == "M1" else np.r_[f["glob"], f["grid"]] for f in fs])


def targets(fs, variant):
    return np.array([[f["labels"][variant][k] / SCALES[k] for k in VARIANTS[variant]] for f in fs])


def to_params(row, variant):
    p = {k: v for k, v in LOOK["params"].items()}
    for k, v in zip(VARIANTS[variant], row):
        p[k] = float(np.clip(v * SCALES[k], *BOUNDS[k]))
    return p


def predict_all(frames):
    """Stage 1: {(experiment, variant, method, frame id): dict(params, fold, alpha)}."""
    out = {}
    for name, e in experiments(frames).items():
        test_rolls = list(dict.fromkeys(f["roll"] for f in e["test"]))
        for roll in test_rolls:
            test = [f for f in e["test"] if f["roll"] == roll]
            train = [f for f in e["test"] if f["roll"] != roll] + e["extra"]
            groups = np.array([f["roll"] for f in train])
            inner = set(r for r in test_rolls if r != roll)  # inner validation: the other test-domain rolls
            for v in VARIANTS:
                Y = targets(train, v)
                for f in test:
                    out[(name, v, "M0_fixed", f["id"])] = dict(params=dict(LOOK["params"]), fold=roll)
                    out[(name, v, "M0_median", f["id"])] = dict(params=to_params(np.median(Y, 0), v), fold=roll)
                for m in ("M1", "M2"):
                    X = design(train, m)
                    alpha, _ = select_alpha(X, Y, groups, validate=inner)
                    model = Ridge(alpha).fit(X, Y)
                    for f, row in zip(test, model.predict(design(test, m))):
                        out[(name, v, m, f["id"])] = dict(params=to_params(row, v), fold=roll, alpha=alpha)
    return out


def evaluate(args):
    """Stage 2 for one frame and a list of parameter sets."""
    f, param_sets = args
    cache = data_dir("s3_ml", "cache", f["domain"])
    P, R = load_cache(cache / f"{f['stem']}_P8.npy"), load_cache(cache / f"{f['stem']}_R8.npy")
    X8 = load_cache(cache / f"{f['stem']}_X8.npy")
    rec = json.loads((cache / f"{f['stem']}.json").read_text())
    Pr, Rr = P.reshape(-1, 3)[None], R.reshape(-1, 3).astype(np.float64)
    ref_clip = reference_clipped(Rr)
    t, l, h, w = rec["positive"]["image_region_x8"]
    mask = np.zeros(X8.shape[:2], bool)
    mask[t : t + h, l : l + w] = True
    res = []
    for p in param_sets:
        lab, diag, pre = render_lab(Pr, RenderParams(**p), with_pre_clip=True)
        de = delta_e2000(lab[0], Rr)
        d = lab[0] - Rr
        over = overexposure_stops(pre[0]) > 0
        disp, _ = render(X8, RenderParams(**p), mask)
        qc = quality_checks(disp, mask, rec["positive"]["report"], rec["positive"]["base_inside_image"])
        res.append(dict(de_median=float(np.median(de)), de_mean=float(de.mean()),
                        dL=float(d[:, 0].mean()), da=float(d[:, 1].mean()), db=float(d[:, 2].mean()),
                        new_clip=float((over & ~ref_clip).any(-1).mean()),
                        below_black=float(diag.get("below_zero_after_colour", 0.0)), qc=qc["status"]))
    return f["id"], res


def param_error(pred, label, variant):
    return float(np.sqrt(np.mean([((pred[k] - label[k]) / SCALES[k]) ** 2 for k in VARIANTS[variant]])))


def compare(rows, a, b):
    """Paired per-frame dE(median) differences a - b with the roll bootstrap and the clipping counts."""
    d = [r[a]["de_median"] - r[b]["de_median"] for r in rows]
    bs = roll_bootstrap(d, [r["roll"] for r in rows])
    clip = {m: sum(r[m]["new_clip"] > NEW_CLIP_FLAG for r in rows) for m in (a, b)}
    return dict(bs, clip_over_2pct=clip, n_frames=len(rows))


def passes(c, level=0.95):
    return c["ci"][1] < 0 and c["improved_rolls"] >= 0.6 and c["clip_over_2pct"][list(c["clip_over_2pct"])[0]] <= \
        c["clip_over_2pct"][list(c["clip_over_2pct"])[1]]


def summary(rows, frames_by_id):
    out = {}
    for m in METHODS:
        out[m] = dict(de_median=st.median(r[m]["de_median"] for r in rows), de_mean=st.median(r[m]["de_mean"] for r in rows),
                      param_error=st.median(r[m]["param_error"] for r in rows),
                      prediction_loss=st.median(r[m]["de_median"] - r["label_de"] for r in rows),
                      new_clip_over_2pct=sum(r[m]["new_clip"] > NEW_CLIP_FLAG for r in rows),
                      below_black_over_5pct=sum(r[m]["below_black"] > 0.05 for r in rows),
                      qc_fail=sum(r[m]["qc"] == "fail" for r in rows))
    out["label"] = dict(de_median=st.median(r["label_de"] for r in rows))
    out["space_limit"] = st.median(r["label_de"] - r["ub7_de"] for r in rows)
    out["n"] = len(rows)
    return out


def main(frames_json, label_dir, out_dir, workers="6"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = load_frames(frames_json, label_dir)
    by_id = {f["id"]: f for f in frames}
    preds = predict_all(frames)
    print(len(frames), "frames,", len(preds), "predictions", flush=True)
    per_frame = {}
    for (e, v, m, fid), p in preds.items():
        per_frame.setdefault(fid, []).append(p["params"])
    uniq = {fid: [json.loads(s) for s in dict.fromkeys(json.dumps(p, sort_keys=True) for p in ps)] for fid, ps in per_frame.items()}
    metrics = {}
    with ProcessPoolExecutor(int(workers)) as ex:
        for fid, res in ex.map(evaluate, [(by_id[fid], ps) for fid, ps in uniq.items()]):
            for p, r in zip(uniq[fid], res):
                metrics[(fid, json.dumps(p, sort_keys=True))] = r
    print("evaluated", len(metrics), "renders", flush=True)
    table = {}
    for (e, v, m, fid), p in preds.items():
        r = dict(metrics[(fid, json.dumps(p["params"], sort_keys=True))], fold=p["fold"], alpha=p.get("alpha"),
                 params=p["params"], param_error=param_error(p["params"], by_id[fid]["labels"][v], v))
        row = table.setdefault((e, v), {}).setdefault(fid, dict(id=fid, roll=by_id[fid]["roll"], assumed=by_id[fid]["assumed"],
                                                                 label_de=by_id[fid]["label_de"][v], ub7_de=by_id[fid]["label_de"]["ub7"]))
        row[m] = r
    results = {}
    for (e, v), rows_d in table.items():
        rows = list(rows_d.values())
        res = dict(overall=summary(rows, by_id))
        if e.startswith("A"):
            res["tiers"] = {"1 M1 vs M0_median": compare(rows, "M1", "M0_median"), "2 M2 vs M1": compare(rows, "M2", "M1")}
            res["exploratory"] = {"M1 vs M0_fixed": compare(rows, "M1", "M0_fixed"), "M2 vs M0_fixed": compare(rows, "M2", "M0_fixed"),
                                  "M2 vs M0_median": compare(rows, "M2", "M0_median"),
                                  "M0_median vs M0_fixed": compare(rows, "M0_median", "M0_fixed")}
            t1 = passes(res["tiers"]["1 M1 vs M0_median"])
            res["conclusion"] = dict(tier1_pass=t1, tier2_pass=t1 and passes(res["tiers"]["2 M2 vs M1"]),
                                     note="tier 2 only counts if tier 1 passes (fixed sequence); A_b is an ablation, descriptive")
        else:
            res["per_roll"] = {roll: summary([r for r in rows if r["roll"] == roll], by_id) for roll in sorted({r["roll"] for r in rows})}
            res["without_color_space_assumed"] = summary([r for r in rows if not r["assumed"]], by_id)
            res["per_roll_mean_diff"] = {f"{a} - {b}": {roll: float(np.mean([r[a]["de_median"] - r[b]["de_median"] for r in rows if r["roll"] == roll]))
                                                         for roll in sorted({r["roll"] for r in rows})}
                                         for a, b in (("M1", "M0_median"), ("M2", "M1"), ("M2", "M0_fixed"))}
        results[f"{e}:{v}"] = res
        results[f"{e}:{v}"]["frames"] = rows
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=float))
    for k, res in results.items():
        o = res["overall"]
        print(k, "n", o["n"], " ".join(f"{m} {o[m]['de_median']:.2f}" for m in METHODS), "| label %.2f" % o["label"]["de_median"],
              "|", res.get("conclusion", ""), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
