"""Separating the colour and the level role of the film base (experiments/s3_separate/DESIGN.md), frames with references.

Base in log density: log10 b = l + c, l = mean over channels, c = per-channel deviation (sums to 0).
  B0    per-frame l, per-frame c   (the engine)
  B1    roll l, roll c             (full roll-base vector, as the two validation rounds)
  L0C1  per-frame l, roll c        (colour only; = B1c of the S3 diagnosis with the level anchored on the channel
                                    mean instead of G)
  L1C0  roll l, per-frame c        (level only)
Everything else as diag_rollcheck.py: black point = base (d_lo = 0), full-resolution 99.9 % white point of each
variant's own base, gamma 0.6. Per variant three tone readings: the fixed look (global_look_v1); level matched
(ev only, solved so that the median L* over the grid equals that of B0 under the fixed look; never reads the
reference); P4 (reference fit, diagnostic). Metrics: diag_common.evaluate plus the median a*b* distance.
Roll bases are the ones each earlier round computed (DIAG_ROLL_BASE); nothing is re-estimated.

  sep_check.py FRAMES_JSON ROLL_BASE_JSON OUT_DIR WORKERS ROLL [ROLL ...]
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from diag_common import LOOK, evaluate, grid_T, positive_T, render_lab  # noqa: E402

import numpy as np  # noqa: E402

from filmcolor.fit import fit_reference_s3  # noqa: E402
from filmcolor.frozen import check_source, load_cache  # noqa: E402
from filmcolor.invert import EPS  # noqa: E402
from filmcolor.paths import data_dir  # noqa: E402
from filmcolor.render import BOUNDS, RenderParams  # noqa: E402

VARIANTS = ("B0", "B1", "L0C1", "L1C0")


def split(b):
    lb = np.log10(np.asarray(b, np.float64))
    return lb.mean(), lb - lb.mean()


def bases(b0, broll):
    (l0, c0), (l1, c1) = split(b0), split(broll)
    return {"B0": b0, "B1": broll, "L0C1": 10 ** (l0 + c1), "L1C0": 10 ** (l1 + c0)}


def median_L(P, params):
    lab, _, _ = render_lab(P[None], RenderParams(**params), with_pre_clip=True)
    return float(np.median(lab[0][:, 0]))


def match_ev(P, target, iters=30):
    """ev such that the median L* equals target (bisection inside the ev bounds; clamped if out of reach)."""
    lo, hi = BOUNDS["ev"]
    p = dict(LOOK["params"])
    f = lambda ev: median_L(P, p | dict(ev=ev)) - target
    flo, fhi = f(lo), f(hi)
    if flo > 0 or fhi < 0:
        return (lo if flo > 0 else hi), True
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi), False


def ab_distance(P, R, params):
    lab, _, _ = render_lab(P[None], RenderParams(**params), with_pre_clip=True)
    return float(np.median(np.hypot(*(lab[0][:, 1:] - R[:, 1:]).T)))


def task(args):
    f, ref_kind, roll_base, out_dir = args
    t0 = time.time()
    stem = f["stem"]
    check_source(f["raw"])
    trec = json.loads((data_dir("s3_diag", "T8") / f"{stem}.json").read_text())
    T8 = load_cache(data_dir("s3_diag", "T8") / f"{stem}_T8.npy")
    crec = json.loads((data_dir("s3_ml", "cache", f["domain"]) / f"{stem}.json").read_text())
    if ref_kind == "shop":
        cdir, al = data_dir("s3_ml", "cache", f["domain"]), dict(crec["alignment"])
        if al["factor"] != 2:
            raise SystemExit(f"{f['id']}: expected the S3 alignment factor 2 (P4 grid)")
        al["factor"] = 1
    else:
        cdir = data_dir("s3_ml", "cache_user", f["domain"])
        al = json.loads(Path(f"experiments/s3_ml/user_labels/{stem}.json").read_text())["alignment"]
    R = load_cache(cdir / f"{stem}_R8.npy").astype(np.float64)
    P_cached = load_cache(cdir / f"{stem}_P8.npy")
    inv0 = crec["positive"]["inversion"]
    Tg, grid_info = grid_T(T8, al, R.shape, P_cached, lambda T: positive_T(T, inv0["base"], inv0["d_lo"], inv0["d_hi"]))
    T, R = Tg.reshape(-1, 3).astype(np.float64), R.reshape(-1, 3)
    LG = -np.log10(np.maximum(T[:, 1], EPS))
    b0 = np.array(trec["base_cache"])
    broll = np.array(json.loads(Path(roll_base).read_text())["rolls"][f["roll"]]["frames"][stem]["base_roll"])
    full = np.array(trec["L_percentiles"]["99.9"])
    out = dict(id=f["id"], roll=f["roll"], ref_kind=ref_kind, grid=grid_info, variants={},
               own_minus_roll_density=np.log10(broll / b0).tolist(),
               level_diff=float(split(b0)[0] - split(broll)[0]),
               chroma_diff=(split(b0)[1] - split(broll)[1]).tolist())
    target = None
    for name, b in bases(b0, broll).items():
        d_hi = full + np.log10(b)
        P = positive_T(T, b, (0, 0, 0), d_hi)
        res = dict(base=b.tolist(), d_hi=d_hi.tolist(), below_black=[float(x) for x in (P < 0).mean(0)],
                   below_black_any=float((P < 0).any(-1).mean()))
        res["fixed_look"], _ = evaluate(P, R, LOOK["params"], LG)
        res["fixed_look"]["ab_median"] = ab_distance(P, R, LOOK["params"])
        if target is None:  # B0 comes first
            target = median_L(P, LOOK["params"])
        ev, clamped = match_ev(P, target)
        pm = LOOK["params"] | dict(ev=ev)
        res["matched"], _ = evaluate(P, R, pm, LG)
        res["matched"]["ab_median"] = ab_distance(P, R, pm)
        res["matched"].update(ev=ev, ev_clamped=clamped)
        p4 = fit_reference_s3(P, R, "p4", LOOK)
        res["p4"], _ = evaluate(P, R, p4["params"], LG)
        res["p4"]["ab_median"] = ab_distance(P, R, p4["params"])
        res["p4"]["params"] = p4["params"]
        out["variants"][name] = res
    out["target_L"] = target
    out["seconds"] = round(time.time() - t0, 1)
    (Path(out_dir) / f"{stem}_{ref_kind}.json").write_text(json.dumps(out, indent=1, default=float))
    v = out["variants"]
    return f["id"], ref_kind, " ".join("%s m %.2f" % (k, v[k]["matched"]["de2000_median"]) for k in v), out["seconds"]


def main(frames_json, roll_base, out_dir, workers, *rolls):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    jobs = []
    for f in json.load(open(frames_json))["frames"]:
        if f["roll"] not in rolls or not (data_dir("s3_diag", "T8") / f"{f['stem']}.json").exists():
            continue
        kinds = ["shop"] + (["user"] if Path(f"experiments/s3_ml/user_labels/{f['stem']}.json").exists() else [])
        jobs += [(f, k, roll_base, out_dir) for k in kinds if not (Path(out_dir) / f"{f['stem']}_{k}.json").exists()]
    print(len(jobs), "tasks to do", flush=True)
    with ProcessPoolExecutor(int(workers)) as ex:
        for r in ex.map(task, jobs):
            print(*r, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
