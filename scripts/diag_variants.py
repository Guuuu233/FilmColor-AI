"""Base-inversion diagnosis, step 3: inversion variants against both references (experiments/s3_diag).

One task per (frame, reference): shop print (S3 cache grid) and, where it exists, the user's Lightroom edit
(cache_user grid). The positive of every variant is computed from the T4 cache on that grid; only one link of the
chain changes per variant:

  film base (experiment 1)   B0   per-frame automatic base (the engine; the S3 labels' positive)
                             B1   roll base (diag_roll_base.py), black point = base (d_lo = 0)
                             B1c  roll base colour, neutral black point at the frame's own base level in G:
                                  d_lo = (c, c, c), c = log10(b_roll_G / b0_G); overall contrast as B0
  white point (exp. 2)       B0w4, B1cw4  the same base, white point taken on the 4x4 block means instead of full
                                  resolution (found in the A023 trial: full-resolution sensor noise raises the
                                  99.9th percentile of the low-signal blue channel by ~0.14 D)
  device colour (exp. 3)     MCt  camera matrix (camera RGB -> linear sRGB from the RAW's rgb_xyz_matrix, rows sum
                                  to 1) on T / b before the density
                             MCp  the same matrix on the positive P, before the render
                             (each on B0 and on B1c)
White points are the engine's (99.9th percentile of density in the image region at full resolution; exact for a
base change, see diag_build_T.py; for MCt computed at full resolution by diag_build_T.py) except the w4 variants. Per variant: fixed look (global_look_v1, no fit), P4 label (fit_config_s3_p4), ub7.
Diagnostic fits that see the reference (experiments 2 and 3), on B0 and B1c, from the ub7 optimum:
  chan_gamma  ub7 + per-channel contrast factors of the inversion (R, B; G = 1)
  p_matrix    ub7 + a 3x3 matrix on P (rows sum to 1, off-diagonal within +-0.8, condition <= 10)
Residual maps (ub7 and fixed look, B0 and B1c) are stored for the two-reference comparison.

  diag_variants.py FRAMES_JSON OUT_DIR [WORKERS] [ID ...]
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from diag_common import LOOK, camera_to_srgb, evaluate, fit_extra, grid_T, positive_T  # noqa: E402

import numpy as np  # noqa: E402
import rawpy  # noqa: E402

from filmcolor.compare import block_mean  # noqa: E402
from filmcolor.fit import FIT_CONFIG_V1, fit_reference, fit_reference_s3  # noqa: E402
from filmcolor.frozen import check_source, load_cache, save_cache  # noqa: E402
from filmcolor.invert import EPS, central_region  # noqa: E402
from filmcolor.paths import data_dir  # noqa: E402

FRAMES = ("ext:A002", "ext:A008", "ext:A019", "ext:A023", "ext:A140", "ext:A196")
ROLL_BASE = Path("experiments/s3_diag/roll_base.json")
SETS = {
    "main": dict(variants=("B0", "B1", "B1c", "B0w4", "B1cw4", "MCt_B0", "MCt_B1c"), extras_on=("B0", "B1c"), suffix=""),
    "b1": dict(variants=("B1", "B1w4", "MCt_B1"), extras_on=("B1",), suffix="_b1"),
}


def white_T4(T4, sat4, base, t_matrix=None, pct=99.9):
    t, l, h, w = central_region(T4.shape[:2])
    Tm = T4[t : t + h, l : l + w][sat4[t : t + h, l : l + w] == 0]
    x = Tm / np.asarray(base)
    if t_matrix is not None:
        x = x @ np.asarray(t_matrix).T
    D = -np.log10(np.maximum(x, EPS))
    return np.percentile(D, pct, 0), float((x <= 0).any(-1).mean())


def task(args):
    f, ref_kind, out_dir, vset = args
    t0 = time.time()
    stem = f["stem"]
    raw_sha = check_source(f["raw"])
    tdir = data_dir("s3_diag", "T4")
    T4 = load_cache(tdir / f"{stem}_T4.npy")
    sat4 = load_cache(tdir / f"{stem}_sat4.npy")
    trec = json.loads((tdir / f"{stem}.json").read_text())
    crec = json.loads((data_dir("s3_ml", "cache", f["domain"]) / f"{stem}.json").read_text())
    if ref_kind == "shop":
        cdir, al, T_in = data_dir("s3_ml", "cache", f["domain"]), crec["alignment"], T4
        ref_sha = f["ref_sha256"]
    else:
        ul = json.loads(Path(f"experiments/s3_ml/user_labels/{stem}.json").read_text())
        cdir, al, T_in = data_dir("s3_ml", "cache_user", f["domain"]), ul["alignment"], block_mean(T4, 2)
        ref_sha = ul["user_ref_sha256"]
    R = load_cache(cdir / f"{stem}_R8.npy").astype(np.float64)
    P_cached = load_cache(cdir / f"{stem}_P8.npy")
    shape = R.shape
    inv0 = crec["positive"]["inversion"]
    pct = np.array(trec["L_percentiles"]["99.9"])
    b0 = np.array(trec["base_cache"])
    roll = json.loads(ROLL_BASE.read_text())["rolls"][f["roll"]]["frames"][stem]
    broll = np.array(roll["base_roll"])

    def B0_P(T):
        return positive_T(T, inv0["base"], inv0["d_lo"], inv0["d_hi"])

    Tg, grid_info = grid_T(T_in, al, shape, P_cached, B0_P)
    T = Tg.reshape(-1, 3).astype(np.float64)
    R = R.reshape(-1, 3)
    Pc = P_cached.reshape(-1, 3)
    LG = -np.log10(np.maximum(T[:, 1], EPS))

    with rawpy.imread(f["raw"]) as r:
        C = camera_to_srgb(np.asarray(r.rgb_xyz_matrix)[:3])
    c = float(np.log10(broll[1] / b0[1]))
    inv = {
        "B0": dict(base=b0, d_lo=(0, 0, 0), d_hi=pct + np.log10(b0)),
        "B1": dict(base=broll, d_lo=(0, 0, 0), d_hi=pct + np.log10(broll)),
        "B1c": dict(base=broll, d_lo=(c, c, c), d_hi=pct + np.log10(broll)),
    }
    wp_B0_T4, _ = white_T4(T4, sat4, b0)
    for name in ("B0", "B1c", "B1"):
        # white point estimated on the 4x4 block means instead of full resolution (less sensor noise)
        inv[f"{name}w4"] = dict(inv[name], d_hi=white_T4(T4, sat4, inv[name]["base"])[0])
        mw = trec["mct_white"]["B0" if name == "B0" else "B1c"]  # B1 and B1c share the roll base (d_hi ignores d_lo)
        if not np.allclose(mw["base"], inv[name]["base"]):
            raise SystemExit(f"{f['id']}: camera-matrix white point computed for another base")
        inv[f"MCt_{name}"] = dict(inv[name], d_hi=np.array(mw["d_hi"]), t_matrix=C)
        inv[f"MCt_{name}"]["_nonpositive_frac"] = mw["nonpositive_frac"]
    out = dict(id=f["id"], roll=f["roll"], ref_kind=ref_kind, set=vset, grid=grid_info, grid_shape=list(shape),
               camera_to_srgb=C.tolist(), black_offset_B1c=c, white_point_T4_vs_full_B0=(wp_B0_T4 - inv["B0"]["d_hi"]).tolist(),
               variants={})
    # the T-first block mean against the S3 cache (P averaged after inversion)
    P_B0 = positive_T(T, **{k: v for k, v in inv["B0"].items() if not k.startswith("_")})
    m_cache, _ = evaluate(Pc, R, LOOK["params"])
    out["grid_check"] = dict(fixed_look_de_cache=m_cache["de2000_median"],
                             P_median_abs_log_ratio=float(np.median(np.abs(np.log(np.maximum(P_B0, 1e-4) / np.maximum(Pc, 1e-4))))))
    inv = {k: v for k, v in inv.items() if k in SETS[vset]["variants"]}
    extras_on = SETS[vset]["extras_on"]
    cfg = FIT_CONFIG_V1
    resid = {}
    for name, iv in inv.items():
        kw = {k: v for k, v in iv.items() if not k.startswith("_")}
        P = positive_T(T, **kw)
        res = dict(inversion={k: (np.asarray(v).tolist() if not isinstance(v, float) else v) for k, v in iv.items()})
        res["fixed_look"], lab_fl = evaluate(P, R, LOOK["params"], LG)
        p4 = fit_reference_s3(P, R, "p4", LOOK)
        ub7 = fit_reference(P, R, "7", lam=0.0, highlight_weight=cfg["highlight_weight"], area_weight=cfg["area_weight"])
        res["p4"], _ = evaluate(P, R, p4["params"], LG)
        res["p4"]["params"], res["p4"]["at_bound"] = p4["params"], p4["at_bound"]
        res["ub7"], lab_ub = evaluate(P, R, ub7["params"], LG)
        res["ub7"]["params"], res["ub7"]["at_bound"] = ub7["params"], ub7["at_bound"]
        if name in extras_on:
            resid[f"{name}_fixed_look"], resid[f"{name}_ub7"] = lab_fl - R, lab_ub - R
            for kind in ("chan_gamma", "p_matrix"):
                fx, Px = fit_extra(kind, T, kw, R, ub7["params"])
                m, _ = evaluate(Px, R, fx["params"], LG)
                res[f"ub7_{kind}"] = m | {k: v for k, v in fx.items() if k != "params"} | dict(params=fx["params"])
            # camera matrix on P (MCp): same inversion, P -> P C^T before the render
            Pm = (P @ C.T.astype(np.float32)).astype(np.float32)
            mres = dict(inversion=res["inversion"], p_matrix="camera_to_srgb")
            mres["fixed_look"], _ = evaluate(Pm, R, LOOK["params"], LG)
            p4m = fit_reference_s3(Pm, R, "p4", LOOK)
            ub7m = fit_reference(Pm, R, "7", lam=0.0, highlight_weight=cfg["highlight_weight"], area_weight=cfg["area_weight"])
            mres["p4"], _ = evaluate(Pm, R, p4m["params"], LG)
            mres["ub7"], _ = evaluate(Pm, R, ub7m["params"], LG)
            mres["p4"]["params"], mres["ub7"]["params"] = p4m["params"], ub7m["params"]
            out["variants"][f"MCp_{name}"] = mres
        out["variants"][name] = res
        print(f["id"], ref_kind, name, "fl %.2f p4 %.2f ub7 %.2f" % (res["fixed_look"]["de2000_median"],
              res["p4"]["de2000_median"], res["ub7"]["de2000_median"]), flush=True)
    rdir = data_dir("s3_diag", "resid")
    for k, v in resid.items():
        save_cache(rdir / f"{stem}_{ref_kind}_{k}.npy", v.reshape(shape).astype(np.float16), [raw_sha, ref_sha])
    out["seconds"] = round(time.time() - t0, 1)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"{stem}_{ref_kind}{SETS[vset]['suffix']}.json").write_text(json.dumps(out, indent=1, default=float))
    return f["id"], ref_kind, out["seconds"]


def main(frames_json, out_dir, workers="6", *ids):
    """``--set b1`` before the IDs: the follow-up set on the roll base B1 (added after the main set showed that B1c
    fails and B1 is the base to test further)."""
    vset = "main"
    if ids and ids[0] == "--set":
        vset, ids = ids[1], ids[2:]
    frames = {f["id"]: f for f in json.load(open(frames_json))["frames"]}
    jobs = []
    for i in ids or FRAMES:
        f = frames[i]
        jobs.append((f, "shop", out_dir, vset))
        if Path(f"experiments/s3_ml/user_labels/{f['stem']}.json").exists():
            jobs.append((f, "user", out_dir, vset))
    with ProcessPoolExecutor(int(workers)) as ex:
        for r in ex.map(task, jobs):
            print("done", *r, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
