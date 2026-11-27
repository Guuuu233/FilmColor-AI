"""Blind frozen-holdout benchmark for one FilmColor source tree (shipped estimate + render path).

usage: python eval_ho179.py SRC_DIR TAG CACHE_DIR OUT_JSON [WORKERS]
SRC_DIR holds src/filmcolor and scripts/eval_blueneg.py of the version under test.
Each cache npz (fc_holdout_cache_183) carries the decoded preview positive, the 4x4 estimation input,
the reference Lab image and its registration, so every version sees identical pixels.
"""
import gc
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

SRC, TAG, CACHE, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
WORKERS = int(sys.argv[5]) if len(sys.argv) > 5 else 4
sys.path.insert(0, str(Path(SRC) / "src"))
sys.path.insert(0, str(Path(SRC) / "scripts"))

import numpy as np  # noqa: E402

_M = {}


def _init():
    import filmcolor.frozen as ff
    ff.check_hashes = lambda *a, **k: None  # evaluation of the frozen holdout itself
    from filmcolor import colorimetry, compare, noref, pipeline, render
    from filmcolor.export import icc_profile
    from filmcolor.invert import invert_t18_from_positive
    from eval_blueneg import pair as pair_blueneg
    _M.update(colorimetry=colorimetry, compare=compare, noref=noref, pipeline=pipeline, render=render,
              inv18=invert_t18_from_positive, pair_blueneg=pair_blueneg,
              prof=colorimetry.MatrixTRCProfile(icc_profile("srgb")),
              cfg=getattr(noref, "NOREF_T25", noref.NOREF_T24))


def _load(p):
    with np.load(p, allow_pickle=False) as d:
        meta = json.loads(str(d["meta_json"]))

        def g(k, dt):
            return np.array(d[k], dtype=dt, copy=True) if (k in d.files and d[k].size > 0) else None
        return (meta, g("P_prev", np.float32), g("mask_prev", bool), g("P4", np.float32), g("mask4", bool),
                g("ref_small", np.float32), g("cam_xyz", np.float64))


def _metrics(disp, ref_small, meta):
    c, compare = _M["colorimetry"], _M["compare"]
    lab = c.xyz_to_lab(_M["prof"].to_xyz(disp.astype(np.float64)))
    al = meta["alignment"]
    if meta["ref_kind"] == "srgb_no_icc":
        a, r = _M["pair_blueneg"](lab, ref_small, al)
    else:
        a, r = compare.aligned_pair(lab, ref_small, compare.Alignment(
            factor=al["factor"], rot90=al["rot90"], scale=al["scale"], offset=tuple(al["offset"]), ncc=al["ncc"]))
    mt = compare.metrics(a, r)
    q20, q80 = np.percentile(r[..., 0], [20, 80])
    db = a[..., 2] - r[..., 2]
    cross = abs(float(np.median(db[r[..., 0] >= q80]) - np.median(db[r[..., 0] <= q20])))
    return dict(de50=round(float(mt["de2000_median"]), 3), de90=round(float(mt["de2000_p90"]), 3),
                dL=round(float(mt["mean_dL"]), 2), da=round(float(mt["mean_da"]), 2), db=round(float(mt["mean_db"]), 2),
                cr=None if mt["chroma_ratio"] is None else round(float(mt["chroma_ratio"]), 3), cross=round(cross, 3))


def _cfg_and_spec(cam):
    cfg = dict(_M["cfg"], camera_matrix_mix=0.35)
    if cam is not None:
        cfg["matrix"] = _M["render"].camera_color_matrix(cam, 0.35)
    return cfg, _M["pipeline"].t18_spec(cfg)


def _render(P_prev, mask_prev, inv, spec, params):
    rp = _M["render"].RenderParamsT24.from_dict(params)
    t18 = _M["inv18"](P_prev, inv["d_lo"], inv["d_hi"], float(inv.get("gamma", 0.6)),
                      has_cam_matrix=spec["has_cam_matrix"], version=spec["version"])
    disp, _ = _M["render"].render_t24(t18, rp, mask_prev)
    return disp


def eval_single(path):
    meta, P_prev, mask_prev, P4, mask4, ref_small, cam = _load(path)
    inv = meta["inversion"]
    cfg, spec = _cfg_and_spec(cam)
    est_in = (P4, mask4) if P4 is not None else (P_prev, mask_prev)
    info, rp, _ = _M["pipeline"].run_auto_estimate(est_in, cfg, d_lo=inv["d_lo"], d_hi=inv["d_hi"],
                                                    gamma=float(inv.get("gamma", 0.6)))
    params = rp.to_dict()
    disp = _render(P_prev, mask_prev, inv, spec, params)
    m = _metrics(disp, ref_small, meta)
    del P_prev, P4, ref_small, disp
    gc.collect()
    return dict(_src=str(path), fid=meta["id"], domain=meta["domain"], roll=meta["roll"], has_cam=cam is not None,
                branch=info.get("branch"), params=params, single=m)


def eval_pooled(item):
    path, params = item
    meta, P_prev, mask_prev, _, _, ref_small, cam = _load(path)
    _, spec = _cfg_and_spec(cam)
    disp = _render(P_prev, mask_prev, meta["inversion"], spec, params)
    m = _metrics(disp, ref_small, meta)
    del P_prev, ref_small, disp
    gc.collect()
    return m


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def main():
    t0 = time.time()
    files = sorted(str(p) for p in Path(CACHE).glob("*.npz"))
    assert len(files) == 179, f"expected 179 holdout caches, got {len(files)}"
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx, initializer=_init, max_tasks_per_child=25) as ex:
        rows = []
        for i, r in enumerate(ex.map(eval_single, files, chunksize=2), 1):
            rows.append(r)
            if i % 30 == 0 or i == len(files):
                print(f"[{TAG}] single {i}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    _init()
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["has_cam"], r["domain"], r["roll"]), []).append(i)
    pooled = [None] * len(rows)
    for idxs in groups.values():
        out = _M["noref"].pool_roll_params_t24([rows[i]["params"] for i in idxs])
        for i, p in zip(idxs, out):
            pooled[i] = dict(p)
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx, initializer=_init, max_tasks_per_child=25) as ex:
        for i, m in enumerate(ex.map(eval_pooled, [(rows[i]["_src"], pooled[i]) for i in range(len(rows))], chunksize=2)):
            rows[i]["pooled"] = m
    out_rows = [{k: v for k, v in r.items() if k != "_src"} for r in rows]
    for key in ("single", "pooled"):
        a = np.array([r[key]["de50"] for r in out_rows])
        print(f"[{TAG}] {key}: n={len(a)} med={np.median(a):.3f} mean={a.mean():.3f} p90={np.percentile(a, 90):.3f}", flush=True)
    Path(OUT).write_text(json.dumps(_jsonable(dict(tag=TAG, elapsed_s=round(time.time() - t0, 1), rows=out_rows)), indent=1))
    print(f"DONE_{TAG} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
