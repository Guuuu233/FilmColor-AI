"""T4c evaluation: the app's automatic starting point vs the reference print, and negative density ratios of
reference-neutral / reference-blue-sky pixels (the evidence for a roll neutral white-point ratio).

  t4c_ref_eval.py SRC_DIR OUT_JSON ROLL_FOLDER [--cfg=JSON] [--ratio=roll|frame] REL=REF_JPG [REL=REF_JPG ...]

``--cfg``: NOREF_V3 entries overridden for this run (variants; recorded in the output). ``--cam-mix``: the app's
camera matrix share (filmcolor.app.CAMERA_MATRIX_MIX) for this run. ``--ratio`` (T4c code only):
open with ``white_ratio`` = the roll's neutral density ratio (``roll``, as /api/roll reports it) or the frame's own
0.25 <= D_G <= 0.75 ratio on its cached blocks with the base it opens with (``frame``); the app does not do this.

SRC_DIR: the ``src`` folder whose ``filmcolor`` package is used (a ``git worktree`` of an earlier commit, or this
repo's src), so before / after are measured with the same script. Per frame:

- the frame is opened as the app opens it (State.open, automatic base choice) with a 1000 px preview, rendered with
  its automatic parameters (``pv.auto["params"]``) and converted to CIELAB D50 with the export's sRGB ICC profile;
- the reference (embedded ICC, checked against the frozen sets first) is block-averaged to the preview's size and
  aligned on L* (filmcolor.compare.align; rotation, scale ``SCALES``, translation);
- ``metrics``: filmcolor.compare.metrics on the aligned grids (median / p90 dE2000, mean dL*, da*, db*, chroma ratio);
- ``density_ratios``: the preview positive is turned back into density above the base with the inversion's own
  parameters (exact inverse of invert.invert, d_lo = 0), and medians of D_R / D_G and D_B / D_G are taken over
  (a) pixels the reference shows near-neutral (C* < 5, 45 < L* < 92), (b) pixels it shows as blue sky
  (b* < -12, L* > 60, -6 < a* < 8), (c) all pixels with 0.25 <= D_G <= 0.75, and (d) the white point d_hi itself.
The references are dev-set prints (user edits / shop scans); nothing here is read by the estimator.
"""
import json
import subprocess
import sys
from pathlib import Path

SRC = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

import filmcolor  # noqa: E402
from filmcolor.app import State  # noqa: E402
from filmcolor.colorimetry import MatrixTRCProfile, xyz_to_lab  # noqa: E402
from filmcolor.compare import align, aligned_pair, block_mean, load_lab, metrics, resize_bilinear  # noqa: E402
from filmcolor.export import icc_profile  # noqa: E402
from filmcolor.frozen import check_source  # noqa: E402

SCALES = np.arange(0.80, 1.0301, 0.01)


def density_of(P, inv):
    d_hi = np.asarray(inv["d_hi"], np.float64)
    R = (d_hi[1] - inv["d_lo"][1]) / inv["gamma"]
    f = 10.0 ** -R
    arg = np.maximum(P.astype(np.float64) / inv["exposure"] * (1 - f) + f, 1e-12)
    return d_hi * (1 + np.log10(arg) / R)


DG_BINS = (0.05, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5)


def ratio_stats(D, sel):
    if sel.sum() < 30:
        return dict(n=int(sel.sum()))
    d = D[sel]
    ok = d[:, 1] > 0.05
    d = d[ok]
    out = dict(n=int(len(d)), r=float(np.median(d[:, 0] / d[:, 1])), b=float(np.median(d[:, 2] / d[:, 1])),
               d_g_median=float(np.median(d[:, 1])), by_d_g=[])
    for lo, hi in zip(DG_BINS[:-1], DG_BINS[1:]):  # ratios per D_G band: they change with density (toe, shoulder)
        b = d[(d[:, 1] >= lo) & (d[:, 1] < hi)]
        if len(b) >= 30:
            out["by_d_g"].append(dict(d_g=[lo, hi], n=int(len(b)), r=float(np.median(b[:, 0] / b[:, 1])),
                                      b=float(np.median(b[:, 2] / b[:, 1]))))
    return out


def frame_ratio(st, rel):
    """The frame's 0.25 <= D_G <= 0.75 density ratio on its cached blocks, densities above the base it opens with."""
    fe = st.roll_cache["."]["frames"][rel]
    flagged = fe.get("own_error") or (fe.get("chroma_part") or 0) >= 0.1
    base = st._roll_base_kwargs(rel, "roll_nat", 0.5)["base_rgb"] if flagged else fe["own_rgb"]
    return st._density_ratio(st.frame_scans[rel], base)


def main(out_json: str, folder: str, pairs: list[str]):
    cfg_over, ratio_mode = {}, None
    while pairs and pairs[0].startswith("--"):
        opt = pairs.pop(0)
        if opt.startswith("--cfg="):
            cfg_over = json.loads(opt.split("=", 1)[1])
        elif opt.startswith("--ratio="):
            ratio_mode = opt.split("=", 1)[1]
        elif opt.startswith("--cam-mix="):
            import filmcolor.app as fc_app
            fc_app.CAMERA_MATRIX_MIX = float(opt.split("=", 1)[1])
            cfg_over["_cam_mix"] = fc_app.CAMERA_MATRIX_MIX
    over = {k: v for k, v in cfg_over.items() if not k.startswith("_")}
    if over:
        from filmcolor.noref import NOREF_V3
        unknown = set(over) - set(NOREF_V3)
        assert not unknown, unknown
        NOREF_V3.update({k: tuple(v) if isinstance(v, list) else v for k, v in over.items()})
    assert Path(filmcolor.__file__).resolve().is_relative_to(SRC), filmcolor.__file__
    commit = subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(SRC), "status", "--porcelain", "--", "."], capture_output=True,
                                text=True).stdout.strip())
    prof = MatrixTRCProfile(icc_profile("srgb"))
    st = State(Path(folder), Path("/tmp/fc_t4c_out"), allow_frozen=False, max_side=1000)
    st.roll_info()
    res = dict(src=str(SRC), git_commit=commit, src_dirty=dirty, cfg_override=cfg_over, ratio_mode=ratio_mode,
               frames={})
    if ratio_mode:
        orig = st._colour_kwargs
        nr = st.roll_cache["."]["neutral_ratio"]["ratio"]

        def with_ratio(p, film, _rel=[None]):
            kw = orig(p, film)
            fr = frame_ratio(st, _rel[0]) if ratio_mode == "frame" else dict(r=nr[0], b=nr[1])
            return dict(kw, white_ratio=(fr["r"], fr["b"]) if fr else None)
        st._colour_kwargs = with_ratio
    for pair in pairs:
        rel, ref = pair.split("=", 1)
        check_source(Path(ref))
        if ratio_mode:
            st._colour_kwargs.__defaults__[0][0] = rel
        m = st.open(rel, "color")
        pv = st.preview
        lab = xyz_to_lab(prof.to_xyz(pv.render(pv.auto["params"]).astype(np.float64)))
        ref_full = load_lab(ref, factor=1 if max(lab.shape[:2]) * 2 > 8000 else 2)
        f = max(1, int(round(max(ref_full.shape[:2]) / max(lab.shape[:2]))))
        ref_lab = block_mean(ref_full, f) if f > 1 else ref_full
        del ref_full
        al = align(lab[..., 0], ref_lab[..., 0], 1, scales=SCALES)
        a, r = aligned_pair(lab, ref_lab, al)
        y, x = al.offset
        P = pv.positive[y : y + r.shape[0], x : x + r.shape[1]]
        D = density_of(P, pv.inversion).reshape(-1, 3)
        rr = r.reshape(-1, 3)
        C = np.hypot(rr[:, 1], rr[:, 2])
        neutral = (C < 5) & (rr[:, 0] > 45) & (rr[:, 0] < 92)
        sky = (rr[:, 2] < -12) & (rr[:, 0] > 60) & (rr[:, 1] > -6) & (rr[:, 1] < 8)
        mid = (D[:, 1] >= 0.25) & (D[:, 1] <= 0.75)
        d_hi = pv.inversion["d_hi"]
        res["frames"][rel] = dict(
            ref=Path(ref).name, base_method=m["base"]["method"], params=pv.auto["params"], flags=pv.auto.get("flags"),
            white_point_info=pv.report.get("white_point"), backlit_check=pv.auto["diagnostics"].get("backlit_check"),
            alignment=al.to_dict(), metrics=metrics(a, r),
            density_ratios=dict(ref_neutral=ratio_stats(D, neutral), ref_blue_sky=ratio_stats(D, sky),
                                mid_density=ratio_stats(D, mid),
                                white_point=dict(r=d_hi[0] / d_hi[1], b=d_hi[2] / d_hi[1], d_hi=list(d_hi))))
        fr = res["frames"][rel]
        print(rel, m["base"]["method"], "ncc %.3f" % al.ncc, "dE %.2f dL %+.1f chroma %.2f" % (
            fr["metrics"]["de2000_median"], fr["metrics"]["mean_dL"], fr["metrics"]["chroma_ratio"] or float("nan")),
            {k: (round(v.get("r", np.nan), 3), round(v.get("b", np.nan), 3), v.get("n")) for k, v in fr["density_ratios"].items()},
            flush=True)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[2], sys.argv[3], sys.argv[4:])
