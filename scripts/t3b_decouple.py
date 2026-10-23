"""T3b (task list & confirmation 2026-10-23): black-point / film-base decoupling experiment.

Implements the pre-registered design of experiments/t3_inversion/DESIGN.md without changing the engine:
every variant is an effective base vector b_eff passed to diag_common.positive_T(T, b_eff, (0, 0, 0), p_999 + log10 b_eff).

  t3b_decouple.py OUT_DIR [--p4 WORKERS]

Steps:
  1. Bit-identity check on 3 frames (A008_shop, A140_shop, 19941217A-08-tokyo_shop) against experiments/s3_separate/check/.
  2. Layer 1 (physical measured-base reference from T3a) across the three confirmed scopes:
       primary_25   strictly passing e4dc94b boxes (25 frames; formal pass-criterion scope)
       expanded_28  primary_25 + the 3 remaining A7R5 frames with their clean rebate boxes (28 frames)
       clean_26     expanded_28 minus the 2 physically anomalous frames A1006 and A252 (26 frames)
     and roll-interior chroma c_roll_int on both primary_6 (A7R5 3 + X-T2 3) and expanded_9 (A7R5 6 + X-T2 3).
  3. Layer 2 (no-reference diagnostics) & Layer 3 (reference comparisons) on all 315 tasks of s3_separate/check/,
     reusing B0 / B1 / L0C1 (= L0C1_hard) and computing DQ(c_roll) and DQ(c0). With --p4 WORKERS, also fits P4
     for DQ(c_roll) in parallel.
  4. Evaluates the 5 pre-registered pass criteria of DESIGN.md section 4 and writes OUT_DIR/layer1.json,
     OUT_DIR/summary.json, and representative side-by-side preview sheets in OUT_DIR/previews/t3b/.
"""
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from diag_common import LOOK, evaluate, positive_T, render_lab  # noqa: E402
from rebate_verify import dD, parts, rect  # noqa: E402
from sep_check import ab_distance, match_ev, median_L, split  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from filmcolor.base import estimate_base_roll  # noqa: E402
from filmcolor.colorimetry import delta_e2000  # noqa: E402
from filmcolor.frozen import check_hashes, check_name, check_source, load_cache  # noqa: E402
from filmcolor.fit import fit_reference_s3  # noqa: E402
from filmcolor.paths import data_dir  # noqa: E402
from filmcolor.render import RenderParams, render as render_image  # noqa: E402

DOM = {"blueneg": "blueneg", "shop": "shop", "ext": "ext_own"}
EXCLUDE_L23 = {"blueneg:199807A-21-orlando-disney"}  # NCC 0.575 < 0.7, as sep_summary.py
ANOMALOUS_L1 = {"A1006", "A252"}
A7_CLEAN_EXTRA = {
    "A7R5_RAW_DSC01688": {"right": [96.0, 99.0, 10, 90]},
    "A7R5_RAW_DSC01692": {"left": [2.8, 5.0, 10, 85]},
    "A7R5_RAW_DSC01694": {"left": [4.0, 7.0, 10, 85]},
}
SPLIT = 0.1


def dq_base(c_vec, p01):
    """Effective base for DQ(c*): d_lo = min_c(p01_c + c*_c), b_eff = 10^(c* - d_lo)."""
    c_vec, p01 = np.asarray(c_vec, np.float64), np.asarray(p01, np.float64)
    d_lo = float(np.min(p01 + c_vec))
    return 10.0 ** (c_vec - d_lo), d_lo


def invert_safe(T, b_eff, p999):
    b_eff, p999 = np.asarray(b_eff, np.float64), np.asarray(p999, np.float64)
    d_hi = p999 + np.log10(b_eff)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        P = positive_T(T, b_eff, (0, 0, 0), d_hi)
    if not np.isfinite(P).all():
        return None, d_hi, "positive not finite (white point too close to black in a channel)"
    return P, d_hi, None


def check_bit_identity():
    checked = []
    for stem, rk in (("A008", "shop"), ("A140", "shop"), ("19941217A-08-tokyo", "shop")):
        old = json.loads(Path(f"experiments/s3_separate/check/{stem}_{rk}.json").read_text())
        trec = json.loads((data_dir("s3_diag", "T8") / f"{stem}.json").read_text())
        check_source(trec["raw"])
        T8 = load_cache(data_dir("s3_diag", "T8") / f"{stem}_T8.npy")
        domain = DOM[old["id"].split(":")[0]]
        cdir = data_dir("s3_ml", "cache" if rk == "shop" else "cache_user", domain)
        R = load_cache(cdir / f"{stem}_R8.npy").astype(np.float64)
        y0, x0 = old["grid"]["start"]
        T = T8[y0 : y0 + R.shape[0], x0 : x0 + R.shape[1]].reshape(-1, 3).astype(np.float64)
        R = R.reshape(-1, 3)
        b0, broll = np.array(trec["base_cache"]), np.array(old["variants"]["B1"]["base"])
        (l0, c0), (_, c1) = split(b0), split(broll)
        p999 = np.array(trec["L_percentiles"]["99.9"])
        for vname, beff in (("B0", b0), ("B1", broll), ("L0C1", 10.0 ** (l0 + c1))):
            P, d_hi, err = invert_safe(T, beff, p999)
            assert err is None
            m_fix, _ = evaluate(P, R, LOOK["params"])
            ev, _ = match_ev(P, old["target_L"])
            m_mat, _ = evaluate(P, R, LOOK["params"] | dict(ev=ev))
            ref_v = old["variants"][vname]
            assert np.allclose(d_hi, ref_v["d_hi"], atol=0, rtol=0)
            assert m_fix["de2000_median"] == ref_v["fixed_look"]["de2000_median"]
            assert m_mat["de2000_median"] == ref_v["matched"]["de2000_median"]
        checked.append(f"{stem}_{rk}")
    return checked


def eval_l1_variant(T, b_eff, p999, bm_fix_lab, bm_target_L, is_bm=False):
    P, d_hi, err = invert_safe(T, b_eff, p999)
    if err is not None:
        return dict(base=np.asarray(b_eff, float).tolist(), d_hi=d_hi.tolist(), undefined=True, reason=err)
    if is_bm:
        return dict(
            base=np.asarray(b_eff, float).tolist(), d_hi=d_hi.tolist(),
            below_black=[float(x) for x in (P < 0).mean(0)],
            below_black_any=float((P < 0).any(-1).mean()),
            clipped_frac_fixed=0.0,
            fixed_L_median=bm_target_L, fixed_abs_dL_vs_BM=0.0,
            fixed_dE_vs_BM=0.0, fixed_ab_vs_BM=0.0,
            ev_matched=float(LOOK["params"]["ev"]), ev_clamped=False,
            matched_dE_vs_BM=0.0, matched_ab_vs_BM=0.0,
        )
    lab_fix, _, pre_fix = render_lab(P[None], RenderParams(**LOOK["params"]), with_pre_clip=True)
    L_fix = float(np.median(lab_fix[0][:, 0]))
    ev, clamped = match_ev(P, bm_target_L)
    lab_mat, _, _ = render_lab(P[None], RenderParams(**(LOOK["params"] | dict(ev=ev))), with_pre_clip=True)
    return dict(
        base=np.asarray(b_eff, float).tolist(), d_hi=d_hi.tolist(),
        below_black=[float(x) for x in (P < 0).mean(0)],
        below_black_any=float((P < 0).any(-1).mean()),
        clipped_frac_fixed=float((pre_fix[0] > 1).any(-1).mean()),
        fixed_L_median=L_fix,
        fixed_abs_dL_vs_BM=abs(L_fix - bm_target_L),
        fixed_dE_vs_BM=float(np.median(delta_e2000(lab_fix[0], bm_fix_lab))),
        fixed_ab_vs_BM=float(np.median(np.hypot(*(lab_fix[0][:, 1:] - bm_fix_lab[:, 1:]).T))),
        ev_matched=float(ev), ev_clamped=bool(clamped),
        matched_dE_vs_BM=float(np.median(delta_e2000(lab_mat[0], bm_fix_lab))),
        matched_ab_vs_BM=float(np.median(np.hypot(*(lab_mat[0][:, 1:] - bm_fix_lab[:, 1:]).T))),
    )


def l1_frame_worker(args):
    stem, r, roll_targets = args
    cache = data_dir("t3_inversion", "T8")
    T8 = load_cache(cache / f"{stem}_T8.npy").astype(np.float64)
    S8 = load_cache(cache / f"{stem}_S8.npy")
    box_T = rect(T8, r["image_box"])
    box_S = rect(S8, r["image_box"])
    T = box_T[box_S == 0].reshape(-1, 3)
    b0 = np.array(r["b_int"], np.float64)
    b_meas = np.array(r["b_meas_used"], np.float64)
    (l0, c0), (l_m, c_m) = split(b0), split(b_meas)
    p01 = np.array(r["L_percentiles_box"]["0.1"], np.float64)
    p999 = np.array(r["L_percentiles_box"]["99.9"], np.float64)

    P_bm, d_hi_bm, err_bm = invert_safe(T, b_meas, p999)
    if err_bm is not None:
        raise RuntimeError(f"{stem}: BM undefined: {err_bm}")
    lab_bm_fix, _, pre_bm = render_lab(P_bm[None], RenderParams(**LOOK["params"]), with_pre_clip=True)
    bm_fix_lab = lab_bm_fix[0]
    bm_target_L = float(np.median(bm_fix_lab[:, 0]))

    beff_dq_m, d_lo_m = dq_base(c_m, p01)
    beff_dq_0, d_lo_0 = dq_base(c0, p01)
    v_meas = {
        "BM": eval_l1_variant(T, b_meas, p999, bm_fix_lab, bm_target_L, is_bm=True)
        | {"clipped_frac_fixed": float((pre_bm[0] > 1).any(-1).mean())},
        "B0": eval_l1_variant(T, b0, p999, bm_fix_lab, bm_target_L),
        "L0C1_hard": eval_l1_variant(T, 10.0 ** (l0 + c_m), p999, bm_fix_lab, bm_target_L),
        "DQ(c_meas)": eval_l1_variant(T, beff_dq_m, p999, bm_fix_lab, bm_target_L) | {"d_lo_shared": d_lo_m},
        "DQ(c0)": eval_l1_variant(T, beff_dq_0, p999, bm_fix_lab, bm_target_L) | {"d_lo_shared": d_lo_0},
    }
    v_roll = {}
    for tag, b_ri_list in roll_targets:
        b_ri = np.array(b_ri_list, np.float64)
        _, c_ri = split(b_ri)
        beff_dq_ri, d_lo_ri = dq_base(c_ri, p01)
        v_roll[tag] = {
            "b_roll_int": b_ri.tolist(),
            "roll_int_vs_meas": parts(dD(b_ri, b_meas)),
            "B0": v_meas["B0"],
            "B1": eval_l1_variant(T, b_ri, p999, bm_fix_lab, bm_target_L),
            "L0C1_hard": eval_l1_variant(T, 10.0 ** (l0 + c_ri), p999, bm_fix_lab, bm_target_L),
            "DQ(c_roll_int)": eval_l1_variant(T, beff_dq_ri, p999, bm_fix_lab, bm_target_L)
            | {"d_lo_shared": d_lo_ri},
            "DQ(c0)": v_meas["DQ(c0)"],
        }
    return stem, dict(
        id=r["id"], camera=r["camera"], source=r["source"], scope_origin=r["scope_origin"],
        anomalous=r["id"] in ANOMALOUS_L1,
        int_vs_meas=r["int_vs_meas"], own_vs_meas=r["own_vs_meas"],
        bm_target_L=bm_target_L, variants_c_meas=v_meas, variants_c_roll_int=v_roll)


def run_layer1(rv_path, n_Pool=8):
    rv = json.loads(Path(rv_path).read_text())
    frames = rv["frames"]
    cache = data_dir("t3_inversion", "T8")

    selected = {}
    for stem, r in sorted(frames.items()):
        if r["usable"]:
            selected[stem] = dict(r, scope_origin="primary_25", b_meas_used=r["b_meas"],
                                  rebate_boxes_used=r["rebate_boxes"])
        elif stem in A7_CLEAN_EXTRA:
            T8 = load_cache(cache / f"{stem}_T8.npy").astype(np.float64)
            boxes = A7_CLEAN_EXTRA[stem]
            pooled = np.concatenate([rect(T8, box).reshape(-1, 3) for box in boxes.values()])
            b_meas = np.median(pooled, 0).tolist()
            selected[stem] = dict(
                r, scope_origin="expanded_a7r5", b_meas_used=b_meas, rebate_boxes_used=boxes,
                int_vs_meas=parts(dD(r["b_int"], b_meas)), own_vs_meas=parts(dD(r["b_own"], b_meas)))

    a7_stems_3 = sorted(s for s in selected if s.startswith("A7R5_RAW_") and selected[s]["scope_origin"] == "primary_25")
    a7_stems_6 = sorted(s for s in selected if s.startswith("A7R5_RAW_"))
    xt2_stems_3 = ["A1010", "A1011", "A1012"]
    rb_a7_3 = estimate_base_roll({s: selected[s]["b_int"] for s in a7_stems_3},
                                 {s: float(selected[s]["exposure_time"]) for s in a7_stems_3})
    rb_a7_6 = estimate_base_roll({s: selected[s]["b_int"] for s in a7_stems_6},
                                 {s: float(selected[s]["exposure_time"]) for s in a7_stems_6})
    rb_xt2_3 = estimate_base_roll({s: selected[s]["b_int"] for s in xt2_stems_3},
                                  {s: float(selected[s]["exposure_time"]) for s in xt2_stems_3})

    jobs = []
    for stem, r in sorted(selected.items()):
        rt = []
        if stem in a7_stems_3:
            rt.append(("a7_3", list(rb_a7_3.for_exposure(float(r["exposure_time"])))))
        if stem in a7_stems_6:
            rt.append(("a7_6", list(rb_a7_6.for_exposure(float(r["exposure_time"])))))
        if stem in xt2_stems_3:
            rt.append(("xt2_3", list(rb_xt2_3.for_exposure(float(r["exposure_time"])))))
        jobs.append((stem, r, rt))

    l1_rows = {}
    with ProcessPoolExecutor(n_Pool) as ex:
        for stem, row in ex.map(l1_frame_worker, jobs):
            l1_rows[stem] = row

    def summarize_scope(stems):
        rs = [l1_rows[s] for s in stems]
        dev_b0 = [r for r in rs if r["variants_c_meas"]["B0"]["matched_dE_vs_BM"] >= 2.0]
        dense_b0 = [r for r in rs if r["int_vs_meas"]["level"] >= SPLIT]
        out = dict(n=len(rs), n_B0_matched_dE_ge_2=len(dev_b0), n_int_level_ge_0_1=len(dense_b0), variants={})
        for v in ("BM", "B0", "L0C1_hard", "DQ(c_meas)", "DQ(c0)"):
            ok = [r for r in rs if not r["variants_c_meas"][v].get("undefined")]
            bb = [r["variants_c_meas"][v]["below_black_any"] for r in ok]
            f_de = [r["variants_c_meas"][v]["fixed_dE_vs_BM"] for r in ok]
            f_ab = [r["variants_c_meas"][v]["fixed_ab_vs_BM"] for r in ok]
            f_dl = [r["variants_c_meas"][v]["fixed_abs_dL_vs_BM"] for r in ok]
            m_de = [r["variants_c_meas"][v]["matched_dE_vs_BM"] for r in ok]
            m_ab = [r["variants_c_meas"][v]["matched_ab_vs_BM"] for r in ok]
            d = dict(
                n=len(rs), undefined=len(rs) - len(ok),
                below_black_median=round(float(np.median(bb)), 5),
                below_black_max=round(float(np.max(bb)), 5),
                fixed_dE_median=round(float(np.median(f_de)), 3),
                fixed_dE_iqr=[round(float(x), 3) for x in np.percentile(f_de, [25, 75])],
                fixed_ab_median=round(float(np.median(f_ab)), 3),
                fixed_abs_dL_median=round(float(np.median(f_dl)), 3),
                matched_dE_median=round(float(np.median(m_de)), 3),
                matched_dE_iqr=[round(float(x), 3) for x in np.percentile(m_de, [25, 75])],
                matched_ab_median=round(float(np.median(m_ab)), 3),
            )
            if v != "BM" and dev_b0:
                diffs = [
                    (float("inf") if r["variants_c_meas"][v].get("undefined")
                     else r["variants_c_meas"][v]["matched_dE_vs_BM"])
                    - r["variants_c_meas"]["B0"]["matched_dE_vs_BM"]
                    for r in dev_b0
                ]
                fin = [x for x in diffs if np.isfinite(x)]
                d["on_B0_dE_ge_2"] = dict(
                    n=len(dev_b0),
                    matched_dE_median=round(float(np.median([
                        r["variants_c_meas"][v]["matched_dE_vs_BM"] for r in dev_b0
                        if not r["variants_c_meas"][v].get("undefined")
                    ])), 3),
                    improved=int(sum(x < 0 for x in diffs)),
                    improved_share=round(float(np.mean([x < 0 for x in diffs])), 4),
                    median_drop=round(float(np.median([-x for x in fin])), 3) if fin else None,
                )
            if v != "BM" and dense_b0:
                d["on_int_level_ge_0_1"] = dict(
                    n=len(dense_b0),
                    fixed_dE_median=round(float(np.median([
                        r["variants_c_meas"][v]["fixed_dE_vs_BM"] for r in dense_b0
                        if not r["variants_c_meas"][v].get("undefined")
                    ])), 3),
                    matched_dE_median=round(float(np.median([
                        r["variants_c_meas"][v]["matched_dE_vs_BM"] for r in dense_b0
                        if not r["variants_c_meas"][v].get("undefined")
                    ])), 3),
                    below_black_median=round(float(np.median([
                        r["variants_c_meas"][v]["below_black_any"] for r in dense_b0
                        if not r["variants_c_meas"][v].get("undefined")
                    ])), 5),
                )
            out["variants"][v] = d
        return out

    def summarize_roll_group(stems, tag):
        rs = [l1_rows[s]["variants_c_roll_int"][tag] for s in stems]
        out = dict(n=len(rs), frames=stems, variants={})
        for v in ("B0", "B1", "L0C1_hard", "DQ(c_roll_int)", "DQ(c0)"):
            ok = [r[v] for r in rs if not r[v].get("undefined")]
            out["variants"][v] = dict(
                n=len(rs), undefined=len(rs) - len(ok),
                below_black_median=round(float(np.median([x["below_black_any"] for x in ok])), 5),
                below_black_max=round(float(np.max([x["below_black_any"] for x in ok])), 5),
                fixed_dE_median=round(float(np.median([x["fixed_dE_vs_BM"] for x in ok])), 3),
                fixed_ab_median=round(float(np.median([x["fixed_ab_vs_BM"] for x in ok])), 3),
                fixed_abs_dL_median=round(float(np.median([x["fixed_abs_dL_vs_BM"] for x in ok])), 3),
                matched_dE_median=round(float(np.median([x["matched_dE_vs_BM"] for x in ok])), 3),
                matched_ab_median=round(float(np.median([x["matched_ab_vs_BM"] for x in ok])), 3),
            )
        return out

    p25 = sorted(s for s, r in l1_rows.items() if r["scope_origin"] == "primary_25")
    e28 = sorted(l1_rows)
    c26 = sorted(s for s, r in l1_rows.items() if not r["anomalous"])
    scopes = dict(
        primary_25=summarize_scope(p25),
        expanded_28=summarize_scope(e28),
        clean_26=summarize_scope(c26),
    )
    roll_scopes = dict(
        a7r5_primary_3=summarize_roll_group(a7_stems_3, "a7_3"),
        a7r5_expanded_6=summarize_roll_group(a7_stems_6, "a7_6"),
        xt2_20180613_3=summarize_roll_group(xt2_stems_3, "xt2_3"),
        pooled_primary_6=summarize_roll_group(a7_stems_3 + xt2_stems_3,
                                              None) if False else None,
    )
    # Pooled roll summaries (primary_6 = a7_3 + xt2_3; expanded_9 = a7_6 + xt2_3)
    for pname, pairs in (("primary_6", [(s, "a7_3") for s in a7_stems_3] + [(s, "xt2_3") for s in xt2_stems_3]),
                         ("expanded_9", [(s, "a7_6") for s in a7_stems_6] + [(s, "xt2_3") for s in xt2_stems_3])):
        rs = [l1_rows[s]["variants_c_roll_int"][t] for s, t in pairs]
        vdict = {}
        for v in ("B0", "B1", "L0C1_hard", "DQ(c_roll_int)", "DQ(c0)"):
            ok = [r[v] for r in rs if not r[v].get("undefined")]
            vdict[v] = dict(
                n=len(rs), undefined=len(rs) - len(ok),
                below_black_median=round(float(np.median([x["below_black_any"] for x in ok])), 5),
                below_black_max=round(float(np.max([x["below_black_any"] for x in ok])), 5),
                fixed_dE_median=round(float(np.median([x["fixed_dE_vs_BM"] for x in ok])), 3),
                fixed_ab_median=round(float(np.median([x["fixed_ab_vs_BM"] for x in ok])), 3),
                fixed_abs_dL_median=round(float(np.median([x["fixed_abs_dL_vs_BM"] for x in ok])), 3),
                matched_dE_median=round(float(np.median([x["matched_dE_vs_BM"] for x in ok])), 3),
                matched_ab_median=round(float(np.median([x["matched_ab_vs_BM"] for x in ok])), 3),
            )
        roll_scopes[pname] = dict(n=len(rs), frames=[s for s, _ in pairs], variants=vdict)
    del roll_scopes["pooled_primary_6"]
    return dict(scopes=scopes, roll_scopes=roll_scopes, frames=l1_rows)


def slim_eval(m, ab, L_med=None, target_L=None):
    d = {k: m[k] for k in ("de2000_median", "de2000_mean", "de2000_p90", "dL_median", "dL_abs_median",
                           "da_mean", "db_mean", "clipped_frac", "ref_clipped_frac",
                           "clipped_where_ref_not_frac", "below_zero_after_colour") if k in m}
    d["ab_median"] = float(ab)
    if L_med is not None:
        d["L_median"] = float(L_med)
        if target_L is not None:
            d["abs_dL_vs_B0"] = abs(float(L_med) - float(target_L))
    return d


def eval_l23_variant(T, R, b_eff, p999, target_L, d_lo_shared=None):
    P, d_hi, err = invert_safe(T, b_eff, p999)
    if err is not None:
        return dict(base=np.asarray(b_eff, float).tolist(), d_hi=d_hi.tolist(), undefined=True, reason=err)
    m_fix, lab_fix = evaluate(P, R, LOOK["params"])
    L_fix = float(np.median(lab_fix[:, 0]))
    ab_fix = float(np.median(np.hypot(*(lab_fix[:, 1:] - R[:, 1:]).T)))
    ev, clamped = match_ev(P, target_L)
    pm = LOOK["params"] | dict(ev=ev)
    m_mat, lab_mat = evaluate(P, R, pm)
    ab_mat = float(np.median(np.hypot(*(lab_mat[:, 1:] - R[:, 1:]).T)))
    out = dict(
        base=np.asarray(b_eff, float).tolist(), d_hi=d_hi.tolist(),
        below_black=[float(x) for x in (P < 0).mean(0)],
        below_black_any=float((P < 0).any(-1).mean()),
        fixed_look=slim_eval(m_fix, ab_fix, L_fix, target_L),
        matched=slim_eval(m_mat, ab_mat) | dict(ev=float(ev), ev_clamped=bool(clamped)),
    )
    if d_lo_shared is not None:
        out["d_lo_shared"] = float(d_lo_shared)
    return out


def l23_task_fast(args):
    p_str, out_dir_str = args
    p = Path(p_str)
    out_file = Path(out_dir_str) / p.name
    if out_file.exists():
        return json.loads(out_file.read_text())
    old = json.loads(p.read_text())
    stem, rk = p.stem.rsplit("_", 1)
    trec = json.loads((data_dir("s3_diag", "T8") / f"{stem}.json").read_text())
    check_name(trec["raw"])
    check_hashes([trec["raw_sha256"]])
    T8 = load_cache(data_dir("s3_diag", "T8") / f"{stem}_T8.npy")
    domain = DOM[old["id"].split(":")[0]]
    cdir = data_dir("s3_ml", "cache" if rk == "shop" else "cache_user", domain)
    R = load_cache(cdir / f"{stem}_R8.npy").astype(np.float64)
    y0, x0 = old["grid"]["start"]
    T = T8[y0 : y0 + R.shape[0], x0 : x0 + R.shape[1]].reshape(-1, 3).astype(np.float64)
    R = R.reshape(-1, 3)
    b0 = np.array(trec["base_cache"], np.float64)
    broll = np.array(old["variants"]["B1"]["base"], np.float64)
    (l0, c0), (l1, c1) = split(b0), split(broll)
    p01 = np.array(trec["L_percentiles"]["0.1"], np.float64)
    p999 = np.array(trec["L_percentiles"]["99.9"], np.float64)
    target_L = float(old["target_L"])

    variants = {}
    for v_old, v_new, beff in (("B0", "B0", b0), ("B1", "B1", broll), ("L0C1", "L0C1_hard", 10.0 ** (l0 + c1))):
        ov = old["variants"][v_old]
        if ov.get("undefined"):
            variants[v_new] = dict(ov)
            continue
        P, _, _ = invert_safe(T, beff, p999)
        L_fix = target_L if v_new == "B0" else median_L(P, LOOK["params"])
        variants[v_new] = dict(
            base=ov["base"], d_hi=ov["d_hi"], below_black=ov["below_black"], below_black_any=ov["below_black_any"],
            fixed_look=slim_eval(ov["fixed_look"], ov["fixed_look"]["ab_median"], L_fix, target_L),
            matched=slim_eval(ov["matched"], ov["matched"]["ab_median"])
            | dict(ev=ov["matched"]["ev"], ev_clamped=ov["matched"]["ev_clamped"]),
            p4=slim_eval(ov["p4"], ov["p4"]["ab_median"]) | dict(params=ov["p4"]["params"]),
        )

    beff_dq1, d_lo_1 = dq_base(c1, p01)
    beff_dq0, d_lo_0 = dq_base(c0, p01)
    variants["DQ(c_roll)"] = eval_l23_variant(T, R, beff_dq1, p999, target_L, d_lo_1)
    variants["DQ(c0)"] = eval_l23_variant(T, R, beff_dq0, p999, target_L, d_lo_0)

    rec = dict(
        id=old["id"], stem=stem, roll=old["roll"], ref_kind=rk, grid=old["grid"],
        own_minus_roll_density=old["own_minus_roll_density"],
        level_diff=old["level_diff"], chroma_diff=old["chroma_diff"],
        chroma_part=float(max(abs(x) for x in old["chroma_diff"])),
        level_part=abs(float(old["level_diff"])),
        target_L=target_L, variants=variants,
    )
    out_file.write_text(json.dumps(rec, indent=1))
    return rec


def p4_worker(p_str):
    t0 = time.time()
    p = Path(p_str)
    rec = json.loads(p.read_text())
    v = rec["variants"]["DQ(c_roll)"]
    if v.get("undefined") or "p4" in v:
        return rec["id"], rec["ref_kind"], "cached", 0.0
    stem, rk = rec["stem"], rec["ref_kind"]
    trec = json.loads((data_dir("s3_diag", "T8") / f"{stem}.json").read_text())
    check_name(trec["raw"])
    check_hashes([trec["raw_sha256"]])
    T8 = load_cache(data_dir("s3_diag", "T8") / f"{stem}_T8.npy")
    domain = DOM[rec["id"].split(":")[0]]
    cdir = data_dir("s3_ml", "cache" if rk == "shop" else "cache_user", domain)
    R = load_cache(cdir / f"{stem}_R8.npy").astype(np.float64)
    y0, x0 = rec["grid"]["start"]
    T = T8[y0 : y0 + R.shape[0], x0 : x0 + R.shape[1]].reshape(-1, 3).astype(np.float64)
    R = R.reshape(-1, 3)
    P = positive_T(T, v["base"], (0, 0, 0), v["d_hi"])
    p4 = fit_reference_s3(P, R, "p4", LOOK)
    m_p4, _ = evaluate(P, R, p4["params"])
    ab_p4 = ab_distance(P, R, p4["params"])
    v["p4"] = slim_eval(m_p4, ab_p4) | dict(params=p4["params"])
    p.write_text(json.dumps(rec, indent=1))
    dt = round(time.time() - t0, 1)
    return rec["id"], rk, f"dE={v['p4']['de2000_median']:.2f}", dt


def dataset_of_l2(r):
    if r["id"].startswith("blueneg:"):
        return "BlueNeg_107"
    if r["roll"] in ("own-e5207-20240401-dsc6233", "shop_r01"):
        return "v1_42"
    return "s3_diag_109"


def group_of_l3(r):
    if r["id"].startswith("blueneg:"):
        return "BlueNeg 冲印"
    if r["roll"] in ("own-e5207-20240401-dsc6233", "shop_r01"):
        return "v1 两卷 店家"
    return "S3 诊断三卷 " + ("店家" if r["ref_kind"] == "shop" else "用户")


def summarize_l23(records):
    rows = [r for r in records if r["id"] not in EXCLUDE_L23]
    v_names = ("B0", "B1", "L0C1_hard", "DQ(c_roll)", "DQ(c0)")

    # Layer 2: no-reference diagnostics on the 258 unique frames (shop records)
    shop_rows = [r for r in rows if r["ref_kind"] == "shop"]
    by_ds = defaultdict(list)
    for r in shop_rows:
        by_ds[dataset_of_l2(r)].append(r)
    by_ds["all_258"] = shop_rows
    layer2 = {}
    for ds_name, rs in sorted(by_ds.items()):
        vd = {}
        for v in v_names:
            ok = [r["variants"][v] for r in rs if not r["variants"][v].get("undefined")]
            bb = [x["below_black_any"] for x in ok]
            cl = [x["fixed_look"]["clipped_frac"] for x in ok]
            dl = [x["fixed_look"]["abs_dL_vs_B0"] for x in ok]
            vd[v] = dict(
                n=len(rs), undefined=len(rs) - len(ok),
                below_black_median=round(float(np.median(bb)), 5),
                below_black_p90=round(float(np.percentile(bb, 90)), 5),
                below_black_max=round(float(np.max(bb)), 5),
                clipped_frac_median=round(float(np.median(cl)), 5),
                abs_dL_vs_B0_median=round(float(np.median(dl)), 3),
                abs_dL_vs_B0_iqr=[round(float(x), 3) for x in np.percentile(dl, [25, 75])],
                abs_dL_vs_B0_max=round(float(np.max(dl)), 3),
            )
        layer2[ds_name] = dict(n=len(rs), variants=vd)

    # Layer 3: reference metrics by group and by chroma_part >= 0.1 vs < 0.1
    def l3_stats(rs):
        out = {}
        for v in v_names:
            ok = [r for r in rs if not r["variants"][v].get("undefined")]
            for rd in ("fixed_look", "matched", "p4"):
                have = [r for r in ok if rd in r["variants"][v]]
                if not have:
                    continue
                de_v = lambda r, vn: (r["variants"][vn][rd]["de2000_median"]
                                      if not r["variants"][vn].get("undefined") else float("inf"))
                d = dict(
                    n=len(rs), undefined=len(rs) - len(ok),
                    dE=round(float(np.median([r["variants"][v][rd]["de2000_median"] for r in have])), 3),
                    ab=round(float(np.median([r["variants"][v][rd]["ab_median"] for r in have])), 3),
                    dL=round(float(np.median([r["variants"][v][rd]["dL_median"] for r in have])), 3),
                    below_black=round(float(np.median([r["variants"][v]["below_black_any"] for r in have])), 5),
                )
                if v != "B0" and len(have) == len(ok):
                    diffs = [de_v(r, v) - de_v(r, "B0") for r in rs]
                    fin = [x for x in diffs if np.isfinite(x)]
                    d.update(
                        d_vs_B0=round(float(np.median(fin)), 3) if fin else None,
                        better=int(sum(x < 0 for x in diffs)),
                        worse2=int(sum(x > 2.0 for x in diffs)),
                        max_worse=round(float(max(fin)), 3) if fin else None,
                    )
                out[f"{v}/{rd}"] = d
        return out

    by_grp = defaultdict(list)
    for r in rows:
        by_grp[group_of_l3(r)].append(r)
    layer3 = {}
    for g, rs in sorted(by_grp.items()):
        subs = {
            "all": rs,
            "chroma_part>=0.1": [r for r in rs if r["chroma_part"] >= SPLIT],
            "chroma_part<0.1": [r for r in rs if r["chroma_part"] < SPLIT],
            "both<0.1": [r for r in rs if r["chroma_part"] < SPLIT and r["level_part"] < SPLIT],
        }
        layer3[g] = {k: dict(n=len(v), stats=l3_stats(v) if v else None) for k, v in subs.items()}

    # Per-roll normal-frame non-degradation table (DESIGN.md section 4 item 5)
    roll_c5 = {}
    by_roll = defaultdict(list)
    for r in rows:
        if r["id"].startswith("blueneg:") or r["roll"] in ("own-e5207-20240401-dsc6233", "shop_r01"):
            by_roll[(r["roll"], r["ref_kind"], "all_frames")].append(r)
        elif r["chroma_part"] < SPLIT:
            by_roll[(r["roll"], r["ref_kind"], "chroma_part<0.1")].append(r)
    for (roll, rk, scope), rs in sorted(by_roll.items()):
        st = {}
        for v in ("B1", "L0C1_hard", "DQ(c_roll)", "DQ(c0)"):
            for rd in ("fixed_look", "matched"):
                diffs = [
                    (float("inf") if r["variants"][v].get("undefined") else r["variants"][v][rd]["de2000_median"])
                    - r["variants"]["B0"][rd]["de2000_median"]
                    for r in rs
                ]
                fin = [x for x in diffs if np.isfinite(x)]
                st[f"{v}/{rd}"] = dict(
                    d_vs_B0=round(float(np.median(fin)), 3) if fin else None,
                    better=int(sum(x < 0 for x in diffs)),
                    worse2=int(sum(x > 2.0 for x in diffs)),
                    max_worse=round(float(max(fin)), 3) if fin else None,
                )
        roll_c5[f"{roll}|{rk}|{scope}"] = dict(n=len(rs), stats=st)

    # Also evaluate the 4 R1-triggered BlueNeg frames specifically
    r1_frames = {}
    for r in shop_rows:
        if any(tag in r["id"] for tag in ("199807B-09", "19941220D-18", "19941222E-01", "19941223E-17")):
            r1_frames[r["id"]] = {
                v: dict(
                    fixed_dE=round(r["variants"][v]["fixed_look"]["de2000_median"], 2),
                    matched_dE=round(r["variants"][v]["matched"]["de2000_median"], 2),
                    below_black=round(r["variants"][v]["below_black_any"], 4),
                    abs_dL_vs_B0=round(r["variants"][v]["fixed_look"]["abs_dL_vs_B0"], 2),
                )
                for v in v_names if not r["variants"][v].get("undefined")
            }

    return dict(layer2=layer2, layer3=layer3, criterion5_rolls=roll_c5, blueneg_r1_frames=r1_frames)


def evaluate_pass_criteria(l1, l23):
    # Criterion 1: truncation on Layer 1 (primary_25 + expanded_28) and Layer 2 (all 258 frames)
    l1_p25_dq = l1["scopes"]["primary_25"]["variants"]["DQ(c_meas)"]
    l1_e28_dq = l1["scopes"]["expanded_28"]["variants"]["DQ(c_meas)"]
    l2_all_dq = l23["layer2"]["all_258"]["variants"]["DQ(c_roll)"]
    c1_pass = (
        l1_p25_dq["below_black_median"] <= 0.005
        and l1_p25_dq["below_black_max"] <= 0.02
        and l2_all_dq["below_black_median"] <= 0.005
        and l2_all_dq["below_black_max"] <= 0.02
    )

    # Criterion 2: Layer 1 c* = c_meas on primary_25 (and reported alongside expanded_28, clean_26)
    c2_a_val = l1_p25_dq["matched_dE_median"]
    c2_b = l1_p25_dq.get("on_B0_dE_ge_2", {})
    c2_a_pass = c2_a_val <= 2.0
    c2_b_pass = (c2_b.get("n", 0) >= 5 and c2_b.get("improved_share", 0) >= 2 / 3
                 and (c2_b.get("median_drop") or 0) >= 1.0)

    # Criterion 3: Layer 1 c* = c_roll_int on roll groups (primary_6 and expanded_9, plus per-group)
    p6 = l1["roll_scopes"]["primary_6"]["variants"]
    e9 = l1["roll_scopes"]["expanded_9"]["variants"]
    a7_3 = l1["roll_scopes"]["a7r5_primary_3"]["variants"]
    a7_6 = l1["roll_scopes"]["a7r5_expanded_6"]["variants"]
    xt2_3 = l1["roll_scopes"]["xt2_20180613_3"]["variants"]
    c3_primary6_pass = (
        p6["DQ(c_roll_int)"]["matched_dE_median"] < p6["B0"]["matched_dE_median"]
        and p6["DQ(c_roll_int)"]["fixed_dE_median"] < p6["B0"]["fixed_dE_median"]
    )
    c3_expanded9_pass = (
        e9["DQ(c_roll_int)"]["matched_dE_median"] < e9["B0"]["matched_dE_median"]
        and e9["DQ(c_roll_int)"]["fixed_dE_median"] < e9["B0"]["fixed_dE_median"]
    )
    c3_per_group_pass = (
        a7_3["DQ(c_roll_int)"]["matched_dE_median"] < a7_3["B0"]["matched_dE_median"]
        and a7_3["DQ(c_roll_int)"]["fixed_dE_median"] < a7_3["B0"]["fixed_dE_median"]
        and a7_6["DQ(c_roll_int)"]["matched_dE_median"] < a7_6["B0"]["matched_dE_median"]
        and a7_6["DQ(c_roll_int)"]["fixed_dE_median"] < a7_6["B0"]["fixed_dE_median"]
        and xt2_3["DQ(c_roll_int)"]["matched_dE_median"] < xt2_3["B0"]["matched_dE_median"]
        and xt2_3["DQ(c_roll_int)"]["fixed_dE_median"] < xt2_3["B0"]["fixed_dE_median"]
    )

    # Criterion 4: Layer 2 brightness stability per dataset (median |dL*| <= 2.0 and < B1)
    c4_datasets = {}
    c4_pass = True
    for ds in ("s3_diag_109", "v1_42", "BlueNeg_107"):
        v_dq = l23["layer2"][ds]["variants"]["DQ(c_roll)"]["abs_dL_vs_B0_median"]
        v_b1 = l23["layer2"][ds]["variants"]["B1"]["abs_dL_vs_B0_median"]
        ok = v_dq <= 2.0 and v_dq < v_b1
        c4_pass = c4_pass and ok
        c4_datasets[ds] = dict(dq_abs_dL_median=v_dq, b1_abs_dL_median=v_b1, pass_=ok)

    # Criterion 5: Layer 3 normal frames non-degradation (worse2 <= 1 per roll under fixed_look)
    c5_rolls = {}
    c5_pass = True
    for k, v in sorted(l23["criterion5_rolls"].items()):
        w2_dq = v["stats"]["DQ(c_roll)/fixed_look"]["worse2"]
        w2_b1 = v["stats"]["B1/fixed_look"]["worse2"]
        ok = w2_dq <= 1
        c5_pass = c5_pass and ok
        c5_rolls[k] = dict(n=v["n"], dq_fixed_worse2=w2_dq, dq_fixed_d_vs_B0=v["stats"]["DQ(c_roll)/fixed_look"]["d_vs_B0"],
                           dq_fixed_max_worse=v["stats"]["DQ(c_roll)/fixed_look"]["max_worse"],
                           b1_fixed_worse2=w2_b1, pass_=ok)

    return dict(
        criterion1_truncation=dict(
            pass_=c1_pass,
            l1_primary25_median=l1_p25_dq["below_black_median"],
            l1_primary25_max=l1_p25_dq["below_black_max"],
            l1_expanded28_median=l1_e28_dq["below_black_median"],
            l1_expanded28_max=l1_e28_dq["below_black_max"],
            l2_all258_median=l2_all_dq["below_black_median"],
            l2_all258_max=l2_all_dq["below_black_max"],
        ),
        criterion2_l1_c_meas=dict(
            pass_=bool(c2_a_pass and c2_b_pass),
            part_a_pass=bool(c2_a_pass),
            part_a_matched_dE_median_primary25=c2_a_val,
            part_a_matched_dE_median_expanded28=l1_e28_dq["matched_dE_median"],
            part_a_matched_dE_median_clean26=l1["scopes"]["clean_26"]["variants"]["DQ(c_meas)"]["matched_dE_median"],
            part_b_pass=bool(c2_b_pass),
            part_b_details_primary25=c2_b,
            part_b_details_expanded28=l1_e28_dq.get("on_B0_dE_ge_2"),
            part_b_details_clean26=l1["scopes"]["clean_26"]["variants"]["DQ(c_meas)"].get("on_B0_dE_ge_2"),
        ),
        criterion3_l1_c_roll_int=dict(
            pass_pooled_primary6=bool(c3_primary6_pass),
            pass_pooled_expanded9=bool(c3_expanded9_pass),
            pass_every_group_individually=bool(c3_per_group_pass),
        ),
        criterion4_l2_brightness_stability=dict(pass_=bool(c4_pass), datasets=c4_datasets),
        criterion5_l3_normal_non_degradation=dict(pass_=bool(c5_pass), rolls=c5_rolls),
        all_pass=bool(c1_pass and c2_a_pass and c2_b_pass and c3_primary6_pass and c3_expanded9_pass
                      and c4_pass and c5_pass),
    )


def render_tile(T_img, b_eff, p999, params, label, tile_w=420):
    P, _, err = invert_safe(T_img, b_eff, p999)
    h, w = T_img.shape[:2]
    th = int(h * tile_w / w)
    if err is not None:
        img = Image.new("RGB", (tile_w, th), (40, 0, 40))
    else:
        disp, _ = render_image(P, RenderParams(**params))
        u8 = (np.clip(disp, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(u8).resize((tile_w, th), Image.BILINEAR)
    out = Image.new("RGB", (tile_w, th + 22), (20, 20, 20))
    out.paste(img, (0, 22))
    ImageDraw.Draw(out).text((6, 4), label, fill=(240, 240, 240))
    return out


def make_preview_sheets(l1, out_dir):
    pdir = Path(out_dir) / "previews" / "t3b"
    pdir.mkdir(parents=True, exist_ok=True)
    # 1. Layer 1 representative frames: A1010, A1020, A263, A7R5_RAW_DSC01683
    cache_l1 = data_dir("t3_inversion", "T8")
    rv = json.loads(Path("experiments/t3_inversion/rebate_verified.json").read_text())["frames"]
    l1_tiles = []
    for stem in ("A1010", "A1020", "A263", "A7R5_RAW_DSC01683"):
        r = rv[stem]
        fr = l1["frames"][stem]
        T8 = load_cache(cache_l1 / f"{stem}_T8.npy").astype(np.float64)
        T_img = rect(T8, r["image_box"])
        p999 = np.array(r["L_percentiles_box"]["99.9"], np.float64)
        row_imgs = []
        for v in ("BM", "B0", "L0C1_hard", "DQ(c_meas)", "DQ(c0)"):
            vd = fr["variants_c_meas"][v]
            lbl = f"{stem} {v}  fix dE={vd.get('fixed_dE_vs_BM', -1):.2f} mat dE={vd.get('matched_dE_vs_BM', -1):.2f} bb={vd.get('below_black_any', -1):.1%}"
            row_imgs.append(render_tile(T_img, vd["base"], p999, LOOK["params"], lbl))
        w_tot = sum(im.width for im in row_imgs)
        h_max = max(im.height for im in row_imgs)
        strip = Image.new("RGB", (w_tot, h_max), (20, 20, 20))
        x = 0
        for im in row_imgs:
            strip.paste(im, (x, 0))
            x += im.width
        strip.save(pdir / f"L1_{stem}.jpg", quality=88)
        l1_tiles.append(strip)

    # 2. Layer 2/3 representative frames: A008, A140, DSC_3827, 19941222E-01-nara
    for stem in ("A008", "A140", "DSC_3827", "19941222E-01-nara"):
        rec = json.loads((Path(out_dir) / "t3b" / f"{stem}_shop.json").read_text())
        trec = json.loads((data_dir("s3_diag", "T8") / f"{stem}.json").read_text())
        T8 = load_cache(data_dir("s3_diag", "T8") / f"{stem}_T8.npy")
        domain = DOM[rec["id"].split(":")[0]]
        R = load_cache(data_dir("s3_ml", "cache", domain) / f"{stem}_R8.npy")
        y0, x0 = rec["grid"]["start"]
        T_img = T8[y0 : y0 + R.shape[0], x0 : x0 + R.shape[1]].astype(np.float64)
        p999 = np.array(trec["L_percentiles"]["99.9"], np.float64)
        row_imgs = []
        for v in ("B0", "B1", "L0C1_hard", "DQ(c_roll)", "DQ(c0)"):
            vd = rec["variants"][v]
            if vd.get("undefined"):
                lbl = f"{stem} {v}  UNDEFINED"
            else:
                lbl = f"{stem} {v}  fix dE={vd['fixed_look']['de2000_median']:.2f} mat dE={vd['matched']['de2000_median']:.2f} bb={vd['below_black_any']:.1%}"
            row_imgs.append(render_tile(T_img, vd["base"], p999, LOOK["params"], lbl))
        w_tot = sum(im.width for im in row_imgs)
        h_max = max(im.height for im in row_imgs)
        strip = Image.new("RGB", (w_tot, h_max), (20, 20, 20))
        x = 0
        for im in row_imgs:
            strip.paste(im, (x, 0))
            x += im.width
        strip.save(pdir / f"L23_{stem}.jpg", quality=88)


def main(argv):
    out_dir = Path(argv[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    t3b_dir = out_dir / "t3b"
    t3b_dir.mkdir(parents=True, exist_ok=True)
    workers = int(argv[argv.index("--p4") + 1]) if "--p4" in argv else 0

    print("1. Checking 3-frame bit-identity against s3_separate/check/ ...", flush=True)
    checked = check_bit_identity()
    print("   bit-identical on:", checked, flush=True)

    print("2. Running Layer 1 (physical measured-base reference) ...", flush=True)
    l1_path = out_dir / "layer1.json"
    if l1_path.exists():
        l1 = json.loads(l1_path.read_text())
    else:
        l1 = run_layer1("experiments/t3_inversion/rebate_verified.json")
        l1_path.write_text(json.dumps(l1, ensure_ascii=False, indent=1) + "\n")

    print("3. Running Layer 2 & Layer 3 (fixed_look & matched across 315 tasks) ...", flush=True)
    check_files = sorted(Path("experiments/s3_separate/check").glob("*.json"))
    with ProcessPoolExecutor(8) as ex:
        records = list(ex.map(l23_task_fast, [(str(p), str(t3b_dir)) for p in check_files]))
    print(f"   done {len(records)} tasks.", flush=True)

    if workers > 0:
        todo = [str(t3b_dir / p.name) for p in check_files
                if "p4" not in json.loads((t3b_dir / p.name).read_text())["variants"]["DQ(c_roll)"]]
        print(f"   Fitting P4 for DQ(c_roll) on {len(todo)} tasks with {workers} workers ...", flush=True)
        if todo:
            with ProcessPoolExecutor(workers) as ex:
                for i, r in enumerate(ex.map(p4_worker, todo), 1):
                    if i % 25 == 0 or i == len(todo):
                        print(f"   [{i}/{len(todo)}]", *r, flush=True)
            records = [json.loads((t3b_dir / p.name).read_text()) for p in check_files]

    print("4. Summarising Layer 2/3 & evaluating pass criteria ...", flush=True)
    l23 = summarize_l23(records)
    criteria = evaluate_pass_criteria(l1, l23)
    make_preview_sheets(l1, out_dir)

    summary = dict(
        created="2026-10-23",
        design="experiments/t3_inversion/DESIGN.md",
        bit_identity_checked=checked,
        pass_criteria=criteria,
        layer1_scopes=l1["scopes"],
        layer1_roll_scopes=l1["roll_scopes"],
        layer2=l23["layer2"],
        layer3=l23["layer3"],
        criterion5_rolls=l23["criterion5_rolls"],
        blueneg_r1_frames=l23["blueneg_r1_frames"],
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(criteria, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])

