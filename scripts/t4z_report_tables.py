"""Rebuild T4z acceptance from measured artifacts; no original images are decoded.
Usage: python -I scripts/t4z_report_tables.py
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4z"
sys.path.insert(0, str(ROOT / "scripts"))
from t4s_report_tables import valid
sys.path.insert(0, str(ROOT / "src"))
from filmcolor.app import THUMB_REV
from filmcolor.noref import NOREF_V3, STYLE_PRESETS, preset_config

PRESETS = tuple(STYLE_PRESETS)
EXPECTED_ERRORS = {f"ext:{n}" for n in ("A016", "A018", "A025", "A026", "A027", "A028")}
CHECK_EXCLUSION = "blueneg:199807A-21-orlando-disney"
TARGETS = ("A305.tif", "A1040.TIF", "A1041.TIF", "A232.NEF", "A1069.NEF", "A239.ORF", "A281.RAF", "A1006.CR2")
NEW_TARGETS = ("A1004.DNG", "A1005.CR2", "A1007.CR2", "A1033.CR2", "A262.arw", "A263.arw", "A1061.CR2", "A237.cr2")
TIFFS = {"A1310.TIF", "A304.tif", "A1301.TIF", "A255.tif", "A1027.TIF", "A1028.TIF", "B032.tif", "B034.tiff"}
NEW_TIFFS = {"A1228.tif", "A229.TIF", "A655.tif", "A849.tif", "A850.tif", "A1093.tif", "B036.tif"}
EIGHT_BIT_TIFFS = {"A229.TIF", "A655.tif", "A849.tif", "A850.tif", "A1093.tif", "B036.tif"}
MATRIX_TARGETS = ("A1004.DNG", "A1005.CR2", "A1006.CR2", "A1007.CR2", "A1033.CR2", "A1296.CR2", "A1061.CR2", "A237.cr2")
BENCHMARK_VISUAL_KEYS = (
    "ext:A105",
    "ext:A105_print_classic",
    "blueneg:199910-01-korea",
    "ext:A198",
    "blueneg:19970619A-15-white-chapel",
)

# 改动 C 硬守卫帧与要求参数/反相/白点报告逐位一致帧
HARD_GUARD_PARAMS = ("A1004.DNG", "A1033.CR2")
GUARD_C_DEVICES = (
    "A232.NEF", "A1069.NEF",
    "A1004.DNG", "A1005.CR2", "A1006.CR2", "A1007.CR2",
    "A1033.CR2", "A1061.CR2", "A237.cr2", "A281.RAF",
    "A655.tif", "A1027.TIF",
    "B027.NEF", "B028.NEF", "B029.NEF",
)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_sha():
    h = hashlib.sha256()
    for p in sorted(p for p in (ROOT / "src/filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(ROOT / "src")).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def committed_json(version, part):
    commit = {
        "t4y": "4bd014e",
        "t4x": "f6340aa",
        "t4w": "b3a7510",
        "t4v": "0d1d730",
        "t4u": "98ded3c",
        "t4t": "567aaf9",
        "t4s": "7c5107c",
        "t4r": "fbd48a0",
        "t4o": "4cd76b7",
    }[version]
    rel = f"experiments/t4_auto_ui/{version}/candidate_{part}.json"
    data = subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=ROOT)
    assert data == (ROOT / rel).read_bytes(), f"historical artifact modified: {rel}"
    return json.loads(data)


def compare_frames_identity_vs_prior(label, ids, old_rows, cand_rows, presets):
    """Compare per-frame params and metrics, tracking differences and max abs diffs."""
    max_abs_diff_params = 0.0
    max_abs_diff_metrics = 0.0
    param_diffs_by_key = {}
    metric_diffs_by_key = {}
    diff_count = 0
    diff_summary = []

    for k in ids:
        cand_vars = cand_rows[k]["variants"]
        old_vars = old_rows[k]["variants"]
        for p in presets:
            if p not in cand_vars or p not in old_vars:
                diff_count += 1
                if len(diff_summary) < 20:
                    diff_summary.append(dict(frame=k, preset=p, error="variant_missing"))
                continue
            cp = cand_vars[p].get("params") or {}
            rp = old_vars[p].get("params") or {}
            for pk in set(cp) | set(rp):
                cv, rv = cp.get(pk), rp.get(pk)
                if cv != rv:
                    diff_count += 1
                    if isinstance(cv, (int, float)) and isinstance(rv, (int, float)):
                        d = abs(float(cv) - float(rv))
                        if d > max_abs_diff_params:
                            max_abs_diff_params = d
                        param_diffs_by_key[pk] = max(param_diffs_by_key.get(pk, 0.0), d)
                        if len(diff_summary) < 20:
                            diff_summary.append(dict(frame=k, preset=p, type="param_numeric", key=pk, cand=cv, prior=rv, diff=d))
                    else:
                        if len(diff_summary) < 20:
                            diff_summary.append(dict(frame=k, preset=p, type="param_discrete", key=pk, cand=cv, prior=rv))

            cm = cand_vars[p].get("metrics") or {}
            rm = old_vars[p].get("metrics") or {}
            for mk in set(cm) | set(rm):
                cv, rv = cm.get(mk), rm.get(mk)
                if cv != rv:
                    diff_count += 1
                    if isinstance(cv, (int, float)) and isinstance(rv, (int, float)):
                        d = abs(float(cv) - float(rv))
                        if d > max_abs_diff_metrics:
                            max_abs_diff_metrics = d
                        metric_diffs_by_key[mk] = max(metric_diffs_by_key.get(mk, 0.0), d)
                        if len(diff_summary) < 20:
                            diff_summary.append(dict(frame=k, preset=p, type="metric_numeric", key=mk, cand=cv, prior=rv, diff=d))
                    else:
                        if len(diff_summary) < 20:
                            diff_summary.append(dict(frame=k, preset=p, type="metric_discrete", key=mk, cand=cv, prior=rv))

    is_identical = (diff_count == 0 and max_abs_diff_params == 0.0 and max_abs_diff_metrics == 0.0)

    return dict(
        identically_equal=is_identical,
        max_abs_diff_params=max_abs_diff_params,
        max_abs_diff_metrics=max_abs_diff_metrics,
        param_diffs_by_key=param_diffs_by_key,
        metric_diffs_by_key=metric_diffs_by_key,
        diff_count=diff_count,
        diff_summary=diff_summary,
    )


def summarize_preset_comparison(ids, base_rows, cand_rows, preset):
    """Summarize deltaE 2000 metrics comparing candidate vs baseline for a given preset."""
    u = np.array([float(base_rows[k]["variants"][preset]["metrics"]["de2000_median"]) for k in ids], dtype=float)
    v = np.array([float(cand_rows[k]["variants"][preset]["metrics"]["de2000_median"]) for k in ids], dtype=float)
    diff = v - u
    better_025 = int((diff < -0.25).sum())
    worse_025 = int((diff > 0.25).sum())
    better_005 = int((diff < -0.05).sum())
    worse_005 = int((diff > 0.05).sum())
    max_reg = float(diff.max()) if len(diff) else 0.0
    regressions_005 = {
        k: float(diff[i]) for i, k in enumerate(ids) if diff[i] > 0.05
    }
    improvements_005 = {
        k: float(diff[i]) for i, k in enumerate(ids) if diff[i] < -0.05
    }
    return dict(
        better_025=better_025, worse_025=worse_025,
        better_005=better_005, worse_005=worse_005,
        max_regression=max_reg,
        regressions_over_005=regressions_005,
        improvements_over_005=improvements_005,
        stats_before=" / ".join(f"{val:.2f}" for val in (np.median(u), np.mean(u), np.percentile(u, 90))),
        stats_after=" / ".join(f"{val:.2f}" for val in (np.median(v), np.mean(v), np.percentile(v, 90))),
        metrics_before=dict(median=float(np.median(u)), mean=float(np.mean(u)), p90=float(np.percentile(u, 90))),
        metrics_after=dict(median=float(np.median(v)), mean=float(np.mean(v)), p90=float(np.percentile(v, 90)))
    )


def find_item_in_candidate(candidate, name):
    for group in ("targets", "new_targets", "tiff", "new_tiff", "special", "acros"):
        if name in candidate.get(group, {}):
            return candidate[group][name]
    if name == "A1296.CR2":
        return candidate.get("a1296")
    return None


def device_checks(baseline, candidate, t4y_candidate):
    checks = {}
    param_changes = []

    def gate(name, passed):
        checks[name] = bool(passed)

    def common(name, before, after):
        gate(name + ":input_sha", before.get("raw_sha256") == after.get("raw_sha256") and bool(after.get("raw_sha256")))
        gate(name + ":ok_finite_preserved", after.get("status") == "ok" and after.get("full_positive_finite")
             and after.get("render_preserves_positive"))

    def check_params_change(name, before, after):
        bp = before.get("params") or {}
        ap = after.get("params") or {}
        if bp == ap:
            return
        diag = after.get("diagnostics", {})
        ol = diag.get("overlift_damp", {})
        reason = "unexplained"
        if ol.get("applied") is True:
            reason = ol.get("mode", "overlift_damp")
        elif bp.get("matrix") != ap.get("matrix"):
            reason = "camera_matrix_unblock"

        param_changes.append(dict(
            frame=name,
            params_before=bp,
            params_after=ap,
            ev_before=bp.get("ev"),
            ev_after=ap.get("ev"),
            blacks_before=bp.get("blacks"),
            blacks_after=ap.get("blacks"),
            contrast_before=bp.get("contrast"),
            contrast_after=ap.get("contrast"),
            matrix_before=bp.get("matrix"),
            matrix_after=ap.get("matrix"),
            reason=reason,
        ))

    a = candidate["a1296"]
    b_a = baseline["a1296"]
    common("A1296", b_a, a)
    gate("A1296:1600px", candidate["max_side"] == 1600)
    gate("A1296:center_p99_lt_1_35", max(a["center_p99"]) < 1.35)
    gate("A1296:band_blue_p99_le_1_15", a["upper_band_p99"][2] <= 1.15)
    gate("A1296:band_blue_max_lt_2", a["upper_band_max"][2] < 2.0)
    gate("A1296:full_band_any_gt_10_zero", a["full_upper_band_share_any_channel_gt_10"] == 0.0)
    gate("A1296:status_exact_vs_t4y", a.get("status") == b_a.get("status"))
    gate("A1296:inversion_exact_vs_t4y", a.get("inversion") == b_a.get("inversion"))
    check_params_change("A1296.CR2", b_a, a)

    base_info = a.get("base") or {}
    report_base = a.get("report", {}).get("base") or {}
    diag_base = a.get("diagnostics", {}).get("base") or {}
    has_bare_backlight = (
        base_info.get("bare_backlight_excluded") is True
        or report_base.get("bare_backlight_excluded") is True
        or diag_base.get("bare_backlight_excluded") is True
        or (a.get("base_diagnostics") or {}).get("bare_backlight_excluded") is True
        or any("bare backlight" in w for w in base_info.get("warnings", []))
    )
    gate("A1296:bare_backlight_excluded", has_bare_backlight)

    gate("targets:membership", set(candidate["targets"]) == set(TARGETS))
    for name in TARGETS:
        r, b = candidate["targets"][name], baseline["targets"][name]
        common(name, b, r)
        check_params_change(name, b, r)
        if name == "A305.tif":
            gate(name + ":no_white_point_invalid", not any("white_point_invalid" in f for f in r["flags"]))
        elif name == "A1040.TIF":
            sc = r.get("scanner", {})
            gate(name + ":reconstruction_present", bool(sc.get("prestretched_reconstruction")))
            gate(name + ":no_white_point_unrep", "white_point_unrepresentative" not in r.get("flags", []))
            gate(name + ":srgb_q50_R_gt_0_10", r.get("srgb_q50", [0, 0, 0])[0] > 0.10)
            gate(name + ":srgb_q50_G_gt_0_10", r.get("srgb_q50", [0, 0, 0])[1] > 0.10)
            gate(name + ":srgb_q50_B_minus_G_lt_0_08", (r.get("srgb_q50", [0, 0, 0])[2] - r.get("srgb_q50", [0, 0, 0])[1]) < 0.08)
        elif name == "A1041.TIF":
            sc = r.get("scanner", {})
            gate(name + ":reconstruction_present", bool(sc.get("prestretched_reconstruction")))
            gate(name + ":no_white_point_unrep", "white_point_unrepresentative" not in r.get("flags", []))
            gate(name + ":srgb_q50_R_gt_0_15", r.get("srgb_q50", [0, 0, 0])[0] > 0.15)
            gate(name + ":srgb_q50_G_gt_0_15", r.get("srgb_q50", [0, 0, 0])[1] > 0.15)
            base_diag = r.get("base_diagnostics", {})
            gate(name + ":no_near_sat_excluded", "near_saturated_blocks_excluded" not in base_diag)
        elif name in ("A232.NEF", "A1069.NEF"):
            gate(name + ":holder_excluded", r["report"]["white_point"].get("holder_blocks_excluded", 0) > 0)
            gate(name + ":white_points_lt_1_15", max(r["inversion"]["d_hi"]) < 1.15)
            gate(name + ":holder_share_gt_0_50", r["holder_share"] > .50)
            gate(name + ":ev_lt_0_50", r["params"]["ev"] < .50)
        wb = r["diagnostics"].get("wb", {}).get("diagnostics", {})
        if name == "A239.ORF":
            gate(name + ":no_cyan_conf_boost", not wb.get("cyan_conf_boost", False))
            gate(name + ":no_narrow_tint_capped", wb.get("no_narrow_tint_capped", False) is True)
            gate(name + ":tint_lt_0_22", r["params"]["tint"] < .22)
        elif name == "A281.RAF":
            gate(name + ":tint_conflict_damped", wb.get("tint_conflict_damped", False))
            gate(name + ":tint_lt_0_18", r["params"]["tint"] < .18)

    gate("acros:membership", set(candidate["acros"]) == {f"B{i:03d}.NEF" for i in range(2, 30)})
    gate("tiff:membership", set(candidate["tiff"]) == TIFFS)
    for group in ("acros", "tiff"):
        for name, r in candidate[group].items():
            b = baseline[group][name]
            common(name, b, r)
            check_params_change(name, b, r)
            if name == "A1027.TIF":
                base_diag = r.get("base_diagnostics", {})
                gate(name + ":no_near_sat_excluded", "near_saturated_blocks_excluded" not in base_diag)

    gate("new_tiff:membership", set(candidate.get("new_tiff", {})) == NEW_TIFFS)
    new_cand = candidate.get("new_tiff", {})
    new_base = baseline.get("new_tiff", {})
    for name in NEW_TIFFS:
        r = new_cand.get(name, {})
        b = new_base.get(name, {})
        gate(f"{name}:input_sha", bool(r.get("raw_sha256")) and r.get("raw_sha256") == b.get("raw_sha256"))
        gate(f"{name}:candidate_ok", r.get("status") == "ok")
        gate(f"{name}:full_positive_finite", bool(r.get("full_positive_finite")))
        gate(f"{name}:render_preserves_positive", bool(r.get("render_preserves_positive")))
        check_params_change(name, b, r)
        if name == "A1228.tif":
            gate(f"{name}:flags_empty", r.get("flags") == [])
            d_hi = r.get("inversion", {}).get("d_hi", [0, 0, 0])
            gate(f"{name}:green_white_point_gt_0_30", len(d_hi) >= 2 and d_hi[1] > 0.30)
        elif name in EIGHT_BIT_TIFFS:
            sc = r.get("scanner", {})
            gate(f"{name}:bits8", sc.get("bits") == 8)

    gate("new_targets:membership", set(candidate.get("new_targets", {})) == set(NEW_TARGETS))
    new_target_cand = candidate.get("new_targets", {})
    new_target_base = baseline.get("new_targets", {})
    for name in NEW_TARGETS:
        r = new_target_cand.get(name, {})
        b = new_target_base.get(name, {})
        gate(f"{name}:input_sha", bool(r.get("raw_sha256")) and r.get("raw_sha256") == b.get("raw_sha256"))
        gate(f"{name}:candidate_ok", r.get("status") == "ok")
        gate(f"{name}:full_positive_finite", bool(r.get("full_positive_finite")))
        gate(f"{name}:render_preserves_positive", bool(r.get("render_preserves_positive")))
        check_params_change(name, b, r)
        if name in ("A262.arw", "A263.arw"):
            base_rgb = r.get("inversion", {}).get("base") or r.get("base", {}).get("base") or []
            gate(f"{name}:max_inversion_base_lt_0_95", max(base_rgb) < 0.95 if base_rgb else False)
            base_diag = r.get("base_diagnostics", {})
            diag_base = r.get("diagnostics", {}).get("base", {})
            near_sat_cand = base_diag.get("near_saturated_blocks_excluded") or diag_base.get("near_saturated_blocks_excluded") or 0
            gate(f"{name}:near_sat_blocks_excluded_cand_gt_1000", near_sat_cand > 1000)

    # Matrix unblock verification for the 8 specified targets
    for name in MATRIX_TARGETS:
        rec = find_item_in_candidate(candidate, name)
        gate(f"{name}:candidate_present", rec is not None)
        if rec is not None:
            gate(f"{name}:candidate_status_ok", rec.get("status") == "ok")
            mat = (rec.get("params") or {}).get("matrix")
            gate(f"{name}:matrix_present", bool(mat) and len(mat) == 3 and len(mat[0]) == 3)
            if mat:
                gate(f"{name}:matrix_00_gt_1_25", mat[0][0] > 1.25)
                gate(f"{name}:matrix_not_identity", mat != [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # scanner metadata check for all probe frames in candidate
    for grp in ("targets", "tiff", "new_tiff"):
        for name, r in candidate.get(grp, {}).items():
            if r.get("status") == "ok" and Path(name).suffix.lower() in (".tif", ".tiff"):
                sc = r.get("scanner", {})
                gate(f"{name}:scanner_meta_present", ("bits" in sc and "tiff_eotf" in sc
                     and "prestretched_reconstruction" in sc and "single_channel_prestretched" in sc))

    # T4z special target B096.tif checks
    gate("special:membership", set(candidate.get("special", {})) == {"B096.tif"})
    b096_cand = candidate.get("special", {}).get("B096.tif", {})
    b096_base = baseline.get("special", {}).get("B096.tif", {})
    gate("B096.tif:input_sha", bool(b096_cand.get("raw_sha256")) and b096_cand.get("raw_sha256") == b096_base.get("raw_sha256"))
    # B096 baseline must be manual_review_required with white_point_invalid and d_hi > 4.0
    gate("B096.tif:baseline_status_manual_review", b096_base.get("status") == "manual_review_required")
    gate("B096.tif:baseline_white_point_invalid", any("white_point_invalid" in f for f in b096_base.get("flags", [])))
    d_hi_b096_base = b096_base.get("inversion", {}).get("d_hi", [0, 0, 0])
    gate("B096.tif:baseline_d_hi_gt_4", len(d_hi_b096_base) >= 2 and d_hi_b096_base[1] > 4.0)

    # B096 candidate must be ok, finite, and d_hi in [0.9, 1.2]
    gate("B096.tif:candidate_status_ok", b096_cand.get("status") == "ok")
    gate("B096.tif:full_positive_finite", bool(b096_cand.get("full_positive_finite")))
    gate("B096.tif:render_preserves_positive", bool(b096_cand.get("render_preserves_positive")))
    d_hi_b096 = b096_cand.get("inversion", {}).get("d_hi", [0, 0, 0])
    gate("B096.tif:d_hi_range", len(d_hi_b096) >= 2 and 0.9 <= d_hi_b096[1] <= 1.2)

    # 改动 C 门禁：旧 devices 状态通过与指定 C 设备 params/inversion/report.white_point exact
    # （A232/A1069/A1004/5/6/7/A1033/A1061/A237/A281/A655/A1027/B027-29）
    # A1004/A1033 params exact 硬守卫
    for name in GUARD_C_DEVICES:
        r_c = find_item_in_candidate(candidate, name)
        r_y = find_item_in_candidate(t4y_candidate, name)
        gate(f"{name}:guard_c_present", r_c is not None and r_y is not None)
        if r_c is not None and r_y is not None:
            gate(f"{name}:status_exact_vs_t4y", r_c.get("status") == r_y.get("status") == "ok")
            gate(f"{name}:inversion_exact_vs_t4y", r_c.get("inversion") == r_y.get("inversion"))
            gate(f"{name}:params_exact_vs_t4y", r_c.get("params") == r_y.get("params"))
            gate(f"{name}:report_white_point_exact_vs_t4y", r_c.get("report", {}).get("white_point") == r_y.get("report", {}).get("white_point"))
            gate(f"{name}:spike_clamp_exact_vs_t4y", r_c.get("inversion", {}).get("spike_clamp") == r_y.get("inversion", {}).get("spike_clamp"))
            if name in HARD_GUARD_PARAMS:
                gate(f"{name}:hard_guard_params_exact", r_c.get("params") == r_y.get("params"))

    return dict(all_passed=all(checks.values()), gates=checks, failures=[k for k, v in checks.items() if not v],
                param_changes=param_changes)


def main():
    fingerprint = source_sha()
    assert NOREF_V3["rev"] == "t4z-2026-11-05" and THUMB_REV == "t4z"
    assert sha(ROOT / "src/filmcolor/paths.py") == "fced78f38a0787e473d882024d289878b99a224c604242f6561c7fb05e8be913"

    freeze_file = OUT / "source_freeze.json"
    if freeze_file.exists():
        freeze = read(freeze_file)
        assert fingerprint == freeze["src_content_sha256"], f"source changed: expected {freeze['src_content_sha256']} got {fingerprint}"
        assert sha(ROOT / "src/filmcolor/paths.py") == freeze["paths_sha256"], "paths.py changed"

    assert (OUT / "before_after.jpg").exists(), "before_after.jpg must exist before reporting"
    protocol = read(OUT / "input_protocol_sha256.json")
    for rel, digest in protocol["historical_json_sha256"].items():
        assert sha(ROOT / rel) == digest, f"historical artifact changed: {rel}"
    for rel, digest in protocol["frozen_manifest_sha256"].items():
        assert sha(ROOT / rel) == digest, f"frozen guard changed: {rel}"

    baseline_source_meta = read(OUT / "baseline_source.json")
    result = dict(source_sha256=fingerprint, baseline=baseline_source_meta, evaluations={},
                  input_protocol_sha256=sha(OUT / "input_protocol_sha256.json"))
    lines = ["# T4z 可重算验收表", "", f"候选源码 SHA256：`{fingerprint}`（含用户原有 paths.py 改动）。", "",
             "每帧先取 median ΔE2000，再跨帧统计 median / mean / p90。", "",
             "| 集合 | 预设 | N | median / mean / p90 | 对T4y 好/差 >.25 | 对T4y 好/差 >.05 | 对T4o 好/差 >.25 | 对T4o 好/差 >.05 |",
             "|---|---|---:|---|---|---|---|---|"]
    collections = {v: {"dev": {}, "cross": {}} for v in ("t4o", "t4r", "t4s", "t4t", "t4u", "t4v", "t4w", "t4x", "t4y", "t4z")}
    artifact_sha = {}
    for part in ("a", "b", "cross"):
        records = {v: committed_json(v, part) for v in ("t4o", "t4r", "t4s", "t4t", "t4u", "t4v", "t4w", "t4x", "t4y")}
        p = OUT / f"candidate_{part}.json"
        records["t4z"] = read(p)
        artifact_sha[str(p.relative_to(ROOT))] = sha(p)
        assert records["t4z"]["src_content_sha256"] == fingerprint
        assert records["t4y"]["src_content_sha256"] == result["baseline"]["src_content_sha256"]
        for field in ("manifest", "max_side", "ncc_min", "app_config", "roll_folder", "base_mode", "variants"):
            assert records["t4z"].get(field) == records["t4y"].get(field), (part, field)
        for version, record in records.items():
            group = collections[version]["cross" if part == "cross" else "dev"]
            assert not group.keys() & record["frames"].keys()
            group.update(record["frames"])
    identity_lines = []
    preset_improvements_vs_t4y = {}
    preset_worse0_vs_t4y = {}
    preset_worse0_vs_t4o = {}
    nonerror_counts = []
    for label, n, presets in (("dev", 185, PRESETS), ("cross", 123, ("standard",))):
        cand = collections["t4z"][label]
        if label == "dev":
            manifest = read(ROOT / "data/manifests/s3_dev_frames.json")["frames"]
            assert set(cand) == {f["id"] for f in manifest}
            assert {k for k, v in cand.items() if "error" in v} == EXPECTED_ERRORS
        else:
            manifest = read(ROOT / "data/manifests/bn_rb_frames.json")["frames"]
            assert set(cand) == {f["id"] for f in manifest if not f["s3_5"]}
            assert set(cand) - valid(cand) == {CHECK_EXCLUSION}
        ids = sorted(valid(cand))
        assert len(ids) == n
        entry = dict(n=n, identity={}, presets_vs_t4o={}, presets_vs_t4y={})
        for v in ("t4o", "t4r", "t4s", "t4t", "t4u", "t4v", "t4w", "t4x", "t4y"):
            old = collections[v][label]
            assert set(old) == set(cand) and valid(old) == set(ids)
            for k in ids:
                for field in ("raw_sha256", "ref_sha256"):
                    assert old[k][field] == cand[k][field], (label, k, field, v)
            if v != "t4o":
                entry["identity"][v] = compare_frames_identity_vs_prior(label, ids, old, cand, presets)
        for preset in presets:
            for k in ids:
                assert set(cand[k]["variants"]) == set(presets)
                assert np.isfinite(list(cand[k]["variants"][preset]["metrics"].values())).all()
            summary_o = summarize_preset_comparison(ids, collections["t4o"][label], cand, preset)
            summary_y = summarize_preset_comparison(ids, collections["t4y"][label], cand, preset)
            entry["presets_vs_t4o"][preset] = summary_o
            entry["presets_vs_t4y"][preset] = summary_y
            key_name = f"{label}_{preset}"
            preset_improvements_vs_t4y[key_name] = summary_y["better_005"]
            preset_worse0_vs_t4y[key_name] = (summary_y["worse_005"] == 0 and summary_y["worse_025"] == 0)
            preset_worse0_vs_t4o[key_name] = (summary_o["worse_005"] == 0 and summary_o["worse_025"] == 0)
            lines.append(f"| {label} | {preset} | {n} | {summary_y['stats_after']} | "
                         f"{summary_y['better_025']}/{summary_y['worse_025']} | {summary_y['better_005']}/{summary_y['worse_005']} | "
                         f"{summary_o['better_025']}/{summary_o['worse_025']} | {summary_o['better_005']}/{summary_o['worse_005']} |")
        nonerror = [k for k, v in cand.items() if "error" not in v]
        nonerror_counts.append(len(nonerror))
        for v in ("t4y", "t4x", "t4w", "t4v", "t4u", "t4t", "t4s", "t4r"):
            entry[f"inversion_identity_vs_{v}"] = dict(count=len(nonerror), differences=[dict(frame=k, field=field)
                for k in nonerror for field in ("d_hi", "white_point", "spike_clamp", "base_method")
                if cand[k].get(field) != collections[v][label][k].get(field)])
        result["evaluations"][label] = entry
        for v, identity in entry["identity"].items():
            identity_lines += ["", f"{label} 对 {v}：params 最大差 `{identity['max_abs_diff_params']}`，metrics 最大差 `{identity['max_abs_diff_metrics']}`，差异 `{identity['diff_count']}` 条。"]
    assert sum(nonerror_counts) == 309, f"Expected exactly 309 non-error inversion frames across dev and check, got {sum(nonerror_counts)}"
    lines += identity_lines
    result["protected_frames"] = {k: all(collections["t4z"]["dev"][k]["variants"][p][f] == collections["t4y"]["dev"][k]["variants"][p][f] == collections["t4o"]["dev"][k]["variants"][p][f]
        for p in PRESETS for f in ("params", "metrics")) for k in ("ext:A008", "ext:A012", "ext:A013", "shop:DSC_3833", "nlp:_DSF8948")}

    # Benchmark visuals SHA & content verification
    for label in ("baseline", "candidate"):
        version = "t4y" if label == "baseline" else "t4z"
        all_eval_frames = {}
        for grp in ("dev", "cross"):
            all_eval_frames.update(collections[version][grp])
        b_vis_path = OUT / f"benchmark_visuals_{label}.json"
        b_vis = read(b_vis_path)
        expected_src = result["baseline"]["src_content_sha256"] if label == "baseline" else fingerprint
        assert b_vis["source_sha256"] == expected_src, f"benchmark_visuals_{label} source SHA mismatch"
        assert set(b_vis["frames"]) == set(BENCHMARK_VISUAL_KEYS), f"benchmark_visuals_{label} frames mismatch"
        for bkey in BENCHMARK_VISUAL_KEYS:
            b_entry = b_vis["frames"][bkey]
            fid = b_entry["frame_id"]
            variant = b_entry["variant"]
            assert b_entry["raw_sha256"] == all_eval_frames[fid]["raw_sha256"], f"raw_sha256 mismatch for visual {bkey}"
            measured = all_eval_frames[fid]["variants"][variant]
            assert b_entry["params"] == measured["params"], f"visual params mismatch: {label} {bkey}"
            assert b_entry["de2000_median"] == measured["metrics"]["de2000_median"], f"visual metric mismatch: {label} {bkey}"
            img_file = OUT / b_entry["preview_image"]
            assert img_file.exists(), f"missing visual preview image: {img_file}"
            assert sha(img_file) == b_entry["preview_image_sha256"], f"preview image sha mismatch for {bkey}"

    base, cand = read(OUT / "devices_baseline.json"), read(OUT / "devices_candidate.json")
    assert base["src_content_sha256"] == result["baseline"]["src_content_sha256"]
    assert cand["src_content_sha256"] == fingerprint
    assert base["script_sha256"] == cand["script_sha256"] == sha(ROOT / "scripts/t4z_probe_devices.py")

    t4y_cand_path = ROOT / "experiments/t4_auto_ui/t4y/devices_candidate.json"
    assert t4y_cand_path.exists(), "t4y/devices_candidate.json is mandatory for exact comparison"
    t4y_candidate = read(t4y_cand_path)

    devices = device_checks(base, cand, t4y_candidate=t4y_candidate)

    # A1296 scope exact comparison against t4y
    scope = read(OUT / "a1296_scope.json")
    scope_t4y = read(ROOT / "experiments/t4_auto_ui/t4y/a1296_scope.json")
    assert scope["source_sha256"] == fingerprint and scope["raw_sha256"] == cand["a1296"]["raw_sha256"]
    devices["gates"]["A1296:scope_raw_sha_exact_vs_t4y"] = scope["raw_sha256"] == scope_t4y["raw_sha256"]
    devices["gates"]["A1296:scope_full_shape_exact_vs_t4y"] = scope["full_shape"] == scope_t4y["full_shape"]
    devices["gates"]["A1296:scope_all_regions_exact_vs_t4y"] = scope["regions"] == scope_t4y["regions"]
    devices["gates"]["A1296:scope_upper_band_any_gt_10_zero"] = scope["regions"]["upper_band"]["any_channel_gt_10_count"] == 0
    devices["gates"]["A1296:scope_center_any_gt_10_lt_1e_4"] = scope["regions"]["center"]["any_channel_gt_10_share"] < 1e-4

    devices["failures"] = [k for k, v in devices["gates"].items() if not v]
    devices["all_passed"] = all(devices["gates"].values())
    result["devices"] = devices
    result["a1296_scope"] = scope
    (OUT / "device_checks.json").write_text(json.dumps(devices, indent=2) + "\n")
    lines += ["", "## 真实样本", "", "| 样本 | 基线状态 | 候选状态 | 候选 d_hi | 候选 EV / Tint | 候选 Matrix[0][0] |", "|---|---|---|---|---|---|"]
    for name in TARGETS:
        b, r = base["targets"][name], cand["targets"][name]
        m00 = (r.get("params") or {}).get("matrix", [[None]])[0][0]
        lines.append(f"| {name} | {b.get('status', b.get('error'))} | {r.get('status', r.get('error'))} | {r.get('inversion', {}).get('d_hi')} | {r.get('params', {}).get('ev')} / {r.get('params', {}).get('tint')} | {m00} |")
    for name in sorted(set(NEW_TARGETS) | set(MATRIX_TARGETS)):
        rec_b = find_item_in_candidate(base, name) or {}
        rec_c = find_item_in_candidate(cand, name) or {}
        m00 = (rec_c.get("params") or {}).get("matrix", [[None]])[0][0]
        lines.append(f"| {name} | {rec_b.get('status', rec_b.get('error'))} | {rec_c.get('status', rec_c.get('error'))} | {rec_c.get('inversion', {}).get('d_hi')} | {rec_c.get('params', {}).get('ev')} / {rec_c.get('params', {}).get('tint')} | {m00} |")
    for name in sorted(NEW_TIFFS):
        b = base.get("new_tiff", {}).get(name, {})
        r = cand.get("new_tiff", {}).get(name, {})
        lines.append(f"| {name} | {b.get('status', b.get('error'))} | {r.get('status', r.get('error'))} | {r.get('inversion', {}).get('d_hi')} | {r.get('params', {}).get('ev')} / {r.get('params', {}).get('tint')} | - |")
    b096_b = base.get("special", {}).get("B096.tif", {})
    b096_c = cand.get("special", {}).get("B096.tif", {})
    lines.append(f"| B096.tif | {b096_b.get('status', b096_b.get('error'))} | {b096_c.get('status', b096_c.get('error'))} | {b096_c.get('inversion', {}).get('d_hi')} | {b096_c.get('params', {}).get('ev')} / {b096_c.get('params', {}).get('tint')} | - |")

    a = cand["a1296"]
    lines += ["", f"A1296 1600px：中心 p99={a['center_p99']}，上带蓝 p99={a['upper_band_p99'][2]}、max={a['upper_band_max'][2]}，整幅同带任一通道>10 占比={a['full_upper_band_share_any_channel_gt_10']}。",
              "", f"A1296 整张 full_positive 诊断任一通道>10 占比={scope['regions']['whole_image']['any_channel_gt_10_share']}；中心区域占比={scope['regions']['center']['any_channel_gt_10_share']}（计数={scope['regions']['center']['any_channel_gt_10_count']}）；上带计数={scope['regions']['upper_band']['any_channel_gt_10_count']}。",
              "", f"Acros {len(cand['acros'])} 帧、TIFF {len(cand['tiff'])} 帧、新 TIFF {len(cand.get('new_tiff', {}))} 帧、新目标样本 {len(cand.get('new_targets', {}))} 帧、特殊样本 {len(cand.get('special', {}))} 帧；设备失败项：{devices['failures']}。"]
    if devices.get("param_changes"):
        lines += ["", "### 参数变化记录（含解封矩阵与阻尼变化）", "", "| 样本 | EV 前 -> 后 | Blacks 前 -> 后 | Contrast 前 -> 后 | Matrix 变动 | Reason |", "|---|---|---|---|---|---|"]
        for pc in devices["param_changes"]:
            m_change = "Yes" if pc.get("matrix_before") != pc.get("matrix_after") else "No"
            lines.append(f"| {pc['frame']} | {pc['ev_before']} -> {pc['ev_after']} | {pc['blacks_before']} -> {pc['blacks_after']} | {pc['contrast_before']} -> {pc['contrast_after']} | {m_change} | {pc['reason']} |")

    tests = read(OUT / "test_evidence.json")
    assert tests["source_sha256"] == fingerprint
    for p, digest in tests["test_sha256"].items():
        assert sha(ROOT / p) == digest, p
    result["tests"] = tests
    baseline_failures = read(OUT / "baseline_failure_evidence.json")
    assert baseline_failures["failed"] >= 3
    assert baseline_failures["baseline_source_sha256"] == result["baseline"]["src_content_sha256"]
    for p, digest in baseline_failures["test_sha256"].items():
        assert sha(ROOT / p) == digest, p
    result["baseline_test_failures"] = baseline_failures
    service = read(OUT / "service_check.json")
    assert service["source_sha256"] == fingerprint
    result["service"] = service
    configs = {}
    for name in PRESETS:
        p = ROOT / "configs" / ("noref_v3.json" if name == "standard" else f"noref_v3_{name}.json")
        d = read(p)
        configs[name] = dict(sha256=sha(p), matches=(d == json.loads(json.dumps(dict(source_config_rev=NOREF_V3["rev"], preset=name, config=preset_config(name))))))
    result["configs"] = configs
    resources = {}
    for p in OUT.glob("*.log"):
        matches = re.findall(r"(\d+)\s+peak memory footprint", p.read_text(errors="replace"))
        if matches:
            resources[p.stem] = dict(peak_memory_footprint_bytes=max(map(int, matches)), log_sha256=sha(p))
    required = ("candidate_a", "candidate_b", "candidate_cross", "devices_baseline", "devices_candidate", "tests")

    # Diagnostics P2 verification (仅验证交付完整性与候选源指纹，不将特定假说/0匹配作为硬门槛)
    diag_p2_path = OUT / "diagnostics_p2.json"
    diag_p2_ok = False
    if diag_p2_path.exists():
        diag_p2 = read(diag_p2_path)
        diag_src = diag_p2.get("source_content_sha256") or diag_p2.get("source_sha256")
        diag_p2_ok = (
            diag_src == fingerprint
            and "diag_1_b094" in diag_p2
            and "diag_2_8bit_quantization" in diag_p2
            and "diag_3_base_chromaticity" in diag_p2
            and "diag_4_blue_cast_relaxation" in diag_p2
        )
        result["diagnostics_p2"] = dict(
            file=str(diag_p2_path.relative_to(ROOT)),
            sha256=sha(diag_p2_path),
            source_content_sha256=diag_src,
            complete=diag_p2_ok,
        )

    # Bind the diagnostic to actual counterfactual metrics, rather than a delegated placeholder.
    blue = read(OUT / "blue_cast_diagnostic.json")
    blue_eval = read(OUT / "eval_blue_cast_relaxation.json")
    assert blue["provenance"]["source_sha256"] == blue_eval["source_sha256"] == fingerprint
    selected = blue["selection"]
    blue_ids = set(selected["all_toggled_candidate_ids"])
    strict_ids = set(selected["strict_3way_candidate_ids"])
    assert strict_ids <= blue_ids and blue_ids == set(blue_eval["frames"]) == set(blue["frames"])
    assert len(blue_ids) == selected["all_toggled_subset_count"]
    assert len(strict_ids) == selected["strict_3way_subset_count"]
    official = collections["t4z"]["dev"] | collections["t4z"]["cross"]
    for key in blue_ids:
        record = blue_eval["frames"][key]
        for field in ("raw_sha256", "ref_sha256"):
            assert record[field] == official[key][field] == blue["frames"][key][field]
        for variant in ("standard", "relaxed_r"):
            for field in ("params", "metrics"):
                assert record["variants"][variant][field] == blue["frames"][key][variant][field]
        for field in ("params", "metrics"):
            assert record["variants"]["standard"][field] == official[key]["variants"]["standard"][field]
    for name, ids in (("strict_3way", strict_ids), ("all_toggled", blue_ids)):
        stats = blue["statistics_" + name]
        before = np.array([blue_eval["frames"][k]["variants"]["standard"]["metrics"]["de2000_median"] for k in sorted(ids)])
        after = np.array([blue_eval["frames"][k]["variants"]["relaxed_r"]["metrics"]["de2000_median"] for k in sorted(ids)])
        assert stats["n_evaluated"] == len(ids) and len(ids) > 0
        for threshold, suffix in ((0.05, "0_05"), (0.25, "0_25")):
            assert stats["better_delta_de_gt_" + suffix] == int(((after - before) < -threshold).sum())
            assert stats["worse_delta_de_gt_" + suffix] == int(((after - before) > threshold).sum())
        assert np.isclose(stats["mean_de2000_before"], before.mean(), atol=1e-12, rtol=0)
        assert np.isclose(stats["mean_de2000_after"], after.mean(), atol=1e-12, rtol=0)
    assert diag_p2["diag_4_blue_cast_relaxation"]["source_sha256"] == fingerprint
    assert diag_p2["diag_4_blue_cast_relaxation"]["statistics_strict_3way"] == blue["statistics_strict_3way"]
    result["blue_diagnostic"] = dict(diagnostic_sha256=sha(OUT / "blue_cast_diagnostic.json"),
                                     eval_sha256=sha(OUT / "eval_blue_cast_relaxation.json"),
                                     baseline_exact=True, strict_count=len(strict_ids), evaluated_count=len(blue_ids))

    # Blind rolls evaluation verification & provenance audit
    blind_path = OUT / "blind_rolls_eval.json"
    blind_prov_path = OUT / "blind_provenance.json"
    blind_ok = False
    if blind_path.exists() and blind_prov_path.exists():
        blind = read(blind_path)
        blind_prov = read(blind_prov_path)

        blind_src = blind.get("source_content_sha256") or blind.get("source_sha256")
        prov_src = blind_prov.get("source_content_sha256")
        src_ok = (blind_src == fingerprint and prov_src == fingerprint)

        # Provenance verification structure
        ver = blind_prov.get("verification", {})
        prov_complete = (
            ver.get("complete") is True
            and ver.get("all_checks_passed") is True
            and ver.get("all_three_rolls_present") is True
            and ver.get("all_six_mosaics_exist") is True
            and ver.get("data_file_sha256") == sha(blind_path)
            and blind_prov.get("total_physical_raw_files_audited") == 143
            and blind_prov.get("blind_script_sha256") == sha(ROOT / "scripts/t4z_blind_rolls.py")
        )

        rolls = blind.get("rolls", {})
        # Verify sample counts across single-frame and whole-roll modes
        counts_ok = (
            len(rolls.get("A7R5_RAW", {}).get("single_frame", [])) == 6
            and len(rolls.get("A7R5_RAW", {}).get("whole_roll", [])) == 6
            and len(rolls.get("Eastman5294", {}).get("single_frame", [])) == 10
            and len(rolls.get("Eastman5294", {}).get("whole_roll", [])) == 10
            and len(rolls.get("fomapan100", {}).get("single_frame", [])) == 9
            and len(rolls.get("fomapan100", {}).get("whole_roll", [])) == 9
        )

        sheets_ok = True
        sheet_hashes = {}
        for rname, rinfo in rolls.items():
            for skey in ("contact_sheet_single", "contact_sheet_whole"):
                fname = rinfo.get(skey, "")
                sfile = OUT / fname
                if not fname or not sfile.is_file():
                    sheets_ok = False
                else:
                    sheet_hashes[f"{rname}:{skey}"] = sha(sfile)

        for rname, rinfo in rolls.items():
            manifest = blind_prov["rolls_manifest"][rname]
            members = manifest["files"]
            indices = manifest.get("sample_indices", list(range(len(members))))
            expected = {str(Path(members[i]["path"]).resolve()): members[i]["raw_sha256"] for i in indices}
            for mode, requested in (("single_frame", None), ("whole_roll", "roll_nat")):
                records = rinfo[mode]
                assert len(records) == len(expected)
                assert {str(Path(v["full_path"]).resolve()) for v in records} == set(expected)
                for record in records:
                    assert record["base_mode_requested"] == requested
                    assert record["raw_sha256"] == expected[str(Path(record["full_path"]).resolve())]
            audited = ver["rolls_status"][rname]["mosaics"]
            assert sheet_hashes[rname + ":contact_sheet_single"] == audited["single_mosaic_sha256"]
            assert sheet_hashes[rname + ":contact_sheet_whole"] == audited["whole_mosaic_sha256"]
        blind_ok = src_ok and prov_complete and counts_ok and sheets_ok
        result["blind_rolls"] = dict(
            file=str(blind_path.relative_to(ROOT)),
            sha256=sha(blind_path),
            provenance_file=str(blind_prov_path.relative_to(ROOT)),
            provenance_sha256=sha(blind_prov_path),
            source_content_sha256=blind_src,
            blind_script_sha256=blind_prov.get("blind_script_sha256"),
            total_physical_raw_files_audited=blind_prov.get("total_physical_raw_files_audited"),
            rolls=list(rolls.keys()),
            sheet_sha256=sheet_hashes,
            all_passed=blind_ok,
        )

    result["gates"] = dict(
        all_preset_worse0_vs_t4y=all(preset_worse0_vs_t4y.values()),
        all_preset_net_positive_vs_t4y=all(v > 0 for v in preset_improvements_vs_t4y.values()),
        all_preset_worse0_vs_t4o=all(preset_worse0_vs_t4o.values()),
        inversion_diagnostics_exact=all(not e["inversion_identity_vs_t4y"]["differences"] for e in result["evaluations"].values()),
        protected_frames_exact=all(result["protected_frames"].values()),
        devices_all_passed=devices["all_passed"],
        tests_passed_ge_435=tests["passed"] >= 435,
        configs_all_match=all(c["matches"] for c in configs.values()),
        service_ok=service["status"] == "ok" and service["http_config_rev"] == NOREF_V3["rev"] and service["source_thumb_rev"] == "t4z",
        resources_complete=all(k in resources for k in required),
        diagnostics_p2_present=diag_p2_ok,
        blind_rolls_passed=blind_ok,
    )
    result["all_passed"] = all(result["gates"].values())
    artifact_sha.update({str(p.relative_to(ROOT)): sha(p) for p in OUT.glob("devices_*.json")})
    result["artifact_sha256"] = artifact_sha
    lines += ["", f"完整 pytest：{tests['passed']} passed，{tests['seconds']} 秒。", "", f"验收 gates：`{result['gates']}`", "", f"**all_passed={result['all_passed']}**", "",
              "图像对照仅用于检查伪影与算法行为；没有色卡真值，status=ok 不等于颜色正确。本轮独立复验指定样本，未复验研究方完整盲测。"]
    (OUT / "TABLES.md").write_text("\n".join(lines) + "\n")
    (OUT / "resource_evidence.json").write_text(json.dumps(resources, indent=2, sort_keys=True) + "\n")
    (OUT / "acceptance.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    scripts = sorted((ROOT / "scripts").glob("t4z_*.py")) + [ROOT / "scripts/t4d_source_eval.py", ROOT / "scripts/t4s_report_tables.py"]
    (OUT / "script_sha256.json").write_text(json.dumps({str(p.relative_to(ROOT)): sha(p) for p in scripts}, indent=2) + "\n")

    print(json.dumps(dict(gates=result["gates"], all_passed=result["all_passed"]), indent=2))


if __name__ == "__main__":
    main()
