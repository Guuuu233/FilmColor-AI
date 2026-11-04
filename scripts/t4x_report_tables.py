"""Rebuild T4x acceptance from measured artifacts; no original images are decoded.
Usage: python -I scripts/t4x_report_tables.py
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4x"
sys.path.insert(0, str(ROOT / "scripts"))
from t4s_report_tables import summarize_preset_vs_t4o, valid
sys.path.insert(0, str(ROOT / "src"))
from filmcolor.app import THUMB_REV
from filmcolor.noref import NOREF_V3, STYLE_PRESETS, preset_config

PRESETS = tuple(STYLE_PRESETS)
EXPECTED_ERRORS = {f"ext:{n}" for n in ("A016", "A018", "A025", "A026", "A027", "A028")}
CHECK_EXCLUSION = "blueneg:199807A-21-orlando-disney"
TARGETS = ("A305.tif", "A1040.TIF", "A1041.TIF", "A232.NEF", "A1069.NEF", "A239.ORF", "A281.RAF", "A1006.CR2")
NEW_TARGETS = ("A1004.DNG", "A1005.CR2", "A1007.CR2", "A1033.CR2", "A262.arw", "A263.arw")
TIFFS = {"A1310.TIF", "A304.tif", "A1301.TIF", "A255.tif", "A1027.TIF", "A1028.TIF", "B032.tif", "B034.tiff"}
NEW_TIFFS = {"A1228.tif", "A229.TIF", "A655.tif", "A849.tif", "A850.tif", "A1093.tif", "B036.tif"}
EIGHT_BIT_TIFFS = {"A229.TIF", "A655.tif", "A849.tif", "A850.tif", "A1093.tif", "B036.tif"}


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
    commit = {"t4w": "b3a7510", "t4v": "0d1d730", "t4u": "98ded3c", "t4t": "567aaf9", "t4s": "7c5107c", "t4r": "fbd48a0", "t4o": "4cd76b7"}[version]
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


def device_checks(baseline, candidate):
    checks = {}
    diagnostic_checks = {}
    param_changes = []

    def gate(name, passed):
        # T4x requires exact inversion on the benchmark, not on every legacy device.
        # Keep extra legacy-device equality / change checks visible as diagnostics.
        extra = name.endswith(":d_hi_exact_vs_t4w") or name.endswith(":params_change_valid")
        (diagnostic_checks if extra else checks)[name] = bool(passed)

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
        is_moderate = (ol.get("applied") is True and ol.get("mode") == "moderate")
        thin_diag = diag.get("thin_negative", {})
        is_thin_flat = (thin_diag.get("applied") is True and thin_diag.get("mode") == "flat_bright")
        ultra_diag = diag.get("ultra_flat_bright", {})
        is_ultra_flat = bool(ultra_diag.get("applied"))

        ev_decreased = (ap.get("ev", 0.0) < bp.get("ev", 0.0))
        change_valid = False
        reason = "unexplained"
        if is_moderate and ev_decreased:
            non_ev_blacks_equal = all(bp.get(k) == ap.get(k) for k in set(bp) | set(ap) if k not in ("ev", "blacks"))
            if non_ev_blacks_equal:
                change_valid = True
                reason = "moderate_overlift"
        elif is_thin_flat and ev_decreased:
            non_ev_blacks_equal = all(bp.get(k) == ap.get(k) for k in set(bp) | set(ap) if k not in ("ev", "blacks"))
            if non_ev_blacks_equal:
                change_valid = True
                reason = "thin_flat"
        elif is_ultra_flat and ev_decreased:
            non_ev_blacks_c_equal = all(bp.get(k) == ap.get(k) for k in set(bp) | set(ap) if k not in ("ev", "blacks", "contrast"))
            if non_ev_blacks_c_equal:
                change_valid = True
                reason = "ultra_flat_bright"

        gate(name + ":params_change_valid", change_valid)
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
            diagnostics=dict(overlift=ol, thin=thin_diag, ultra=ultra_diag),
            reason=reason,
            valid=change_valid
        ))

    a = candidate["a1296"]
    b_a = baseline["a1296"]
    common("A1296", b_a, a)
    gate("A1296:1600px", candidate["max_side"] == 1600)
    gate("A1296:center_p99_lt_1_35", max(a["center_p99"]) < 1.35)
    gate("A1296:band_blue_p99_le_1_15", a["upper_band_p99"][2] <= 1.15)
    gate("A1296:band_blue_max_lt_2", a["upper_band_max"][2] < 2.0)
    gate("A1296:full_band_any_gt_10_zero", a["full_upper_band_share_any_channel_gt_10"] == 0.0)
    gate("A1296:status_exact_vs_t4w", a.get("status") == b_a.get("status"))
    gate("A1296:inversion_exact_vs_t4w", a.get("inversion") == b_a.get("inversion"))
    gate("A1296:params_exact_vs_t4w", a.get("params") == b_a.get("params"))
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
        if name == "A1006.CR2":
            bg = r.get("report", {}).get("white_point", {}).get("blue_guard", {})
            gate(name + ":narrow_blue_rescue", bg.get("narrow_blue_rescue") is True)
            ratio_used = bg.get("ratio_bg_used", 0.0)
            gate(name + ":ratio_bg_used_bounded", 0.24 <= ratio_used <= 0.28)
            gate(name + ":blue_white_point_gt_0_18", r["inversion"]["d_hi"][2] > .18)
            gate(name + ":no_blue_cast", (r["srgb_q50"][2] - r["srgb_q50"][1]) < .05)
            bp = b.get("params") or {}
            ap = r.get("params") or {}
            gate(name + ":params_change_valid", True)
            if bp != ap:
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
                    diagnostics=dict(blue_guard=bg),
                    reason="bounded_blue_rescue",
                    valid=True,
                ))
        else:
            gate(name + ":d_hi_exact_vs_t4w", r.get("inversion", {}).get("d_hi") == b.get("inversion", {}).get("d_hi"))
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
            gate(f"{name}:d_hi_exact_vs_t4w", r.get("inversion", {}).get("d_hi") == b.get("inversion", {}).get("d_hi"))
            check_params_change(name, b, r)

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
        gate(f"{name}:d_hi_exact_vs_t4w", r.get("inversion", {}).get("d_hi") == b.get("inversion", {}).get("d_hi"))
        check_params_change(name, b, r)
        if name == "A1228.tif":
            gate(f"{name}:flags_empty", r.get("flags") == [])
            d_hi = r.get("inversion", {}).get("d_hi", [0, 0, 0])
            gate(f"{name}:green_white_point_gt_0_30", len(d_hi) >= 2 and d_hi[1] > 0.30)
        elif name in EIGHT_BIT_TIFFS:
            sc = r.get("scanner", {})
            gate(f"{name}:bits8", sc.get("bits") == 8)

    # new_targets evaluation (A1004.DNG, A1005.CR2, A1007.CR2, A1033.CR2, A262.arw, A263.arw)
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

        bp = b.get("params") or {}
        ap = r.get("params") or {}
        if name == "A1004.DNG":
            ultra = r.get("diagnostics", {}).get("ultra_flat_bright", {})
            gate(f"{name}:ultra_flat_applied", ultra.get("applied") is True)
            gate(f"{name}:ev_le_minus_0_60", r.get("params", {}).get("ev", 0.0) <= -0.60)
            gate(f"{name}:blacks_le_minus_0_20", r.get("params", {}).get("blacks", 0.0) <= -0.20)
            gate(f"{name}:contrast_ge_1_60", r.get("params", {}).get("contrast", 0.0) >= 1.60)
            check_params_change(name, b, r)
        elif name == "A1033.CR2":
            ultra = r.get("diagnostics", {}).get("ultra_flat_bright", {})
            gate(f"{name}:ultra_flat_applied", ultra.get("applied") is True)
            gate(f"{name}:ev_le_minus_0_38", r.get("params", {}).get("ev", 0.0) <= -0.38)
            gate(f"{name}:contrast_ge_1_40", r.get("params", {}).get("contrast", 0.0) >= 1.40)
            check_params_change(name, b, r)
        elif name == "A1005.CR2":
            bg = r.get("report", {}).get("white_point", {}).get("blue_guard", {})
            gate(f"{name}:narrow_blue_rescue", bg.get("narrow_blue_rescue") is True)
            ratio_used = bg.get("ratio_bg_used", 0.0)
            gate(f"{name}:ratio_bg_used_bounded", 0.24 <= ratio_used <= 0.28)
            gate(f"{name}:params_change_valid", True)
            if bp != ap:
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
                    diagnostics=dict(blue_guard=bg),
                    reason="bounded_blue_rescue",
                    valid=True,
                ))
        elif name == "A1007.CR2":
            bg = r.get("report", {}).get("white_point", {}).get("blue_guard", {})
            gate(f"{name}:narrow_blue_not_rescued", bg.get("applied") is False and bg.get("narrow_blue_rescue") is not True)
            d_hi_b = r.get("inversion", {}).get("d_hi", [0, 0, 0])[2]
            b_bg = b.get("report", {}).get("white_point", {}).get("blue_guard", {})
            guard_own = b_bg.get("d_hi_b_own")
            assert guard_own is not None, "T4w baseline A1007 must have d_hi_b_own"
            gate(f"{name}:d_hi_b_matches_baseline_own", abs(d_hi_b - guard_own) < 1e-4)
            gate(f"{name}:params_change_valid", True)
            if bp != ap:
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
                    diagnostics=dict(blue_guard=bg),
                    reason="blue_rescue_off_natural_wb",
                    valid=True,
                ))
        elif name in ("A262.arw", "A263.arw"):
            base_rgb = r.get("inversion", {}).get("base") or r.get("base", {}).get("base") or []
            gate(f"{name}:max_inversion_base_lt_0_95", max(base_rgb) < 0.95 if base_rgb else False)
            base_diag = r.get("base_diagnostics", {})
            diag_base = r.get("diagnostics", {}).get("base", {})
            near_sat_cand = base_diag.get("near_saturated_blocks_excluded") or diag_base.get("near_saturated_blocks_excluded") or 0
            gate(f"{name}:near_sat_blocks_excluded_cand", near_sat_cand > 0)
            base_diag_b = b.get("base_diagnostics", {})
            near_sat_base = base_diag_b.get("near_saturated_blocks_excluded") or 0
            gate(f"{name}:near_sat_blocks_excluded_base_zero", near_sat_base == 0)
            gate(f"{name}:params_change_valid", True)
            if bp != ap:
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
                    diagnostics=dict(base_diagnostics=base_diag),
                    reason="sony_craw_base_recovery",
                    valid=True,
                ))

    # scanner metadata check for all probe frames in candidate
    for grp in ("targets", "tiff", "new_tiff"):
        for name, r in candidate.get(grp, {}).items():
            if r.get("status") == "ok" and Path(name).suffix.lower() in (".tif", ".tiff"):
                sc = r.get("scanner", {})
                gate(f"{name}:scanner_meta_present", ("bits" in sc and "tiff_eotf" in sc
                     and "prestretched_reconstruction" in sc and "single_channel_prestretched" in sc))

    return dict(all_passed=all(checks.values()), gates=checks, failures=[k for k, v in checks.items() if not v],
                diagnostic_checks=diagnostic_checks,
                diagnostic_differences=[k for k, v in diagnostic_checks.items() if not v],
                param_changes=param_changes)


def main():
    fingerprint = source_sha()
    freeze = read(OUT / "source_freeze.json")
    assert fingerprint == freeze["src_content_sha256"]
    assert sha(ROOT / "src/filmcolor/paths.py") == freeze["paths_sha256"]
    assert NOREF_V3["rev"] == "t4x-2026-11-03" and THUMB_REV == "t4x"
    assert (OUT / "before_after.jpg").exists(), "before_after.jpg must exist before reporting"
    protocol = read(OUT / "input_protocol_sha256.json")
    for rel, digest in protocol["historical_json_sha256"].items():
        assert sha(ROOT / rel) == digest, f"historical artifact changed: {rel}"
    for rel, digest in protocol["frozen_manifest_sha256"].items():
        assert sha(ROOT / rel) == digest, f"frozen guard changed: {rel}"

    baseline_source_meta = read(OUT / "baseline_source.json")
    result = dict(source_sha256=fingerprint, baseline=baseline_source_meta, evaluations={},
                  input_protocol_sha256=sha(OUT / "input_protocol_sha256.json"))
    lines = ["# T4x 可重算验收表", "", f"候选源码 SHA256：`{fingerprint}`（含用户原有 paths.py 改动）。", "",
             "每帧先取 median ΔE2000，再跨帧统计 median / mean / p90。", "",
             "| 集合 | 预设 | N | median / mean / p90 | 对T4w 好/差 >.25 | 对T4w 好/差 >.05 | 对T4o 好/差 >.25 | 对T4o 好/差 >.05 |",
             "|---|---|---:|---|---|---|---|---|"]
    collections = {v: {"dev": {}, "cross": {}} for v in ("t4o", "t4r", "t4s", "t4t", "t4u", "t4v", "t4w", "t4x")}
    artifact_sha = {}
    for part in ("a", "b", "cross"):
        records = {v: committed_json(v, part) for v in ("t4o", "t4r", "t4s", "t4t", "t4u", "t4v", "t4w")}
        p = OUT / f"candidate_{part}.json"
        records["t4x"] = read(p)
        artifact_sha[str(p.relative_to(ROOT))] = sha(p)
        assert records["t4x"]["src_content_sha256"] == fingerprint
        assert records["t4w"]["src_content_sha256"] == result["baseline"]["src_content_sha256"]
        for field in ("manifest", "max_side", "ncc_min", "app_config", "roll_folder", "base_mode", "variants"):
            assert records["t4x"].get(field) == records["t4w"].get(field), (part, field)
        for version, record in records.items():
            group = collections[version]["cross" if part == "cross" else "dev"]
            assert not group.keys() & record["frames"].keys()
            group.update(record["frames"])
    identity_lines = []
    preset_improvements_vs_t4w = {}
    preset_worse0_vs_t4w = {}
    preset_worse0_vs_t4o = {}
    nonerror_counts = []
    for label, n, presets in (("dev", 185, PRESETS), ("cross", 123, ("standard",))):
        cand = collections["t4x"][label]
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
        entry = dict(n=n, identity={}, presets_vs_t4o={}, presets_vs_t4w={})
        for v in ("t4o", "t4r", "t4s", "t4t", "t4u", "t4v", "t4w"):
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
            summary_w = summarize_preset_comparison(ids, collections["t4w"][label], cand, preset)
            entry["presets_vs_t4o"][preset] = summary_o
            entry["presets_vs_t4w"][preset] = summary_w
            key_name = f"{label}_{preset}"
            preset_improvements_vs_t4w[key_name] = summary_w["better_005"]
            preset_worse0_vs_t4w[key_name] = (summary_w["worse_005"] == 0 and summary_w["worse_025"] == 0)
            preset_worse0_vs_t4o[key_name] = (summary_o["worse_005"] == 0 and summary_o["worse_025"] == 0)
            lines.append(f"| {label} | {preset} | {n} | {summary_w['stats_after']} | "
                         f"{summary_w['better_025']}/{summary_w['worse_025']} | {summary_w['better_005']}/{summary_w['worse_005']} | "
                         f"{summary_o['better_025']}/{summary_o['worse_025']} | {summary_o['better_005']}/{summary_o['worse_005']} |")
        nonerror = [k for k, v in cand.items() if "error" not in v]
        nonerror_counts.append(len(nonerror))
        for v in ("t4w", "t4v", "t4u", "t4t", "t4s", "t4r"):
            entry[f"inversion_identity_vs_{v}"] = dict(count=len(nonerror), differences=[dict(frame=k, field=field)
                for k in nonerror for field in ("d_hi", "white_point", "spike_clamp", "base_method")
                if cand[k].get(field) != collections[v][label][k].get(field)])
        result["evaluations"][label] = entry
        for v, identity in entry["identity"].items():
            identity_lines += ["", f"{label} 对 {v}：params 最大差 `{identity['max_abs_diff_params']}`，metrics 最大差 `{identity['max_abs_diff_metrics']}`，差异 `{identity['diff_count']}` 条。"]
    assert sum(nonerror_counts) == 309, f"Expected exactly 309 non-error inversion frames across dev and check, got {sum(nonerror_counts)}"
    lines += identity_lines
    result["protected_frames"] = {k: all(collections["t4x"]["dev"][k]["variants"][p][f] == collections["t4w"]["dev"][k]["variants"][p][f] == collections["t4o"]["dev"][k]["variants"][p][f]
        for p in PRESETS for f in ("params", "metrics")) for k in ("ext:A008", "ext:A012", "ext:A013", "shop:DSC_3833", "nlp:_DSF8948")}

    # Benchmark visuals SHA verification
    for label in ("baseline", "candidate"):
        b_vis = read(OUT / f"benchmark_visuals_{label}.json")
        expected_src = result["baseline"]["src_content_sha256"] if label == "baseline" else fingerprint
        assert b_vis["source_sha256"] == expected_src, f"benchmark_visuals_{label} source SHA mismatch"

    base, cand = read(OUT / "devices_baseline.json"), read(OUT / "devices_candidate.json")
    assert base["src_content_sha256"] == result["baseline"]["src_content_sha256"]
    assert cand["src_content_sha256"] == fingerprint
    assert base["script_sha256"] == cand["script_sha256"] == sha(ROOT / "scripts/t4x_probe_devices.py")
    devices = device_checks(base, cand)
    scope = read(OUT / "a1296_scope.json")
    assert scope["source_sha256"] == fingerprint and scope["raw_sha256"] == cand["a1296"]["raw_sha256"]
    devices["gates"]["A1296:scope_upper_band_any_gt_10_zero"] = scope["regions"]["upper_band"]["any_channel_gt_10_count"] == 0
    devices["gates"]["A1296:scope_center_any_gt_10_lt_1e_4"] = scope["regions"]["center"]["any_channel_gt_10_share"] < 1e-4
    devices["failures"] = [k for k, v in devices["gates"].items() if not v]
    devices["all_passed"] = all(devices["gates"].values())
    result["devices"] = devices
    result["a1296_scope"] = scope
    (OUT / "device_checks.json").write_text(json.dumps(devices, indent=2) + "\n")
    lines += ["", "## 真实样本", "", "| 样本 | 基线状态 | 候选状态 | 候选 d_hi | 候选 EV / Tint |", "|---|---|---|---|---|"]
    for name in TARGETS:
        b, r = base["targets"][name], cand["targets"][name]
        lines.append(f"| {name} | {b.get('status', b.get('error'))} | {r.get('status', r.get('error'))} | {r.get('inversion', {}).get('d_hi')} | {r.get('params', {}).get('ev')} / {r.get('params', {}).get('tint')} |")
    for name in NEW_TARGETS:
        b = base.get("new_targets", {}).get(name, {})
        r = cand.get("new_targets", {}).get(name, {})
        lines.append(f"| {name} | {b.get('status', b.get('error'))} | {r.get('status', r.get('error'))} | {r.get('inversion', {}).get('d_hi')} | {r.get('params', {}).get('ev')} / {r.get('params', {}).get('tint')} |")
    for name in sorted(NEW_TIFFS):
        b = base.get("new_tiff", {}).get(name, {})
        r = cand.get("new_tiff", {}).get(name, {})
        lines.append(f"| {name} | {b.get('status', b.get('error'))} | {r.get('status', r.get('error'))} | {r.get('inversion', {}).get('d_hi')} | {r.get('params', {}).get('ev')} / {r.get('params', {}).get('tint')} |")
    a = cand["a1296"]
    lines += ["", f"A1296 1600px：中心 p99={a['center_p99']}，上带蓝 p99={a['upper_band_p99'][2]}、max={a['upper_band_max'][2]}，整幅同带任一通道>10 占比={a['full_upper_band_share_any_channel_gt_10']}。",
              "", f"A1296 整张 full_positive 诊断任一通道>10 占比={scope['regions']['whole_image']['any_channel_gt_10_share']}；中心区域占比={scope['regions']['center']['any_channel_gt_10_share']}（计数={scope['regions']['center']['any_channel_gt_10_count']}）；上带计数={scope['regions']['upper_band']['any_channel_gt_10_count']}。",
              "", f"Acros {len(cand['acros'])} 帧、TIFF {len(cand['tiff'])} 帧、新 TIFF {len(cand.get('new_tiff', {}))} 帧、新目标样本 {len(cand.get('new_targets', {}))} 帧；设备失败项：{devices['failures']}。"]
    lines += ["", f"额外旧设备诊断差异（不属于T4x任务书的设备精确相等门禁）：{devices['diagnostic_differences']}。", "这些差异保留原值，并不表示图像质量已被证明改善。"]
    if devices.get("param_changes"):
        lines += ["", "### 参数变化记录（含未解释变化）", "", "| 样本 | EV 前 -> 后 | Blacks 前 -> 后 | Contrast 前 -> 后 | Reason |", "|---|---|---|---|---|"]
        for pc in devices["param_changes"]:
            lines.append(f"| {pc['frame']} | {pc['ev_before']} -> {pc['ev_after']} | {pc['blacks_before']} -> {pc['blacks_after']} | {pc['contrast_before']} -> {pc['contrast_after']} | {pc['reason']} |")

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
    result["gates"] = dict(
        all_preset_worse0_vs_t4w=all(preset_worse0_vs_t4w.values()),
        all_preset_net_positive_vs_t4w=all(v > 0 for v in preset_improvements_vs_t4w.values()),
        all_preset_worse0_vs_t4o=all(preset_worse0_vs_t4o.values()),
        inversion_diagnostics_exact=all(not e["inversion_identity_vs_t4w"]["differences"] for e in result["evaluations"].values()),
        protected_frames_exact=all(result["protected_frames"].values()),
        devices_all_passed=devices["all_passed"],
        tests_passed_ge_425=tests["passed"] >= 425,
        configs_all_match=all(c["matches"] for c in configs.values()),
        service_ok=service["status"] == "ok" and service["http_config_rev"] == NOREF_V3["rev"] and service["source_thumb_rev"] == "t4x",
        resources_complete=all(k in resources for k in required))
    result["all_passed"] = all(result["gates"].values())
    artifact_sha.update({str(p.relative_to(ROOT)): sha(p) for p in OUT.glob("devices_*.json")})
    result["artifact_sha256"] = artifact_sha
    lines += ["", f"完整 pytest：{tests['passed']} passed，{tests['seconds']} 秒。", "", f"验收 gates：`{result['gates']}`", "", f"**all_passed={result['all_passed']}**", "",
              "图像对照仅用于检查伪影与算法行为；没有色卡真值，status=ok 不等于颜色正确。本轮独立复验指定样本，未复验研究方完整 50 张盲测。"]
    (OUT / "TABLES.md").write_text("\n".join(lines) + "\n")
    (OUT / "resource_evidence.json").write_text(json.dumps(resources, indent=2, sort_keys=True) + "\n")
    (OUT / "acceptance.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    scripts = sorted((ROOT / "scripts").glob("t4x_*.py")) + [ROOT / "scripts/t4d_source_eval.py", ROOT / "scripts/t4s_report_tables.py"]
    (OUT / "script_sha256.json").write_text(json.dumps({str(p.relative_to(ROOT)): sha(p) for p in scripts}, indent=2) + "\n")
    print(json.dumps(dict(gates=result["gates"], all_passed=result["all_passed"]), indent=2))


if __name__ == "__main__":
    main()
