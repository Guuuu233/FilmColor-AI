"""Rebuild T4y acceptance from measured artifacts; no original images are decoded.
Usage: python -I scripts/t4y_report_tables.py
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4y"
sys.path.insert(0, str(ROOT / "scripts"))
from t4s_report_tables import summarize_preset_vs_t4o, valid
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
BENCHMARK_VISUAL_IDS = ("blueneg:199807B-20-seattle", "blueneg:19960811B-25-arizona-london-bridge", "ext:A205")


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
    for group in ("targets", "new_targets", "tiff", "new_tiff"):
        if name in candidate.get(group, {}):
            return candidate[group][name]
    if name == "A1296.CR2":
        return candidate.get("a1296")
    return None


def device_checks(baseline, candidate, t4w_candidate):
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
        elif name in ("A1041.TIF", "A1027.TIF"):
            reason = "scanner_near_sat_isolation"
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
    gate("A1296:status_exact_vs_t4x", a.get("status") == b_a.get("status"))
    gate("A1296:inversion_exact_vs_t4x", a.get("inversion") == b_a.get("inversion"))
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
            t4w_r = t4w_candidate["targets"][name]
            gate(name + ":base_exact_vs_t4w", r.get("inversion", {}).get("base") == t4w_r.get("inversion", {}).get("base"))
            gate(name + ":d_hi_exact_vs_t4w", r.get("inversion", {}).get("d_hi") == t4w_r.get("inversion", {}).get("d_hi"))
            gate(name + ":params_exact_vs_t4w", r.get("params") == t4w_r.get("params"))
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
                t4w_r = t4w_candidate["tiff"][name]
                gate(name + ":base_exact_vs_t4w", r.get("inversion", {}).get("base") == t4w_r.get("inversion", {}).get("base"))
                gate(name + ":d_hi_exact_vs_t4w", r.get("inversion", {}).get("d_hi") == t4w_r.get("inversion", {}).get("d_hi"))

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
                # Verify not identity
                gate(f"{name}:matrix_not_identity", mat != [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # scanner metadata check for all probe frames in candidate
    for grp in ("targets", "tiff", "new_tiff"):
        for name, r in candidate.get(grp, {}).items():
            if r.get("status") == "ok" and Path(name).suffix.lower() in (".tif", ".tiff"):
                sc = r.get("scanner", {})
                gate(f"{name}:scanner_meta_present", ("bits" in sc and "tiff_eotf" in sc
                     and "prestretched_reconstruction" in sc and "single_channel_prestretched" in sc))

    return dict(all_passed=all(checks.values()), gates=checks, failures=[k for k, v in checks.items() if not v],
                param_changes=param_changes)


def main():
    fingerprint = source_sha()
    freeze = read(OUT / "source_freeze.json")
    assert fingerprint == freeze["src_content_sha256"]
    assert sha(ROOT / "src/filmcolor/paths.py") == freeze["paths_sha256"]
    assert NOREF_V3["rev"] == "t4y-2026-11-04" and THUMB_REV == "t4y"
    assert (OUT / "before_after.jpg").exists(), "before_after.jpg must exist before reporting"
    protocol = read(OUT / "input_protocol_sha256.json")
    for rel, digest in protocol["historical_json_sha256"].items():
        assert sha(ROOT / rel) == digest, f"historical artifact changed: {rel}"
    for rel, digest in protocol["frozen_manifest_sha256"].items():
        assert sha(ROOT / rel) == digest, f"frozen guard changed: {rel}"

    baseline_source_meta = read(OUT / "baseline_source.json")
    result = dict(source_sha256=fingerprint, baseline=baseline_source_meta, evaluations={},
                  input_protocol_sha256=sha(OUT / "input_protocol_sha256.json"))
    lines = ["# T4y 可重算验收表", "", f"候选源码 SHA256：`{fingerprint}`（含用户原有 paths.py 改动）。", "",
             "每帧先取 median ΔE2000，再跨帧统计 median / mean / p90。", "",
             "| 集合 | 预设 | N | median / mean / p90 | 对T4x 好/差 >.25 | 对T4x 好/差 >.05 | 对T4o 好/差 >.25 | 对T4o 好/差 >.05 |",
             "|---|---|---:|---|---|---|---|---|"]
    collections = {v: {"dev": {}, "cross": {}} for v in ("t4o", "t4r", "t4s", "t4t", "t4u", "t4v", "t4w", "t4x", "t4y")}
    artifact_sha = {}
    for part in ("a", "b", "cross"):
        records = {v: committed_json(v, part) for v in ("t4o", "t4r", "t4s", "t4t", "t4u", "t4v", "t4w", "t4x")}
        p = OUT / f"candidate_{part}.json"
        records["t4y"] = read(p)
        artifact_sha[str(p.relative_to(ROOT))] = sha(p)
        assert records["t4y"]["src_content_sha256"] == fingerprint
        assert records["t4x"]["src_content_sha256"] == result["baseline"]["src_content_sha256"]
        for field in ("manifest", "max_side", "ncc_min", "app_config", "roll_folder", "base_mode", "variants"):
            assert records["t4y"].get(field) == records["t4x"].get(field), (part, field)
        for version, record in records.items():
            group = collections[version]["cross" if part == "cross" else "dev"]
            assert not group.keys() & record["frames"].keys()
            group.update(record["frames"])
    identity_lines = []
    preset_improvements_vs_t4x = {}
    preset_worse0_vs_t4x = {}
    preset_worse0_vs_t4o = {}
    nonerror_counts = []
    for label, n, presets in (("dev", 185, PRESETS), ("cross", 123, ("standard",))):
        cand = collections["t4y"][label]
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
        entry = dict(n=n, identity={}, presets_vs_t4o={}, presets_vs_t4x={})
        for v in ("t4o", "t4r", "t4s", "t4t", "t4u", "t4v", "t4w", "t4x"):
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
            summary_x = summarize_preset_comparison(ids, collections["t4x"][label], cand, preset)
            entry["presets_vs_t4o"][preset] = summary_o
            entry["presets_vs_t4x"][preset] = summary_x
            key_name = f"{label}_{preset}"
            preset_improvements_vs_t4x[key_name] = summary_x["better_005"]
            preset_worse0_vs_t4x[key_name] = (summary_x["worse_005"] == 0 and summary_x["worse_025"] == 0)
            preset_worse0_vs_t4o[key_name] = (summary_o["worse_005"] == 0 and summary_o["worse_025"] == 0)
            lines.append(f"| {label} | {preset} | {n} | {summary_x['stats_after']} | "
                         f"{summary_x['better_025']}/{summary_x['worse_025']} | {summary_x['better_005']}/{summary_x['worse_005']} | "
                         f"{summary_o['better_025']}/{summary_o['worse_025']} | {summary_o['better_005']}/{summary_o['worse_005']} |")
        nonerror = [k for k, v in cand.items() if "error" not in v]
        nonerror_counts.append(len(nonerror))
        for v in ("t4x", "t4w", "t4v", "t4u", "t4t", "t4s", "t4r"):
            entry[f"inversion_identity_vs_{v}"] = dict(count=len(nonerror), differences=[dict(frame=k, field=field)
                for k in nonerror for field in ("d_hi", "white_point", "spike_clamp", "base_method")
                if cand[k].get(field) != collections[v][label][k].get(field)])
        result["evaluations"][label] = entry
        for v, identity in entry["identity"].items():
            identity_lines += ["", f"{label} 对 {v}：params 最大差 `{identity['max_abs_diff_params']}`，metrics 最大差 `{identity['max_abs_diff_metrics']}`，差异 `{identity['diff_count']}` 条。"]
    assert sum(nonerror_counts) == 309, f"Expected exactly 309 non-error inversion frames across dev and check, got {sum(nonerror_counts)}"
    lines += identity_lines
    result["protected_frames"] = {k: all(collections["t4y"]["dev"][k]["variants"][p][f] == collections["t4x"]["dev"][k]["variants"][p][f] == collections["t4o"]["dev"][k]["variants"][p][f]
        for p in PRESETS for f in ("params", "metrics")) for k in ("ext:A008", "ext:A012", "ext:A013", "shop:DSC_3833", "nlp:_DSF8948")}

    # Benchmark visuals SHA & content verification
    for label in ("baseline", "candidate"):
        version = "t4x" if label == "baseline" else "t4y"
        all_eval_frames = {}
        for grp in ("dev", "cross"):
            all_eval_frames.update(collections[version][grp])
        b_vis_path = OUT / f"benchmark_visuals_{label}.json"
        b_vis = read(b_vis_path)
        expected_src = result["baseline"]["src_content_sha256"] if label == "baseline" else fingerprint
        assert b_vis["source_sha256"] == expected_src, f"benchmark_visuals_{label} source SHA mismatch"
        assert set(b_vis["frames"]) == set(BENCHMARK_VISUAL_IDS), f"benchmark_visuals_{label} frames mismatch"
        for bid in BENCHMARK_VISUAL_IDS:
            b_entry = b_vis["frames"][bid]
            assert b_entry["raw_sha256"] == all_eval_frames[bid]["raw_sha256"], f"raw_sha256 mismatch for visual {bid}"
            measured = all_eval_frames[bid]["variants"]["standard"]
            assert b_entry["params"] == measured["params"], f"visual params mismatch: {label} {bid}"
            assert b_entry["de2000_median"] == measured["metrics"]["de2000_median"], f"visual metric mismatch: {label} {bid}"
            img_file = OUT / b_entry["preview_image"]
            assert img_file.exists(), f"missing visual preview image: {img_file}"
            assert sha(img_file) == b_entry["preview_image_sha256"], f"preview image sha mismatch for {bid}"

    base, cand = read(OUT / "devices_baseline.json"), read(OUT / "devices_candidate.json")
    assert base["src_content_sha256"] == result["baseline"]["src_content_sha256"]
    assert cand["src_content_sha256"] == fingerprint
    assert base["script_sha256"] == cand["script_sha256"] == sha(ROOT / "scripts/t4y_probe_devices.py")

    t4w_cand_path = ROOT / "experiments/t4_auto_ui/t4w/devices_candidate.json"
    assert t4w_cand_path.exists(), "t4w/devices_candidate.json is mandatory for exact scanner restoration comparison"
    t4w_candidate = read(t4w_cand_path)

    devices = device_checks(base, cand, t4w_candidate=t4w_candidate)

    # A1296 scope exact comparison against t4x
    scope = read(OUT / "a1296_scope.json")
    scope_t4x = read(ROOT / "experiments/t4_auto_ui/t4x/a1296_scope.json")
    assert scope["source_sha256"] == fingerprint and scope["raw_sha256"] == cand["a1296"]["raw_sha256"]
    devices["gates"]["A1296:scope_raw_sha_exact_vs_t4x"] = scope["raw_sha256"] == scope_t4x["raw_sha256"]
    devices["gates"]["A1296:scope_full_shape_exact_vs_t4x"] = scope["full_shape"] == scope_t4x["full_shape"]
    devices["gates"]["A1296:scope_all_regions_exact_vs_t4x"] = scope["regions"] == scope_t4x["regions"]
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
    a = cand["a1296"]
    lines += ["", f"A1296 1600px：中心 p99={a['center_p99']}，上带蓝 p99={a['upper_band_p99'][2]}、max={a['upper_band_max'][2]}，整幅同带任一通道>10 占比={a['full_upper_band_share_any_channel_gt_10']}。",
              "", f"A1296 整张 full_positive 诊断任一通道>10 占比={scope['regions']['whole_image']['any_channel_gt_10_share']}；中心区域占比={scope['regions']['center']['any_channel_gt_10_share']}（计数={scope['regions']['center']['any_channel_gt_10_count']}）；上带计数={scope['regions']['upper_band']['any_channel_gt_10_count']}。",
              "", f"Acros {len(cand['acros'])} 帧、TIFF {len(cand['tiff'])} 帧、新 TIFF {len(cand.get('new_tiff', {}))} 帧、新目标样本 {len(cand.get('new_targets', {}))} 帧；设备失败项：{devices['failures']}。"]
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
    result["gates"] = dict(
        all_preset_worse0_vs_t4x=all(preset_worse0_vs_t4x.values()),
        all_preset_net_positive_vs_t4x=all(v > 0 for v in preset_improvements_vs_t4x.values()),
        all_preset_worse0_vs_t4o=all(preset_worse0_vs_t4o.values()),
        inversion_diagnostics_exact=all(not e["inversion_identity_vs_t4x"]["differences"] for e in result["evaluations"].values()),
        protected_frames_exact=all(result["protected_frames"].values()),
        devices_all_passed=devices["all_passed"],
        tests_passed_ge_430=tests["passed"] >= 430,
        configs_all_match=all(c["matches"] for c in configs.values()),
        service_ok=service["status"] == "ok" and service["http_config_rev"] == NOREF_V3["rev"] and service["source_thumb_rev"] == "t4y",
        resources_complete=all(k in resources for k in required))
    result["all_passed"] = all(result["gates"].values())
    artifact_sha.update({str(p.relative_to(ROOT)): sha(p) for p in OUT.glob("devices_*.json")})
    result["artifact_sha256"] = artifact_sha
    lines += ["", f"完整 pytest：{tests['passed']} passed，{tests['seconds']} 秒。", "", f"验收 gates：`{result['gates']}`", "", f"**all_passed={result['all_passed']}**", "",
              "图像对照仅用于检查伪影与算法行为；没有色卡真值，status=ok 不等于颜色正确。本轮独立复验指定样本，未复验研究方完整盲测。"]
    (OUT / "TABLES.md").write_text("\n".join(lines) + "\n")
    (OUT / "resource_evidence.json").write_text(json.dumps(resources, indent=2, sort_keys=True) + "\n")
    (OUT / "acceptance.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    scripts = sorted((ROOT / "scripts").glob("t4y_*.py")) + [ROOT / "scripts/t4d_source_eval.py", ROOT / "scripts/t4s_report_tables.py"]
    (OUT / "script_sha256.json").write_text(json.dumps({str(p.relative_to(ROOT)): sha(p) for p in scripts}, indent=2) + "\n")
    print(json.dumps(dict(gates=result["gates"], all_passed=result["all_passed"]), indent=2))


if __name__ == "__main__":
    main()
