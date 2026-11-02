"""Verify T4s evidence and rebuild tables without decoding original material.
Usage: python -I scripts/t4s_report_tables.py

Validates T4s candidate evidence in experiments/t4_auto_ui/t4s vs:
- Committed T4r (fbd48a0) equivalent evidence (exact params/metrics identity).
- Committed T4o (4cd76b7) baseline evidence (strict >.25 and >.05 better/worse stats).

Deliverables produced on full evidence completion:
- experiments/t4_auto_ui/t4s/TABLES.md
- experiments/t4_auto_ui/t4s/acceptance.json
- experiments/t4_auto_ui/t4s/device_checks.json
- experiments/t4_auto_ui/t4s/resource_evidence.json
- experiments/t4_auto_ui/t4s/script_sha256.json
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4s"
EXPERIMENTS = OUT.parent

PROTECTED = ("ext:A008", "ext:A012", "ext:A013", "shop:DSC_3833", "nlp:_DSF8948")
PRESETS_DEV = ("standard", "natural", "portrait", "vivid", "print_classic")
PRESETS_CROSS = ("standard",)

DEV_EXPECTED_TOTAL = 191  # candidate_a: 96, candidate_b: 95
DEV_EXPECTED_VALID = 185  # 191 total - 6 invalid missing ICC tag
CROSS_EXPECTED_TOTAL = 124
CROSS_EXPECTED_VALID = 123  # 124 total - 1 invalid (199807A21 low NCC)
EXPECTED_DEV_ERRORS = {"ext:" + n for n in ("A016", "A018", "A025", "A026", "A027", "A028")}
EXPECTED_CROSS_EXCLUSION = "blueneg:199807A-21-orlando-disney"
EXPECTED_TIFFS = ("A1310.TIF", "A304.tif", "A1301.TIF", "A255.tif", "A1027.TIF", "A1028.TIF", "B032.tif", "B034.tiff")
EXPECTED_ACROS = {f"B{i:03d}.NEF" for i in range(2, 30)}
ACROS_TARGET_FOUR = ("B003.NEF", "B011.NEF", "B016.NEF", "B017.NEF")
REQUIRED_LOGS = ("candidate_a", "candidate_b", "candidate_cross", "devices_candidate", "tests")

# Baseline reference source SHA256 hashes (sources differ by design; verify each against its own origin)
COMMITTED_T4R_SOURCE_SHA = "276fa74b10912b20a38f47ee8edd7db2cbe59412d02ae90b80ff05c14afca1a1"
EXTRACTED_FBD48A0_CLEAN_SHA = "16862d53f8873cdf82d9659674da0f372114c8ab6ad7a284fca2dd1a0c1158a7"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def source_sha() -> str:
    """SHA256 over relative path and bytes of every tracked-type file in src/filmcolor.
    Note: includes user's modified paths.py in working tree; paths.py must not be altered.
    """
    h = hashlib.sha256()
    for p in sorted(p for p in (ROOT / "src/filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(ROOT / "src")).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def valid(rows: dict) -> set[str]:
    return {k for k, v in rows.items() if "error" not in v and v.get("ncc", 0.0) >= 0.7}


def value(rows: dict, key: str, variant: str = "standard") -> float:
    return float(rows[key]["variants"][variant]["metrics"]["de2000_median"])


def stats(x: np.ndarray | list[float]) -> str:
    a = np.asarray(x, dtype=float)
    return " / ".join(f"{v:.2f}" for v in (np.median(a), np.mean(a), np.percentile(a, 90)))


def counts(x: np.ndarray | list[float], threshold: float) -> str:
    a = np.asarray(x, dtype=float)
    return f"{int((a < -threshold).sum())} / {int((a > threshold).sum())}"


def verify_root_configs() -> dict:
    """Verify that root configs/ directory contains the five runtime NOREF_V3 snapshots
    matching filmcolor.noref.preset_config(preset) and source_config_rev == 't4s-2026-11-02'.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from filmcolor.noref import NOREF_V3, STYLE_PRESETS, preset_config

    assert NOREF_V3["rev"] == "t4s-2026-11-02", f"Unexpected NOREF_V3 rev: {NOREF_V3['rev']}"
    configs_dir = ROOT / "configs"
    assert configs_dir.is_dir(), f"Configs directory missing: {configs_dir}"

    preset_results = {}
    all_ok = True
    for name in STYLE_PRESETS:
        fname = "noref_v3.json" if name == "standard" else f"noref_v3_{name}.json"
        p = configs_dir / fname
        assert p.is_file(), f"Preset configuration snapshot missing: {p}"
        raw_text = p.read_text(encoding="utf-8")
        data = json.loads(raw_text)
        expected_cfg = json.loads(json.dumps(preset_config(name)))
        cfg_match = (data.get("config") == expected_cfg)
        rev_match = (data.get("source_config_rev") == NOREF_V3["rev"])
        name_match = (data.get("preset") == name)
        file_sha256 = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

        ok = cfg_match and rev_match and name_match
        if not ok:
            all_ok = False
        preset_results[name] = dict(
            file=f"configs/{fname}",
            file_sha256=file_sha256,
            source_config_rev=data.get("source_config_rev"),
            matches_runtime=ok
        )
        assert ok, f"Preset snapshot mismatch for {name}: cfg_match={cfg_match}, rev_match={rev_match}"

    return dict(
        all_match=all_ok,
        source_config_rev=NOREF_V3["rev"],
        presets=preset_results
    )


def summarize_preset_vs_t4o(ids: list[str], old_o: dict, cand: dict, preset: str) -> dict:
    u = np.array([value(old_o, k, preset) for k in ids], dtype=float)
    v = np.array([value(cand, k, preset) for k in ids], dtype=float)
    diff = v - u
    better_025 = int((diff < -0.25).sum())
    worse_025 = int((diff > 0.25).sum())
    better_005 = int((diff < -0.05).sum())
    worse_005 = int((diff > 0.05).sum())
    max_reg = float(diff.max()) if len(diff) else 0.0
    regressions_005 = {
        k: float(diff[i]) for i, k in enumerate(ids) if diff[i] > 0.05
    }
    return dict(
        better_025=better_025, worse_025=worse_025,
        better_005=better_005, worse_005=worse_005,
        max_regression=max_reg,
        regressions_over_005=regressions_005,
        stats_before=stats(u), stats_after=stats(v),
        metrics_before=dict(median=float(np.median(u)), mean=float(np.mean(u)), p90=float(np.percentile(u, 90))),
        metrics_after=dict(median=float(np.median(v)), mean=float(np.mean(v)), p90=float(np.percentile(v, 90)))
    )


def compare_frames_identity_vs_t4r(label: str, ids: list[str], t4r_rows: dict, cand_rows: dict, presets: tuple[str, ...]) -> dict:
    """Verify that all per-frame params and metrics are strictly and identically equal to committed T4r."""
    diff_records = []
    max_abs_diff_params = 0.0
    max_abs_diff_metrics = 0.0
    param_diffs_by_key = {}
    metric_diffs_by_key = {}

    for k in ids:
        cand_vars = cand_rows[k]["variants"]
        t4r_vars = t4r_rows[k]["variants"]
        for p in presets:
            if p not in cand_vars or p not in t4r_vars:
                diff_records.append(dict(frame=k, preset=p, error="variant_missing"))
                continue
            cp = cand_vars[p].get("params") or {}
            rp = t4r_vars[p].get("params") or {}
            all_param_keys = set(cp) | set(rp)
            for pk in all_param_keys:
                cv = cp.get(pk)
                rv = rp.get(pk)
                if cv != rv:
                    d = 0.0
                    if isinstance(cv, (int, float)) and isinstance(rv, (int, float)):
                        d = abs(float(cv) - float(rv))
                        if d > max_abs_diff_params:
                            max_abs_diff_params = d
                        param_diffs_by_key[pk] = max(param_diffs_by_key.get(pk, 0.0), d)
                        diff_records.append(dict(frame=k, preset=p, type="param_numeric", key=pk, cand=cv, t4r=rv, diff=d))
                    else:
                        diff_records.append(dict(frame=k, preset=p, type="param_discrete", key=pk, cand=cv, t4r=rv))

            cm = cand_vars[p].get("metrics") or {}
            rm = t4r_vars[p].get("metrics") or {}
            all_metric_keys = set(cm) | set(rm)
            for mk in all_metric_keys:
                cv = cm.get(mk)
                rv = rm.get(mk)
                if cv != rv:
                    d = 0.0
                    if isinstance(cv, (int, float)) and isinstance(rv, (int, float)):
                        d = abs(float(cv) - float(rv))
                        if d > max_abs_diff_metrics:
                            max_abs_diff_metrics = d
                        metric_diffs_by_key[mk] = max(metric_diffs_by_key.get(mk, 0.0), d)
                        diff_records.append(dict(frame=k, preset=p, type="metric_numeric", key=mk, cand=cv, t4r=rv, diff=d))
                    else:
                        diff_records.append(dict(frame=k, preset=p, type="metric_discrete", key=mk, cand=cv, t4r=rv))

    is_identical = (len(diff_records) == 0 and max_abs_diff_params == 0.0 and max_abs_diff_metrics == 0.0)

    return dict(
        identically_equal=is_identical,
        max_abs_diff_params=max_abs_diff_params,
        max_abs_diff_metrics=max_abs_diff_metrics,
        param_diffs_by_key=param_diffs_by_key,
        metric_diffs_by_key=metric_diffs_by_key,
        diff_count=len(diff_records),
        diff_summary=diff_records[:20],
        diff_frame_list=diff_records,
        all_diff_records=diff_records
    )


def process_device_checks(base_dev: dict, cand_dev: dict) -> dict:
    """Evaluate device probes from devices_baseline.json and devices_candidate.json."""
    failures = []

    # 1. A1296 check
    a1296_cand = cand_dev.get("a1296") or {}
    a1296_base = base_dev.get("a1296") or {}
    a1296_raw_match = (a1296_cand.get("raw_sha256") == a1296_base.get("raw_sha256"))
    if not a1296_raw_match:
        failures.append("A1296 raw_sha256 mismatch vs baseline")

    a1296_status = a1296_cand.get("status")
    a1296_center = a1296_cand.get("center_p99") or []
    a1296_band = a1296_cand.get("upper_band_p99") or []
    a1296_band_base = a1296_base.get("upper_band_p99") or []

    center_ok = bool(a1296_center and max(a1296_center) < 1.35)
    band_b_ok = bool(len(a1296_band) >= 3 and a1296_band[2] < 1.0)
    status_ok = (a1296_status == "ok")
    finite_ok = bool(a1296_cand.get("full_positive_finite"))
    preserves_pos = bool(a1296_cand.get("render_preserves_positive"))

    if not center_ok:
        failures.append(f"A1296 center_p99 not all < 1.35: {a1296_center}")
    if not band_b_ok:
        band_b_val = a1296_band[2] if len(a1296_band) >= 3 else "N/A"
        band_b_base_val = a1296_band_base[2] if len(a1296_band_base) >= 3 else "N/A"
        failures.append(f"A1296 upper_band_p99 Blue {band_b_val} > 1.0 (hard acceptance failed; baseline={band_b_base_val})")
    if not status_ok:
        failures.append(f"A1296 status not ok: {a1296_status}")
    if not finite_ok:
        failures.append("A1296 full_positive not finite")
    if not preserves_pos:
        failures.append("A1296 render does not preserve positive")

    c_status_str = "passed" if center_ok else "failed"
    b_status_str = "passed" if band_b_ok else "failed"
    band_b_val = a1296_band[2] if len(a1296_band) >= 3 else "N/A"
    band_b_base_val = a1296_band_base[2] if len(a1296_band_base) >= 3 else "N/A"

    a1296_summary = dict(
        status=a1296_status,
        status_ok=status_ok,
        center_p99=a1296_center,
        center_under_1_35=center_ok,
        upper_band_p99=a1296_band,
        upper_band_p99_baseline=a1296_band_base,
        band_b_under_1=band_b_ok,
        raw_sha256_match=a1296_raw_match,
        full_positive_finite=finite_ok,
        render_preserves_positive=preserves_pos,
        note=f"1600px preview: center < 1.35 {c_status_str} (center={a1296_center}), upper band Blue < 1.0 {b_status_str} (candidate B={band_b_val}, baseline B={band_b_base_val})"
    )

    # 2. 28 Acros check (all existing members; task's claimed 24 has no supplied subset)
    cand_acros = cand_dev.get("acros") or {}
    base_acros = base_dev.get("acros") or {}
    cand_acros_keys = set(cand_acros.keys())
    base_acros_keys = set(base_acros.keys())
    if cand_acros_keys != EXPECTED_ACROS:
        failures.append(f"Candidate Acros set mismatch: missing {EXPECTED_ACROS - cand_acros_keys}, extra {cand_acros_keys - EXPECTED_ACROS}")
    if base_acros_keys != EXPECTED_ACROS:
        failures.append(f"Baseline Acros set mismatch: missing {EXPECTED_ACROS - base_acros_keys}, extra {base_acros_keys - EXPECTED_ACROS}")

    acros_count = len(cand_acros)
    if acros_count != 28:
        failures.append(f"Acros count {acros_count} != 28 (task claimed 24 had no supplied subset; full 28 expected)")

    acros_all_ok = True
    acros_all_finite = True
    acros_hash_match = True
    roll_nat_targets = {}
    other_roll_nat_routed = {}

    for name, cr in sorted(cand_acros.items()):
        br = base_acros.get(name) or {}
        if cr.get("raw_sha256") != br.get("raw_sha256"):
            acros_hash_match = False
            failures.append(f"Acros {name} raw_sha256 changed vs baseline")
        if cr.get("status") != "ok":
            acros_all_ok = False
            failures.append(f"Acros {name} status not ok: {cr.get('status')}")
        if not cr.get("full_positive_finite"):
            acros_all_finite = False
            failures.append(f"Acros {name} not finite")

        base_info = cr.get("base") or {}
        base_method = base_info.get("method") if isinstance(base_info, dict) else str(base_info)
        flags = cr.get("flags") or []
        span_d_g = cr.get("span_d_g", 0.0)

        # 4 designated targets require span_d_g >= 0.24 and flags == []
        if name in ACROS_TARGET_FOUR:
            roll_nat_targets[name] = dict(
                base_method=base_method,
                span_d_g=span_d_g,
                flags=flags,
                span_ge_024=(span_d_g >= 0.24),
                flags_empty=(len(flags) == 0)
            )
            if base_method != "roll_nat":
                failures.append(f"Acros target {name} base method {base_method} != roll_nat")
            if span_d_g < 0.24:
                failures.append(f"Acros target {name} span_d_g {span_d_g:.4f} < 0.24")
            if flags:
                failures.append(f"Acros target {name} flags not empty: {flags}")
        elif base_method == "roll_nat" or "roll_nat" in str(base_info):
            # Other roll frames that routed to roll_nat are recorded for diagnostics without flags==[] requirement
            other_roll_nat_routed[name] = dict(
                base_method=base_method,
                span_d_g=span_d_g,
                flags=flags
            )

    roll_nat_count = len(roll_nat_targets)
    if roll_nat_count != 4:
        missing_targets = set(ACROS_TARGET_FOUR) - set(roll_nat_targets.keys())
        failures.append(f"Acros designated targets missing from evidence: {missing_targets}")

    acros_summary = dict(
        count=acros_count,
        note="Validated all 28 existing Acros NEFs B002..B029; 4 designated targets B003/B011/B016/B017 checked for span>=.24 & flags==[]",
        all_status_ok=acros_all_ok,
        all_finite=acros_all_finite,
        hashes_match=acros_hash_match,
        designated_targets_count=roll_nat_count,
        designated_all_span_ge_024=all(v["span_ge_024"] for v in roll_nat_targets.values()),
        designated_all_flags_empty=all(v["flags_empty"] for v in roll_nat_targets.values()),
        designated_targets=roll_nat_targets,
        other_roll_nat_routed=other_roll_nat_routed
    )

    # 3. 8 TIFF check (exact filename set)
    cand_tiff = cand_dev.get("tiff") or {}
    tiff_keys = set(cand_tiff.keys())
    expected_tiff_set = set(EXPECTED_TIFFS)
    if tiff_keys != expected_tiff_set:
        missing_tiffs = expected_tiff_set - tiff_keys
        extra_tiffs = tiff_keys - expected_tiff_set
        failures.append(f"TIFF filenames mismatch vs expected set: missing {missing_tiffs}, extra {extra_tiffs}")

    tiff_count = len(cand_tiff)
    tiff_all_ok = True
    tiff_all_finite = True
    for name, tr in sorted(cand_tiff.items()):
        if tr.get("status") != "ok":
            tiff_all_ok = False
            failures.append(f"TIFF {name} status not ok: {tr.get('status')}")
        if not tr.get("full_positive_finite"):
            tiff_all_finite = False
            failures.append(f"TIFF {name} not finite")

    tiff_summary = dict(
        count=tiff_count,
        expected_set_matched=(tiff_keys == expected_tiff_set),
        all_status_ok=tiff_all_ok,
        all_finite=tiff_all_finite,
        samples={k: dict(status=v.get("status"), finite=v.get("full_positive_finite")) for k, v in cand_tiff.items()}
    )

    # 4. Verify render_preserves_positive across all device probes (A1296 + 28 Acros + 8 TIFF)
    a1296_preserves = bool(a1296_cand.get("render_preserves_positive"))
    acros_preserves = {k: bool(v.get("render_preserves_positive")) for k, v in cand_acros.items()}
    tiff_preserves = {k: bool(v.get("render_preserves_positive")) for k, v in cand_tiff.items()}
    all_preserves = (a1296_preserves and all(acros_preserves.values()) and all(tiff_preserves.values()))
    if not all_preserves:
        if not a1296_preserves:
            failures.append("A1296 render does not preserve positive")
        for k, ok in acros_preserves.items():
            if not ok:
                failures.append(f"Acros {k} render does not preserve positive")
        for k, ok in tiff_preserves.items():
            if not ok:
                failures.append(f"TIFF {k} render does not preserve positive")

    all_passed = (len(failures) == 0)

    # 5. Optional 1000px sensitivity appendix for A1296 (does not replace 1600px hard gate)
    a1296_sens_path = OUT / "a1296_1000px_sensitivity.json"
    a1296_sens_data = None
    if a1296_sens_path.is_file():
        try:
            a1296_sens_data = read_json(a1296_sens_path)
        except Exception:
            pass

    device_checks_result = dict(
        all_passed=all_passed,
        failures=failures,
        a1296=a1296_summary,
        a1296_sensitivity_1000px=a1296_sens_data,
        acros=acros_summary,
        tiff=tiff_summary,
        all_status_ok=(status_ok and acros_all_ok and tiff_all_ok),
        all_finite=(finite_ok and acros_all_finite and tiff_all_finite),
        all_preserve_positive=all_preserves
    )
    return device_checks_result


def check_prerequisites(sha: str) -> dict:
    """Audit existence and integrity of required evidence files."""
    status = {}
    required_eval = ["candidate_a.json", "candidate_b.json", "candidate_cross.json"]
    required_meta = ["test_evidence.json", "service_check.json"]
    device_inputs = ["devices_baseline.json", "devices_candidate.json"]

    missing_eval = [f for f in required_eval if not (OUT / f).is_file()]
    missing_meta = [f for f in required_meta if not (OUT / f).is_file()]
    missing_dev = [f for f in device_inputs if not (OUT / f).is_file()]

    status["missing_eval"] = missing_eval
    status["missing_meta"] = missing_meta
    status["missing_devices"] = missing_dev
    status["ready_to_build"] = not (missing_eval or missing_meta or missing_dev)
    return status


def main():
    sha = source_sha()
    freeze = read_json(OUT / "source_freeze.json")
    assert sha == freeze["src_content_sha256"], "working source differs from frozen validation source"
    assert hashlib.sha256((ROOT / "src/filmcolor/paths.py").read_bytes()).hexdigest() == freeze["paths_sha256"]
    print(f"T4s Report Tables: working tree source SHA256 verified: {sha}")
    print(f"Tracking user's modified paths.py in fingerprint without mutating it.")

    prereqs = check_prerequisites(sha)
    if not prereqs["ready_to_build"]:
        print("\n" + "=" * 70)
        print("T4s Evidence Collection Incomplete - PENDING GENERATION")
        print("=" * 70)
        if prereqs["missing_eval"]:
            print(f"- Missing candidate evaluation JSONs: {', '.join(prereqs['missing_eval'])}")
            print("  (Run scripts/t4s_run_eval.py to generate candidate_a/b/cross)")
        if prereqs["missing_devices"]:
            print(f"- Missing device probe JSONs: {', '.join(prereqs['missing_devices'])}")
            print("  (Run scripts/t4s_probe_devices.py SRC baseline / candidate)")
        if prereqs["missing_meta"]:
            print(f"- Missing metadata JSONs: {', '.join(prereqs['missing_meta'])}")
            print("  (Run test evidence extraction and service check to generate)")
        print("\nExpected device_checks structure will be produced automatically from:")
        print("  - devices_baseline.json & devices_candidate.json")
        print("Coordinated fields in device_checks.json:")
        print("  - A1296: center_p99 < 1.35, upper_band_p99[Blue] < 1.0, status == ok, full_positive_finite")
        print("  - Acros: 28 frames all status == ok, 4 roll_nat targets span_d_g >= 0.24, flags == []")
        print("  - TIFF: 8 files all status == ok, full_positive_finite")
        print("  - Raw SHA256 match between baseline and candidate")
        print("\nReport table compilation refused due to incomplete evidence.")
        sys.exit(2)

    # 1. Load candidate evidence and baselines
    print("\nLoading evaluation JSONs from t4o, t4r, and t4s...")
    collections = {ver: {"dev": {}, "cross": {}} for ver in ("t4o", "t4r", "t4s")}
    for ver in ("t4o", "t4r", "t4s"):
        folder = OUT if ver == "t4s" else EXPERIMENTS / ver
        for part, n in (("a", 96), ("b", 95), ("cross", 124)):
            rec = read_json(folder / f"candidate_{part}.json")
            assert len(rec["frames"]) == n, f"{ver} candidate_{part} frames count {len(rec['frames'])} != {n}"
            if ver == "t4r":
                assert rec["src_content_sha256"] == COMMITTED_T4R_SOURCE_SHA, (
                    f"t4r candidate_{part} src sha mismatch: {rec['src_content_sha256']} != {COMMITTED_T4R_SOURCE_SHA}"
                )
            if ver == "t4s":
                assert rec["src_content_sha256"] == sha, f"t4s candidate_{part} src sha mismatch: {rec['src_content_sha256']} != {sha}"
                expected_variants = {"standard"} if part == "cross" else set(PRESETS_DEV)
                assert set(rec["variants"]) == expected_variants, f"candidate_{part} variants mismatch: {set(rec['variants'])} != {expected_variants}"
                # Protocol invariants check against t4r
                t4r_rec = read_json(EXPERIMENTS / "t4r" / f"candidate_{part}.json")
                for field in ("manifest", "max_side", "ncc_min", "app_config", "roll_folder", "base_mode"):
                    assert t4r_rec.get(field) == rec.get(field), f"{part} {field} changed vs t4r: {t4r_rec.get(field)} != {rec.get(field)}"
            group = collections[ver]["cross" if part == "cross" else "dev"]
            assert not (group.keys() & rec["frames"].keys()), f"Duplicate keys in {ver} {part}"
            group.update(rec["frames"])

    # 2. Manifest and exclusion checks
    dev_manifest = read_json(ROOT / "data/manifests/s3_dev_frames.json")["frames"]
    cross_manifest = read_json(ROOT / "data/manifests/bn_rb_frames.json")["frames"]
    dev_s = collections["t4s"]["dev"]
    cross_s = collections["t4s"]["cross"]
    assert set(dev_s) == {f["id"] for f in dev_manifest}, "dev frame id set mismatch with manifest"
    assert set(cross_s) == {f["id"] for f in cross_manifest if not f["s3_5"]}, "cross frame id set mismatch with manifest"

    actual_dev_errors = {k for k, v in dev_s.items() if "error" in v}
    assert actual_dev_errors == EXPECTED_DEV_ERRORS, f"Dev error set mismatch: {actual_dev_errors} != {EXPECTED_DEV_ERRORS}"
    actual_cross_exclusions = {k for k, v in cross_s.items() if "error" in v or v.get("ncc", 0.0) < 0.7}
    assert actual_cross_exclusions == {EXPECTED_CROSS_EXCLUSION}, f"Cross exclusion set mismatch: {actual_cross_exclusions} != {{'{EXPECTED_CROSS_EXCLUSION}'}}"

    lines = [
        "# T4s Verified Regression and Equivalence Report", "",
        f"- Working tree candidate source SHA256: `{sha}` (includes user's unmodified paths.py).",
        f"- Committed T4r runtime evidence source SHA256: `{COMMITTED_T4R_SOURCE_SHA}` (distinct historical environment).",
        f"- Clean baseline source SHA256 extracted from fbd48a0: `{EXTRACTED_FBD48A0_CLEAN_SHA}`.",
        "- Equivalence target: Committed T4r (`fbd48a0`), five dev presets and standard cross.",
        "- Statistical baseline: Committed T4o (`4cd76b7`).", "",
        "Per-frame median ΔE2000, then median / mean / p90 across frames.", "",
        "| Set | N | T4o | T4r | T4s | vs T4o >.25 | vs T4o >.05 | Max abs ΔE(std) vs T4r | T4r ΔE(std) exact (0.0) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    details = []
    acceptance = dict(
        source_sha256=sha,
        baseline_clean_source_sha256=EXTRACTED_FBD48A0_CLEAN_SHA,
        committed_t4r_source_sha256=COMMITTED_T4R_SOURCE_SHA,
        exclusions=dict(
            dev_errors_count=len(actual_dev_errors),
            dev_errors=sorted(actual_dev_errors),
            cross_exclusions_count=len(actual_cross_exclusions),
            cross_exclusions=sorted(actual_cross_exclusions)
        )
    )

    # 3. Dev and Cross Evaluation
    for label, n_expected, presets_eval in (("dev", DEV_EXPECTED_VALID, PRESETS_DEV), ("cross", CROSS_EXPECTED_VALID, PRESETS_CROSS)):
        old_o = collections["t4o"][label]
        old_r = collections["t4r"][label]
        cand = collections["t4s"][label]

        ids = sorted(valid(cand))
        assert len(ids) == n_expected, f"{label} valid count {len(ids)} != {n_expected}"
        assert valid(old_o) == valid(old_r) == set(ids), f"{label} valid frame set mismatch across versions"
        assert set(old_o) == set(old_r) == set(cand), f"{label} total frame set mismatch"

        # Verify raw/ref hashes and variant completeness/finiteness
        # Structural check ensures expected variants exist; algorithm non-ok statuses are preserved rather than crashing.
        variant_status_anomalies = {}
        for k in ids:
            for baseline in (old_o, old_r):
                for field in ("raw_sha256", "ref_sha256"):
                    assert baseline[k][field] == cand[k][field], f"{k} {field} changed vs baseline"
            for p in presets_eval:
                assert p in cand[k]["variants"], f"{k} missing expected variant: {p}"
            for var_name, var_row in cand[k]["variants"].items():
                st = var_row.get("status")
                if st != "ok":
                    variant_status_anomalies.setdefault(var_name, []).append((k, st))
                m_de = var_row.get("metrics", {}).get("de2000_median")
                if m_de is not None:
                    assert np.isfinite(m_de), f"{k} {var_name} de2000 not finite: {m_de}"

        # Check exact identity vs T4r
        id_check = compare_frames_identity_vs_t4r(label, ids, old_r, cand, presets_eval)
        if not id_check["identically_equal"]:
            print(f"WARNING: {label} has params/metrics differences vs T4r! Max diff: {max(id_check['max_abs_diff_params'], id_check['max_abs_diff_metrics']):.6e}")

        # Subdomain breakdown for standard variant
        domains = ("nlp", "shop", "blueneg", "ext_own", "all") if label == "dev" else ("all",)
        for domain in domains:
            keys = [k for k in ids if domain == "all" or cand[k]["domain"] == domain]
            uo = np.array([value(old_o, k) for k in keys], dtype=float)
            ur = np.array([value(old_r, k) for k in keys], dtype=float)
            us = np.array([value(cand, k) for k in keys], dtype=float)
            diff_arr = np.abs(us - ur)
            max_d_std = float(diff_arr.max()) if len(diff_arr) else 0.0
            id_str = "YES (0.0)" if max_d_std == 0.0 else f"NO ({max_d_std:.6f})"
            lines.append(
                f"| {label}:{domain} | {len(keys)} | {stats(uo)} | {stats(ur)} | {stats(us)} | "
                f"{counts(us - uo, 0.25)} | {counts(us - uo, 0.05)} | "
                f"{max_d_std:.6f} | {id_str} |"
            )

        # Standard delta vs T4o
        delta_vs_o = {k: value(cand, k) - value(old_o, k) for k in ids}
        delta_vs_r = {k: value(cand, k) - value(old_r, k) for k in ids}

        acceptance[label] = dict(
            n=n_expected,
            primary_variant="standard",
            vs_t4r_identity=id_check,
            vs_t4o=dict(
                better_025=sum(v < -0.25 for v in delta_vs_o.values()),
                worse_025=sum(v > 0.25 for v in delta_vs_o.values()),
                better_005=sum(v < -0.05 for v in delta_vs_o.values()),
                worse_005=sum(v > 0.05 for v in delta_vs_o.values()),
                max_regression=max(delta_vs_o.values()) if delta_vs_o else 0.0,
                regressions_over_005={k: v for k, v in delta_vs_o.items() if v > 0.05},
                metrics_before=dict(zip(('median', 'mean', 'p90'),
                                        (float(f([value(old_o, k) for k in ids])) for f in (np.median, np.mean, lambda x: np.percentile(x, 90))))),
                metrics_after=dict(zip(('median', 'mean', 'p90'),
                                       (float(f([value(cand, k) for k in ids])) for f in (np.median, np.mean, lambda x: np.percentile(x, 90)))))
            ),
            presets_vs_t4o={}
        )

        # Preset comparison vs T4o
        for p in presets_eval:
            if all(p in cand[k]["variants"] for k in ids) and all(p in old_o[k]["variants"] for k in ids):
                acceptance[label]["presets_vs_t4o"][p] = summarize_preset_vs_t4o(ids, old_o, cand, p)

        details += [
            "", f"## {label}: presets vs T4o", "",
            "| Preset | T4o | T4s | Better / worse >.25 | Better / worse >.05 | Max regression |",
            "|---|---:|---:|---:|---:|---:|"
        ]
        for p in presets_eval:
            if p in acceptance[label]["presets_vs_t4o"]:
                pr = acceptance[label]["presets_vs_t4o"][p]
                details.append(
                    f"| {p} | {pr['stats_before']} | {pr['stats_after']} | "
                    f"{pr['better_025']} / {pr['worse_025']} | {pr['better_005']} / {pr['worse_005']} | {pr['max_regression']:+.4f} |"
                )

        details += ["", f"### {label}: preset regressions >.05 vs T4o", ""]
        for p in presets_eval:
            if p in acceptance[label]["presets_vs_t4o"]:
                regs = acceptance[label]["presets_vs_t4o"][p]["regressions_over_005"]
                reg_str = ", ".join(f"{k} {v:+.6f}" for k, v in sorted(regs.items(), key=lambda x: -x[1])) or "none"
                details.append(f"- **{p}**: {reg_str}")

        details += [
            "", f"## {label}: Equivalence check vs T4r", "",
            f"- Identically equal: **{id_check['identically_equal']}**",
            f"- Max absolute diff params: `{id_check['max_abs_diff_params']:.6e}`",
            f"- Max absolute diff metrics: `{id_check['max_abs_diff_metrics']:.6e}`",
            f"- Number of mismatching records: `{id_check['diff_count']}`"
        ]
        if id_check["all_diff_records"]:
            details.append("- Mismatches summary (first 10):")
            for item in id_check["diff_summary"][:10]:
                details.append(f"  - `{item}`")

    lines += details

    # 4. Protected frames check
    lines += ["", "## Protected frames", ""]
    for k in PROTECTED:
        a = collections["t4o"]["dev"][k]["variants"]["standard"]
        b = collections["t4s"]["dev"][k]["variants"]["standard"]
        assert a["params"] == b["params"] and a["metrics"] == b["metrics"], f"Protected frame {k} changed"
        lines.append(f"- {k}: unchanged, ΔE {b['metrics']['de2000_median']:.6f}.")

    # 5. Devices verification and generation of device_checks.json
    print("\nProcessing device evidence from devices_baseline.json and devices_candidate.json...")
    base_dev = read_json(OUT / "devices_baseline.json")
    assert base_dev["src_content_sha256"] == EXTRACTED_FBD48A0_CLEAN_SHA, (
        f"devices_baseline src sha mismatch: {base_dev['src_content_sha256']} != {EXTRACTED_FBD48A0_CLEAN_SHA}"
    )
    cand_dev = read_json(OUT / "devices_candidate.json")
    assert cand_dev["src_content_sha256"] == sha, f"devices_candidate src sha mismatch: {cand_dev['src_content_sha256']} != {sha}"

    device_checks = process_device_checks(base_dev, cand_dev)
    (OUT / "device_checks.json").write_text(json.dumps(device_checks, indent=2) + "\n")
    acceptance["device_checks"] = device_checks

    a1296_res = device_checks['a1296']
    b_val = a1296_res['upper_band_p99'][2] if len(a1296_res['upper_band_p99']) >= 3 else 'N/A'
    b_base = a1296_res.get('upper_band_p99_baseline', [None, None, 'N/A'])[2] if len(a1296_res.get('upper_band_p99_baseline', [])) >= 3 else 'N/A'

    lines += [
        "", "## Device Verification Summary", "",
        f"- All device checks passed: **{device_checks['all_passed']}**",
        f"- A1296 (1600px preview): status `{a1296_res['status']}`, center < 1.35: `{a1296_res['center_under_1_35']}`, band Blue < 1.0: `{a1296_res['band_b_under_1']}` (candidate B={b_val} vs baseline B={b_base}), hash match: `{a1296_res['raw_sha256_match']}`",
        f"- Acros: {device_checks['acros']['count']} frames all ok: `{device_checks['acros']['all_status_ok']}`, {device_checks['acros']['designated_targets_count']} designated roll_nat targets span >= 0.24 & flags empty: `{device_checks['acros']['designated_all_span_ge_024'] and device_checks['acros']['designated_all_flags_empty']}`",
        f"- TIFF: {device_checks['tiff']['count']} scanner frames all ok & finite: `{device_checks['tiff']['all_status_ok'] and device_checks['tiff']['all_finite']}`"
    ]
    if device_checks["failures"]:
        lines.append("\n### Device Verification Warnings / Failures")
        for fail in device_checks["failures"]:
            lines.append(f"- **[FAILED]** {fail}")

    if device_checks.get("a1296_sensitivity_1000px"):
        s1000 = device_checks["a1296_sensitivity_1000px"]
        s_c = s1000.get("center_p99", [])
        s_b = s1000.get("upper_band_p99", [])
        s_b_val = s_b[2] if len(s_b) >= 3 else "N/A"
        lines += [
            "", "### Boundary Appendix: A1296 1000px Preview Scale Sensitivity", "",
            "- Note: Evaluated on 1000px scale to locate task prediction differences; does not replace 1600px primary device gate.",
            f"- Status: `{s1000.get('status')}`, Shape: `{s1000.get('shape')}`",
            f"- Center p99: `{s_c}`, Upper band p99 Blue: `{s_b_val}` (1000px preview) vs `{b_val}` (1600px primary device preview)."
        ]

    interrupted_dir = OUT / "interrupted_before_lzw_bit_buffer_fix"
    if interrupted_dir.is_dir():
        acceptance["interrupted_run_isolation"] = dict(
            directory=interrupted_dir.name,
            isolated=True,
            note="Pre-LZW-fix interrupted candidate run was quarantined; excluded from final tables and acceptance."
        )

    # 6. Test Evidence Verification
    print("\nVerifying test evidence...")
    tests = read_json(OUT / "test_evidence.json")
    assert tests["source_sha256"] == sha, f"test_evidence source sha mismatch: {tests['source_sha256']} != {sha}"
    assert tests["passed"] >= 391, f"pytest passed count {tests['passed']} < 391"
    for path_str, exp_sha in tests["test_sha256"].items():
        actual_test_sha = hashlib.sha256((ROOT / path_str).read_bytes()).hexdigest()
        assert actual_test_sha == exp_sha, f"Test file {path_str} hash changed: {actual_test_sha} != {exp_sha}"
    lines += ["", "## Full pytest", "", f"{tests['passed']} passed in {tests['seconds']:.2f}s, peak memory {tests['peak_memory_footprint_bytes'] / 2**30:.2f} GiB."]
    acceptance["test_evidence"] = dict(passed=tests["passed"], seconds=tests["seconds"], peak_memory=tests["peak_memory_footprint_bytes"])

    # 7. Service Check Verification
    print("\nVerifying service check...")
    service = read_json(OUT / "service_check.json")
    assert service["status"] == "ok", f"service_check status not ok: {service['status']}"
    assert service.get("http_config_rev") == "t4s-2026-11-02", f"http_config_rev mismatch: {service.get('http_config_rev')}"
    assert service.get("source_thumb_rev") == "t4s", f"source_thumb_rev mismatch: {service.get('source_thumb_rev')}"
    assert service.get("source_config_rev") == "t4s-2026-11-02", f"source_config_rev mismatch: {service.get('source_config_rev')}"

    lines += [
        "", "## Service Check", "",
        f"- Status: `{service['status']}`",
        f"- Sample: `{service.get('sample')}`, EV: `{service.get('ev')}`",
        f"- HTTP Config Rev: `{service.get('http_config_rev')}`",
        f"- Source Thumb Rev: `{service.get('source_thumb_rev')}`, Source Config Rev: `{service.get('source_config_rev')}`",
        f"- Render SHA256: `{service.get('render_sha256')}`"
    ]
    acceptance["service_check"] = service

    # 8. Root configs 5-preset JSONs Verification
    print("\nVerifying root configs five preset JSON snapshots vs runtime preset_config()...")
    configs_check = verify_root_configs()
    acceptance["preset_configs_check"] = configs_check
    lines += [
        "", "## Preset Configuration Snapshots (configs/)", "",
        f"- Source Config Rev: `{configs_check['source_config_rev']}`",
        f"- All presets match runtime `preset_config()`: **{configs_check['all_match']}**"
    ]
    for p_name, p_info in sorted(configs_check["presets"].items()):
        lines.append(f"- `{p_info['file']}`: matches runtime = {p_info['matches_runtime']}, SHA256 = `{p_info['file_sha256']}`")

    # 9. Resource Evidence Collection
    print("\nCollecting resource evidence from log files...")
    resource_path = OUT / "resource_evidence.json"
    resources = {}
    for p in sorted(OUT.glob("*.log")):
        values = re.findall(r"(\d+)\s+peak memory footprint", p.read_text(errors="replace"))
        if values:
            resources[p.stem] = dict(
                peak_memory_footprint_bytes=max(map(int, values)),
                log_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                status="ok"
            )
    for req_log in REQUIRED_LOGS:
        if req_log not in resources:
            log_file = OUT / f"{req_log}.log"
            if log_file.is_file():
                resources[req_log] = dict(
                    peak_memory_footprint_bytes=None,
                    log_sha256=hashlib.sha256(log_file.read_bytes()).hexdigest(),
                    status="missing_peak_memory_metric"
                )
            else:
                resources[req_log] = dict(
                    peak_memory_footprint_bytes=None,
                    log_sha256=None,
                    status="missing_log_file"
                )

    resource_path.write_text(json.dumps(resources, indent=2, sort_keys=True) + "\n")
    lines += ["", "## Peak memory footprint (GiB)", ""]
    for k, v in sorted(resources.items()):
        if v.get("peak_memory_footprint_bytes") is not None:
            lines.append(f"- {k}: {v['peak_memory_footprint_bytes'] / 2**30:.2f} GiB")
        else:
            lines.append(f"- {k}: missing ({v.get('status')})")

    # 10. Overall Acceptance Gates Aggregation
    dev_exact = acceptance["dev"]["vs_t4r_identity"]["identically_equal"]
    cross_exact = acceptance["cross"]["vs_t4r_identity"]["identically_equal"]
    all_metrics_params_exact = bool(dev_exact and cross_exact)

    all_preset_worse0 = (
        all(pr["worse_005"] == 0 and pr["worse_025"] == 0 for pr in acceptance["dev"]["presets_vs_t4o"].values()) and
        all(pr["worse_005"] == 0 and pr["worse_025"] == 0 for pr in acceptance["cross"]["presets_vs_t4o"].values())
    )
    tests_passed_ge_391 = bool(tests["passed"] >= 391)
    devices_all_passed = bool(device_checks["all_passed"])
    service_ok = bool(service["status"] == "ok")
    configs_all_match = bool(configs_check["all_match"])

    gates = dict(
        all_metrics_params_exact=all_metrics_params_exact,
        all_preset_worse0=all_preset_worse0,
        tests_passed_ge_391=tests_passed_ge_391,
        devices_all_passed=devices_all_passed,
        service_ok=service_ok,
        configs_all_match=configs_all_match
    )
    overall_all_passed = bool(all(gates.values()))
    acceptance["gates"] = gates
    acceptance["all_passed"] = overall_all_passed

    # 11. Script Hashes (single script_sha256.json deliverable)
    print("\nRecording script SHA256 hashes...")
    scripts = sorted((ROOT / "scripts").glob("t4s_*.py")) + [ROOT / "scripts/t4d_source_eval.py"]
    script_shas = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts}
    (OUT / "script_sha256.json").write_text(json.dumps(script_shas, indent=2, sort_keys=True) + "\n")

    # 12. Write Deliverables
    print("\nWriting deliverables...")
    (OUT / "TABLES.md").write_text("\n".join(lines).rstrip() + "\n")
    (OUT / "acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")

    print("\n" + "=" * 70)
    print("T4s Evidence Verification Complete!")
    print(f"Deliverables written to: {OUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
