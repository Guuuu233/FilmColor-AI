"""Rebuild T4u acceptance from measured artifacts; no original images are decoded.
Usage: python -I scripts/t4u_report_tables.py
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4u"
sys.path.insert(0, str(ROOT / "scripts"))
from t4s_report_tables import compare_frames_identity_vs_t4r, summarize_preset_vs_t4o, valid
sys.path.insert(0, str(ROOT / "src"))
from filmcolor.app import THUMB_REV
from filmcolor.noref import NOREF_V3, STYLE_PRESETS, preset_config

PRESETS = tuple(STYLE_PRESETS)
EXPECTED_ERRORS = {f"ext:{n}" for n in ("A016", "A018", "A025", "A026", "A027", "A028")}
CHECK_EXCLUSION = "blueneg:199807A-21-orlando-disney"
TARGETS = ("A305.tif", "A1040.TIF", "A1041.TIF", "A232.NEF", "A1069.NEF", "A239.ORF", "A281.RAF", "A1006.CR2")
TIFFS = {"A1310.TIF", "A304.tif", "A1301.TIF", "A255.tif", "A1027.TIF", "A1028.TIF", "B032.tif", "B034.tiff"}


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
    commit = {"t4t": "567aaf9", "t4s": "7c5107c", "t4r": "fbd48a0", "t4o": "4cd76b7"}[version]
    rel = f"experiments/t4_auto_ui/{version}/candidate_{part}.json"
    data = subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=ROOT)
    assert data == (ROOT / rel).read_bytes(), f"historical artifact modified: {rel}"
    return json.loads(data)


def device_checks(baseline, candidate):
    checks = {}

    def gate(name, passed):
        checks[name] = bool(passed)

    def common(name, before, after):
        gate(name + ":input_sha", before.get("raw_sha256") == after.get("raw_sha256") and bool(after.get("raw_sha256")))
        gate(name + ":ok_finite_preserved", after.get("status") == "ok" and after.get("full_positive_finite")
             and after.get("render_preserves_positive"))

    a = candidate["a1296"]
    common("A1296", baseline["a1296"], a)
    gate("A1296:1600px", candidate["max_side"] == 1600)
    gate("A1296:center_p99_lt_1_35", max(a["center_p99"]) < 1.35)
    gate("A1296:band_blue_p99_le_1_15", a["upper_band_p99"][2] <= 1.15)
    gate("A1296:band_blue_max_lt_2", a["upper_band_max"][2] < 2.0)
    gate("A1296:full_band_any_gt_10_zero", a["full_upper_band_share_any_channel_gt_10"] == 0.0)
    gate("targets:membership", set(candidate["targets"]) == set(TARGETS))
    for name in TARGETS:
        r, b = candidate["targets"][name], baseline["targets"][name]
        common(name, b, r)
        if name == "A305.tif":
            gate(name + ":no_white_point_invalid", not any("white_point_invalid" in f for f in r["flags"]))
        if name in ("A1040.TIF", "A1041.TIF"):
            gate(name + ":channel_cap_B", "B" in r["report"]["white_point"].get("channel_cap", {}))
            gate(name + ":blue_white_point_lt_0_80", r["inversion"]["d_hi"][2] < .80)
            gate(name + ":green_q50_gt_0_05", r["srgb_q50"][1] > .05)
        if name in ("A232.NEF", "A1069.NEF"):
            gate(name + ":holder_excluded", r["report"]["white_point"].get("holder_blocks_excluded", 0) > 0)
            gate(name + ":white_points_lt_1_15", max(r["inversion"]["d_hi"]) < 1.15)
            gate(name + ":holder_share_gt_0_50", r["holder_share"] > .50)
            gate(name + ":ev_lt_0_50", r["params"]["ev"] < .50)
        wb = r["diagnostics"].get("wb", {}).get("diagnostics", {})
        if name == "A239.ORF":
            gate(name + ":no_cyan_conf_boost", not wb.get("cyan_conf_boost", False))
            gate(name + ":no_narrow_tint_capped", wb.get("no_narrow_tint_capped", False) is True)
            gate(name + ":tint_lt_0_22", r["params"]["tint"] < .22)
        if name == "A281.RAF":
            gate(name + ":tint_conflict_damped", wb.get("tint_conflict_damped", False))
            gate(name + ":tint_lt_0_18", r["params"]["tint"] < .18)
        if name == "A1006.CR2":
            gate(name + ":narrow_blue_rescue", r["report"]["white_point"].get("blue_guard", {}).get("narrow_blue_rescue") is True)
            gate(name + ":blue_white_point_gt_0_18", r["inversion"]["d_hi"][2] > .18)
            gate(name + ":no_blue_cast", (r["srgb_q50"][2] - r["srgb_q50"][1]) < .05)
    gate("acros:membership", set(candidate["acros"]) == {f"B{i:03d}.NEF" for i in range(2, 30)})
    gate("tiff:membership", set(candidate["tiff"]) == TIFFS)
    for group in ("acros", "tiff"):
        for name, r in candidate[group].items():
            common(name, baseline[group][name], r)
    return dict(all_passed=all(checks.values()), gates=checks, failures=[k for k, v in checks.items() if not v])


def main():
    fingerprint = source_sha()
    freeze = read(OUT / "source_freeze.json")
    assert fingerprint == freeze["src_content_sha256"]
    assert sha(ROOT / "src/filmcolor/paths.py") == freeze["paths_sha256"]
    assert NOREF_V3["rev"] == "t4u-2026-11-03" and THUMB_REV == "t4u"
    protocol = read(OUT / "input_protocol_sha256.json")
    for rel, digest in protocol["historical_json_sha256"].items():
        assert sha(ROOT / rel) == digest, f"historical artifact changed: {rel}"
    for rel, digest in protocol["frozen_manifest_sha256"].items():
        assert sha(ROOT / rel) == digest, f"frozen guard changed: {rel}"
    result = dict(source_sha256=fingerprint, baseline=read(OUT / "baseline_source.json"), evaluations={},
                  input_protocol_sha256=sha(OUT / "input_protocol_sha256.json"))
    lines = ["# T4u 可重算验收表", "", f"候选源码 SHA256：`{fingerprint}`（含用户原有 paths.py 改动）。", "",
             "每帧先取 median ΔE2000，再跨帧统计 median / mean / p90。", "",
             "| 集合 | 预设 | N | median / mean / p90 | 对T4o 好/差 >.25 | 对T4o 好/差 >.05 |", "|---|---|---:|---|---|---|"]
    collections = {v: {"dev": {}, "cross": {}} for v in ("t4o", "t4r", "t4s", "t4t", "t4u")}
    artifact_sha = {}
    for part in ("a", "b", "cross"):
        records = {v: committed_json(v, part) for v in ("t4o", "t4r", "t4s", "t4t")}
        p = OUT / f"candidate_{part}.json"
        records["t4u"] = read(p)
        artifact_sha[str(p.relative_to(ROOT))] = sha(p)
        assert records["t4u"]["src_content_sha256"] == fingerprint
        assert records["t4t"]["src_content_sha256"] == result["baseline"]["src_content_sha256"]
        for field in ("manifest", "max_side", "ncc_min", "app_config", "roll_folder", "base_mode", "variants"):
            assert records["t4u"].get(field) == records["t4t"].get(field), (part, field)
        for version, record in records.items():
            group = collections[version]["cross" if part == "cross" else "dev"]
            assert not group.keys() & record["frames"].keys()
            group.update(record["frames"])
    identity_lines = []
    for label, n, presets in (("dev", 185, PRESETS), ("cross", 123, ("standard",))):
        cand = collections["t4u"][label]
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
        entry = dict(n=n, identity={}, presets_vs_t4o={})
        for v in ("t4o", "t4r", "t4s", "t4t"):
            old = collections[v][label]
            assert set(old) == set(cand) and valid(old) == set(ids)
            for k in ids:
                for field in ("raw_sha256", "ref_sha256"):
                    assert old[k][field] == cand[k][field], (label, k, field, v)
            if v != "t4o":
                entry["identity"][v] = compare_frames_identity_vs_t4r(label, ids, old, cand, presets)
        for preset in presets:
            for k in ids:
                assert set(cand[k]["variants"]) == set(presets)
                assert np.isfinite(list(cand[k]["variants"][preset]["metrics"].values())).all()
            summary = summarize_preset_vs_t4o(ids, collections["t4o"][label], cand, preset)
            entry["presets_vs_t4o"][preset] = summary
            lines.append(f"| {label} | {preset} | {n} | {summary['stats_after']} | {summary['better_025']}/{summary['worse_025']} | {summary['better_005']}/{summary['worse_005']} |")
        nonerror = [k for k, v in cand.items() if "error" not in v]
        for v in ("t4t", "t4s", "t4r"):
            entry[f"inversion_identity_vs_{v}"] = dict(count=len(nonerror), differences=[dict(frame=k, field=field)
                for k in nonerror for field in ("d_hi", "white_point", "spike_clamp", "base_method")
                if cand[k].get(field) != collections[v][label][k].get(field)])
        result["evaluations"][label] = entry
        for v, identity in entry["identity"].items():
            identity_lines += ["", f"{label} 对 {v}：params 最大差 `{identity['max_abs_diff_params']}`，metrics 最大差 `{identity['max_abs_diff_metrics']}`，差异 `{identity['diff_count']}` 条。"]
    lines += identity_lines
    result["protected_frames"] = {k: all(collections["t4u"]["dev"][k]["variants"][p][f] == collections["t4o"]["dev"][k]["variants"][p][f]
        for p in PRESETS for f in ("params", "metrics")) for k in ("ext:A008", "ext:A012", "ext:A013", "shop:DSC_3833", "nlp:_DSF8948")}
    base, cand = read(OUT / "devices_baseline.json"), read(OUT / "devices_candidate.json")
    assert base["src_content_sha256"] == result["baseline"]["src_content_sha256"]
    assert cand["src_content_sha256"] == fingerprint
    assert base["script_sha256"] == cand["script_sha256"] == sha(ROOT / "scripts/t4u_probe_devices.py")
    devices = device_checks(base, cand)
    saturation = read(OUT / "scanner_saturation.json")
    assert saturation["source_sha256"] == fingerprint
    for name, row in saturation["frames"].items():
        assert row["raw_sha256"] == cand["targets"][name]["raw_sha256"]
        devices["gates"][name + ":single_channel_prestretched_R"] = row["single_channel_prestretched"] == "R" and row["gb_mask_exact"]
    result["scanner_saturation"] = saturation
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
    a = cand["a1296"]
    lines += ["", f"A1296 1600px：中心 p99={a['center_p99']}，上带蓝 p99={a['upper_band_p99'][2]}、max={a['upper_band_max'][2]}，整幅同带任一通道>10 占比={a['full_upper_band_share_any_channel_gt_10']}。",
              "", f"A1296 整张 full_positive 诊断任一通道>10 占比={scope['regions']['whole_image']['any_channel_gt_10_share']}；中心区域占比={scope['regions']['center']['any_channel_gt_10_share']}（计数={scope['regions']['center']['any_channel_gt_10_count']}）；上带计数={scope['regions']['upper_band']['any_channel_gt_10_count']}。",
              "", "T4u 按任务书将 A1296 门禁定义为 scope_upper_band_any_gt_10_zero 与 scope_center_any_gt_10_lt_1e_4；整图统计保留为诊断，中心仍有极端值，不能将整图高值全部归于外框。", "",
              f"Acros {len(cand['acros'])} 帧、TIFF {len(cand['tiff'])} 帧；设备失败项：{devices['failures']}。"]
    tests = read(OUT / "test_evidence.json")
    assert tests["source_sha256"] == fingerprint
    for p, digest in tests["test_sha256"].items():
        assert sha(ROOT / p) == digest, p
    baseline_failures = read(OUT / "baseline_failure_evidence.json")
    assert baseline_failures["failed"] >= 3
    assert baseline_failures["baseline_source_sha256"] == result["baseline"]["src_content_sha256"]
    for p, digest in baseline_failures["test_sha256"].items():
        assert sha(ROOT / p) == digest, p
    result["baseline_test_failures"] = baseline_failures
    result["tests"] = tests
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
        all_metrics_params_exact=all(e["identity"][v]["identically_equal"] for e in result["evaluations"].values() for v in ("t4t", "t4s", "t4r")),
        all_preset_worse0=all(s["worse_025"] == s["worse_005"] == 0 for e in result["evaluations"].values() for s in e["presets_vs_t4o"].values()),
        inversion_diagnostics_exact=all(not e[f"inversion_identity_vs_{v}"]["differences"] for e in result["evaluations"].values() for v in ("t4t", "t4s", "t4r")),
        protected_frames_exact=all(result["protected_frames"].values()), devices_all_passed=devices["all_passed"],
        tests_passed_ge_409=tests["passed"] >= 409, configs_all_match=all(c["matches"] for c in configs.values()),
        service_ok=service["status"] == "ok" and service["http_config_rev"] == NOREF_V3["rev"] and service["source_thumb_rev"] == "t4u",
        resources_complete=all(k in resources for k in required))
    result["all_passed"] = all(result["gates"].values())
    artifact_sha.update({str(p.relative_to(ROOT)): sha(p) for p in OUT.glob("devices_*.json")})
    result["artifact_sha256"] = artifact_sha
    lines += ["", f"完整 pytest：{tests['passed']} passed，{tests['seconds']} 秒。", "", f"验收 gates：`{result['gates']}`", "", f"**all_passed={result['all_passed']}**", "",
              "图像对照仅用于检查伪影与算法行为；没有色卡真值，status=ok 不等于颜色正确。80 张研究盲测未提供可读取的完整逐张清单，本轮只声明指定样本的独立复验。"]
    (OUT / "TABLES.md").write_text("\n".join(lines) + "\n")
    (OUT / "resource_evidence.json").write_text(json.dumps(resources, indent=2, sort_keys=True) + "\n")
    (OUT / "acceptance.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    scripts = sorted((ROOT / "scripts").glob("t4u_*.py")) + [ROOT / "scripts/t4d_source_eval.py", ROOT / "scripts/t4s_report_tables.py"]
    (OUT / "script_sha256.json").write_text(json.dumps({str(p.relative_to(ROOT)): sha(p) for p in scripts}, indent=2) + "\n")
    print(json.dumps(dict(gates=result["gates"], all_passed=result["all_passed"]), indent=2))


if __name__ == "__main__":
    main()
