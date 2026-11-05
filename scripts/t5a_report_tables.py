"""Rebuild T5a acceptance from measured artifacts; no original images are decoded.
Usage: python -I scripts/t5a_report_tables.py

T5a changes no render parameter, so the gates are exact identity against T4z (acee65d): every benchmark record
(Dev 185 x 5 presets, Check 123 x standard; params, metrics and inversion diagnostics; only the wall-clock timing
fields ``seconds`` / ``estimate_seconds`` and the new ``input_warnings`` / ``input_center_p99`` are left out) and every
device record (params / inversion / status / flags / report without the two new keys); display render hashes may
change only for B096. Plus: no benchmark frame carries an input warning, the §4 hit lists, the new gate 4 (devices with
min(core q50) < 0.03 or channel q50 max / min > 6 carry a warning), and the §3 routing changes.
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t5a"
sys.path.insert(0, str(ROOT / "scripts"))
from t4s_report_tables import valid  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from filmcolor.app import ROLL_BASE_MAX, THUMB_REV  # noqa: E402
from filmcolor.noref import NOREF_V3, STYLE_PRESETS, preset_config  # noqa: E402
from filmcolor.preview import INPUT_DHI_G_MAX, INPUT_EMPTY_CHANNEL  # noqa: E402

PRESETS = tuple(STYLE_PRESETS)
T4Z, T4O = "acee65d", "4cd76b7"
PATHS_SHA = "fced78f38a0787e473d882024d289878b99a224c604242f6561c7fb05e8be913"
EXPECTED_ERRORS = {f"ext:{n}" for n in ("A016", "A018", "A025", "A026", "A027", "A028")}
CHECK_EXCLUSION = "blueneg:199807A-21-orlando-disney"
PROTECTED = ("ext:A008", "ext:A012", "ext:A013", "shop:DSC_3833", "nlp:_DSF8948")
TIMING = {"seconds", "estimate_seconds"}
NEW_KEYS = {"input_warnings", "input_center_p99"}
GROUPS = ("targets", "new_targets", "acros", "tiff", "new_tiff", "special")
EXPECTED_DEVICE_HITS = {"A850.tif": ["input_density_implausible"], "A229.TIF": ["input_channel_empty"]}
EXPECTED_BATCH_HITS = {"A851.tif": ["input_density_implausible"], "A852.tif": ["input_density_implausible"],
                       "B094.tif": ["input_density_implausible"]}
RESEARCH = Path("/Users/davidliu/Downloads/fresh_batches_t4y.json")
TASK = Path("/Users/davidliu/Downloads/FilmColor_AI_任务书_T5a_主干快进合入与片夹涂灰避让过曝窗景及卷片基越界保护与输入可信度提示_2026-11-05.md")
AUDIT = Path("/Users/davidliu/Downloads/t4z_audit_t5a_checks.json")


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_sha():
    h = hashlib.sha256()
    for p in sorted(p for p in (ROOT / "src/filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(ROOT / "src")).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def committed(commit, rel):
    data = subprocess.check_output(["git", "show", f"{commit}:{rel}"], cwd=ROOT)
    assert data == (ROOT / rel).read_bytes(), f"historical artifact modified: {rel}"
    return json.loads(data)


def strip_bench(v):
    v = {k: x for k, x in v.items() if k not in TIMING | NEW_KEYS}
    if "variants" in v:
        v["variants"] = {n: {k: x for k, x in var.items() if k not in TIMING} for n, var in v["variants"].items()}
    return v


def strip_device(r):
    r = {k: x for k, x in r.items() if k not in NEW_KEYS | {"display_sha256", "preview_image", "preview_image_sha256",
                                                            "srgb_q50", "core_srgb_q50"}}
    if "report" in r:
        r["report"] = {k: x for k, x in r["report"].items() if k not in NEW_KEYS}
    return r


def codes(ws):
    return [w["code"] for w in ws or []]


def de_summary(ids, before, after, preset):
    u = np.array([before[k]["variants"][preset]["metrics"]["de2000_median"] for k in ids], float)
    v = np.array([after[k]["variants"][preset]["metrics"]["de2000_median"] for k in ids], float)
    d = v - u
    return dict(better_005=int((d < -.05).sum()), worse_005=int((d > .05).sum()), better_025=int((d < -.25).sum()),
                worse_025=int((d > .25).sum()), max_abs_delta=float(np.abs(d).max()),
                stats=" / ".join(f"{x:.2f}" for x in (np.median(v), np.mean(v), np.percentile(v, 90))))


def device_records(d):
    out = {"A1296.CR2": d["a1296"]}
    for g in GROUPS:
        for k, r in d[g].items():
            assert k not in out, k
            out[k] = r
    return out


def main():
    fingerprint = source_sha()
    freeze = read(OUT / "source_freeze.json")
    assert fingerprint == freeze["src_content_sha256"], "source changed after the freeze"
    assert sha(ROOT / "src/filmcolor/paths.py") == PATHS_SHA == freeze["paths_sha256"]
    assert sha(ROOT / "scripts/t4d_source_eval.py") == freeze["eval_script_sha256"]
    assert THUMB_REV == "t5a" and NOREF_V3["rev"] == "t4z-2026-11-05"
    assert ROLL_BASE_MAX == 1.25 and INPUT_DHI_G_MAX == 2.40 and INPUT_EMPTY_CHANNEL == (0.02, 0.30)
    baseline = read(OUT / "baseline_source.json")
    assert baseline["commit"].startswith(T4Z) and baseline["paths_runtime_sha256"] == PATHS_SHA
    t4z_protocol = read(ROOT / "experiments/t4_auto_ui/t4z/input_protocol_sha256.json")
    for rel, digest in t4z_protocol["historical_json_sha256"].items():
        assert sha(ROOT / rel) == digest, rel
    for rel, digest in t4z_protocol["frozen_manifest_sha256"].items():
        assert sha(ROOT / rel) == digest, rel
    result = dict(source_sha256=fingerprint, baseline=baseline, gates={}, evaluations={},
                  inputs=dict(task_sha256=sha(TASK), audit_sha256=sha(AUDIT), research_sha256=sha(RESEARCH)))
    lines = ["# T5a 可重算验收表", "", f"候选源码 SHA256：`{fingerprint}`（含用户原有 paths.py 改动）；基线 T4z `{baseline['src_content_sha256']}`。", "",
             "T5a 不改渲染参数，验收按逐位一致：基准每条记录除耗时字段（seconds / estimate_seconds）与新增 input_warnings / input_center_p99 外完全相同。", "",
             "| 集合 | 预设 | N | median / mean / p90 | 对T4z 好/差 >.05 | 对T4z 最大 \\|ΔΔE\\| | 对T4o 好/差 >.05 | 对T4o 好/差 >.25 |",
             "|---|---|---:|---|---|---:|---|---|"]

    # 1-2. benchmark identity, protected frames, zero benchmark hints
    bench_identical, worse0_t4o, no_hint, inv_identical = True, True, True, True
    collections = {}
    for part in ("a", "b", "cross"):
        rel = f"experiments/t4_auto_ui/t4z/candidate_{part}.json"
        z = committed(T4Z, rel)
        o = committed(T4O, f"experiments/t4_auto_ui/t4o/candidate_{part}.json")
        c = read(OUT / f"candidate_{part}.json")
        assert c["src_content_sha256"] == fingerprint and z["src_content_sha256"] == baseline["src_content_sha256"]
        for field in ("manifest", "max_side", "ncc_min", "app_config", "roll_folder", "base_mode", "variants"):
            assert c.get(field) == z.get(field), (part, field)
        assert set(c["frames"]) == set(z["frames"]) == set(o["frames"])
        diffs = [k for k in c["frames"] if strip_bench(c["frames"][k]) != strip_bench(z["frames"][k])]
        inv = [(k, f) for k, v in c["frames"].items() if "error" not in v
               for f in ("d_hi", "white_point", "spike_clamp", "base_method") if v.get(f) != z["frames"][k].get(f)]
        hinted = [k for k, v in c["frames"].items() if v.get("input_warnings")]
        unset = [k for k, v in c["frames"].items() if "error" not in v and v.get("input_warnings") != []]
        bench_identical &= not diffs
        inv_identical &= not inv
        no_hint &= not hinted and not unset
        result["evaluations"][part] = dict(records=len(c["frames"]), non_timing_diffs=diffs, inversion_diffs=inv,
                                           hinted=hinted, warnings_field_missing=unset, sha256=sha(OUT / f"candidate_{part}.json"))
        label = "cross" if part == "cross" else "dev"
        for name, rows in (("t5a", c), ("t4z", z), ("t4o", o)):
            collections.setdefault(label, {}).setdefault(name, {}).update(rows["frames"])
    counts = []
    for label, n, presets in (("dev", 185, PRESETS), ("cross", 123, ("standard",))):
        cand = collections[label]["t5a"]
        if label == "dev":
            assert set(cand) == {f["id"] for f in read(ROOT / "data/manifests/s3_dev_frames.json")["frames"]}
            assert {k for k, v in cand.items() if "error" in v} == EXPECTED_ERRORS
        else:
            assert set(cand) == {f["id"] for f in read(ROOT / "data/manifests/bn_rb_frames.json")["frames"] if not f["s3_5"]}
            assert set(cand) - valid(cand) == {CHECK_EXCLUSION}
        ids = sorted(valid(cand))
        assert len(ids) == n and valid(collections[label]["t4z"]) == set(ids) == valid(collections[label]["t4o"])
        counts.append(len([k for k, v in cand.items() if "error" not in v]))
        for k in ids:
            for field in ("raw_sha256", "ref_sha256"):
                assert cand[k][field] == collections[label]["t4z"][k][field] == collections[label]["t4o"][k][field]
        for preset in presets:
            sz = de_summary(ids, collections[label]["t4z"], cand, preset)
            so = de_summary(ids, collections[label]["t4o"], cand, preset)
            bench_identical &= sz["better_005"] == sz["worse_005"] == 0 and sz["max_abs_delta"] == 0.0
            worse0_t4o &= so["worse_005"] == 0 and so["worse_025"] == 0
            result["evaluations"][f"{label}_{preset}"] = dict(vs_t4z=sz, vs_t4o=so)
            lines.append(f"| {label} | {preset} | {n} | {sz['stats']} | {sz['better_005']}/{sz['worse_005']} | "
                         f"{sz['max_abs_delta']} | {so['better_005']}/{so['worse_005']} | {so['better_025']}/{so['worse_025']} |")
    assert sum(counts) == 309
    dev = collections["dev"]
    result["protected_frames"] = {k: all(dev["t5a"][k]["variants"][p][f] == dev["t4z"][k]["variants"][p][f]
                                         == dev["t4o"][k]["variants"][p][f] for p in PRESETS for f in ("params", "metrics"))
                                  for k in PROTECTED}

    # 3-4. devices
    base, cand = read(OUT / "devices_baseline.json"), read(OUT / "devices_candidate.json")
    assert base["src_content_sha256"] == baseline["src_content_sha256"] and cand["src_content_sha256"] == fingerprint
    assert base["script_sha256"] == cand["script_sha256"] == sha(ROOT / "scripts/t5a_probe_devices.py")
    t4z_dev = committed(T4Z, "experiments/t4_auto_ui/t4z/devices_candidate.json")
    rb, rc, rz = device_records(base), device_records(cand), device_records(t4z_dev)
    assert set(rb) == set(rc) == set(rz)
    dev_rows, dev_ok = {}, True
    for k in sorted(rc):
        b, c, z = rb[k], rc[k], rz[k]
        assert "error" not in b and "error" not in c, k
        same_base_as_t4z = all(b.get(f) == z.get(f) for f in ("params", "inversion", "status", "flags", "report",
                                                              "core_srgb_q50", "raw_sha256"))
        same = strip_device(b) == strip_device(c)
        display_changed = b["display_sha256"] != c["display_sha256"]
        image_changed = b.get("preview_image_sha256") != c.get("preview_image_sha256")
        q = c["core_srgb_q50"]
        gate4_needs = min(q) < 0.03 or max(q) / max(min(q), 1e-9) > 6
        row = dict(identical_except_display=same, baseline_reproduces_t4z=same_base_as_t4z,
                   display_changed=display_changed, preview_image_changed=image_changed,
                   input_warnings=codes(c["input_warnings"]), baseline_has_field=(b.get("input_warnings") is not None),  # T4z: None
                   core_srgb_q50=q, gate4_needs_warning=gate4_needs, status=c["status"])
        expected_display_change = k == "B096.tif"
        row["ok"] = (same and same_base_as_t4z and display_changed == expected_display_change
                     and image_changed == (expected_display_change and "preview_image_sha256" in c)
                     and row["input_warnings"] == EXPECTED_DEVICE_HITS.get(k, [])
                     and (not gate4_needs or bool(row["input_warnings"])) and not row["baseline_has_field"])
        dev_ok &= row["ok"]
        dev_rows[k] = row
    scope, scope_z = read(OUT / "a1296_scope.json"), committed(T4Z, "experiments/t4_auto_ui/t4z/a1296_scope.json")
    scope_ok = (scope["source_sha256"] == fingerprint and scope["raw_sha256"] == scope_z["raw_sha256"]
                == rc["A1296.CR2"]["raw_sha256"] and scope["regions"] == scope_z["regions"]
                and scope["full_shape"] == scope_z["full_shape"])
    b096 = rc["B096.tif"]
    result["devices"] = dict(records=len(rc), rows=dev_rows, failures=[k for k, v in dev_rows.items() if not v["ok"]],
                             gate4_frames=sorted(k for k, v in dev_rows.items() if v["gate4_needs_warning"]),
                             hits={k: v["input_warnings"] for k, v in dev_rows.items() if v["input_warnings"]},
                             b096_core_q50=dict(t4z=rb["B096.tif"]["core_srgb_q50"], t5a=b096["core_srgb_q50"]),
                             a1296_scope_identical=scope_ok)

    # §4 batch hit list
    research = read(RESEARCH)
    rows = [r for rs in research["batches"].values() for r in rs]
    bb, bc = read(OUT / "batches_baseline.json"), read(OUT / "batches_candidate.json")
    assert bb["src_content_sha256"] == baseline["src_content_sha256"] and bc["src_content_sha256"] == fingerprint
    assert bb["script_sha256"] == bc["script_sha256"] == sha(ROOT / "scripts/t5a_batch_hits.py")
    assert bb["research_sha256"] == bc["research_sha256"] == sha(RESEARCH)
    assert set(bb["frames"]) == set(bc["frames"]) == {r["rel"] for r in rows} and len(rows) == 41
    batch_rows, batch_ok = {}, True
    for rel in sorted(bc["frames"]):
        b, c = bb["frames"][rel], bc["frames"][rel]
        assert "error" not in b and "error" not in c, rel
        assert b["raw_sha256"] == c["raw_sha256"]
        same = all(b[f] == c[f] for f in ("status", "flags", "params", "inversion", "base_method"))
        disp = b["display_sha256"] != c["display_sha256"]
        row = dict(name=c["name"], batch=c["batch"], status=c["status"], identical=same, display_changed=disp,
                   input_warnings=codes(c["input_warnings"]), d_hi_g=c["inversion"]["d_hi"][1],
                   input_center_p99=c["input_center_p99"])
        row["ok"] = same and disp == (c["name"] == "B096.tif") and row["input_warnings"] == EXPECTED_BATCH_HITS.get(c["name"], [])
        batch_ok &= row["ok"]
        batch_rows[rel] = row
    result["batches"] = dict(frames=len(batch_rows), rows=batch_rows,
                             hits={v["name"]: v["input_warnings"] for v in batch_rows.values() if v["input_warnings"]},
                             failures=[v["name"] for v in batch_rows.values() if not v["ok"]],
                             baseline_sha256=sha(OUT / "batches_baseline.json"),
                             candidate_sha256=sha(OUT / "batches_candidate.json"))

    # §3 routing and roll hit list
    routing = read(OUT / "roll_routing_changes.json")
    assert routing["source_sha256"] == fingerprint and routing["baseline_source_sha256"] == baseline["src_content_sha256"]
    assert routing["rolls_baseline_sha256"] == sha(OUT / "rolls_baseline.json")
    assert routing["rolls_candidate_sha256"] == sha(OUT / "rolls_candidate.json")
    for name, m in routing["mosaics"].items():
        assert sha(OUT / m["file"]) == m["sha256"], name
    rolls_c = read(OUT / "rolls_candidate.json")
    roll_hits = {f"{n}/{f}": codes(x.get("input_warnings")) for n, r in rolls_c["rolls"].items() if r["opened"]
                 for f, x in r["frames"].items() if x.get("input_warnings")}
    foma, eastman = routing["rolls"]["foma_pan100"], routing["rolls"]["eastman_5294"]
    routing_ok = (len(foma["moved_to_auto"]) >= 24 and len(eastman["moved_to_auto"]) >= 4
                  and set(foma["moved_to_auto"]) == set(foma["roll_gt_limit"]) & set(foma["routed_before"])
                  and set(eastman["moved_to_auto"]) == set(eastman["roll_gt_limit"]) & set(eastman["routed_before"])
                  and all(not routing["rolls"][n]["moved_to_auto"] for n in ("eastman_5294-1", "eastman_5294-2"))
                  and all(x["status_after"] == "ok" for r in (foma, eastman) for x in r["frames"].values()))
    result["rolls"] = dict(routing_sha256=sha(OUT / "roll_routing_changes.json"),
                           moved_to_auto={n: r["moved_to_auto"] for n, r in routing["rolls"].items()},
                           routed=dict((n, [len(r["routed_before"]), len(r["routed_after"])]) for n, r in routing["rolls"].items()),
                           roll_gt_limit={n: len(r["roll_gt_limit"]) for n, r in routing["rolls"].items()},
                           input_warning_hits=roll_hits, mosaics=routing["mosaics"])

    # 5. tests, configs, service, resources
    tests = read(OUT / "test_evidence.json")
    assert tests["source_sha256"] == fingerprint
    for p, digest in tests["test_sha256"].items():
        assert sha(ROOT / p) == digest, p
    red = read(OUT / "baseline_failure_evidence.json")
    assert red["baseline_source_sha256"] == baseline["src_content_sha256"] and red["failed"] == 7 and red["passed"] == 0
    for p, digest in red["test_sha256"].items():
        assert sha(ROOT / p) == digest, p
    configs = {}
    for name in PRESETS:
        p = ROOT / "configs" / ("noref_v3.json" if name == "standard" else f"noref_v3_{name}.json")
        d = read(p)
        configs[name] = dict(sha256=sha(p), unchanged_since_t4z=(p.read_bytes() == subprocess.check_output(
            ["git", "show", f"{T4Z}:{p.relative_to(ROOT)}"], cwd=ROOT)),
            matches=(d == json.loads(json.dumps(dict(source_config_rev=NOREF_V3["rev"], preset=name, config=preset_config(name))))))
    service = read(OUT / "service_check.json")
    assert service["source_sha256"] == fingerprint
    resources = {}
    for p in OUT.glob("*.log"):
        m = re.findall(r"(\d+)\s+peak memory footprint", p.read_text(errors="replace"))
        if m:
            resources[p.stem] = dict(peak_memory_footprint_bytes=max(map(int, m)), log_sha256=sha(p))
    required = ("candidate_a", "candidate_b", "candidate_cross", "devices_baseline", "devices_candidate", "tests",
                "rolls_baseline", "rolls_candidate", "batches_baseline", "batches_candidate")
    result.update(tests=tests, baseline_test_failures=red, configs=configs, service=service)

    result["gates"] = dict(
        benchmark_identical_vs_t4z=bench_identical,
        benchmark_worse0_vs_t4o=worse0_t4o,
        inversion_diagnostics_identical=inv_identical,
        protected_frames_identical=all(result["protected_frames"].values()),
        benchmark_no_input_warning=no_hint,
        devices_identical_display_only_b096_hits_expected_gate4=dev_ok and scope_ok,
        batches_hits_expected=batch_ok,
        roll_routing_changes_present=routing_ok,
        tests_passed_ge_441=tests["passed"] >= 441,
        configs_unchanged=all(c["matches"] and c["unchanged_since_t4z"] for c in configs.values()),
        service_ok=(service["status"] == "ok" and service["source_thumb_rev"] == "t5a"
                    and service["http_config_rev"] == NOREF_V3["rev"] and service["input_warnings"] == []),
        paths_unchanged=sha(ROOT / "src/filmcolor/paths.py") == PATHS_SHA,
        resources_complete=all(k in resources for k in required),
    )
    result["all_passed"] = all(result["gates"].values())

    d = result["devices"]
    lines += ["", f"309 条非错误记录反演诊断差异：{sum(len(result['evaluations'][p]['inversion_diffs']) for p in ('a', 'b', 'cross'))}；"
              f"基准 input_warnings 命中：{sum(len(result['evaluations'][p]['hinted']) for p in ('a', 'b', 'cross'))}。",
              "", "## 设备（T4z 设备集）", "",
              f"{d['records']} 条记录；除显示外逐位一致且基线重现 T4z 提交记录：{sum(v['identical_except_display'] and v['baseline_reproduces_t4z'] for v in d['rows'].values())}；"
              f"显示渲染哈希变化：{[k for k, v in d['rows'].items() if v['display_changed']]}；失败项：{d['failures']}。", "",
              "| 样本 | 状态 | core sRGB q50 | 门 4 需提示 | input_warnings |", "|---|---|---|---|---|"]
    for k, v in d["rows"].items():
        if v["input_warnings"] or v["gate4_needs_warning"] or v["display_changed"]:
            lines.append(f"| {k} | {v['status']} | {[round(x, 3) for x in v['core_srgb_q50']]} | {v['gate4_needs_warning']} | {', '.join(v['input_warnings']) or '-'} |")
    lines += ["", f"B096 core q50：T4z {[round(x, 3) for x in d['b096_core_q50']['t4z']]} → T5a {[round(x, 3) for x in d['b096_core_q50']['t5a']]}；A1296 全分辨率作用域统计与 T4z 一致：{d['a1296_scope_identical']}。",
              "", "## Batch 19–21（41 帧）", "", f"命中：{result['batches']['hits']}；失败项：{result['batches']['failures']}。", "",
              "## §3 整卷片基越界保护", "", "| 卷 | 帧 | 卷片基 > 1.25 | 默认 roll_nat 前 → 后 | 改走单帧 |", "|---|---:|---:|---|---:|"]
    for n, r in routing["rolls"].items():
        lines.append(f"| {n} | {r['n']} | {len(r['roll_gt_limit'])} | {len(r['routed_before'])} → {len(r['routed_after'])} | {len(r['moved_to_auto'])} |")
    lines += ["", f"改走单帧清单：foma {foma['moved_to_auto']}；Eastman（无后缀）{eastman['moved_to_auto']}。",
              "", f"两卷默认路由下 input_warnings 命中：{roll_hits}。", "",
              f"完整 pytest：{tests['passed']} passed，{tests['seconds']} 秒；新增/改动测试在 T4z 源码上 {red['failed']} 项失败。", "",
              f"验收 gates：`{result['gates']}`", "", f"**all_passed={result['all_passed']}**", "",
              "提示只说明输入可能不是可用的负片扫描，不改变 status、参数或反相；status=ok 与无提示都不等于颜色正确。"]
    (OUT / "TABLES.md").write_text("\n".join(lines) + "\n")
    (OUT / "resource_evidence.json").write_text(json.dumps(resources, indent=2, sort_keys=True) + "\n")
    (OUT / "acceptance.json").write_text(json.dumps(result, indent=2, allow_nan=False, ensure_ascii=False) + "\n")
    scripts = sorted((ROOT / "scripts").glob("t5a_*.py")) + [ROOT / "scripts/t4d_source_eval.py", ROOT / "scripts/t4s_report_tables.py"]
    (OUT / "script_sha256.json").write_text(json.dumps({str(p.relative_to(ROOT)): sha(p) for p in scripts}, indent=2) + "\n")
    print(json.dumps(dict(gates=result["gates"], all_passed=result["all_passed"]), indent=2))


if __name__ == "__main__":
    main()
