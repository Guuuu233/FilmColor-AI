"""Rebuild T5b acceptance from measured artifacts; no original images are decoded.
Usage: python -I scripts/t5b_report_tables.py

T5b changes the camera-RAW base only when near-saturated blocks were excluded and the base still has a channel
>= CLIPPED_BASE_MAX; no benchmark frame or device record is in that case, so the gates are exact identity against
committed T5a (1c4c9aa): every benchmark record (Dev 185 x 5 presets, Check 123 x standard; only the wall-clock
timing fields ``seconds`` / ``estimate_seconds`` are left out, and no ``clipped_base_excluded`` may appear) and every
device record (params / inversion / status / flags / report / display and preview hashes, baseline = isolated T5a
tree, which must also reproduce the committed T5a records). Plus the Sony A7R4 before / after, the blind camera-RAW
base scan (only A1298 / A1299 / A1300 may change), the material polarity list, tests and the service check.
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t5b"
sys.path.insert(0, str(ROOT / "scripts"))
from t4s_report_tables import valid  # noqa: E402
sys.path.insert(0, str(ROOT / "src"))
from filmcolor.app import ROLL_BASE_MAX, THUMB_REV  # noqa: E402
from filmcolor.base import CLIPPED_BASE_MAX  # noqa: E402
from filmcolor.noref import NOREF_V3, STYLE_PRESETS, preset_config  # noqa: E402

PRESETS = tuple(STYLE_PRESETS)
T5A = "1c4c9aa"
PATHS_SHA = "fced78f38a0787e473d882024d289878b99a224c604242f6561c7fb05e8be913"
EXPECTED_ERRORS = {f"ext:{n}" for n in ("A016", "A018", "A025", "A026", "A027", "A028")}
CHECK_EXCLUSION = "blueneg:199807A-21-orlando-disney"
PROTECTED = ("ext:A008", "ext:A012", "ext:A013", "shop:DSC_3833", "nlp:_DSF8948")
TIMING = {"seconds", "estimate_seconds"}
GROUPS = ("targets", "new_targets", "acros", "tiff", "new_tiff", "special")
SONY = ("A1298.ARW", "A1299.ARW", "A1300.ARW")
EXPECTED_CHANGED = tuple(f"01_彩色负片_未反相/{n}" for n in SONY)
TASK = Path("/Users/davidliu/Downloads/FilmColor_AI_任务书_T5b_主干快进合入与相机RAW削顶片基剔除及素材极性清单_2026-11-05.md")
AUDIT = Path("/Users/davidliu/Downloads/FilmColor_AI_T5a_审计_2026-11-05-2.md")
RESEARCH_BLIND = Path("/Users/davidliu/Downloads/blind_raw_base_t5a.json")


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
    v = {k: x for k, x in v.items() if k not in TIMING}
    if "variants" in v:
        v["variants"] = {n: {k: x for k, x in var.items() if k not in TIMING} for n, var in v["variants"].items()}
    return v


def de_summary(ids, before, after, preset):
    u = np.array([before[k]["variants"][preset]["metrics"]["de2000_median"] for k in ids], float)
    v = np.array([after[k]["variants"][preset]["metrics"]["de2000_median"] for k in ids], float)
    d = v - u
    return dict(better_005=int((d < -.05).sum()), worse_005=int((d > .05).sum()), better_025=int((d < -.25).sum()),
                worse_025=int((d > .25).sum()), max_abs_delta=float(np.abs(d).max()),
                stats=" / ".join(f"{x:.2f}" for x in (np.median(v), np.mean(v), np.percentile(v, 90))))


def collections_method(k, collections):
    for label in ("dev", "cross"):
        if k in collections.get(label, {}).get("t5b", {}):
            return collections[label]["t5b"][k]["base_method"]
    raise KeyError(k)


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
    assert THUMB_REV == "t5b" and NOREF_V3["rev"] == "t4z-2026-11-05" and ROLL_BASE_MAX == 1.25 and CLIPPED_BASE_MAX == 0.99
    baseline = read(OUT / "baseline_source.json")
    assert baseline["commit"].startswith(T5A) and baseline["paths_runtime_sha256"] == PATHS_SHA
    t5a_freeze = committed(T5A, "experiments/t4_auto_ui/t5a/source_freeze.json")
    assert baseline["src_content_sha256"] == t5a_freeze["src_content_sha256"]
    result = dict(source_sha256=fingerprint, baseline=baseline, gates={}, evaluations={},
                  inputs=dict(task_sha256=sha(TASK), audit_sha256=sha(AUDIT), research_blind_sha256=sha(RESEARCH_BLIND)))
    lines = ["# T5b 可重算验收表", "",
             f"候选源码 SHA256：`{fingerprint}`（含用户原有 paths.py 改动）；基线 T5a `{baseline['src_content_sha256']}`。", "",
             "T5b 的规则在基准和设备帧上都不触发，验收按逐位一致：基准每条记录除耗时字段（seconds / estimate_seconds）外与 T5a 提交记录完全相同。", "",
             "| 集合 | 预设 | N | median / mean / p90 | 对T5a 好/差 >.05 | 对T5a 最大 \\|ΔΔE\\| |",
             "|---|---|---:|---|---|---:|"]

    # 1. benchmark identity vs committed T5a, protected frames, no clipped_base_excluded
    bench_identical, inv_identical, no_clipped = True, True, True
    collections = {}
    for part in ("a", "b", "cross"):
        z = committed(T5A, f"experiments/t4_auto_ui/t5a/candidate_{part}.json")
        c = read(OUT / f"candidate_{part}.json")
        assert c["src_content_sha256"] == fingerprint and z["src_content_sha256"] == baseline["src_content_sha256"]
        for field in ("manifest", "max_side", "ncc_min", "app_config", "roll_folder", "base_mode", "variants"):
            assert c.get(field) == z.get(field), (part, field)
        assert set(c["frames"]) == set(z["frames"])
        diffs = [k for k in c["frames"] if strip_bench(c["frames"][k]) != strip_bench(z["frames"][k])]
        inv = [(k, f) for k, v in c["frames"].items() if "error" not in v
               for f in ("d_hi", "white_point", "spike_clamp", "base_method", "base", "base_diagnostics")
               if v.get(f) != z["frames"][k].get(f)]
        clipped = [k for k, v in c["frames"].items() if (v.get("base_diagnostics") or {}).get("clipped_base_excluded")]
        bench_identical &= not diffs
        inv_identical &= not inv
        no_clipped &= not clipped
        result["evaluations"][part] = dict(records=len(c["frames"]), non_timing_diffs=diffs, inversion_diffs=inv,
                                           clipped_base_excluded=clipped, sha256=sha(OUT / f"candidate_{part}.json"))
        label = "cross" if part == "cross" else "dev"
        for name, rows in (("t5b", c), ("t5a", z)):
            collections.setdefault(label, {}).setdefault(name, {}).update(rows["frames"])
    counts = []
    bench_base_max, bench_own_max = {}, {}
    for label, n, presets in (("dev", 185, PRESETS), ("cross", 123, ("standard",))):
        cand = collections[label]["t5b"]
        if label == "dev":
            assert set(cand) == {f["id"] for f in read(ROOT / "data/manifests/s3_dev_frames.json")["frames"]}
            assert {k for k, v in cand.items() if "error" in v} == EXPECTED_ERRORS
        else:
            assert set(cand) == {f["id"] for f in read(ROOT / "data/manifests/bn_rb_frames.json")["frames"] if not f["s3_5"]}
            assert set(cand) - valid(cand) == {CHECK_EXCLUSION}
        ids = sorted(valid(cand))
        assert len(ids) == n and valid(collections[label]["t5a"]) == set(ids)
        counts.append(len([k for k, v in cand.items() if "error" not in v]))
        for k in ids:
            for field in ("raw_sha256", "ref_sha256"):
                assert cand[k][field] == collections[label]["t5a"][k][field]
        for k, v in cand.items():
            if "error" not in v:
                bench_base_max[k] = max(v["base"]["rgb"])  # base used for rendering (roll_nat on routed frames)
                bench_own_max[k] = max(v["base"]["own_rgb"])  # the frame's own block estimate: what the T5b rule tests
        for preset in presets:
            sz = de_summary(ids, collections[label]["t5a"], cand, preset)
            bench_identical &= sz["better_005"] == sz["worse_005"] == 0 and sz["max_abs_delta"] == 0.0
            result["evaluations"][f"{label}_{preset}"] = dict(vs_t5a=sz)
            lines.append(f"| {label} | {preset} | {n} | {sz['stats']} | {sz['better_005']}/{sz['worse_005']} | {sz['max_abs_delta']} |")
    assert sum(counts) == 309
    top, top_own = max(bench_base_max, key=bench_base_max.get), max(bench_own_max, key=bench_own_max.get)
    result["benchmark_base_max_channel"] = dict(own_estimate=dict(frame=top_own, value=bench_own_max[top_own]),
                                                render_base=dict(frame=top, value=bench_base_max[top],
                                                                 method=collections_method(top, collections)))
    dev = collections["dev"]
    result["protected_frames"] = {k: all(dev["t5b"][k]["variants"][p][f] == dev["t5a"][k]["variants"][p][f]
                                         for p in PRESETS for f in ("params", "metrics")) for k in PROTECTED}

    # 2. devices: candidate vs isolated T5a baseline, and the baseline vs committed T5a records
    base, cand = read(OUT / "devices_baseline.json"), read(OUT / "devices_candidate.json")
    assert base["src_content_sha256"] == baseline["src_content_sha256"] and cand["src_content_sha256"] == fingerprint
    assert base["script_sha256"] == cand["script_sha256"] == sha(ROOT / "scripts/t5b_probe_devices.py")
    t5a_dev = committed(T5A, "experiments/t4_auto_ui/t5a/devices_candidate.json")
    rb, rc, rz = device_records(base), device_records(cand), device_records(t5a_dev)
    assert set(rb) == set(rc) == set(rz) and len(rc) == 61
    dev_rows, dev_ok = {}, True
    fields = ("params", "inversion", "status", "flags", "report", "base", "diagnostics", "display_sha256",
              "core_srgb_q50", "input_warnings", "raw_sha256", "preview_image_sha256", "base_diagnostics")
    for k in sorted(rc):
        b, c, z = rb[k], rc[k], rz[k]
        assert "error" not in b and "error" not in c, k
        reproduces = all(b.get(f) == z.get(f) for f in fields)
        same = all(b.get(f) == c.get(f) for f in fields)
        clipped = bool((c.get("base_diagnostics") or {}).get("clipped_base_excluded"))
        row = dict(identical=same, baseline_reproduces_t5a=reproduces, clipped_base_excluded=clipped,
                   status=c["status"], base_max_channel=max(c["base"]["rgb"]),  # render base
                   own_base_max_channel=max(c["base"]["own_rgb"]),  # the frame's own estimate: what the rule tests
                   near_sat=(c.get("base_diagnostics") or {}).get("near_saturated_blocks_excluded"),
                   has_preview_image=c.get("preview_image_sha256") is not None)
        row["ok"] = (same and reproduces and not clipped
                     and (not row["near_sat"] or row["own_base_max_channel"] < CLIPPED_BASE_MAX))
        dev_ok &= row["ok"]
        dev_rows[k] = row
    result["devices"] = dict(records=len(rc), rows=dev_rows, failures=[k for k, v in dev_rows.items() if not v["ok"]],
                             near_sat_frames={k: dict(blocks=v["near_sat"], own_base_max_channel=v["own_base_max_channel"])
                                              for k, v in dev_rows.items() if v["near_sat"]},
                             preview_image_records=sum(v["has_preview_image"] for v in dev_rows.values()),
                             baseline_sha256=sha(OUT / "devices_baseline.json"),
                             candidate_sha256=sha(OUT / "devices_candidate.json"))

    # 3. Sony A7R4 before / after
    sony = read(OUT / "sony_a7r4.json")
    assert sony["baseline_src_content_sha256"] == baseline["src_content_sha256"]
    assert sony["candidate_src_content_sha256"] == fingerprint
    assert sony["baseline_sha256"] == sha(OUT / "sony_a7r4_baseline.json")
    assert sony["candidate_sha256"] == sha(OUT / "sony_a7r4_candidate.json")
    for label in ("baseline", "candidate"):  # the probe records were written by the current script
        assert read(OUT / f"sony_a7r4_{label}.json")["script_sha256"] == sha(ROOT / "scripts/t5b_sony_a7r4.py"), label
    sony_ok = True
    for n in SONY:
        r = sony["frames"][n]
        a, b = r["t5a"], r["t5b"]
        sony_ok &= (max(a["base_rgb"]) >= CLIPPED_BASE_MAX and max(b["base_rgb"]) < 0.5
                    and b["base_diagnostics"].get("clipped_base_excluded") is True and not a["base_diagnostics"].get("clipped_base_excluded")
                    and b["base_RG_BG"][1] < 0.43 and b["d_hi_g"] < a["d_hi_g"] and b["status"] == a["status"] == "ok")

    # 4. blind camera-RAW base scan (research's 235-file T5a dump cross-checked)
    scan = read(OUT / "blind_raw_scan.json")
    assert scan["src_content_sha256"] == fingerprint and scan["baseline_src_content_sha256"] == baseline["src_content_sha256"]
    assert scan["script_sha256"] == sha(ROOT / "scripts/t5b_blind_raw_scan.py")
    s = scan["summary"]
    research = read(RESEARCH_BLIND)
    agree, disagree = 0, []
    assert all(rel in scan["files"] for rel in research), "research file missing from the scan"
    for rel, v in research.items():
        mine = scan["files"][rel]
        if "rgb" in v:
            if v["rgb"] == (mine.get("t5a") or {}).get("rgb"):
                agree += 1
            else:
                disagree.append(rel)
    research_with_rgb = sum("rgb" in v for v in research.values())
    files = scan["files"]
    scan_counts = dict(
        compared_scanner=sum(bool(v.get("is_scan")) for v in files.values() if "identical" in v),
        compared_camera=sum(not v.get("is_scan") for v in files.values() if "identical" in v),
        required_scanner=sum(bool(files[k].get("is_scan")) for k in scan["required"]),
        by_folder={d: dict(compared=sum(k.startswith(d + "/") and "identical" in v for k, v in files.items()),
                           scanner=sum(k.startswith(d + "/") and bool(v.get("is_scan")) for k, v in files.items()))
                   for d in ("01_彩色负片_未反相", "02_黑白负片_未反相", "A7R5_RAW")})
    scan_ok = (not s["unexpected_changes"] and not s["missing_expected"] and s["required_changed"] == list(EXPECTED_CHANGED)
               and s["required_compared"] == len(scan["required"]) and sorted(s["clipped_hits"]) == list(EXPECTED_CHANGED)
               and not disagree and agree == research_with_rgb)
    result["blind_raw_scan"] = dict(summary=s, counts=scan_counts, research_t5a_agree=agree,
                                    research_t5a_with_rgb=research_with_rgb, research_t5a_disagree=disagree,
                                    sha256=sha(OUT / "blind_raw_scan.json"))

    # 5. polarity list, tests, red-before-green, configs, service, resources
    polarity = read(ROOT / "data/manifests/material_polarity_20261105.json")
    pol_ok = (len(polarity["entries"]) == 9 and {e["kind"] for e in polarity["entries"]} == {"positive_image", "reversal_film"}
              and all({"rel", "kind", "evidence", "checked_by"} <= set(e) for e in polarity["entries"]))
    tests = read(OUT / "test_evidence.json")
    assert tests["source_sha256"] == fingerprint
    for p, digest in tests["test_sha256"].items():
        assert sha(ROOT / p) == digest, p
    red = read(OUT / "baseline_failure_evidence.json")
    assert red["baseline_source_sha256"] == baseline["src_content_sha256"] and red["failed"] == 2 and red["passed"] == 2
    for p, digest in red["test_sha256"].items():
        assert sha(ROOT / p) == digest, p
    configs = {}
    for name in PRESETS:
        p = ROOT / "configs" / ("noref_v3.json" if name == "standard" else f"noref_v3_{name}.json")
        d = read(p)
        configs[name] = dict(sha256=sha(p), unchanged_since_t5a=(p.read_bytes() == subprocess.check_output(
            ["git", "show", f"{T5A}:{p.relative_to(ROOT)}"], cwd=ROOT)),
            matches=(d == json.loads(json.dumps(dict(source_config_rev=NOREF_V3["rev"], preset=name, config=preset_config(name))))))
    service = read(OUT / "service_check.json")
    assert service["source_sha256"] == fingerprint
    resources = {}
    for p in OUT.glob("*.log"):
        m = re.findall(r"(\d+)\s+peak memory footprint", p.read_text(errors="replace"))
        if m:
            resources[p.stem] = dict(peak_memory_footprint_bytes=max(map(int, m)), log_sha256=sha(p))
    required = ("candidate_a", "candidate_b", "candidate_cross", "devices_baseline", "devices_candidate", "tests",
                "sony_baseline", "sony_candidate", "blind_raw_scan")
    result.update(sony=sony, polarity=dict(entries=len(polarity["entries"]), sha256=sha(ROOT / "data/manifests/material_polarity_20261105.json")),
                  tests=tests, baseline_test_failures=red, configs=configs, service=service)

    result["gates"] = dict(
        benchmark_identical_vs_t5a=bench_identical,
        inversion_and_base_identical=inv_identical,
        benchmark_no_clipped_base_excluded=no_clipped,
        protected_frames_identical=all(result["protected_frames"].values()),
        devices_identical_61=dev_ok,
        sony_a7r4_orange_base=sony_ok,
        blind_raw_only_a7r4_changed=scan_ok,
        material_polarity_list=pol_ok,
        tests_passed_ge_444=tests["passed"] >= 444,
        configs_unchanged=all(c["matches"] and c["unchanged_since_t5a"] for c in configs.values()),
        service_ok=(service["status"] == "ok" and service["source_thumb_rev"] == "t5b"
                    and service["http_config_rev"] == NOREF_V3["rev"]),
        paths_unchanged=sha(ROOT / "src/filmcolor/paths.py") == PATHS_SHA,
        resources_complete=all(k in resources for k in required),
    )
    result["all_passed"] = all(result["gates"].values())

    d = result["devices"]
    lines += ["", f"309 条非错误记录的反演诊断、片基与片基诊断差异："
              f"{sum(len(result['evaluations'][p]['inversion_diffs']) for p in ('a', 'b', 'cross'))}；"
              f"`clipped_base_excluded` 命中：{sum(len(result['evaluations'][p]['clipped_base_excluded']) for p in ('a', 'b', 'cross'))}。"
              f"基准帧单帧片基估计（新规则检查的量）最大通道最大 {result['benchmark_base_max_channel']['own_estimate']['value']:.3f}"
              f"（`{result['benchmark_base_max_channel']['own_estimate']['frame']}`），低于 0.99；"
              f"渲染所用片基最大 {result['benchmark_base_max_channel']['render_base']['value']:.3f}"
              f"（`{result['benchmark_base_max_channel']['render_base']['frame']}`，"
              f"{result['benchmark_base_max_channel']['render_base']['method']}，不经过本规则）。",
              "", "## 设备（T4z 设备集）", "",
              f"{d['records']} 条记录；候选与隔离 T5a 基线逐位相同、基线重现 T5a 提交记录："
              f"{sum(v['identical'] and v['baseline_reproduces_t5a'] for v in d['rows'].values())}；失败项：{d['failures']}。"
              f"带预览图哈希的记录 {d['preview_image_records']} 条（其余记录没有生成预览图，靠显示哈希比较）。"
              f"有近饱和剔除的设备帧及其单帧片基估计最大通道："
              f"{ {k: (v['blocks'], round(v['own_base_max_channel'], 3)) for k, v in d['near_sat_frames'].items()} }，"
              f"都低于 {CLIPPED_BASE_MAX}（已在门禁中检查）。", "",
              "## Sony ILCE-7RM4（A1298 / A1299 / A1300）", "",
              "| 帧 | 片基 T5a | 片基 T5b | R/G, B/G T5b | d_hi_G T5a → T5b | core q50 T5a → T5b | EV T5a → T5b |",
              "|---|---|---|---|---|---|---|"]
    for n in SONY:
        r = sony["frames"][n]
        a, b = r["t5a"], r["t5b"]
        lines.append(f"| {n} | {[round(x, 3) for x in a['base_rgb']]} | {[round(x, 3) for x in b['base_rgb']]} | "
                     f"{b['base_RG_BG']} | {a['d_hi_g']:.3f} → {b['d_hi_g']:.3f} | "
                     f"{[round(x, 3) for x in a['core_srgb_q50']]} → {[round(x, 3) for x in b['core_srgb_q50']]} | "
                     f"{a['ev']:.3f} → {b['ev']:.3f} |")
    lines += ["", "## 盲测彩负 / 黑白相机 RAW 片基（T5a 与 T5b 在同一组块上）", "",
              f"必查集（Batch 19–22 彩负目录下的 RAW / DNG + A1298 / A1300）{len(scan['required'])} 个，其中扫描仪 DNG "
              f"{scan_counts['required_scanner']} 个（Batch 19），相机 RAW {len(scan['required']) - scan_counts['required_scanner']} 个；"
              f"比较 {s['required_compared']} 个，变化：{s['required_changed']}。",
              f"扩展集（两个负片目录和 A7R5_RAW 的全部 RAW / DNG）{s['files']} 个：比较 {s['compared']}"
              f"（相机 RAW {scan_counts['compared_camera']}，扫描仪 DNG {scan_counts['compared_scanner']}，后者不适用本规则；"
              f"按目录 {scan_counts['by_folder']}），冻结未解码 "
              f"{s['frozen_not_decoded']}，解码失败 {len(s['decode_errors'])}（两版相同的读取限制）；变化 {s['changed']}；"
              f"意外变化 {s['unexpected_changes']}；`clipped_base_excluded` 命中 {s['clipped_hits']}；"
              f"T5a 片基最大通道 ≥ 0.99：{s['t5a_base_max_ge_099']}。",
              f"研究端 {len(research)} 个文件中带 T5a 片基的 {research_with_rgb} 个，与本轮 T5a 片基逐位相同 {agree} 个，"
              f"不同 {len(disagree)} 个。", "",
              f"素材极性清单：{len(polarity['entries'])} 条。完整 pytest：{tests['passed']} passed，{tests['seconds']} 秒；"
              f"新增 / 改动测试在 T5a 源码上 {red['failed']} 项失败、{red['passed']} 项通过（不触发的两项本应与 T5a 相同）。", "",
              f"验收 gates：`{result['gates']}`", "", f"**all_passed={result['all_passed']}**", "",
              "规则只修正密度范围（片基），不保证颜色正确；status=ok 不等于颜色正确。"]
    (OUT / "TABLES.md").write_text("\n".join(lines) + "\n")
    (OUT / "resource_evidence.json").write_text(json.dumps(resources, indent=2, sort_keys=True) + "\n")
    (OUT / "acceptance.json").write_text(json.dumps(result, indent=2, allow_nan=False, ensure_ascii=False) + "\n")
    scripts = sorted((ROOT / "scripts").glob("t5b_*.py")) + [ROOT / "scripts/t4d_source_eval.py", ROOT / "scripts/t4s_report_tables.py"]
    (OUT / "script_sha256.json").write_text(json.dumps({str(p.relative_to(ROOT)): sha(p) for p in scripts}, indent=2) + "\n")
    print(json.dumps(dict(gates=result["gates"], all_passed=result["all_passed"]), indent=2))


if __name__ == "__main__":
    main()
