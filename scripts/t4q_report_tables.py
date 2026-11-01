"""Verify T4q evidence and rebuild tables without decoding original material.
Usage: python -I scripts/t4q_report_tables.py
Baseline: committed T4o evidence; T4p evidence documents the spatial-radius rollback.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4q"
EXPERIMENTS = OUT.parent
PROTECTED = ("ext:A008", "ext:A012", "ext:A013", "shop:DSC_3833", "nlp:_DSF8948")


def read(folder, name):
    return json.loads((folder / f"{name}.json").read_text())


def source_sha():
    h = hashlib.sha256()
    for p in sorted(p for p in (ROOT / "src/filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(ROOT / "src")).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def committed_source_sha():
    h = hashlib.sha256()
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', '4cd76b7', 'src/filmcolor'],
                                    cwd=ROOT, text=True).splitlines()
    for name in sorted(n for n in names if Path(n).suffix in ('.py', '.json', '.html')):
        h.update(str(Path(name).relative_to('src')).encode() + b'\0'
                 + subprocess.check_output(['git', 'show', f'4cd76b7:{name}'], cwd=ROOT) + b'\0')
    return h.hexdigest()


def valid(rows):
    return {k for k, v in rows.items() if "error" not in v and v.get("ncc", 0) >= .7}


def value(rows, key, variant="standard"):
    return rows[key]["variants"][variant]["metrics"]["de2000_median"]


def stats(x):
    return " / ".join(f"{v:.2f}" for v in (np.median(x), np.mean(x), np.percentile(x, 90)))


def counts(x, threshold):
    a = np.asarray(x)
    return f"{int((a < -threshold).sum())} / {int((a > threshold).sum())}"


def main():
    sha = source_sha()
    collections = {version: {"dev": {}, "cross": {}} for version in ("t4o", "t4p", "t4q")}
    for version, groups in collections.items():
        for part, n in (("a", 96), ("b", 95), ("cross", 124)):
            rec = read(EXPERIMENTS / version, f"candidate_{part}")
            assert len(rec["frames"]) == n
            if version == "t4q":
                assert rec["src_content_sha256"] == sha
                expected = {"standard", "no_overlift"} | (set() if part == "cross" else {"natural", "portrait", "vivid", "print_classic"})
                assert set(rec["variants"]) == expected
                old = read(EXPERIMENTS / "t4o", f"candidate_{part}")
                for field in ("manifest", "max_side", "ncc_min", "app_config", "roll_folder", "base_mode"):
                    assert old.get(field) == rec.get(field), (part, field)
            group = groups["cross" if part == "cross" else "dev"]
            assert not group.keys() & rec["frames"].keys()
            group.update(rec["frames"])
    manifest = read(ROOT / "data/manifests", "s3_dev_frames")["frames"]
    dev = collections["t4q"]["dev"]
    assert set(dev) == {f["id"] for f in manifest}
    assert {k for k, v in dev.items() if "error" in v} == {"ext:" + n for n in ("A016", "A018", "A025", "A026", "A027", "A028")}
    lines = ["# T4q verified regression", "", "Per-frame median ΔE2000, then median / mean / p90 across frames.", "",
             "| Set | N | T4o | T4q | Better / worse >.25 | Better / worse >.05 |", "|---|---:|---:|---:|---:|---:|"]
    details = []
    acceptance = {}
    for label, n in (("dev", 185), ("cross", 123)):
        old, previous, new = (collections[v][label] for v in ("t4o", "t4p", "t4q"))
        ids = sorted(valid(new))
        assert len(ids) == n and valid(old) == valid(previous) == set(ids)
        assert set(old) == set(previous) == set(new)
        for k in ids:
            for source in (old, previous):
                for field in ("raw_sha256", "ref_sha256"):
                    assert source[k][field] == new[k][field], (k, field)
            for variant, row in new[k]["variants"].items():
                assert row["status"] == "ok" and np.isfinite(row["metrics"]["de2000_median"]), (k, variant)
        for domain in (("nlp", "shop", "blueneg", "ext_own", "all") if label == "dev" else ("all",)):
            keys = [k for k in ids if domain == "all" or new[k]["domain"] == domain]
            u, v = (np.array([value(r, k) for k in keys]) for r in (old, new))
            lines.append(f"| {label}:{domain} | {len(keys)} | {stats(u)} | {stats(v)} | {counts(v-u,.25)} | {counts(v-u,.05)} |")
        delta = {k: value(new, k) - value(old, k) for k in ids}
        for k in ids:
            assert np.isfinite([new[k]['variants']['standard']['params'].get(p, 0.0) for p in ('ev', 'temp', 'tint', 'blacks')]).all(), k
        acceptance[label] = dict(n=n, better_025=sum(v < -.25 for v in delta.values()),
                                 worse_025=sum(v > .25 for v in delta.values()),
                                 better_005=sum(v < -.05 for v in delta.values()),
                                 worse_005=sum(v > .05 for v in delta.values()),
                                 max_regression=max(delta.values()),
                                 metrics_before=dict(zip(('median', 'mean', 'p90'),
                                                         (float(f([value(old,k) for k in ids])) for f in
                                                          (np.median, np.mean, lambda x: np.percentile(x,90))))),
                                 metrics_after=dict(zip(('median', 'mean', 'p90'),
                                                        (float(f([value(new,k) for k in ids])) for f in
                                                         (np.median, np.mean, lambda x: np.percentile(x,90))))),
                                 source_sha256=sha)
        details += ["", f"## {label}: changes vs T4o", "", "| Frame | T4o | T4q | Change | mean_dL before → after |", "|---|---:|---:|---:|---:|"]
        for k in sorted(ids, key=delta.get):
            if abs(delta[k]) > 1e-6:
                a, b = (r[k]["variants"]["standard"]["metrics"]["mean_dL"] for r in (old, new))
                details.append(f"| {k} | {value(old,k):.6f} | {value(new,k):.6f} | {delta[k]:+.6f} | {a:+.3f} → {b:+.3f} |")
        applied = [k for k in ids if new[k]["variants"]["standard"].get("overlift_damp")]
        touched = [k for k in ids if new[k]["variants"]["standard"]["params"] != new[k]["variants"]["no_overlift"]["params"]]
        assert set(applied) == set(touched)
        for k in applied:
            d = new[k]['variants']['standard']['overlift_damp']
            assert d['applied'] and 0 < d['pull'] <= .65 and 0 <= d['blacks_lowered'] <= .06
            assert d['ev_before'] >= 1.5 and d['q25_before'] >= .35
            assert d['q25_before'] == new[k]['variants']['standard']['backlit']['p25']
        differences = [value(new,k) - value(new,k,"no_overlift") for k in ids]
        acceptance[label]['overlift_applied'] = len(applied)
        acceptance[label]['overlift_better_025'] = sum(v < -.25 for v in differences)
        acceptance[label]['overlift_worse_005'] = sum(v > .05 for v in differences)
        details += ["", f"## {label}: over-lift ablation", "",
                    f"Applied {len(applied)}; >.25 {counts(differences,.25)}, >.05 {counts(differences,.05)}.", "",
                    "| Frame | Off ΔE | On ΔE | Diagnostic |", "|---|---:|---:|---|"]
        for k in ids:
            row = new[k]['variants']['standard']
            off = new[k]['variants']['no_overlift']
            for field in ('backlit', 'highlight_hold'):
                assert row.get(field) == off.get(field), (k, field)
        for k in applied:
            row = new[k]["variants"]["standard"]
            details.append(f"| {k} | {value(new,k,'no_overlift'):.6f} | {value(new,k):.6f} | `{json.dumps(row['overlift_damp'],sort_keys=True)}` |")
        a_delta = [value(new,k,"no_overlift") - value(old,k) for k in ids]
        a_touched = [k for k in ids if new[k]['variants']['no_overlift']['params'] != old[k]['variants']['standard']['params']]
        details += ["", f"Module A alone vs T4o: >.25 {counts(a_delta,.25)}, >.05 {counts(a_delta,.05)}.",
                    'Module A parameter changes: ' + (', '.join(a_touched) or 'none') + '.']
        gains = [k for k in applied if value(new,k) < value(new,k,'no_overlift')]
        losses = [k for k in applied if value(new,k) > value(new,k,'no_overlift')]
        details.append(f'Applied frames with strictly lower / higher ΔE (no tolerance): {len(gains)} / {len(losses)}.')
        over = [(k,new[k]["variants"]["standard"]["clip_share"]) for k in ids if new[k]["variants"]["standard"]["clip_share"] > .002]
        details.append("Final clip > .002: " + (", ".join(f"{k} {v:.6f}" for k,v in over) or "none") + ".")
        details += ["", f"## {label}: presets vs T4o", "", "| Preset | T4o | T4q | Better / worse >.25 | Better / worse >.05 |", "|---|---:|---:|---:|---:|"]
        acceptance[label]['primary_variant'] = 'standard'
        acceptance[label]['presets'] = {}
        for preset in ("standard", "natural", "portrait", "vivid", "print_classic"):
            if not all(preset in new[k]["variants"] for k in ids):
                continue
            u,v = (np.array([value(r,k,preset) for k in ids]) for r in (old,new))
            acceptance[label]['presets'][preset] = dict(
                better_025=int(((v-u)<-.25).sum()), worse_025=int(((v-u)>.25).sum()),
                better_005=int(((v-u)<-.05).sum()), worse_005=int(((v-u)>.05).sum()),
                max_regression=float((v-u).max()),
                regressions_over_005={k: value(new,k,preset)-value(old,k,preset) for k in ids
                                      if value(new,k,preset)-value(old,k,preset) > .05})
            details.append(f"| {preset} | {stats(u)} | {stats(v)} | {counts(v-u,.25)} | {counts(v-u,.05)} |")
        for preset in ('natural', 'portrait', 'vivid', 'print_classic'):
            if not all(preset in new[k]['variants'] for k in ids):
                continue
            losses = [(k, value(new,k,preset)-value(old,k,preset)) for k in ids
                      if value(new,k,preset)-value(old,k,preset) > .05]
            details += ['', f'{preset} regressions >.05: ' +
                        (', '.join(f'{k} {v:+.6f}' for k,v in losses) or 'none') + '.']
        wb_changes = [k for k in ids if any(old[k]["variants"]["standard"]["params"][p] != new[k]["variants"]["standard"]["params"][p] for p in ("temp","tint"))]
        cool_count = sum(bool(new[k]['variants']['standard']['wb_cool_cast']) for k in ids)
        details += ["", "WB parameter changes vs T4o: " + (", ".join(wb_changes) or "none") + ".",
                    f"Cool-cast count: {cool_count}."]
        regressions = [k for k in ids if value(previous,k)-value(old,k) > .05]
        remaining = [k for k in regressions if delta[k] > .05]
        details += ["", "Prior T4p regressions >.05 remaining: " + (", ".join(remaining) or "none") + "."]
        stage_delta = [value(new,k) - value(previous,k) for k in ids]
        previous_delta = [value(previous,k) - value(old,k) for k in ids]
        details += ["", f'T4q vs uncommitted T4p: >.25 {counts(stage_delta,.25)}, >.05 {counts(stage_delta,.05)}.',
                    f'Historical T4p vs T4o recomputed: >.25 {counts(previous_delta,.25)}, >.05 {counts(previous_delta,.05)}.']
    lines += details + ["", "## Protected frames", ""]
    for k in PROTECTED:
        a,b = (collections[v]["dev"][k]["variants"]["standard"] for v in ("t4o","t4q"))
        assert a["params"] == b["params"] and a["metrics"] == b["metrics"], k
        lines.append(f"- {k}: unchanged, ΔE {b['metrics']['de2000_median']:.6f}.")
    lines += ["", "## Devices", "", "| File / mode | 4cd76b7 source status (probed during T4p) | T4q status | EV | Temp | Clip |", "|---|---|---|---:|---:|---:|"]
    baseline_sha = committed_source_sha()
    for film,n in (("color",18),("bw",3)):
        baseline = read(EXPERIMENTS / 't4p', f'devices_baseline_{film}')
        assert baseline['src_content_sha256'] == baseline_sha
        before = baseline['samples']
        rec = read(OUT, f"devices_candidate_{film}")
        assert rec["src_content_sha256"] == sha and len(rec["samples"]) == n and set(before) == set(rec["samples"])
        for k,row in rec["samples"].items():
            assert before[k]["raw_sha256"] == row["raw_sha256"]
            assert not row.get("error") and row["status"] == "ok" and row["full_positive_finite"] and row["render_preserves_positive"], (k,film)
            if k in ("B086.dng","B087.dng"):
                assert row["scanner"]["monochrome_dng"] and row["scanner"]["infrared_layer"] is None
            p=row["params"]
            lines.append(f"| {k} / {film} | {before[k].get('error') or before[k]['status']} | {row['status']} | {p['ev']:+.4f} | {p['temp']:+.4f} | {row['diagnostics']['clip_share']:.6f} |")
    device_color = read(OUT, 'devices_candidate_color')['samples']
    device_bw = read(OUT, 'devices_candidate_bw')['samples']
    assert device_color['A277.dng']['params']['temp'] == .0707
    assert device_color['A1170.NEF']['params']['temp'] == .2508
    assert device_bw['B091.RW2']['params']['ev'] == .2349
    lines += ['', '## Device over-lift and spatial diagnostics', '']
    for k, row in device_color.items():
        d = row['diagnostics']
        w = (d.get('wb') or {}).get('diagnostics', {})
        if k in ('A277.dng', 'A1170.NEF', 'A1036.RW2') or d.get('overlift_damp'):
            selected = dict(overlift=d.get('overlift_damp'),
                            wb={f:w.get(f) for f in ('split_sky', 'bright_px_share', 'top_bottom_spread', 'zone_disagreement')})
            lines.append(f'- {k}: `{json.dumps(selected,sort_keys=True)}`.')
    tests = read(OUT,"test_evidence")
    assert tests["source_sha256"] == sha
    for path,expected in tests["test_sha256"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, path
    lines += ["", "## Full pytest", "", f"{tests['passed']} passed in {tests['seconds']:.2f}s."]
    resource_path = OUT / "resource_evidence.json"
    resources = json.loads(resource_path.read_text()) if resource_path.exists() else {}
    for p in sorted(OUT.glob("*.log")):
        values = re.findall(r"(\d+)\s+peak memory footprint",p.read_text(errors="replace"))
        if values:
            resources[p.stem] = dict(peak_memory_footprint_bytes=max(map(int,values)), log_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    resource_path.write_text(json.dumps(resources,indent=2,sort_keys=True)+"\n")
    lines += ["", "## Peak memory footprint (GiB)", ""]
    lines += [f"- {k}: {v['peak_memory_footprint_bytes']/2**30:.2f}" for k,v in sorted(resources.items())]
    for name in ('candidate_a', 'candidate_b', 'candidate_cross', 'devices_candidate_color', 'devices_candidate_bw', 'tests'):
        if name not in resources:
            lines.append(f'- {name}: unavailable (no peak-memory statistic in log).')
    service = read(OUT,"service_check")
    assert service["status"] == "ok" and service["source_thumb_rev"] == "t4q" and service["source_config_rev"] == "t4q-2026-11-01"
    lines += ["", "## Service", "", f"`{json.dumps(service,sort_keys=True)}`", "", f"Source SHA256: `{sha}`."]
    (OUT/"TABLES.md").write_text("\n".join(lines).rstrip()+"\n")
    (OUT/"acceptance.json").write_text(json.dumps(acceptance,indent=2)+"\n")
    scripts = sorted((ROOT/"scripts").glob("t4q_*.py")) + [ROOT/"scripts/t4d_source_eval.py"]
    (OUT/"evidence_script_sha256.json").write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},indent=2)+"\n")
    print(json.dumps(acceptance,indent=2))
    print("Verified inputs, source, protocol, 185+123 valid frames, 18 color + 3 bw probes, tests and service.")


if __name__ == "__main__":
    main()
