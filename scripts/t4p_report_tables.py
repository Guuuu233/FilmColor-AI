"""Verify and summarize T4p vs 4cd76b7 without reading original images.
Usage: python -I scripts/t4p_report_tables.py
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4p"
BEFORE = ROOT / "experiments/t4_auto_ui/t4o"
PROTECTED = ("ext:A008", "ext:A012", "ext:A013", "shop:DSC_3833", "nlp:_DSF8948")


def read(folder, name):
    return json.loads((folder / f"{name}.json").read_text())


def source_sha():
    h = hashlib.sha256()
    for p in sorted(p for p in (ROOT / "src/filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(ROOT / "src")).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def committed_source_sha(commit):
    h = hashlib.sha256()
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, 'src/filmcolor'],
                                    cwd=ROOT, text=True).splitlines()
    for name in sorted(n for n in names if Path(n).suffix in ('.py', '.json', '.html')):
        h.update(str(Path(name).relative_to('src')).encode() + b'\0'
                 + subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=ROOT) + b'\0')
    return h.hexdigest()


def valid(rows):
    return {k for k, v in rows.items() if "error" not in v and v.get("ncc", 0) >= .7}


def de(rows, key, variant="standard"):
    return rows[key]["variants"][variant]["metrics"]["de2000_median"]


def stats(x):
    return " / ".join(f"{v:.2f}" for v in (np.median(x), np.mean(x), np.percentile(x, 90)))


def counts(x, threshold=.25):
    a = np.asarray(x)
    return f"{int((a < -threshold).sum())} / {int((a > threshold).sum())}"


def main():
    sha = source_sha()
    old, new = {}, {}
    for part, count in (("a", 96), ("b", 95)):
        a, b = read(BEFORE, f"candidate_{part}"), read(OUT, f"candidate_{part}")
        assert b["src_content_sha256"] == sha and len(b["frames"]) == count
        assert set(b['variants']) == {'standard', 'natural', 'portrait', 'vivid', 'print_classic', 'old_highlight_drop', 'old_sky_gate'}
        for field in ('manifest', 'max_side', 'ncc_min', 'app_config', 'roll_folder', 'base_mode'):
            assert a.get(field) == b.get(field), (part, field)
        assert not new.keys() & b["frames"].keys()
        old.update(a["frames"])
        new.update(b["frames"])
    manifest = read(ROOT / "data/manifests", "s3_dev_frames")["frames"]
    assert set(old) == set(new) == {f["id"] for f in manifest}
    assert {k for k, v in new.items() if "error" in v} == {"ext:" + n for n in ("A016", "A018", "A025", "A026", "A027", "A028")}
    ca, cb_record = read(BEFORE, "candidate_cross")["frames"], read(OUT, "candidate_cross")
    cb = cb_record["frames"]
    assert cb_record["src_content_sha256"] == sha and set(ca) == set(cb) and len(cb) == 124
    for field in ('manifest', 'max_side', 'ncc_min', 'app_config', 'roll_folder', 'base_mode'):
        assert read(BEFORE, 'candidate_cross').get(field) == cb_record.get(field), field
    assert set(cb_record['variants']) == {'standard', 'old_highlight_drop', 'old_sky_gate'}
    summary, details = [], []
    for label, a, b, expected in (("dev", old, new, 185), ("cross", ca, cb, 123)):
        ids = sorted(valid(b))
        assert valid(a) == valid(b) and len(ids) == expected
        for key in ids:
            assert a[key]["raw_sha256"] == b[key]["raw_sha256"] and a[key]["ref_sha256"] == b[key]["ref_sha256"], key
            for variant, row in b[key]['variants'].items():
                assert row['status'] == 'ok', (label, key, variant, row['status'])
                assert np.isfinite(row['metrics']['de2000_median']), (label, key, variant)
                assert np.isfinite([row['params'][p] for p in ('ev', 'temp', 'tint', 'midtone', 'shoulder')]).all()
        for domain in (("nlp", "shop", "blueneg", "ext_own", "all") if label == "dev" else ("all",)):
            keys = [k for k in ids if domain == "all" or b[k]["domain"] == domain]
            u, v = (np.array([de(r, k) for k in keys]) for r in (a, b))
            summary.append(f"| {label}:{domain} | {len(keys)} | {stats(u)} | {stats(v)} | {counts(v-u)} | {counts(v-u,.05)} |")
        details += ["", f"## {label}: changes", "", "| Frame | T4o | T4p | ΔE change | mean_dL before → after |", "|---|---:|---:|---:|---:|"]
        for key in sorted(ids, key=lambda k: de(b,k)-de(a,k)):
            diff = de(b,key)-de(a,key)
            if abs(diff) > 1e-6:
                u, v = (r[key]["variants"]["standard"]["metrics"]["mean_dL"] for r in (a,b))
                details.append(f"| {key} | {de(a,key):.6f} | {de(b,key):.6f} | {diff:+.6f} | {u:+.3f} → {v:+.3f} |")
        details += ["", f"## {label}: ablations and highlight budget", ""]
        for variant in ("old_highlight_drop", "old_sky_gate"):
            changes = [de(b,k)-de(b,k,variant) for k in ids]
            touched = [k for k in ids if b[k]["variants"]["standard"]["params"] != b[k]["variants"][variant]["params"]]
            details.append(f"- {variant}: >.25 {counts(changes)}, >.05 {counts(changes,.05)}; parameters changed {len(touched)}: {', '.join(touched) or 'none'}.")
        over = [(v["variants"]["standard"].get("clip_share") or 0,k) for k,v in b.items() if k in ids and (v["variants"]["standard"].get("clip_share") or 0) > .002]
        details.append("- clip_share > .002: " + (", ".join(f"{k} {clip:.6f}" for clip,k in sorted(over,reverse=True)) or "none") + ".")
        details.append(f"- Cool-cast count: {sum(bool(b[k]['variants']['standard'].get('wb_cool_cast')) for k in ids)}.")
        lowered = [(k, b[k]['variants']['standard'].get('highlight_hold')) for k in ids
                   if (b[k]['variants']['standard'].get('highlight_hold') or {}).get('ev_lowered', 0) > 0]
        details.append('- Step 5 EV reductions: ' + (', '.join(f'{k} {v["ev_lowered"]:.4f}' for k, v in lowered) or 'none') + '.')
        rescues = [(k, b[k]['variants']['standard'].get('backlit', {})) for k in ids
                   if (b[k]['variants']['standard'].get('backlit') or {}).get('rescue_weight', 0) > 0]
        details.append('- Active capped rescues: ' + ', '.join(f'{k} {v["rescue_weight"]:.6f}' for k, v in rescues) + '.')
        wb_changed = [k for k in ids if any(a[k]['variants']['standard']['params'][p] != b[k]['variants']['standard']['params'][p]
                                           for p in ('temp', 'tint'))]
        details.append('- WB parameters changed vs T4o: ' + (', '.join(wb_changed) or 'none') + '.')
        details += ['', f'## {label}: preset results', '', '| Preset | N | T4o | T4p | Better / worse >.25 | Better / worse >.05 |',
                    '|---|---:|---:|---:|---:|---:|']
        for preset in ('standard', 'natural', 'portrait', 'vivid', 'print_classic'):
            if not all(preset in a[k]['variants'] and preset in b[k]['variants'] for k in ids):
                continue
            u, v = (np.array([de(r, k, preset) for k in ids]) for r in (a, b))
            details.append(f'| {preset} | {len(ids)} | {stats(u)} | {stats(v)} | {counts(v-u)} | {counts(v-u,.05)} |')
    lines = ["# T4p regression vs 4cd76b7", "", "Per-frame median ΔE2000, then median / mean / p90 across frames.", "",
             "| Set | N | T4o | T4p | Better / worse >.25 | Better / worse >.05 |", "|---|---:|---:|---:|---:|---:|"] + summary + details
    lines += ['', '## Additional comparison vs T4n', '',
              '| Set | T4n | T4p | Better / worse >.25 | Better / worse >.05 |',
              '|---|---:|---:|---:|---:|']
    t4n_dev = {}
    for part in ('a', 'b'):
        t4n_dev.update(read(ROOT / 'experiments/t4_auto_ui/t4n', f'candidate_{part}')['frames'])
    t4n_cross = read(ROOT / 'experiments/t4_auto_ui/t4n', 'candidate_cross')['frames']
    for label, a, b in (('dev', t4n_dev, new), ('cross', t4n_cross, cb)):
        ids = sorted(valid(b))
        assert valid(a) == set(ids)
        for key in ids:
            assert a[key]['raw_sha256'] == b[key]['raw_sha256'] and a[key]['ref_sha256'] == b[key]['ref_sha256']
        u, v = (np.array([de(r, k) for k in ids]) for r in (a, b))
        lines.append(f'| {label} | {stats(u)} | {stats(v)} | {counts(v-u)} | {counts(v-u,.05)} |')
    lines += ["", "## Protected frames", "", "| Frame | T4o | T4p |", "|---|---:|---:|"]
    for key in PROTECTED:
        lines.append(f"| {key} | {de(old,key):.6f} | {de(new,key):.6f} |")
    lines += ["", "## Devices by mode", "", "| Sample / mode | Before status | After status | EV before → after | Temp before → after | final clip |", "|---|---|---|---:|---:|---:|"]
    baseline_sha = committed_source_sha('4cd76b7')
    for film, expected in (("color",18),("bw",3)):
        baseline = read(OUT, f'devices_baseline_{film}')
        assert baseline['src_content_sha256'] == baseline_sha
        a = baseline['samples']
        rec = read(OUT,f"devices_candidate_{film}")
        assert rec["src_content_sha256"] == sha and set(a) == set(rec["samples"]) and len(a) == expected
        for key,b in rec["samples"].items():
            before = a[key]
            assert before["raw_sha256"] == b["raw_sha256"]
            assert not b.get('error') and b.get('status') == 'ok', (key, film)
            assert b['full_positive_finite'] and b['render_preserves_positive'], (key, film)
            if key in ('B086.dng', 'B087.dng'):
                assert b['scanner']['monochrome_dng'] is True and b['scanner']['infrared_layer'] is None
            def val(r,name):
                x=(r.get("params") or {}).get(name)
                return "—" if x is None else f"{x:+.4f}"
            lines.append(f"| {key} / {film} | {before.get('error') or before.get('status')} | {b.get('error') or b.get('status')} | {val(before,'ev')} → {val(b,'ev')} | {val(before,'temp')} → {val(b,'temp')} | {(b.get('diagnostics') or {}).get('clip_share','—')} |")
    lines += ['', '## Device parameter changes and rescue evidence', '']
    before = read(OUT, 'devices_baseline_color')['samples']
    after = read(OUT, 'devices_candidate_color')['samples']
    for key, row in after.items():
        old_params = before[key].get('params') or {}
        params = row.get('params') or {}
        changes = {k: [old_params.get(k), v] for k, v in params.items() if old_params.get(k) != v}
        rescue = row['diagnostics'].get('backlit_check') or {}
        old_rescue = (before[key].get('diagnostics') or {}).get('backlit_check') or {}
        if changes or rescue.get('rescue_weight', 0) > 0:
            lines.append(f'- {key}: parameter changes `{json.dumps(changes, sort_keys=True)}`; '
                         f'rescue weight {old_rescue.get("rescue_weight", 0):.6f} → {rescue.get("rescue_weight", 0):.6f}.')
    lines += ['', '## Target diagnostics', '']
    for film, names in (('color', ('A1170.NEF', 'A1036.RW2')), ('bw', ('B091.RW2',))):
        before = read(OUT, f'devices_baseline_{film}')['samples']
        after = read(OUT, f'devices_candidate_{film}')['samples']
        for key in names:
            row = after[key]
            diag = row['diagnostics']
            wb = (diag.get('wb') or {}).get('diagnostics', {})
            selected = dict(raw_sha256=row['raw_sha256'], highlight_hold=diag.get('highlight_hold'),
                            wb={k: wb.get(k) for k in ('bright_px_share', 'zone_disagreement', 'top_bottom_spread', 'split_sky', 'cool_cast')},
                            display_rgb_q50_before=before[key].get('display_rgb_q50'),
                            display_rgb_q50_after=row.get('display_rgb_q50'))
            lines += [f'- {key} / {film}: `{json.dumps(selected, sort_keys=True)}`.']
    lines += ["", "## Peak memory footprint (GiB)", ""]
    resource_path = OUT / 'resource_evidence.json'
    resources = json.loads(resource_path.read_text()) if resource_path.exists() else {}
    for p in sorted(OUT.glob("*.log")):
        values = re.findall(r"(\d+)\s+peak memory footprint",p.read_text(errors="replace"))
        if values:
            resources[p.stem] = dict(peak_memory_footprint_bytes=max(map(int, values)),
                                     log_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    resource_path.write_text(json.dumps(resources, indent=2, sort_keys=True) + '\n')
    for name, value in sorted(resources.items()):
        lines.append(f'- {name}: {value["peak_memory_footprint_bytes"] / 2**30:.2f}')
    tests = read(OUT, 'test_evidence')
    assert tests['source_sha256'] == sha
    for path, expected in tests['test_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    lines += ['', '## Full pytest', '', f'{tests["passed"]} passed in {tests["seconds"]:.2f}s; '
              f'peak memory footprint {tests["peak_memory_footprint_bytes"] / 2**30:.2f} GiB.', '',
              f"Candidate source SHA256: `{sha}`."]
    service = read(OUT, 'service_check')
    assert service['status'] == 'ok' and service['source_thumb_rev'] == 't4p'
    assert service['source_config_rev'] == 't4p-2026-10-31' and service['highlight_hold']['max_drop'] == .6
    lines += ['', '## Local service', '', f'`{json.dumps(service, sort_keys=True)}`.']
    (OUT / "TABLES.md").write_text("\n".join(lines).rstrip()+"\n")
    scripts = [ROOT/"scripts"/n for n in ("t4p_run_eval.py","t4p_run_devices.py","t4p_probe_devices.py","t4p_device_sheet.py","t4p_prepare_baseline.py","t4p_service_check.py","t4p_test_evidence.py","t4p_report_tables.py","t4d_source_eval.py")]
    (OUT/"evidence_script_sha256.json").write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},indent=2)+"\n")
    print("Verified 191 dev / 185 valid, 124 cross / 123 valid, input/source hashes and 18 color + 3 bw probes.")
    print(OUT/"TABLES.md")


if __name__ == "__main__":
    main()
