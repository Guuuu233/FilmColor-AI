"""Verify T4r evidence and rebuild tables without decoding original material.
Usage: python -I scripts/t4r_report_tables.py
Baseline: committed T4o and T4q evidence. Candidate evidence in experiments/t4_auto_ui/t4r.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4r"
EXPERIMENTS = OUT.parent
BASE_4D31A38 = "4d31a38"
PROTECTED = ("ext:A008", "ext:A012", "ext:A013", "shop:DSC_3833", "nlp:_DSF8948")
PRESETS_ALL = ("standard", "natural", "portrait", "vivid", "print_classic")
DEV_VARIANTS_EXPECTED = {
    "standard", "natural", "portrait", "vivid", "print_classic",
    "old_wb_gate",
    "old_overlift_standard", "old_overlift_natural", "old_overlift_portrait",
    "old_overlift_vivid", "old_overlift_print_classic"
}
CROSS_VARIANTS_EXPECTED = {"standard", "old_wb_gate", "old_overlift_standard"}


def read(folder: Path, name: str) -> dict:
    return json.loads((folder / f"{name}.json").read_text())


def source_sha() -> str:
    h = hashlib.sha256()
    for p in sorted(p for p in (ROOT / "src/filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(ROOT / "src")).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def committed_source_sha(commit: str) -> str:
    h = hashlib.sha256()
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit, "src/filmcolor"],
        cwd=ROOT, text=True
    ).splitlines()
    for name in sorted(n for n in names if Path(n).suffix in (".py", ".json", ".html")):
        h.update(
            str(Path(name).relative_to("src")).encode() + b"\0"
            + subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT) + b"\0"
        )
    return h.hexdigest()


def valid(rows: dict) -> set[str]:
    return {k for k, v in rows.items() if "error" not in v and v.get("ncc", 0) >= 0.7}


def value(rows: dict, key: str, variant: str = "standard") -> float:
    return float(rows[key]["variants"][variant]["metrics"]["de2000_median"])


def stats(x: np.ndarray | list[float]) -> str:
    a = np.asarray(x, dtype=float)
    return " / ".join(f"{v:.2f}" for v in (np.median(a), np.mean(a), np.percentile(a, 90)))


def counts(x: np.ndarray | list[float], threshold: float) -> str:
    a = np.asarray(x, dtype=float)
    return f"{int((a < -threshold).sum())} / {int((a > threshold).sum())}"


def summarize_preset_comparison(ids: list[str], old_rows: dict, new_rows: dict, preset: str) -> dict:
    u = np.array([value(old_rows, k, preset) for k in ids], dtype=float)
    v = np.array([value(new_rows, k, preset) for k in ids], dtype=float)
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


def main():
    sha = source_sha()
    base_commit_sha = committed_source_sha(BASE_4D31A38)

    # 1. Load dev (a96, b95) and cross (124) from t4o, t4q, and t4r
    collections = {ver: {"dev": {}, "cross": {}} for ver in ("t4o", "t4q", "t4r")}
    for version, groups in collections.items():
        folder = OUT if version == "t4r" else EXPERIMENTS / version
        for part, n in (("a", 96), ("b", 95), ("cross", 124)):
            rec = read(folder, f"candidate_{part}")
            assert len(rec["frames"]) == n, f"{version} candidate_{part} frames count {len(rec['frames'])} != {n}"
            if version == "t4r":
                assert rec["src_content_sha256"] == sha, f"candidate_{part} src sha mismatch: {rec['src_content_sha256']} != {sha}"
                expected_variants = CROSS_VARIANTS_EXPECTED if part == "cross" else DEV_VARIANTS_EXPECTED
                assert set(rec["variants"]) == expected_variants, f"candidate_{part} variants mismatch: {set(rec['variants'])} != {expected_variants}"
                # Protocol invariants check against t4q
                t4q_rec = read(EXPERIMENTS / "t4q", f"candidate_{part}")
                for field in ("manifest", "max_side", "ncc_min", "app_config", "roll_folder", "base_mode"):
                    assert t4q_rec.get(field) == rec.get(field), f"{part} {field} changed vs t4q: {t4q_rec.get(field)} != {rec.get(field)}"
            group = groups["cross" if part == "cross" else "dev"]
            assert not (group.keys() & rec["frames"].keys()), f"Duplicate keys in {version} {part}"
            group.update(rec["frames"])

    # Verify dev manifest membership & exclusions
    dev_manifest = read(ROOT / "data/manifests", "s3_dev_frames")["frames"]
    cross_manifest = read(ROOT / "data/manifests", "bn_rb_frames")["frames"]
    dev_r = collections["t4r"]["dev"]
    cross_r = collections["t4r"]["cross"]
    assert set(dev_r) == {f["id"] for f in dev_manifest}, "dev frame id set mismatch with manifest"
    assert set(cross_r) == {f["id"] for f in cross_manifest if not f["s3_5"]}, "cross frame id set mismatch with manifest"

    # Exclusions check: fixed 6 frames without ICC tag in dev; 1 frame with low NCC in cross
    expected_dev_errors = {"ext:" + n for n in ("A016", "A018", "A025", "A026", "A027", "A028")}
    actual_dev_errors = {k for k, v in dev_r.items() if "error" in v}
    assert actual_dev_errors == expected_dev_errors, f"Dev error set mismatch: {actual_dev_errors} != {expected_dev_errors}"

    lines = [
        "# T4r verified regression", "",
        "Per-frame median ΔE2000, then median / mean / p90 across frames.", "",
        "| Set | N | T4o | T4q | T4r | vs T4o >.25 | vs T4o >.05 | vs T4q >.25 | vs T4q >.05 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    details = []
    acceptance = {}

    for label, n_expected in (("dev", 185), ("cross", 123)):
        old_o = collections["t4o"][label]
        old_q = collections["t4q"][label]
        cand = collections["t4r"][label]

        ids = sorted(valid(cand))
        assert len(ids) == n_expected, f"{label} valid count {len(ids)} != {n_expected}"
        assert valid(old_o) == valid(old_q) == set(ids), f"{label} valid frame set mismatch across versions"
        assert set(old_o) == set(old_q) == set(cand), f"{label} total frame set mismatch"

        # Verify raw/ref hashes and variant status/finiteness
        for k in ids:
            for baseline in (old_o, old_q):
                for field in ("raw_sha256", "ref_sha256"):
                    assert baseline[k][field] == cand[k][field], f"{k} {field} changed vs baseline"
            for var_name, var_row in cand[k]["variants"].items():
                assert var_row["status"] == "ok", f"{k} {var_name} status not ok: {var_row['status']}"
                assert np.isfinite(var_row["metrics"]["de2000_median"]), f"{k} {var_name} de2000 not finite"
            std_params = cand[k]["variants"]["standard"]["params"]
            assert np.isfinite([std_params.get(p, 0.0) for p in ("ev", "temp", "tint", "blacks")]).all(), k

        # Subdomain breakdown for standard variant
        domains = ("nlp", "shop", "blueneg", "ext_own", "all") if label == "dev" else ("all",)
        for domain in domains:
            keys = [k for k in ids if domain == "all" or cand[k]["domain"] == domain]
            uo = np.array([value(old_o, k) for k in keys], dtype=float)
            uq = np.array([value(old_q, k) for k in keys], dtype=float)
            ur = np.array([value(cand, k) for k in keys], dtype=float)
            lines.append(
                f"| {label}:{domain} | {len(keys)} | {stats(uo)} | {stats(uq)} | {stats(ur)} | "
                f"{counts(ur - uo, 0.25)} | {counts(ur - uo, 0.05)} | "
                f"{counts(ur - uq, 0.25)} | {counts(ur - uq, 0.05)} |"
            )

        # Standard delta vs T4o and T4q
        delta_vs_o = {k: value(cand, k) - value(old_o, k) for k in ids}
        delta_vs_q = {k: value(cand, k) - value(old_q, k) for k in ids}

        acceptance[label] = dict(
            n=n_expected,
            source_sha256=sha,
            primary_variant="standard",
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
            vs_t4q=dict(
                better_025=sum(v < -0.25 for v in delta_vs_q.values()),
                worse_025=sum(v > 0.25 for v in delta_vs_q.values()),
                better_005=sum(v < -0.05 for v in delta_vs_q.values()),
                worse_005=sum(v > 0.05 for v in delta_vs_q.values()),
                max_regression=max(delta_vs_q.values()) if delta_vs_q else 0.0,
                regressions_over_005={k: v for k, v in delta_vs_q.items() if v > 0.05},
                metrics_before=dict(zip(('median', 'mean', 'p90'),
                                        (float(f([value(old_q, k) for k in ids])) for f in (np.median, np.mean, lambda x: np.percentile(x, 90))))),
                metrics_after=dict(zip(('median', 'mean', 'p90'),
                                       (float(f([value(cand, k) for k in ids])) for f in (np.median, np.mean, lambda x: np.percentile(x, 90)))))
            ),
            presets_vs_t4o={},
            presets_vs_t4q={}
        )

        details += [
            "", f"## {label}: changes vs T4q (standard)", "",
            "| Frame | T4q | T4r | Change | mean_dL before → after |",
            "|---|---:|---:|---:|---:|"
        ]
        for k in sorted(ids, key=delta_vs_q.get):
            if abs(delta_vs_q[k]) > 1e-6:
                a_dl = old_q[k]["variants"]["standard"]["metrics"]["mean_dL"]
                b_dl = cand[k]["variants"]["standard"]["metrics"]["mean_dL"]
                details.append(f"| {k} | {value(old_q, k):.6f} | {value(cand, k):.6f} | {delta_vs_q[k]:+.6f} | {a_dl:+.3f} → {b_dl:+.3f} |")

        details += [
            "", f"## {label}: changes vs T4o (standard)", "",
            "| Frame | T4o | T4r | Change | mean_dL before → after |",
            "|---|---:|---:|---:|---:|"
        ]
        for k in sorted(ids, key=delta_vs_o.get):
            if abs(delta_vs_o[k]) > 1e-6:
                a_dl = old_o[k]["variants"]["standard"]["metrics"]["mean_dL"]
                b_dl = cand[k]["variants"]["standard"]["metrics"]["mean_dL"]
                details.append(f"| {k} | {value(old_o, k):.6f} | {value(cand, k):.6f} | {delta_vs_o[k]:+.6f} | {a_dl:+.3f} → {b_dl:+.3f} |")

        # Preset comparison vs T4o and vs T4q
        presets_eval = PRESETS_ALL if label == "dev" else ("standard",)
        for p in presets_eval:
            if all(p in cand[k]["variants"] for k in ids):
                if all(p in old_o[k]["variants"] for k in ids):
                    acceptance[label]["presets_vs_t4o"][p] = summarize_preset_comparison(ids, old_o, cand, p)
                if all(p in old_q[k]["variants"] for k in ids):
                    acceptance[label]["presets_vs_t4q"][p] = summarize_preset_comparison(ids, old_q, cand, p)

        details += [
            "", f"## {label}: presets vs T4o", "",
            "| Preset | T4o | T4r | Better / worse >.25 | Better / worse >.05 | Max regression |",
            "|---|---:|---:|---:|---:|---:|"
        ]
        for p in presets_eval:
            if p in acceptance[label]["presets_vs_t4o"]:
                pr = acceptance[label]["presets_vs_t4o"][p]
                details.append(
                    f"| {p} | {pr['stats_before']} | {pr['stats_after']} | "
                    f"{pr['better_025']} / {pr['worse_025']} | {pr['better_005']} / {pr['worse_005']} | {pr['max_regression']:+.4f} |"
                )

        details += [
            "", f"## {label}: presets vs T4q", "",
            "| Preset | T4q | T4r | Better / worse >.25 | Better / worse >.05 | Max regression |",
            "|---|---:|---:|---:|---:|---:|"
        ]
        for p in presets_eval:
            if p in acceptance[label]["presets_vs_t4q"]:
                pr = acceptance[label]["presets_vs_t4q"][p]
                details.append(
                    f"| {p} | {pr['stats_before']} | {pr['stats_after']} | "
                    f"{pr['better_025']} / {pr['worse_025']} | {pr['better_005']} / {pr['worse_005']} | {pr['max_regression']:+.4f} |"
                )

        # Regressions list per preset (>0.05)
        details += ["", f"### {label}: preset regressions >.05 vs T4q", ""]
        for p in presets_eval:
            if p in acceptance[label]["presets_vs_t4q"]:
                regs = acceptance[label]["presets_vs_t4q"][p]["regressions_over_005"]
                reg_str = ", ".join(f"{k} {v:+.6f}" for k, v in sorted(regs.items(), key=lambda x: -x[1])) or "none"
                details.append(f"- **{p}**: {reg_str}")

        details += ["", f"### {label}: preset regressions >.05 vs T4o", ""]
        for p in presets_eval:
            if p in acceptance[label]["presets_vs_t4o"]:
                regs = acceptance[label]["presets_vs_t4o"][p]["regressions_over_005"]
                reg_str = ", ".join(f"{k} {v:+.6f}" for k, v in sorted(regs.items(), key=lambda x: -x[1])) or "none"
                details.append(f"- **{p}**: {reg_str}")

        # Ablation 1: old_wb_gate (gate closed vs open)
        details += ["", f"## {label}: old_wb_gate ablation (standard WB new gate vs old gate)", ""]
        wb_gate_diff = [value(cand, k) - value(cand, k, "old_wb_gate") for k in ids]
        wb_touched = [k for k in ids if cand[k]["variants"]["standard"]["params"] != cand[k]["variants"]["old_wb_gate"]["params"]]
        acceptance[label]["wb_gate_touched"] = len(wb_touched)
        acceptance[label]["wb_gate_better_025"] = int((np.array(wb_gate_diff) < -0.25).sum())
        acceptance[label]["wb_gate_worse_005"] = int((np.array(wb_gate_diff) > 0.05).sum())
        details.append(
            f"Touched frames: {len(wb_touched)}: {', '.join(wb_touched) or 'none'}. "
            f">.25 {counts(wb_gate_diff, 0.25)}, >.05 {counts(wb_gate_diff, 0.05)}."
        )
        if wb_touched:
            details += [
                "", "| Frame | Old WB gate ΔE | New WB gate ΔE | Diff | New Temp/Tint | Old Temp/Tint |",
                "|---|---:|---:|---:|---|---|"
            ]
            for k in sorted(wb_touched):
                p_new = cand[k]["variants"]["standard"]["params"]
                p_old = cand[k]["variants"]["old_wb_gate"]["params"]
                details.append(
                    f"| {k} | {value(cand, k, 'old_wb_gate'):.6f} | {value(cand, k):.6f} | "
                    f"{value(cand, k) - value(cand, k, 'old_wb_gate'):+.6f} | "
                    f"({p_new.get('temp', 0):+.3f}, {p_new.get('tint', 0):+.3f}) | "
                    f"({p_old.get('temp', 0):+.3f}, {p_old.get('tint', 0):+.3f}) |"
                )

        # Ablation 2: old_overlift_{preset} (wide-bright exemption on vs off)
        details += ["", f"## {label}: over-lift wide-bright exemption ablation (new standard vs old_overlift_*)", ""]
        acceptance[label]["overlift_exemptions"] = {}
        for p in presets_eval:
            var_old = f"old_overlift_{p}"
            if var_old not in cand[ids[0]]["variants"]:
                continue
            ol_diff = [value(cand, k, p) - value(cand, k, var_old) for k in ids]
            ol_touched = [k for k in ids if cand[k]["variants"][p]["params"] != cand[k]["variants"][var_old]["params"]]
            acceptance[label]["overlift_exemptions"][p] = dict(
                touched=len(ol_touched),
                better_025=int((np.array(ol_diff) < -0.25).sum()),
                worse_005=int((np.array(ol_diff) > 0.05).sum()),
                touched_frames=ol_touched
            )
            details.append(
                f"- **{p}**: touched {len(ol_touched)} frames: {', '.join(ol_touched) or 'none'}; "
                f">.25 {counts(ol_diff, 0.25)}, >.05 {counts(ol_diff, 0.05)}."
            )
            if ol_touched:
                for k in sorted(ol_touched):
                    details.append(
                        f"  - `{k}`: with exemption {value(cand, k, p):.6f} vs without {value(cand, k, var_old):.6f} "
                        f"({value(cand, k, p) - value(cand, k, var_old):+.6f})"
                    )

        # Base diagnostics & spike clamp statistics
        bare_backlight_frames = [
            k for k in ids if (cand[k].get("base_diagnostics") or {}).get("bare_backlight_excluded")
        ]
        spike_clamped_frames = [
            k for k in ids if (cand[k].get("spike_clamp") or {}).get("pixels", 0) > 0
        ]
        spike_pixels_total = sum(
            (cand[k].get("spike_clamp") or {}).get("pixels", 0) for k in ids
        )
        acceptance[label]["base_diagnostics"] = dict(
            bare_backlight_excluded_count=len(bare_backlight_frames),
            bare_backlight_excluded_frames=bare_backlight_frames
        )
        spike_changed = [k for k in ids if cand[k].get('spike_clamp') != old_q[k].get('spike_clamp')]
        acceptance[label]["spike_clamp"] = dict(
            clamped_frames_count=len(spike_clamped_frames),
            total_pixels_clamped=spike_pixels_total,
            report_changed_vs_t4q=spike_changed
        )
        details += [
            "", f"### {label}: base diagnostics and spike clamp statistics", "",
            f"- `bare_backlight_excluded` active on {len(bare_backlight_frames)} frames: {', '.join(bare_backlight_frames) or 'none'}.",
            f"- `spike_clamp` active on {len(spike_clamped_frames)} frames, total clamped pixels: {spike_pixels_total}.",
            '- Spike-clamp report changed vs T4q: ' + (', '.join(spike_changed) or 'none') + '.'
        ]

        # Clip share check
        over_clip = [(k, cand[k]["variants"]["standard"].get("clip_share", 0.0))
                     for k in ids if (cand[k]["variants"]["standard"].get("clip_share") or 0.0) > 0.002]
        details.append(
            f"- Final clip > .002: {', '.join(f'{k} {v:.6f}' for k, v in over_clip) or 'none'}."
        )

    lines += details

    # Protected frames check
    lines += ["", "## Protected frames", ""]
    for k in PROTECTED:
        a = collections["t4o"]["dev"][k]["variants"]["standard"]
        b = collections["t4r"]["dev"][k]["variants"]["standard"]
        assert a["params"] == b["params"] and a["metrics"] == b["metrics"], f"Protected frame {k} changed"
        lines.append(f"- {k}: unchanged, ΔE {b['metrics']['de2000_median']:.6f}.")

    # Devices baseline & candidate checks
    lines += [
        "", "## Devices", "",
        "| File / mode | 4d31a38 baseline status | T4r status | Full finite | Preserves pos | EV | Temp | Clip |",
        "|---|---|---|---|---|---:|---:|---:|"
    ]
    devices_summary = {}
    for film, n_expected in (("color", 20), ("bw", 3)):
        baseline = read(OUT, f"devices_baseline_{film}")
        assert baseline["src_content_sha256"] == base_commit_sha, (
            f"devices_baseline_{film} source sha mismatch: {baseline['src_content_sha256']} != {base_commit_sha}"
        )
        before = baseline["samples"]
        assert len(before) == n_expected, f"devices_baseline_{film} samples count {len(before)} != {n_expected}"

        rec = read(OUT, f"devices_candidate_{film}")
        assert rec["src_content_sha256"] == sha, f"devices_candidate_{film} src sha mismatch: {rec['src_content_sha256']} != {sha}"
        after = rec["samples"]
        assert len(after) == n_expected and set(before) == set(after), f"devices_candidate_{film} samples mismatch"

        for k, row in after.items():
            assert before[k]["raw_sha256"] == row["raw_sha256"], f"Device {k} raw sha changed"
            # Verify status and finiteness; list failures without blocking table generation
            b_status = before[k].get("error") or before[k].get("status", "unknown")
            a_status = row.get("error") or row.get("status", "unknown")
            is_finite = bool(row.get("full_positive_finite", False))
            preserves_pos = bool(row.get("render_preserves_positive", False))
            p = row.get("params") or {}
            ev_str = f"{p['ev']:+.4f}" if "ev" in p else "N/A"
            temp_str = f"{p['temp']:+.4f}" if "temp" in p else "N/A"
            clip_value = (row.get('diagnostics') or {}).get('clip_share')
            clip_str = f'{clip_value:.6f}' if clip_value is not None else 'N/A'
            lines.append(f"| {k} / {film} | {b_status} | {a_status} | {is_finite} | {preserves_pos} | {ev_str} | {temp_str} | {clip_str} |")
        devices_summary[film] = {
            k: dict(baseline_status=before[k].get('error') or before[k].get('status'),
                    candidate_status=after[k].get('error') or after[k].get('status'),
                    full_positive_finite=bool(after[k].get('full_positive_finite')),
                    render_preserves_positive=bool(after[k].get('render_preserves_positive')))
            for k in after
        }
    acceptance['devices'] = devices_summary
    all_devices = [row for mode in devices_summary.values() for row in mode.values()]
    acceptance['device_checks'] = dict(
        all_status_ok=all(row['candidate_status'] == 'ok' for row in all_devices),
        all_finite=all(row['full_positive_finite'] for row in all_devices),
        all_preserve_positive=all(row['render_preserves_positive'] for row in all_devices),
        status_regressions={f'{name}/{film}': row for film, mode in devices_summary.items()
                            for name, row in mode.items()
                            if row['baseline_status'] == 'ok' and row['candidate_status'] != 'ok'})

    lines += ["", "## Device spatial and over-lift diagnostics", ""]
    cand_color_samples = read(OUT, "devices_candidate_color")["samples"]
    for k, row in sorted(cand_color_samples.items()):
        d = row.get("diagnostics") or {}
        w = (d.get("wb") or {}).get("diagnostics", {})
        if k in ("A277.dng", "A1170.NEF", "A1036.RW2", "A1296.CR2", "A1102.ARW") or d.get("overlift_damp"):
            scan = row.get('scan') or {}
            estimate = scan.get('estimate') or {}
            diag_entry = dict(
                status=row.get('status'), flags=row.get('flags'),
                own_base=scan.get('own_rgb'),
                bare_backlight_excluded=(estimate.get('diagnostics') or {}).get('bare_backlight_excluded', False),
                positive_q=row.get('positive_q'), holder_share=row.get('holder_share'),
                overlift=d.get("overlift_damp"),
                wb={f: w.get(f) for f in ("split_sky", "bright_px_share", "top_bottom_spread", "zone_disagreement") if f in w}
            )
            lines.append(f"- {k}: `{json.dumps(diag_entry, sort_keys=True)}`")

    # Tests evidence
    tests = read(OUT, "test_evidence")
    assert tests["source_sha256"] == sha, f"test_evidence source sha mismatch: {tests['source_sha256']} != {sha}"
    for path_str, exp_sha in tests["test_sha256"].items():
        actual_test_sha = hashlib.sha256((ROOT / path_str).read_bytes()).hexdigest()
        assert actual_test_sha == exp_sha, f"Test file {path_str} hash changed: {actual_test_sha} != {exp_sha}"
    lines += ["", "## Full pytest", "", f"{tests['passed']} passed in {tests['seconds']:.2f}s."]

    # Resources evidence
    resource_path = OUT / "resource_evidence.json"
    resources = json.loads(resource_path.read_text()) if resource_path.exists() else {}
    for p in sorted(OUT.glob("*.log")):
        values = re.findall(r"(\d+)\s+peak memory footprint", p.read_text(errors="replace"))
        if values:
            resources[p.stem] = dict(
                peak_memory_footprint_bytes=max(map(int, values)),
                log_sha256=hashlib.sha256(p.read_bytes()).hexdigest()
            )
    resource_path.write_text(json.dumps(resources, indent=2, sort_keys=True) + "\n")
    lines += ["", "## Peak memory footprint (GiB)", ""]
    lines += [f"- {k}: {v['peak_memory_footprint_bytes'] / 2**30:.2f}" for k, v in sorted(resources.items())]
    for required_log in ("candidate_a", "candidate_b", "candidate_cross", "devices_candidate_color", "devices_candidate_bw", "tests"):
        if required_log not in resources:
            lines.append(f"- {required_log}: unavailable (no peak-memory statistic in log).")

    # Service check
    service = read(OUT, "service_check")
    assert service["status"] == "ok" and service["source_thumb_rev"] == "t4r" and service["source_config_rev"] == "t4r-2026-11-01"
    lines += [
        "", "## Service", "",
        f"`{json.dumps(service, sort_keys=True)}`", "",
        f"Source SHA256: `{sha}`."
    ]

    # Write deliverables
    (OUT / "TABLES.md").write_text("\n".join(lines).rstrip() + "\n")
    (OUT / "acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")

    # Evidence script hashes
    scripts = sorted((ROOT / "scripts").glob("t4r_*.py")) + [ROOT / "scripts/t4d_source_eval.py"]
    script_shas = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts}
    (OUT / "evidence_script_sha256.json").write_text(json.dumps(script_shas, indent=2, sort_keys=True) + "\n")

    print(json.dumps(acceptance, indent=2))
    print(f"Verified T4r evidence: dev 185 valid, cross 123 valid, 20 color + 3 bw device probes, tests and service.")


if __name__ == "__main__":
    main()
