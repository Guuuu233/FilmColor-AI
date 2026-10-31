"""Verify and report T4o full-resolution evaluation against c76e588, without image reads.
Usage: python -I scripts/t4o_report_tables.py
"""
import hashlib
import json
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4o"
BEFORE = ROOT / "experiments/t4_auto_ui/t4n"
PROTECTED = ("ext:A008", "ext:A012", "ext:A013", "shop:DSC_3833", "nlp:_DSF8948")
NAMED = ("ext:A104", "ext:A138", "ext:A006", "ext:A011", "ext:A196", "ext:A103",
         "blueneg:19970620C-10-st-michel")


def read(folder, name):
    return json.loads((folder / f"{name}.json").read_text())


def source_sha():
    h = hashlib.sha256()
    for p in sorted(p for p in (ROOT / "src/filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(ROOT / "src")).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def valid(frames):
    return {k for k, v in frames.items() if "error" not in v and v.get("ncc", 0) >= .7}


def de(frames, key, variant="standard"):
    return frames[key]["variants"][variant]["metrics"]["de2000_median"]


def stats(values):
    return " / ".join(f"{v:.2f}" for v in (np.median(values), np.mean(values), np.percentile(values, 90)))


def counts(values, threshold=.25):
    a = np.asarray(values)
    return f"{int((a < -threshold).sum())} / {int((a > threshold).sum())}"


def main():
    sha = source_sha()
    old, new = {}, {}
    for part, n in (("a", 96), ("b", 95)):
        a, b = read(BEFORE, f"candidate_{part}"), read(OUT, f"candidate_{part}")
        assert b["src_content_sha256"] == sha
        assert len(b["frames"]) == n and not new.keys() & b["frames"].keys()
        old.update(a["frames"])
        new.update(b["frames"])
    manifest = read(ROOT / "data/manifests", "s3_dev_frames")["frames"]
    assert set(old) == set(new) == {f["id"] for f in manifest}
    assert {k for k, v in new.items() if "error" in v} == {"ext:" + n for n in ("A016", "A018", "A025", "A026", "A027", "A028")}
    ca, cb_record = read(BEFORE, "candidate_cross")["frames"], read(OUT, "candidate_cross")
    cb = cb_record["frames"]
    assert cb_record["src_content_sha256"] == sha and set(ca) == set(cb) and len(cb) == 124
    summaries, detail = [], []
    for label, a, b, expected in (("dev", old, new, 185), ("cross", ca, cb, 123)):
        assert valid(a) == valid(b) and len(valid(b)) == expected
        ids = sorted(valid(b))
        for key in ids:
            assert a[key]["raw_sha256"] == b[key]["raw_sha256"] and a[key]["ref_sha256"] == b[key]["ref_sha256"], key
        for domain in (("nlp", "shop", "blueneg", "ext_own", "all") if label == "dev" else ("all",)):
            keys = [k for k in ids if domain == "all" or b[k]["domain"] == domain]
            x, y = (np.array([de(d, k) for k in keys]) for d in (a, b))
            summaries.append(f"| {label}:{domain} | {len(keys)} | {stats(x)} | {stats(y)} | {counts(y-x)} | {counts(y-x,.05)} |")
        detail += ["", f"## {label}: mechanism checks", ""]
        for name in ("no_rescue", "no_spatial_gate"):
            diffs = [de(b, k) - de(b, k, name) for k in ids]
            changed = [k for k in ids if b[k]["variants"]["standard"]["params"] != b[k]["variants"][name]["params"]]
            detail.append(f"- {name}: parameters changed {len(changed)}; on minus off >.25 {counts(diffs)}, >.05 {counts(diffs,.05)}: {', '.join(changed) or 'none'}.")
        holder = [k for k in ids if (b[k].get("white_point") or {}).get("holder_blocks_excluded")]
        rescue = [k for k in ids if (b[k]["variants"]["standard"].get("backlit") or {}).get("rescue_weight", 0) > 0]
        cool = [k for k in ids if b[k]["variants"]["standard"].get("wb_cool_cast")]
        over = [k for k in ids if (b[k]["variants"]["standard"].get("clip_share") or 0) > .0021]
        detail += [f"- Holder white-point filter: {len(holder)}: {', '.join(holder) or 'none'}.",
                   f"- Rescue: {len(rescue)}: {', '.join(rescue) or 'none'}.",
                   f"- Cool cast: {len(cool)}; clip_share > .0021: {len(over)}: {', '.join(over) or 'none'}."]
        detail += ["", f"## {label}: metric changes > .000001", "", "| Frame | T4n | T4o | ΔE change | mean_dL T4n → T4o |", "|---|---:|---:|---:|---:|"]
        for key in sorted(ids, key=lambda k: de(b, k) - de(a, k)):
            delta = de(b, key) - de(a, key)
            if abs(delta) <= 1e-6:
                continue
            u, v = (d[key]["variants"]["standard"]["metrics"]["mean_dL"] for d in (a, b))
            detail.append(f"| {key} | {de(a,key):.6f} | {de(b,key):.6f} | {delta:+.6f} | {u:+.3f} → {v:+.3f} |")
    lines = ["# T4o vs c76e588 (T4n) reproducible results", "", "Per-frame median ΔE2000, then median / mean / p90 across frames; lower is closer to the reference.", "",
             "| Set | N | T4n | T4o | Better / worse > .25 | Better / worse > .05 |", "|---|---:|---:|---:|---:|---:|"] + summaries + detail
    lines += ["", "## Protected and named dev frames", "", "| Frame | T4n | T4o |", "|---|---:|---:|"]
    for key in PROTECTED + NAMED:
        lines.append(f"| {key} | {de(old,key):.6f} | {de(new,key):.6f} |")
    lines += ["", "## Devices (no paired reference)", "", "| Sample | T4n status | T4o status | EV T4n → T4o | Temp T4n → T4o |", "|---|---|---|---:|---:|"]
    da, db_record = read(OUT, "devices_baseline")["samples"], read(OUT, "devices_candidate")
    assert db_record["src_content_sha256"] == sha
    expected_devices = {"A1195.CR2", "A240.CR3", "A328.CR3", "A329.ARW", "A099.NEF", "A1094.NEF", "A1170.NEF",
                        "A1095.RAF", "A1042.RAF", "A277.dng", "A281.RAF", "A1005.CR2", "A1089.DNG", "A322.cr2"}
    assert set(db_record["samples"]) == set(da) == expected_devices
    for key, b in db_record["samples"].items():
        a = da[key]
        assert a["raw_sha256"] == b["raw_sha256"]
        assert "error" not in b and b.get("full_positive_finite") and b.get("render_preserves_positive"), key
        def val(r, name):
            x = (r.get("params") or {}).get(name)
            return "—" if x is None else f"{x:+.4f}"
        lines.append(f"| {key} | {a.get('error') or a.get('status')} | {b.get('error') or b.get('status')} | {val(a,'ev')} → {val(b,'ev')} | {val(a,'temp')} → {val(b,'temp')} |")
    lines += ["", "## Check-set rescue targets: post-Step-5 inputs", "",
              "| Frame | span G | textured median | q25 | q90 |", "|---|---:|---:|---:|---:|"]
    traces = read(OUT, "cross_target_diagnostics")
    for key, rec in traces.items():
        assert rec["params"] == cb[key]["variants"]["standard"]["params"], key
        q25, q90 = rec["rescue_input_q25_q90"]
        lines.append(f"| {key} | {rec['span_d_g']:.6f} | {rec['rescue_input_median']:.6f} | {q25:.6f} | {q90:.6f} |")
    lines += ["", "## Peak memory footprint (GiB)", ""]
    for p in sorted(OUT.glob("*.log")):
        values = re.findall(r"(\d+)\s+peak memory footprint", p.read_text(errors="replace"))
        if values:
            lines.append(f"- {p.stem}: {max(map(int,values))/2**30:.2f}")
    lines += ["", f"Candidate source SHA256: `{sha}`."]
    (OUT / "TABLES.md").write_text("\n".join(lines).rstrip() + "\n")
    scripts = [ROOT / "scripts" / n for n in ("t4o_run_eval.py", "t4o_run_devices.py", "t4o_prepare_baseline.py", "t4o_device_sheet.py", "t4o_probe_devices.py", "t4o_probe_cross.py", "t4o_report_tables.py", "t4d_source_eval.py")]
    (OUT / "evidence_script_sha256.json").write_text(json.dumps({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts}, indent=2) + "\n")
    print("Verified 191 dev / 185 valid, 124 cross / 123 valid, input hashes, source hash and fixed membership.")
    print(OUT / "TABLES.md")


if __name__ == "__main__":
    main()
