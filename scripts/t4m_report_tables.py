"""Recompute T4m tables and verify membership/input hashes, without decoding images.
Usage: python -I scripts/t4m_report_tables.py
Reads experiments/t4_auto_ui/t4m/{baseline,candidate}_{a,b,cross}.json and devices_*.json.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/t4_auto_ui/t4m"
PROTECTED = ("ext:A008", "ext:A012", "ext:A013", "shop:DSC_3833", "nlp:_DSF8948")
NAMED = ("shop:DSC_3827", "ext:A006", "ext:A011", "ext:A196", "ext:A216", "ext:A193", "ext:A199", "ext:A208",
         "ext:A209", "ext:A222", "ext:A223", "ext:A224", "ext:A225", "ext:A226", "ext:A227", "ext:A228",
         "nlp:_DSF9239", "blueneg:19960812D-09-las-vegas")
EXCLUDED = {"ext:" + n for n in ("A016", "A018", "A025", "A026", "A027", "A028")}


def load(name):
    return json.loads((OUT / f"{name}.json").read_text())


def merged(prefix):
    a, b = load(f"{prefix}_a"), load(f"{prefix}_b")
    assert not (a["frames"].keys() & b["frames"].keys())
    assert len(a["frames"]) == 96 and len(b["frames"]) == 95
    assert a["src_content_sha256"] == b["src_content_sha256"]
    records = a["frames"] | b["frames"]
    manifest = json.loads((REPO / "data/manifests/s3_dev_frames.json").read_text())["frames"]
    assert set(records) == {f["id"] for f in manifest}
    assert {k for k, v in records.items() if "error" in v} == EXCLUDED
    return records


def valid(records):
    return {k for k, v in records.items() if "error" not in v and v.get("ncc", 0) >= .7}


def de(records, fid, variant="standard"):
    return records[fid]["variants"][variant]["metrics"]["de2000_median"]


def stats(values):
    return " / ".join(f"{v:.2f}" for v in (np.median(values), np.mean(values), np.percentile(values, 90)))


def better_worse(delta, threshold):
    d = np.asarray(delta)
    return f"{int((d < -threshold).sum())} / {int((d > threshold).sum())}"


def paired(a, b, ids):
    for fid in ids:
        assert a[fid]["raw_sha256"] == b[fid]["raw_sha256"], fid
        assert a[fid]["ref_sha256"] == b[fid]["ref_sha256"], fid


def table(a, b, ids, title):
    lines = [f"## {title}", "", "Per-frame median ΔE2000, then median / mean / p90 over frames. Negative change helps.", "",
             "| Subset | N | T4l e8a8c3d | T4m | Better / worse > .25 | Better / worse > .05 | median mean_dL T4l → T4m |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for domain in ("nlp", "shop", "blueneg", "ext_own", "all"):
        sub = sorted(k for k in ids if domain == "all" or b[k]["domain"] == domain)
        if not sub:
            continue
        x, y = (np.array([de(d, k) for k in sub]) for d in (a, b))
        dl = [np.median([d[k]["variants"]["standard"]["metrics"]["mean_dL"] for k in sub]) for d in (a, b)]
        lines.append(f"| {domain} | {len(sub)} | {stats(x)} | {stats(y)} | {better_worse(y-x,.25)} | "
                     f"{better_worse(y-x,.05)} | {dl[0]:+.2f} → {dl[1]:+.2f} |")
    return lines + [""]


def main():
    a, b = merged("baseline"), merged("candidate")
    va, vb = valid(a), valid(b)
    dev_valid_a, dev_valid_b = set(va), set(vb)
    vb = va & vb
    paired(a, b, vb)
    ca, cb = load("baseline_cross"), load("candidate_cross")
    cross_ids = {f["id"] for f in json.loads((REPO / "data/manifests/bn_rb_frames.json").read_text())["frames"] if not f["s3_5"]}
    assert set(ca["frames"]) == set(cb["frames"]) == cross_ids
    cva, cvb = valid(ca["frames"]), valid(cb["frames"])
    cross_valid_a, cross_valid_b = set(cva), set(cvb)
    cvb = cva & cvb
    paired(ca["frames"], cb["frames"], cvb)
    h = hashlib.sha256()
    for p in sorted(p for p in (REPO / "src/filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(REPO / "src")).encode() + b"\0" + p.read_bytes() + b"\0")
    for name in ("candidate_a", "candidate_b", "candidate_cross", "devices_candidate"):
        assert load(name)["src_content_sha256"] == h.hexdigest(), name
    assert load("baseline_a")["src_content_sha256"] == ca["src_content_sha256"] == load("devices_baseline")["src_content_sha256"]
    lines = ["# T4m reproducible evaluation tables", "", "Baseline is a fresh run of e8a8c3d, not the research side's quick evaluation.",
             "The BlueNeg check set contains other frames of existing development rolls; it is not a set of unseen rolls.", ""]
    lines += [f"Development validity: baseline {len(dev_valid_a)}, candidate {len(dev_valid_b)}, paired {len(vb)}; "
              f"membership difference: {', '.join(sorted(dev_valid_a ^ dev_valid_b)) or 'none'}.", "",
              f"Cross validity: baseline {len(cross_valid_a)}, candidate {len(cross_valid_b)}, paired {len(cvb)}; "
              f"membership difference: {', '.join(sorted(cross_valid_a ^ cross_valid_b)) or 'none'}.", ""]
    lines += table(a, b, vb, "Development set")
    lines += table(ca["frames"], cb["frames"], cvb, "BlueNeg check set")
    lines += [f"BlueNeg candidates {len(cross_ids)}, valid {len(cvb)}; excluded: " + ", ".join(sorted(cross_ids-cvb)) + ".", "",
              "## Protected and named frames", "", "| Frame | T4l | T4m | ΔE change | EV T4l → T4m | T4m flags |", "|---|---:|---:|---:|---:|---|"]
    for fid in PROTECTED + NAMED:
        if fid not in vb:
            continue
        u, v = (d[fid]["variants"]["standard"] for d in (a,b))
        lines.append(f"| {fid} | {de(a,fid):.2f} | {de(b,fid):.2f} | {de(b,fid)-de(a,fid):+.2f} | "
                     f"{u['params']['ev']:+.3f} → {v['params']['ev']:+.3f} | {', '.join(v.get('flags') or [])} |")
    lines += ["", "## Estimator ablations (on minus off, standard preset)", ""]
    for label, records, ids in (("dev", b, vb), ("cross", cb["frames"], cvb)):
        for variant in ("thin_t4l", "no_cyan"):
            delta = [de(records,k)-de(records,k,variant) for k in sorted(ids)]
            lines.append(f"- {label} {variant}: >.25 {better_worse(delta,.25)}, >.05 {better_worse(delta,.05)}.")
        guard = [k for k in sorted(ids) if ((records[k].get("white_point") or {}).get("blue_guard") or {}).get("applied")]
        cyan = [k for k in sorted(ids) if records[k]["variants"]["standard"].get("t4m",{}).get("cyan_conf_boost")]
        thin = [k for k in sorted(ids) if (records[k]["variants"]["standard"].get("t4k",{}).get("thin_negative") or {}).get("applied")]
        suppressed = [k for k in sorted(ids) if (records[k]["variants"]["standard"].get("backlit") or {}).get("thin_suppressed")]
        for name, hits in (("blue guard",guard),("cyan confidence",cyan),("thin cap",thin),("backlit suppressed",suppressed)):
            lines.append(f"- {label} {name}: {len(hits)}: {', '.join(hits)}.")
    for mode, label in (("guard_off", "Orange-mask blue white guard"), ("holder_off", "Holder enhancement")):
        if (OUT / f"{mode}_a.json").exists() and (OUT / f"{mode}_b.json").exists():
            off = merged(mode)
            ids = vb & valid(off)
            paired(b, off, ids)
            diffs = [de(b,k)-de(off,k) for k in sorted(ids)]
            lines.append(f"- dev {label}, on minus off ({len(ids)}): >.25 {better_worse(diffs,.25)}, >.05 {better_worse(diffs,.05)}.")
        if (OUT / f"{mode}_cross.json").exists():
            off = load(f"{mode}_cross")["frames"]
            ids = cvb & valid(off)
            paired(cb["frames"], off, ids)
            diffs = [de(cb["frames"],k)-de(off,k) for k in sorted(ids)]
            lines.append(f"- cross {label}, on minus off ({len(ids)}): >.25 {better_worse(diffs,.25)}, >.05 {better_worse(diffs,.05)}.")
    lines += ["", "## Largest standard changes vs T4l", "", "| Frame | T4l | T4m | Change |", "|---|---:|---:|---:|"]
    delta = sorted((de(b,k)-de(a,k),k) for k in vb)
    for change, fid in delta[:17] + list(reversed(delta[-10:])):
        lines.append(f"| {fid} | {de(a,fid):.2f} | {de(b,fid):.2f} | {change:+.2f} |")
    lines += ["", "## BlueNeg changes exceeding .05", "", "| Frame | T4l | T4m | Change | Thin on minus off | Cyan on minus off |",
              "|---|---:|---:|---:|---:|---:|"]
    for change, fid in sorted((de(cb["frames"],k)-de(ca["frames"],k),k) for k in cvb):
        if abs(change) <= .05:
            continue
        r = cb["frames"]
        lines.append(f"| {fid} | {de(ca['frames'],fid):.2f} | {de(r,fid):.2f} | {change:+.2f} | "
                     f"{de(r,fid)-de(r,fid,'thin_t4l'):+.2f} | {de(r,fid)-de(r,fid,'no_cyan'):+.2f} |")
    lines += ["", "## Device samples (no reference)", "", "| Sample | T4l | T4m | Holder T4l → T4m | T4m film_box |", "|---|---|---|---:|---|"]
    da, db = (load(f"devices_{n}")["samples"] for n in ("baseline","candidate"))
    for fid in db:
        u,v = da.get(fid,{}),db[fid]
        def status(x):
            return x.get("error") or x.get("status", "—")
        def holder(x):
            return f"{x['holder_share']:.5f}" if x.get("holder_share") is not None else "—"
        assert u.get("raw_sha256") == v["raw_sha256"], fid
        if "error" not in v:
            assert v["render_preserves_positive"], fid
            assert v.get("outer_display_grey_max_error", 0) < 1e-7, fid
        lines.append(f"| {fid} | {status(u)} | {status(v)} | {holder(u)} → {holder(v)} | {v.get('film_box','—')} |")
    assert db["A1089.DNG"]["scanner"]["infrared_layer"]["source"] == "interleaved_4ch"
    assert db["A1283.DNG"]["scanner"]["linearization_table_len"] == 256
    assert db["A1141.NEF"]["status"] == "manual_review_required"
    lines += ["", "## Source fingerprints", ""]
    for name in ("baseline_a","candidate_a","baseline_cross","candidate_cross","devices_baseline","devices_candidate"):
        lines.append(f"- {name}: `{load(name)['src_content_sha256']}`.")
    lines += ["", "## Peak memory footprint (GiB, /usr/bin/time -l)", ""]
    for p in sorted(OUT.glob("*.log")):
        hits = re.findall(r"(\d+)\s+peak memory footprint",p.read_text(errors="replace"))
        if hits:
            lines.append(f"- {p.stem}: {max(map(int,hits))/2**30:.2f} GiB (largest recorded run, including resumed invocations).")
    (OUT / "TABLES.md").write_text("\n".join(lines)+"\n")
    hashes = {str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [REPO / "scripts" / n for n in ("t4m_run_eval.py","t4m_probe_devices.py","t4m_prepare_sources.py","t4m_pipeline.py","t4m_device_sheet.py","t4m_report_tables.py","t4d_source_eval.py")]}
    (OUT / "evidence_script_sha256.json").write_text(json.dumps(hashes,indent=2)+"\n")
    print(f"Verified 191 dev records / {len(vb)} paired valid, disjoint halves, fixed reference exclusions, paired input hashes, cross membership.")
    print(OUT / "TABLES.md")


if __name__ == "__main__":
    main()
