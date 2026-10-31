"""Verify and summarize T4n vs T4m full-resolution evidence; no image reads.
Usage: python -I scripts/t4n_report_tables.py
"""
import hashlib
import json
import re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"experiments/t4_auto_ui/t4n"
OLD=ROOT/"experiments/t4_auto_ui/t4m"
PROTECTED=("ext:A008","ext:A012","ext:A013","shop:DSC_3833","nlp:_DSF8948")


def read(folder,name):
    return json.loads((folder/f"{name}.json").read_text())


def valid(frames):
    return {k for k,v in frames.items() if "error" not in v and v.get("ncc",0)>=.7}


def de(frames,k):
    return frames[k]["variants"]["standard"]["metrics"]["de2000_median"]


def stats(x):
    return " / ".join(f"{y:.2f}" for y in (np.median(x),np.mean(x),np.percentile(x,90)))


def main():
    h=hashlib.sha256()
    for p in sorted(p for p in (ROOT/"src/filmcolor").rglob("*") if p.is_file() and p.suffix in (".py",".json",".html")):
        h.update(str(p.relative_to(ROOT/"src")).encode()+b"\0"+p.read_bytes()+b"\0")
    a,b={},{}
    for part,n in (("a",96),("b",95)):
        prev,cur=read(OLD,f"candidate_{part}"),read(OUT,f"candidate_{part}")
        assert cur["src_content_sha256"]==h.hexdigest()
        assert len(cur["frames"])==n and not b.keys()&cur["frames"].keys()
        a.update(prev["frames"]);b.update(cur["frames"])
    manifest=read(ROOT/"data/manifests","s3_dev_frames")["frames"]
    assert set(a)==set(b)=={f["id"] for f in manifest}
    excludes={"ext:"+x for x in ("A016","A018","A025","A026","A027","A028")}
    assert {k for k,v in b.items() if "error" in v}==excludes
    ca,cb=read(OLD,"candidate_cross")["frames"],read(OUT,"candidate_cross")["frames"]
    assert read(OUT,"candidate_cross")["src_content_sha256"]==h.hexdigest()
    assert set(ca)==set(cb) and len(cb)==124
    lines=["# T4n vs T4m full-resolution regression", "", "T4m evidence is from commit 13db565; T4n is rerun with the same script/protocol.","",
           "| Set | N | T4m median / mean / p90 | T4n median / mean / p90 | better / worse > .25 | better / worse > .05 |",
           "|---|---:|---:|---:|---:|---:|"]
    summaries=[]
    details=[]
    for label,x,y in (("dev",a,b),("cross",ca,cb)):
        assert valid(x)==valid(y)
        ids=sorted(valid(y))
        assert len(ids)==(185 if label=="dev" else 123)
        for k in ids:
            assert x[k]["raw_sha256"]==y[k]["raw_sha256"] and x[k]["ref_sha256"]==y[k]["ref_sha256"],k
        domains=("nlp","shop","blueneg","ext_own","all") if label=="dev" else ("all",)
        for domain in domains:
            keys=[k for k in ids if domain=="all" or y[k]["domain"]==domain]
            u,v=(np.array([de(d,k) for k in keys]) for d in (x,y))
            diff=v-u
            summaries.append(f"| {label}:{domain} | {len(keys)} | {stats(u)} | {stats(v)} | {sum(diff<-.25)} / {sum(diff>.25)} | {sum(diff<-.05)} / {sum(diff>.05)} |")
        changed=[k for k in ids if x[k]["variants"]["standard"]["params"]!=y[k]["variants"]["standard"]["params"]]
        base_changed=[k for k in ids if x[k]["base"]!=y[k]["base"]]
        if label=="dev":
            for name in ("standard","natural","portrait","vivid","print_classic"):
                params_changed=[k for k in ids if x[k]["variants"][name]["params"]!=y[k]["variants"][name]["params"]]
                metrics_changed=[k for k in ids if x[k]["variants"][name]["metrics"]!=y[k]["variants"][name]["metrics"]]
                details.append(f"{label} {name}: params changed {len(params_changed)}, metrics changed {len(metrics_changed)}.")
        details += ["",f"{label}: parameter changes {len(changed)}: {', '.join(changed) or 'none'}.",
                  f"{label}: base metadata changes {len(base_changed)}: {', '.join(base_changed) or 'none'}."]
        for diag in ("mode_seed","relaxed_uniformity"):
            hits=[k for k in ids if (y[k].get("base_diagnostics") or {}).get(diag)]
            details.append(f"{label}: {diag} triggers {len(hits)}: {', '.join(hits) or 'none'}.")
        details += ["",f"## {label} metric changes > .000001", "", "| Frame | T4m | T4n | Difference |", "|---|---:|---:|---:|"]
        for k in sorted(ids,key=lambda k:de(y,k)-de(x,k)):
            if abs(de(y,k)-de(x,k))>1e-6:
                details.append(f"| {k} | {de(x,k):.6f} | {de(y,k):.6f} | {de(y,k)-de(x,k):+.6f} |")
    lines += summaries + [""] + details
    lines += ["","## Protected and cyan target frames","","| Frame | T4m | T4n |","|---|---:|---:|"]
    for k in PROTECTED+("ext:A006","ext:A011","ext:A196"):
        lines.append(f"| {k} | {de(a,k):.6f} | {de(b,k):.6f} |")
    target="blueneg:199807B-08-orlando-seaworld"
    lines += ["",f"Target {target}: T4m {de(ca,target):.6f}, T4n {de(cb,target):.6f}; T4m no_cyan "
              f"{ca[target]['variants']['no_cyan']['metrics']['de2000_median']:.6f}."]
    key=read(OUT,"key_cross")
    assert key["src_content_sha256"]==h.hexdigest()
    assert key["frames"][target]["variants"]["standard"]["params"]==cb[target]["variants"]["standard"]["params"]
    devices=read(OUT,"devices_candidate")
    assert devices["src_content_sha256"]==h.hexdigest()
    lines += ["","## Fault samples","","| Sample | T4m | T4n | Base | Span G | Finite positive |","|---|---|---|---|---:|---|"]
    old=read(OUT,"devices_baseline")["samples"]
    assert set(devices["samples"])=={"A1095.RAF","A277.dng","A281.RAF","A1005.CR2","A1234.DNG"}
    scans=devices["samples"]
    assert scans["A1095.RAF"]["scan"]["estimate"]["diagnostics"].get("mode_seed")
    for name in ("A277.dng","A281.RAF"):
        assert scans[name]["scan"]["estimate"]["diagnostics"].get("relaxed_uniformity")
    assert scans["A1005.CR2"]["white_point"]["blue_guard"]["reason"]=="below_valid_ratio"
    assert scans["A1234.DNG"]["params"]==old["A1234.DNG"]["params"]
    for name,r in devices["samples"].items():
        before=old.get(name,{}).get("error") or old.get(name,{}).get("status", "not probed")
        assert old[name]["raw_sha256"]==r["raw_sha256"],name
        assert "error" not in r and r["full_positive_finite"] and r["render_preserves_positive"],name
        lines.append(f"| {name} | {before} | {r.get('error') or r.get('status')} | {r.get('base',{}).get('rgb_render')} | {r.get('span_d_g',0):.4f} | {r.get('full_positive_finite')} |")
    lines += ["","## Peak memory footprint (GiB)",""]
    for p in sorted(OUT.glob("*.log")):
        matches=re.findall(r"(\d+)\s+peak memory footprint",p.read_text(errors="replace"))
        if matches: lines.append(f"- {p.stem}: {max(map(int,matches))/2**30:.2f}")
    lines += ["",f"Candidate source SHA256: `{h.hexdigest()}`.",""]
    (OUT/"TABLES.md").write_text("\n".join(lines).rstrip()+"\n")
    scripts=[ROOT/"scripts"/n for n in ("t4n_run_eval.py","t4n_probe_devices.py","t4n_prepare_baseline.py","t4n_device_sheet.py","t4n_probe_wb.py","t4n_report_tables.py","t4d_source_eval.py")]
    (OUT/"evidence_script_sha256.json").write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},indent=2)+"\n")
    print("Verified 191 dev records / 185 valid, 124 cross / 123 valid, fixed membership and matching inputs/source.")
    print(OUT/"TABLES.md")


if __name__=="__main__":
    main()
