"""Audit T4t stored narrow-blue guard evidence and manifest suffixes, without image reads."""
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4u"


def main():
    rows, hashes = {}, {}
    for part in ("a", "b", "cross"):
        rel = f"experiments/t4_auto_ui/t4t/candidate_{part}.json"
        content = subprocess.check_output(["git", "show", f"567aaf9:{rel}"], cwd=ROOT)
        hashes[rel] = hashlib.sha256(content).hexdigest()
        record = json.loads(content)
        assert not rows.keys() & record["frames"].keys()
        rows.update(record["frames"])
    candidates, low_ratio = {}, {}
    for name, row in rows.items():
        if "error" in row:
            continue
        guard = (row.get("white_point") or {}).get("blue_guard", {})
        bg, ratio = guard.get("base_bg"), guard.get("ratio_bg")
        if bg is not None and ratio is not None and ratio < .55:
            low_ratio[name] = dict(base_bg=bg, ratio_bg=ratio)
            if bg < .48:
                candidates[name] = low_ratio[name]
    suffixes = {}
    for name in ("s3_dev_frames.json", "bn_rb_frames.json"):
        frames = json.loads((ROOT / "data/manifests" / name).read_text())["frames"]
        suffixes[name] = dict(Counter(Path(f["raw"]).suffix.lower() for f in frames))
    result = dict(nonerror_records=sum("error" not in v for v in rows.values()),
                  low_ratio_records=low_ratio, narrow_blue_new_branch_candidates=candidates,
                  manifest_suffix_counts=suffixes, historical_artifact_sha256=hashes,
                  limits="Rounded stored params cannot prove exact pre-boost Tint maxima; full rerun checks exact params/metrics equivalence.")
    (OUT / "historical_gate_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Historical low blue ratio records:", len(low_ratio), "; new branch candidates:", len(candidates))


if __name__ == "__main__":
    main()
