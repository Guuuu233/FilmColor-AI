"""Audit the task's historical gate claims using committed T4s evaluation records only."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4t"


def main():
    frames, artifacts = {}, {}
    for part in ("a", "b", "cross"):
        name = f"experiments/t4_auto_ui/t4s/candidate_{part}.json"
        content = subprocess.check_output(["git", "show", f"7c5107c:{name}"], cwd=ROOT)
        artifacts[name] = hashlib.sha256(content).hexdigest()
        record = json.loads(content)
        assert record["src_content_sha256"] == "27ba01f3aa13d29905af25f17284b68bc99da2897a40c957c2a5f4d898b9efb5"
        assert not frames.keys() & record["frames"].keys()
        frames.update(record["frames"])
    rows = {k: v for k, v in frames.items() if "error" not in v}
    channel_cap_hits = []
    mount_initial_hits = []
    ratio = {}
    max_density = {}
    zone = {}
    cyan = {}
    for name, row in rows.items():
        d = row["d_hi"]
        ratio[name] = max(d[0], d[2]) / d[1]
        max_density[name] = max(d)
        if d[1] >= .20 and any(d[c] > max(2.5, 2.2*d[1]) for c in (0, 2)):
            channel_cap_hits.append(name)
        if d[1] > 1.15 and d[0] > d[1] + .25:
            mount_initial_hits.append(name)
        for preset, variant in row["variants"].items():
            z = variant.get("wb_spatial", {}).get("zone_disagreement")
            if z is not None:
                zone[f"{name}:{preset}"] = z
            if variant.get("t4m", {}).get("cyan_conf_boost"):
                cyan[f"{name}:{preset}"] = variant.get("wb_spatial", {}).get("bright_px_share")
    def maximum(values):
        key = max(values, key=values.get)
        return dict(record=key, value=values[key])
    out = dict(nonerror_records=len(rows), includes_low_ncc_check_record=True,
               channel_cap_hits=channel_cap_hits, slide_mount_initial_signature_hits=mount_initial_hits,
               max_non_green_white_point_ratio=maximum(ratio), max_white_point=maximum(max_density),
               max_zone_disagreement=maximum(zone), cyan_boost_bright_shares=cyan,
               historical_artifact_sha256=artifacts,
               limits="Stored diagnostics cannot recompute Dm escape rates or edge-ring density shares; no claim is made for those.")
    (OUT / "reference_gate_audit.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
