#!/usr/bin/env python3
"""T4z Blue Cast Relaxation Diagnostic (Section 5.4):
Evaluates blue negative frames (d_hi_B / d_hi_G < 0.44) under valid308 protocol.
Constructs baseline standard vs counterfactual override cfg (R thresholds relaxed to +inf,
retaining B thresholds) via t4d_source_eval evaluation protocol.
Outputs:
  - experiments/t4_auto_ui/t4z/blue_cast_diagnostic.json
  - experiments/t4_auto_ui/t4z/eval_blue_cast_relaxation.json
"""
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / "scripts"))

from filmcolor.frozen import check_source
from filmcolor import noref
from filmcolor.app import State
from t4s_report_tables import valid

OUT_DIR = ROOT / "experiments/t4_auto_ui/t4z"
EVAL_SCRIPT = ROOT / "scripts/t4d_source_eval.py"

PROD_FROZEN_SHA = "136a7dbb951ed4c59497adfd2aa789ee0f180246f4e525c6822a78c9253a4756"


def get_source_sha256() -> str:
    """Exact protocol source hash: relative path + NUL + bytes + NUL for .py, .json, .html."""
    h = hashlib.sha256()
    for p in sorted(p for p in (SRC / "filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(SRC)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def load_valid308_frames() -> dict:
    """Load valid308 frames from T4y benchmark files using protocol valid()."""
    candidate_files = [
        ROOT / "experiments/t4_auto_ui/t4y/candidate_a.json",
        ROOT / "experiments/t4_auto_ui/t4y/candidate_b.json",
        ROOT / "experiments/t4_auto_ui/t4y/candidate_cross.json",
    ]
    all_valid = {}
    for cf in candidate_files:
        if cf.exists():
            data = json.loads(cf.read_text()).get("frames", {})
            v_ids = valid(data)
            for fid in v_ids:
                all_valid[fid] = data[fid]
    return all_valid


def filter_low_ratio_frames(valid_frames: dict) -> dict:
    """Find frames where d_hi_B / d_hi_G < 0.44."""
    low_ratio = {}
    for fid, f in valid_frames.items():
        d_hi = f.get("d_hi", [0, 0, 0])
        if len(d_hi) >= 3 and d_hi[1] > 0:
            if (d_hi[2] / d_hi[1]) < 0.44:
                low_ratio[fid] = f
    return low_ratio


def inspect_wb_zones_and_filter(low_ratio_frames: dict, manifest_map: dict) -> list[dict]:
    """Inspect WB zones on positive samples for all low_ratio frames.
    Filter criteria:
      - cool_cast is False under default cfg
      - log2(B/G) in three places: z0 > 0.20 AND z1 > 0.25 AND gw > 0.40
      - R not low: r_z0 > -0.05 or r_z1 > -0.10 or r_gw > -0.20
      - cool_cast toggles to True when R thresholds relaxed to +inf
    """
    candidates = []
    rolls_state = {}

    for fid in sorted(low_ratio_frames.keys()):
        f_info = manifest_map[fid]
        raw_path = ROOT / f_info["raw"] if not os.path.isabs(f_info["raw"]) else Path(f_info["raw"])
        check_source(raw_path, allow_frozen=False)

        roll_key = (f_info["domain"], f_info["roll"])
        if roll_key not in rolls_state:
            rolls_state[roll_key] = State(raw_path.parent, Path("/tmp/fc_diag_test"), allow_frozen=False, max_side=1000)
        st = rolls_state[roll_key]

        m = st.open(raw_path.name, "color")
        pv = st.preview
        P4, mask4 = noref.estimation_input(pv.full_positive, pv.image_region)
        px = P4[mask4]

        # 1. Default estimate
        res_def = noref.estimate_wb_v3(px)
        diag_def = res_def["diagnostics"]
        cool_cast_def = diag_def.get("cool_cast", False)

        # Zones & GW chroma
        zones = diag_def.get("zones", [])
        z0_c = zones[0].get("chroma") if len(zones) > 0 else None
        z1_c = zones[1].get("chroma") if len(zones) > 1 else None

        # Exact grey-world percentiles from noref.py [15, 88]
        y = px @ noref.LUMA.astype(np.float32)
        ok = y > noref.NOREF_V3["min_luma"]
        px_ok, y_ok = px[ok], y[ok]
        lo, hi = np.percentile(y_ok, [15, 88])
        mid = (y_ok >= lo) & (y_ok <= hi)
        c = noref._chroma(px_ok)
        est_gw = noref._huber_location(c[mid], noref.NOREF_V3["wb_v3_gw_huber"]) if mid.sum() >= noref.NOREF_V3["wb_v3_min_px"] else None

        b_z0 = z0_c[1] if z0_c else -999.0
        b_z1 = z1_c[1] if z1_c else -999.0
        b_gw = est_gw[1] if est_gw is not None else -999.0

        r_z0 = z0_c[0] if z0_c else 999.0
        r_z1 = z1_c[0] if z1_c else 999.0
        r_gw = est_gw[0] if est_gw is not None else 999.0

        # Relaxed R test:
        # Override R thresholds to +inf while keeping B thresholds
        cfg_relaxed = dict(noref.NOREF_V3)
        cfg_relaxed["wb_v3_cool_z0"] = (float("inf"), cfg_relaxed["wb_v3_cool_z0"][1])
        cfg_relaxed["wb_v3_cool_z1"] = (float("inf"), cfg_relaxed["wb_v3_cool_z1"][1])
        cfg_relaxed["wb_v3_cool_gw"] = (float("inf"), cfg_relaxed["wb_v3_cool_gw"][1])

        res_rel = noref.estimate_wb_v3(px, cfg=cfg_relaxed)
        cool_cast_rel = res_rel["diagnostics"].get("cool_cast", False)

        # Check eligibility:
        # High blue: B/G in zones > 0.20 AND > 0.25 AND > 0.40 (all three places high)
        high_b_3way = bool(b_z0 > 0.20 and b_z1 > 0.25 and b_gw > 0.40)
        # R not low: blocked from default cool_cast because R is not suppressed
        r_not_low = bool(r_z0 > -0.05 or r_z1 > -0.10 or r_gw > -0.20)
        cool_cast_toggled = bool(not cool_cast_def and cool_cast_rel)

        record = {
            "id": fid,
            "domain": f_info["domain"],
            "roll": f_info["roll"],
            "raw": f_info["raw"],
            "ref": f_info["ref"],
            "cool_cast_default": cool_cast_def,
            "cool_cast_relaxed": cool_cast_rel,
            "b_z0": float(b_z0) if z0_c else None,
            "b_z1": float(b_z1) if z1_c else None,
            "b_gw": float(b_gw) if est_gw is not None else None,
            "r_z0": float(r_z0) if z0_c else None,
            "r_z1": float(r_z1) if z1_c else None,
            "r_gw": float(r_gw) if est_gw is not None else None,
            "high_b_3way": high_b_3way,
            "r_not_low": r_not_low,
            "cool_cast_toggled": cool_cast_toggled,
        }
        candidates.append(record)

    return candidates


def run_subset_eval(subset_ids: list[str], manifest_path: Path, out_json: Path, variants: dict):
    """Run t4d_source_eval.py with baseline vs relaxed variants."""
    ids_file = out_json.with_suffix(".ids.txt")
    ids_file.write_text("\n".join(sorted(subset_ids)) + "\n")

    cmd = [
        str(ROOT / ".venv/bin/python"),
        "-I",
        str(EVAL_SCRIPT),
        str(SRC),
        str(out_json),
        f"--ids-from={ids_file}",
        f"--manifest={manifest_path}",
        "--variants=" + json.dumps(variants),
    ]
    env = dict(os.environ, FILMCOLOR_DATA_ROOT="/Volumes/waizhiyingpan/filmcolor_data")
    log_file = out_json.with_suffix(".log")
    with open(log_file, "w") as log:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    source_sha = get_source_sha256()

    print(f"=== T4z Blue Cast Diagnostic (5.4) ===")
    print(f"Source SHA256: {source_sha}")
    print(f"Production Frozen SHA256: {PROD_FROZEN_SHA}")
    assert source_sha == PROD_FROZEN_SHA, f"Source SHA mismatch: {source_sha} != {PROD_FROZEN_SHA}"

    # 1. Load valid308 protocol frames
    all_valid = load_valid308_frames()
    print(f"Valid308 protocol frames loaded: {len(all_valid)}")
    assert len(all_valid) == 308, f"Expected 308 valid frames, got {len(all_valid)}"

    # 2. Locate d_hi_B / d_hi_G < 0.44
    low_ratio = filter_low_ratio_frames(all_valid)
    print(f"Frames with d_hi_B / d_hi_G < 0.44: {len(low_ratio)}")
    assert len(low_ratio) == 114, f"Expected 114 low ratio frames, got {len(low_ratio)}"

    # Load manifests for raw/ref mapping
    m1 = json.loads((ROOT / "data/manifests/s3_dev_frames.json").read_text())["frames"]
    m2 = json.loads((ROOT / "data/manifests/bn_rb_frames.json").read_text())["frames"]
    manifest_map = {f["id"]: f for f in m1 + m2}

    # 3. Fast scan using existing artifact or fresh WB scan
    scan_jsonl_path = Path("/tmp/wb_scan_clean.jsonl")
    scan_cache_path = Path("/tmp/low_ratio_wb_scan.json")
    scan_records = []

    if scan_jsonl_path.exists():
        raw_lines = [json.loads(l) for l in scan_jsonl_path.read_text().splitlines() if l.strip()]
        if len(raw_lines) == len(low_ratio) and {r["fid"] for r in raw_lines} == set(low_ratio.keys()):
            print(f"Loading complete clean scan ({len(raw_lines)} frames) from {scan_jsonl_path}...")
            for r in raw_lines:
                fid = r["fid"]
                b_z0 = r.get("b_z0") or -999.0
                b_z1 = r.get("b_z1") or -999.0
                b_gw = r.get("b_gw") or -999.0
                r_z0 = r.get("r_z0") or 999.0
                r_z1 = r.get("r_z1") or 999.0
                r_gw = r.get("r_gw") or 999.0
                cool_def = r.get("cool_cast", False)
                cool_rel = r.get("cool_rel", False)
                high_b_3way = bool(b_z0 > 0.20 and b_z1 > 0.25 and b_gw > 0.40)
                r_not_low = bool(r_z0 > -0.05 or r_z1 > -0.10 or r_gw > -0.20)
                cool_cast_toggled = bool(not cool_def and cool_rel)
                scan_records.append({
                    "id": fid,
                    "cool_cast_default": cool_def,
                    "cool_cast_relaxed": cool_rel,
                    "b_z0": b_z0, "b_z1": b_z1, "b_gw": b_gw,
                    "r_z0": r_z0, "r_z1": r_z1, "r_gw": r_gw,
                    "high_b_3way": high_b_3way,
                    "r_not_low": r_not_low,
                    "cool_cast_toggled": cool_cast_toggled,
                })

    if not scan_records:
        print("Running lightweight WB zone inspection on 114 frames...")
        scan_records = inspect_wb_zones_and_filter(low_ratio, manifest_map)

    # Filter actual target candidates:
    # 1. cool_cast_default is False
    # 2. cool_cast_toggled is True (actual effect under relaxed R)
    # 3. High blue tendency (log2 B/G in zones > 0.20/0.25/0.40) and R not low
    target_candidates = []
    strict_3way_candidates = []
    for rec in scan_records:
        fid = rec["id"]
        if rec.get("cool_cast_default") is True:
            continue
        if rec.get("cool_cast_toggled"):
            target_candidates.append(fid)
            if rec.get("high_b_3way") and rec.get("r_not_low"):
                strict_3way_candidates.append(fid)

    print(f"Target candidate subset selected: {len(target_candidates)} frames with cool_cast toggled (strict 3-way AND: {len(strict_3way_candidates)})")

    # 4. Construct variants for counterfactual comparison
    # baseline: standard preset
    # relaxed: standard preset + override cfg (wb_v3_cool_z0/z1/gw R-threshold = +inf, B kept)
    variants = {
        "standard": {"preset": "standard"},
        "relaxed_r": {
            "preset": "standard",
            "cfg": {
                "wb_v3_cool_z0": [1e9, 0.05],
                "wb_v3_cool_z1": [1e9, 0.10],
                "wb_v3_cool_gw": [1e9, 0.25],
            },
        },
    }

    # Partition target candidates by manifest
    m1_ids = [fid for fid in target_candidates if fid in {f["id"] for f in m1}]
    m2_ids = [fid for fid in target_candidates if fid in {f["id"] for f in m2}]

    eval_out_json = OUT_DIR / "eval_blue_cast_relaxation.json"
    print(f"Running subset dE evaluation on {len(target_candidates)} frames...")

    results_frames = {}
    if m2_ids:
        tmp_m2 = OUT_DIR / "temp_blue_cross.json"
        run_subset_eval(m2_ids, ROOT / "data/manifests/bn_rb_frames.json", tmp_m2, variants)
        if tmp_m2.exists():
            results_frames.update(json.loads(tmp_m2.read_text()).get("frames", {}))
    if m1_ids:
        tmp_m1 = OUT_DIR / "temp_blue_dev.json"
        run_subset_eval(m1_ids, ROOT / "data/manifests/s3_dev_frames.json", tmp_m1, variants)
        if tmp_m1.exists():
            results_frames.update(json.loads(tmp_m1.read_text()).get("frames", {}))

    eval_out_json.write_text(json.dumps({
        "src": str(SRC),
        "source_sha256": source_sha,
        "variants": variants,
        "frames": results_frames,
    }, indent=2))
    print(f"Subset evaluation complete: {len(results_frames)} frames recorded in {eval_out_json}")

    # 5. Compute dE shift statistics
    def compute_stats(frame_dict, candidate_id_list):
        better_005 = 0
        worse_005 = 0
        better_025 = 0
        worse_025 = 0
        db_before_list = []
        db_after_list = []
        de_before_list = []
        de_after_list = []
        for fid in candidate_id_list:
            f = frame_dict.get(fid)
            if not f:
                continue
            v_std = f.get("variants", {}).get("standard", {})
            v_rel = f.get("variants", {}).get("relaxed_r", {})
            if not v_std or not v_rel:
                continue
            m_std = v_std.get("metrics", {})
            m_rel = v_rel.get("metrics", {})
            de_std = m_std.get("de2000_median")
            de_rel = m_rel.get("de2000_median")
            db_std = m_std.get("mean_db")
            db_rel = m_rel.get("mean_db")
            if de_std is not None and de_rel is not None:
                delta_de = de_rel - de_std
                de_before_list.append(de_std)
                de_after_list.append(de_rel)
                if delta_de < -0.05:
                    better_005 += 1
                elif delta_de > 0.05:
                    worse_005 += 1
                if delta_de < -0.25:
                    better_025 += 1
                elif delta_de > 0.25:
                    worse_025 += 1
            if db_std is not None and db_rel is not None:
                db_before_list.append(db_std)
                db_after_list.append(db_rel)
        return {
            "n_evaluated": len(de_before_list),
            "better_delta_de_gt_0_05": better_005,
            "worse_delta_de_gt_0_05": worse_005,
            "better_delta_de_gt_0_25": better_025,
            "worse_delta_de_gt_0_25": worse_025,
            "mean_de2000_before": float(np.mean(de_before_list)) if de_before_list else None,
            "mean_de2000_after": float(np.mean(de_after_list)) if de_after_list else None,
            "mean_db_before": float(np.mean(db_before_list)) if db_before_list else None,
            "mean_db_after": float(np.mean(db_after_list)) if db_after_list else None,
        }

    stats_all_toggled = compute_stats(results_frames, target_candidates)
    stats_strict_3way = compute_stats(results_frames, strict_3way_candidates)

    per_frame_details = {}
    for fid, f in results_frames.items():
        v_std = f.get("variants", {}).get("standard", {})
        v_rel = f.get("variants", {}).get("relaxed_r", {})
        if not v_std or not v_rel:
            continue
        m_std = v_std.get("metrics", {})
        m_rel = v_rel.get("metrics", {})
        de_std = m_std.get("de2000_median")
        de_rel = m_rel.get("de2000_median")
        db_std = m_std.get("mean_db")
        db_rel = m_rel.get("mean_db")

        per_frame_details[fid] = {
            "raw_sha256": f.get("raw_sha256"),
            "ref_sha256": f.get("ref_sha256"),
            "d_hi": f.get("d_hi"),
            "is_strict_3way": bool(fid in strict_3way_candidates),
            "standard": {
                "params": v_std.get("params"),
                "metrics": m_std,
                "wb_cool_cast": v_std.get("wb_cool_cast"),
            },
            "relaxed_r": {
                "params": v_rel.get("params"),
                "metrics": m_rel,
                "wb_cool_cast": v_rel.get("wb_cool_cast"),
            },
            "delta_de2000_median": (de_rel - de_std) if (de_std is not None and de_rel is not None) else None,
            "delta_mean_db": (db_rel - db_std) if (db_std is not None and db_rel is not None) else None,
        }

    summary_report = {
        "task": "T4z Section 5.4 Blue Cast Relaxation Diagnostic",
        "provenance": {
            "source_dir": str(SRC),
            "source_sha256": source_sha,
            "production_frozen_sha256": PROD_FROZEN_SHA,
            "source_frozen_verified": bool(source_sha == PROD_FROZEN_SHA),
            "protocol": "valid308_benchmark",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "selection": {
            "valid308_count": len(all_valid),
            "low_ratio_count": len(low_ratio),
            "low_ratio_threshold": "d_hi_B / d_hi_G < 0.44",
            "criteria_relaxed": "wb zones log2(B/G) > 0.05 (z0) / 0.10 (z1) / 0.25 (gw), R relaxed to +inf, cool_cast toggled False -> True",
            "criteria_strict_3way": "wb zones log2(B/G) > 0.20 (z0) AND > 0.25 (z1) AND > 0.40 (gw), R not low, cool_cast toggled False -> True",
            "all_toggled_subset_count": len(target_candidates),
            "all_toggled_candidate_ids": sorted(target_candidates),
            "strict_3way_subset_count": len(strict_3way_candidates),
            "strict_3way_candidate_ids": sorted(strict_3way_candidates),
        },
        "statistics_strict_3way": stats_strict_3way,
        "statistics_all_toggled": stats_all_toggled,
        "frames": per_frame_details,
    }

    final_diag_path = OUT_DIR / "blue_cast_diagnostic.json"
    final_diag_path.write_text(json.dumps(summary_report, indent=2))
    print(f"Summary diagnostic saved to {final_diag_path}")

    # Also update diagnostics_p2.json with full diag_4 summary
    diag_p2_path = OUT_DIR / "diagnostics_p2.json"
    if diag_p2_path.exists():
        try:
            diag_p2 = json.loads(diag_p2_path.read_text())
            diag_p2["diag_4_blue_cast_relaxation"] = {
                "task": "T4z Section 5.4 Blue Cast Relaxation Diagnostic",
                "source_sha256": source_sha,
                "counts": {
                    "valid308_total": len(all_valid),
                    "low_ratio_total": len(low_ratio),
                    "all_toggled_subset": len(target_candidates),
                    "strict_3way_subset": len(strict_3way_candidates),
                },
                "statistics_strict_3way": stats_strict_3way,
                "statistics_all_toggled": stats_all_toggled,
                "target_ids_strict_3way": sorted(strict_3way_candidates),
                "all_toggled_ids": sorted(target_candidates),
                "files": {
                    "diagnostic_json": "experiments/t4_auto_ui/t4z/blue_cast_diagnostic.json",
                    "eval_json": "experiments/t4_auto_ui/t4z/eval_blue_cast_relaxation.json",
                },
            }
            diag_p2_path.write_text(json.dumps(diag_p2, indent=2))
            print(f"Updated diag_4 in {diag_p2_path}")
        except Exception as e:
            print(f"Warning: could not update diagnostics_p2.json: {e}")

    print(f"Execution finished in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
