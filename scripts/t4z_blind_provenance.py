#!/usr/bin/env python3
"""T4z blind roll provenance and complete member audit suite.
Reads and verifies:
  1. All 143 raw files across the 3 blind rolls (A7R5: 6, Eastman: 99, Fomapan: 38).
     Pre-guards each with check_source(allow_frozen=False) and computes raw_sha256.
  2. Scripts and source hashes (source_content_sha256, scripts/t4z_blind_rolls.py sha256).
  3. Verifies checkpoint and final results of blind_rolls_eval.json / blind_rolls_checkpoint.json:
     - Exact length counts (A7R5: 6, Eastman: 10, Fomapan: 9)
     - Mode verification: single_frame has base_mode_requested=None, whole_roll has base_mode_requested='roll_nat'
     - Verified raw_sha256 matches between manifest and inference records using full resolved path mapping
     - Verified exact set equality of sampled paths against expected sample_indices
  4. Verifies all 6 generated contact sheet mosaics exist on disk and computes their SHA256.
  5. Strict overall 'complete' verification boolean requiring all 3 rolls, all booleans true, and all 6 mosaics present.

Usage: python -I scripts/t4z_blind_provenance.py
Outputs: experiments/t4_auto_ui/t4z/blind_provenance.json
"""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from filmcolor.frozen import check_source

OUT_DIR = ROOT / "experiments/t4_auto_ui/t4z"
MATERIAL = Path("/Volumes/waizhiyingpan/胶片负片素材")
EXPECTED_FREEZE_SHA = "136a7dbb951ed4c59497adfd2aa789ee0f180246f4e525c6822a78c9253a4756"


def file_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source_sha() -> str:
    h = hashlib.sha256()
    for p in sorted(p for p in (SRC / "filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(SRC)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def audit_all_blind_members():
    """Enumerate and verify all 143 physical raw files across the 3 blind rolls."""
    manifest = {}

    # 1. A7R5_RAW (6 ARWs)
    a7r5_dir = MATERIAL / "A7R5_RAW"
    a7r5_files = sorted(list(a7r5_dir.glob("*.ARW")) + list(a7r5_dir.glob("*.arw")))
    assert len(a7r5_files) == 6, f"Expected 6 A7R5 files, found {len(a7r5_files)}"
    a7r5_list = []
    for f in a7r5_files:
        check_source(f, allow_frozen=False)
        a7r5_list.append({
            "name": f.name,
            "path": str(f.resolve()),
            "size": f.stat().st_size,
            "raw_sha256": file_sha256(f),
        })
    manifest["A7R5_RAW"] = {
        "folder": str(a7r5_dir),
        "total_count": len(a7r5_list),
        "expected_count": 6,
        "sample_count": 6,
        "sample_indices": list(range(6)),
        "files": a7r5_list,
    }

    # 2. Eastman 5294 (99 NEFs across 3 sub-folders)
    eastman_subdirs = sorted(MATERIAL.glob("Eastman 5294(尼康数字化)-*/原件"))
    assert len(eastman_subdirs) == 3, f"Expected 3 Eastman sub-folders, found {len(eastman_subdirs)}"
    eastman_list = []
    for ed in eastman_subdirs:
        nefs = sorted(list(ed.glob("*.NEF")) + list(ed.glob("*.nef")))
        for f in nefs:
            check_source(f, allow_frozen=False)
            eastman_list.append({
                "name": f.name,
                "path": str(f.resolve()),
                "subfolder": ed.parent.name,
                "size": f.stat().st_size,
                "raw_sha256": file_sha256(f),
            })
    assert len(eastman_list) == 99, f"Expected 99 Eastman files, found {len(eastman_list)}"
    indices_e = [int(round(i * (len(eastman_list) - 1) / 9)) for i in range(10)]
    manifest["Eastman5294"] = {
        "subfolders": [str(d) for d in eastman_subdirs],
        "total_count": len(eastman_list),
        "expected_count": 99,
        "sample_count": 10,
        "sample_indices": indices_e,
        "files": eastman_list,
    }

    # 3. fomapan 100 (38 NEFs)
    foma_dir = MATERIAL / "foma pan100(d67 尼康数字化)-/原件"
    assert foma_dir.exists(), f"Fomapan directory missing: {foma_dir}"
    foma_files = sorted(list(foma_dir.glob("*.NEF")) + list(foma_dir.glob("*.nef")))
    assert len(foma_files) == 38, f"Expected 38 Fomapan files, found {len(foma_files)}"
    foma_list = []
    for f in foma_files:
        check_source(f, allow_frozen=False)
        foma_list.append({
            "name": f.name,
            "path": str(f.resolve()),
            "size": f.stat().st_size,
            "raw_sha256": file_sha256(f),
        })
    indices_f = [int(round(i * (len(foma_list) - 1) / 8)) for i in range(9)]
    manifest["fomapan100"] = {
        "folder": str(foma_dir),
        "total_count": len(foma_list),
        "expected_count": 38,
        "sample_count": 9,
        "sample_indices": indices_f,
        "files": foma_list,
    }

    return manifest


def verify_blind_results(manifest, cur_source_sha):
    """Audit the output JSON and mosaics against the physical manifest with full path mapping and strict completeness."""
    eval_file = OUT_DIR / "blind_rolls_eval.json"
    ckpt_file = OUT_DIR / "blind_rolls_checkpoint.json"

    target_data = None
    source_file = None
    target_file_sha256 = None
    is_eval_final = False

    if eval_file.exists():
        try:
            target_data = json.loads(eval_file.read_text())
            source_file = str(eval_file)
            target_file_sha256 = file_sha256(eval_file)
            is_eval_final = True
        except Exception:
            pass

    if target_data is None and ckpt_file.exists():
        try:
            raw_ckpt = json.loads(ckpt_file.read_text())
            target_data = raw_ckpt.get("results")
            source_file = str(ckpt_file)
            target_file_sha256 = file_sha256(ckpt_file)
        except Exception:
            pass

    verification = {
        "data_source_file": source_file,
        "data_file_sha256": target_file_sha256,
        "is_eval_final": is_eval_final,
        "rolls_status": {},
        "all_three_rolls_present": False,
        "all_checks_passed": False,
        "all_six_mosaics_exist": False,
        "complete": False,
    }

    if not target_data or "rolls" not in target_data:
        verification["status"] = "in_progress_no_completed_rolls_yet"
        return verification

    # Check target_data source SHA
    data_source_sha = target_data.get("source_content_sha256")
    verification["data_recorded_source_sha256"] = data_source_sha
    assert data_source_sha == EXPECTED_FREEZE_SHA, f"Data recorded source SHA ({data_source_sha}) != expected ({EXPECTED_FREEZE_SHA})"
    assert cur_source_sha == EXPECTED_FREEZE_SHA, f"Current source SHA ({cur_source_sha}) != expected ({EXPECTED_FREEZE_SHA})"

    rolls_data = target_data.get("rolls", {})
    expected_samples = {"A7R5_RAW": 6, "Eastman5294": 10, "fomapan100": 9}

    all_rolls_ok = True
    total_mosaics_present = 0

    for roll_name, exp_n in expected_samples.items():
        if roll_name not in rolls_data:
            all_rolls_ok = False
            verification["rolls_status"][roll_name] = {"present": False}
            continue

        rdata = rolls_data[roll_name]
        single_list = rdata.get("single_frame", [])
        whole_list = rdata.get("whole_roll", [])

        # Expected sample paths from manifest indices
        all_files = manifest[roll_name]["files"]
        sample_indices = manifest[roll_name]["sample_indices"]
        expected_paths = {str(Path(all_files[i]["path"]).resolve()) for i in sample_indices}
        path_to_hash = {str(Path(f["path"]).resolve()): f["raw_sha256"] for f in all_files}

        # Actual paths in results
        single_paths = {str(Path(item.get("full_path")).resolve()) for item in single_list if item.get("full_path")}
        whole_paths = {str(Path(item.get("full_path")).resolve()) for item in whole_list if item.get("full_path")}

        # Exact set equality
        single_paths_match_expected = (single_paths == expected_paths)
        whole_paths_match_expected = (whole_paths == expected_paths)

        # Check lengths
        single_len_ok = len(single_list) == exp_n
        whole_len_ok = len(whole_list) == exp_n

        # Check modes
        single_modes_ok = all(item.get("base_mode_requested") is None for item in single_list)
        whole_modes_ok = all(item.get("base_mode_requested") == "roll_nat" for item in whole_list)

        # Check raw hashes via full_path mapping (handles identical names across Eastman folders)
        single_hashes_ok = all(
            item.get("raw_sha256") == path_to_hash.get(str(Path(item.get("full_path")).resolve()))
            for item in single_list
        )
        whole_hashes_ok = all(
            item.get("raw_sha256") == path_to_hash.get(str(Path(item.get("full_path")).resolve()))
            for item in whole_list
        )

        # Contact sheet mosaics check
        sheet_s = OUT_DIR / f"blind_{roll_name}_single_mosaic.jpg"
        sheet_w = OUT_DIR / f"blind_{roll_name}_whole_mosaic.jpg"
        mosaics = {}
        s_exists = sheet_s.is_file()
        w_exists = sheet_w.is_file()
        if s_exists:
            mosaics["single_mosaic_sha256"] = file_sha256(sheet_s)
            total_mosaics_present += 1
        if w_exists:
            mosaics["whole_mosaic_sha256"] = file_sha256(sheet_w)
            total_mosaics_present += 1

        roll_passed = (
            single_len_ok and whole_len_ok and
            single_paths_match_expected and whole_paths_match_expected and
            single_modes_ok and whole_modes_ok and
            single_hashes_ok and whole_hashes_ok and
            s_exists and w_exists
        )
        if not roll_passed:
            all_rolls_ok = False

        verification["rolls_status"][roll_name] = {
            "present": True,
            "expected_sample_count": exp_n,
            "single_frame_count": len(single_list),
            "whole_roll_count": len(whole_list),
            "counts_match": single_len_ok and whole_len_ok,
            "single_paths_exact_set_match": single_paths_match_expected,
            "whole_paths_exact_set_match": whole_paths_match_expected,
            "single_mode_verified_none": single_modes_ok,
            "whole_mode_verified_roll_nat": whole_modes_ok,
            "single_raw_hashes_verified": single_hashes_ok,
            "whole_raw_hashes_verified": whole_hashes_ok,
            "single_mosaic_exists": s_exists,
            "whole_mosaic_exists": w_exists,
            "mosaics": mosaics,
            "roll_passed": roll_passed,
        }

    verification["all_three_rolls_present"] = set(rolls_data.keys()) >= set(expected_samples.keys())
    verification["all_checks_passed"] = all_rolls_ok
    verification["all_six_mosaics_exist"] = (total_mosaics_present == 6)
    verification["complete"] = (
        verification["all_three_rolls_present"] and
        verification["all_checks_passed"] and
        verification["all_six_mosaics_exist"] and
        is_eval_final
    )

    return verification


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cur_source_sha = source_sha()
    assert cur_source_sha == EXPECTED_FREEZE_SHA, f"Source SHA mismatch: {cur_source_sha} != {EXPECTED_FREEZE_SHA}"

    script_p = ROOT / "scripts/t4z_blind_rolls.py"
    blind_script_sha = file_sha256(script_p) if script_p.exists() else None

    print(f"Auditing blind rolls provenance...")
    print(f"  source_content_sha256: {cur_source_sha}")
    print(f"  scripts/t4z_blind_rolls.py sha256: {blind_script_sha}")

    print("Enumerating and pre-guarding all 143 raw files across 3 rolls...")
    manifest = audit_all_blind_members()

    print("Verifying checkpoint / evaluation results...")
    verification = verify_blind_results(manifest, cur_source_sha)

    report = {
        "source_content_sha256": cur_source_sha,
        "expected_freeze_sha256": EXPECTED_FREEZE_SHA,
        "source_sha_matches_freeze": True,
        "blind_script_sha256": blind_script_sha,
        "total_physical_raw_files_audited": 143,
        "rolls_manifest": manifest,
        "verification": verification,
    }

    out_file = OUT_DIR / "blind_provenance.json"
    out_file.write_text(json.dumps(report, indent=2))
    print(f"Audit complete! Provenance report saved to: {out_file}")
    print(f"Verification complete status: {verification['complete']}")


if __name__ == "__main__":
    main()
