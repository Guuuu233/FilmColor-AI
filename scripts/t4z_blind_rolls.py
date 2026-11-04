#!/usr/bin/env python3
"""T4z blind roll evaluation and contact sheet generator.
Evaluates 3 unseen, non-frozen film rolls:
  1. A7R5_RAW (6 ARWs - all 6 frames sampled)
  2. Eastman 5294 (99 NEFs - 10 frames uniformly sampled)
  3. fomapan 100 (38 NEFs - 9 frames uniformly sampled)

Modes:
  a) Single-frame mode: isolated temporary directory per frame containing ONLY that frame's symlink.
  b) Whole-roll mode: folder-level State with all directory inputs pre-guarded, opened with base_mode="roll_nat".

Requirements:
  - Source SHA assertion against frozen fingerprint.
  - Pre-guard all inputs with check_source(path, allow_frozen=False).
  - Record raw_sha256 for every frame.
  - Per-frame checkpointing so partial progress is preserved.
  - Contact sheets with clearly rendered titles and mode markers on the canvas.

Usage: python -I scripts/t4z_blind_rolls.py
"""
import gc
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from filmcolor.app import State, to8
from filmcolor.frozen import check_source
from filmcolor.holder import film_box

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


def get_blind_rolls():
    # 1. A7R5_RAW (6 frames)
    a7r5_dir = MATERIAL / "A7R5_RAW"
    a7r5_files = sorted(list(a7r5_dir.glob("*.ARW")) + list(a7r5_dir.glob("*.arw")))
    assert len(a7r5_files) == 6, f"Expected 6 A7R5 frames, found {len(a7r5_files)}"
    a7r5_sample = [(f, "color", a7r5_dir) for f in a7r5_files]

    # 2. Eastman 5294 (99 NEFs across 3 sub-folders)
    eastman_dirs = sorted(MATERIAL.glob("Eastman 5294(尼康数字化)-*/原件"))
    assert len(eastman_dirs) == 3, f"Expected 3 Eastman sub-folders, found {len(eastman_dirs)}"
    eastman_all = []
    for ed in eastman_dirs:
        nefs = sorted(list(ed.glob("*.NEF")) + list(ed.glob("*.nef")))
        for f in nefs:
            eastman_all.append((f, "bw", ed))
    assert len(eastman_all) == 99, f"Expected 99 Eastman frames, found {len(eastman_all)}"
    # 10 uniformly spaced indices out of 99: [0, 11, 22, 33, 44, 55, 66, 77, 88, 98]
    indices_e = [int(round(i * (len(eastman_all) - 1) / 9)) for i in range(10)]
    eastman_sample = [eastman_all[i] for i in indices_e]

    # 3. fomapan 100 (38 NEFs)
    foma_dir = MATERIAL / "foma pan100(d67 尼康数字化)-/原件"
    assert foma_dir.exists(), f"Fomapan directory missing: {foma_dir}"
    foma_all = [(f, "bw", foma_dir) for f in sorted(list(foma_dir.glob("*.NEF")) + list(foma_dir.glob("*.nef")))]
    assert len(foma_all) == 38, f"Expected 38 Fomapan frames, found {len(foma_all)}"
    # 9 uniformly spaced indices out of 38: [0, 4, 9, 14, 18, 23, 27, 32, 37]
    indices_f = [int(round(i * (len(foma_all) - 1) / 8)) for i in range(9)]
    foma_sample = [foma_all[i] for i in indices_f]

    return {
        "A7R5_RAW": a7r5_sample,
        "Eastman5294": eastman_sample,
        "fomapan100": foma_sample,
    }


def process_frame(st, rel_name, film, base_mode=None):
    meta = st.open(rel_name, film, base_mode=base_mode)
    pv = st.preview
    disp = pv.render(pv.auto["params"])
    fb = film_box(pv.holder)
    core = disp if fb is None else disp[fb[0]:fb[0]+fb[2], fb[1]:fb[1]+fb[3]]
    record = {
        "rel": rel_name,
        "base_mode_requested": base_mode,
        "status": pv.auto.get("status"),
        "flags": pv.auto.get("flags"),
        "params": pv.auto.get("params"),
        "base": meta.get("base"),
        "d_hi": pv.inversion.get("d_hi"),
        "ev": float(pv.auto["params"].get("ev", 0.0)),
        "srgb_q50": [float(x) for x in np.median(disp, axis=(0, 1))],
        "core_srgb_q50": [float(x) for x in np.median(core, axis=(0, 1))],
    }
    img = Image.fromarray(to8(disp))
    img.thumbnail((300, 300))
    st.preview = None
    del pv, disp, core
    gc.collect()
    return record, img


def make_annotated_contact_sheet(images, titles, mode_label, roll_name, out_path, cols=5):
    """Draws a contact sheet with banner, frame titles and mode labels."""
    if not images:
        return
    n = len(images)
    rows = (n + cols - 1) // cols
    cell_w, cell_h = 320, 360
    header_h = 60
    sheet_w = cols * cell_w
    sheet_h = header_h + rows * cell_h

    sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)

    # Header banner
    banner_text = f"T4z Blind Roll: {roll_name} | Mode: {mode_label} | Sample: {n} frames | Source: {EXPECTED_FREEZE_SHA[:12]}"
    draw.text((20, 20), banner_text, fill=(240, 240, 240))

    for idx, (img, title) in enumerate(zip(images, titles)):
        r = idx // cols
        c = idx % cols
        cell_x = c * cell_w
        cell_y = header_h + r * cell_h

        # Center thumbnail in upper part of cell
        img_x = cell_x + (cell_w - img.width) // 2
        img_y = cell_y + 10 + (300 - img.height) // 2
        sheet.paste(img, (img_x, img_y))

        # Draw frame title and mode below image
        caption = f"{title} [{mode_label}]"
        draw.text((cell_x + 10, cell_y + 320), caption, fill=(200, 200, 200))

    sheet.save(out_path, quality=92)
    print(f"Saved contact sheet: {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cur_sha = source_sha()
    print(f"Checking source SHA-256: {cur_sha}")
    assert cur_sha == EXPECTED_FREEZE_SHA, f"Source SHA mismatch! Expected {EXPECTED_FREEZE_SHA}, got {cur_sha}"

    checkpoint_file = OUT_DIR / "blind_rolls_checkpoint.json"
    checkpoint = {}
    if checkpoint_file.exists():
        try:
            checkpoint = json.loads(checkpoint_file.read_text())
        except Exception:
            checkpoint = {}

    all_results = checkpoint.get("results", {
        "source_content_sha256": cur_sha,
        "rolls": {},
    })

    rolls = get_blind_rolls()

    # Pre-guard all roll inputs
    print("Pre-guarding all sample frames and directories with check_source...")
    for roll_name, sample in rolls.items():
        for p, _, _ in sample:
            check_source(p, allow_frozen=False)
    print("All sample frames passed check_source successfully.")

    for roll_name, sample in rolls.items():
        if roll_name in all_results["rolls"]:
            print(f"Roll {roll_name} already in checkpoint, checking completeness...")
            if "single_frame" in all_results["rolls"][roll_name] and "whole_roll" in all_results["rolls"][roll_name]:
                print(f"Skipping completed roll: {roll_name}")
                continue

        print(f"\n==========================================")
        print(f"Processing Roll: {roll_name} ({len(sample)} frames)")
        print(f"==========================================")
        roll_res = {"single_frame": [], "whole_roll": []}
        single_imgs, single_titles = [], []
        whole_imgs, whole_titles = [], []

        # --- A. Single-frame mode: isolated symlink per frame ---
        print("  Running Mode A: Single-frame (independent isolated State)...")
        for p, film, _ in sample:
            raw_sha = file_sha256(p)
            with tempfile.TemporaryDirectory() as td:
                tmp_dir = Path(td)
                link = tmp_dir / p.name
                link.symlink_to(p)
                st = State(tmp_dir, tmp_dir / "out", allow_frozen=False)
                rec, img = process_frame(st, p.name, film, base_mode=None)
                rec["raw_sha256"] = raw_sha
                rec["full_path"] = str(p)
                roll_res["single_frame"].append(rec)
                single_imgs.append(img)
                single_titles.append(p.name)

        # --- B. Whole-roll mode: folder State, base_mode='roll_nat' ---
        print("  Running Mode B: Whole-roll (roll State, base_mode='roll_nat')...")
        folder_groups = {}
        for p, film, parent_dir in sample:
            folder_groups.setdefault(parent_dir, []).append((p, film))

        for parent_dir, items in folder_groups.items():
            # Pre-guard all files in this parent directory before State instantiation
            for child in parent_dir.iterdir():
                if child.is_file() and child.suffix.lower() in (".nef", ".arw", ".cr2", ".tif"):
                    check_source(child, allow_frozen=False)
            with tempfile.TemporaryDirectory() as td:
                st = State(parent_dir, Path(td), allow_frozen=False)
                # Compute roll base and roll info across folder
                st.roll_info(".")
                for p, film in items:
                    raw_sha = file_sha256(p)
                    rec, img = process_frame(st, p.name, film, base_mode="roll_nat")
                    rec["raw_sha256"] = raw_sha
                    rec["full_path"] = str(p)
                    roll_res["whole_roll"].append(rec)
                    whole_imgs.append(img)
                    whole_titles.append(p.name)

        # Contact sheets with canvas titles
        sheet_single = OUT_DIR / f"blind_{roll_name}_single_mosaic.jpg"
        sheet_whole = OUT_DIR / f"blind_{roll_name}_whole_mosaic.jpg"
        make_annotated_contact_sheet(single_imgs, single_titles, "single_frame", roll_name, sheet_single)
        make_annotated_contact_sheet(whole_imgs, whole_titles, "roll_nat", roll_name, sheet_whole)

        roll_res["contact_sheet_single"] = sheet_single.name
        roll_res["contact_sheet_whole"] = sheet_whole.name
        all_results["rolls"][roll_name] = roll_res

        # Checkpoint save
        checkpoint_file.write_text(json.dumps({"results": all_results}, indent=2))
        print(f"Checkpoint saved for roll {roll_name}.")

    final_out = OUT_DIR / "blind_rolls_eval.json"
    final_out.write_text(json.dumps(all_results, indent=2))
    print(f"\nAll blind rolls complete! Final report saved to {final_out}")


if __name__ == "__main__":
    main()
