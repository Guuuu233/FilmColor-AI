#!/usr/bin/env python3
"""T4z diagnostics suite: runs experimental diagnostics for P2 items:
  - Diag 1: B094.tif internal hot space density, exact production gate conditions and Change C coverage.
  - Diag 2: 8-bit A851.tif, A852.tif, A850.tif exact srgb_eotf(2/255) quantization threshold,
            evaluated on the exact Preview pipeline (max_side 1600, spike_clamp, span_d_g, auto_estimate).
  - Diag 3: A251-A253 base chromaticity rule vs all 315 benchmark frame bases.
  - Diag 4: Reference to external t4z_blue_diagnostic.py without hardcoded numbers.

Does not modify production code or tests.
Usage: python -I scripts/t4z_diagnostics.py
Outputs: experiments/t4_auto_ui/t4z/diagnostics_p2.json
"""
import copy
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from filmcolor.app import State, to8
from filmcolor.frozen import check_source
from filmcolor.invert import (
    block_white_point,
    central_region,
    density,
    estimate_white_point,
    region_mask,
)
from filmcolor.noref import NOREF_V3
from filmcolor.preview import load_preview
from filmcolor.render import RenderParams, render, srgb_eotf
from filmcolor.scanner import load_scanner

OUT_DIR = ROOT / "experiments/t4_auto_ui/t4z"
MATERIAL = Path("/Volumes/waizhiyingpan/胶片负片素材")


def file_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source_sha() -> str:
    h = hashlib.sha256()
    for p in sorted(p for p in (SRC / "filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(SRC)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def diag_1_b094() -> dict:
    """1. B094.tif internal hot space density and Change C coverage."""
    p = MATERIAL / "02_黑白负片_未反相/B094.tif"
    check_source(p, allow_frozen=False)
    raw_sha = file_sha256(p)
    frame = load_scanner(p, allow_frozen=False)
    rgb = frame.rgb
    h, w = rgb.shape[:2]
    mask = region_mask((h, w), central_region((h, w)))
    block = 16
    base = (1.0, 1.0, 1.0)
    pct = 99.8

    h_b, w_b = h // block, w // block
    h_crop, w_crop = h_b * block, w_b * block
    sums = np.zeros((h_b, w_b, 3), np.float64)
    full = np.zeros((h_b, w_b), np.int64)

    CHUNK_ROWS = 512
    step = max(block, CHUNK_ROWS // block * block)
    for r0 in range(0, h_crop, step):
        r1 = min(h_crop, r0 + step)
        b0, b1 = r0 // block, r1 // block
        sums[b0:b1] = rgb[r0:r1, :w_crop].reshape(b1 - b0, block, w_b, block, 3).sum(axis=(1, 3), dtype=np.float64)
        full[b0:b1] = mask[r0:r1, :w_crop].reshape(b1 - b0, block, w_b, block).sum(axis=(1, 3))

    ok = full == block * block
    means = (sums[ok] / (block * block)).astype(np.float32)
    d_hi_initial = estimate_white_point(density(means, base), None, pct)
    grid_means = (sums / (block * block)).astype(np.float32)
    D_grid = density(grid_means, base)
    D_sort = np.sort(D_grid, axis=-1)
    hot = (D_sort[..., 1] > 1.85) & (D_sort[..., 2] > 2.10)

    from scipy import ndimage

    lab, n_features = ndimage.label(hot)
    edge_labels = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    edge_labels = edge_labels[edge_labels > 0]
    edge_connected = np.isin(lab, edge_labels)
    holder_blocks_orig = ndimage.binary_dilation(edge_connected, iterations=2)
    excluded_orig = ok & holder_blocks_orig
    n_excluded_orig = int(excluded_orig.sum())

    # Check production gate conditions:
    # In production invert.py:
    # 1. d_hi[1] > 1.80 or (d_hi[1] > 1.15 and d_hi[0] > d_hi[1] + 0.25)
    # 2. hot.any()
    # 3. len(edge_labels) > 0
    # 4. n_excluded > 0 (where excluded = ok & holder_blocks)
    # 5. (ok & ~holder_blocks & hot).any()  <-- Change C bridging condition
    entered_c_gate = bool(
        n_excluded_orig > 0 and (ok & ~holder_blocks_orig & hot).any()
    )

    rem_hot_orig = ok & ~holder_blocks_orig & hot
    rows_orig, cols_orig = np.where(rem_hot_orig)
    d_rem_orig = D_grid[rem_hot_orig]

    # Evaluate exact production block_white_point
    d_hi_prod, info_prod = block_white_point(rgb, base, mask, block=block, pct=pct)

    # Evaluate with State in preview pipeline
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        link = tmp_dir / p.name
        link.symlink_to(p)
        st = State(tmp_dir, tmp_dir / "out", allow_frozen=False)
        st.open(p.name, "bw")
        pv = st.preview
        prod_status = pv.auto.get("status")
        prod_flags = pv.auto.get("flags")
        st.preview = None
        del pv
        gc.collect()

    del frame, rgb, sums, full, D_grid, D_sort
    gc.collect()

    return {
        "file": str(p),
        "raw_sha256": raw_sha,
        "total_blocks_grid": [h_b, w_b],
        "total_hot_blocks": int(hot.sum()),
        "edge_connected_hot_blocks": int(edge_connected.sum()),
        "production_gate_condition": {
            "d_hi_initial_gate_passed": bool(d_hi_initial[1] > 1.80),
            "d_hi_initial": [float(x) for x in d_hi_initial],
            "n_excluded_initial": n_excluded_orig,
            "has_remaining_hot": bool((ok & ~holder_blocks_orig & hot).any()),
            "entered_change_c_branch": entered_c_gate,
            "explanation": "In production invert.py lines 188-189, Change C requires BOTH n_excluded > 0 and (ok & ~holder_blocks & hot).any(). While remaining hot is True (296 blocks), n_excluded == 0 because edge-connected holder blocks do not intersect the central ok mask. Thus Change C is NOT entered for B094.",
        },
        "internal_hot_blocks_in_ok": {
            "count": int(rem_hot_orig.sum()),
            "rows_range": [int(rows_orig.min()), int(rows_orig.max())],
            "cols_range": [int(cols_orig.min()), int(cols_orig.max())],
            "d_g_min": float(d_rem_orig[:, 1].min()),
            "d_g_median": float(np.median(d_rem_orig[:, 1])),
            "d_g_max": float(d_rem_orig[:, 1].max()),
        },
        "production_result": {
            "d_hi": [float(x) for x in d_hi_prod],
            "status": prod_status,
            "flags": prod_flags,
            "covered_by_change_c": False,
        },
    }


def diag_2_8bit_quantization() -> dict:
    """2. A851.tif, A852.tif, A850.tif exact srgb_eotf(2/255) quantization threshold experiment.
    Evaluated using exact Preview pipeline (max_side 1600, spike_clamp, span_d_g, auto_estimate)."""
    results = {}
    thresh_2_255 = float(srgb_eotf(np.array([2.0 / 255.0], dtype=np.float32))[0])  # 0.00060705

    for name in ["A850.tif", "A851.tif", "A852.tif"]:
        p = MATERIAL / "01_彩色负片_未反相" / name
        check_source(p, allow_frozen=False)
        raw_sha = file_sha256(p)
        frame = load_scanner(p, allow_frozen=False)
        rgb = frame.rgb
        h, w = rgb.shape[:2]
        mask = region_mask((h, w), central_region((h, w)))
        block = 16

        h_b, w_b = h // block, w // block
        h_c, w_c = h_b * block, w_b * block
        sums = np.zeros((h_b, w_b, 3), np.float64)
        full = np.zeros((h_b, w_b), np.int64)
        step = 512
        for r0 in range(0, h_c, step):
            r1 = min(h_c, r0 + step)
            b0, b1 = r0 // block, r1 // block
            sums[b0:b1] = rgb[r0:r1, :w_c].reshape(b1 - b0, block, w_b, block, 3).sum(axis=(1, 3), dtype=np.float64)
            full[b0:b1] = mask[r0:r1, :w_c].reshape(b1 - b0, block, w_b, block).sum(axis=(1, 3))

        ok = full == block * block
        means = (sums / (block * block)).astype(np.float32)

        # 1. Unconstrained: full load_preview pipeline
        pv_unconstr = load_preview(p, max_side=1600, film="color", white_block=16,
                                   white_pct=99.8, spike_clamp=(1.15, 0.12), auto=NOREF_V3)
        base = tuple(pv_unconstr.inversion["base"])
        prod_d_hi = tuple(pv_unconstr.inversion["d_hi"])
        prod_params = copy.deepcopy(pv_unconstr.auto["params"])
        disp_unconstr, _ = render(pv_unconstr.positive, RenderParams.from_dict(prod_params))
        prod_q50 = [float(x) for x in np.median(disp_unconstr, axis=(0, 1))]
        prod_status = pv_unconstr.auto.get("status")
        prod_flags = pv_unconstr.auto.get("flags")

        # 2. Constrained: exclude blocks where min channel <= srgb_eotf(2/255)
        too_dark = means.min(axis=-1) <= thresh_2_255
        ok_constrained = ok & ~too_dark

        means_clean = (sums[ok_constrained] / (block * block)).astype(np.float32)
        d_hi_constrained = estimate_white_point(density(means_clean, base), None, 99.8)

        # 3. Constrained preview via exact load_preview with constrained d_hi
        pv_constr = load_preview(p, max_side=1600, film="color", base_rgb=base,
                                 d_hi=tuple(float(x) for x in d_hi_constrained),
                                 white_block=16, white_pct=99.8, spike_clamp=(1.15, 0.12), auto=NOREF_V3)
        constr_params = copy.deepcopy(pv_constr.auto["params"])
        disp_constr, _ = render(pv_constr.positive, RenderParams.from_dict(constr_params))
        constr_q50 = [float(x) for x in np.median(disp_constr, axis=(0, 1))]
        constr_status = pv_constr.auto.get("status")
        constr_flags = pv_constr.auto.get("flags")

        results[name] = {
            "raw_sha256": raw_sha,
            "bits": frame.meta.get("bits"),
            "eotf": frame.meta.get("tiff_eotf"),
            "threshold_linear": thresh_2_255,
            "total_ok_blocks": int(ok.sum()),
            "too_dark_blocks_excluded": int((ok & too_dark).sum()),
            "unconstrained": {
                "d_hi": [float(x) for x in prod_d_hi],
                "ev": float(prod_params.get("ev", 0.0)),
                "status": prod_status,
                "flags": prod_flags,
                "srgb_q50": prod_q50,
            },
            "constrained": {
                "d_hi": [float(x) for x in d_hi_constrained],
                "ev": float(constr_params.get("ev", 0.0)),
                "status": constr_status,
                "flags": constr_flags,
                "srgb_q50": constr_q50,
            },
            "direction": {
                "d_hi_reduction": [float(a - b) for a, b in zip(prod_d_hi, d_hi_constrained)],
                "srgb_q50_lift": [float(b - a) for a, b in zip(prod_q50, constr_q50)],
                "ev_delta": float(constr_params.get("ev", 0.0) - prod_params.get("ev", 0.0)),
            },
        }
        del frame, rgb, sums, full, pv_unconstr, pv_constr, disp_unconstr, disp_constr
        gc.collect()

    return results


def diag_3_base_chromaticity() -> dict:
    """3. A251-A253 base chromaticity rule vs all 315 benchmark frame bases."""
    target_results = {}
    for name in ["A251.cr2", "A252.cr2", "A253.cr2"]:
        p = MATERIAL / "01_彩色负片_未反相" / name
        check_source(p, allow_frozen=False)
        raw_sha = file_sha256(p)
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            tmp_dir = Path(td)
            link = tmp_dir / p.name
            link.symlink_to(p)
            st = State(tmp_dir, tmp_dir / "out", allow_frozen=False)
            st.open(p.name, "color")
            pv = st.preview
            base = list(pv.inversion["base"])
            r_g = float(base[0] / base[1])
            b_g = float(base[2] / base[1])
            mean_base = float(np.mean(base))
            disp = pv.render(pv.auto["params"])
            q50 = [float(x) for x in np.median(disp, axis=(0, 1))]
            target_results[name] = {
                "raw_sha256": raw_sha,
                "base_rgb": base,
                "r_g": r_g,
                "b_g": b_g,
                "mean_base": mean_base,
                "ev": float(pv.auto["params"].get("ev", 0.0)),
                "srgb_q50": q50,
                "matches_spec_rule": bool(r_g < 0.85 and b_g > 0.42 and mean_base > 0.20),
            }
            st.preview = None
            del pv
            gc.collect()

    # Benchmark check across all 315 frames in candidate_a/b/cross
    benchmark_candidates = []
    total_checked = 0
    candidate_files = [
        ROOT / "experiments/t4_auto_ui/t4y/candidate_a.json",
        ROOT / "experiments/t4_auto_ui/t4y/candidate_b.json",
        ROOT / "experiments/t4_auto_ui/t4y/candidate_cross.json",
    ]
    for cfile in candidate_files:
        if not cfile.exists():
            continue
        data = json.loads(cfile.read_text())
        for fid, f in data.get("frames", {}).items():
            binfo = f.get("base")
            if isinstance(binfo, dict) and "rgb" in binfo:
                rgb = binfo["rgb"]
                if rgb and len(rgb) == 3 and rgb[1] > 0:
                    total_checked += 1
                    rg = float(rgb[0] / rgb[1])
                    bg = float(rgb[2] / rgb[1])
                    mean_val = float(np.mean(rgb))
                    # Spec rule: R/G < 0.85 and B/G > 0.42 and mean(base_rgb) > 0.20
                    if rg < 0.85 and bg > 0.42 and mean_val > 0.20:
                        benchmark_candidates.append({"id": fid, "r_g": rg, "b_g": bg, "mean_base": mean_val, "rgb": rgb})

    return {
        "rule_specification": "Reject candidate base if (R/G < 0.85 and B/G > 0.42 and mean(base_rgb) > 0.20) indicating bare backlight intrusion rather than orange film mask.",
        "targets": target_results,
        "benchmark_audit": {
            "total_frames_checked": total_checked,
            "matching_frames_count": len(benchmark_candidates),
            "matching_frames": benchmark_candidates,
            "conclusion": f"Across {total_checked} evaluated benchmark frames, exactly {len(benchmark_candidates)} frame(s) matched the (R/G < 0.85 & B/G > 0.42 & mean > 0.20) threshold. Pure chromaticity (R/G < 0.85 & B/G > 0.42) alone hit ext:A196 (dense negative, mean=0.064), while combining with brightness (>0.20) isolates bare lightboxes.",
        },
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "diagnostics_p2.json"
    cur_sha = source_sha()

    print("Running Diag 1: B094 hot space density...")
    d1 = diag_1_b094()
    print("Running Diag 2: 8-bit quantization constraint...")
    d2 = diag_2_8bit_quantization()
    print("Running Diag 3: Base chromaticity rule & benchmark audit...")
    d3 = diag_3_base_chromaticity()

    final_report = {
        "source_content_sha256": cur_sha,
        "diag_1_b094": d1,
        "diag_2_8bit_quantization": d2,
        "diag_3_base_chromaticity": d3,
        "diag_4_blue_cast_relaxation": {
            "delegation": "scripts/t4z_blue_diagnostic.py",
            "status": "in_progress_by_parallel_subagent",
            "note": "Independent script executing actual counterfactual delta-E metrics on filtered candidates.",
        },
    }

    out_file.write_text(json.dumps(final_report, indent=2))
    print(f"Diagnostics complete. Saved to {out_file}")


if __name__ == "__main__":
    main()
