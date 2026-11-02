"""Recompute failed-target explanations from probe evidence, without reading originals."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4t"


def main():
    candidate = json.loads((OUT / "devices_candidate.json").read_text())
    baseline = json.loads((OUT / "devices_baseline.json").read_text())
    a239 = candidate["targets"]["A239.ORF"]
    wb = a239["diagnostics"]["wb"]
    d = wb["diagnostics"]
    cfg = json.loads((ROOT / "configs/noref_v3.json").read_text())["config"]
    expected_raw = -sum(wb["chroma"]) * cfg["wb_v3_strength"] * wb["confidence"] / 2
    expected_wb = expected_raw * cfg["wb_v3_tint_boost"]
    expected_final = round(expected_wb + cfg["base_tint"] + cfg["tint_offset"], 4)
    assert abs(expected_raw - d["temp_tint_raw"][1]) < 1e-12
    assert abs(expected_wb - wb["tint"]) < 1e-12
    assert expected_final == a239["params"]["tint"]
    a1040 = candidate["targets"]["A1040.TIF"]
    out = dict(source_sha256=candidate["src_content_sha256"],
               probe_sha256=hashlib.sha256((OUT / "devices_candidate.json").read_bytes()).hexdigest(),
               a239=dict(bright_px_share=d["bright_px_share"], cyan_conf_boost=d.get("cyan_conf_boost", False),
                         confidence_without_boost=wb["confidence"], chroma=wb["chroma"],
                         tint_raw_recomputed=expected_raw, tint_wb_recomputed=expected_wb,
                         tint_final_recomputed=expected_final, task_tint_gate_lt=.22,
                         passes=expected_final < .22,
                         note="The specified bright-share gate removes the 0.72 floor, but does not cap the remaining confidence."),
               a1040=dict(baseline_white_point=baseline["targets"]["A1040.TIF"]["inversion"]["d_hi"],
                          candidate_white_point=a1040["inversion"]["d_hi"],
                          channel_cap=a1040["report"]["white_point"]["channel_cap"],
                          srgb_q50=a1040["srgb_q50"], task_green_q50_gate_gt=.05,
                          passes=a1040["srgb_q50"][1] > .05,
                          upper_band_p99=a1040["upper_band_p99"],
                          note="Capping the blue white point does not repair lost channels, the base estimate, or all out-of-mask zero clips."))
    (OUT / "target_diagnostics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
