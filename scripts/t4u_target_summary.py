"""Recompute real-target comparisons and record scope limitations without decoding images."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4u"


def main():
    records = {label: json.loads((OUT / f"devices_{label}.json").read_text()) for label in ("baseline", "candidate")}
    base, candidate = records["baseline"], records["candidate"]
    out = dict(source_sha256=candidate["src_content_sha256"], frames={})
    for name, c in candidate["targets"].items():
        b = base["targets"][name]
        assert c["raw_sha256"] == b["raw_sha256"]
        wb = c["diagnostics"].get("wb", {})
        out["frames"][name] = dict(raw_sha256=c["raw_sha256"], baseline_status=b["status"], status=c["status"],
            baseline_srgb_q50=b["srgb_q50"], srgb_q50=c["srgb_q50"],
            baseline_tint=b["params"]["tint"], tint=c["params"]["tint"],
            d_hi=c["inversion"]["d_hi"], no_narrow_tint_capped=wb.get("diagnostics", {}).get("no_narrow_tint_capped", False),
            narrow_blue_rescue=c["report"]["white_point"].get("blue_guard", {}).get("narrow_blue_rescue", False),
            upper_band_p99=c["upper_band_p99"])
    scope = json.loads((OUT / "a1296_scope.json").read_text())
    assert scope["source_sha256"] == candidate["src_content_sha256"]
    out["a1296"] = scope["regions"]
    out["limits"] = [
        "R-saturation decoupling changes sample eligibility; it cannot recover the clipped R signal.",
        "A1040 still has visible blue bias and grain; the green-median gate does not prove accurate colour.",
        "A1006 passes the whole-image B-G median gate but retains local yellow/blue colour differences.",
        "A1296 whole-image and center extreme values remain diagnostic; only the explicitly named regions and thresholds are accepted.",
    ]
    out["artifact_sha256"] = {f"devices_{label}.json": hashlib.sha256((OUT / f"devices_{label}.json").read_bytes()).hexdigest() for label in records}
    (OUT / "target_summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
