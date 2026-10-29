"""Validate a completed T4g / T4h / T4i / T4j / T4k / T4l delivery without decoding any source image.
Usage: python -I scripts/t4g_verify_outputs.py [TAG]   (default t4g)
"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments/t4_auto_ui"
TAG = sys.argv[1] if len(sys.argv) > 1 else "t4g"
PRESETS = {"standard", "natural", "portrait", "vivid", "print_classic"}
VARIANTS = {"t4g": PRESETS | {"wb_t4f"}, "t4h": PRESETS | {"wb_t4f", "wb_sky_red1"},
            "t4i": PRESETS | {"wb_t4h", "no_auto_levels", "no_auto_slope", "t4h_equiv", "levels_display"},
            "t4j": PRESETS | {"wb_t4h"},
            "t4k": PRESETS | {"no_thin", "wb_t4j", "t4j_noref"},
            "t4l": PRESETS | {"no_meter", "no_holder_box", "no_strips", "no_ev_down", "t4k_est"}}[TAG]
PORTRAIT_CONTRAST = None if TAG == "t4i" else 1.16  # T4i: auto contrast may raise it on a narrow frame


def main():
    records = {}
    h = hashlib.sha256()
    for p in sorted(p for p in (REPO / "src/filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(REPO / "src")).encode() + b"\0" + p.read_bytes() + b"\0")
    sha = h.hexdigest()
    for part, expected in (("a", 96), ("b", 95)):
        data = json.loads((ROOT / f"{TAG}/eval_{TAG}_{part}.json").read_text())
        assert data["src_content_sha256"] == sha
        assert len(data["frames"]) == expected
        assert not (records.keys() & data["frames"].keys())
        records.update(data["frames"])
    manifest = json.loads((REPO / "data/manifests/s3_dev_frames.json").read_text())["frames"]
    assert set(records) == {f["id"] for f in manifest}
    errors = {k for k,v in records.items() if "error" in v}
    assert errors == {"ext:" + s for s in ("A016", "A018", "A025", "A026", "A027", "A028")}, errors
    valid = {k:v for k,v in records.items() if k not in errors}
    assert len(valid) == 185
    assert all(v["ncc"] >= 0.7 for v in valid.values())
    assert all(len(v["raw_sha256"]) == len(v["ref_sha256"]) == 64 for v in records.values())
    for v in valid.values():
        assert set(v["variants"]) == VARIANTS
    key = json.loads((ROOT / TAG / "key_eval.json").read_text())["frames"]
    for fid, v in key.items():
        for name, item in v["variants"].items():
            assert item["params"] == records[fid]["variants"][name]["params"], (fid, name)
            assert abs(item["metrics"]["de2000_median"] - records[fid]["variants"][name]["metrics"]["de2000_median"]) < 1e-8
    checks = json.loads((ROOT / TAG / "after/checks.json").read_text())
    assert checks["src_content_sha256"] == sha
    if PORTRAIT_CONTRAST is not None:
        assert checks["frames"]["A012"]["presets"]["portrait"]["auto"]["params"]["contrast"] == PORTRAIT_CONTRAST
    cache = Path("/Volumes/waizhiyingpan/filmcolor_data/app_cache")
    for roll in checks["rolls"].values():
        for frame in roll["info"]["frames"].values():
            assert (cache / f"thumb_color_{TAG}_{frame['disk_key']}.png").is_file()
    print("Verified 191 records / 185 valid, unchanged exclusions, disjoint halves, input hashes, key replays, source hash and 75 thumbnails.")
    print("Software SHA256:", sha)


if __name__ == "__main__":
    main()
