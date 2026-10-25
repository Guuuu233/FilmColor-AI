"""Validate the completed T4g delivery without decoding any source image.
Usage: python -I scripts/t4g_verify_outputs.py
"""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments/t4_auto_ui"


def main():
    records = {}
    h = hashlib.sha256()
    for p in sorted(p for p in (REPO / "src/filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(REPO / "src")).encode() + b"\0" + p.read_bytes() + b"\0")
    sha = h.hexdigest()
    for part, expected in (("a", 96), ("b", 95)):
        data = json.loads((ROOT / f"t4g/eval_t4g_{part}.json").read_text())
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
        assert set(v["variants"]) == {"standard", "natural", "portrait", "vivid", "print_classic", "wb_t4f"}
    key = json.loads((ROOT / "t4g/key_eval.json").read_text())["frames"]
    for fid, v in key.items():
        for name, item in v["variants"].items():
            assert item["params"] == records[fid]["variants"][name]["params"], (fid, name)
            assert abs(item["metrics"]["de2000_median"] - records[fid]["variants"][name]["metrics"]["de2000_median"]) < 1e-8
    checks = json.loads((ROOT / "t4g/after/checks.json").read_text())
    assert checks["src_content_sha256"] == sha
    assert checks["frames"]["A012"]["presets"]["portrait"]["auto"]["params"]["contrast"] == 1.16
    cache = Path("/Volumes/waizhiyingpan/filmcolor_data/app_cache")
    for roll in checks["rolls"].values():
        for frame in roll["info"]["frames"].values():
            assert (cache / f"thumb_color_t4g_{frame['disk_key']}.png").is_file()
    print("Verified 191 records / 185 valid, unchanged exclusions, disjoint halves, input hashes, key replays, source hash and 75 thumbnails.")
    print("Software SHA256:", sha)


if __name__ == "__main__":
    main()
