"""Build a local before/after contact sheet from guarded probe derivatives and benchmark visuals. No original reads.
Usage: python -I scripts/t4y_visual_compare.py
The JPG remains untracked; quantitative image checks live in devices_*.json and benchmark_visuals_*.json.
Five rows: two new devices (A1061.CR2, A237.cr2) + blueneg:199807B-20-seattle, blueneg:19960811B-25, ext:A205.
"""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4y"
PROBE_NAMES = ("A1061.CR2", "A237.cr2")
BENCHMARK_IDS = ("blueneg:199807B-20-seattle", "blueneg:19960811B-25-arizona-london-bridge", "ext:A205")


def main():
    records = {label: json.loads((OUT / f"devices_{label}.json").read_text()) for label in ("baseline", "candidate")}
    benchmarks = {}
    for label in ("baseline", "candidate"):
        bp = OUT / f"benchmark_visuals_{label}.json"
        assert bp.exists(), f"missing benchmark visuals evidence: {bp}"
        benchmarks[label] = json.loads(bp.read_text())["frames"]
        assert set(benchmarks[label]) == set(BENCHMARK_IDS), f"mismatched benchmark ids in {bp}"

    rows = []
    for name in PROBE_NAMES:
        rows.append(("probe", name))
    for key in BENCHMARK_IDS:
        rows.append(("benchmark", key))

    sheet = Image.new("RGB", (1040, 360 * len(rows)), "#222222")
    draw = ImageDraw.Draw(sheet)

    for row_idx, (kind, item_id) in enumerate(rows):
        for col, label in enumerate(("baseline", "candidate")):
            x, y = col * 520, row_idx * 360
            if kind == "probe":
                name = item_id
                if name in records[label].get("new_targets", {}):
                    group = "new_targets"
                elif name in records[label].get("targets", {}):
                    group = "targets"
                elif name in records[label].get("new_tiff", {}):
                    group = "new_tiff"
                else:
                    group = "tiff"
                record = records[label][group][name]
                assert "error" not in record, f"probe record contains error for {name} ({label}): {record.get('error')}"
                assert record.get("status") == "ok", f"probe status not ok for {name} ({label}): {record.get('status')}"
                assert "preview_image" in record and "preview_image_sha256" in record, f"missing preview image in probe record for {name} ({label})"
                p = OUT / record["preview_image"]
                assert p.exists(), f"missing preview image file: {p}"
                assert hashlib.sha256(p.read_bytes()).hexdigest() == record["preview_image_sha256"]
                with Image.open(p) as opened:
                    image = opened.convert("RGB")
                image.thumbnail((510, 310))
                sheet.paste(image, (x + (520 - image.width) // 2, y + 42 + (310 - image.height) // 2))
                draw.text((x + 10, y + 5), f"{name}  {label}  {record['status']}", fill="white")
                draw.text((x + 10, y + 22), f"EV {record['params']['ev']:.4f}  Tint {record['params']['tint']:.4f}", fill="white")
            else:
                key = item_id
                b_rec = benchmarks[label][key]
                assert "preview_image" in b_rec and "preview_image_sha256" in b_rec, f"missing preview image in benchmark record for {key} ({label})"
                p = OUT / b_rec["preview_image"]
                assert p.exists(), f"missing benchmark image file: {p}"
                assert hashlib.sha256(p.read_bytes()).hexdigest() == b_rec["preview_image_sha256"]
                with Image.open(p) as opened:
                    image = opened.convert("RGB")
                image.thumbnail((510, 310))
                sheet.paste(image, (x + (520 - image.width) // 2, y + 42 + (310 - image.height) // 2))
                params = b_rec["params"]
                de = b_rec.get("de2000_median")
                de_str = f"  dE {de:.2f}" if de is not None else ""
                draw.text((x + 10, y + 5), f"{key}  {label}{de_str}", fill="white")
                draw.text((x + 10, y + 22), f"EV {params['ev']:.4f}  Tint {params['tint']:.4f}", fill="white")

    p = OUT / "before_after.jpg"
    sheet.save(p, quality=94)
    print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
