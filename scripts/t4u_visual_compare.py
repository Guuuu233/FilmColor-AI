"""Build a local before/after contact sheet from guarded probe derivatives. No original reads.
Usage: python -I scripts/t4u_visual_compare.py
The JPG remains untracked; quantitative image checks live in devices_*.json.
"""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4u"
NAMES = ("A305.tif", "A1040.TIF", "A1041.TIF", "A232.NEF", "A1069.NEF", "A239.ORF", "A281.RAF", "A1296.CR2", "A1006.CR2")


def main():
    records = {label: json.loads((OUT / f"devices_{label}.json").read_text()) for label in ("baseline", "candidate")}
    sheet = Image.new("RGB", (1040, 360 * len(NAMES)), "#222222")
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(NAMES):
        for col, label in enumerate(("baseline", "candidate")):
            record = records[label]["a1296"] if name == "A1296.CR2" else records[label]["targets"][name]
            p = OUT / record["preview_image"]
            assert hashlib.sha256(p.read_bytes()).hexdigest() == record["preview_image_sha256"]
            with Image.open(p) as opened:
                image = opened.convert("RGB")
            image.thumbnail((510, 310))
            x, y = col * 520, row * 360
            sheet.paste(image, (x + (520-image.width)//2, y + 42 + (310-image.height)//2))
            draw.text((x+10, y+5), f"{name}  {label}  {record['status']}", fill="white")
            draw.text((x+10, y+22), f"EV {record['params']['ev']:.4f}  Tint {record['params']['tint']:.4f}", fill="white")
    p = OUT / "before_after.jpg"
    sheet.save(p, quality=94)
    print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
