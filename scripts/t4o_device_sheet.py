"""Create the local T4o before/after comparison from probe JPEGs, never original material.
Usage: python -I scripts/t4o_device_sheet.py
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "experiments/t4_auto_ui/t4o"
NAMES = ("A1195.CR2", "A240.CR3", "A328.CR3", "A329.ARW", "A099.NEF", "A1094.NEF", "A1170.NEF")


def main():
    records = {v: json.loads((OUT / f"devices_{v}.json").read_text())["samples"] for v in ("baseline", "candidate")}
    image = Image.new("RGB", (360 * len(NAMES), 640), (25, 25, 25))
    draw = ImageDraw.Draw(image)
    for row, version in enumerate(("baseline", "candidate")):
        for col, name in enumerate(NAMES):
            rec = records[version][name]
            x, y = col * 360, row * 320
            draw.text((x + 8, y + 8), f"{version}: {name}", fill="white")
            path = OUT / rec.get("preview_image", "missing.jpg")
            if path.exists():
                with Image.open(path) as saved:
                    tile = saved.convert("RGB")
                    tile.thumbnail((350, 260))
                    image.paste(tile, (x + (360 - tile.width) // 2, y + 30 + (260 - tile.height) // 2))
            else:
                draw.text((x + 8, y + 140), rec.get("error", "preview unavailable")[:45], fill=(255, 160, 120))
            p = rec.get("params") or {}
            draw.text((x + 8, y + 298), f"EV {p.get('ev',0):+.3f} Temp {p.get('temp',0):+.3f}", fill="white")
    image.save(OUT / "targets_before_after.jpg", quality=92)
    print(OUT / "targets_before_after.jpg")


if __name__ == "__main__":
    main()
