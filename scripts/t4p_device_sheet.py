"""Generate T4p local comparisons from derived probe images, not original material.
Usage: python -I scripts/t4p_device_sheet.py
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "experiments/t4_auto_ui/t4p"
SAMPLES = (("B086.dng", "bw"), ("B087.dng", "bw"), ("B091.RW2", "bw"), ("A1170.NEF", "color"), ("A1036.RW2", "color"))


def main():
    records = {(version, film): json.loads((OUT / f"devices_{version}_{film}.json").read_text())["samples"]
               for version in ("baseline", "candidate") for film in ("color", "bw")}
    image = Image.new("RGB", (400 * len(SAMPLES), 680), (25, 25, 25))
    draw = ImageDraw.Draw(image)
    for row, version in enumerate(("baseline", "candidate")):
        for col, (name, film) in enumerate(SAMPLES):
            rec = records[(version, film)][name]
            x, y = col * 400, row * 340
            draw.text((x + 8, y + 8), f"{version}: {name} ({film})", fill="white")
            p = OUT / rec.get("preview_image", "missing.jpg")
            if p.exists():
                with Image.open(p) as saved:
                    tile = saved.convert("RGB")
                    tile.thumbnail((390, 280))
                    image.paste(tile, (x + (400 - tile.width) // 2, y + 30 + (280 - tile.height) // 2))
            else:
                draw.text((x + 8, y + 150), (rec.get("error") or "preview unavailable")[:48], fill=(255, 160, 120))
            params = rec.get("params") or {}
            draw.text((x + 8, y + 318), f"EV {params.get('ev',0):+.3f} Temp {params.get('temp',0):+.3f}", fill="white")
    image.save(OUT / "targets_before_after.jpg", quality=92)
    print(OUT / "targets_before_after.jpg")


if __name__ == "__main__":
    main()
