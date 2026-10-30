"""Make a local before/after device sheet from T4m probe JPEGs (no original reads).
Usage: python -I scripts/t4m_device_sheet.py
JPEGs are evidence for local inspection only, not staged.
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "experiments/t4_auto_ui/t4m"
NAMES = ("A1089.DNG", "A322.cr2", "A223.NEF", "A224.NEF", "DSC_3827.NEF")


def main():
    records = {n:json.loads((OUT / f"devices_{n}.json").read_text())["samples"] for n in ("baseline","candidate")}
    cell_w, cell_h = 380, 310
    image = Image.new("RGB",(cell_w*len(NAMES),cell_h*2),(28,28,28))
    draw = ImageDraw.Draw(image)
    for row, version in enumerate(("baseline","candidate")):
        for col, name in enumerate(NAMES):
            rec = records[version][name]
            x,y = col*cell_w,row*cell_h
            draw.text((x+10,y+8),f"{version}: {name}",fill="white")
            p = OUT / rec.get("preview_image", "missing.jpg")
            if p.exists():
                with Image.open(p) as src:
                    tile = src.convert("RGB")
                    tile.thumbnail((cell_w-10,cell_h-60))
                    image.paste(tile,(x+(cell_w-tile.width)//2,y+35+(cell_h-60-tile.height)//2))
            else:
                draw.text((x+10,y+120),"open error (T4l)",fill=(255,180,120))
            label = f"EV {rec['params']['ev']:+.3f}" if rec.get("params") else "unsupported RGBI"
            draw.text((x+10,y+cell_h-20),label,fill="white")
    image.save(OUT / "devices_before_after.jpg",quality=92)
    print(OUT / "devices_before_after.jpg")


if __name__ == "__main__":
    main()
