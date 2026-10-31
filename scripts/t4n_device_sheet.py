"""Generate the local four-fault before/after sheet from saved probe JPEGs, not raw material.
Usage: python -I scripts/t4n_device_sheet.py
"""
import json
from pathlib import Path
from PIL import Image, ImageDraw

OUT=Path(__file__).resolve().parents[1]/"experiments/t4_auto_ui/t4n"
NAMES=("A1095.RAF","A277.dng","A281.RAF","A1005.CR2")


def main():
    records={version:json.loads((OUT/f"devices_{version}.json").read_text())["samples"] for version in ("baseline","candidate")}
    image=Image.new("RGB",(1600,680),(25,25,25))
    draw=ImageDraw.Draw(image)
    for row,version in enumerate(("baseline","candidate")):
        for col,name in enumerate(NAMES):
            rec=records[version][name]
            x,y=col*400,row*340
            draw.text((x+10,y+8),f"{version}: {name}",fill="white")
            path=OUT/rec.get("preview_image","missing.jpg")
            if path.exists():
                with Image.open(path) as saved:
                    tile=saved.convert("RGB")
                    tile.thumbnail((390,290))
                    image.paste(tile,(x+(400-tile.width)//2,y+32+(290-tile.height)//2))
            else:
                draw.text((x+15,y+160),rec.get("error","preview unavailable")[:48],fill=(255,160,120))
    image.save(OUT/"faults_before_after.jpg",quality=92)
    print(OUT/"faults_before_after.jpg")


if __name__=="__main__":
    main()
