"""Build the local T5a before / after sheet (T4z left, T5a right) from guarded probe derivatives. No original reads.
Usage: python -I scripts/t5a_visual_compare.py
Rows: B096 (scene highlight released from the holder paint), B029 (real holder band kept), A850 and A229 (input
warnings, pixels unchanged). The JPG stays untracked; the checks live in devices_*.json.
"""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t5a"
ROWS = (("special", "B096.tif"), ("acros", "B029.NEF"), ("new_tiff", "A850.tif"), ("new_tiff", "A229.TIF"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    records = {label: json.loads((OUT / f"devices_{label}.json").read_text()) for label in ("baseline", "candidate")}
    sheet = Image.new("RGB", (1040, 380 * len(ROWS)), "#222222")
    draw = ImageDraw.Draw(sheet)
    out = {}
    for row, (group, name) in enumerate(ROWS):
        for col, label in enumerate(("baseline", "candidate")):
            r = records[label][group][name]
            assert "error" not in r and r["status"] == "ok", (label, name)
            p = OUT / r["preview_image"]
            assert sha(p) == r["preview_image_sha256"], (label, name)
            with Image.open(p) as opened:
                image = opened.convert("RGB")
            image.thumbnail((510, 310))
            x, y = col * 520, row * 380
            sheet.paste(image, (x + (520 - image.width) // 2, y + 62 + (310 - image.height) // 2))
            version = "T4z" if label == "baseline" else "T5a"
            draw.text((x + 10, y + 5), f"{name}  {version}  {r['status']}  holder {r['holder_share']:.3f}", fill="white")
            draw.text((x + 10, y + 22), f"EV {r['params']['ev']:.4f}  d_hi G {r['inversion']['d_hi'][1]:.3f}  "
                                        f"core q50 {[round(v, 3) for v in r['core_srgb_q50']]}", fill="white")
            warn = "; ".join(w["code"] for w in r.get("input_warnings") or []) or "-"
            draw.text((x + 10, y + 39), f"input_warnings: {warn}", fill="#ffb080" if warn != "-" else "white")
            out[f"{name}:{label}"] = r["preview_image_sha256"]
    p = OUT / "before_after.jpg"
    sheet.save(p, quality=94)
    print(p.relative_to(ROOT), sha(p))


if __name__ == "__main__":
    main()
