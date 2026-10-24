"""Three-column before / after / reference sheet from t4b_render_frame.py outputs (T4b deliverables).

  t4b_sheets.py [--tile-h=PX] OUT_JPG BEFORE_DIR AFTER_DIR STEM=REF_JPG [STEM=REF_JPG ...]

``--tile-h``: tile height in pixels (default 520; the renders are 1000 px on their long side).

Each row: BEFORE_DIR/<stem>.png | AFTER_DIR/<stem>.png | the reference print. Our renders are turned upright to the
reference: by ``alignment.rot90`` of experiments/s3_ml/user_labels/<frame>.json when it exists (the reference turned
that many quarter turns counter-clockwise matches ours), else by the quarter turn whose luminance (64 px, aspect
matching the reference) correlates best with the reference. References are scaled with their own aspect, never
stretched. The turn used and its source are printed under each row. References are checked against the frozen
sets (filmcolor.frozen.check_source) before they are read.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from filmcolor.frozen import check_source

REPO = Path(__file__).resolve().parents[1]
TILE_H = 520
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def luma64(im: Image.Image) -> np.ndarray:
    a = np.asarray(im.convert("RGB").resize((64, 64), Image.BILINEAR), np.float32) @ LUMA
    return (a - a.mean()) / (a.std() + 1e-6)


def upright_turns(stem: str, ours: Image.Image, ref: Image.Image) -> tuple[int, str]:
    lab = REPO / "experiments/s3_ml/user_labels" / f"{stem.split('_')[0]}.json"
    if lab.exists():
        k = int(json.loads(lab.read_text())["alignment"]["rot90"])
        return (4 - k) % 4, "user_labels alignment"
    ref_portrait = ref.height > ref.width
    best = None
    for k in range(4):
        t = ours.rotate(90 * k, expand=True)
        if (t.height > t.width) != ref_portrait:
            continue
        score = float(np.mean(luma64(t) * luma64(ref)))
        best = max(best or (score, k), (score, k))
    return best[1], f"luminance correlation {best[0]:.2f}"


def tile(im: Image.Image, title: str, sub: str) -> Image.Image:
    w = int(round(im.width * TILE_H / im.height))
    out = Image.new("RGB", (w, TILE_H + 40), (22, 22, 25))
    out.paste(im.resize((w, TILE_H), Image.LANCZOS), (0, 40))
    d = ImageDraw.Draw(out)
    d.text((8, 4), title, fill=(235, 235, 240))
    d.text((8, 21), sub, fill=(170, 195, 220))
    return out


def params_line(info: dict) -> str:
    p = info["params"]
    keys = ("ev", "temp", "tint", "contrast", "shadows", "blacks", "highlights")
    return f"{info['base_method']} | " + " ".join(f"{k}={p.get(k, 0.0):+.2f}" for k in keys) + \
        " | d_hi=" + ",".join(f"{v:.3f}" for v in info["d_hi"])


def main(out_jpg: str, before_dir: str, after_dir: str, pairs: list[str]):
    rows = []
    for pair in pairs:
        stem, ref_path = pair.split("=", 1)
        check_source(Path(ref_path))
        ref = Image.open(ref_path).convert("RGB")
        cols = []
        k = src = None
        for label, d in (("BEFORE 4eb2678", before_dir), ("AFTER T4b", after_dir)):
            ours = Image.open(Path(d) / f"{stem}.png").convert("RGB")
            info = json.loads((Path(d) / f"{stem}.json").read_text())
            if k is None:
                k, src = upright_turns(stem, ours, ref)
            cols.append(tile(ours.rotate(90 * k, expand=True), f"{label}  {stem}", params_line(info)))
        name = Path(ref_path).name.replace("编辑", "edit")  # the default bitmap font has no CJK glyphs
        cols.append(tile(ref, f"REFERENCE  {name}", "reference print / user edit (not used by the estimator)"))
        w = sum(c.width for c in cols) + 8 * (len(cols) - 1)
        row = Image.new("RGB", (w, cols[0].height + 18), (15, 15, 18))
        x = 0
        for c in cols:
            row.paste(c, (x, 0))
            x += c.width + 8
        ImageDraw.Draw(row).text((8, cols[0].height + 3), f"turned {k} x 90 deg CCW for display ({src})", fill=(140, 140, 150))
        rows.append(row)
        print(stem, "turn", k, src, flush=True)
    W = max(r.width for r in rows)
    sheet = Image.new("RGB", (W, sum(r.height for r in rows)), (15, 15, 18))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    Path(out_jpg).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_jpg, quality=90)
    print("saved", out_jpg, sheet.size)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0].startswith("--tile-h="):
        TILE_H = int(args.pop(0).split("=", 1)[1])
    main(args[0], args[1], args[2], args[3:])
