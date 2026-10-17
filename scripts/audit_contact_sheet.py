"""Contact sheets for checking negative -> print pairs by eye, from the audit thumbnails.

Each cell: the negative thumbnail (shown inverted, as a grey positive) next to its print, labelled with
both ids and the similarity. Pairs come from PAIRING_JSON; SELECT picks which:
  roll:<source dir>   every print of that roll with its best negative
  ids:R001,R002,...   the listed prints
  low:<margin>        prints whose best negative wins by less than <margin>

  audit_contact_sheet.py PAIRING_JSON INVENTORY_JSON THUMB_DIR SELECT OUT_JPG [COLS]
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

CELL_W, CELL_H, LABEL = 288, 192, 16


def select(pairs, inv, sel):
    kind, _, arg = sel.partition(":")
    if kind == "roll":
        return [p for p in pairs if inv[p["ref"]]["orig_dir"] == arg]
    if kind == "ids":
        want = arg.split(",")
        return [p for p in pairs if p["ref"] in want]
    if kind == "low":
        return [p for p in pairs if p["margin"] < float(arg)]
    raise SystemExit(f"unknown selection {sel}")


def cell(thumbs, p):
    out = Image.new("RGB", (2 * CELL_W, CELL_H + LABEL), "white")
    for k, (i, inv) in enumerate(((p["best"], True), (p["ref"], False))):
        f = Path(thumbs) / f"{i}.jpg"
        if f.exists():
            im = Image.open(f).convert("L" if inv else "RGB")
            out.paste((ImageOps.invert(im) if inv else im).convert("RGB").resize((CELL_W, CELL_H)), (k * CELL_W, LABEL))
    ImageDraw.Draw(out).text((4, 2), f"{p['best']} -> {p['ref']}  r={p['r']:.3f}  margin={p['margin']:.3f}"
                             f"  index={p.get('index_pair') or '-'}", fill="black")
    return out


def main(pairing, inventory, thumbs, sel, out_path, cols=3):
    pairs = json.load(open(pairing))["pairs"]
    inv = {r["id"]: r for r in json.load(open(inventory)) if r.get("id")}
    chosen = sorted(select(pairs, inv, sel), key=lambda p: p["ref"])
    cols = int(cols)
    rows = max(1, -(-len(chosen) // cols))
    sheet = Image.new("RGB", (cols * 2 * CELL_W, rows * (CELL_H + LABEL)), "white")
    for n, p in enumerate(chosen):
        sheet.paste(cell(thumbs, p), ((n % cols) * 2 * CELL_W, (n // cols) * (CELL_H + LABEL)))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=85)
    print(len(chosen), "pairs ->", out_path)


if __name__ == "__main__":
    main(*sys.argv[1:7])
