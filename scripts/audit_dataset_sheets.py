"""Contact sheets for the by-eye part of the dataset report (rebate visible? negative or positive?).

Read-only on ROOT; frozen files are never read. Selects frames of catalog.json by source (and optionally samples
PER_ROLL frames per roll, seed 0), draws numbered tiles, writes OUT_PREFIX_<n>.jpg and OUT_PREFIX.json (tile
number -> file) so a judgement made on the sheet can be traced back to a file.

  audit_dataset_sheets.py ROOT CATALOG_JSON OUT_PREFIX KIND SOURCES(comma) [PER_ROLL]
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from audit_dataset_pairing import load  # noqa: E402

TILE, COLS, PER_SHEET = 220, 8, 48


def main(root, catalog, prefix, kind, sources, per_roll=None):
    root = Path(root)
    c = json.loads(Path(catalog).read_text())
    files = {f["rel"]: f for f in c["files"]}
    sel = [fr for fr in c["frames"] if fr["kind"] == kind and fr["source"] in sources.split(",")]
    if per_roll:
        by = defaultdict(list)
        for fr in sel:
            by[fr["roll"] or fr["source"]].append(fr)
        rng = random.Random(0)
        sel = [x for k in sorted(by) for x in rng.sample(sorted(by[k], key=lambda f: f["frame"]), min(int(per_roll), len(by[k])))]
    rels = [next(r for r in fr["files"] if not files[r]["frozen"]) for fr in sel
            if any(not files[r]["frozen"] for r in fr["files"])]
    index = {}
    for s in range(0, len(rels), PER_SHEET):
        chunk = rels[s: s + PER_SHEET]
        img = Image.new("RGB", (COLS * TILE, ((len(chunk) + COLS - 1) // COLS) * (TILE + 14)), "white")
        d = ImageDraw.Draw(img)
        for i, rel in enumerate(chunk):
            n = s + i
            index[n] = rel
            try:
                t = load(root / rel)
            except Exception:  # noqa: BLE001 - leave the tile empty, the number still traces the file
                t = None
            x, y = (i % COLS) * TILE, (i // COLS) * (TILE + 14)
            if t is not None:
                t.thumbnail((TILE - 4, TILE - 4))
                img.paste(t.convert("RGB"), (x, y + 14))
            d.text((x + 2, y + 1), f"{n} {rel.rsplit('/', 1)[-1][:24]}", fill="black")
        img.save(f"{prefix}_{s // PER_SHEET}.jpg", quality=80)
    Path(f"{prefix}.json").write_text(json.dumps(index, ensure_ascii=False, indent=0))
    print(len(rels), "tiles,", (len(rels) + PER_SHEET - 1) // PER_SHEET, "sheets")


if __name__ == "__main__":
    main(*sys.argv[1:7])
