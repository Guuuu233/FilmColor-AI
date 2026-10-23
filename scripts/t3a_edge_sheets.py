"""T3a, missed rebate (task list 2026-10-23): contact sheets of every engine-readable development colour frame of the
own rolls and of BlueNeg, to check by eye whether film outside the picture (rebate, gap, sprocket area) is in the
scan. The 10-18 visual review saw only a sample (12 per own roll, 2 per BlueNeg roll); all frames are shown here.

  t3a_edge_sheets.py MATERIAL_ROOT T1_FRAMES_JSON OUT_PREFIX

Frames: T1 frames.json, partition dev, source own / blueneg, colour negative, engine-readable (frozen frames are
not in that selection and every file passes check_source before it is read). Tile image: the S3 1/8 cache when one
exists (load_cache), else the camera's embedded JPEG preview (own NEF) or the engine's scanner decode at 1/8
(BlueNeg DNG); shown as the negative, whole scan area, no crop. Writes OUT_PREFIX_<n>.jpg and OUT_PREFIX.json
(tile number -> frame, file, tile source).
"""
import glob
import io
import json
import sys
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image, ImageDraw

from filmcolor.compare import block_mean
from filmcolor.frozen import check_source, load_cache
from filmcolor.paths import data_dir
from filmcolor.scanner import load_scanner

REPO = Path(__file__).resolve().parents[1]
TILE, COLS, PER_SHEET = 300, 6, 36


def linear_tile(T):
    return Image.fromarray((np.clip(T / np.percentile(T, 99.9), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8))


def main(mat, t1_frames, prefix):
    mat = Path(mat)
    fr = json.loads(Path(t1_frames).read_text())
    sel = [r for r in fr if r["partition"] == "dev" and r["source"] in ("own", "blueneg")
           and r["kind"] == "colour_neg" and r["engine_readable"]]
    t8 = data_dir("s3_diag", "T8")
    dngs = {Path(p).stem: p for p in glob.glob(str(REPO / "data/raw/public/blueneg/**/*.dng"), recursive=True)}
    index, tiles = {}, []
    for n, r in enumerate(sel):
        name = r["id"].split(":", 1)[1].rsplit("/", 1)[-1]
        path = Path(dngs[name]) if r["source"] == "blueneg" else mat / r["files"][0]
        check_source(path)
        if (t8 / f"{name}_T8.npy").exists():
            img, how = linear_tile(load_cache(t8 / f"{name}_T8.npy").astype(np.float64)), "s3 T8 cache"
        elif r["source"] == "own":
            with rawpy.imread(str(path)) as raw:
                th = raw.extract_thumb()
            img, how = Image.open(io.BytesIO(th.data)).convert("RGB"), "embedded JPEG preview"
        else:
            img, how = linear_tile(block_mean(load_scanner(path).rgb, 8).astype(np.float64)), "scanner decode 1/8"
        img.thumbnail((TILE - 4, TILE - 4))
        tiles.append(img)
        index[n] = dict(frame=r["id"], roll=r["roll"], file=str(path.relative_to(REPO) if r["source"] == "blueneg"
                                                                  else r["files"][0]), tile=how)
        print(n, name, how, flush=True)
    for s in range(0, len(tiles), PER_SHEET):
        chunk = tiles[s: s + PER_SHEET]
        sheet = Image.new("RGB", (COLS * TILE, ((len(chunk) + COLS - 1) // COLS) * (TILE + 14)), "white")
        d = ImageDraw.Draw(sheet)
        for i, t in enumerate(chunk):
            x, y = (i % COLS) * TILE, (i // COLS) * (TILE + 14)
            sheet.paste(t, (x, y + 14))
            d.text((x + 2, y + 1), f"{s + i} {index[s + i]['frame'].rsplit('/', 1)[-1].split(':')[-1][:30]}",
                   fill="black")
        sheet.save(f"{prefix}_{s // PER_SHEET}.jpg", quality=82)
    Path(f"{prefix}.json").write_text(json.dumps(index, ensure_ascii=False, indent=0) + "\n")
    print(len(tiles), "tiles")


if __name__ == "__main__":
    main(*sys.argv[1:4])
