"""T3a step 1 (task list 2026-10-23): candidate frames with visible rebate, decode, grid previews (read-only).

  t3a_inspect.py MATERIAL_ROOT CATALOG_JSON VISUAL_REVIEW_JSON OUT_DIR

Candidates: the colour frames marked "rebate visible" in the 10-18 visual review (data/manifests/
dataset_visual_review_20261018.json, A7R5 included). Kept for measurement: colour negatives (not listed by that
review as not a negative, B&W or possibly reversal) whose file is a linear camera RAW or scanner DNG (catalog fmt
camera_raw / camera_dng); 8-bit TIFF and JPEG are listed with the reason and not decoded.
Each kept file: check_source (frozen guard), decoded as the engine's camera path does (rawpy_io.demosaic_linear,
raw.saturation_mask), 1/8 block-mean caches T8 / S8 under $FILMCOLOR_DATA_ROOT/t3_inversion/T8 (provenance
sidecar), and a grid preview OUT_DIR/previews/grids/<id>_grid.jpg (gamma-encoded negative, grid every 5 %, labels
every 10 %, saturated blocks magenta, blocks with G >= 95 % of the frame's maximum red) on which the rebate and
picture boxes are placed by eye before anything is measured. EXIF (exiftool: model, date, exposure, aperture, ISO,
lens) for grouping frames into sessions; the body serial is only used to form a code body-sha256(model|serial)[:10]
as in t1b_v3c_candidates.py, the serial itself is not written.
Writes OUT_DIR/candidates.json.
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from filmcolor.compare import block_mean
from filmcolor.frozen import check_source, save_cache
from filmcolor.paths import data_dir
from filmcolor.raw import saturation_mask
from filmcolor.rawpy_io import demosaic_linear, load_raw

LINEAR = ("camera_raw", "camera_dng")
SERIAL_KEYS = ("SerialNumber", "InternalSerialNumber", "BodySerialNumber", "CameraSerialNumber")
EXIF_KEYS = ("Model", "DateTimeOriginal", "ExposureTime", "FNumber", "ISO", "LensModel", "LensID")


def exif(paths):
    out = subprocess.run(["exiftool", "-json", "-n", "-q", "-q", *(f"-{k}" for k in EXIF_KEYS + SERIAL_KEYS),
                          *map(str, paths)], capture_output=True, text=True).stdout
    res = {}
    for d in json.loads(out or "[]"):
        s = next((str(d[k]) for k in SERIAL_KEYS if d.get(k) not in (None, "", 0)), None)
        res[d["SourceFile"]] = dict({k: d.get(k) for k in EXIF_KEYS},
                                    body=("body-" + hashlib.sha256(f"{d.get('Model')}|{s}".encode()).hexdigest()[:10])
                                    if s else None)
    return res


def grid_preview(T8, S8, path):
    v = (np.clip(T8 / np.percentile(T8, 99.9), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    v[T8[..., 1] >= 0.95 * T8[..., 1].max()] = (255, 0, 0)
    v[S8 > 0] = (255, 0, 255)
    img = Image.fromarray(v)
    d = ImageDraw.Draw(img)
    h, w = T8.shape[:2]
    for i in range(0, 101, 5):
        x, y = int(w * i / 100), int(h * i / 100)
        col = (0, 255, 0) if i % 10 == 0 else (0, 110, 0)
        d.line([(x, 0), (x, h)], fill=col)
        d.line([(0, y), (w, y)], fill=col)
        if i % 10 == 0:
            d.text((x + 2, 2), str(i), fill=(255, 255, 0))
            d.text((2, y + 2), str(i), fill=(255, 255, 0))
    img.save(path, quality=90)


def main(mat, catalog, review, out_dir):
    mat, out = Path(mat), Path(out_dir)
    (out / "previews" / "grids").mkdir(parents=True, exist_ok=True)
    cat = json.loads(Path(catalog).read_text())
    rv = json.loads(Path(review).read_text())
    by_stem = {Path(f["rel"]).stem: f for f in cat["files"] if f["rel"].startswith(("01_", "A7R5_RAW/"))}
    by_rel = {f["rel"]: f for f in cat["files"]}
    nan = rv["not_a_negative"]
    not_colour = dict({i: "positive (visual review)" for k in ("positive_certain", "positive_likely")
                       for i in nan[k]},
                      **{i: "unclear: maybe reversal or odd colour (visual review)"
                         for i in nan["unclear_maybe_reversal_or_odd_colour"]},
                      **{i: "not film (visual review)" for i in nan["not_film"]},
                      **{i: "B&W negative in the colour folder (visual review)"
                         for i in rv["other"]["bw_negative_in_colour_folder"]})
    rows = []
    for cid in rv["rebate_visible"]["colour_group"]:
        f = by_rel.get(cid + ".ARW") if "/" in cid else None
        f = f or by_stem.get(Path(cid).name)
        r = dict(id=cid, rel=f["rel"] if f else None, fmt=f["fmt"] if f else None,
                 source=f["source"] if f else None, film_index=f["film"] if f else None)
        if f is None:
            r["excluded"] = "file not in the 10-18 catalog"
        elif cid in not_colour:
            r["excluded"] = not_colour[cid]
        elif f["fmt"] not in LINEAR:
            r["excluded"] = f"not a linear input ({f['fmt']})"
        rows.append(r)
    kept = [r for r in rows if "excluded" not in r]
    ex = exif([mat / r["rel"] for r in kept])
    cache = data_dir("t3_inversion", "T8")
    for r in kept:
        p = mat / r["rel"]
        sha = check_source(p)
        stem = r["id"].replace("/", "_")
        fr = load_raw(p)
        sat = saturation_mask(fr)
        cfa_shape = list(fr.cfa.shape)
        del fr
        rgb = demosaic_linear(p)
        T8 = block_mean(rgb, 8).astype(np.float32)
        S8 = block_mean(sat.astype(np.float32), 8).astype(np.float32)
        save_cache(cache / f"{stem}_T8.npy", T8, [sha])
        save_cache(cache / f"{stem}_S8.npy", S8, [sha])
        r.update(sha256=sha, stem=stem, cfa_shape=cfa_shape, render_shape=list(rgb.shape[:2]),
                 sat_frac=float(sat.mean()), exif=ex.get(str(p), {}))
        del rgb, sat
        grid_preview(T8, S8, out / "previews" / "grids" / f"{stem}_grid.jpg")
        print(r["id"], r["exif"].get("Model"), r["render_shape"], f"sat {r['sat_frac']:.4f}", flush=True)
    res = dict(created="2026-10-23", material_root_note="paths relative to the material folder",
               candidates=len(rows), kept=len(kept),
               excluded=dict(sorted(Counter(r["excluded"] for r in rows if "excluded" in r).items())),
               frames=rows)
    (out / "candidates.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(len(rows), "candidates,", len(kept), "kept")


if __name__ == "__main__":
    main(*sys.argv[1:5])
