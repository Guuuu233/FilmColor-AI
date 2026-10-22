"""T1 spot check (task list T1b item 7): 20 roll assignments and 20 negative -> reference pairings of the 344 key
frames, drawn with a fixed seed.

  t1b_spot_check.py T1_FRAMES CATALOG MATCH_JSON MATERIAL_ROOT PHOTO_ROOT OUT_DIR

Roll assignments are checked against evidence other than the rule that assigned them, where there is any:
  own / shop_r01  exiftool on the negative: same camera body and capture time inside the scan session of the
                  other frames of the roll (the folder / list assigned it)
  BlueNeg         release metadata: film type and date agree with the roll's other frames (the roll id assigned it)
  NLP groups      none besides the capture day that assigned them: listed as "no independent evidence"
Pairings: one relation per sampled frame (material partner id, photo-folder match, repo reference file, BlueNeg
official print), drawn at random; a sheet (inverted negative preview | reference) goes to OUT_DIR/outputs/
(git-ignored) to be looked at by eye; the verdicts are recorded by hand in OUT_DIR/spot_check_eye.json.
Every file is checked against the frozen guard (frozen.check_source) before it is read.
"""
import csv
import io
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rawpy
import tifffile
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from filmcolor import frozen  # noqa: E402

SEED, N = 20261022, 20
TILE = 300
RAW = {".nef", ".cr2", ".cr3", ".arw", ".raf", ".orf", ".rw2", ".dng", ".pef", ".srw", ".nrw"}


def key_frames(t1):
    return [r for r in t1 if r["kind"] == "colour_neg" and r["usable_negative"] and r["partition"] == "dev"
            and r["engine_readable"] and r["has_reference"]]


def negative_path(r, mat):
    for f in r["files"]:
        p = (REPO / f) if f.startswith("data/") else mat / f
        if p.suffix.lower() in RAW:
            return p
    if r["source"] == "blueneg":
        stem = r["id"].split(":", 1)[1]
        return REPO / "data/raw/public/blueneg/negative-16bit/blue-intact" / f"{stem}.dng"
    return None


def relations(r, cat_files, cat_by_id, photos_by_best, mat, photo_root):
    out = []
    for f in r["files"]:
        c = cat_files.get(f)
        if c:
            out += [("material partner", mat / cat_by_id[p]) for p in c["partners"] if p in cat_by_id]
            out += [("photo folder", photo_root / x) for x in photos_by_best.get(c["id"], [])]
    if r["source"] == "blueneg":
        stem = r["id"].split(":", 1)[1]
        p = REPO / "data/raw/public/blueneg/printed-16bit/blue-intact" / f"{stem}.tif"
        if p.exists():
            out.append(("BlueNeg official print", p))
    if r["source"] == "shop_r01":
        row = next(x for x in csv.DictReader(open(REPO / "data/manifests/ultramax400_r01.csv"))
                   if f"data/raw/{x['file']}" in r["files"])
        out += [("shop list", REPO / "data" / x) for x in row["reference_files"].split(";") if x]
    if r["source"] == "nlp_official":
        stem = r["id"].split(":", 1)[1]
        out.append(("NLP default output", REPO / f"data/reference/nlp_default/{stem}-positive.tif"))
    if r["source"].startswith("nlp_forum"):
        d = REPO / "data/raw/public" / r["source"]
        out += [("forum attachment", p) for p in sorted(d.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg")]
    return [(k, p) for k, p in out if p.exists()]


def exif(paths):
    out = subprocess.run(["exiftool", "-json", "-n", "-q", "-q", "-DateTimeOriginal", "-SerialNumber", "-Model",
                          *map(str, paths)], capture_output=True, text=True, check=True).stdout
    return {d["SourceFile"]: d for d in json.loads(out or "[]")}


def roll_check(sample, by_roll, mat, bmeta):
    res = []
    need = sorted({negative_path(x, mat) for r in sample if r["source"] in ("own", "shop_r01")
                   for x in by_roll[r["roll"]]} - {None})
    for p in need:
        frozen.check_source(p)
    ex = exif(need)
    for r in sample:
        item = dict(id=r["id"], roll=r["roll"], level=r["roll_level"], source=r["source"])
        if r["source"] in ("own", "shop_r01"):
            mates = [x for x in by_roll[r["roll"]] if x is not r]
            me = ex.get(str(negative_path(r, mat)), {})
            ts = sorted(ex[str(negative_path(x, mat))]["DateTimeOriginal"] for x in mates
                        if str(negative_path(x, mat)) in ex and ex[str(negative_path(x, mat))].get("DateTimeOriginal"))
            bodies = {ex[str(negative_path(x, mat))].get("SerialNumber") for x in mates if str(negative_path(x, mat)) in ex}
            t = me.get("DateTimeOriginal")
            item.update(method="exif: body and capture time vs the other frames of the roll",
                        same_body=me.get("SerialNumber") in bodies and len(bodies) == 1,
                        time=t, session=[ts[0], ts[-1]] if ts else None,
                        inside_session=bool(ts and t and (ts[0] <= t <= ts[-1] or t[:10] in {x[:10] for x in ts})))
            item["agrees"] = item["same_body"] and item["inside_session"]
        elif r["source"] == "blueneg":
            stem = r["id"].split(":", 1)[1]
            m = bmeta[stem]
            mates = [bmeta[x["id"].split(":", 1)[1]] for x in by_roll[r["roll"]] if x is not r]
            films = {x["film_type"] for x in mates}
            dates = sorted(x["date"] for x in mates)
            item.update(method="release metadata: film type and date vs the other frames of the roll",
                        film=m["film_type"], roll_films=sorted(films), date=m["date"],
                        roll_dates=[dates[0], dates[-1]] if dates else None)
            item["agrees"] = films == {m["film_type"]} and bool(dates) and dates[0][:6] == m["date"][:6]
        else:
            item.update(method="none independent of the assigning rule (capture day)", agrees=None)
        res.append(item)
    return res


def preview(p):
    frozen.check_source(p)
    ext = p.suffix.lower()
    if ext in RAW:
        with rawpy.imread(str(p)) as r:
            try:
                t = r.extract_thumb()
                im = Image.open(io.BytesIO(t.data)) if t.format == rawpy.ThumbFormat.JPEG else Image.fromarray(t.data)
            except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
                im = Image.fromarray(r.postprocess(half_size=True, no_auto_bright=True, output_bps=8))
    elif ext in (".tif", ".tiff"):
        a = tifffile.imread(p)
        a = a[..., :3] if a.ndim == 3 else np.repeat(a[..., None], 3, 2)
        step = max(1, max(a.shape[:2]) // 1200)
        a = a[::step, ::step].astype(np.float64)
        a = a / (a.max() or 1)
        im = Image.fromarray((np.clip(a, 0, 1) ** (1 / 2.2 if a.mean() < 0.2 else 1) * 255).astype(np.uint8))
    else:
        im = Image.open(p)
        im.draft("RGB", (1200, 1200))
    im = im.convert("RGB")
    im.thumbnail((TILE, TILE))
    return im


def invert(im):
    a = np.asarray(im, np.float64)
    lo, hi = np.percentile(a.reshape(-1, 3), [0.5, 99.5], axis=0)
    a = 1 - np.clip((a - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8))


def main(t1_frames, catalog, match_json, mat, photo_root, out_dir):
    mat, photo_root, out = Path(mat), Path(photo_root), Path(out_dir)
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    t1 = json.loads(Path(t1_frames).read_text())
    cat = json.loads(Path(catalog).read_text())
    cat_files = {f["rel"]: f for f in cat["files"]}
    cat_by_id = {f["id"]: f["rel"] for f in cat["files"] if f["id"]}
    photos_by_best = defaultdict(list)
    for p in json.loads(Path(match_json).read_text())["photos"]:
        if p.get("best") and p["r"] >= 0.75 and p["margin"] >= 0.05:
            photos_by_best[p["best"]].append(p["photo"])
    bmeta = {m["filename"]: m for m in json.loads((REPO / "data/raw/public/blueneg/meta.json").read_text())}
    key = key_frames(t1)
    by_roll = defaultdict(list)
    for r in t1:
        if r["roll"] and r["partition"] == "dev" and r["usable_negative"] and r["kind"] == "colour_neg":
            by_roll[r["roll"]].append(r)

    rng = random.Random(SEED)
    roll_sample = rng.sample(sorted((r for r in key if r["roll"]), key=lambda r: r["id"]), N)
    rolls = roll_check(roll_sample, by_roll, mat, bmeta)

    rel = {r["id"]: relations(r, cat_files, cat_by_id, photos_by_best, mat, photo_root) for r in key}
    with_rel = sorted((r for r in key if rel[r["id"]]), key=lambda r: r["id"])
    pair_sample = rng.sample(with_rel, N)
    pairs, tiles = [], []
    for i, r in enumerate(pair_sample):
        kind, ref = rng.choice(rel[r["id"]])
        neg = negative_path(r, mat)
        pairs.append(dict(n=i + 1, id=r["id"], negative=str(neg.relative_to(REPO) if neg.is_relative_to(REPO) else
                                                            neg.relative_to(mat)),
                          relation=kind, reference=str(ref.relative_to(REPO) if ref.is_relative_to(REPO) else
                                                       ref.relative_to(photo_root) if ref.is_relative_to(photo_root)
                                                       else ref.relative_to(mat)),
                          ref_strength=r["ref_strength"]))
        tiles.append((invert(preview(neg)), preview(ref)))
    sheet = Image.new("RGB", (2 * TILE + 60, N * (TILE + 10)), "white")
    d = ImageDraw.Draw(sheet)
    for i, (a, b) in enumerate(tiles):
        y = i * (TILE + 10)
        d.text((4, y + 4), str(i + 1), fill="black")
        sheet.paste(a, (40, y))
        sheet.paste(b, (40 + TILE + 10, y))
    for j in range(0, N, 5):  # five pairs per page so a page can be looked at in one go
        sheet.crop((0, j * (TILE + 10), sheet.width, (j + 5) * (TILE + 10))).save(
            out / "outputs" / f"spot_pairs_{j + 1:02d}-{j + 5:02d}.jpg", quality=88)
    res = dict(seed=SEED, key_frames=len(key), key_frames_with_relation=len(with_rel),
               relations_by_kind={k: sum(1 for v in rel.values() for x in v if x[0] == k)
                                  for k in sorted({x[0] for v in rel.values() for x in v})},
               rolls=rolls, rolls_agree=sum(x["agrees"] is True for x in rolls),
               rolls_disagree=sum(x["agrees"] is False for x in rolls),
               rolls_no_independent_evidence=sum(x["agrees"] is None for x in rolls), pairs=pairs)
    (out / "spot_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("pairs",)}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:7])
