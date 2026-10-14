"""Inventory RAW files: checksum, EXIF, sensor metadata, clipping stats; writes CSV manifest + contact sheet.

  inventory_raw.py SRC_DIR SOURCE_ID [--license L] [--film-stock F] [--roll-id R] [--has-film-border B]
"""
import argparse, csv, hashlib, io, json
from pathlib import Path

import exifread
import numpy as np
import rawpy
import tifffile

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("source_id")
ap.add_argument("--license", default="dev-test-only"); ap.add_argument("--film-stock", default="unknown")
ap.add_argument("--roll-id", default=""); ap.add_argument("--has-film-border", default="")
args = ap.parse_args()
src = Path(args.src).resolve()
source_id = args.source_id
root = Path(__file__).resolve().parents[1]
raw_root = root / "data" / "raw"
out_csv = root / "data" / "manifests" / f"{source_id}.csv"
out_json = root / "data" / "manifests" / f"{source_id}_sensor.json"
sheet_path = root / "experiments" / "inventory" / f"{source_id}_contact.tif"
sheet_path.parent.mkdir(parents=True, exist_ok=True)

rows, sensor, thumbs = [], {}, []
for f in sorted(src.glob("*.RAF")) + sorted(src.glob("*.ARW")) + sorted(src.glob("*.DNG")) + sorted(src.glob("*.NEF")):
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    with rawpy.imread(str(f)) as r:
        # TIFF-based RAWs (NEF, ARW, DNG) carry EXIF directly; RAF is not parsed by exifread, so fall back to
        # the embedded JPEG preview
        with f.open("rb") as fh:
            tags = exifread.process_file(fh, details=False)
        if "Image Model" not in tags:
            try:
                th_ = r.extract_thumb()
                if th_.format == rawpy.ThumbFormat.JPEG:
                    tags = exifread.process_file(io.BytesIO(th_.data), details=False)
            except rawpy.LibRawError:
                pass
        g = lambda k: str(tags.get(k, ""))
        raw = r.raw_image_visible
        white = int(r.white_level)
        clip = {}
        colors = r.raw_colors_visible
        for c, name in enumerate(r.color_desc.decode()):
            if c > 2:
                break
            v = raw[colors == c]
            clip[name] = round(float((v >= white).mean()) * 100, 4)
        black = [int(x) for x in r.black_level_per_channel]
        sensor[f.name] = dict(
            raw_shape=list(r.raw_image.shape), visible_shape=list(raw.shape),
            sizes=r.sizes._asdict(), raw_pattern=r.raw_pattern.tolist() if r.raw_pattern is not None else None,
            color_desc=r.color_desc.decode(), black_level_per_channel=black, white_level=white,
            camera_whitebalance=list(r.camera_whitebalance), daylight_whitebalance=list(r.daylight_whitebalance),
            rgb_xyz_matrix=r.rgb_xyz_matrix[:3].tolist(),
            raw_min=int(raw.min()), raw_max=int(raw.max()), clip_pct=clip,
        )
        rgb = r.postprocess(half_size=True, use_camera_wb=False, user_wb=[1, 1, 1, 1],
                            no_auto_bright=False, output_bps=8)
    h, w = rgb.shape[:2]
    step = max(1, max(h, w) // 400)
    thumbs.append(rgb[::step, ::step])
    rows.append(dict(
        file=str(f.relative_to(raw_root)), sha256=sha, source_id=source_id, format=f.suffix[1:].upper(),
        camera_make=g("Image Make"), camera_model=g("Image Model"),
        datetime=g("EXIF DateTimeOriginal"), lens=g("EXIF LensModel"),
        exposure=g("EXIF ExposureTime"), fnumber=g("EXIF FNumber"), iso=g("EXIF ISOSpeedRatings"),
        width=raw.shape[1], height=raw.shape[0], bit_depth=int(np.ceil(np.log2(white + 1))),
        film_stock=args.film_stock, roll_id=args.roll_id, has_film_border=args.has_film_border, reference_files="",
        license=args.license, notes="",
    ))
    print(f.name, rows[-1]["camera_model"], raw.shape, "black", black, "white", white, "clip%", clip)

with out_csv.open("w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
    wr.writeheader(); wr.writerows(rows)
out_json.write_text(json.dumps(sensor, indent=1))

# contact sheet, 4 columns, labelled by index order
th = max(t.shape[0] for t in thumbs); tw = max(t.shape[1] for t in thumbs)
cols = 4; nrows = (len(thumbs) + cols - 1) // cols
sheet = np.full((nrows * (th + 8), cols * (tw + 8), 3), 255, np.uint8)
for i, t in enumerate(thumbs):
    y, x = (i // cols) * (th + 8), (i % cols) * (tw + 8)
    sheet[y:y + t.shape[0], x:x + t.shape[1]] = t
tifffile.imwrite(sheet_path, sheet)
print("wrote", out_csv, out_json, sheet_path)
