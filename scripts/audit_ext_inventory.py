"""Read-only inventory of an external material folder: SHA256, technical metadata, index cross-check.

Nothing under ROOT is written. The raw exiftool dump (may hold GPS / serial numbers) goes to
OUT/outputs/ (git-ignored); the committed inventory keeps technical fields only.

  audit_ext_inventory.py ROOT OUT_DIR
"""
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

KEEP = ("FileType", "ImageWidth", "ImageHeight", "BitsPerSample", "SamplesPerPixel", "Compression",
        "PhotometricInterpretation", "Make", "Model", "LensModel", "Software", "CreatorTool", "DateTimeOriginal",
        "ProfileDescription", "ColorSpace", "Orientation", "WhiteBalance", "ISO", "ExposureTime", "FNumber",
        "HasSettings", "ProcessVersion", "Temperature", "Tint", "RawFileName", "PreservedFileName")


def sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def main(root, out_dir):
    root, out = Path(root), Path(out_dir)
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    raw = json.loads(subprocess.run(["exiftool", "-json", "-n", "-r", "-q", "-q", str(root)],
                                    capture_output=True, text=True, check=True).stdout)
    (out / "outputs" / "exif_raw.json").write_text(json.dumps(raw, ensure_ascii=False, indent=0))
    exif = {Path(d["SourceFile"]).resolve(): d for d in raw}
    index = {r["文件名"]: r for r in csv.DictReader(open(root / "索引.csv", encoding="utf-8-sig"))}

    rows, problems = [], []
    for p in files:
        rel = p.relative_to(root).as_posix()
        e = exif.get(p.resolve(), {})
        ix = index.get(p.name)
        row = dict(id=ix["编号"] if ix else "", rel=rel, ext=p.suffix.upper().lstrip("."), bytes=p.stat().st_size,
                   sha256=sha256(p), **{k: e.get(k, "") for k in KEEP})
        if ix:
            row.update(group=ix["编号"][0], film_or_source=ix["胶片/来源"], orig_dir=ix["原路径"].split("/")[0],
                       orig_name=ix["原文件名"], index_note=ix["说明"], index_review=ix["需目视复核"],
                       index_pair=ix["配对编号"], licence=ix["许可"])
            if ix["类型"] != p.parent.name and not ("非照片" in ix["类型"] and "非照片" in p.parent.name):
                problems.append(f"{p.name}: index type {ix['类型']} vs folder {p.parent.name}")
        elif p.parent != root and not p.parent.name.startswith("_"):
            problems.append(f"{rel}: not in index")
        rows.append(row)
        print(row["id"] or rel, row["ext"], row["bytes"], flush=True)

    seen = {r["rel"].split("/")[-1] for r in rows}
    problems += [f"index row {n}: file missing" for n in index if n not in seen]
    by_hash = defaultdict(list)
    for r in rows:
        by_hash[r["sha256"]].append(r["id"] or r["rel"])
    dups = [v for v in by_hash.values() if len(v) > 1]

    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "inventory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    (out / "inventory.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    summary = dict(files=len(rows), bytes=sum(r["bytes"] for r in rows),
                   by_group_ext={g: Counter(r["ext"] for r in rows if r.get("group") == g) for g in "ABCDRTX"},
                   exact_duplicates=dups, problems=problems)
    (out / "inventory_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
