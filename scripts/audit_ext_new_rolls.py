"""Read-only roll-level summary of the files an incremental audit found new.

Joins new files with the material index and exiftool technical fields, assigns a roll key, counts
negatives and paired references per roll. Nothing under ROOT is written; output goes to OUT_DIR.

Roll key: original names of the form <date><letter>-<frame>-<place> (e.g. 19960815F-23-san-francisco)
carry the shooting date and a roll letter; one letter spans several days within a trip with a single
film stock (e.g. 19960808B..19960811B all Fuji NPH400), so the roll is <yyyymm><letter>. Names without
a letter keep their full date. Anything else is grouped by (film, camera) and marked ungrouped, i.e.
its roll membership is unknown.

  audit_ext_new_rolls.py ROOT INCREMENT_JSON OUT_DIR
"""
import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROLL_RE = re.compile(r"^(\d{6,8}[A-Z]?)-(\d+)-")
FIELDS = ("FileType", "ImageWidth", "ImageHeight", "BitsPerSample", "Make", "Model", "Software",
          "ProfileDescription", "DateTimeOriginal", "ExposureTime", "FNumber", "ISO")


def main(root, inc_json, out_dir):
    root, out = Path(root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    inc = json.loads(Path(inc_json).read_text())
    new = [r for r in inc["new"] if "/" in r and not r.startswith("_")]
    dup_of = {}
    for v in inc["new_dups"].values():
        for r in sorted(v)[1:]:
            dup_of[r] = sorted(v)[0]

    args = out / "exif_args.txt"
    args.write_text("".join(f"{root / r}\n" for r in new))
    raw = json.loads(subprocess.run(["exiftool", "-json", "-n", "-fast2", "-q", "-q", "-@", str(args)],
                                    capture_output=True, text=True, check=True).stdout)
    exif = {Path(d["SourceFile"]).relative_to(root).as_posix(): {k: d.get(k) for k in FIELDS} for d in raw}

    index = {}
    for row in csv.DictReader(open(root / "索引.csv", encoding="utf-8-sig")):
        index[f"{row['类型']}/{row['文件名']}"] = row
    by_id = {row["编号"]: k for k, row in index.items()}

    files = []
    for r in new:
        row, e = index.get(r), exif.get(r, {})
        top = r.split("/", 1)[0]
        cam = " ".join(x for x in (e.get("Make"), e.get("Model")) if x) or "-"
        f = dict(rel=r, top=top, id=row["编号"] if row else None, in_index=row is not None,
                 film=row["胶片/来源"] if row else None, orig=row["原文件名"] if row else None,
                 pair=row["配对编号"] if row else "", license=row.get("许可") if row else None,
                 note=row["说明"][:200] if row else None, camera=cam, dup_of=dup_of.get(r), exif=e)
        m = ROLL_RE.match(f["orig"] or "")
        key = m and (m.group(1)[:6] + m.group(1)[-1] if m.group(1)[-1].isalpha() else m.group(1))
        f["roll"] = f"ds-{key}" if m else f"ungrouped:{f['film']}|{cam}" if row else f"folder:{top}"
        f["roll_known"] = bool(m)
        files.append(f)
    (out / "new_files.json").write_text(json.dumps(files, ensure_ascii=False, indent=1))

    # rolls of new colour negatives; a pair counts if its partner file exists on disk
    rolls = defaultdict(lambda: dict(neg=0, neg_unique=0, paired=0, films=Counter(), cameras=Counter(),
                                     sizes=Counter(), types=Counter(), known=False))
    for f in files:
        if f["top"] != "01_彩色负片_未反相":
            continue
        g = rolls[f["roll"]]
        g["neg"] += 1
        g["neg_unique"] += f["dup_of"] is None
        partner = by_id.get(f["pair"].split(";")[0].strip()) if f["pair"] else None
        g["paired"] += bool(partner and (root / partner).exists() and f["dup_of"] is None)
        g["films"][f["film"]] += 1
        g["cameras"][f["camera"]] += 1
        g["sizes"][f"{f['exif'].get('ImageWidth')}x{f['exif'].get('ImageHeight')}"] += 1
        g["types"][f["exif"].get("FileType")] += 1
        g["known"] = f["roll_known"]
    rolls = {k: {**v, **{c: dict(v[c]) for c in ("films", "cameras", "sizes", "types")}} for k, v in rolls.items()}
    (out / "new_rolls.json").write_text(json.dumps(rolls, ensure_ascii=False, indent=1))

    known = {k: v for k, v in rolls.items() if v["known"]}
    print("new files", len(files), "not in index", sum(not f["in_index"] for f in files))
    print("colour-negative rolls: known", len(known), "ungrouped", len(rolls) - len(known))
    print("known rolls with >=5 unique paired frames", sum(v["paired"] >= 5 for v in known.values()))
    print("unique negs / paired: known", sum(v["neg_unique"] for v in known.values()),
          sum(v["paired"] for v in known.values()),
          "ungrouped", sum(v["neg_unique"] for k, v in rolls.items() if not v["known"]),
          sum(v["paired"] for k, v in rolls.items() if not v["known"]))
    for k, v in sorted(rolls.items(), key=lambda kv: (-kv[1]["known"], -kv[1]["paired"])):
        print(f"{k[:60]:60s} neg {v['neg_unique']:3d} paired {v['paired']:3d} {list(v['films'])[:2]} "
              f"{list(v['cameras'])[:2]} {list(v['sizes'])[:2]}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
