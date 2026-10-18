"""Read-only dataset report of the whole external material folder: what kind of negative each file is,
which roll / film / device it comes from, whether it is a duplicate, and which reference positives pair with it.

Nothing under ROOT is written. Inputs are the SHA256 list of an incremental audit (same file set), an exiftool
JSON dump of ROOT, the material index, the BlueNeg release metadata, the repo manifests and the photo-folder
match. Output: OUT/catalog.json (one row per file, technical fields only) and OUT/summary.json.

Units:
  file      a file on disk
  unique    distinct content (SHA256), first path kept
  frame     a distinct exposure: unique files merged further when they share source folder and original
            file stem (e.g. NEF + JPG of the same DSC_xxxx), or the same BlueNeg frame name
  roll      own rolls: the original folder; BlueNeg: <yyyymm><letter> (see audit_ext_new_rolls.py);
            Auckland Museum sheets: one plate / sheet each, no roll; everything else: unknown
Pair levels (strongest first):
  dev       named by s3_dev_frames.json (registered and checked by S3-3)
  frozen    named by a frozen_test_*.json manifest (checked, sealed)
  official  BlueNeg release metadata gives a printed scan for the frame
  photos    a frame of the user's photo folder matched by content (audit_photos_match.py, r >= 0.75)
  index     only the material index names a partner (index author's matching, not checked by us)

  audit_dataset_report.py ROOT SHA256_TXT EXIF_JSON BLUENEG_META OUT_DIR
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from filmcolor import frozen

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "data" / "manifests"
ROLL_RE = re.compile(r"^(\d{6,8}[A-Z]?)-(\d+)-")
AM_RE = re.compile(r"\(AM [0-9][0-9.\-]*\)")
RAW_EXT = {".nef", ".raf", ".cr2", ".cr3", ".arw", ".rw2", ".orf"}
KIND = {"01_彩色负片_未反相": "colour_neg", "02_黑白负片_未反相": "bw_neg", "03_正片幻灯片_已反相": "slide",
        "04_已反转成品_对照答案": "positive", "05_数码设备照_非素材": "other", "06_非照片图表": "other",
        "07_工具与参考": "tool"}
# folders outside the index; both foma folders hold positives by eye (reversal-processed Fomapan 100 NEF captures
# and edited JPEGs of the same frames, contact sheet 2026-10-18), not negatives
LOOSE = {"Eastman 5294": "slide", "foma pan100(d67 尼康数字化)": "bw_reversal",
         "foma pan100(d67 负片正冲": "bw_reversal", "A7R5_RAW": "colour_neg"}
FIELDS = ("FileType", "ImageWidth", "ImageHeight", "BitsPerSample", "Make", "Model", "Software")
# photo-folder matches need a clear winner: at r >= 0.75 alone one group-pack negative "matched" six unrelated photos
# with margins 0.002-0.034 (2026-10-18)
PHOTO_MARGIN = 0.05


def read_sha(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        h, rel = line.split("  ", 1)
        out[rel.removeprefix("./")] = h
    return out


def fmt(rel, e):
    ext = Path(rel).suffix.lower()
    if ext in RAW_EXT:
        return "camera_raw"
    if ext == ".dng":
        return "scanner_dng" if "SilverFast" in str(e.get("Software", "")) or not e.get("Make") else "camera_dng"
    if ext in (".tif", ".tiff"):
        b = e.get("BitsPerSample")
        b = int(str(b).split()[0]) if b else 0
        return "tiff16" if b >= 16 else "tiff8"
    return {".jpg": "jpeg", ".jpeg": "jpeg", ".png": "png"}.get(ext, ext.lstrip(".") or "none")


def main(root, sha_txt, exif_json, bn_meta, out_dir, pairing_json=None):
    root, out = Path(root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sha = {r: h for r, h in read_sha(sha_txt).items() if "/" in r and not r.startswith("_")}
    exif = {}
    for d in json.loads(Path(exif_json).read_text()):
        rel = Path(d["SourceFile"]).relative_to(root).as_posix()
        exif[rel] = {k: d.get(k) for k in FIELDS}
    index, by_name = {}, defaultdict(list)
    reader = csv.reader(open(root / "索引.csv", encoding="utf-8-sig"))
    head = next(reader)
    for vals in reader:
        if len(vals) > 2 and vals[2] == "已采集":  # shifted rows (A1034-A1078, C0036-): remap by position
            row = dict(编号=vals[0], 文件名=vals[1], 类型="", 许可=vals[3], 说明=vals[4], 原路径=vals[6],
                       **{"胶片/来源": vals[7]}, 需目视复核=vals[8], 原文件名=vals[1], 配对编号="", shifted=True)
        else:
            row = dict(zip(head, vals))
        by_name[row["文件名"].lower()].append(row)
        if row["类型"]:
            index.setdefault(f"{row['类型']}/{row['文件名']}", row)
    for rel in sha:  # fall back to a unique file name
        rows_ = by_name.get(rel.rsplit("/", 1)[-1].lower(), [])
        if rel not in index and len(rows_) == 1:
            index[rel] = rows_[0]
    rel_of_id = {row["编号"]: k for k, row in index.items() if k in sha}
    bn = {m["filename"]: m for m in json.loads(Path(bn_meta).read_text())}
    fz = frozen.frozen_hashes()

    dev_pairs = defaultdict(set)  # negative sha -> ref shas
    for f in json.loads((MANIFESTS / "s3_dev_frames.json").read_text())["frames"]:
        if f.get("raw_sha256") and f.get("ref_sha256"):
            dev_pairs[f["raw_sha256"]].add(f["ref_sha256"])
    photo_pairs = defaultdict(list)  # material id -> photo kinds
    m = json.loads((REPO / "experiments/s3_photos_audit/match.json").read_text())
    for p in m["photos"]:
        if p.get("r", 0) >= m["same_frame_r"] and p.get("margin", 0) >= PHOTO_MARGIN and p.get("best"):
            photo_pairs[p["best"]].append("user_lightroom" if "编辑" in p["photo"] else "photo_folder_other")

    content, candidate = defaultdict(list), defaultdict(list)  # negative rel -> positive rels
    if pairing_json:
        for p in json.loads(Path(pairing_json).read_text())["pairs"]:
            {"content": content, "candidate": candidate}.get(p["level"], {}).setdefault(p["best"], []).append(p["ref"])

    first = {}
    rows = []
    for rel in sorted(sha):
        top = rel.split("/", 1)[0]
        h, e, ix = sha[rel], exif.get(rel, {}), index.get(rel)
        kind = KIND.get(top) or next((v for k, v in LOOSE.items() if top.startswith(k)), "unknown")
        if kind == "tool" or Path(rel).suffix.lower() in (".xmp", ".cube", ".json", ".md", ".csv"):
            kind = "tool"
        orig = ix["原文件名"] if ix else Path(rel).name
        src_dir = ix["原路径"].split("/")[0] if ix else top
        r = dict(rel=rel, id=ix["编号"] if ix else None, kind=kind, fmt=fmt(rel, e), w=e.get("ImageWidth"),
                 h=e.get("ImageHeight"), bits=e.get("BitsPerSample"),
                 camera=" ".join(str(x) for x in (e.get("Make"), e.get("Model")) if x) or None,
                 software=(str(e.get("Software"))[:40] if e.get("Software") else None),
                 film=ix["胶片/来源"] if ix else None, licence=ix["许可"] if ix else None, orig=orig, src_dir=src_dir,
                 in_index=ix is not None, dup_of=first.get(h), frozen=h in fz, review=ix["需目视复核"] if ix else "")
        first.setdefault(h, rel)

        mm = ROLL_RE.match(orig or "")
        stem = Path(orig).stem
        meta = bn.get(stem) if mm else None
        if meta:
            key = mm.group(1)
            r.update(source="blueneg", roll=f"ds-{key[:6] + key[-1] if key[-1].isalpha() else key}",
                     film=meta["film_type"], partition=meta["partition"], frame_key=f"blueneg:{stem}")
        elif AM_RE.search(orig or ""):
            r.update(source="auckland_museum", roll=None, frame_key=f"am:{AM_RE.search(orig).group(0)}")
        elif ix and ix["许可"] == "自有素材":
            r.update(source="own", roll=f"own:{src_dir}", frame_key=f"own:{src_dir}/{stem.lower()}")
        elif not ix:
            r.update(source=f"loose:{top}", roll=f"loose:{top}", frame_key=f"loose:{top}/{stem.lower()}")
        else:
            src = {"交付包": "group_pack", "discuss.pixls.us": "pixls"}.get(src_dir, "group_single")
            r.update(source=src, roll=None, frame_key=f"{src}:{h[:16]}")
        try:
            if r.get("source") == "blueneg":
                frozen.check_name(stem)
                frozen.check_roll(r["roll"])
        except frozen.FrozenDataError:
            r["frozen"] = True
        r["sha256"] = h

        # partners: index names, then upgrade the level if a stronger source confirms
        partners = [p.strip() for p in re.split(r"[;,，、\s]+", ix["配对编号"]) if re.fullmatch(r"[A-Z]\d+", p.strip())] \
            if ix else []
        r["partners"] = [p for p in partners if p in rel_of_id]
        r["partners_missing"] = [p for p in partners if p not in rel_of_id]
        levels = set()
        if r["partners"]:
            levels.add("index")
        if h in dev_pairs or any(sha[rel_of_id[p]] in dev_pairs.get(h, ()) for p in r["partners"]):
            levels.add("dev")
        if r["frozen"] and r["partners"]:
            levels.add("frozen")
        if meta and meta.get("printed_path"):
            levels.add("official")
        if r["id"] in photo_pairs:
            levels.update({"photos"})
            r["photo_refs"] = Counter(photo_pairs[r["id"]])
        if rel in content:
            levels.add("content")
            r["content_refs"] = content[rel]
        if rel in candidate:
            r["candidate_refs"] = candidate[rel]
        if r["partners"] and r["kind"] in ("colour_neg", "bw_neg") and pairing_json and not r["frozen"] \
                and r.get("source") != "blueneg":
            prel = {rel_of_id[p] for p in r["partners"]}
            r["index_check"] = ("agree" if prel & set(content.get(rel, ())) else
                                "disagree" if content.get(rel) else
                                "weak" if prel & set(candidate.get(rel, ())) else "unconfirmed")
        r["pair_levels"] = sorted(levels)
        rows.append(r)

    # frames: merge unique files that share frame_key; a frame is paired at the best level of any member
    order = ["dev", "frozen", "official", "photos", "content", "index"]
    frames = defaultdict(list)
    for r in rows:
        if r["dup_of"] is None and r["kind"] in ("colour_neg", "bw_neg"):
            frames[r["frame_key"]].append(r)
    fr = []
    for k, members in frames.items():
        lv = set().union(*(set(x["pair_levels"]) for x in members))
        best = next((o for o in order if o in lv), None)
        x = members[0]
        fr.append(dict(frame=k, kind=x["kind"], source=x["source"], roll=x["roll"], film=x["film"],
                       camera=x["camera"], fmts=sorted({y["fmt"] for y in members}), files=[y["rel"] for y in members],
                       pair=best, pair_levels=sorted(lv), frozen=any(y["frozen"] for y in members),
                       partition=x.get("partition"), user_lr=sum(y.get("photo_refs", {}).get("user_lightroom", 0)
                                                                 for y in members),
                       review=next((y["review"] for y in members if y["review"] not in ("", "否")), "")))

    def tab(sel):
        g = defaultdict(lambda: dict(frames=0, paired=Counter(), rolls=set(), films=Counter(), cameras=Counter(),
                                     fmts=Counter(), frozen=0, partitions=Counter(), user_lr=0))
        for f in sel:
            s = g[f["source"]]
            s["frames"] += 1
            s["paired"][f["pair"] or "none"] += 1
            if f["roll"]:
                s["rolls"].add(f["roll"])
            s["films"][f["film"]] += 1
            s["cameras"][f["camera"]] += 1
            s["fmts"]["+".join(f["fmts"])] += 1
            s["frozen"] += f["frozen"]
            s["partitions"][f["partition"]] += 1
            s["user_lr"] += f["user_lr"] > 0
        return {k: {**v, "rolls": sorted(v["rolls"]), "n_rolls": len(v["rolls"]),
                    **{c: dict(v[c].most_common()) for c in ("paired", "films", "cameras", "fmts", "partitions")}}
                for k, v in sorted(g.items())}

    rolls = defaultdict(lambda: dict(frames=0, paired=0, frozen=0, films=Counter(), cameras=Counter()))
    for f in fr:
        if f["roll"]:
            q = rolls[f["roll"]]
            q["frames"] += 1
            q["paired"] += f["pair"] is not None
            q["frozen"] += f["frozen"]
            q["films"][f["film"]] += 1
            q["cameras"][f["camera"]] += 1
    rolls = {k: {**v, "films": dict(v["films"]), "cameras": dict(v["cameras"]), "kind": None} for k, v in rolls.items()}
    for f in fr:
        if f["roll"]:
            rolls[f["roll"]]["kind"] = f["kind"]

    summary = dict(
        files=len(rows), unique=sum(r["dup_of"] is None for r in rows),
        files_by_kind=dict(Counter(r["kind"] for r in rows)),
        unique_by_kind=dict(Counter(r["kind"] for r in rows if r["dup_of"] is None)),
        frames_by_kind=dict(Counter(f["kind"] for f in fr)),
        not_in_index=sum(not r["in_index"] for r in rows),
        partners_missing=sorted({p for r in rows for p in r["partners_missing"]}),
        index_check=dict(Counter(r["index_check"] for r in rows if "index_check" in r)),
        colour=tab([f for f in fr if f["kind"] == "colour_neg"]),
        bw=tab([f for f in fr if f["kind"] == "bw_neg"]),
        rolls=rolls,
    )
    (out / "catalog.json").write_text(json.dumps(dict(files=rows, frames=fr), ensure_ascii=False, indent=0, default=dict))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=dict))
    for kind in ("colour", "bw"):
        print("==", kind)
        for k, v in summary[kind].items():
            print(f"  {k:22s} frames {v['frames']:5d} rolls {v['n_rolls']:3d} frozen {v['frozen']:4d} "
                  f"paired {v['paired']} userLR {v['user_lr']} fmts {list(v['fmts'].items())[:3]}")
    print({k: summary[k] for k in ("files", "unique", "files_by_kind", "unique_by_kind", "frames_by_kind",
                                   "not_in_index")})
    print("partners missing", len(summary["partners_missing"]), summary["partners_missing"][:10])
    print("index pairs vs content", summary["index_check"])


if __name__ == "__main__":
    main(*sys.argv[1:7])
