"""T1b: identity check of the material added since 2026-10-18 (task list 2026-10-22), before anyone looks at it.

  t1b_increment.py ROOT INC_DIR T1_FRAMES_JSON OUT_DIR

INC_DIR is the output of audit_ext_increment.py ROOT <10-18b sha256.txt> INC_DIR (increment.json, sha256.txt).
Nothing under ROOT is written and no image is decoded: SHA256, exiftool technical fields, the material index.
The raw exiftool dump (serial numbers) goes to OUT_DIR/outputs/ (git-ignored); committed files keep technical
fields, a per-run body label and whether the body is the one that scanned the user's own rolls.

Index rows come in three layouts (2026-10-22): the header's; "collected" (类型 = 已采集, columns shifted, no original
name / path / pairing; the 配对编号 column holds the licence text); format-in-类型 (类型 holds the film format, the
other columns as the header). A pairing counts only if 配对编号 is a list of material ids whose files exist.

Links to data already used or sealed, per new file:
  copy      byte copy of a file registered on 2026-10-18 (T1 record of that file: dev / frozen / excluded)
  frozen    SHA256 in a frozen manifest, or the original name is a blocked frozen frame (frozen.frozen_blocks)
  blueneg   original name of a BlueNeg frame: release roll, dev / frozen as in T1
  own_body  scanned with the body that scanned the user's own rolls (serial number, dev files only compared)
Units: a BlueNeg roll when the name and the release metadata agree (verified); otherwise a scan session = same body
and same capture day (inferred, one kind of evidence); otherwise the file alone (unknown).
Seal rule (fixed in the task list before this run): a candidate unit holds normal colour negatives the engine can
read, at least 5 frames with a reference, and no link above; candidates sorted by SHA256 of the unit name, every
other one sealed starting with the first. This script only lists the proposal; it writes no frozen manifest except
the addendum for copies of frames that are already frozen (blocked-name hits), see --no-addendum.
"""
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from filmcolor import frozen  # noqa: E402

ENGINE = {".nef", ".nrw", ".raf", ".cr2", ".cr3", ".arw", ".rw2", ".orf", ".pef", ".srw", ".dng"}  # app.EXTENSIONS
ROLL_RE = re.compile(r"^(\d{6,8})([A-Z]?)-(\d+)-")
ID_RE = re.compile(r"^[A-Z]\d{3,4}$")
POSITIVE_WORDS = ("反转片", "正像", "正片", "幻灯")
BW_WORDS = ("黑白",)
EXIF_KEEP = ("FileType", "ImageWidth", "ImageHeight", "BitsPerSample", "Make", "Model", "Software",
             "DateTimeOriginal", "CreateDate")
SERIAL_KEYS = ("SerialNumber", "InternalSerialNumber", "BodySerialNumber", "CameraSerialNumber")
MIN_REF, SEAL_START = 5, 0
ADDENDUM = REPO / "data/manifests/frozen_test_v2_ds_addendum_20261022.json"


def exif(paths, dump):
    args = dump.with_suffix(".args")
    args.write_text("".join(f"{p}\n" for p in paths))
    out = subprocess.run(["exiftool", "-json", "-n", "-q", "-q", *(f"-{k}" for k in EXIF_KEEP + SERIAL_KEYS),
                          "-@", str(args)], capture_output=True, text=True, check=True).stdout
    dump.write_text(out)
    return {d["SourceFile"]: d for d in json.loads(out or "[]")}


def serial(e):
    return next((str(e[k]) for k in SERIAL_KEYS if e.get(k) not in (None, "", 0)), None)


def index_rows(root, tops):
    rows = {}
    for r in list(csv.reader(open(root / "索引.csv", encoding="utf-8-sig")))[1:]:
        if r[2] == "已采集":
            row = dict(layout="collected", id=r[0], film=r[7], orig=None, path=None, note=r[4], pair=None,
                       licence=r[3], extra=r[10])
        elif r[2] in tops:
            row = dict(layout="header", id=r[0], film=r[4], orig=r[5], path=r[6], note=r[7], pair=r[9], licence=r[10])
        else:
            row = dict(layout="format_in_type", id=r[0], film=None, format=r[2], orig=r[5], path=r[6], note=r[7],
                       pair=r[9], licence=r[4])
        rows[r[1]] = row
    return rows


def pair_ids(text):
    xs = [x.strip() for x in re.split(r"[;,，；\s]+", text or "") if x.strip()]
    return xs if xs and all(ID_RE.match(x) for x in xs) else []


def main(root, inc_dir, t1_frames, out_dir, addendum=True):
    root, inc_dir, out = Path(root), Path(inc_dir), Path(out_dir)
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    inc = json.loads((inc_dir / "increment.json").read_text())
    now = {}
    for line in (inc_dir / "sha256.txt").read_text().splitlines():
        h, r = line.split("  ", 1)
        now[r.removeprefix("./")] = h
    tops = {p.name for p in root.iterdir() if p.is_dir()}
    new_paths = sorted(r for r in set(inc["new"]) | set(inc["moved"]) if "/" in r)
    old_paths = {r for v in inc["moved"].values() for r in v}

    t1 = json.loads(Path(t1_frames).read_text())
    t1_by_file = {f: r for r in t1 for f in r["files"]}
    sealed = [m for m in frozen.MANIFESTS if m.resolve() != ADDENDUM.resolve()]  # not the addendum this writes
    fz_hash = frozen.frozen_hashes(sealed)
    blocks = frozen.frozen_blocks(sealed)
    meta = json.loads((REPO / "data/raw/public/blueneg/meta.json").read_text())
    release = {m["filename"]: m for m in meta}
    dev_rolls = {r["roll"] for r in t1 if r["partition"] == "dev" and r["roll"]}
    frozen_rolls = {r["roll"] for r in t1 if r["partition"] == "frozen" and r["roll"]}

    # byte copies of repo raws (data/raw is not in the material list): hash only files of a matching size
    sizes = {(root / r).stat().st_size for r in new_paths}
    repo_raw = {}
    for q in sorted((REPO / "data/raw").rglob("*")):
        if q.is_file() and not q.name.startswith(".") and q.stat().st_size in sizes:
            repo_raw.setdefault(frozen.sha256_file(q), []).append(q.relative_to(REPO).as_posix())

    idx = index_rows(root, tops)
    by_id = {row["id"]: name for name, row in idx.items()}
    on_disk = {Path(r).name: r for r in now}

    own_files = sorted({r["files"][0] for r in t1 if r["source"] == "own" and r["partition"] == "dev"
                        and r["files"][0].lower().endswith(".nef")})
    own_one = {}
    for f in own_files:
        own_one.setdefault(t1_by_file[f]["roll"], f)
    ex = exif([root / r for r in new_paths] + [root / f for f in own_one.values()], out / "outputs/exif_raw.json")
    own_serials = {serial(ex.get(str(root / f), {})) for f in own_one.values()} - {None}
    bodies = sorted({serial(ex.get(str(root / r), {})) for r in new_paths} - {None})
    body_label = {s: f"body{i + 1:02d}" for i, s in enumerate(bodies)}

    files = []
    for r in new_paths:
        name, top, ext = Path(r).name, r.split("/", 1)[0], Path(r).suffix.lower()
        row = idx.get(name) or {}
        e = ex.get(str(root / r), {})
        s = serial(e)
        note = " ".join(x for x in (row.get("note"), row.get("film"), row.get("extra")) if x)
        orig = row.get("orig") or ""
        links = []
        copy_of = sorted(inc["moved"].get(r, []))
        if copy_of:
            recs = [t1_by_file.get(x) for x in copy_of]
            parts = sorted({x["partition"] or "none" for x in recs if x}) or ["not in T1"]
            links.append(f"copy:{'/'.join(parts)}")
        sha = now[r]
        if sha in repo_raw:
            recs = [t1_by_file.get(x.removeprefix("data/raw/")) or t1_by_file.get(x) for x in repo_raw[sha]]
            parts = sorted({x["partition"] or "none" for x in recs if x}) or ["not in T1"]
            links.append(f"copy_repo:{'/'.join(parts)}")
            copy_of = copy_of + repo_raw[sha]
        if sha in fz_hash:
            links.append("frozen:hash")
        m = ROLL_RE.match(Path(orig).name)
        bn = None
        if m:
            stem = Path(orig).name.split(".", 1)[0]
            rel = release.get(stem)
            roll = f"ds-{m.group(1)[:6]}{m.group(2)}" if m.group(2) else f"ds-{m.group(1)[:6]}"
            bn = dict(stem=stem, roll=roll, in_release=rel is not None,
                      release_roll=rel and rel["roll_id"], partition=rel and rel["partition"])
            if stem in blocks["names"] or m.group(1) + m.group(2) in blocks["rolls"] or roll in blocks["rolls"]:
                links.append("frozen:name")
            elif roll in frozen_rolls:
                links.append("frozen:roll")
            elif roll in dev_rolls:
                links.append("blueneg:dev")
            else:
                links.append("blueneg:other")
        if s and s in own_serials:
            links.append("own_body")
        pairs = pair_ids(row.get("pair"))
        partners = [by_id.get(p) for p in pairs]
        refs = [on_disk[p] for p in partners if p and p in on_disk]
        day = (e.get("DateTimeOriginal") or e.get("CreateDate") or "")[:10] or None
        if bn and bn["in_release"]:
            unit, level = bn["roll"], "verified"
        elif bn:
            unit, level = bn["roll"], "inferred"
        elif s and day:
            unit, level = f"session:{e.get('Make')} {e.get('Model')}|{body_label[s]}|{day}", "inferred"
        else:
            unit, level = f"file:{r}", "unknown"
        files.append(dict(
            rel=r, id=row.get("id"), sha256=sha, copy_of=copy_of or None, top=top, ext=ext, engine_readable=ext in ENGINE,
            in_index=bool(row), index_layout=row.get("layout"), orig_name=orig or None,
            source_kind=(lambda p: None if not p else p.split("/", 1)[0] if "." in p.split("/", 1)[0] else "群友分享"
                         if p.startswith("群友分享") else "other")(row.get("path")),
            pair_text_is_ids=bool(pairs), references=refs,
            possible_positive=top == "03_正片幻灯片_已反相" or any(w in note for w in POSITIVE_WORDS),
            possible_bw=any(w in note for w in BW_WORDS),
            camera=" ".join(x for x in (e.get("Make"), e.get("Model")) if x) or None, body=body_label.get(s),
            exif={k: e.get(k) for k in EXIF_KEEP}, blueneg=bn, links=links, unit=unit, unit_level=level))

    units = defaultdict(list)
    for f in files:
        units[f["unit"]].append(f)
    unit_rows = []
    for k, fs in sorted(units.items()):
        normal = [f for f in fs if f["top"] == "01_彩色负片_未反相" and not f["possible_positive"]
                  and not f["possible_bw"] and f["engine_readable"]]
        u = dict(unit=k, level=fs[0]["unit_level"], files=len(fs), normal_readable=len(normal),
                 with_reference=sum(bool(f["references"]) for f in normal),
                 links=sorted({x for f in fs for x in f["links"]}), cameras=dict(Counter(f["camera"] for f in fs)),
                 exts=dict(Counter(f["ext"] for f in fs)))
        u["seal_candidate"] = u["with_reference"] >= MIN_REF and not u["links"]
        unit_rows.append(u)
    cands = sorted((u["unit"] for u in unit_rows if u["seal_candidate"]),
                   key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    proposal = dict(rule=f"units with >= {MIN_REF} referenced normal readable colour negatives and no link; sorted "
                         f"by sha256(unit name); every other one sealed starting at position {SEAL_START}",
                    candidates=cands, proposed_sealed=cands[SEAL_START::2], proposed_dev=[c for i, c in
                                                                                         enumerate(cands) if i % 2 != SEAL_START],
                    status="proposal only; nothing sealed")

    frozen_copies = [f for f in files if "frozen:name" in f["links"] or "frozen:roll" in f["links"]]
    if addendum and frozen_copies:
        ADDENDUM.write_text(json.dumps(dict(
            id=ADDENDUM.stem, created="2026-10-22", addendum_to="frozen_test_v2_ds",
            rule="Copies of frames sealed in frozen_test_v2_ds that were added to the material folder after "
                 "2026-10-18 under new names (other format, other bytes); found by the original name the material "
                 "index records (scripts/t1b_increment.py). Not opened. Copies without a recorded original name are "
                 "not found by this rule.",
            files=[dict(id=f"material:{f['rel']}", sha256=f["sha256"], orig_name=f["orig_name"],
                        roll=f["blueneg"]["roll"]) for f in frozen_copies]), ensure_ascii=False, indent=1) + "\n")

    new_only = [f for f in files if not f["rel"] in inc["moved"]]
    summary = dict(
        root_files_now=inc["n_now"], root_files_1018=inc["n_old"], unchanged=inc["unchanged"],
        new_paths=len(files), new_contents=len(new_only), copies_of_registered=len(files) - len(new_only),
        paths_of_1018_gone=inc_paths_gone(inc, now), contents_gone=inc["gone"],
        manifest_checks=inc["manifest_checks"],
        new_by_top=dict(Counter(f["top"] for f in files)), new_by_ext=dict(Counter(f["ext"] for f in new_only)),
        index=dict(rows_for_new_paths=sum(f["in_index"] for f in files),
                   layouts=dict(Counter(f["index_layout"] for f in files)),
                   pairing_column_raw_non_empty=raw_pair_non_empty(root, {Path(f["rel"]).name for f in files}),
                   pairing_ids_valid=sum(f["pair_text_is_ids"] for f in files),
                   with_reference_on_disk=sum(bool(f["references"]) for f in files)),
        engine_readable_new=sum(f["engine_readable"] for f in new_only),
        possible_positive=sum(f["possible_positive"] for f in files), possible_bw=sum(f["possible_bw"] for f in files),
        frozen_hash_overlap=sum("frozen:hash" in f["links"] for f in files),
        links=dict(Counter(x for f in files for x in f["links"])),
        files_with_any_link=sum(bool(f["links"]) for f in files),
        blueneg_named=[dict(rel=f["rel"], **f["blueneg"], links=f["links"]) for f in files if f["blueneg"]],
        own_body_serials_compared=len(own_serials), bodies_new=len(bodies),
        units=dict(Counter(u["level"] for u in unit_rows)),
        units_with_reference=sum(u["with_reference"] > 0 for u in unit_rows),
        seal=proposal, frozen_addendum=dict(path=str(ADDENDUM.relative_to(REPO)) if frozen_copies and addendum else None,
                                           files=len(frozen_copies)))
    (out / "new_files.json").write_text(json.dumps(files, ensure_ascii=False, indent=1) + "\n")
    (out / "units.json").write_text(json.dumps(unit_rows, ensure_ascii=False, indent=1) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def raw_pair_non_empty(root, names):
    """Rows whose 配对编号 column (as the header names it, whatever the layout) is not empty."""
    rows = list(csv.reader(open(root / "索引.csv", encoding="utf-8-sig")))[1:]
    return sum(bool(r[9].strip()) for r in rows if r[1] in names)


def inc_paths_gone(inc, now):
    """Paths of the 2026-10-18 list that are no longer there, with where their content sits now (if anywhere)."""
    by_hash = defaultdict(list)
    for r, h in now.items():
        by_hash[h].append(r)
    out = {}
    for line in (REPO / inc["old_list"]).read_text().splitlines():
        h, r = line.split("  ", 1)
        r = r.removeprefix("./")
        if r not in now:
            out[r] = sorted(by_hash.get(h, []))
    return out


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(*a[:4], addendum="--no-addendum" not in sys.argv)
