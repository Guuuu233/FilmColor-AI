"""Candidate table for the v3c hold-out rule (T1b review, 2026-10-23), metadata only, nothing sealed.

  t1b_v3c_candidates.py MATERIAL_ROOT PHOTO_ROOT T1B_DIR T1_FRAMES OUT_JSON

Rule (written by the research side before any new image was decoded, docs/T1B_INCREMENT_20261022.md section 9):
  unit       a camera body of the new material (EXIF serial number); code = "body-" + sha256(model|serial)[:10],
             the serial itself is not written to committed files
  candidate  >= 5 engine-readable normal colour negatives; no file of the body has a link found by T1b (copy of a
             registered file or repo raw, BlueNeg, own scanning body, frozen); the body's serial does not appear
             in any older material (material folder files of the 10-18 list, repo data/raw camera raws, photo
             folder): if it appears only on development data the body goes to development, if on frozen data
             to frozen
  source     every candidate needs a known source and a licence for research use; unknown -> development
             (confirmed by the user, not by this script; listed here as recorded in the material index)
  selection  strata: >= 20 frames (large) and < 20 (small); within a stratum sort by sha256(unit code)
             ascending; seal the 1st, 3rd, ... of each stratum
Older files are read with exiftool (serial tags only, -fast2); frozen files are not decoded and only their serial
tag is read, to place a body (rule above). Serials go to T1B_DIR/outputs/ (git-ignored).
"""
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from filmcolor import frozen  # noqa: E402

SERIAL_KEYS = ("SerialNumber", "InternalSerialNumber", "BodySerialNumber", "CameraSerialNumber")
RAWISH = {".nef", ".nrw", ".raf", ".cr2", ".cr3", ".arw", ".rw2", ".orf", ".pef", ".srw", ".dng", ".tif", ".tiff",
          ".jpg", ".jpeg", ".png"}
MIN_FRAMES, LARGE = 5, 20


def serial(e):
    return next((str(e[k]) for k in SERIAL_KEYS if e.get(k) not in (None, "", 0)), None)


def code(model, s):
    return "body-" + hashlib.sha256(f"{model}|{s}".encode()).hexdigest()[:10]


def exif_serials(paths, dump):
    args = dump.with_suffix(".args")
    args.write_text("".join(f"{p}\n" for p in paths))
    out = subprocess.run(["exiftool", "-json", "-fast2", "-q", "-q", "-Model", *(f"-{k}" for k in SERIAL_KEYS),
                          "-@", str(args)], capture_output=True, text=True).stdout
    dump.write_text(out)
    return {d["SourceFile"]: d for d in json.loads(out or "[]")}


def main(mat, photo_root, t1b_dir, t1_frames, out_json):
    mat, photo_root, t1b = Path(mat), Path(photo_root), Path(t1b_dir)
    new = json.loads((t1b / "new_files.json").read_text())
    ex_new = {d["SourceFile"]: d for d in json.loads((t1b / "outputs/exif_raw.json").read_text())}
    inc = json.loads((t1b / "outputs/inc/increment.json").read_text())
    t1 = json.loads(Path(t1_frames).read_text())
    t1_part = {f: r["partition"] for r in t1 for f in r["files"]}
    fz = frozen.frozen_hashes([m for m in frozen.MANIFESTS if m.name != "frozen_test_v3.json"])

    # older material: files of the 10-18 list still in place, repo camera raws, photo folder
    old_sha = {}
    for line in (REPO / inc["old_list"]).read_text().splitlines():
        h, r = line.split("  ", 1)
        old_sha[r.removeprefix("./")] = h
    old_files = [(mat / r, "material", r) for r in sorted(old_sha) if (mat / r).exists()
                 and Path(r).suffix.lower() in RAWISH]
    old_files += [(q, "data/raw", q.relative_to(REPO).as_posix()) for q in sorted((REPO / "data/raw").rglob("*"))
                  if q.is_file() and not q.name.startswith(".") and q.suffix.lower() in RAWISH]
    old_files += [(q, "photos", q.relative_to(photo_root).as_posix()) for q in sorted(photo_root.rglob("*"))
                  if q.is_file() and not q.name.startswith(".") and q.suffix.lower() in RAWISH]
    ex_old = exif_serials([p for p, _, _ in old_files], t1b / "outputs/old_serials.json")
    old_by_body = defaultdict(Counter)
    for p, where, rel in old_files:
        e = ex_old.get(str(p), {})
        s = serial(e)
        if not s:
            continue
        sha = old_sha.get(rel) if where == "material" else None
        if (sha and sha in fz) or t1_part.get(rel) == "frozen" or t1_part.get(f"data/raw/{rel}") == "frozen":
            part = "frozen"
        elif where == "photos":
            part = "photos"  # the user's finished photos; frozen copies there are caught by hash only
            try:
                frozen.check_hashes([frozen.sha256_file(p)], frozen=fz)
            except frozen.FrozenDataError:
                part = "frozen"
        else:
            part = "dev"
        old_by_body[code(e.get("Model"), s)][part] += 1

    # bodies of the new material (contents new to the project)
    units = defaultdict(list)
    for f in new:
        if f["copy_of"]:
            continue
        e = ex_new.get(str(mat / f["rel"]), {})
        s = serial(e)
        units[code(e.get("Model"), s) if s else "no-serial"].append(f)
    rows = []
    for k, fs in sorted(units.items()):
        normal = [f for f in fs if f["top"] == "01_彩色负片_未反相" and not f["possible_positive"]
                  and not f["possible_bw"] and f["engine_readable"]]
        links = sorted({x for f in fs for x in f["links"]})
        old = dict(old_by_body.get(k, {}))
        if k == "no-serial":
            status = "development (no body serial)"
        elif links:
            status = "development (linked: " + ", ".join(links) + ")"
        elif old.get("frozen"):
            status = "frozen (body on frozen data)"
        elif old:
            status = "development (body on older " + "/".join(sorted(old)) + " data)"
        elif len(normal) < MIN_FRAMES:
            status = f"development (< {MIN_FRAMES} frames)"
        else:
            status = "candidate"
        rows.append(dict(
            unit=k, run_label=sorted({f["body"] for f in fs if f["body"]}) or None,
            camera=sorted({f["camera"] or "-" for f in fs}), files=len(fs), normal_readable=len(normal),
            sessions=sorted({f["unit"].rsplit("|", 1)[-1] for f in fs if f["unit"].startswith("session:")}),
            links=links, older_material=old, status=status,
            index_layouts=dict(Counter(f["index_layout"] for f in fs)),
            index_sources=dict(Counter(f["source_kind"] or "(none recorded)" for f in fs)),
            frames=[f["rel"] for f in normal]))
    cands = [r for r in rows if r["status"] == "candidate"]
    for r in cands:
        r["stratum"] = "large" if r["normal_readable"] >= LARGE else "small"
        r["sort_key"] = hashlib.sha256(r["unit"].encode()).hexdigest()
    sel = {}
    for st in ("large", "small"):
        order = sorted((r for r in cands if r["stratum"] == st), key=lambda r: r["sort_key"])
        for i, r in enumerate(order):
            r["position"] = i + 1
            sel[r["unit"]] = "seal" if i % 2 == 0 else "development"
    for r in cands:
        r["proposal"] = sel[r["unit"]]
    res = dict(
        rule="v3c (docs/T1B_INCREMENT_20261022.md section 9); unit code body-sha256(model|serial)[:10]",
        older_files_read=len(old_files), older_files_with_serial=sum(1 for p, _, _ in old_files
                                                                     if serial(ex_old.get(str(p), {}))),
        new_contents=sum(len(v) for v in units.values()), bodies=len([k for k in units if k != "no-serial"]),
        no_serial_files=len(units.get("no-serial", [])),
        status=dict(Counter(r["status"].split(" (")[0] for r in rows)),
        candidates=[{k: r[k] for k in ("unit", "run_label", "camera", "normal_readable", "stratum", "position",
                                       "sort_key", "proposal", "index_sources", "index_layouts", "sessions")}
                    for r in sorted(cands, key=lambda r: (r["stratum"], r["position"]))],
        proposed_seal_frames=sum(r["normal_readable"] for r in cands if r["proposal"] == "seal"),
        proposed_dev_frames_from_candidates=sum(r["normal_readable"] for r in cands if r["proposal"] != "seal"),
        note="proposal only: source / licence of every candidate to be confirmed by the user; a candidate moved to "
             "development changes the order of its stratum, so the selection is recomputed after confirmation",
        units=rows)
    Path(out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "units"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:6])
