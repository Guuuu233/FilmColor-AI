"""Who graded the 16 dsc6233 references that come only from the photo folder (T1: medium evidence, content match)?

  t1b_dsc6233_grader.py PHOTO_ROOT OUT_JSON

Compares exiftool metadata (no pixels) of the 22 shop references of the S3 dev manifest (material folder) with the
38 photos of the photo-folder export of the roll: Photoshop document / instance ids, save time, software, size.
Same document and instance id with different bytes = the same Photoshop export saved again; the remaining photos
are placed by their save time within that export. Read-only on both folders.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROLL, FOLDER = "own-e5207-20240401-dsc6233", "#013 5207 兴庆公园2023"
KEYS = ("EXIF:ModifyDate", "EXIF:Software", "XMP:DocumentID", "XMP:InstanceID", "XMP:OriginalDocumentID",
        "File:ImageWidth", "File:ImageHeight", "ICC_Profile:ProfileDescription")


def meta(paths):
    out = subprocess.run(["exiftool", "-json", "-G0", "-q", "-q", *(f"-{k.split(':')[1]}" for k in KEYS), *map(str, paths)],
                         capture_output=True, text=True, check=True).stdout
    return {Path(d["SourceFile"]).name: {k: d.get(k) for k in KEYS} for d in json.loads(out)}


def main(photo_root, out_json):
    dev = [f for f in json.loads((REPO / "data/manifests/s3_dev_frames.json").read_text())["frames"] if f["roll"] == ROLL]
    refs = meta(sorted(f["ref"] for f in dev))
    folder = Path(photo_root) / FOLDER
    photos = meta(sorted(p for p in folder.iterdir() if p.suffix.lower() == ".jpg"))
    match = {Path(p["photo"]).name: p for p in json.loads((REPO / "experiments/s3_photos_audit/match.json").read_text())["photos"]
             if p["photo"].startswith(FOLDER + "/")}
    ref_by_id = {Path(f["ref"]).stem: Path(f["ref"]).name for f in dev}
    rows = []
    for name, m in sorted(photos.items()):
        best = match[name]["best"]
        r = refs.get(ref_by_id.get(best, ""), None)
        rows.append(dict(photo=name, best_match=best, r=round(match[name]["r"], 4), save_time=m["EXIF:ModifyDate"],
                         is_dev_ref=r is not None,
                         same_document=r is not None and all(m[k] == r[k] for k in ("XMP:DocumentID", "XMP:InstanceID")),
                         same_save_time=r is not None and m["EXIF:ModifyDate"] == r["EXIF:ModifyDate"]))
    dev_rows = [x for x in rows if x["is_dev_ref"]]
    other = [x for x in rows if not x["is_dev_ref"]]
    t_dev = sorted(x["save_time"] for x in dev_rows)
    common = lambda d, k: sorted({str(v[k]) for v in d.values()})
    res = dict(
        roll=ROLL, folder=FOLDER, photos=len(rows), dev_refs=len(refs),
        photos_matching_a_dev_ref=len(dev_rows), same_document_and_instance=sum(x["same_document"] for x in dev_rows),
        same_save_time=sum(x["same_save_time"] for x in dev_rows),
        other_photos=len(other),
        other_inside_dev_save_span=sum(t_dev[0] <= x["save_time"] <= t_dev[-1] for x in other),
        dev_save_span=[t_dev[0], t_dev[-1]],
        software=dict(refs=common(refs, "EXIF:Software"), photos=common(photos, "EXIF:Software")),
        size=dict(refs=sorted({f"{v['File:ImageWidth']}x{v['File:ImageHeight']}" for v in refs.values()}),
                  photos=sorted({f"{v['File:ImageWidth']}x{v['File:ImageHeight']}" for v in photos.values()})),
        rows=sorted(rows, key=lambda x: x["save_time"]),
        reading="same Photoshop export: evidence that the 16 other photos were graded in the same session as the 22 "
                "shop references; whether by the same person is an inference, not shown by metadata")
    Path(out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
