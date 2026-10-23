"""How many negative frames the reference-print folder 04_已反转成品_对照答案 covers (question of 2026-10-23: the
user sees 734 items there). Reads no pixels: file count on disk, then the 10-18 catalog and content pairing.

  ref_folder_count.py MATERIAL_ROOT DATASET_DIR OUT_JSON

DATASET_DIR = experiments/s3_ext_audit/dataset_20261018 (catalog.json is git-ignored; its sha256 is written out).
A print is linked to a negative frame by any of: the negative's partner ids (index pairing), the print's partner
ids, a pairing.json pair of level "content". Frozen frames are counted, not opened.
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REF = "04_已反转成品_对照答案"


def main(mat, dataset_dir, out_json):
    mat, d = Path(mat), Path(dataset_dir)
    on_disk = [p for p in (mat / REF).iterdir()]
    cat_raw = (d / "catalog.json").read_bytes()
    cat, pr = json.loads(cat_raw), json.loads((d / "pairing.json").read_text())
    files = {f["rel"]: f for f in cat["files"]}
    byid = {f["id"]: f for f in cat["files"] if f["id"]}
    prints = [f for f in cat["files"] if f["rel"].startswith(REF + "/")]
    neg_frame = {r: fr for fr in cat["frames"] for r in fr["files"]}
    fr_by = {fr["frame"]: fr for fr in cat["frames"]}

    link = defaultdict(set)
    for fr in cat["frames"]:
        for r in fr["files"]:
            for p in files[r]["partners"]:
                if p in byid and byid[p]["rel"].startswith(REF + "/"):
                    link[byid[p]["rel"]].add(fr["frame"])
    for f in prints:
        for p in f["partners"]:
            if p in byid and byid[p]["rel"] in neg_frame:
                link[f["rel"]].add(neg_frame[byid[p]["rel"]]["frame"])
    for p in pr["pairs"]:
        if p["level"] == "content" and p["ref"].startswith(REF + "/") and p["best"] in neg_frame:
            link[p["ref"]].add(neg_frame[p["best"]]["frame"])

    linked = [f for f in prints if link[f["rel"]]]
    unlinked = [f for f in prints if not link[f["rel"]]]
    covered = defaultdict(set)
    for f in linked:
        for x in link[f["rel"]]:
            covered[x].add(f["rel"])
    with_ref = [fr for fr in cat["frames"] if fr["pair"]]
    res = dict(
        catalog_sha256=hashlib.sha256(cat_raw).hexdigest(),
        on_disk=dict(entries=len(on_disk), files=sum(p.is_file() for p in on_disk),
                     dirs=sum(p.is_dir() for p in on_disk), hidden=sum(p.name.startswith(".") for p in on_disk)),
        catalog_prints=len(prints), distinct_contents=len({f["sha256"] for f in prints}),
        prints_linked=len(linked),
        frames_covered=len(covered),
        frames_covered_by_kind=dict(Counter(fr_by[x]["kind"] for x in covered)),
        frames_covered_frozen=sum(bool(fr_by[x]["frozen"]) for x in covered),
        prints_per_covered_frame=dict(sorted(Counter(len(v) for v in covered.values()).items())),
        prints_unlinked=len(unlinked),
        unlinked_byte_duplicates=sum(1 for f in unlinked if f["dup_of"]),
        unlinked_other_by_source=dict(Counter(f["source"].split(":")[0] for f in unlinked if not f["dup_of"])),
        frames_with_reference=len(with_ref), frames_with_reference_by_kind=dict(Counter(fr["kind"] for fr in with_ref)),
        frames_with_reference_none_in_folder=dict(Counter(fr["pair"] for fr in with_ref if fr["frame"] not in covered)))
    Path(out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
