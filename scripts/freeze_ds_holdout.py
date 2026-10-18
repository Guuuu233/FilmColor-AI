"""Seal untouched rolls of the 2026-10-18 group-pack scan dataset as a held-out test set (manifest only).

Read-only on ROOT; writes one manifest that filmcolor.frozen loads automatically (frozen_test_*.json).

The dataset is the public BlueNeg release (Tien-Tsin Wong; prints byte-identical to data/raw/public/blueneg,
negatives re-encoded to 4000 px TIFF by the group member). Candidates: dataset rolls (audit_ext_new_rolls.py
key <yyyymm><letter>) with >= MIN_PAIRED unique paired frames that nobody has touched: no frame in
s3_dev_frames.json (blueneg_xdev27 took 1-4 frames from 14 real rolls) and no frame viewed (VIEWED: one
frame shown on a contact sheet, 2026-10-18). All candidates are sealed.

First run (same day, never used): the rule sealed half of 13 candidates by sha256(SALT + roll) without
the dev-manifest exclusion; the frozen guard test caught two contaminated rolls (199412D, 199412E).

Frozen per sealed roll: every negative whose original name belongs to the roll, every print whose original
name belongs to the roll, every index pair partner of those, and every other path with identical content.
Over-freezing is the safe side. SHA256 comes from the incremental audit list and is re-checked here.

  freeze_ds_holdout.py ROOT NEW_FILES_JSON NEW_ROLLS_JSON SHA256_TXT OUT_JSON
"""
import csv
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

MIN_PAIRED = 5
VIEWED = {"ds-199608D", "ds-199807A", "ds-199608I", "ds-199608J", "ds-199910", "ds-199412C", "ds-199608B",
          "ds-199412B"}
ROLL_RE = re.compile(r"^(\d{6,8}[A-Z]?)-(\d+)-")


def roll_of(orig):
    m = ROLL_RE.match(orig or "")
    if not m:
        return None
    k = m.group(1)
    return "ds-" + (k[:6] + k[-1] if k[-1].isalpha() else k)


def sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def main(root, files_json, rolls_json, sha_txt, out_path):
    root = Path(root)
    rolls = json.loads(Path(rolls_json).read_text())
    dev = json.loads((Path(__file__).resolve().parents[1] / "data/manifests/s3_dev_frames.json").read_text())
    dev_used = {roll_of(f["stem"] + "-") for f in dev["frames"] if f["domain"] == "blueneg"}
    cand = sorted(k for k, v in rolls.items()
                  if v["known"] and v["paired"] >= MIN_PAIRED and k not in VIEWED and k not in dev_used)
    sealed = cand

    now = {}
    for line in Path(sha_txt).read_text().splitlines():
        h, rel = line.split("  ", 1)
        now[rel.removeprefix("./")] = h
    by_hash = {}
    for r, h in now.items():
        by_hash.setdefault(h, []).append(r)
    index = {f"{r['类型']}/{r['文件名']}": r for r in csv.DictReader(open(root / "索引.csv", encoding="utf-8-sig"))}
    by_id = {r["编号"]: k for k, r in index.items()}

    picked = {}  # rel -> why
    for rel, row in index.items():
        rk = roll_of(row["原文件名"])
        if rk in sealed and rel in now:
            picked[rel] = (rk, "negative of the roll" if rel.startswith("01_") else "file of the roll")
    for rel in list(picked):
        for pid in filter(None, (x.strip() for x in index[rel]["配对编号"].split(";"))):
            if by_id.get(pid) in now and by_id[pid] not in picked:
                picked[by_id[pid]] = (picked[rel][0], f"index pair of {index[rel]['编号']}")
    for rel in list(picked):
        for other in by_hash[now[rel]]:
            if other not in picked:
                picked[other] = (picked[rel][0], f"identical content to {rel}")

    files = []
    for rel in sorted(picked):
        assert sha256(root / rel) == now[rel], rel
        row = index.get(rel, {})
        files.append(dict(id=row.get("编号") or rel, roll=picked[rel][0], why=picked[rel][1], rel=rel,
                          orig_name=row.get("原文件名"), pair=row.get("配对编号") or None,
                          bytes=(root / rel).stat().st_size, sha256=now[rel]))
    n_neg = sum(f["rel"].startswith("01_") for f in files)
    out = dict(
        id="frozen_test_v2_ds", created=str(date.today()),
        decision="2026-10-18 review: keep untouched rolls of the new material for validating the next base-estimation "
                 "version; never read for development, threshold choice, failure analysis or fitting",
        rule=dict(min_paired=MIN_PAIRED, viewed_dev_rolls=sorted(VIEWED), dev_manifest_rolls=sorted(dev_used),
                  candidates=cand, sealed=sealed, seal="every untouched candidate"),
        source="BlueNeg release (https://huggingface.co/datasets/ttgroup/blueneg-release), via the group pack; "
               "credit 'Copyrighted by Tien-Tsin Wong'",
        caveat="all sealed rolls are Kodak Gold 100-5 from one 1996 trip: same photographer, scanner, conversion "
               "and film stock as dev rolls 199608B/D/I/J. A held-out set, not an independent population. "
               "Frames that exist elsewhere only as visually similar (non-identical) files are not detected; the "
               "original BlueNeg DNGs of these frames are frozen by name only if downloaded later.",
        counts=dict(files=len(files), negatives=n_neg, prints=len(files) - n_neg), files=files)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print("candidates", len(cand), "sealed", sealed)
    print(out["counts"])


if __name__ == "__main__":
    main(*sys.argv[1:6])
