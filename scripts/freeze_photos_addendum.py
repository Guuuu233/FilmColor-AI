"""Addendum to frozen_test_v1: copies of the frozen frames found in another photo folder (other exports, the
user's own edits), by content (audit_photos_match.py). A folder counts as a copy of a frozen roll when at least
FOLDER_SHARE of its photos match frames of that roll with r >= FOLDER_R; every photo of such a folder whose best
match is a frozen frame is frozen, by SHA256. Candidates outside such folders were checked by eye and are listed
as rejected (different pictures with a similar light layout), not frozen.

  freeze_photos_addendum.py MATCH_JSON PHOTO_ROOT OUT_JSON
"""
import json
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

FOLDER_SHARE, FOLDER_R = 0.8, 0.9
EYE_CHECK = "2026-10-17: every rejected candidate looked at next to its matched frozen frame; none is the same frame"


def main(match_json, photo_root, out_json):
    m = json.load(open(match_json))
    by_folder = defaultdict(list)
    for x in m["photos"]:
        by_folder[x["photo"].rsplit("/", 1)[0] if "/" in x["photo"] else "."].append(x)
    copies = {}
    for folder, xs in by_folder.items():
        c = Counter(x["frozen_roll"] for x in xs if x["frozen_roll"] and x["r"] >= FOLDER_R)
        if c and c.most_common(1)[0][1] >= FOLDER_SHARE * len(xs):
            copies[folder] = c.most_common(1)[0][0]
    files, rejected = [], []
    for folder, xs in sorted(by_folder.items()):
        for x in xs:
            if not x["frozen_roll"]:
                continue
            rec = dict(id=f"photos:{x['photo']}", sha256=x["sha256"], matched_frame=x["best"], r=round(x["r"], 4))
            (files if folder in copies else rejected).append(dict(rec, roll=x["frozen_roll"]))
    out = dict(id="frozen_test_v1_photos", created=date.today().isoformat(), addendum_to="frozen_test_v1",
               source=str(Path(photo_root)), rule=__doc__.split("  freeze_photos_addendum.py")[0].strip(),
               folders=copies, eye_check=EYE_CHECK, files=files, rejected_candidates=rejected)
    Path(out_json).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(copies, len(files), "frozen,", len(rejected), "rejected")


if __name__ == "__main__":
    main(*sys.argv[1:4])
