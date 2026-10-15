"""Deterministic BlueNeg cross-device sample (S3-3), fixed before any frame was viewed.

Rule: blue-intact frames that have a printed photo. (1) One frame per roll: the frame with the
median number_in_roll (lower median), except rolls already sampled in blueneg_sample3, which keep
that frame. (2) For the EXTRA_ROLLS largest rolls, one more frame whose scene class
(indoor/outdoor x day/night from meta.json) is not yet covered in that roll, preferring night, then
indoor; ties by number_in_roll.

usage: select_blueneg.py meta.json out.json
"""
import json
import sys
from collections import defaultdict

ALREADY = {"19960808B-03-new-orleans-siggraph96", "199807A-06-hk", "199910-01-korea"}
EXTRA_ROLLS = 4


def scene(r):
    s = r["scene_property"]
    return s["is_indoor"], s["is_daytime"]


def pref(r):
    ind, day = scene(r)
    return (day != "night", ind != "indoor", int(r["number_in_roll"]))


def main():
    meta, out = sys.argv[1], sys.argv[2]
    rows = [r for r in json.load(open(meta)) if r["partition"] == "blue-intact" and r["printed_path"]]
    rolls = defaultdict(list)
    for r in rows:
        rolls[r["roll_id"]].append(r)
    picked = []
    for rid in sorted(rolls):
        rs = sorted(rolls[rid], key=lambda r: int(r["number_in_roll"]))
        have = [r for r in rs if r["filename"] in ALREADY]
        first = have[0] if have else rs[(len(rs) - 1) // 2]
        picked.append(dict(first, pick="roll_median" if not have else "already_downloaded"))
    largest = sorted(rolls, key=lambda k: (-len(rolls[k]), k))[:EXTRA_ROLLS]
    for rid in largest:
        taken = {p["filename"] for p in picked}
        covered = {scene(p) for p in picked if p["roll_id"] == rid}
        cands = [r for r in rolls[rid] if r["filename"] not in taken and scene(r) not in covered]
        if cands:
            picked.append(dict(min(cands, key=pref), pick="extra_scene"))
    json.dump(dict(rule=__doc__.split("usage:")[0].strip(), n=len(picked), frames=picked), open(out, "w"),
              indent=1, ensure_ascii=False)
    for p in picked:
        print(p["filename"], p["film_type"], scene(p), p["pick"])
    print(len(picked), "frames,", len({p["roll_id"] for p in picked}), "rolls,",
          len({p["film_type"] for p in picked}), "film types")


if __name__ == "__main__":
    main()
