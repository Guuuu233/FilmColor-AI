"""Freeze whole rolls of the external material folder as a held-out test set (manifest only).

Read-only on ROOT; writes one manifest. A roll is identified by its source folder (index column 原路径),
its camera file-number range and its capture time, never by film type alone. Frozen are: every file
of the roll (negatives, the roll-label frame, finished prints, sidecars), plus every file elsewhere
that may show the same frame - a negative or print listed as a near-duplicate of a frozen one, or a
print whose best-matching negative is frozen (audit_ext_pairing.py). Over-freezing is the safe side.
SHA256 is recomputed for every frozen file and must equal the inventory.

  freeze_ext_rolls.py ROOT INVENTORY_JSON PAIRING_JSON OUT_JSON
"""
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

ROLLS = {  # user decision 2026-10-17
    "own-e5219-20240323-dsc8202": dict(film="Eastman 5219", source_dir="Eastman 5219(ecn-2次抛冲洗 尼康数字化)-"),
    "own-e5207-20240401-dsc6199": dict(film="Eastman 5207", source_dir="Eastman 5207(ecn-2次抛冲洗 尼康数字化)-2"),
}
PRINT_R = 0.75  # a print elsewhere may be the same frame from this similarity on (confirmed pairs: 0.77-0.99)
# pairs the thumbnail check could not find, settled by eye; scored=False keeps them out of the test metrics
MANUAL = {"R119": dict(negative="A072", scored=False,
                       note="last frame of the roll runs into the leader; the print is a 4236x5300 portrait crop of "
                            "its left part. Confirmed by eye 2026-10-17. The index pair A066 is wrong (A066 -> R095)")}
EYE_CHECK = ("2026-10-17: every print of both rolls checked against its negative on contact sheets "
             "(experiments/s3_data_audit/outputs/sheets/frozen_*.jpg); all automatic pairs confirmed")


def sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def main(root, inv_path, pairing_path, out_path):
    root = Path(root)
    inv = {r["id"]: r for r in json.load(open(inv_path)) if r.get("id")}
    pairing = json.load(open(pairing_path))
    by_dir = {v["source_dir"]: k for k, v in ROLLS.items()}
    frozen = {i: dict(roll=by_dir[r["orig_dir"]], why="file of the roll")
              for i, r in inv.items() if r.get("orig_dir") in by_dir}
    for _ in range(3):  # follow duplicate chains a few steps
        for d in pairing["negative_near_duplicates"] + pairing["ref_near_duplicates"]:
            for a, b in ((d["a"], d["b"]), (d["b"], d["a"])):
                if a in frozen and b not in frozen:
                    frozen[b] = dict(roll=frozen[a]["roll"], why=f"near-duplicate of {a} (r={d['r']:.3f})")
        for p in pairing["pairs"]:
            if p["best"] in frozen and p["ref"] not in frozen and p["r"] >= PRINT_R:
                frozen[p["ref"]] = dict(roll=frozen[p["best"]]["roll"],
                                        why=f"print whose best negative is {p['best']} (r={p['r']:.3f})")
    pair_of = {p["ref"]: p for p in pairing["pairs"]}
    files = []
    for i in sorted(frozen):
        r = inv[i]
        h = sha256(root / r["rel"])
        if h != r["sha256"]:
            raise SystemExit(f"{i}: sha256 differs from the inventory ({h} != {r['sha256']})")
        p = pair_of.get(i)
        pair = None
        if i in MANUAL:
            pair = dict(negative=MANUAL[i]["negative"], how="eye", scored=MANUAL[i]["scored"], note=MANUAL[i]["note"])
        elif p and p["mutual"] and inv[p["best"]]["orig_dir"] == r["orig_dir"]:
            pair = dict(negative=p["best"], how="thumbnail mutual best, confirmed by eye", scored=True)
        files.append(dict(id=i, roll=frozen[i]["roll"], why=frozen[i]["why"], group=r["group"], rel=r["rel"],
                          orig_dir=r["orig_dir"], orig_name=r["orig_name"], bytes=r["bytes"], sha256=h,
                          captured=r.get("DateTimeOriginal", ""), index_pair=r.get("index_pair", ""),
                          thumbnail_check=(dict(negative=p["best"], r=round(p["r"], 4), margin=round(p["margin"], 4),
                                                mutual=p["mutual"]) if p else None),
                          pair=pair))
    rolls = []
    for k, v in ROLLS.items():
        own = [inv[f["id"]] for f in files if f["roll"] == k and f["why"] == "file of the roll"]
        negs = [x for x in own if x["group"] in ("A", "B")]
        nums = sorted(int(m.group(1)) for x in negs if (m := re.search(r"_(\d+)\.", x["orig_name"])))
        times = sorted(x["DateTimeOriginal"] for x in negs if x.get("DateTimeOriginal"))
        label = [x["id"] for x in negs if x["ext"] == "JPG"]
        pairs = [f for f in files if f["roll"] == k and f["pair"]]
        paired_negs = {f["pair"]["negative"] for f in pairs}
        rolls.append(dict(roll_id=k, **v, camera="Nikon Z 7_2 (lab digitisation)", file_numbers=[nums[0], nums[-1]],
                          captured=[times[0], times[-1]], label_frame=label, negatives=len(negs),
                          prints=sum(x["group"] == "R" for x in own), pairs=len(pairs),
                          scored_pairs=sum(f["pair"]["scored"] for f in pairs),
                          negatives_without_print=sorted(x["id"] for x in negs if x["id"] not in paired_negs),
                          other_files=sum(x["group"] not in ("A", "B", "R") for x in own)))
    out = dict(id="frozen_test_v1", created=date.today().isoformat(),
               decision="user 2026-10-17: freeze Eastman 5219 (37 pairs) and the Eastman 5207 roll with 32 pairs",
               rule=("Held-out test set. These files (and any other file with one of these sha256) must not be used "
                     "for training, tuning, model selection, fixed-look fitting, failure analysis or label-config "
                     "choices. Opened once per pre-registered experiment, after the model is frozen, and the opening "
                     "is recorded with the commit of the frozen model."),
               inventory=Path(inv_path).name, inventory_sha256=sha256(inv_path),
               pairing=Path(pairing_path).name, pairing_sha256=sha256(pairing_path),
               eye_check=EYE_CHECK, rolls=rolls, files=files)
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    for r in rolls:
        print(r)
    print(len(files), "files;", sum(f["why"] != "file of the roll" for f in files), "added as possible duplicates")


if __name__ == "__main__":
    main(*sys.argv[1:5])
