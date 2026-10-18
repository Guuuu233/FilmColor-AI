"""Extend frozen_test_v2_ds with the official BlueNeg files of its sealed rolls, without downloading them.

From the BlueNeg meta.json (data/raw/public/blueneg): every frame whose roll (<yyyymm><letter>, see
audit_ext_new_rolls.py) is sealed. Adds to the manifest
  blocked.names     the frame stems (filmcolor.frozen.check_name refuses them in downloaders and readers)
  blocked.roll_ids  the sealed roll keys and the dataset's own roll_id spellings (<yyyymmdd><letter>)
  files             one entry per official file (negative DNG, print TIFF, previews, pseudo-GT) with the
                    sha256 of its Hugging Face LFS record, so a later download is caught by content as well.
Consistency check: every print already sealed through the group pack must have the LFS sha256 of the
official print of the same name. Idempotent: earlier official entries are replaced.

  freeze_ds_blueneg_official.py MANIFEST_JSON META_JSON
"""
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://huggingface.co/api/datasets/ttgroup/blueneg-release/paths-info/main"
KINDS = ("negative_path", "printed_path", "preview_path", "pseudogt_path")


def roll_of(x):
    rid = x["roll_id"]
    return "ds-" + (x["date"][:6] + rid[-1] if rid[-1].isalpha() else rid)


def lfs(paths, batch=50):
    out = {}
    for i in range(0, len(paths), batch):
        data = urllib.parse.urlencode([("paths", p) for p in paths[i:i + batch]]).encode()
        with urllib.request.urlopen(urllib.request.Request(API, data=data), timeout=60) as r:
            out |= {e["path"]: e for e in json.load(r)}
    return out


def main(manifest, meta_json):
    m = json.loads(Path(manifest).read_text())
    sealed = set(m["rule"]["sealed"])
    frames = [x for x in json.loads(Path(meta_json).read_text()) if roll_of(x) in sealed]
    paths = [x[k] for x in frames for k in KINDS if x.get(k)]
    recs = lfs(paths)
    missing = [p for p in paths if p not in recs or "lfs" not in recs[p]]
    if missing:
        raise SystemExit(f"no LFS record for {len(missing)} paths, e.g. {missing[:3]}")

    m["files"] = [f for f in m["files"] if not f["id"].startswith("hf:")]
    have = {f["sha256"] for f in m["files"]}
    by_orig = {Path(f["orig_name"]).stem: f for f in m["files"] if f.get("orig_name") and f["rel"].startswith("04_")}
    mismatch, matched = [], 0
    for x in frames:
        for k in KINDS:
            p = x.get(k)
            if not p:
                continue
            sha = recs[p]["lfs"]["oid"]
            if k == "printed_path" and x["filename"] in by_orig:
                if by_orig[x["filename"]]["sha256"] != sha:
                    mismatch.append(x["filename"])
                else:
                    matched += 1
            m["files"].append(dict(id=f"hf:{p}", roll=roll_of(x), why=f"official BlueNeg {k[:-5]} of a sealed roll "
                                   "(Hugging Face LFS sha256, not downloaded)", rel=None, orig_name=Path(p).name,
                                   pair=None, bytes=recs[p]["lfs"]["size"], sha256=sha))
    m["blocked"] = dict(
        names=sorted({x["filename"] for x in frames}),
        roll_ids=sorted(sealed | {x["roll_id"] for x in frames}),
        source="BlueNeg meta.json; official files listed in files with id hf:<path>",
        # group-pack prints whose content matches no official print (smaller files: re-saved or other versions);
        # both versions stay frozen
        group_prints_not_official=sorted(mismatch))
    n_new = sum(f["id"].startswith("hf:") and f["sha256"] not in have for f in m["files"])
    m["counts"] = dict(m["counts"], official_files=sum(f["id"].startswith("hf:") for f in m["files"]),
                       official_frames=len(frames))
    Path(manifest).write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n")
    print("sealed frames in meta.json", len(frames), "official files", len(paths), "new hashes", n_new,
          "prints matching the group pack", matched)


if __name__ == "__main__":
    main(*sys.argv[1:3])
