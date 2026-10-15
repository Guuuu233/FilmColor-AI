"""Download the negative DNG and printed TIFF of each selected BlueNeg frame and verify them against
the Hugging Face LFS records (sha256 + size). Files land in DATA_ROOT under the dataset's own
relative paths; a file is only renamed into place after its hash matches. transformations.pkl is
never fetched (pickle executes code on load).

usage: fetch_blueneg.py selection.json DATA_ROOT out.json
"""
import hashlib
import json
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = "https://huggingface.co/datasets/ttgroup/blueneg-release"
API = "https://huggingface.co/api/datasets/ttgroup/blueneg-release/paths-info/main"


def lfs_records(paths):
    data = urllib.parse.urlencode([("paths", p) for p in paths]).encode()
    with urllib.request.urlopen(urllib.request.Request(API, data=data), timeout=60) as r:
        info = json.load(r)
    return {e["path"]: e for e in info}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    sel, root, out = json.load(open(sys.argv[1])), Path(sys.argv[2]), sys.argv[3]
    paths = [p for fr in sel["frames"] for p in (fr["negative_path"], fr["printed_path"])]
    recs = lfs_records(paths)
    results = []
    for rel in paths:
        rec = recs.get(rel)
        if rec is None or "lfs" not in rec:
            results.append(dict(path=rel, status="no LFS record"))
            continue
        oid, size = rec["lfs"]["oid"], rec["lfs"]["size"]
        dst = root / rel
        if dst.exists() and dst.stat().st_size == size and sha256(dst) == oid:
            results.append(dict(path=rel, size=size, sha256=oid, status="present, verified"))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        part = dst.with_name(dst.name + ".part")
        url = f"{REPO}/resolve/main/{urllib.parse.quote(rel)}"
        rc = subprocess.run(["curl", "-fsSL", "--retry", "3", "-C", "-", "-o", str(part), url]).returncode
        got = sha256(part) if part.exists() else None
        ok = rc == 0 and got == oid and part.stat().st_size == size
        if ok:
            part.rename(dst)
        results.append(dict(path=rel, size=size, sha256=oid, downloaded_sha256=got, curl_rc=rc,
                            status="downloaded, verified" if ok else "FAILED"))
        print(results[-1]["status"], rel, flush=True)
    json.dump(dict(source=REPO, files=results), open(out, "w"), indent=1)
    bad = [r for r in results if not r["status"].endswith("verified")]
    print(len(results) - len(bad), "verified,", len(bad), "not verified")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
