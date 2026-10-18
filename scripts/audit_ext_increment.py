"""Read-only incremental audit of the external material folder against an earlier SHA256 list.

Nothing under ROOT is written. Classifies every current file as unchanged (same path, same content),
moved (content known, new path), new (content never seen) and every earlier path as kept, changed
(same path, different content) or gone; then checks that every file the dev and frozen manifests name
still sits at its recorded path with its recorded hash.

  audit_ext_increment.py ROOT OLD_SHA256_TXT OUT_DIR
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "data" / "manifests"


def sha256(path, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def read_list(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        h, rel = line.split("  ", 1)
        out[rel.removeprefix("./")] = h
    return out


def main(root, old_txt, out_dir):
    root, out = Path(root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.name.startswith("."))
    rels = [p.relative_to(root).as_posix() for p in files]
    with ThreadPoolExecutor(4) as ex:
        hashes = list(ex.map(sha256, files))
    now = dict(zip(rels, hashes))
    (out / "sha256.txt").write_text("".join(f"{h}  ./{r}\n" for r, h in now.items()))

    old = read_list(old_txt)
    old_by_hash = defaultdict(list)
    for r, h in old.items():
        old_by_hash[h].append(r)
    now_hashes = set(now.values())

    unchanged = [r for r, h in now.items() if old.get(r) == h]
    moved = {r: old_by_hash[h] for r, h in now.items() if old.get(r) != h and h in old_by_hash}
    new = [r for r, h in now.items() if h not in old_by_hash]
    changed = {r: dict(old=old[r], now=now[r]) for r in old if r in now and now[r] != old[r]}
    gone = [r for r, h in old.items() if h not in now_hashes]

    by_hash = defaultdict(list)
    for r in new:
        by_hash[now[r]].append(r)
    top = lambda r: r.split("/", 1)[0]
    ext = lambda r: Path(r).suffix.lower()

    # manifests: every named file must still be at its recorded path with its recorded hash
    checks = []
    dev = json.loads((MANIFESTS / "s3_dev_frames.json").read_text())
    for fr in dev["frames"]:
        for k in ("raw", "ref"):
            p, h = fr.get(k), fr.get(f"{k}_sha256")
            if not p or not h:
                continue
            q = Path(p) if Path(p).is_absolute() else REPO / p
            try:
                rel = q.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                continue  # outside the material folder; not this audit's business
            checks.append(dict(manifest="s3_dev_frames", id=fr["id"], kind=k, rel=rel,
                               ok=now.get(rel) == h, at=[r for r, x in now.items() if x == h]))
    for m in sorted(MANIFESTS.glob("frozen_test_*.json")):
        for f in json.loads(m.read_text())["files"]:
            rel = f.get("rel")
            if rel is None or rel not in old:
                continue  # photo-folder addendum etc.
            checks.append(dict(manifest=m.stem, id=f["id"], kind="frozen", rel=rel,
                               ok=now.get(rel) == f["sha256"], at=[r for r, x in now.items() if x == f["sha256"]]))

    res = dict(
        root=str(root), old_list=str(old_txt), n_now=len(now), n_old=len(old),
        unchanged=len(unchanged), moved=moved, changed=changed, gone=gone,
        new_files=len(new), new_unique=len(by_hash),
        new_dups={h: v for h, v in by_hash.items() if len(v) > 1},
        new_by_top=Counter(top(r) for r in new), new_by_top_ext=Counter(f"{top(r)} {ext(r)}" for r in new),
        manifest_checks=dict(n=len(checks), bad=[c for c in checks if not c["ok"]]),
        new=new,
    )
    (out / "increment.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("new", "new_dups", "moved", "changed", "gone")},
                     ensure_ascii=False, indent=1))
    print("moved", len(moved), "changed", len(changed), "gone", len(gone), "new_dup_groups", len(res["new_dups"]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
