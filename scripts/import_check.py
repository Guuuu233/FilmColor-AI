"""Check files before they enter the material folder (T1b review 2026-10-22, item 4): nothing frozen gets in under
a new name, and every file records where it really came from. Reads no pixels.

  import_check.py STAGING_DIR SOURCES_CSV [--material-sha SHA256_TXT] [--out OUT_JSON]

SOURCES_CSV has one row per file of STAGING_DIR: file (path relative to STAGING_DIR), original_name, roll, source,
licence.
  source   the download link or the provider (a person, a group, a site), as it really was
  licence  what that source allows for research use; the index template "群友提供，已明确授权" is refused when the
           source is a link (a web download is not a group member's grant)
Refused (exit status 1): SHA256 in a frozen manifest; original name or roll id on the frozen block lists (the roll is
also derived from BlueNeg-style names, <date><letter>-<frame>-<place>); a file without a row or a row without a file;
an empty source or licence; the template licence with a link as source.
Listed, not refused: byte copies of a file already in the material list given with --material-sha.
"""
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from filmcolor import frozen  # noqa: E402

TEMPLATE = "群友提供，已明确授权"
LINK = re.compile(r"^(https?://|www\.|[\w-]+(\.[\w-]+)+/)")
ROLL_RE = re.compile(r"^(\d{6,8})([A-Z]?)-(\d+)-")


def rolls_of(name, roll):
    out = {r for r in [roll] if r}
    m = ROLL_RE.match(Path(name or "").name)
    if m:
        out |= {m.group(1) + m.group(2), f"ds-{m.group(1)[:6]}{m.group(2)}"}
    return out


def check(staging, rows, fz=None, blocks=None, material=None):
    """Per-file verdicts; ``fz`` / ``blocks`` default to every frozen manifest, ``material`` maps sha256 -> path."""
    staging = Path(staging)
    fz = frozen.frozen_hashes() if fz is None else fz
    blocks = frozen.frozen_blocks() if blocks is None else blocks
    material = material or {}
    on_disk = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*")
                     if p.is_file() and not p.name.startswith("."))
    by_file = {r["file"]: r for r in rows}
    res = []
    for f in sorted(set(on_disk) | set(by_file)):
        r, why = by_file.get(f), []
        if f not in on_disk:
            res.append(dict(file=f, refused=["row without a file"]))
            continue
        sha = frozen.sha256_file(staging / f)
        if sha in fz:
            why.append(f"frozen by hash: {fz[sha]}")
        if r is None:
            why.append("no row in the sources list")
            r = {}
        names = {frozen.stem_of(f), frozen.stem_of(r.get("original_name") or "")} - {""}
        why += [f"frozen by name: {blocks['names'][n]}" for n in sorted(names) if n in blocks["names"]]
        why += [f"frozen roll: {blocks['rolls'][x]}" for x in sorted(rolls_of(r.get("original_name"), r.get("roll")))
                if x in blocks["rolls"]]
        src, lic = (r.get("source") or "").strip(), (r.get("licence") or "").strip()
        if r and not src:
            why.append("no source")
        if r and not lic:
            why.append("no licence")
        if lic == TEMPLATE and LINK.match(src):
            why.append("template licence with a link as source: record what the link's owner allows")
        res.append(dict(file=f, sha256=sha, refused=why, already_in_material=material.get(sha)))
    return res


def main(argv):
    a = [x for x in argv if not x.startswith("--")]
    opt = {k: argv[argv.index(k) + 1] for k in ("--material-sha", "--out") if k in argv}
    staging, sources = a[0], a[1]
    rows = list(csv.DictReader(open(sources, encoding="utf-8-sig")))
    material = {}
    if "--material-sha" in opt:
        for line in Path(opt["--material-sha"]).read_text().splitlines():
            h, r = line.split("  ", 1)
            material.setdefault(h, r.removeprefix("./"))
    res = check(staging, rows, material=material)
    out = dict(files=len(res), refused=sum(bool(r["refused"]) for r in res),
               already_in_material=sum(bool(r.get("already_in_material")) for r in res), results=res)
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if "--out" in opt:
        Path(opt["--out"]).write_text(text + "\n")
    print(text)
    return 1 if out["refused"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
