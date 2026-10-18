"""Guard for held-out test data (docs/S3_ML_DESIGN.md 5.4).

Frozen files are identified by the SHA256 of their content (data/manifests/frozen_test_*.json), so a copy, a
renamed file or a symlink is caught like the original. Derived caches (P8, R8, features, labels, model inputs)
are written with a sidecar that lists the SHA256 of their source files; reading a cache checks those sources,
and a cache without a sidecar is refused. Only the final-test entry point passes ``allow_frozen=True``.
Frames of a sealed dataset roll that are not on disk yet (no local hash) are also blocked by name and roll id
(``blocked`` in the manifest): downloaders, readers and ``check_source`` refuse them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MANIFESTS = sorted((Path(__file__).resolve().parents[2] / "data" / "manifests").glob("frozen_test_*.json"))
SIDECAR = ".sources.json"


class FrozenDataError(RuntimeError):
    """A held-out test file (or something derived from one) was about to be used."""


def sha256_file(path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:  # open() follows symlinks: the content is what is hashed
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def frozen_hashes(manifests=None) -> dict:
    """sha256 -> "<manifest id>:<file id>" for every file of the given (default: all) frozen manifests."""
    out = {}
    for m in (MANIFESTS if manifests is None else manifests):
        d = json.loads(Path(m).read_text())
        out |= {f["sha256"]: f"{d['id']}:{f['id']}" for f in d["files"]}
    if manifests is None and not out:
        raise FrozenDataError("no frozen manifest found: refusing to run without the guard")
    return out


def check_hashes(shas, *, allow_frozen: bool = False, frozen=None) -> None:
    """Raise if any of ``shas`` is frozen (unless ``allow_frozen``)."""
    frozen = frozen_hashes() if frozen is None else frozen
    hit = sorted({frozen[s] for s in shas if s in frozen})
    if hit and not allow_frozen:
        raise FrozenDataError(f"held-out test data: {', '.join(hit[:5])}{' ...' if len(hit) > 5 else ''}")


def frozen_blocks(manifests=None) -> dict:
    """Name-level guard for frozen data that may not be on disk yet (no local hash): ``{"names": {stem: label},
    "rolls": {roll: label}}`` from the optional ``blocked`` field of the manifests (``names``: dataset file stems,
    e.g. BlueNeg ``19960815F-23-san-francisco``; ``roll_ids``: every spelling of a sealed roll)."""
    names, rolls = {}, {}
    for m in (MANIFESTS if manifests is None else manifests):
        d = json.loads(Path(m).read_text())
        b = d.get("blocked") or {}
        names |= {n: f"{d['id']}:{n}" for n in b.get("names", [])}
        rolls |= {r: f"{d['id']}:{r}" for r in b.get("roll_ids", [])}
    return dict(names=names, rolls=rolls)


def stem_of(path) -> str:
    """Dataset stem of a file name: everything before the first dot (``x.preview.png`` -> ``x``)."""
    return Path(path).name.split(".", 1)[0]


def check_name(path, *, allow_frozen: bool = False, blocks=None) -> None:
    """Raise if the file name (or a URL / dataset path) belongs to a frozen dataset frame."""
    blocks = frozen_blocks() if blocks is None else blocks
    hit = blocks["names"].get(stem_of(path))
    if hit and not allow_frozen:
        raise FrozenDataError(f"held-out test data (by name): {hit}")


def check_roll(roll, *, allow_frozen: bool = False, blocks=None) -> None:
    """Raise if ``roll`` is a sealed roll (any registered spelling)."""
    blocks = frozen_blocks() if blocks is None else blocks
    hit = blocks["rolls"].get(str(roll))
    if hit and not allow_frozen:
        raise FrozenDataError(f"held-out test roll: {hit}")


def check_source(path, *, allow_frozen: bool = False, frozen=None) -> str:
    """SHA256 of a source file (negative or reference), after checking it is not frozen (by name and by
    content); pass the result to ``save_cache`` as provenance."""
    check_name(path, allow_frozen=allow_frozen)
    sha = sha256_file(path)
    check_hashes([sha], allow_frozen=allow_frozen, frozen=frozen)
    return sha


def save_cache(path, array, sources) -> None:
    """``np.save`` plus a sidecar listing the SHA256 of the source files it was derived from."""
    path = Path(path)
    if path.suffix != ".npy":  # np.save would append .npy and the sidecar would not match
        raise ValueError(f"{path}: cache path must end in .npy")
    np.save(path, array)
    Path(str(path) + SIDECAR).write_text(json.dumps(dict(sources=sorted(sources))))


def load_cache(path, *, allow_frozen: bool = False, frozen=None):
    """``np.load`` of a cache written by ``save_cache``; refuses caches without provenance or with a frozen
    source."""
    path = Path(path)
    side = Path(str(path) + SIDECAR)
    if not side.exists():
        raise FrozenDataError(f"{path}: cache without source record, not read")
    check_hashes(json.loads(side.read_text())["sources"], allow_frozen=allow_frozen, frozen=frozen)
    return np.load(path)
