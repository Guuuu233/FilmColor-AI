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
import os
from pathlib import Path

import numpy as np


def _default_manifests() -> list[Path]:
    env_dir = os.environ.get("FILMCOLOR_MANIFEST_DIR")
    candidates = [
        Path(env_dir) if env_dir else None,
        Path(__file__).resolve().parents[2] / "data" / "manifests",
        Path(__file__).resolve().parent / "_manifests",
    ]
    for d in candidates:
        if d is not None and d.is_dir():
            found = sorted(d.glob("frozen_test_*.json"))
            if found:
                return found
    return []


MANIFESTS = _default_manifests()
SIDECAR = ".sources.json"
_HASH_CACHE: dict[tuple, str] = {}
_FROZEN_HASHES_CACHE: tuple[tuple[Path, ...], dict[str, str]] | None = None
_FROZEN_BLOCKS_CACHE: tuple[tuple[Path, ...], dict[str, dict[str, str]]] | None = None


class FrozenDataError(RuntimeError):
    """A held-out test file (or something derived from one) was about to be used."""


def sha256_file(path, chunk: int = 8 << 20) -> str:
    p = Path(path)
    rp = p.resolve()
    st = rp.stat()
    key = (str(rp), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    cached = _HASH_CACHE.get(key)
    if cached is not None:
        return cached
    with open(rp, "rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if len(_HASH_CACHE) >= 2048:
        _HASH_CACHE.clear()
    _HASH_CACHE[key] = digest
    return digest


def frozen_hashes(manifests=None) -> dict:
    """sha256 -> "<manifest id>:<file id>" for every file of the given (default: all) frozen manifests."""
    global _FROZEN_HASHES_CACHE
    use_default = manifests is None
    m_list = tuple(MANIFESTS or _default_manifests()) if use_default else tuple(Path(m) for m in manifests)
    if use_default and _FROZEN_HASHES_CACHE is not None and _FROZEN_HASHES_CACHE[0] == m_list:
        return dict(_FROZEN_HASHES_CACHE[1])
    out = {}
    for m in m_list:
        d = json.loads(Path(m).read_text())
        out |= {f["sha256"]: f"{d['id']}:{f['id']}" for f in d["files"]}
    if use_default and not out:
        raise FrozenDataError("no frozen manifest found: refusing to run without the guard")
    if use_default:
        _FROZEN_HASHES_CACHE = (m_list, out)
        return dict(out)
    return out


def check_hashes(shas, *, allow_frozen: bool = False, frozen=None) -> None:
    """Raise if any of ``shas`` is frozen (unless ``allow_frozen``)."""
    if allow_frozen:
        if frozen is None and not (MANIFESTS or _default_manifests()):
            raise FrozenDataError("no frozen manifest found: refusing to run without the guard")
        return
    if frozen is None:
        frozen = _cached_default_hashes()
    hit = sorted({frozen[s] for s in shas if s in frozen})
    if hit:
        raise FrozenDataError(f"held-out test data: {', '.join(hit[:5])}{' ...' if len(hit) > 5 else ''}")


def _cached_default_hashes() -> dict[str, str]:
    global _FROZEN_HASHES_CACHE
    m_list = tuple(MANIFESTS or _default_manifests())
    if _FROZEN_HASHES_CACHE is not None and _FROZEN_HASHES_CACHE[0] == m_list:
        return _FROZEN_HASHES_CACHE[1]
    frozen_hashes()
    return _FROZEN_HASHES_CACHE[1]


def frozen_blocks(manifests=None) -> dict:
    """Name-level guard for frozen data that may not be on disk yet (no local hash): ``{"names": {stem: label},
    "rolls": {roll: label}}`` from the optional ``blocked`` field of the manifests (``names``: dataset file stems,
    e.g. BlueNeg ``19960815F-23-san-francisco``; ``roll_ids``: every spelling of a sealed roll)."""
    global _FROZEN_BLOCKS_CACHE
    use_default = manifests is None
    m_list = tuple(MANIFESTS or _default_manifests()) if use_default else tuple(Path(m) for m in manifests)
    if use_default and _FROZEN_BLOCKS_CACHE is not None and _FROZEN_BLOCKS_CACHE[0] == m_list:
        b0 = _FROZEN_BLOCKS_CACHE[1]
        return dict(names=dict(b0["names"]), rolls=dict(b0["rolls"]))
    names, rolls = {}, {}
    for m in m_list:
        d = json.loads(Path(m).read_text())
        b = d.get("blocked") or {}
        names |= {n: f"{d['id']}:{n}" for n in b.get("names", [])}
        rolls |= {r: f"{d['id']}:{r}" for r in b.get("roll_ids", [])}
    res = dict(names=names, rolls=rolls)
    if use_default:
        _FROZEN_BLOCKS_CACHE = (m_list, res)
        return dict(names=dict(names), rolls=dict(rolls))
    return res


def _cached_default_blocks() -> dict[str, dict[str, str]]:
    global _FROZEN_BLOCKS_CACHE
    m_list = tuple(MANIFESTS or _default_manifests())
    if _FROZEN_BLOCKS_CACHE is not None and _FROZEN_BLOCKS_CACHE[0] == m_list:
        return _FROZEN_BLOCKS_CACHE[1]
    frozen_blocks()
    return _FROZEN_BLOCKS_CACHE[1]


def stem_of(path) -> str:
    """Dataset stem of a file name: everything before the first dot (``x.preview.png`` -> ``x``)."""
    return Path(path).name.split(".", 1)[0]


def check_name(path, *, allow_frozen: bool = False, blocks=None) -> None:
    """Raise if the file name (or a URL / dataset path) belongs to a frozen dataset frame."""
    if allow_frozen:
        return
    blocks = _cached_default_blocks() if blocks is None else blocks
    hit = blocks["names"].get(stem_of(path))
    if hit:
        raise FrozenDataError(f"held-out test data (by name): {hit}")


def check_roll(roll, *, allow_frozen: bool = False, blocks=None) -> None:
    """Raise if ``roll`` is a sealed roll (any registered spelling)."""
    if allow_frozen:
        return
    blocks = _cached_default_blocks() if blocks is None else blocks
    hit = blocks["rolls"].get(str(roll))
    if hit:
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
