"""Where large, reproducible data lives (caches, intermediate experiment data).

The root is given by the environment variable FILMCOLOR_DATA_ROOT (typically a directory on an external disk);
no disk name is built in. If the variable is unset or the directory does not exist (e.g. the disk is not
mounted) this fails loudly instead of creating anything elsewhere. Code, configs, labels and reports stay in
the repository.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV = "FILMCOLOR_DATA_ROOT"


class DataRootError(RuntimeError):
    pass


def data_root() -> Path:
    v = os.environ.get(ENV)
    if not v:
        raise DataRootError(f"{ENV} is not set")
    root = Path(v)
    if not root.is_dir():
        raise DataRootError(f"{ENV}={v} does not exist (disk not mounted?); not creating it")
    return root


def data_dir(*parts) -> Path:
    """A directory below the data root, created if missing (the root itself is never created)."""
    d = data_root().joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d
