#!/usr/bin/env python3
"""Install FilmColor.lrplugin into Lightroom Classic Modules folder (symlink) and verify setup."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
LR_SRC = REPO / "plugins" / "lightroom" / "FilmColor.lrplugin"
LR_MODULES = Path.home() / "Library" / "Application Support" / "Adobe" / "Lightroom" / "Modules"
LR_DST = LR_MODULES / "FilmColor.lrplugin"
PS_SRC = REPO / "plugins" / "photoshop-uxp"


def install_lightroom(*, force: bool = False) -> Path:
    if not LR_SRC.is_dir():
        raise FileNotFoundError(f"missing plugin source: {LR_SRC}")
    LR_MODULES.mkdir(parents=True, exist_ok=True)
    if LR_DST.is_symlink() or LR_DST.exists():
        if LR_DST.is_symlink() and LR_DST.resolve() == LR_SRC.resolve():
            print(f"[OK] Lightroom plugin already linked: {LR_DST} -> {LR_SRC}")
            return LR_DST
        if not force:
            raise FileExistsError(f"{LR_DST} already exists (pass --force to replace symlink)")
        if LR_DST.is_symlink():
            LR_DST.unlink()
        else:
            raise RuntimeError(f"refusing to delete non-symlink directory {LR_DST}; remove manually if intended")
    LR_DST.symlink_to(LR_SRC)
    print(f"[OK] Linked Lightroom plugin: {LR_DST} -> {LR_SRC}")
    return LR_DST


def main(argv=None):
    ap = argparse.ArgumentParser(description="Install FilmColor T24 plugins for Lightroom Classic & Photoshop")
    ap.add_argument("--force", action="store_true", help="replace existing FilmColor.lrplugin symlink")
    a = ap.parse_args(argv)
    install_lightroom(force=a.force)
    venv_py = REPO / ".venv" / "bin" / "python"
    print(f"[INFO] FilmColor Python venv: {venv_py} ({'exists' if venv_py.exists() else 'NOT FOUND'})")
    print(f"[INFO] Photoshop UXP folder : {PS_SRC}")
    print("\nNext steps:")
    print("  1. Restart Adobe Lightroom Classic (or open 文件 → 增效工具管理器 → 添加 FilmColor.lrplugin).")
    print("  2. Select 1 or more negatives -> 图库/文件 → 增效工具额外信息 → FilmColor T24 负片转换…")
    print("  3. Photoshop UXP: Load plugins/photoshop-uxp/manifest.json via Adobe UXP Developer Tool.")


if __name__ == "__main__":
    main()
