#!/usr/bin/env python3
"""Install FilmColor.lrplugin into Lightroom Classic Modules folder (macOS & Windows) and verify setup."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
LR_SRC = REPO / "plugins" / "lightroom" / "FilmColor.lrplugin"
PS_SRC = REPO / "plugins" / "photoshop-uxp"


def lr_modules_dir() -> Path:
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "Adobe" / "Lightroom" / "Modules"
        return Path.home() / "AppData" / "Roaming" / "Adobe" / "Lightroom" / "Modules"
    return Path.home() / "Library" / "Application Support" / "Adobe" / "Lightroom" / "Modules"


def _venv_python() -> Path:
    if sys.platform == "win32":
        pyw = REPO / ".venv" / "Scripts" / "pythonw.exe"
        return pyw if pyw.exists() else (REPO / ".venv" / "Scripts" / "python.exe")
    return REPO / ".venv" / "bin" / "python"


def _write_local_paths_lua(dst_plugin: Path) -> None:
    """Write LocalPaths.lua inside a copied plugin bundle so Lightroom finds the repo .venv automatically."""
    def _lua_esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')

    content = (
        "return {\n"
        f'    repoRoot = "{_lua_esc(str(REPO))}",\n'
        f'    pythonPath = "{_lua_esc(str(_venv_python()))}",\n'
        "}\n"
    )
    (dst_plugin / "LocalPaths.lua").write_text(content, encoding="utf-8")


def install_lightroom(*, force: bool = False) -> Path:
    if not LR_SRC.is_dir():
        raise FileNotFoundError(f"missing plugin source: {LR_SRC}")
    lr_modules = lr_modules_dir()
    lr_modules.mkdir(parents=True, exist_ok=True)
    lr_dst = lr_modules / "FilmColor.lrplugin"

    if sys.platform == "win32":
        if lr_dst.exists() or lr_dst.is_symlink():
            if not force:
                raise FileExistsError(f"{lr_dst} already exists (pass --force to replace)")
            if lr_dst.is_symlink():
                lr_dst.unlink()
            else:
                # Handle both NTFS directory junction (rmdir) and copied folder (rmtree)
                res = subprocess.run(["cmd", "/c", "rmdir", str(lr_dst)], capture_output=True)
                if res.returncode != 0 and lr_dst.exists():
                    shutil.rmtree(lr_dst)
        # Try NTFS directory junction first (no admin rights needed on Windows)
        jres = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(lr_dst), str(LR_SRC)],
            capture_output=True, text=True,
        )
        if jres.returncode == 0 and lr_dst.exists():
            print(f"[OK] Linked Lightroom plugin (junction): {lr_dst} -> {LR_SRC}")
            return lr_dst
        # Fallback: copy plugin directory and embed repo/.venv paths
        shutil.copytree(LR_SRC, lr_dst)
        _write_local_paths_lua(lr_dst)
        print(f"[OK] Installed Lightroom plugin (copy): {lr_dst}")
        return lr_dst

    # macOS / POSIX symlink
    if lr_dst.is_symlink() or lr_dst.exists():
        if lr_dst.is_symlink() and lr_dst.resolve() == LR_SRC.resolve():
            print(f"[OK] Lightroom plugin already linked: {lr_dst} -> {LR_SRC}")
            return lr_dst
        if not force:
            raise FileExistsError(f"{lr_dst} already exists (pass --force to replace symlink)")
        if lr_dst.is_symlink():
            lr_dst.unlink()
        else:
            raise RuntimeError(f"refusing to delete non-symlink directory {lr_dst}; remove manually if intended")
    lr_dst.symlink_to(LR_SRC)
    print(f"[OK] Linked Lightroom plugin: {lr_dst} -> {LR_SRC}")
    return lr_dst


def install_ps_bridge_launcher() -> Path:
    """Install a silent on-demand launcher at a fixed user-data path so Photoshop UXP can auto-start the bridge."""
    venv_py = _venv_python()
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        dst_dir = Path(appdata) / "FilmColor"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "start_bridge.bat"
        bat = (
            "@echo off\r\n"
            f'start "" /B /D "{REPO}" "{venv_py}" -m filmcolor.plugin_bridge --port 18765\r\n'
        )
        dst.write_text(bat, encoding="utf-8")
        print(f"[OK] Installed Photoshop auto-start launcher: {dst}")
        return dst
    dst_dir = Path.home() / "Library" / "Application Support" / "FilmColor"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "start_bridge.command"
    sh = (
        "#!/bin/bash\n"
        f'cd "{REPO}" && nohup "{venv_py}" -m filmcolor.plugin_bridge --port 18765 >/tmp/filmcolor_ps_bridge.log 2>&1 </dev/null &\n'
    )
    dst.write_text(sh, encoding="utf-8")
    dst.chmod(dst.stat().st_mode | 0o111)
    print(f"[OK] Installed Photoshop auto-start launcher: {dst}")
    return dst


def main(argv=None):
    ap = argparse.ArgumentParser(description="Install FilmColor v2.0 plugins for Lightroom Classic & Photoshop")
    ap.add_argument("--force", action="store_true", help="replace existing FilmColor.lrplugin installation")
    a = ap.parse_args(argv)
    install_lightroom(force=a.force)
    install_ps_bridge_launcher()
    venv_py = _venv_python()
    print(f"[INFO] FilmColor Python venv: {venv_py} ({'exists' if venv_py.exists() else 'NOT FOUND'})")
    print(f"[INFO] Photoshop UXP folder : {PS_SRC}")
    print("\nNext steps:")
    print("  1. Restart Adobe Lightroom Classic (or open 文件 → 增效工具管理器 → 添加 FilmColor.lrplugin).")
    print("  2. Select 1 or more negatives -> 图库/文件 → 增效工具额外信息 → FilmColor 负片转换…")
    print("  3. Photoshop UXP: Load plugins/photoshop-uxp/manifest.json via Adobe UXP Developer Tool.")


if __name__ == "__main__":
    main()
