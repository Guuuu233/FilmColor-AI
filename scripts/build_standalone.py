#!/usr/bin/env python3
"""Build a self-contained, zero-dependency FilmColor v1.0 release bundle (macOS or Windows).

Embeds a PyInstaller standalone `filmcolor-bridge` binary directly inside `FilmColor.lrplugin/bin/`,
so end users do NOT need Python, uv, or internet access — just like Negative Lab Pro.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
VERSION = "1.0"


def _platform_tag() -> tuple[str, str]:
    if sys.platform == "win32":
        return "win", "Windows-x64"
    arch = platform.machine().lower()
    if arch in ("arm64", "aarch64"):
        return "mac", "macOS-arm64"
    return "mac", f"macOS-{arch}"


def _write_entry_script(path: Path) -> None:
    path.write_text(
        "from filmcolor.plugin_bridge import main\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )


def build_bundle(*, out_dir: Path, make_zip: bool = True) -> Path:
    plat_short, plat_tag = _platform_tag()
    work_dir = REPO / "build" / f"pyinstaller_{plat_short}"
    dist_py = work_dir / "dist"
    spec_dir = work_dir / "spec"
    work_dir.mkdir(parents=True, exist_ok=True)
    entry = work_dir / "_entry_bridge.py"
    _write_entry_script(entry)

    data_sep = ";" if sys.platform == "win32" else ":"
    cfg_src = REPO / "src" / "filmcolor" / "configs"
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--onedir",
        "--name", "filmcolor-bridge",
        "--distpath", str(dist_py),
        "--workpath", str(work_dir / "work"),
        "--specpath", str(spec_dir),
        "--paths", str(REPO / "src"),
        "--add-data", f"{cfg_src}{data_sep}filmcolor/configs",
        "--hidden-import", "rawpy",
        "--hidden-import", "tifffile",
        "--hidden-import", "exifread",
        "--hidden-import", "PIL.ImageFilter",
        "--hidden-import", "scipy.ndimage",
        "--exclude-module", "torch",
        "--exclude-module", "transformers",
        "--exclude-module", "matplotlib",
        "--exclude-module", "tkinter",
        "--exclude-module", "pytest",
        "--exclude-module", "IPython",
    ]
    if sys.platform == "win32":
        cmd.append("--noconsole")
    cmd.append(str(entry))

    print(f"[1/3] Running PyInstaller ({plat_tag})...")
    subprocess.run(cmd, check=True)

    built_dir = dist_py / "filmcolor-bridge"
    if not built_dir.is_dir():
        raise RuntimeError(f"PyInstaller output missing: {built_dir}")

    # Assemble out-of-the-box release folder
    release_name = f"FilmColor-v{VERSION}-{plat_tag}"
    rel_dir = out_dir / release_name
    if rel_dir.exists():
        shutil.rmtree(rel_dir)
    rel_dir.mkdir(parents=True, exist_ok=True)

    print(f"[2/3] Assembling self-contained plugin bundle -> {rel_dir}")
    lr_dst = rel_dir / "FilmColor.lrplugin"
    shutil.copytree(REPO / "plugins" / "lightroom" / "FilmColor.lrplugin", lr_dst)
    bin_dst = lr_dst / "bin" / plat_short / "filmcolor-bridge"
    bin_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(built_dir, bin_dst)

    ps_dst = rel_dir / "FilmColor-Photoshop-UXP"
    shutil.copytree(REPO / "plugins" / "photoshop-uxp", ps_dst)
    shutil.copy2(REPO / "plugins" / "README.md", rel_dir / "README.md")

    if sys.platform == "win32":
        inst_bat = (
            "@echo off\r\nchcp 65001 >nul\r\nsetlocal\r\n"
            'set "SRC=%~dp0FilmColor.lrplugin"\r\n'
            'set "DST=%APPDATA%\\Adobe\\Lightroom\\Modules\\FilmColor.lrplugin"\r\n'
            'if not exist "%APPDATA%\\Adobe\\Lightroom\\Modules" mkdir "%APPDATA%\\Adobe\\Lightroom\\Modules"\r\n'
            'if exist "%DST%" rmdir /s /q "%DST%"\r\n'
            'xcopy /E /I /Y /Q "%SRC%" "%DST%" >nul\r\n'
            "echo.\r\n"
            "echo [OK] FilmColor v1.0 已安装到 Lightroom Classic（内置引擎，免装 Python）！\r\n"
            "echo 重启 Adobe Lightroom Classic 即可在「图库 → 增效工具额外信息 → FilmColor 负片转换…」使用。\r\n"
            "pause\r\n"
        )
        (rel_dir / "Install_FilmColor_Windows.bat").write_text(inst_bat, encoding="utf-8")

        ps_bat = (
            "@echo off\r\nchcp 65001 >nul\r\n"
            'start "" /B "%~dp0FilmColor.lrplugin\\bin\\win\\filmcolor-bridge\\filmcolor-bridge.exe" --port 18765\r\n'
            "echo FilmColor v1.0 引擎已在后台启动（端口 18765），可直接使用 Photoshop UXP 插件。\r\n"
            "timeout /t 3 >nul\r\n"
        )
        (rel_dir / "Start_Bridge_for_Photoshop.bat").write_text(ps_bat, encoding="utf-8")
    else:
        inst_cmd = (
            "#!/bin/bash\nset -e\n"
            'DIR="$(cd "$(dirname "$0")" && pwd)"\n'
            'DST="$HOME/Library/Application Support/Adobe/Lightroom/Modules/FilmColor.lrplugin"\n'
            'mkdir -p "$(dirname "$DST")"\n'
            'rm -rf "$DST"\n'
            'cp -R "$DIR/FilmColor.lrplugin" "$DST"\n'
            'chmod +x "$DST/bin/mac/filmcolor-bridge/filmcolor-bridge"\n'
            'xattr -dr com.apple.quarantine "$DST" 2>/dev/null || true\n'
            'echo ""\n'
            'echo "[OK] FilmColor v1.0 已安装到 Lightroom Classic（内置独立引擎，免装 Python）！"\n'
            'echo "重启 Adobe Lightroom Classic 即可在「图库 → 增效工具额外信息 → FilmColor 负片转换…」使用。"\n'
        )
        p_inst = rel_dir / "Install_FilmColor_macOS.command"
        p_inst.write_text(inst_cmd, encoding="utf-8")
        p_inst.chmod(p_inst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        ps_cmd = (
            "#!/bin/bash\nset -e\n"
            'DIR="$(cd "$(dirname "$0")" && pwd)"\n'
            'xattr -dr com.apple.quarantine "$DIR/FilmColor.lrplugin" 2>/dev/null || true\n'
            'exec "$DIR/FilmColor.lrplugin/bin/mac/filmcolor-bridge/filmcolor-bridge" --port 18765\n'
        )
        p_ps = rel_dir / "Start_Bridge_for_Photoshop.command"
        p_ps.write_text(ps_cmd, encoding="utf-8")
        p_ps.chmod(p_ps.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    total_mb = sum(f.stat().st_size for f in rel_dir.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"[OK] Uncompressed bundle size: {total_mb:.1f} MB")

    if make_zip:
        zip_path = out_dir / f"{release_name}.zip"
        print(f"[3/3] Creating archive -> {zip_path}")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for f in sorted(rel_dir.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(out_dir))
        print(f"[OK] Archive ready: {zip_path} ({zip_path.stat().st_size / (1024*1024):.1f} MB)")
        return zip_path
    return rel_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build standalone zero-dependency FilmColor v1.0 plugin bundle")
    ap.add_argument("--out", type=Path, default=REPO / "dist", help="Output directory (default: dist/)")
    ap.add_argument("--no-zip", action="store_true", help="Skip creating .zip archive")
    a = ap.parse_args(argv)
    build_bundle(out_dir=a.out, make_zip=not a.no_zip)


if __name__ == "__main__":
    main()
