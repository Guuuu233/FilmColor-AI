"""Preserve the T4f showcase, write the T4g showcase, and warm both own-roll thumbnail caches.
Usage: python -I scripts/t4g_finish_assets.py
"""
import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments/t4_auto_ui"
PY = str(REPO / ".venv/bin/python")


def main():
    for suffix in (".json", ".jpg"):
        src, dst = ROOT / ("presets_showcase" + suffix), ROOT / ("presets_showcase_t4f" + suffix)
        if not dst.exists():
            shutil.copy2(src, dst)
    env = dict(os.environ, FILMCOLOR_DATA_ROOT="/Volumes/waizhiyingpan/filmcolor_data")
    cmd = [PY, "-I", str(REPO / "scripts/t4d_presets_showcase.py"), str(ROOT / "presets_showcase.jpg"),
           "--tile-h=300", "ext:A008", "ext:A012", "ext:A013", "nlp:_DSF9237", "nlp:_DSF8948", "ext:A019",
           "ext:A021", "ext:A075", "ext:A091"]
    subprocess.run(["/usr/bin/time", "-l", *cmd], env=env, check=True)
    cmd = [PY, "-I", str(REPO / "scripts/t4g_checks.py"), str(REPO / "src"), str(ROOT / "t4g/after"), "--thumbs"]
    subprocess.run(["/usr/bin/time", "-l", *cmd], env=env, check=True)


if __name__ == "__main__":
    main()
