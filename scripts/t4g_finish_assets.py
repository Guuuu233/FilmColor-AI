"""Preserve the previous showcase, write the new one, and warm both own-roll thumbnail caches.
Usage: python -I scripts/t4g_finish_assets.py [TAG [PREVIOUS]]   (default t4g t4f; T4h: t4h t4g)
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "experiments/t4_auto_ui"
PY = str(REPO / ".venv/bin/python")
TAG = sys.argv[1] if len(sys.argv) > 1 else "t4g"
PREVIOUS = sys.argv[2] if len(sys.argv) > 2 else "t4f"


def main():
    for suffix in (".json", ".jpg"):
        src, dst = ROOT / ("presets_showcase" + suffix), ROOT / (f"presets_showcase_{PREVIOUS}" + suffix)
        if not dst.exists():
            shutil.copy2(src, dst)
    env = dict(os.environ, FILMCOLOR_DATA_ROOT="/Volumes/waizhiyingpan/filmcolor_data")
    cmd = [PY, "-I", str(REPO / "scripts/t4d_presets_showcase.py"), str(ROOT / "presets_showcase.jpg"),
           "--tile-h=300", "ext:A008", "ext:A012", "ext:A013", "nlp:_DSF9237", "nlp:_DSF8948", "ext:A019",
           "ext:A021", "ext:A075", "ext:A091"]
    subprocess.run(["/usr/bin/time", "-l", *cmd], env=env, check=True)
    cmd = [PY, "-I", str(REPO / "scripts/t4g_checks.py"), str(REPO / "src"), str(ROOT / TAG / "after"), "--thumbs"]
    subprocess.run(["/usr/bin/time", "-l", *cmd], env=env, check=True)


if __name__ == "__main__":
    main()
