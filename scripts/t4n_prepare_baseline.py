"""Create T4m source with frozen manifests for T4n fault reproduction.
Usage: python -I scripts/t4n_prepare_baseline.py NEW_DIR
The destination must not exist; no image material is read or copied.
"""
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE="13db565"


def main():
    out=Path(sys.argv[1]).resolve()
    out.mkdir(parents=True,exist_ok=False)
    files=subprocess.check_output(["git","ls-tree","-r","--name-only",BASE,"src","data/manifests"],cwd=ROOT,text=True).splitlines()
    for rel in files:
        dest=out/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(subprocess.check_output(["git","show",f"{BASE}:{rel}"],cwd=ROOT))
    print(out/"src")


if __name__=="__main__":
    main()
