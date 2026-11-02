"""Extract committed T4r source and guard manifests into a new empty directory.
Usage: python -I scripts/t4s_prepare_baseline.py NEW_DIR
Does not read or copy original image material.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', 'fbd48a0', 'src', 'data/manifests'],
                                    cwd=ROOT, text=True).splitlines()
    for name in names:
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(subprocess.check_output(['git', 'show', f'fbd48a0:{name}'], cwd=ROOT))
    print(out / 'src')


if __name__ == '__main__':
    main()
