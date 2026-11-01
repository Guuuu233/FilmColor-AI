"""Extract committed T4q source and guard manifests into a new empty directory.
Usage: python -I scripts/t4r_prepare_baseline.py NEW_DIR
Does not read or copy original image material.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', '4d31a38', 'src', 'data/manifests'],
                                    cwd=ROOT, text=True).splitlines()
    for name in names:
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(subprocess.check_output(['git', 'show', f'4d31a38:{name}'], cwd=ROOT))
    print(out / 'src')


if __name__ == '__main__':
    main()
