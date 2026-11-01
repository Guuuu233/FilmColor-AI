"""Extract T4o 4cd76b7 source and frozen manifests for device reproduction.
Usage: python -I scripts/t4p_prepare_baseline.py NEW_DIR
No original image material is read or copied.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "4cd76b7", "src", "data/manifests"],
                                    cwd=ROOT, text=True).splitlines()
    for name in files:
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(subprocess.check_output(["git", "show", f"4cd76b7:{name}"], cwd=ROOT))
    print(out / "src")


if __name__ == "__main__":
    main()
