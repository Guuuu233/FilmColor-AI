"""Extract the authorized c76e588 T4o baseline with frozen guards; no original image reads.
Usage: python -I scripts/t4o_prepare_baseline.py NEW_DIR
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=False)
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "c76e588", "src", "data/manifests"],
                                    cwd=ROOT, text=True).splitlines()
    for name in files:
        p = out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(subprocess.check_output(["git", "show", f"c76e588:{name}"], cwd=ROOT))
    print(out / "src")


if __name__ == "__main__":
    main()
