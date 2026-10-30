"""Create isolated source snapshots used by T4m evaluations, retaining frozen manifests.
Usage: python -I scripts/t4m_prepare_sources.py baseline|holder_off OUT_DIR
OUT_DIR must not exist. Prints the src path; images are never read or copied.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = "e8a8c3da88bdd896c6eb5f4a57d3bf38b7d5f687"


def main():
    mode, out = sys.argv[1], Path(sys.argv[2]).resolve()
    if mode not in ("baseline", "holder_off"):
        raise SystemExit(__doc__)
    out.mkdir(parents=True, exist_ok=False)
    tracked = subprocess.run(["git", "ls-tree", "-r", "--name-only", BASELINE, "src", "data/manifests"],
                             cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()
    for rel in tracked:
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if mode == "holder_off" and rel.startswith("src/") and rel not in ("src/filmcolor/holder.py", "src/filmcolor/preview.py"):
            p.write_bytes((REPO / rel).read_bytes())
        else:
            p.write_bytes(subprocess.run(["git", "show", f"{BASELINE}:{rel}"], cwd=REPO, capture_output=True,
                                         check=True).stdout)
    print(out / "src")


if __name__ == "__main__":
    main()
