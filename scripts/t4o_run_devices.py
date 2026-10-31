"""Run the fourteen T4o target/protection samples from one selected source, sequentially.
Usage: python -I scripts/t4o_run_devices.py SRC baseline|candidate
Only symlinks and derived evidence are written; originals are checked by the probe's frozen guard.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4o"
MATERIAL = Path("/Volumes/waizhiyingpan/胶片负片素材/01_彩色负片_未反相")
NAMES = ("A1195.CR2", "A240.CR3", "A328.CR3", "A329.ARW", "A099.NEF", "A1094.NEF", "A1170.NEF",
         "A1095.RAF", "A1042.RAF", "A277.dng", "A281.RAF", "A1005.CR2", "A1089.DNG", "A322.cr2")


def main():
    src, label = Path(sys.argv[1]).resolve(), sys.argv[2]
    if label not in ("baseline", "candidate"):
        raise SystemExit(__doc__)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"devices_{label}.log").open("a") as log:
        subprocess.run(["/usr/bin/time", "-l", str(ROOT / ".venv/bin/python"), "-I", str(ROOT / "scripts/t4o_probe_devices.py"),
                        str(src), str(OUT / f"devices_{label}.json"), *[str(MATERIAL / name) for name in NAMES]],
                       cwd=ROOT, env=dict(os.environ, FILMCOLOR_DATA_ROOT="/Volumes/waizhiyingpan/filmcolor_data"),
                       stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
