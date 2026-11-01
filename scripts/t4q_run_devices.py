"""T4q 18-device color probes and three black/white probes; originals stay read-only.
Usage: python -I scripts/t4q_run_devices.py SRC baseline|candidate
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4q"
MATERIAL = Path("/Volumes/waizhiyingpan/胶片负片素材")
COLOR = ("A1195.CR2", "A240.CR3", "A328.CR3", "A329.ARW", "A099.NEF", "A1094.NEF", "A1170.NEF",
         "A1095.RAF", "A1042.RAF", "A277.dng", "A281.RAF", "A1005.CR2", "A1089.DNG", "A322.cr2", "A1036.RW2")
BW = ("B086.dng", "B087.dng", "B091.RW2")


def main():
    src, label = Path(sys.argv[1]).resolve(), sys.argv[2]
    if label not in ("baseline", "candidate"):
        raise SystemExit(__doc__)
    OUT.mkdir(parents=True, exist_ok=True)
    color_paths = [MATERIAL / "01_彩色负片_未反相" / n for n in COLOR]
    bw_paths = [MATERIAL / "02_黑白负片_未反相" / n for n in BW]
    for film, paths in (("color", color_paths + bw_paths), ("bw", bw_paths)):
        name = f"devices_{label}_{film}"
        with (OUT / f"{name}.log").open("a") as log:
            subprocess.run(["/usr/bin/time", "-l", str(ROOT / ".venv/bin/python"), "-I", str(ROOT / "scripts/t4q_probe_devices.py"),
                            str(src), str(OUT / f"{name}.json"), *map(str, paths)], cwd=ROOT,
                           env=dict(os.environ, FILMCOLOR_DATA_ROOT="/Volumes/waizhiyingpan/filmcolor_data", FILMCOLOR_PROBE_FILM=film),
                           stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
