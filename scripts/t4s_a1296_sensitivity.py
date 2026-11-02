"""Recompute A1296 quantiles at an explicit preview size, retaining input/source provenance.
Usage: python -I scripts/t4s_a1296_sensitivity.py [MAX_SIDE=1000 [SRC_DIR LABEL]]
SRC_DIR (default: this checkout's src) selects another extracted source tree, e.g. the T4r baseline; its output
file gets the LABEL suffix.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else ROOT / "src"
LABEL = f"_{sys.argv[3]}" if len(sys.argv) > 3 else ""
sys.path.insert(0, str(SRC))
from filmcolor.app import State
from filmcolor.frozen import check_source


def main():
    assert os.environ.get("FILMCOLOR_DATA_ROOT") == "/Volumes/waizhiyingpan/filmcolor_data"
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    p = Path("/Volumes/waizhiyingpan/胶片负片素材/01_彩色负片_未反相/A1296.CR2")
    sha = check_source(p, allow_frozen=False)
    folder = Path("/tmp/fc_t4s_devices/sensitivity")
    folder.mkdir(parents=True, exist_ok=True)
    link = folder / p.name
    if not link.exists():
        link.symlink_to(p)
    assert link.is_symlink() and link.resolve() == p.resolve()
    st = State(folder, folder / "exports", False, max_side=size)
    st.roll_info()
    meta = st.open(p.name, "color")
    pos = st.preview.positive
    h, w = pos.shape[:2]
    source = hashlib.sha256()
    for f in sorted(f for f in (SRC / "filmcolor").rglob("*") if f.is_file() and f.suffix in (".py", ".json", ".html")):
        source.update(str(f.relative_to(SRC)).encode() + b"\0" + f.read_bytes() + b"\0")
    out = dict(max_side=size, shape=list(pos.shape), raw_sha256=sha, src=str(SRC), src_content_sha256=source.hexdigest(),
               center_p99=np.percentile(pos[int(.2*h):int(.8*h), int(.2*w):int(.8*w)], 99, axis=(0, 1)).tolist(),
               upper_band_p99=np.percentile(pos[int(.1*h):int(.2*h), int(.2*w):int(.8*w)], 99, axis=(0, 1)).tolist(),
               upper_band_max=pos[int(.1*h):int(.2*h), int(.2*w):int(.8*w)].max(axis=(0, 1)).tolist(),
               upper_band_share_any_channel_gt_10=float((pos[int(.1*h):int(.2*h), int(.2*w):int(.8*w)] > 10).any(-1).mean()),
               status=meta["auto"]["status"])
    (ROOT / f"experiments/t4_auto_ui/t4s/a1296_{size}px_sensitivity{LABEL}.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
