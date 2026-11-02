"""Disambiguate A1296 full-resolution whole-image vs upper-band extreme-pixel shares.
Usage: python -I scripts/t4t_a1296_scope.py
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filmcolor.app import State
from filmcolor.frozen import check_source


def main():
    assert os.environ.get("FILMCOLOR_DATA_ROOT") == "/Volumes/waizhiyingpan/filmcolor_data"
    raw = Path("/Volumes/waizhiyingpan/胶片负片素材/01_彩色负片_未反相/A1296.CR2")
    digest = check_source(raw, allow_frozen=False)
    with tempfile.TemporaryDirectory(prefix="fc_t4t_a1296_scope_") as temp:
        folder = Path(temp)
        (folder / raw.name).symlink_to(raw)
        st = State(folder, folder / "exports", False, max_side=1600)
        st.roll_info()
        st.open(raw.name, "color")
        pos = st.preview.full_positive
        h, w = pos.shape[:2]
        regions = dict(whole_image=pos, center=pos[int(.2*h):int(.8*h), int(.2*w):int(.8*w)],
                       upper_band=pos[int(.1*h):int(.2*h), int(.2*w):int(.8*w)])
        result = dict(raw_sha256=digest, full_shape=list(pos.shape), regions={})
        for name, image in regions.items():
            result["regions"][name] = dict(n_pixels=int(image.shape[0]*image.shape[1]),
                any_channel_gt_10_count=int((image > 10).any(-1).sum()),
                any_channel_gt_10_share=float((image > 10).any(-1).mean()),
                max_rgb=image.max(axis=(0, 1)).tolist())
    source = hashlib.sha256()
    for p in sorted(p for p in (ROOT / "src/filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        source.update(str(p.relative_to(ROOT / "src")).encode() + b"\0" + p.read_bytes() + b"\0")
    result["source_sha256"] = source.hexdigest()
    (ROOT / "experiments/t4_auto_ui/t4t/a1296_scope.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
