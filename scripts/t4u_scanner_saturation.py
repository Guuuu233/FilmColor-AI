"""Recompute TIFF saturation-mask evidence without storing raw metadata or modifying originals."""
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filmcolor.frozen import check_source
from filmcolor.scanner import load_scanner


def main():
    assert os.environ.get("FILMCOLOR_DATA_ROOT") == "/Volumes/waizhiyingpan/filmcolor_data"
    result = dict(frames={})
    for name in ("A1040.TIF", "A1041.TIF"):
        p = Path("/Volumes/waizhiyingpan/胶片负片素材/01_彩色负片_未反相") / name
        digest = check_source(p, allow_frozen=False)
        frame = load_scanner(p, allow_frozen=False)
        saturated = frame.rgb >= 1.0
        result["frames"][name] = dict(raw_sha256=digest,
            saturation_share_rgb=saturated.mean(axis=(0, 1)).tolist(),
            saturation_mask_share=float(frame.sat.mean()),
            single_channel_prestretched=frame.meta.get("single_channel_prestretched"),
            gb_mask_exact=bool(np.array_equal(frame.sat, saturated[:, :, 1] | saturated[:, :, 2])))
        del frame, saturated
        gc.collect()
    h = hashlib.sha256()
    for p in sorted(p for p in (ROOT / "src/filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(ROOT / "src")).encode() + b"\0" + p.read_bytes() + b"\0")
    result["source_sha256"] = h.hexdigest()
    (ROOT / "experiments/t4_auto_ui/t4u/scanner_saturation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
