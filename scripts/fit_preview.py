"""Side-by-side preview: ours (defaults) | ours (fitted, given mode) | reference, on the aligned grid.

  fit_preview.py FIT_DIR COMPARE_JSON MODE STEM ...
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

from filmcolor.colorimetry import lab_to_xyz, xyz_to_srgb_preview
from filmcolor.compare import Alignment, aligned_pair, load_lab
from filmcolor.render import RenderParams, render


def main(fit_dir, compare_json, mode, *stems):
    fit_dir = Path(fit_dir)
    comp = json.loads(Path(compare_json).read_text())
    for stem in stems:
        fit = json.loads((fit_dir / f"{stem}.json").read_text())["fits"][mode]
        P8, R8 = aligned_pair(np.load(fit_dir / "cache" / f"{stem}_P4.npy"), load_lab(comp[stem]["reference"], 4),
                              Alignment(**comp[stem]["alignment"]))
        a, _ = render(P8.astype(np.float32), RenderParams())
        b, _ = render(P8.astype(np.float32), RenderParams.from_dict(fit["params"]))
        c = xyz_to_srgb_preview(lab_to_xyz(R8))
        gap = np.ones((a.shape[0], 6, 3))
        sheet = np.concatenate([a, gap, b, gap, c], 1)
        tifffile.imwrite(fit_dir / f"{stem}_{mode}_preview.tif", (np.clip(sheet, 0, 1) * 255 + 0.5).astype(np.uint8))


if __name__ == "__main__":
    main(*sys.argv[1:])
