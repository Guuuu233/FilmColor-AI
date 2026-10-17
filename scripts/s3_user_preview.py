"""Side by side for the two-grader comparison: the shop's print, the user's own print, our default conversion,
and the renders of the P4 parameters fitted to each print (1/8 grid; prints converted from their Lab).

  s3_user_preview.py SHOP_LABEL_DIR USER_LABEL_DIR OUT_DIR STEM [STEM ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from filmcolor.colorimetry import lab_to_xyz, xyz_to_srgb_preview
from filmcolor.frozen import load_cache
from filmcolor.paths import data_dir
from filmcolor.render import RenderParams, render

sys.path.insert(0, str(Path(__file__).parent))
from s3_preview import W, tile  # noqa: E402


def main(shop_dir, user_dir, out_dir, *stems):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for stem in stems:
        shop = json.loads((Path(shop_dir) / "ext_own" / f"{stem}.json").read_text())
        user = json.loads((Path(user_dir) / f"{stem}.json").read_text())
        sc, uc = data_dir("s3_ml", "cache", "ext_own"), data_dir("s3_ml", "cache_user", "ext_own")
        Rs, Ru = load_cache(sc / f"{stem}_R8.npy"), load_cache(uc / f"{stem}_R8.npy")
        Pu = load_cache(uc / f"{stem}_P8.npy")
        prev = lambda R: xyz_to_srgb_preview(lab_to_xyz(R.astype(np.float64)))  # noqa: E731
        cols = [tile(prev(Rs), f"{stem} shop print"), tile(prev(Ru), "user's own print"),
                tile(render(Pu, RenderParams())[0], "our default"),
                tile(render(Pu, RenderParams(**shop["p4"]["params"]))[0],
                     f"shop-fitted P4 (dE to user {user['shop_p4_on_user']['de2000_median']:.1f})"),
                tile(render(Pu, RenderParams(**user["p4"]["params"]))[0], f"user-fitted P4 (dE {user['p4']['de2000_median']:.1f})")]
        row = Image.new("RGB", (W * 5, max(c.height for c in cols)), "white")
        for i, c in enumerate(cols):
            row.paste(c, (i * W, 0))
        row.save(out_dir / f"{stem}.jpg", quality=90)
        rows.append(row)
    sheet = Image.new("RGB", (W * 5, sum(r.height for r in rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(out_dir / "overview.jpg", quality=88)
    print(out_dir / "overview.jpg")


if __name__ == "__main__":
    main(*sys.argv[1:])
