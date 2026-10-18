"""Roll-base validation: side-by-side previews from diag_rollcheck.py records (not in git).

Per frame one JPEG: the negative as scanned (linear, scaled to its 99.9th percentile, sRGB-encoded), the fixed look
(global_look_v1, no reference) on B0 (current engine), on R1 (the pre-registered rule: B1 when the base chroma
disagreement >= 0.1 D, else B0) and on B1 (roll base for every frame), then the reference print. Panels from the
negative are the full frame on the 1/8 grid in sensor orientation; prints are shown as delivered.

  rb_previews.py FRAMES_JSON ROLLCHECK_DIR OUT_DIR ID [ID ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from diag_common import LOOK, positive_T  # noqa: E402
from diag_previews import fit_h, font, to_img, H  # noqa: E402

from filmcolor.frozen import check_source, load_cache  # noqa: E402
from filmcolor.invert import srgb_oetf  # noqa: E402
from filmcolor.paths import data_dir  # noqa: E402
from filmcolor.render import RenderParams, render  # noqa: E402

SPLIT = 0.1


def print_img(path):
    check_source(path)
    if Path(path).suffix.lower() in (".tif", ".tiff"):
        a = tifffile.imread(path)[..., :3].astype(np.float64)
        a /= 65535.0 if a.max() > 255 else 255.0
        return to_img(a)
    with Image.open(path) as im:
        return im.convert("RGB")


def main(frames_json, rc_dir, out_dir, *ids):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = {f["id"]: f for f in json.loads(Path(frames_json).read_text())["frames"]}
    fnt = font(22)
    for fid in ids:
        f = frames[fid]
        rec = json.loads((Path(rc_dir) / f"{f['stem']}_shop.json").read_text())
        o = rec["own_minus_roll_density"]
        c = max(abs(o[0] - o[1]), abs(o[2] - o[1]))
        T8 = load_cache(data_dir("s3_diag", "T8") / f"{f['stem']}_T8.npy").astype(np.float64)
        panels = [("原始负片（未反转）", to_img(srgb_oetf(T8 / np.percentile(T8, 99.9))))]
        for name, title in (("B0", "当前反转"), ("R1", f"规则 R1（色度差 {c:.2f}{'，触发' if c >= SPLIT else '，不触发'}）"),
                            ("B1", "整卷片基（全换）")):
            v = rec["variants"]["B1" if name == "B1" or (name == "R1" and c >= SPLIT) else "B0"]
            P = positive_T(T8, np.asarray(v["base"]), (0, 0, 0), np.asarray(v["d_hi"]))
            e, _ = render(P, RenderParams(**LOOK["params"]))
            panels.append((title, to_img(e)))
        panels.append(("参考成品", print_img(f["ref"])))
        tiles = [(t, fit_h(im)) for t, im in panels]
        W = sum(im.width for _, im in tiles) + 10 * (len(tiles) - 1)
        sheet = Image.new("RGB", (W, H + 40), (40, 40, 40))
        d = ImageDraw.Draw(sheet)
        x = 0
        for t, im in tiles:
            sheet.paste(im, (x, 40))
            d.text((x + 6, 8), t, fill=(235, 235, 235), font=fnt)
            x += im.width + 10
        d.text((W - 260, 8), f["stem"][:24], fill=(255, 210, 120), font=fnt)
        sheet.save(out / f"{f['stem']}.jpg", quality=90)
        print("wrote", out / f"{f['stem']}.jpg", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
