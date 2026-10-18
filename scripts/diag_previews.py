"""Base-inversion diagnosis: side-by-side previews for visual comparison (experiments/s3_diag/previews, not in git).

Per frame, one JPEG, all panels at the same height: the negative as scanned (linear camera RGB, scaled to its
99.9th percentile, sRGB-encoded, no inversion), the fixed look (global_look_v1, no reference) on the current
inversion (B0) and on the experimental inversions (B1, B1w4, MCt_B1), the shop print and the user's print. Panels
from the negative are the full frame on the 1/8 grid in sensor orientation; prints are shown as delivered.
Inversion parameters are read from the diag_variants.py records of the shop task (main and b1 sets).

  diag_previews.py VARIANTS_DIR OUT_DIR ID [ID ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from diag_common import LOOK, positive_T  # noqa: E402

from filmcolor.compare import block_mean  # noqa: E402
from filmcolor.frozen import check_source, load_cache  # noqa: E402
from filmcolor.invert import srgb_oetf  # noqa: E402
from filmcolor.paths import data_dir  # noqa: E402
from filmcolor.render import RenderParams, render  # noqa: E402

H = 520
FONTS = ("/System/Library/Fonts/PingFang.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc",
         "/System/Library/Fonts/STHeiti Medium.ttc")
# (variant, record file suffix, title): B0 from the main set, the B1 family from the follow-up set
PANELS = (("B0", "", "当前反转（逐帧片基）"), ("B1", "_b1", "实验：整卷片基"),
          ("B1w4", "_b1", "实验：整卷片基+块平均白点"), ("MCt_B1", "_b1", "实验：整卷片基+相机矩阵"))


def font(size):
    for p in FONTS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def to_img(e):
    return Image.fromarray((np.clip(e, 0, 1) * 255 + 0.5).astype(np.uint8))


def fit_h(im):
    return im.resize((max(1, round(im.width * H / im.height)), H), Image.LANCZOS)


def main(variants_dir, out_dir, *stems):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = {f["stem"]: f for f in json.load(open("data/manifests/s3_dev_frames.json"))["frames"]}
    fnt = font(22)
    for stem in stems:
        f = frames[stem]
        T8 = block_mean(load_cache(data_dir("s3_diag", "T4") / f"{stem}_T4.npy"), 2).astype(np.float64)
        panels = [("原始负片（未反转）", to_img(srgb_oetf(T8 / np.percentile(T8, 99.9))))]
        for v, suffix, title in PANELS:
            rec = json.loads((Path(variants_dir) / f"{stem}_shop{suffix}.json").read_text())
            iv = {k: np.asarray(x) if isinstance(x, list) else x for k, x in rec["variants"][v]["inversion"].items()
                  if not k.startswith("_")}
            P = positive_T(T8, **iv)
            e, _ = render(P, RenderParams(**LOOK["params"]))
            panels.append((title, to_img(e)))
        refs = [("店家成品", f["ref"])]
        ul = Path(f"experiments/s3_ml/user_labels/{stem}.json")
        if ul.exists():
            refs.append(("你的成品（Lightroom）", json.loads(ul.read_text())["user_ref"]))
        for title, p in refs:
            check_source(p)
            with Image.open(p) as im:
                panels.append((title, im.convert("RGB")))
        tiles = [(t, fit_h(im)) for t, im in panels]
        W = sum(im.width for _, im in tiles) + 10 * (len(tiles) - 1)
        sheet = Image.new("RGB", (W, H + 40), (40, 40, 40))
        d = ImageDraw.Draw(sheet)
        x = 0
        for t, im in tiles:
            sheet.paste(im, (x, 40))
            d.text((x + 6, 8), t, fill=(235, 235, 235), font=fnt)
            x += im.width + 10
        d.text((W - 90, 8), stem, fill=(255, 210, 120), font=fnt)
        sheet.save(out / f"{stem}.jpg", quality=90)
        print("wrote", out / f"{stem}.jpg", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
