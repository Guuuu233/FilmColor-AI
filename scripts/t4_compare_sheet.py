"""Generate 3-column side-by-side comparison sheet on 8 representative frames of dsc6161 and dsc6233 (T4 v2 deliverable).

  t4_compare_sheet.py OUT_JPG

Columns per frame:
  1. Old Auto (per-frame B0 + NOREF_V2 / global_look_v1, fixed contrast=2.0)
  2. New Auto (T4 v2: Roll Natural base + NOREF_V3 NegPy textured-cell Auto Density / Auto Grade / Highlight & Shadow Hold)
  3. Reference Print (user reference edit when available, else shop reference print)
"""
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

from filmcolor.app import State
from filmcolor.base import natural_roll_base
from filmcolor.noref import NOREF_V2, NOREF_V3, estimate_params_v2, estimate_params_v3, estimation_input
from filmcolor.render import LUMA, RenderParams

REPO = Path(__file__).resolve().parents[1]
MAT_ROOT = Path("/Volumes/waizhiyingpan/胶片负片素材")
FRAMES = [
    ("dsc6161", "A008_dsc_6168.NEF", "A008 (dsc6161: B0 warm cast dC=0.28)",
     "/Volumes/waizhiyingpan/照片/新疆/#012 5207/DSC_6168-编辑.jpg", "User Ref (DSC_6168-编辑)"),
    ("dsc6161", "A019_dsc_6179.NEF", "A019 (dsc6161: dark frame + cast dC=0.22)",
     "/Volumes/waizhiyingpan/照片/新疆/#012 5207/DSC_6179-编辑.jpg", "User Ref (DSC_6179-编辑)"),
    ("dsc6161", "A021_dsc_6181.NEF", "A021 (dsc6161: bright open desert)",
     "/Volumes/waizhiyingpan/照片/新疆/#012 5207/DSC_6181-编辑.jpg", "User Ref (DSC_6181-编辑)"),
    ("dsc6161", "A034_dsc_6194.NEF", "A034 (dsc6161: high-key sky/water dC=0.25)",
     "/Volumes/waizhiyingpan/照片/新疆/#012 5207/DSC_6194-编辑.jpg", "User Ref (DSC_6194-编辑)"),
    ("dsc6233", "A075_dsc_6235.NEF", "A075 (dsc6233: underexposed park)",
     str(MAT_ROOT / "04_已反转成品_对照答案/R005.jpg"), "Shop Ref (R005)"),
    ("dsc6233", "A081_dsc_6241.NEF", "A081 (dsc6233: deep underexposed)",
     str(MAT_ROOT / "04_已反转成品_对照答案/R026.jpg"), "Shop Ref (R026)"),
    ("dsc6233", "A091_dsc_6251.NEF", "A091 (dsc6233: high-contrast branches)",
     str(MAT_ROOT / "04_已反转成品_对照答案/R060.jpg"), "Shop Ref (R060)"),
    ("dsc6233", "A104_dsc_6264.NEF", "A104 (dsc6233: B0 fails completely)",
     str(MAT_ROOT / "04_已反转成品_对照答案/R110.jpg"), "Shop Ref (R110)"),
]
TILE_W = 540


def tile_of(pv, rp: RenderParams, title: str) -> Image.Image:
    disp, stats = pv.render_stats(rp)
    y = disp @ LUMA.astype(np.float32)
    med = float(np.median(y[pv.mask]))
    h, w = disp.shape[:2]
    th = int(h * TILE_W / w)
    u8 = np.round(np.clip(disp, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(u8).resize((TILE_W, th), Image.BILINEAR)
    out = Image.new("RGB", (TILE_W, th + 38), (22, 22, 25))
    out.paste(im, (0, 38))
    d = ImageDraw.Draw(out)
    d.text((8, 4), title, fill=(235, 235, 240))
    d.text((8, 20), f"ev={rp.ev:+.2f} c={rp.contrast:.2f} hl={rp.highlights:+.2f} sw={rp.shadows:+.2f} bk={rp.blacks:+.2f} | med={med:.2f} clip={stats['above_white']:.2%}",
           fill=(170, 195, 220))
    return out


def ref_tile(ref_path: str, target_h: int, title: str) -> Image.Image:
    im = Image.open(ref_path).convert("RGB")
    im = im.resize((TILE_W, target_h), Image.BILINEAR)
    arr = np.asarray(im, np.float32) / 255.0
    med = float(np.median(arr @ LUMA.astype(np.float32)))
    out = Image.new("RGB", (TILE_W, target_h + 38), (22, 22, 25))
    out.paste(im, (0, 38))
    d = ImageDraw.Draw(out)
    d.text((8, 4), title, fill=(235, 235, 240))
    d.text((8, 20), f"Reference Print | sRGB luma median = {med:.2f}", fill=(180, 215, 180))
    return out


def blank_tile(w: int, h: int, title: str, msg: str) -> Image.Image:
    out = Image.new("RGB", (w, h + 38), (42, 22, 25))
    d = ImageDraw.Draw(out)
    d.text((8, 4), title, fill=(255, 160, 140))
    d.text((14, h // 2), msg, fill=(255, 190, 170))
    return out


def main(out_jpg: str):
    states = {
        sub: State(REPO / "data/raw/own" / sub, Path("/tmp/fc_t4_out"), allow_frozen=False, max_side=1000)
        for sub in ("dsc6161", "dsc6233")
    }
    for st in states.values():
        st.roll_info()

    rows = []
    for sub, fname, desc, ref_p, ref_lbl in FRAMES:
        st = states[sub]
        col_imgs = []
        ref_h = 360

        # Column 1: Old B0 + NOREF_V2 (global_look_v1, fixed contrast=2.0)
        try:
            st.open(fname, "color", "auto")
            pv_b0 = st.preview
            est_v2 = estimate_params_v2(*estimation_input(pv_b0.full_positive, pv_b0.image_region), NOREF_V2)
            rp_old = RenderParams.from_dict(est_v2["params"])
            t0 = tile_of(pv_b0, rp_old, f"[1] Old Auto (B0 + NOREF_V2, contrast=2.0)  {fname.split('_')[0]}")
            ref_h = t0.height - 38
            col_imgs.append(t0)
        except Exception as e:  # noqa: BLE001
            col_imgs.append(blank_tile(TILE_W, ref_h, f"[1] Old Auto (B0 + NOREF_V2)  {fname.split('_')[0]}", f"B0 FAILED: {e}"))

        # Column 2: New T4 v2 Auto (Roll Natural + NOREF_V3 NegPy textured-cell auto estimator)
        st.open(fname, "color", "roll_nat", softness=0.5)
        pv_nat = st.preview
        rp_new = RenderParams.from_dict(pv_nat.auto["params"])
        t_new = tile_of(pv_nat, rp_new, f"[2] NEW Auto (Roll Natural + NOREF_V3)  {desc}")
        ref_h = t_new.height - 38
        if col_imgs[0].height != t_new.height:
            col_imgs[0] = blank_tile(TILE_W, ref_h, f"[1] Old Auto (B0 + NOREF_V2)  {fname.split('_')[0]}",
                                     "B0 FAILED: no chromatically consistent base candidates")
        col_imgs.append(t_new)

        # Column 3: Reference Print (User or Shop)
        col_imgs.append(ref_tile(ref_p, ref_h, f"[3] {ref_lbl}"))

        w_tot = sum(im.width for im in col_imgs)
        h_max = max(im.height for im in col_imgs)
        row = Image.new("RGB", (w_tot, h_max), (18, 18, 20))
        x = 0
        for im in col_imgs:
            row.paste(im, (x, 0))
            x += im.width
        rows.append(row)
        print("rendered row:", desc, flush=True)

    W = max(r.width for r in rows)
    H = sum(r.height for r in rows)
    sheet = Image.new("RGB", (W, H), (15, 15, 18))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    out_p = Path(out_jpg)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_p, quality=90)
    print("saved:", out_p, sheet.size)


if __name__ == "__main__":
    main(sys.argv[1])
