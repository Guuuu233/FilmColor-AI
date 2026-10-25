"""T4g base ablations on listed own development frames, separate processes recommended for memory.
Usage: python -I scripts/t4g_base_probe.py OUT MODE [FRAME_PREFIX]
MODE: auto | roll_nat | old_roll | uncapped (new roll, no upward black cap).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from filmcolor.app import State
from filmcolor.base import estimate_base_roll, natural_roll_base
from filmcolor.preview import load_preview
from filmcolor.colorimetry import MatrixTRCProfile, xyz_to_lab
from filmcolor.compare import align, aligned_pair, block_mean, load_lab, metrics
from filmcolor.export import icc_profile
from filmcolor.frozen import check_source
from filmcolor.render import RenderParams

REPO = Path(__file__).resolve().parents[1]


def main():
    out, mode = Path(sys.argv[1]), sys.argv[2]
    out.mkdir(parents=True, exist_ok=True)
    st = State(REPO / "data/raw/own/dsc6161", out / "unused_exports", allow_frozen=False, max_side=1000)
    ri = st.roll_info()
    rel = next(n for n in st.files() if n.startswith(sys.argv[3] if len(sys.argv) > 3 else "A008_"))
    if mode in ("auto", "roll_nat"):
        m = st.open(rel, "color", base_mode=mode)
        pv, auto, base = st.preview, m["auto"], m["base"]
    else:
        fe = ri["frames"][rel]
        roll = fe["roll_rgb"]
        if mode == "old_roll":
            cand = {k:v["own_rgb"] for k,v in ri["frames"].items() if v["own_rgb"]}
            exp = {k:ri["frames"][k]["exposure"] for k in cand}
            roll = estimate_base_roll(cand, exp, orange_mask_prior=False).for_exposure(fe["exposure"])
        base, diag = natural_roll_base(roll, fe["block48_L_p01"], own_rgb=fe["own_rgb"],
                                      max_black_pull_d=10.0 if mode == "uncapped" else 0.04)
        pv = load_preview(st.resolve(rel), film="color", max_side=1000, base_rgb=base, **st._colour_kwargs(st.resolve(rel), "color"),
                          white_block=16, white_pct=99.8)
        auto = pv.auto
        base = dict(rgb=base, diagnostics=diag)
    p = auto["params"]
    img = np.round(np.clip(pv.render(p), 0, 1)*255).astype(np.uint8)
    Image.fromarray(img).save(out / f"{rel.split('_')[0]}_{mode}.jpg", quality=90)
    manifest = json.loads((REPO / "data/manifests/s3_dev_frames.json").read_text())["frames"]
    f = next(f for f in manifest if f["id"] == "ext:" + rel.split("_")[0])
    ref = Path(f["ref"])
    if not ref.is_absolute():
        ref = REPO / ref
    check_source(ref)
    prof = MatrixTRCProfile(icc_profile("srgb"))
    lab = xyz_to_lab(prof.to_xyz(pv.render(p).astype(np.float64)))
    lab0 = xyz_to_lab(prof.to_xyz(pv.render(RenderParams()).astype(np.float64)))
    ref_full = load_lab(ref, factor=2)
    fb = max(1, int(round(max(ref_full.shape[:2]) / max(lab0.shape[:2]))))
    ref_small = block_mean(ref_full, fb)
    al = align(lab0[..., 0], ref_small[..., 0], 1, scales=np.arange(0.80, 1.0301, 0.01))
    a, r = aligned_pair(lab, ref_small, al)
    mt = metrics(a, r)
    (out / f"{rel.split('_')[0]}_{mode}.json").write_text(json.dumps(dict(base=base, auto=auto, alignment=al.to_dict(), metrics=mt), indent=1))
    print("metrics", mt["de2000_median"], mt["mean_da"], mt["mean_db"], flush=True)
    print(mode, p, flush=True)


if __name__ == "__main__":
    main()
