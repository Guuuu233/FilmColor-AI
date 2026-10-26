"""T4i diagnostics on development frames, opened as the evaluation opens them (roll folder of symlinks, State.open,
standard preset). Prints and writes JSON: automatic parameters, white-balance diagnostics, encoded quantiles, per
channel quantiles of the positive / our render / the reference (encoded values as stored), and the colours of preview
pixels outside the estimator mask (the light-table border). No original is written; files are checked against the
frozen sets before they are read.

  python -I scripts/t4i_probe.py SRC OUT_JSON ID [ID ...]   (e.g. nlp:_DSF8948)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(SRC))

import numpy as np
import tifffile
from PIL import Image

import filmcolor.app as fc_app
from filmcolor.frozen import check_source

REPO = Path(__file__).resolve().parents[1]
FRAMES = json.loads((REPO / "data/manifests/s3_dev_frames.json").read_text())["frames"]
ROLLS = Path("/tmp/fc_t4i/rolls")
QS = (1, 2, 5, 50, 95, 98, 99)


def roll_state(f):
    d = ROLLS / f["domain"] / f["roll"]
    d.mkdir(parents=True, exist_ok=True)
    for g in FRAMES:
        if (g["domain"], g["roll"]) == (f["domain"], f["roll"]):
            link = d / Path(g["raw"]).name
            if not link.is_symlink():
                link.symlink_to((REPO / g["raw"]).resolve())
    st = fc_app.State(d, Path("/tmp/fc_t4i/out"), allow_frozen=False, max_side=1000)
    st.roll_info()
    return st


def q(x):
    return np.percentile(x.reshape(-1, 3), QS, axis=0).round(4).tolist()


def main():
    out_json, ids = Path(sys.argv[2]), sys.argv[3:]
    out_dir = out_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = dict(git_commit=git, quantiles=list(QS), frames={})
    for fid in ids:
        f = next(g for g in FRAMES if g["id"] == fid)
        raw, ref = (REPO / f["raw"]).resolve(), (REPO / f["ref"]).resolve()
        check_source(raw), check_source(ref)
        st = roll_state(f)
        m = st.open(raw.name, "color")
        pv, auto = st.preview, m["auto"]
        p = auto["params"]
        dg = auto["diagnostics"]
        disp = pv.render(p)
        msk = pv.mask
        outside = ~msk
        # colours of the preview outside the mask (saturated light table, outside the image region)
        e_out = disp[outside]
        magenta = (e_out[:, 0] > 0.7) & (e_out[:, 2] > 0.7) & (e_out[:, 1] < 0.35)
        yellow = (e_out[:, 0] > 0.7) & (e_out[:, 1] > 0.7) & (e_out[:, 2] < 0.35)
        cols = np.where(outside.mean(0) > 0.9)[0]
        stem = raw.stem
        Image.fromarray(np.round(disp * 255).astype(np.uint8)).save(out_dir / f"{stem}_standard.jpg", quality=90)
        refimg = tifffile.imread(ref) if ref.suffix.lower() in (".tif", ".tiff") else np.asarray(Image.open(ref))
        scale = 65535.0 if refimg.dtype == np.uint16 else 255.0
        rec = dict(
            base_method=m["base"]["method"], params=dict(fc_app.RenderParams.from_dict(p).to_dict(full=True), matrix=None, tone_curve=None),
            wb={k: v for k, v in dg["wb"].items() if k != "diagnostics"},
            wb_diagnostics=dg["wb"]["diagnostics"],
            contrast_check=dg.get("contrast_check"), encoded=dict(p2=dg.get("encoded_p2"), p25=dg.get("encoded_p25"),
                                                                  median=dg.get("encoded_median"), p98=dg.get("encoded_p98")),
            image_region=list(pv.image_region), preview_shape=list(pv.positive.shape[:2]), factor=pv.factor,
            mask_share=float(msk.mean()), outside_share=float(outside.mean()),
            outside_magenta_share=float(magenta.mean()) if len(e_out) else 0.0,
            outside_yellow_share=float(yellow.mean()) if len(e_out) else 0.0,
            outside_columns=[int(cols.min()), int(cols.max())] if len(cols) else None,
            outside_render_quantiles=q(e_out) if len(e_out) else None,
            positive_quantiles_mask=q(pv.positive[msk]), render_quantiles_mask=q(disp[msk]),
            reference_quantiles=q(refimg[..., :3].astype(np.float64) / scale), reference_shape=list(refimg.shape),
        )
        res["frames"][fid] = rec
        print(fid, json.dumps({k: rec[k] for k in ("base_method", "params", "wb", "contrast_check", "encoded",
                                                    "outside_share", "outside_magenta_share", "outside_yellow_share",
                                                    "outside_columns")}), flush=True)
        st.preview = None
    out_json.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
