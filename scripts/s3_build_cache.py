"""S3 caches for one or more development frames (docs/S3_ML_DESIGN.md sections 2.3 and 3).

Per frame, under $FILMCOLOR_DATA_ROOT/s3_ml/cache/<domain>/:
  <stem>_X8.npy  full-frame scene-linear positive P, 8x8 block mean (model-input source; depends on the
                 negative only, never on the reference)
  <stem>_P8.npy / <stem>_R8.npy  P and the reference Lab on the reference-aligned 1/8 grid, 3 % margin cropped
                 (labels and evaluation only)
  <stem>.json    inversion record, base estimate and warnings, alignment, hashes
  <stem>.failed.json  instead, when the automatic film base finds no usable candidates (no P under the design
                 definition; no fallback is substituted)
P is pipeline.positive with the design defaults (colour, automatic per-frame film base, white_pct 99.9,
gamma 0.6, exposure 1.0); the full-resolution P is not stored. Every file is hashed again and checked against
the frame list and the frozen test manifests before it is read.

Alignment: camera frames as in S1-9 / S2 (compare_nlp.py): L* of P rendered with default parameters (in
memory) against the reference L*, both block-averaged by 4, aligned at factor 2. BlueNeg: the S3-3
registration (rotation, mirror, scale, offset; eval_blueneg.py) is reused and its NCC recomputed.

  s3_build_cache.py FRAMES_JSON [ID ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from filmcolor.colorimetry import D50, xyz_to_lab
from filmcolor.compare import align, aligned_pair, block_mean, load_lab
from filmcolor.fit import render_lab
from filmcolor.frozen import check_source, save_cache
from filmcolor.paths import data_dir
from filmcolor.pipeline import positive
from filmcolor.render import RenderParams

sys.path.insert(0, str(Path(__file__).parent))
from eval_blueneg import SRGB_D50, pair as pair_blueneg, print_lab  # noqa: E402
from sensitivity import margin_crop  # noqa: E402

LOAD_FACTOR, ALIGN_FACTOR = 4, 2  # 1/8 grid, as in compare_nlp.py
PRIOR_ALIGNMENT = {"nlp": "experiments/s1_nlp_compare/compare.json", "shop": "experiments/s2_shop/compare/compare.json"}
BLUENEG_EVAL = "experiments/s3_blueneg/eval.json"


def srgb_assumed_lab(path, factor):
    """Lab (D50) of an 8-bit JPEG without ICC, read as sRGB (an assumption, flagged in the record); block
    averaged by ``factor`` in XYZ, as load_lab does."""
    from PIL import Image

    with Image.open(path) as im:
        a8 = np.asarray(im.convert("RGB"))
    h = (a8.shape[0] // factor) * factor
    m = SRGB_D50.T.astype(np.float32)
    chunks = []
    for y in range(0, h, 64 * factor):  # row chunks keep the float copies small
        a = a8[y : min(y + 64 * factor, h)].astype(np.float32) / 255
        lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
        chunks.append(block_mean(lin @ m, factor))
    return xyz_to_lab(np.concatenate(chunks).astype(np.float64), white=D50)


def ncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def build(f, out_dir):
    t0 = time.time()
    raw_sha, ref_sha = check_source(f["raw"]), check_source(f["ref"])
    if (raw_sha, ref_sha) != (f["raw_sha256"], f["ref_sha256"]):
        raise SystemExit(f"{f['id']}: file content differs from the frame list")
    p = positive(f["raw"])
    pos, inv = p["positive"], p["inversion"]
    X8 = block_mean(pos, 8).astype(np.float32)
    P4 = block_mean(pos, LOAD_FACTOR).astype(np.float32)
    del pos, p["positive"]
    ours_lab4, _ = render_lab(P4, RenderParams())
    if f["ref_kind"] == "srgb_no_icc":
        reg = next(r["registration"] for r in json.load(open(BLUENEG_EVAL))["frames"] if r["frame"] == f["stem"])
        ref = print_lab(f["ref"])
        P8, R8 = pair_blueneg(P4, ref, reg)
        L8, _ = pair_blueneg(ours_lab4, ref, reg)
        alignment = dict(source="S3-3 registration reused", **reg, ncc_recomputed=ncc(L8[..., 0], R8[..., 0]))
    else:
        ref4 = load_lab(f["ref"], LOAD_FACTOR) if f["ref_kind"] == "icc" else srgb_assumed_lab(f["ref"], LOAD_FACTOR)
        al = align(ours_lab4[..., 0], ref4[..., 0], ALIGN_FACTOR)
        P8, R8 = aligned_pair(P4, ref4, al)
        alignment = dict(source="aligned now", ref_kind=f["ref_kind"], **al.to_dict())
        prior = json.load(open(PRIOR_ALIGNMENT[f["domain"]])).get(f["stem"]) if f["domain"] in PRIOR_ALIGNMENT else None
        if prior:
            alignment["prior"] = prior["alignment"]
    P8, R8 = margin_crop(P8).astype(np.float32), margin_crop(R8).astype(np.float32)
    stem = f["stem"]
    save_cache(out_dir / f"{stem}_X8.npy", X8, [raw_sha])
    save_cache(out_dir / f"{stem}_P8.npy", P8, [raw_sha, ref_sha])
    save_cache(out_dir / f"{stem}_R8.npy", R8, [raw_sha, ref_sha])
    t, l, h, w = p["image_region"]
    rec = dict(id=f["id"], raw=f["raw"], raw_sha256=raw_sha, ref=f["ref"], ref_sha256=ref_sha,
               positive=dict(definition="pipeline.positive defaults: colour, auto base, white_pct 99.9, gamma 0.6, "
                                        "exposure 1.0", inversion=inv.params.to_dict(), report=inv.report,
                             image_region=list(p["image_region"]), image_region_x8=[t // 8, l // 8, h // 8, w // 8],
                             region_source=p["region_source"], base_inside_image=p["base_inside_image"]),
               base=p["base_info"], alignment=alignment,
               shapes=dict(X8=list(X8.shape), P8=list(P8.shape), R8=list(R8.shape)), seconds=round(time.time() - t0, 1))
    (out_dir / f"{stem}.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def main(frames_json, *ids):
    frames = json.load(open(frames_json))["frames"]
    for f in frames:
        if ids and f["id"] not in ids:
            continue
        out_dir = data_dir("s3_ml", "cache", f["domain"])
        try:
            rec = build(f, out_dir)
        except ValueError as e:  # automatic film base found nothing usable: no P under the design definition
            if "base candidates" not in str(e):
                raise
            (out_dir / f"{f['stem']}.failed.json").write_text(json.dumps(dict(id=f["id"], stage="film base", error=str(e))))
            print(f["id"], "FAILED film base:", e, flush=True)
            continue
        a = rec["alignment"]
        print(f["id"], "ncc %.3f" % a.get("ncc_recomputed", a["ncc"]), "rot", a["rot90"], "scale %.3f" % a["scale"],
              "P8", rec["shapes"]["P8"], "base warnings", rec["base"].get("estimate", {}).get("warnings"),
              f"{rec['seconds']} s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
