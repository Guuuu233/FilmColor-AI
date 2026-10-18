"""Content pairing for the dataset report: every negative without an official pairing against every positive.

Read-only on ROOT. Same luminance-thumbnail correlation as audit_ext_pairing.py (inverted negative vs positive,
2 rotations x mirror, central 80 %), so a match says "same picture", nothing about colour. 16-bit RGB TIFFs,
which PIL cannot open, are read with tifffile. Frozen files are never read (frozen guard by hash / name / roll);
BlueNeg frames are left out because the release metadata pairs them. Dev-verified pairs stay in as a control:
the method must find them again.

Second pass: negatives and positives left unpaired are compared again with each negative cropped to its picture
(crop_picture: drop the black holder, then 6 % per side for the rebate), because scans that include the holder and
the rebate put the picture in the middle only and the central-80 % comparison misses them (A617-A637, 2026-10-18).
A pair is "content" when the positive and the negative are each other's best match and r >= PAIR_R;
CANDIDATE_R <= r < PAIR_R or a non-mutual best match is a "candidate" (to be looked at, not counted).

  audit_dataset_pairing.py ROOT CATALOG_JSON OUT_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from audit_ext_pairing import lum, load_small, near_duplicates, similarity  # noqa: E402

PAIR_R, CANDIDATE_R = 0.75, 0.6  # PAIR_R as freeze_ext_rolls.py PRINT_R


def load(path):
    try:
        return load_small(path, path.suffix.upper().lstrip("."))
    except Exception:  # noqa: BLE001 - fall back to tifffile for 16-bit RGB TIFF, else re-raise
        if path.suffix.lower() not in (".tif", ".tiff"):
            raise
    a = tifffile.imread(path)
    a = a[..., :3] if a.ndim == 3 else np.repeat(a[..., None], 3, -1)
    a = a[::8, ::8].astype(np.float64)
    a = (255 * a / (65535.0 if a.max() > 255 else 255.0)).astype(np.uint8)
    im = Image.fromarray(a)
    return im.rotate(90, expand=True) if im.height > im.width else im


def crop_picture(im, dark=0.15, inset=0.06):
    """Crop a negative scanned with its holder / rebate: bounding box of rows and columns whose mean
    luminance is above `dark` x max (drops the black holder), then `inset` per side (drops the rebate)."""
    a = np.asarray(im.convert("L"), np.float64) / 255
    rows, cols = np.flatnonzero(a.mean(1) > dark * a.max()), np.flatnonzero(a.mean(0) > dark * a.max())
    if len(rows) < 8 or len(cols) < 8:
        return im
    t, b, l, r = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    dy, dx = int((b - t) * inset), int((r - l) * inset)
    return im.crop((l + dx, t + dy, r - dx, b - dy))


def second_pass(root, use, first):
    """Negatives and positives left unpaired by the first pass, negatives cropped to the picture."""
    paired_n = {x["best"] for x in first if x["level"] == "content"}
    paired_r = {x["ref"] for x in first if x["level"] == "content"}
    negs, refs = {}, {}
    for f in use:
        if f["rel"] in paired_n or f["rel"] in paired_r or f["source"] == "auckland_museum":
            continue
        try:
            im = load(root / f["rel"])
        except Exception:  # noqa: BLE001 - already recorded by the first pass
            continue
        if im is None:
            continue
        if f["kind"] == "positive":
            refs[f["rel"]] = lum(im)
        else:
            negs[f["rel"]] = 1 - lum(crop_picture(im))
    nid, rid = sorted(negs), sorted(refs)
    S = similarity([negs[i] for i in nid], [refs[i] for i in rid])
    res = []
    for j, ref in enumerate(rid):
        o = np.argsort(-S[j])
        mutual = rid[int(np.argmax(S[:, o[0]]))] == ref
        r = float(S[j, o[0]])
        res.append(dict(ref=ref, best=nid[o[0]], r=r, second=nid[o[1]], r2=float(S[j, o[1]]), mutual=bool(mutual),
                        level="content" if mutual and r >= PAIR_R else "candidate" if r >= CANDIDATE_R else None,
                        pass_=2))
    return res, len(nid), len(rid)


def main(root, catalog, out_path):
    root = Path(root)
    files = json.loads(Path(catalog).read_text())["files"]
    use = [f for f in files if f["dup_of"] is None and not f["frozen"] and f["source"] != "blueneg"
           and f["kind"] in ("colour_neg", "bw_neg", "positive") and f["fmt"] not in ("none",)]
    negs, refs, skipped = {}, {}, {}
    for f in use:
        key = f["rel"]
        try:
            im = load(root / f["rel"])
        except Exception as e:  # noqa: BLE001 - recorded, audit continues
            skipped[key] = f"{type(e).__name__}: {e}"[:200]
            continue
        if im is None:
            skipped[key] = "no JPEG preview"
            continue
        y = lum(im)
        (refs if f["kind"] == "positive" else negs)[key] = y if f["kind"] == "positive" else 1 - y
        print(key, flush=True)
    nid, rid = sorted(negs), sorted(refs)
    S = similarity([negs[i] for i in nid], [refs[i] for i in rid])  # S[ref, neg]
    res = []
    for j, ref in enumerate(rid):
        o = np.argsort(-S[j])
        best = nid[o[0]]
        mutual = rid[int(np.argmax(S[:, o[0]]))] == ref
        r = float(S[j, o[0]])
        level = "content" if mutual and r >= PAIR_R else "candidate" if r >= CANDIDATE_R else None
        res.append(dict(ref=ref, best=best, r=r, second=nid[o[1]], r2=float(S[j, o[1]]), mutual=bool(mutual),
                        level=level, pass_=1))
    res2, n2, r2 = second_pass(root, use, res)
    better = {x["ref"]: x for x in res2 if x["level"] == "content"}
    res = [better.get(x["ref"], x) for x in res]
    json.dump(dict(method=__doc__.split("  audit_dataset_pairing.py")[0].strip(), pair_r=PAIR_R,
                   candidate_r=CANDIDATE_R, negatives=len(nid), refs=len(rid), skipped=skipped, pairs=res,
                   second_pass=dict(negatives=n2, refs=r2, content=len(better)),
                   ref_near_duplicates=near_duplicates(rid, [refs[i] for i in rid])),
              open(out_path, "w"), ensure_ascii=False, indent=1)
    print("negatives", len(nid), "refs", len(rid), "skipped", len(skipped),
          "content", sum(x["level"] == "content" for x in res), "candidate", sum(x["level"] == "candidate" for x in res))


if __name__ == "__main__":
    main(*sys.argv[1:4])
