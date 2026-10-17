"""Independent check of the negative -> finished-print pairing in an external material folder.

Read-only on ROOT. For every negative (camera preview embedded in the raw file, or the JPEG itself)
and every finished print R*, build a small luminance thumbnail, invert the negative, and correlate
every negative with every print over 2 rotations x mirror (central 80 % of the frame, so the small
crop between scan and print does not matter). Uses no pairing information from the index; the index
pairing is only compared afterwards. Also lists near-duplicates among the negatives and among the
prints (re-scans / other versions of the same frame), with the same similarity.

Luminance only: a match says the picture is the same frame, nothing about its colour.

  audit_ext_pairing.py ROOT INVENTORY_JSON OUT_JSON [THUMB_DIR]
"""
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image

H, W = 48, 72  # thumbnail in landscape orientation
RAW_EXT = {"NEF", "CR2", "CR3", "ARW", "RAF", "ORF", "RW2", "DNG"}
DUP_R = 0.95  # similarity listed as a possible duplicate (reviewed by eye, not decided by this number)


def load_small(path, ext):
    if ext in RAW_EXT:
        with rawpy.imread(str(path)) as r:
            t = r.extract_thumb()
        if t.format != rawpy.ThumbFormat.JPEG:
            return None
        im = Image.open(io.BytesIO(t.data))
    else:
        im = Image.open(path)
    im.draft("RGB", (im.width // 8, im.height // 8))
    im = im.convert("RGB")
    if im.height > im.width:
        im = im.rotate(90, expand=True)
    return im


def lum(im):
    a = np.asarray(im.resize((W * 2, H * 2), Image.BILINEAR), np.float64) / 255
    y = a @ [0.2126, 0.7152, 0.0722]
    return y.reshape(H, 2, W, 2).mean((1, 3))


def centre(y):
    h, w = y.shape
    c = y[int(h * 0.1): h - int(h * 0.1), int(w * 0.1): w - int(w * 0.1)]
    c = c - c.mean()
    return c / (np.linalg.norm(c) + 1e-12)


def variants(y):
    out = []
    for m in (False, True):
        b = y[:, ::-1] if m else y
        for k in (0, 2):  # same aspect only; portrait inputs were already rotated to landscape
            out.append(centre(np.rot90(b, k)))
    return out


def similarity(A, B):
    """Best correlation of every row image of A with every image of B over the variants of B."""
    M = np.stack([centre(a).ravel() for a in A])
    S = np.full((len(B), len(A)), -1.0)
    for j, b in enumerate(B):
        for v in variants(b):
            S[j] = np.maximum(S[j], M @ v.ravel())
    return S


def near_duplicates(ids, ys):
    S = similarity(ys, ys)
    return [dict(a=ids[i], b=ids[j], r=float(max(S[i, j], S[j, i])))
            for i in range(len(ids)) for j in range(i + 1, len(ids)) if max(S[i, j], S[j, i]) >= DUP_R]


def main(root, inv_path, out_path, thumb_dir=None):
    root = Path(root)
    inv = [r for r in json.load(open(inv_path)) if r.get("group") in ("A", "B", "R")]
    negs, refs, skipped = {}, {}, {}
    for r in inv:
        try:
            im = load_small(root / r["rel"], r["ext"])
        except Exception as e:  # noqa: BLE001 - recorded, audit continues
            skipped[r["id"]] = f"{type(e).__name__}: {e}"
            print("skip", r["id"], skipped[r["id"]], flush=True)
            continue
        if im is None:
            skipped[r["id"]] = "no JPEG preview"
            print("skip", r["id"], "no JPEG preview", flush=True)
            continue
        if thumb_dir:
            Path(thumb_dir).mkdir(parents=True, exist_ok=True)
            im.resize((W * 4, H * 4)).save(Path(thumb_dir) / f"{r['id']}.jpg", quality=85)
        y = lum(im)
        (refs if r["group"] == "R" else negs)[r["id"]] = (1 - y) if r["group"] != "R" else y
        print(r["id"], flush=True)
    nid, rid = sorted(negs), sorted(refs)
    S = similarity([negs[i] for i in nid], [refs[i] for i in rid])
    meta = {r["id"]: r for r in inv}
    res = []
    for j, i in enumerate(rid):
        o = np.argsort(-S[j])
        best, second = nid[o[0]], nid[o[1]]
        col = S[:, o[0]]
        mutual = rid[int(np.argmax(col))] == i
        res.append(dict(ref=i, best=best, r=float(S[j, o[0]]), second=second, r2=float(S[j, o[1]]),
                        margin=float(S[j, o[0]] - S[j, o[1]]), mutual=bool(mutual),
                        index_pair=meta[i].get("index_pair", ""),
                        same_roll=meta[best].get("orig_dir") == meta[i].get("orig_dir")))
    reverse = defaultdict(list)
    for x in res:
        reverse[x["best"]].append(x["ref"])
    json.dump(dict(method=__doc__.split("  audit_ext_pairing.py")[0].strip(), thumb=[H, W], dup_threshold=DUP_R,
                   negatives=nid, refs=rid, skipped=skipped, pairs=res,
                   negatives_claimed_twice={k: v for k, v in reverse.items() if len(v) > 1},
                   negative_near_duplicates=near_duplicates(nid, [negs[i] for i in nid]),
                   ref_near_duplicates=near_duplicates(rid, [refs[i] for i in rid])),
              open(out_path, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:5])
