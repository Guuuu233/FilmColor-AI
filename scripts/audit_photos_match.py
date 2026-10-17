"""Which frames of the material folder appear in another photo folder (e.g. the user's finished photos), by
content: the same luminance-thumbnail correlation as audit_ext_pairing.py, every photo against every negative
(inverted) and every print of the audit. Read-only on PHOTO_ROOT. Byte hashes are not enough: the same frame
exported or saved again has different bytes.

  audit_photos_match.py PHOTO_ROOT AUDIT_PAIRING_JSON AUDIT_THUMB_DIR FROZEN_JSON OUT_JSON [THUMB_DIR]
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from audit_ext_pairing import load_small, lum, similarity  # noqa: E402

SAME_FRAME_R = 0.75  # as the freeze rule (freeze_ext_rolls.py PRINT_R)
EXT = {".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def main(photo_root, pairing_json, audit_thumbs, frozen_json, out_json, thumb_dir=None):
    root = Path(photo_root)
    pairing = json.load(open(pairing_json))
    frozen = {f["id"]: f["roll"] for f in json.load(open(frozen_json))["files"]}
    thumbs = Path(audit_thumbs)
    ref = {}
    for i in pairing["negatives"]:
        ref[i] = 1 - lum(Image.open(thumbs / f"{i}.jpg").convert("RGB"))
    for i in pairing["refs"]:
        ref[i] = lum(Image.open(thumbs / f"{i}.jpg").convert("RGB"))
    rid = sorted(ref)
    photos, skipped = {}, {}
    for p in sorted(x for x in root.rglob("*") if x.is_file() and x.suffix.lower() in EXT and not x.name.startswith(".")):
        rel = p.relative_to(root).as_posix()
        try:
            im = load_small(p, p.suffix.upper().lstrip("."))
        except Exception as e:  # noqa: BLE001 - recorded, audit continues
            skipped[rel] = f"{type(e).__name__}: {e}"
            continue
        if thumb_dir:
            t = Path(thumb_dir) / (hashlib.sha1(rel.encode()).hexdigest()[:16] + ".jpg")
            t.parent.mkdir(parents=True, exist_ok=True)
            im.resize((288, 192)).save(t, quality=85)
        photos[rel] = lum(im)
    pid = sorted(photos)
    S = similarity([ref[i] for i in rid], [photos[p] for p in pid])  # (photos, refs)
    out = []
    for j, rel in enumerate(pid):
        o = np.argsort(-S[j])
        best = rid[o[0]]
        out.append(dict(photo=rel, sha256=sha256(root / rel), best=best, r=float(S[j, o[0]]),
                        second=rid[o[1]], r2=float(S[j, o[1]]), margin=float(S[j, o[0]] - S[j, o[1]]),
                        frozen_roll=frozen.get(best) if S[j, o[0]] >= SAME_FRAME_R else None))
    json.dump(dict(method=__doc__.split("  audit_photos_match.py")[0].strip(), same_frame_r=SAME_FRAME_R,
                   photos=out, skipped=skipped), open(out_json, "w"), ensure_ascii=False, indent=1)
    hits = [x for x in out if x["frozen_roll"]]
    print(len(out), "photos,", len(skipped), "skipped,", len(hits), "matched to frozen frames")


if __name__ == "__main__":
    main(*sys.argv[1:7])
