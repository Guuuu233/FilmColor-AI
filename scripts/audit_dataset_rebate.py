"""First-pass screen for negatives that show unexposed film (rebate / film edge) next to the picture.

Read-only on ROOT; frozen files are not read. On the small image of audit_dataset_pairing.load (camera
preview, TIFF or JPEG), for each side look at the outer OUTER share of the frame: a rebate is a line
parallel to that side that is about as clear as the clearest picture content (median luminance >=
CLEAR x inner 99th percentile), uniform along its length (MAD / median <= FLAT) and covers the side
(>= COVER of its length above the clear level). The clearest content of a negative is the base, so a
side that holds such a line most likely shows film without an image. This is a screen, not a decision:
its hit rate is checked by eye on a sample (contact sheets written next to the output).

  audit_dataset_rebate.py ROOT CATALOG_JSON OUT_JSON
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from audit_dataset_pairing import load  # noqa: E402

OUTER, CLEAR, FLAT, COVER = 0.15, 0.85, 0.06, 0.6
REBATE_MAX, FREE_MIN = 1.25, 1.4  # line / clearest picture content, linear transmission
TILE = 300


def sides(y):
    """Lines of the outer band of each side, outermost first, as (side, k, line)."""
    h, w = y.shape
    bh, bw = max(2, int(h * OUTER)), max(2, int(w * OUTER))
    for k in range(bh):
        yield "top", k, y[k]
        yield "bottom", k, y[h - 1 - k]
    for k in range(bw):
        yield "left", k, y[:, k]
        yield "right", k, y[:, w - 1 - k]


def screen(im, linear):
    """Per side: 'rebate' = a flat line about as clear as the clearest picture content (film base + fog);
    'film_free' = a flat line much clearer than that (light reaching the sensor without film: holder gap,
    sprocket hole strip, scanner background). Ratios on linear transmission (gamma-encoded inputs undone with 2.2)."""
    a = np.asarray(im.convert("L"), np.float64) / 255
    if not linear:
        a = a ** 2.2
    h, w = a.shape
    inner = a[int(h * OUTER): h - int(h * OUTER), int(w * OUTER): w - int(w * OUTER)]
    level = max(np.percentile(inner, 99), 1e-4)
    found = {}
    for side, k, line in sides(a):
        m = np.median(line)
        if side in found or m < CLEAR * level:
            continue
        mad = np.median(np.abs(line - m))
        if mad / max(m, 1e-6) > FLAT or np.mean(line >= CLEAR * level) < COVER:
            continue
        ratio = m / level
        if ratio <= REBATE_MAX:
            found[side] = ("rebate", k, ratio)
        elif ratio >= FREE_MIN:
            found.setdefault(side + "_free", ("film_free", k, ratio))
    reb = {s: v for s, v in found.items() if v[0] == "rebate"}
    free = {s.removesuffix("_free"): v for s, v in found.items() if v[0] == "film_free"}
    rel = lambda s, k: round(k / (h if s in ("top", "bottom") else w), 3)
    return dict(sides=sorted(reb), film_free=sorted(free),
                depth={s: rel(s, v[1]) for s, v in reb.items()},
                ratio={s: round(float(v[2]), 3) for s, v in {**reb, **{f"{s}_free": v for s, v in free.items()}}.items()},
                inner_p99=round(float(level), 4))


def sheet(root, rels, out):
    cols = 6
    img = Image.new("RGB", (cols * TILE, ((len(rels) + cols - 1) // cols) * (TILE + 16)), "white")
    d = ImageDraw.Draw(img)
    for i, rel in enumerate(rels):
        t = load(root / rel)
        t.thumbnail((TILE, TILE))
        x, y = (i % cols) * TILE, (i // cols) * (TILE + 16)
        img.paste(t.convert("RGB"), (x, y + 16))
        d.text((x + 3, y + 2), rel.rsplit("/", 1)[-1][:40], fill="black")
    img.save(out, quality=85)


def main(root, catalog, out_path):
    root = Path(root)
    c = json.loads(Path(catalog).read_text())
    files = {f["rel"]: f for f in c["files"]}
    res, skipped = {}, {}
    for fr in c["frames"]:
        rel = next((r for r in fr["files"] if not files[r]["frozen"]), None)
        if rel is None:
            continue
        try:
            im = load(root / rel)
        except Exception as e:  # noqa: BLE001 - recorded, screen continues
            skipped[rel] = f"{type(e).__name__}: {e}"[:200]
            continue
        if im is None:
            skipped[rel] = "no JPEG preview"
            continue
        res[fr["frame"]] = dict(rel=rel, kind=fr["kind"], source=fr["source"],
                                **screen(im, linear=files[rel]["fmt"] == "tiff16"))
    rng = random.Random(0)
    hits = sorted(k for k, v in res.items() if v["sides"])
    miss = sorted(k for k, v in res.items() if not v["sides"])
    samples = dict(hit=rng.sample(hits, min(36, len(hits))), miss=rng.sample(miss, min(36, len(miss))))
    out = Path(out_path)
    for name, keys in samples.items():
        sheet(root, [res[k]["rel"] for k in keys], out.with_name(f"rebate_sample_{name}.jpg"))
    json.dump(dict(method=__doc__.split("  audit_dataset_rebate.py")[0].strip(),
                   params=dict(outer=OUTER, clear=CLEAR, flat=FLAT, cover=COVER), frames=res, skipped=skipped,
                   samples=samples), open(out, "w"), ensure_ascii=False, indent=1)
    by = {}
    for v in res.values():
        k = (v["kind"], v["source"])
        by.setdefault(k, [0, 0, 0])
        by[k][0] += 1
        by[k][1] += bool(v["sides"])
        by[k][2] += bool(v["film_free"])
    for k, (n, h, ff) in sorted(by.items()):
        print(k, "frames", n, "rebate", h, "film_free", ff)
    print("screened", len(res), "skipped", len(skipped), "hits", len(hits))


if __name__ == "__main__":
    main(*sys.argv[1:4])
