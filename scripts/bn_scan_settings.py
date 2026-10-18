"""BlueNeg roll-base study, step 1: are SilverFast LinearRaw scans comparable across frames of a roll? (read-only)

For every downloaded BlueNeg negative DNG (development side; sealed frames are refused by name): all SilverFast
XMP key/value pairs and the DNG tags that could rescale the signal (BlackLevel, WhiteLevel, BaselineExposure,
AnalogBalance, AsShotNeutral, LinearResponseLimit). Per real roll (<yyyymm><letter>): which keys vary between
frames, with emphasis on keys whose name suggests exposure, gain, light source or automatic adjustment.

  bn_scan_settings.py META_JSON DATA_ROOT OUT_JSON
"""
import html
import json
import re
import sys
import urllib.parse
from collections import defaultdict
from pathlib import Path

import tifffile

from filmcolor.frozen import check_name

HINT = re.compile(r"expos|gain|analog|time|bright|auto|lamp|light|level|gamma|dens|hdr|multi|white|black|"
                  r"mask|balance|shad|high|contrast|curve|neutral|grey|gray|iso|focus", re.I)
DNG_TAGS = ("BlackLevel", "WhiteLevel", "BaselineExposure", "AnalogBalance", "AsShotNeutral", "LinearResponseLimit",
            "BaselineNoise", "BitsPerSample")


def roll_of(x):
    rid = x["roll_id"]
    return "ds-" + (x["date"][:6] + rid[-1] if rid[-1].isalpha() else rid)


def xmp_pairs(xmp):
    s = xmp.decode("utf8", "replace") if isinstance(xmp, bytes) else str(xmp)
    s = urllib.parse.unquote(html.unescape(s))
    out = {}
    for k, v in re.findall(r"<([A-Za-z0-9_]+)>([^<]*)</\1>", s):
        out.setdefault(k, v.strip())
    for k, v in re.findall(r'\b([A-Za-z0-9_]+:[A-Za-z0-9_]+)="([^"]*)"', s):
        out.setdefault(k, v.strip())
    return out


def tags(path):
    out = {}
    with tifffile.TiffFile(path) as tf:
        pages = list(tf.pages) + [sp for p in tf.pages for sp in (getattr(p, "pages", None) or [])]
        xmp = None
        for p in pages:
            for name in DNG_TAGS:
                t = p.tags.get(name)
                if t is not None and name not in out:
                    v = t.value
                    out[name] = list(v) if isinstance(v, tuple) else v
            t = p.tags.get("XMP")
            if t is not None and xmp is None:
                xmp = t.value
    return out, xmp_pairs(xmp) if xmp is not None else {}


def main(meta_json, root, out_json):
    root = Path(root)
    frames = []
    for x in json.loads(Path(meta_json).read_text()):
        p = root / x["negative_path"]
        if not p.exists():
            continue
        check_name(p)
        t, xmp = tags(p)
        frames.append(dict(name=x["filename"], roll=roll_of(x), partition=x["partition"], film=x["film_type"],
                           dng=t, xmp=xmp))
    by_roll = defaultdict(list)
    for f in frames:
        by_roll[f["roll"]].append(f)
    keys = sorted({k for f in frames for k in f["xmp"]})
    varying_global = {k: sorted({str(f["xmp"].get(k)) for f in frames})[:6] for k in keys
                      if len({str(f["xmp"].get(k)) for f in frames}) > 1}
    rolls = {}
    for r, fs in sorted(by_roll.items()):
        var = {}
        for k in keys:
            vals = [str(f["xmp"].get(k)) for f in fs]
            if len(set(vals)) > 1:
                var[k] = dict(n_distinct=len(set(vals)), values=sorted(set(vals))[:6])
        dvar = {k: sorted({json.dumps(f["dng"].get(k)) for f in fs}) for k in DNG_TAGS
                if len({json.dumps(f["dng"].get(k)) for f in fs}) > 1}
        rolls[r] = dict(frames=len(fs), film=sorted({f["film"] for f in fs}), varying_xmp=var, varying_dng=dvar,
                        varying_hint=sorted(k for k in var if HINT.search(k)))
    out = dict(n_frames=len(frames), n_keys=len(keys), dng_tags_first=frames[0]["dng"] if frames else None,
               constant_hint={k: frames[0]["xmp"].get(k) for k in keys if HINT.search(k) and k not in varying_global},
               varying_global=varying_global, rolls=rolls, frames=frames)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    print("frames", len(frames), "keys", len(keys), "varying across all frames", len(varying_global))
    print("DNG tags (first frame):", frames[0]["dng"])
    for r, v in rolls.items():
        print(r, v["frames"], v["film"], "varying xmp", len(v["varying_xmp"]), "hint", v["varying_hint"][:12],
              "dng", list(v["varying_dng"]))


if __name__ == "__main__":
    main(*sys.argv[1:4])
