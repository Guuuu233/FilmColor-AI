"""Base-inversion diagnosis, step 1: roll film base for development rolls (experiments/s3_diag).

For every development frame of the given rolls: the per-frame automatic film base of the S3 cache (render path,
the base the cached positive was inverted with) and the EXIF exposure time of the scan; then
filmcolor.base.estimate_base_roll per roll (defaults: level window 0.1 D, chroma tolerance 0.05, 3 frames used).
Reads only the S3 cache records and the RAW EXIF of development frames; frozen frames are not in the frame list,
and every RAW is hashed through filmcolor.frozen before exiftool reads it.

  diag_roll_base.py FRAMES_JSON OUT_JSON ROLL [ROLL ...]
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from filmcolor.base import estimate_base_roll
from filmcolor.frozen import check_source
from filmcolor.paths import data_dir

EXIF_TAGS = ("ExposureTime", "FNumber", "ISO", "LensID", "DateTimeOriginal")


def exif(path):
    out = subprocess.run(["exiftool", "-j", "-n", *(f"-{t}" for t in EXIF_TAGS), str(path)],
                         capture_output=True, text=True, check=True).stdout
    return {k: v for k, v in json.loads(out)[0].items() if k in EXIF_TAGS}


def main(frames_json, out_json, *rolls):
    frames = [f for f in json.load(open(frames_json))["frames"] if f["roll"] in rolls]
    out = dict(method="estimate_base_roll on the per-frame automatic bases of the S3 cache (render path), "
                      "exposure normalised by EXIF ExposureTime", rolls={})
    for roll in rolls:
        cand, t, info = {}, {}, {}
        for f in (f for f in frames if f["roll"] == roll):
            rec_path = data_dir("s3_ml", "cache", f["domain"]) / f"{f['stem']}.json"
            if not rec_path.exists():  # base failure (.failed.json) or not built
                info[f["stem"]] = dict(skipped="no S3 cache record")
                continue
            if check_source(f["raw"]) != f["raw_sha256"]:
                raise SystemExit(f"{f['id']}: RAW differs from the frame list")
            rec = json.loads(rec_path.read_text())
            e = exif(f["raw"])
            cand[f["stem"]] = rec["positive"]["inversion"]["base"]
            t[f["stem"]] = float(e["ExposureTime"])
            info[f["stem"]] = dict(exif=e, base_own=cand[f["stem"]])
        same = {k: sorted({str(v["exif"][k]) for v in info.values() if "exif" in v}) for k in ("FNumber", "ISO", "LensID")}
        rb = estimate_base_roll(cand, t)
        for n in cand:
            b = np.array(rb.for_exposure(t[n]))
            d = np.log10(b / np.array(cand[n]))  # density of the frame's own base above the roll base (> 0: denser)
            info[n].update(base_roll=b.tolist(), own_minus_roll_density=d.tolist(),
                           own_colour_error_RG_BG=[float(d[0] - d[1]), float(d[2] - d[1])])
        out["rolls"][roll] = dict(scan_settings=same, roll=rb.to_dict(), frames=info)
        print(roll, "members", len(rb.members), "used", rb.used, "rgb_norm", [round(x, 4) for x in rb.rgb_norm],
              "t_ref", rb.t_ref, "warnings", rb.warnings, "settings", same, flush=True)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
