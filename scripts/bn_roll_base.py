"""BlueNeg roll-base study: roll film base with the scanner gates of PREREG_BLUENEG.md section 3.

Same output layout as diag_roll_base.py (read by diag_build_T.py / diag_rollcheck.py via DIAG_ROLL_BASE). Per roll:
  G1 identical scale-relevant SilverFast settings (GATE_KEYS) and Directory over the roll's cached frames
  G2 >= 3 frames with an S3 cache record
  G3 estimate_base_roll succeeds (defaults, every exposure t = 1: fixed exposure within a scan batch)
A roll failing a gate gets no roll base and is marked not applicable (R1 = B0 for its frames). Inputs: the per-frame
automatic bases recorded in the S3 cache and the SilverFast settings of bn_scan_settings.py; no image or print.

  bn_roll_base.py FRAMES_JSON SCAN_SETTINGS_JSON OUT_JSON
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from filmcolor.base import estimate_base_roll
from filmcolor.frozen import check_roll
from filmcolor.paths import data_dir

GATE_KEYS = ("MultiExposure", "General_HdrMode", "IsImageAutomaticsUsed", "General_Bitdepth", "BrightnessRed",
             "BrightnessGreen", "BrightnessBlue", "HighlightRed", "HighlightGreen", "HighlightBlue", "Directory")


def main(frames_json, settings_json, out_json):
    xmp = {f["name"]: f["xmp"] for f in json.loads(Path(settings_json).read_text())["frames"]}
    by = defaultdict(list)
    for f in json.loads(Path(frames_json).read_text())["frames"]:
        by[f["roll"]].append(f)
    out = dict(method="estimate_base_roll on the per-frame automatic scanner bases of the S3 cache, t = 1 "
                      "(PREREG_BLUENEG.md)", gate_keys=list(GATE_KEYS), rolls={})
    for roll, fs in sorted(by.items()):
        check_roll(roll)
        cand, info = {}, {}
        for f in fs:
            rec_path = data_dir("s3_ml", "cache", f["domain"]) / f"{f['stem']}.json"
            if not rec_path.exists():
                info[f["stem"]] = dict(skipped="no S3 cache record")
                continue
            rec = json.loads(rec_path.read_text())
            cand[f["stem"]] = rec["positive"]["inversion"]["base"]
            info[f["stem"]] = dict(base_own=cand[f["stem"]])
        settings = {k: sorted({str(xmp.get(s, {}).get(k)) for s in cand}) for k in GATE_KEYS}
        g1 = all(len(v) == 1 for v in settings.values()) and all(s in xmp for s in cand)
        g2 = len(cand) >= 3
        r = dict(gates=dict(G1=g1, G2=g2, G3=None), scan_settings=settings, frames=info, applicable=False)
        if g1 and g2:
            try:
                rb = estimate_base_roll(cand, {n: 1.0 for n in cand})
                r["gates"]["G3"], r["applicable"], r["roll"] = True, True, rb.to_dict()
                for n in cand:
                    b = np.array(rb.for_exposure(1.0))
                    d = np.log10(b / np.array(cand[n]))
                    info[n].update(base_roll=b.tolist(), own_minus_roll_density=d.tolist(),
                                   own_colour_error_RG_BG=[float(d[0] - d[1]), float(d[2] - d[1])])
            except ValueError as e:
                r["gates"]["G3"], r["error"] = False, str(e)
        out["rolls"][roll] = r
        print(roll, "frames", len(cand), "gates", r["gates"], "applicable", r["applicable"],
              "members", len(r["roll"]["members"]) if "roll" in r else "-",
              "warnings", r["roll"]["warnings"] if "roll" in r else r.get("error", ""), flush=True)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
