"""Convert RAW negatives to 16-bit TIFF + sidecar JSON.

  convert.py OUT_DIR RAW... [--encoding srgb|linear] [--image-region T,L,H,W] [--base-region T,L,H,W ...]
                            [--render PARAMS.json | --auto | --preset PRESET | --t22 | --t23 | --t24] [--roll-base ROLL_BASE.json]
                            [--film color|bw]
  convert.py OUT_DIR RAW --replay SIDECAR.json

Each frame is converted on its own: a frame that fails is reported and the rest still run (exit status 1 if any
failed). Quality checks that do not pass and film-base warnings are printed, not only written to the sidecar.
--replay refuses an input whose SHA256 differs from the one the sidecar records.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rawpy

from filmcolor.noref import NOREF_T22, NOREF_T23, NOREF_T24, NOREF_V2, STYLE_PRESETS, preset_config
from filmcolor.pipeline import replay_kwargs, sha256, write
from filmcolor.render import camera_color_matrix
from filmcolor.scanner import is_scanner_dng


def region(s):
    return tuple(int(x) for x in s.split(","))


def notes(sc):
    """Non-passing quality checks and film-base warnings of a sidecar, as short strings."""
    out = [f"qc {c['name']} {c['status']} ({c['value']})" for c in sc["qc"]["checks"] if c["status"] != "pass"]
    out += [f"base: {w}" for w in sc.get("base", {}).get("estimate", {}).get("warnings", [])]
    return out


def _with_camera_matrix(path, cfg: dict, film: str, mix: float) -> dict:
    if film != "color" or is_scanner_dng(path):
        return cfg
    try:
        with rawpy.imread(str(path)) as r:
            m = camera_color_matrix(np.asarray(r.rgb_xyz_matrix)[:3], mix)
        return dict(cfg, matrix=m)
    except Exception:  # noqa: BLE001
        return cfg


def _preset_auto_for(path, preset_name: str, film: str) -> dict:
    cfg = preset_config(preset_name)
    return _with_camera_matrix(path, cfg, film, cfg["camera_matrix_mix"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("raw", nargs="+")
    ap.add_argument("--encoding", default="srgb", choices=["srgb", "linear"])
    ap.add_argument("--image-region", type=region)
    ap.add_argument("--base-region", type=region, action="append")
    ap.add_argument("--render", help="JSON file with render parameters (docs/SPRINT2_PARAMS.md)")
    ap.add_argument("--roll-base", help="roll_base.json from scripts/roll_base.py (per-frame base_rgb)")
    ap.add_argument("--auto", action="store_true", help="reference-free render parameters (noref_v2, global_look_v1)")
    ap.add_argument("--preset", choices=list(STYLE_PRESETS),
                    help="reference-free noref_v3 style preset (standard, natural, portrait, vivid, print_classic)")
    ap.add_argument("--t22", action="store_true",
                    help="reference-free T22 physical pipeline (decoupled inversion + linear physics estimator + T19 curve)")
    ap.add_argument("--t23", action="store_true",
                    help="reference-free T23 physical pipeline (T23 inversion + expanded smooth physics estimator + T19 curve)")
    ap.add_argument("--t24", action="store_true",
                    help="reference-free T24 physical+ML residual pipeline (T23 base + pure-NumPy residual trees + T24 render)")
    ap.add_argument("--film", default="color", choices=["color", "bw"],
                    help="film type, given explicitly (no detection); bw = neutral inversion, no colour look")
    ap.add_argument("--replay")
    a = ap.parse_args(argv)
    if a.roll_base and (a.base_region or a.replay):
        ap.error("--roll-base excludes --base-region and --replay")
    if a.auto and (a.render or a.replay or a.preset or a.t22 or a.t23 or a.t24):
        ap.error("--auto excludes --preset, --t22, --t23, --t24, --render and --replay")
    if a.preset and (a.render or a.replay or a.t22 or a.t23 or a.t24):
        ap.error("--preset excludes --t22, --t23, --t24, --render and --replay")
    if (a.t22 or a.t23 or a.t24) and (a.render or a.replay):
        ap.error("--t22/--t23/--t24 excludes --render and --replay")
    if sum(bool(x) for x in (a.t22, a.t23, a.t24)) > 1:
        ap.error("--t22, --t23, and --t24 are mutually exclusive")
    roll = json.load(open(a.roll_base))["frames"] if a.roll_base else None
    if a.replay:
        if len(a.raw) != 1:
            ap.error("--replay takes exactly one RAW")
        side = json.load(open(a.replay))
        if sha256(a.raw[0]) != side["input"]["sha256"]:
            ap.error(f"--replay: {a.raw[0]} is not the input recorded in {a.replay} (SHA256 differs)")
        kw = replay_kwargs(side)
    else:
        kw = dict(encoding=a.encoding, image_region=a.image_region, base_regions=a.base_region,
                  render_params=json.load(open(a.render)) if a.render else None, auto=NOREF_V2 if a.auto else None,
                  film=a.film)
    failed = []
    for p in a.raw:
        try:
            run_kw = dict(kw)
            if a.t24 and not a.replay:
                run_kw["auto"] = _with_camera_matrix(p, NOREF_T24, a.film, 0.35)
            elif a.t23 and not a.replay:
                run_kw["auto"] = _with_camera_matrix(p, NOREF_T23, a.film, 0.35)
            elif a.t22 and not a.replay:
                run_kw["auto"] = _with_camera_matrix(p, NOREF_T22, a.film, 0.35)
            elif a.preset and not a.replay:
                run_kw["auto"] = _preset_auto_for(p, a.preset, a.film)
            if roll is not None:
                if Path(p).name not in roll:
                    raise KeyError(f"{Path(p).name} is not in {a.roll_base}")
                run_kw["base_rgb"] = roll[Path(p).name]["base_rgb"]
            sc = write(p, a.out_dir, **run_kw)
        except Exception as e:  # noqa: BLE001 - one bad frame must not stop the batch
            failed.append(p)
            print(p, "FAILED", f"{type(e).__name__}: {e}", flush=True)
            continue
        r = sc["report"]["range"]
        print(p, "->", sc["output"]["path"], "d_hi", [round(x, 3) for x in sc["inversion"]["params"]["d_hi"]],
              "clip_high%", [round(r[c]["clip_high"] * 100, 2) for c in "RGB"],
              f"auto {sc['auto']['status']} {sc['auto'].get('flags', [])}" if "auto" in sc else "",
              f"qc {sc['qc']['status']}", flush=True)
        for n in notes(sc):
            print("   ", n, flush=True)
    if failed:
        print(f"{len(failed)} of {len(a.raw)} frame(s) failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
