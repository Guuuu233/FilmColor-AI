"""T4l checks beyond the 185-frame run (scripts/t4g_run_eval.py t4l), outputs in experiments/t4_auto_ui/t4l/.
Usage: python -I scripts/t4l_run_checks.py cross

cross: the BlueNeg roll-base frames that are not in the dev manifest (data/manifests/bn_rb_frames.json, s3_5 false:
124 frames of the same 14 rolls, opened as one folder per roll), twice in two processes: the T4l app (cross_t4l.json)
and the T4l app with its two inversion changes off (SPIKE_CLAMP, WHITE_SELF_RATIO None: cross_inv_off.json); the
t4k_est variant of the latter is the T4k estimator on the T4k inversion. (On the dev frames the inversion changes
alone are the t4k_est variant of eval_t4l against T4k: the T4k estimator on the T4l positive.)
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/t4_auto_ui/t4l"
STD = {"preset": "standard"}
T4K_EST = dict(preset="standard", cfg=dict(meter_holder_box=False, meter_trim_strips=False, ev_down_adaptive=False,
                                           min_span_d_g=None))
INV_OFF = {"SPIKE_CLAMP": None, "WHITE_SELF_RATIO": None}


def run(out, ids, variants, extra=()):
    OUT.mkdir(parents=True, exist_ok=True)
    ids_file = OUT / f"{out}_ids.txt"
    ids_file.write_text("\n".join(sorted(ids)) + "\n")
    cmd = ["/usr/bin/time", "-l", str(REPO / ".venv/bin/python"), "-I", str(REPO / "scripts/t4d_source_eval.py"),
           str(REPO / "src"), str(OUT / f"{out}.json"), f"--ids-from={ids_file}", "--variants=" + json.dumps(variants),
           *extra]
    env = dict(os.environ, FILMCOLOR_DATA_ROOT="/Volumes/waizhiyingpan/filmcolor_data")
    with (OUT / f"{out}.log").open("a") as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    what = sys.argv[1]
    if what == "cross":
        frames = json.loads((REPO / "data/manifests/bn_rb_frames.json").read_text())["frames"]
        ids = [f["id"] for f in frames if not f["s3_5"]]
        man = ("--manifest=data/manifests/bn_rb_frames.json",)
        jobs = [("cross_t4l", dict(standard=STD, no_meter=dict(preset="standard", cfg=dict(
                    meter_holder_box=False, meter_trim_strips=False)), no_ev_down=dict(preset="standard", cfg=dict(
                    ev_down_adaptive=False)), t4k_est=T4K_EST), man),
                ("cross_inv_off", dict(standard=STD, t4k_est=T4K_EST), (*man, "--app-config=" + json.dumps(INV_OFF)))]
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda j: run(j[0], ids, j[1], j[2]), jobs))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
