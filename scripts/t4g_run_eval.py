"""Run T4g's unchanged 191-frame manifest split in two processes, resuming matching checkpoints.
Usage: python -I scripts/t4g_run_eval.py
"""
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/t4_auto_ui/t4g"
VARIANTS = {n: dict(preset=n) for n in ("standard", "natural", "portrait", "vivid", "print_classic")}
VARIANTS["wb_t4f"] = dict(preset="standard", cfg=dict(wb_v3_adaptive_share=0.0, wb_v3_gw_weight=0.0, wb_max_shift=0.35))


def main():
    frames = json.loads((REPO / "data/manifests/s3_dev_frames.json").read_text())["frames"]
    halves = {"a": [], "b": []}
    # Same partition as the T4d/T4e/T4f outputs (membership checked from their records).
    old_a = json.loads((REPO / "experiments/t4_auto_ui/t4f/eval_t4f_a.json").read_text())["frames"]
    for f in frames:
        halves["a" if f["id"] in old_a else "b"].append(f["id"])
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, FILMCOLOR_DATA_ROOT="/Volumes/waizhiyingpan/filmcolor_data")
    def run(item):
        part, ids = item
        cmd = ["/usr/bin/time", "-l", str(REPO / ".venv/bin/python"), "-I", str(REPO / "scripts/t4d_source_eval.py"),
               str(REPO / "src"), str(OUT / f"eval_t4g_{part}.json"), "--ids=" + ",".join(ids),
               "--variants=" + json.dumps(VARIANTS)]
        print(part, len(ids), flush=True)
        with (OUT / f"eval_{part}.log").open("a") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, halves.items()))


if __name__ == "__main__":
    main()
