"""T4r dev/cross regression vs committed T4o and T4q evidence.
Usage: python -I scripts/t4r_run_eval.py
Old WB and old over-lift gate ablations share the candidate positive; base/spike changes use full regression.
"""
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4r"
VARIANTS = {p: dict(preset=p) for p in ("standard", "natural", "portrait", "vivid", "print_classic")}
VARIANTS['old_wb_gate'] = dict(preset='standard', cfg=dict(wb_v3_cast_max_bright_share_no_tb=.45,
                                                         wb_v3_cyan_max_zone_dis=1e9))
for preset in ('standard', 'natural', 'portrait', 'vivid', 'print_classic'):
    VARIANTS['old_overlift_' + preset] = dict(preset=preset, cfg=dict(overlift_max_span=1e9))


def run(name, ids, manifest, variants):
    ids_file = OUT / f"{name}_ids.txt"
    ids_file.write_text("\n".join(sorted(ids)) + "\n")
    cmd = ["/usr/bin/time", "-l", str(ROOT / ".venv/bin/python"), "-I", str(ROOT / "scripts/t4d_source_eval.py"),
           str(ROOT / "src"), str(OUT / f"{name}.json"), f"--ids-from={ids_file}", f"--manifest={manifest}",
           "--variants=" + json.dumps(variants)]
    with (OUT / f"{name}.log").open("a") as log:
        subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, FILMCOLOR_DATA_ROOT="/Volumes/waizhiyingpan/filmcolor_data"),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print(name, "complete", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for part in ("a", "b"):
        ids = json.loads((ROOT / f"experiments/t4_auto_ui/t4o/candidate_{part}.json").read_text())["frames"]
        jobs.append((f"candidate_{part}", ids, "data/manifests/s3_dev_frames.json", VARIANTS))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda args: run(*args), jobs))
    frames = json.loads((ROOT / "data/manifests/bn_rb_frames.json").read_text())["frames"]
    variants = {k: v for k, v in VARIANTS.items() if k in ('standard', 'old_wb_gate', 'old_overlift_standard')}
    run("candidate_cross", [f["id"] for f in frames if not f["s3_5"]], "data/manifests/bn_rb_frames.json", variants)


if __name__ == "__main__":
    main()
