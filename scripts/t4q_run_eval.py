"""T4q full-resolution dev/cross regression vs T4o, with over-lift damping ablation.
Usage: python -I scripts/t4q_run_eval.py
The no_overlift variant is module A alone; standard is modules A+B.
"""
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4q"
VARIANTS = {p: dict(preset=p) for p in ("standard", "natural", "portrait", "vivid", "print_classic")}
VARIANTS["no_overlift"] = dict(preset="standard", cfg=dict(overlift_damp_check=False))


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
    variants = {k: v for k, v in VARIANTS.items() if k in ("standard", "no_overlift")}
    run("candidate_cross", [f["id"] for f in frames if not f["s3_5"]], "data/manifests/bn_rb_frames.json", variants)


if __name__ == "__main__":
    main()
