"""Run the remaining T4m checks in a bounded sequence (two evaluation children at most).
Usage: python -I scripts/t4m_pipeline.py BASELINE_SRC
Runs fresh baseline and candidate dev/cross evaluations, then eight device probes. Resumes matching checkpoints.
"""
import os
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/t4_auto_ui/t4m"
PY = str(REPO / ".venv/bin/python")
ENV = dict(os.environ, FILMCOLOR_DATA_ROOT="/Volumes/waizhiyingpan/filmcolor_data", FILMCOLOR_EVAL_WORKERS="1")
MATERIAL = Path("/Volumes/waizhiyingpan/胶片负片素材/01_彩色负片_未反相")
SAMPLES = [MATERIAL / n for n in ("A1089.DNG", "A322.cr2", "A1106.CR2", "A1283.DNG", "A1141.NEF", "A223.NEF", "A224.NEF")]
SAMPLES.append(REPO / "data/raw/own/ultramax400_r01/DSC_3827.NEF")


def run(script, *args):
    subprocess.run([PY, "-I", str(REPO / "scripts" / script), *map(str,args)], cwd=REPO, env=ENV, check=True)


def devices(name, src):
    evidence = OUT / f"devices_{name}.json"
    if evidence.exists():
        samples = json.loads(evidence.read_text())["samples"]
        if set(samples) == {p.name for p in SAMPLES}:
            print(name, "device evidence already complete", flush=True)
            return
    with (OUT / f"devices_{name}.log").open("a") as log:
        subprocess.run(["/usr/bin/time", "-l", PY, "-I", str(REPO / "scripts/t4m_probe_devices.py"), str(src),
                        str(OUT / f"devices_{name}.json"), *map(str,SAMPLES)], cwd=REPO, env=ENV,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    baseline = Path(sys.argv[1]).resolve()
    jobs = (("baseline", baseline), ("candidate", REPO / "src"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda job: run("t4m_run_eval.py", *job), jobs))
    devices("baseline", baseline)
    devices("candidate", REPO / "src")
    run("t4m_report_tables.py")


if __name__ == "__main__":
    main()
