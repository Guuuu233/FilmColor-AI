"""T4m full-resolution app evaluation against e8a8c3d, with the unchanged T4f halves and NCC >= .7.

Usage: python -I scripts/t4m_run_eval.py baseline BASELINE_SRC
       python -I scripts/t4m_run_eval.py candidate SRC
       python -I scripts/t4m_run_eval.py guard_off SRC
       python -I scripts/t4m_run_eval.py holder_off SRC

Each invocation runs at most two child processes. The baseline source is a git archive of e8a8c3d's src and
 data/manifests (the frozen guard remains active). holder_off uses the candidate source with only holder.py and
preview.py restored from e8a8c3d. No image caches or frozen images are read by this runner.
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/t4_auto_ui/t4m"
THIN_T4L = dict(thin_bright_share_max=0.35, thin_span_zero=0.42, thin_span_ramp=0.20, thin_ev_min=0.0,
                thin_blacks_span=0.52, thin_blacks_gain=0.8, thin_blacks_max=0.12, thin_suppress_backlit_span=0.0)
EST_T4L = dict(THIN_T4L, wb_v3_cyan_conf_check=False)
STD = dict(preset="standard")


def run(src, name, ids, variants, manifest, app_config):
    ids_path = OUT / f"{name}_ids.txt"
    ids_path.write_text("\n".join(sorted(ids)) + "\n")
    cmd = ["/usr/bin/time", "-l", str(REPO / ".venv/bin/python"), "-I", str(REPO / "scripts/t4d_source_eval.py"),
           str(src), str(OUT / f"{name}.json"), f"--ids-from={ids_path}", f"--manifest={manifest}",
           "--variants=" + json.dumps(variants), "--app-config=" + json.dumps(app_config)]
    with (OUT / f"{name}.log").open("a") as log:
        subprocess.run(cmd, cwd=REPO, env=dict(os.environ, FILMCOLOR_DATA_ROOT="/Volumes/waizhiyingpan/filmcolor_data"),
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    print(name, "complete", flush=True)


def main():
    mode, src = sys.argv[1], Path(sys.argv[2]).resolve()
    if mode not in ("baseline", "candidate", "guard_off", "holder_off"):
        raise SystemExit(__doc__)
    OUT.mkdir(parents=True, exist_ok=True)
    variants = dict(standard=STD)
    if mode == "candidate":
        variants.update({p: dict(preset=p) for p in ("natural", "portrait", "vivid", "print_classic")})
        variants.update(thin_t4l=dict(preset="standard", cfg=THIN_T4L),
                        no_cyan=dict(preset="standard", cfg=dict(wb_v3_cyan_conf_check=False)))
    elif mode == "guard_off":
        variants["t4l_est"] = dict(preset="standard", cfg=EST_T4L)
    app_config = dict(WHITE_B_GUARD=None) if mode == "guard_off" else {}
    frames = json.loads((REPO / "data/manifests/s3_dev_frames.json").read_text())["frames"]
    old_a = json.loads((REPO / "experiments/t4_auto_ui/t4f/eval_t4f_a.json").read_text())["frames"]
    halves = {"a": [], "b": []}
    for frame in frames:
        halves["a" if frame["id"] in old_a else "b"].append(frame["id"])
    with ThreadPoolExecutor(max_workers=int(os.environ.get("FILMCOLOR_EVAL_WORKERS", "2"))) as pool:
        list(pool.map(lambda part: run(src, f"{mode}_{part}", halves[part], variants,
                                      "data/manifests/s3_dev_frames.json", app_config), halves))
    cross = json.loads((REPO / "data/manifests/bn_rb_frames.json").read_text())["frames"]
    # 124 candidates; the fixed NCC filter is applied when reporting (historically 123 pass).
    cross_variants = {k: v for k, v in variants.items() if k not in ("natural", "portrait", "vivid", "print_classic")}
    run(src, f"{mode}_cross", [f["id"] for f in cross if not f["s3_5"]], cross_variants,
        "data/manifests/bn_rb_frames.json", app_config)


if __name__ == "__main__":
    main()
