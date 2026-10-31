"""Reproduce the disputed mid-grey gate on 199807B-08 using the full app evaluation path.
Usage: python -I scripts/t4n_probe_wb.py
Run with FILMCOLOR_DATA_ROOT set, as for the full evaluation. No thresholds are fitted here.
"""
import json
import runpy
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"experiments/t4_auto_ui/t4n"


def main():
    sys.argv=[str(ROOT/"scripts/t4d_source_eval.py"),str(ROOT/"src")]
    evaluator=runpy.run_path(sys.argv[0],run_name="t4n_wb_probe")
    frames=json.loads((ROOT/"data/manifests/bn_rb_frames.json").read_text())["frames"]
    evaluator["roll_state"].__globals__["FRAMES"]=frames
    target=next(f for f in frames if f["id"]=="blueneg:199807B-08-orlando-seaworld")
    state=evaluator["roll_state"](target,{},1000)
    state.open(Path(target["raw"]).name,"color")
    wb=state.preview.auto["diagnostics"]["wb"]
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"target_wb.json").write_text(json.dumps(wb,indent=2)+"\n")
    print(json.dumps(wb,indent=2))


if __name__=="__main__":
    main()
