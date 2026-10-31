"""Trace post-Highlight-Hold quantiles for the two T4o check-set rescue targets.
Usage: python -I scripts/t4o_probe_cross.py
Uses the evaluation's default roll folders and frozen guards. Tracing does not change numerical results.
"""
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4o"


def main():
    sys.argv = [str(ROOT / "scripts/t4d_source_eval.py"), str(ROOT / "src")]
    ev = runpy.run_path(sys.argv[0], run_name="t4o_cross_probe")
    frames = json.loads((ROOT / "data/manifests/bn_rb_frames.json").read_text())["frames"]
    ev["roll_state"].__globals__["FRAMES"] = frames
    noref = ev["noref"]
    original = noref._encoded_quantiles
    result = {}
    states = {}
    for key in ("blueneg:199807A-19-orlando-disney", "blueneg:199807A-33-orlando-disney"):
        f = next(f for f in frames if f["id"] == key)
        state = ev["roll_state"](f, states, 1000)
        path = Path(f["raw"]) if os.path.isabs(f["raw"]) else ROOT / f["raw"]
        state.open(path.name, "color")
        calls = []

        def traced(px, params, qs=(2, 50, 98)):
            values = original(px, params, qs)
            calls.append(dict(qs=list(qs), values=values))
            return values

        noref._encoded_quantiles = traced
        try:
            pv = state.preview
            est = noref.estimate_params_v3(*noref.estimation_input(pv.full_positive, pv.image_region),
                                          span_d_g=pv.inversion["d_hi"][1] - pv.inversion["d_lo"][1])
            pair = [c for c in calls if c["qs"] == [25, 90]][-1]
            median = [c for c in calls if c["qs"] == [50]][-1]
            result[key] = dict(span_d_g=pv.inversion["d_hi"][1] - pv.inversion["d_lo"][1],
                               rescue_input_q25_q90=pair["values"], rescue_input_median=median["values"][0],
                               backlit=est["diagnostics"]["backlit_check"], params=est["params"])
        finally:
            noref._encoded_quantiles = original
            state.preview = None
    (OUT / "cross_target_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
