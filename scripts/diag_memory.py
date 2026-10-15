"""Per-stage memory of one conversion: wraps the stage functions used by filmcolor.pipeline and
records, per call, the tracemalloc peak above the stage's starting level (numpy allocations; not
LibRaw's own C++ buffers) and the process max RSS after the stage (monotonic, includes LibRaw).
Whole-process peak footprint must still be measured with /usr/bin/time -l (memory caliber).

usage: diag_memory.py out.json raw [--auto]
"""
import json
import resource
import sys
import time
import tracemalloc

from filmcolor import pipeline

STAGES = ["load_raw", "linearize", "saturation_mask", "estimate_base_auto", "demosaic_linear",
          "base_from_regions", "load_scanner", "estimate_base_scanner", "invert_negative",
          "estimation_input", "estimate_params_v2", "render_image", "quality_checks", "srgb_eotf",
          "write_tiff16", "sha256"]
GiB = 2 ** 30
log = []


def wrap(name, fn):
    def inner(*a, **k):
        cur0, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        t = time.time()
        out = fn(*a, **k)
        cur1, peak = tracemalloc.get_traced_memory()
        log.append(dict(stage=name, seconds=round(time.time() - t, 2), start_gib=round(cur0 / GiB, 3),
                        peak_gib=round(peak / GiB, 3), stage_extra_gib=round((peak - cur0) / GiB, 3),
                        end_gib=round(cur1 / GiB, 3),
                        maxrss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GiB, 3)))
        return out
    return inner


def main():
    out, raw = sys.argv[1], sys.argv[2]
    auto = "--auto" in sys.argv[3:]
    for name in STAGES:
        if hasattr(pipeline, name):
            setattr(pipeline, name, wrap(name, getattr(pipeline, name)))
    tracemalloc.start()
    kw = {}
    if auto:
        from filmcolor.noref import NOREF_V2
        kw["auto"] = NOREF_V2
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        pipeline.write(raw, d, **kw)
    _, peak = tracemalloc.get_traced_memory()
    res = dict(input=raw, auto=auto, traced_peak_gib=round(peak / GiB, 3),
               maxrss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / GiB, 3), stages=log)
    json.dump(res, open(out, "w"), indent=1)
    for r in log:
        print(f"{r['stage']:22s} start {r['start_gib']:6.2f}  peak {r['peak_gib']:6.2f}  "
              f"(+{r['stage_extra_gib']:5.2f})  end {r['end_gib']:6.2f}  maxrss {r['maxrss_gib']:6.2f}  {r['seconds']}s")
    print("traced peak", res["traced_peak_gib"], "GiB; maxrss", res["maxrss_gib"], "GiB")


if __name__ == "__main__":
    main()
