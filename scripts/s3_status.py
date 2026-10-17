"""Processing status of every S3 development frame: labelled, pipeline failure (the base inversion could not
produce P under the design definition), or excluded before processing by the design (separate categories;
a pipeline failure is not a design exclusion).

  s3_status.py FRAMES_JSON LABEL_DIR OUT_JSON      (needs FILMCOLOR_DATA_ROOT for the failure records)
"""
import json
import sys
from collections import Counter
from pathlib import Path

from filmcolor.paths import data_dir


def main(frames_json, label_dir, out_json):
    fl = json.load(open(frames_json))
    rows = []
    for f in fl["frames"]:
        cache = data_dir("s3_ml", "cache", f["domain"])
        failed = cache / f"{f['stem']}.failed.json"
        if (Path(label_dir) / f["domain"] / f"{f['stem']}.json").exists():
            status, detail = "labelled", None
        elif failed.exists():
            status, detail = "pipeline_failure", json.loads(failed.read_text())
        else:
            status, detail = "missing", None
        rows.append(dict(id=f["id"], domain=f["domain"], roll=f["roll"], status=status, detail=detail))
    if any(r["status"] == "missing" for r in rows):
        raise SystemExit("frames without label or failure record: " + ", ".join(r["id"] for r in rows if r["status"] == "missing"))
    by = {d: Counter(r["status"] for r in rows if r["domain"] == d) for d in sorted({r["domain"] for r in rows})}
    out = dict(design_exclusions=dict(fl["excluded"], reason=dict(
                   nlp="black-and-white negative", shop="roll leader / mostly unexposed (S2)",
                   ext_own="blank frames (audit)")),
               pipeline_failures=[r for r in rows if r["status"] == "pipeline_failure"],
               counts=by,
               success_rate={d: round(c["labelled"] / sum(c.values()), 4) for d, c in by.items()},
               frames=rows)
    Path(out_json).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(by, out["success_rate"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
