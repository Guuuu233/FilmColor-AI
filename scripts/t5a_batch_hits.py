"""T5a §4: input_warnings hit list on the research's Batch 19-21 frames (41), with status / params / inversion.
Usage: python -I scripts/t5a_batch_hits.py SRC baseline|candidate RESEARCH_JSON

The frame list (name, rel, film) is read from the research JSON (fresh_batches_t4y.json; its results are not used).
Each frame is opened alone (temporary folder holding only its symlink, check_source(allow_frozen=False) first) with
State.open(frame, film) as the app does, at max_side 1600. No images are written.
"""
import gc
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]).resolve()
LABEL = sys.argv[2]
RESEARCH = Path(sys.argv[3])
assert LABEL in ("baseline", "candidate") and len(sys.argv) == 4
sys.path.insert(0, str(SRC))
import filmcolor.app as fc_app  # noqa: E402
from filmcolor.app import State, to8  # noqa: E402
from filmcolor.frozen import check_source  # noqa: E402
from filmcolor.holder import film_box  # noqa: E402

OUT = ROOT / "experiments/t4_auto_ui/t5a"
MATERIAL = Path("/Volumes/waizhiyingpan/胶片负片素材")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_sha():
    h = hashlib.sha256()
    for p in sorted(p for p in (SRC / "filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(SRC)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def main():
    assert os.environ.get("FILMCOLOR_DATA_ROOT") == "/Volumes/waizhiyingpan/filmcolor_data"
    assert Path(fc_app.__file__).resolve().is_relative_to(SRC)
    fingerprint, script_sha = source_sha(), sha(__file__)
    research = json.loads(RESEARCH.read_text())
    rows = [dict(batch=b, name=r["name"], rel=r["rel"], film=r["film"]) for b, rs in research["batches"].items() for r in rs]
    assert len(rows) == 41 and len({r["rel"] for r in rows}) == 41
    target = OUT / f"batches_{LABEL}.json"
    result = dict(label=LABEL, src=str(SRC), src_content_sha256=fingerprint, script_sha256=script_sha,
                  research_sha256=sha(RESEARCH), max_side=1600, frames={})
    if target.exists():
        result = json.loads(target.read_text())
        assert result["src_content_sha256"] == fingerprint and result["script_sha256"] == script_sha
        assert result["research_sha256"] == sha(RESEARCH)

    def save():
        assert source_sha() == fingerprint and sha(__file__) == script_sha
        target.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")

    for row in rows:
        if row["rel"] in result["frames"]:
            continue
        p = MATERIAL / row["rel"]
        digest = check_source(p, allow_frozen=False)
        rec = dict(row, raw_sha256=digest)
        with tempfile.TemporaryDirectory(prefix=f"fc_t5a_batch_{LABEL}_") as temp:
            d = Path(temp) / "f"
            d.mkdir()
            (d / p.name).symlink_to(p)
            try:
                st = State(d, Path(temp) / "out", False, max_side=1600)
                st.roll_info()
                meta = st.open(p.name, row["film"])
                pv = st.preview
                disp = pv.render(pv.auto["params"])
                fb = film_box(pv.holder)
                core = disp if fb is None else disp[fb[0]:fb[0] + fb[2], fb[1]:fb[1] + fb[3]]
                rec.update(status=pv.auto["status"], flags=pv.auto["flags"], params=pv.auto["params"],
                           inversion=pv.inversion, base_method=meta["base"]["method"],
                           input_warnings=meta.get("input_warnings"),
                           input_center_p99=pv.report.get("input_center_p99"),
                           holder_share=float(pv.holder.mean()),
                           core_srgb_q50=np.median(core, axis=(0, 1)).tolist(),
                           display_sha256=hashlib.sha256(to8(disp).tobytes()).hexdigest())
                st.preview = None
                del pv, disp, core
            except Exception as e:  # noqa: BLE001 - recorded
                rec["error"] = f"{type(e).__name__}: {e}"
        gc.collect()
        result["frames"][row["rel"]] = rec
        save()
        print(LABEL, row["name"], rec.get("status", rec.get("error")),
              [w["code"] for w in rec.get("input_warnings") or []], flush=True)
    save()


if __name__ == "__main__":
    main()
