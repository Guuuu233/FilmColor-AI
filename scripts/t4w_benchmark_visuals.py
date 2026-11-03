"""Render two manifest-listed improvement examples with each source's measured parameters."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]).resolve()
LABEL = sys.argv[2]
assert LABEL in ("baseline", "candidate")
sys.path.insert(0, str(SRC))
from filmcolor.app import State, to8
from filmcolor.frozen import check_source

OUT = ROOT / "experiments/t4_auto_ui/t4w"
IDS = ("ext:A124", "blueneg:19960812D-25-las-vegas")


def main():
    assert os.environ.get("FILMCOLOR_DATA_ROOT") == "/Volumes/waizhiyingpan/filmcolor_data"
    frames = {}
    for manifest in ("s3_dev_frames.json", "bn_rb_frames.json"):
        frames.update({f["id"]: f for f in json.loads((ROOT / "data/manifests" / manifest).read_text())["frames"]})
    measured = {}
    version = "t4v" if LABEL == "baseline" else "t4w"
    for part in ("a", "b", "cross"):
        measured.update(json.loads((ROOT / f"experiments/t4_auto_ui/{version}/candidate_{part}.json").read_text())["frames"])
    result = {}
    for key in IDS:
        f, record = frames[key], measured[key]
        raw = (ROOT / f["raw"]).resolve()
        digest = check_source(raw, allow_frozen=False)
        assert digest == record["raw_sha256"]
        # Use exactly the evaluator's roll folder (already populated from the guarded manifest).
        folder = Path("/tmp/fc_t4d/rolls") / f["domain"] / f["roll"]
        for p in folder.iterdir():
            if p.is_file():
                check_source(p, allow_frozen=False)
        st = State(folder, OUT / "exports", False, max_side=1000)
        st.roll_info()
        st.open(raw.name, "color")
        assert list(st.preview.inversion["d_hi"]) == record["d_hi"]
        params = record["variants"]["standard"]["params"]
        before = hashlib.sha256(st.preview.positive.tobytes()).hexdigest()
        disp = st.preview.render(params)
        assert before == hashlib.sha256(st.preview.positive.tobytes()).hexdigest()
        assert np.isfinite(disp).all()
        target = OUT / (key.replace(":", "_") + "_" + LABEL + ".jpg")
        image = Image.fromarray(to8(disp))
        image.thumbnail((900, 900))
        image.save(target, quality=92)
        result[key] = dict(raw_sha256=digest, preview_image=target.name,
                           preview_image_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                           params=params, de2000_median=record["variants"]["standard"]["metrics"]["de2000_median"])
        st.preview = None
    h = hashlib.sha256()
    for p in sorted(p for p in (SRC / "filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(SRC)).encode() + b"\0" + p.read_bytes() + b"\0")
    (OUT / f"benchmark_visuals_{LABEL}.json").write_text(json.dumps(dict(source_sha256=h.hexdigest(), frames=result), indent=2) + "\n")


if __name__ == "__main__":
    main()
