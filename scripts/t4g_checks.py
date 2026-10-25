"""T4g roll-base / preset evidence and thumbnail prewarming, using only the two own development rolls.

  python -I scripts/t4g_checks.py SRC OUT_DIR [--thumbs]

Run once on T4f before changing the base, then on T4g. Keeps actual rendered before images: parameters alone cannot
replay the old inversion after a roll-base change. No original, saved orientation or user adjustment is written.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(SRC))

import numpy as np
from PIL import Image

from filmcolor.app import State
from filmcolor.frozen import check_source
from filmcolor.noref import STYLE_PRESETS

REPO = Path(__file__).resolve().parents[1]


def main():
    out = Path(sys.argv[2]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    record = dict(git_commit=subprocess.check_output(["git", "-C", str(SRC), "rev-parse", "HEAD"], text=True).strip(),
                  src_dirty=bool(subprocess.check_output(["git", "-C", str(SRC), "status", "--porcelain", "--", "."],
                                                        text=True).strip()), rolls={}, frames={})
    h = hashlib.sha256()
    for p in sorted(p for p in (SRC / "filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(SRC)).encode() + b"\0" + p.read_bytes() + b"\0")
    record["src_content_sha256"] = h.hexdigest()
    for roll in ("dsc6161", "dsc6233"):
        st = State(REPO / "data/raw/own" / roll, out / "unused_exports", allow_frozen=False, max_side=1000)
        t0 = time.perf_counter()
        ri = st.roll_info()
        record["rolls"][roll] = dict(info=ri, roll_seconds=time.perf_counter() - t0)
        print(roll, "used", ri["roll"]["used"], "flagged", sum(f["flag"] for f in ri["frames"].values()), flush=True)
        if "--thumbs" in sys.argv:
            for rel in st.files():
                st.thumbnail_png(rel)
            record["rolls"][roll]["roll_and_thumbs_seconds"] = time.perf_counter() - t0
        if roll != "dsc6161":
            continue
        for prefix in ("A008_", "A012_", "A013_"):
            rel = next(n for n in st.files() if n.startswith(prefix))
            path = st.resolve(rel)
            check_source(path)
            raw_hash = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
            m = st.open(rel, "color", preset="standard")
            rec = dict(raw_sha256=raw_hash, base=m["base"], rot90=m["rot90"], presets={})
            for name in STYLE_PRESETS:
                mm = st.set_preset(dict(token=st.token, preset=name))
                p = mm["auto"]["params"]
                disp = st.preview.render(p)
                img = np.round(np.clip(np.rot90(disp, m["rot90"]), 0, 1) * 255).astype(np.uint8)
                image = f"{prefix[:-1]}_{name}.png"
                Image.fromarray(img).save(out / image)
                rec["presets"][name] = dict(auto=mm["auto"], image=image)
                print(prefix[:-1], name, {k: p.get(k, 0) for k in ("ev", "temp", "tint", "contrast", "midtone", "shadows", "highlights")}, flush=True)
            record["frames"][prefix[:-1]] = rec
            st.preview = None
    (out / "checks.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
