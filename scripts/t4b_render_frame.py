"""Render frames exactly as the app opens them (automatic base choice + NOREF_V3 starting point), with a given
source tree, so the code of an earlier commit can be rendered next to the current one (T4b before / after sheets).

  t4b_render_frame.py SRC_DIR OUT_DIR ROLL_FOLDER REL [REL ...]

SRC_DIR: the ``src`` folder whose ``filmcolor`` package is used (e.g. a ``git worktree`` of 4eb2678, or this repo's
src). Writes OUT_DIR/<stem>.png (sRGB 8-bit preview, longest side 1000, sensor orientation) and OUT_DIR/<stem>.json
(base mode, white point, auto parameters and diagnostics, software commit). Frozen sets are refused by State.resolve.
"""
import json
import subprocess
import sys
from pathlib import Path

SRC = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import filmcolor  # noqa: E402
from filmcolor.app import State  # noqa: E402


def main(out_dir: str, folder: str, rels: list[str]):
    assert Path(filmcolor.__file__).resolve().is_relative_to(SRC), filmcolor.__file__
    commit = subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(SRC), "status", "--porcelain", "--", "."], capture_output=True,
                                text=True).stdout.strip())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    st = State(Path(folder), Path("/tmp/fc_t4b_out"), allow_frozen=False, max_side=1000)
    st.roll_info()
    for rel in rels:
        m = st.open(rel, "color")
        pv = st.preview
        disp = pv.render(pv.auto["params"])
        Image.fromarray(np.round(np.clip(disp, 0, 1) * 255).astype(np.uint8)).save(out / f"{Path(rel).stem}.png")
        info = dict(rel=rel, folder=folder, src=str(SRC), git_commit=commit, src_dirty=dirty, base_method=m["base"]["method"],
                    d_hi=list(pv.inversion["d_hi"]), white_point=pv.report.get("white_point"),
                    auto_config=pv.auto.get("config"), auto_rev=pv.auto.get("config_rev"),
                    params=pv.auto["params"], flags=pv.auto.get("flags"), diagnostics=pv.auto.get("diagnostics"))
        (out / f"{Path(rel).stem}.json").write_text(json.dumps(info, indent=1, default=float))
        print(rel, m["base"]["method"], {k: round(v, 3) for k, v in pv.auto["params"].items()
                                         if isinstance(v, float)}, flush=True)


if __name__ == "__main__":
    main(sys.argv[2], sys.argv[3], sys.argv[4:])
