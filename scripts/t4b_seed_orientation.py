"""Seed the app's per-photo orientation (State.set_orientation) from the reference alignment of a roll (T4b).

  t4b_seed_orientation.py ROLL_FOLDER

For each file of ROLL_FOLDER whose frame has experiments/s3_ml/user_labels/<frame>.json (the user's edit aligned to
our render: the reference turned ``alignment.rot90`` quarter turns counter-clockwise matches ours), stores the turn
that shows ours upright, (4 - rot90) % 4, unless an orientation is already saved for it. Prints what it set.
"""
import json
import sys
from pathlib import Path

from filmcolor.app import State

REPO = Path(__file__).resolve().parents[1]


def main(folder: str):
    st = State(Path(folder), Path("/tmp/fc_t4b_out"), allow_frozen=False)
    for rel in st.files():
        lab = REPO / "experiments/s3_ml/user_labels" / f"{Path(rel).name.split('_')[0]}.json"
        if not lab.exists():
            continue
        k = (4 - int(json.loads(lab.read_text())["alignment"]["rot90"])) % 4
        cur, src = st.orientation(st.resolve(rel))
        if src == "saved":
            print(rel, "kept saved", cur)
            continue
        if k:
            st.set_orientation(dict(path=rel, rot90=k))
            print(rel, "set", k)


if __name__ == "__main__":
    main(sys.argv[1])
