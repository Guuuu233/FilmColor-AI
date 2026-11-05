"""T5a §3: default routing of the Program AE copy rolls (foma pan100, Eastman 5294) under one source tree.
Usage: python -I scripts/t5a_roll_routing.py SRC baseline|candidate

Each roll folder is opened as the app opens it: a State on a temporary folder of symlinks to every NEF of the roll
(every member checked with check_source(allow_frozen=False) first), roll_info, then State.open(frame, "bw") with the
automatic base choice (base_mode None) for every frame of the two rolls the task names; the other two Eastman rolls
get roll_info statistics only. film="bw" as the T4z blind run (the film type of Eastman 5294 is not confirmed).
Writes experiments/t4_auto_ui/t5a/rolls_<label>.json and 240 px JPEGs (local, not committed) for the mosaics.
"""
import gc
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]).resolve()
LABEL = sys.argv[2]
assert LABEL in ("baseline", "candidate") and len(sys.argv) == 3
sys.path.insert(0, str(SRC))
import filmcolor.app as fc_app  # noqa: E402
from filmcolor.app import State, to8  # noqa: E402
from filmcolor.frozen import check_source  # noqa: E402
from filmcolor.holder import film_box  # noqa: E402

OUT = ROOT / "experiments/t4_auto_ui/t5a"
THUMBS = OUT / "roll_thumbs"
MATERIAL = Path("/Volumes/waizhiyingpan/胶片负片素材")
ROLLS = {  # name -> (folder, open every frame)
    "foma_pan100": ("foma pan100(d67 尼康数字化)-", True),
    "eastman_5294": ("Eastman 5294(尼康数字化)-", True),
    "eastman_5294-1": ("Eastman 5294(尼康数字化)-1", False),
    "eastman_5294-2": ("Eastman 5294(尼康数字化)-2", False),
}
LIMIT = 1.25  # ROLL_BASE_MAX of the candidate; computed here too so the baseline's roll bases are classified alike


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
    THUMBS.mkdir(parents=True, exist_ok=True)
    target = OUT / f"rolls_{LABEL}.json"
    result = dict(label=LABEL, src=str(SRC), src_content_sha256=fingerprint, script_sha256=script_sha,
                  max_side=1600, film="bw", roll_base_limit=LIMIT, rolls={})
    if target.exists():
        result = json.loads(target.read_text())
        assert result["src_content_sha256"] == fingerprint and result["script_sha256"] == script_sha

    def save():
        assert source_sha() == fingerprint and sha(__file__) == script_sha
        target.write_text(json.dumps(result, indent=1, allow_nan=False) + "\n")

    for name, (folder, open_all) in ROLLS.items():
        if name in result["rolls"] and result["rolls"][name].get("complete"):
            continue
        members = sorted((MATERIAL / folder / "原件").glob("*.NEF"))
        hashes = {p.name: check_source(p, allow_frozen=False) for p in members}
        assert len(set(p.name for p in members)) == len(members)
        with tempfile.TemporaryDirectory(prefix=f"fc_t5a_roll_{LABEL}_") as temp:
            d = Path(temp) / name
            d.mkdir()
            for p in members:
                (d / p.name).symlink_to(p)
            st = State(d, Path(temp) / "out", False, max_side=1600)
            ri = st.roll_info()
            sub = ri["sub_rolls"][""] if "" in ri["sub_rolls"] else None
            frames = {}
            for p in members:
                fe = ri["frames"][p.name]
                rr = fe.get("roll_rgb")
                frames[p.name] = dict(
                    raw_sha256=hashes[p.name], exposure=fe["exposure"], own_rgb=fe["own_rgb"], roll_rgb=rr,
                    level_diff=fe.get("level_diff"), chroma_part=fe.get("chroma_part"), own_error=fe.get("own_error"),
                    roll_max=max(rr) if rr else None, roll_gt_limit=bool(rr and max(rr) > LIMIT),
                    roll_implausible=fe.get("roll_implausible"),  # None under the baseline (field added in T5a)
                    route_roll_nat=bool(st._route_roll_nat(".", p.name, film="bw")))
            roll = dict(folder=folder, n=len(members), mixed_rolls=ri["mixed_rolls"], n_base=sub and sub["n_base"],
                        roll_warnings=(ri["roll"] or {}).get("warnings"), roll_error=ri.get("error"),
                        thumb_rev=fc_app.THUMB_REV, frames=frames, opened=open_all, complete=False)
            result["rolls"][name] = roll
            save()
            if open_all:
                for p in members:
                    rec = frames[p.name]
                    try:
                        meta = st.open(p.name, "bw")
                        pv = st.preview
                        disp = pv.render(pv.auto["params"])
                        fb = film_box(pv.holder)
                        core = disp if fb is None else disp[fb[0]:fb[0] + fb[2], fb[1]:fb[1] + fb[3]]
                        img = Image.fromarray(to8(disp))
                        img.thumbnail((240, 240))
                        jpg = THUMBS / f"{name}_{p.stem}_{LABEL}.jpg"
                        img.save(jpg, quality=90)
                        rec.update(base_method=meta["base"]["method"], base_rgb=meta["base"]["rgb"],
                                   base_warnings=meta["base"]["warnings"], status=pv.auto["status"],
                                   flags=pv.auto["flags"], params=pv.auto["params"], inversion=pv.inversion,
                                   input_warnings=meta.get("input_warnings"),
                                   input_center_p99=pv.report.get("input_center_p99"),
                                   core_srgb_q50=np.median(core, axis=(0, 1)).tolist(),
                                   display_sha256=hashlib.sha256(to8(disp).tobytes()).hexdigest(),
                                   thumb=jpg.name, thumb_sha256=sha(jpg))
                        st.preview = None
                        del pv, disp, core
                    except Exception as e:  # noqa: BLE001 - recorded
                        rec["error"] = f"{type(e).__name__}: {e}"
                    gc.collect()
                    save()
                    print(LABEL, name, p.name, rec.get("base_method"), rec.get("status"), rec.get("error", ""),
                          flush=True)
            roll["complete"] = True
            save()
            print(LABEL, name, "roll", len(members), "frames; routed",
                  sum(f["route_roll_nat"] for f in frames.values()), "roll>1.25",
                  sum(f["roll_gt_limit"] for f in frames.values()), flush=True)


if __name__ == "__main__":
    main()
