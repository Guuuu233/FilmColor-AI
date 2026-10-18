"""BlueNeg roll-base study: frame list in the s3_dev_frames format (data/manifests/bn_rb_frames.json).

Frames of the download selection whose negative and print are on disk and verified against the Hugging Face LFS
sha256 (fetch_blueneg.py records). Roll = <yyyymm><letter> (PREREG_BLUENEG.md); ids, stems and paths as in
s3_dev_frames.json so the S3 cache of the 27 earlier frames is reused. ``s3_5`` marks the 27 frames of the S3-5
base attribution. Sealed frames are refused by name and roll (filmcolor.frozen).

  bn_frames.py SELECTION_JSON FETCH_JSON DATA_ROOT OUT_JSON
"""
import json
import sys
from pathlib import Path

from filmcolor.frozen import check_name, check_roll

DEV = Path(__file__).resolve().parents[1] / "data/manifests/s3_dev_frames.json"


def roll_of(x):
    rid = x["roll_id"]
    return "ds-" + (x["date"][:6] + rid[-1] if rid[-1].isalpha() else rid)


def main(sel_json, fetch_json, root, out_json):
    root = Path(root)
    ok = {r["path"]: r["sha256"] for r in json.loads(Path(fetch_json).read_text())["files"]
          if r["status"].endswith("verified")}
    s35 = {f["stem"] for f in json.loads(DEV.read_text())["frames"] if f["domain"] == "blueneg"}
    frames = []
    for x in json.loads(Path(sel_json).read_text())["frames"]:
        n, p = x["negative_path"], x["printed_path"]
        if n not in ok or p not in ok or not (root / n).exists() or not (root / p).exists():
            continue
        check_name(n)
        check_roll(roll_of(x))
        frames.append(dict(id=f"blueneg:{x['filename']}", domain="blueneg", stem=x["filename"], roll=roll_of(x),
                           roll_id_dataset=x["roll_id"], film=x["film_type"], partition=x["partition"],
                           raw=str(Path("data/raw/public/blueneg") / n), ref=str(Path("data/raw/public/blueneg") / p),
                           ref_kind="srgb_no_icc", color_space_assumed="sRGB", raw_sha256=ok[n], ref_sha256=ok[p],
                           s3_5=x["filename"] in s35))
    out = dict(design="experiments/s3_rollbase_v1/PREREG_BLUENEG.md", source=sel_json,
               counts=dict(frames=len(frames), rolls=len({f["roll"] for f in frames}), s3_5=sum(f["s3_5"] for f in frames)),
               frames=frames)
    Path(out_json).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(out["counts"])


if __name__ == "__main__":
    main(*sys.argv[1:5])
