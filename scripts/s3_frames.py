"""Development frame list for the S3 experiments (docs/S3_ML_DESIGN.md section 1): one row per negative /
reference pair with its domain, roll, files and their SHA256, and the experiments it belongs to.
Every file is checked against the frozen test manifests; the list is refused if any is frozen.

  s3_frames.py EXT_ROOT OUT_JSON

EXT_ROOT: the external material folder (ext_own); its file hashes come from the audit inventory.
"""
import json
import subprocess
import sys
from pathlib import Path

from filmcolor.frozen import check_hashes, sha256_file

sys.path.insert(0, str(Path(__file__).parent))
from noref_global import STEMS  # noqa: E402

NLP_BW = "_DSF9228"
EXT_DEV = {  # audit roll id -> source folder (docs/S3_EXT_DATA_AUDIT.md section 2)
    "own-g200-20240320-dsc4889": "Kodak G200（尼康数字化）-",
    "own-e5207-20240323-dsc8242": "Eastman 5207(ecn-2次抛冲洗 尼康数字化)-",
    "own-e5207-20240401-dsc6161": "Eastman 5207(ecn-2次抛冲洗 尼康数字化)-3",
    "own-e5207-20240401-dsc6233": "Eastman 5207(ecn-2次抛冲洗 尼康数字化)-1",
}
EXT_BLANK = {"A148", "A210"}  # blank frames, kept out of training and development scoring (design section 1)


def capture_date(path):
    out = subprocess.run(["exiftool", "-s3", "-DateTimeOriginal", str(path)], capture_output=True, text=True, check=True)
    return out.stdout.strip()[:10].replace(":", "")


def main(ext_root, out_json):
    ext_root = Path(ext_root)
    frames = []
    for stem in STEMS["nlp"]:
        if stem == NLP_BW:
            continue
        raw = Path(f"data/raw/public/nlp_official/{stem}.RAF")
        frames.append(dict(id=f"nlp:{stem}", domain="nlp", stem=stem, roll=f"nlp-{capture_date(raw)}",
                           raw=str(raw), ref=f"data/reference/nlp_default/{stem}-positive.tif", ref_kind="icc",
                           experiments=["A"]))
    for stem in STEMS["shop"]:
        frames.append(dict(id=f"shop:{stem}", domain="shop", stem=stem, roll="shop_r01",
                           raw=f"data/raw/own/ultramax400_r01/{stem}.NEF",
                           ref=str(Path(f"experiments/s2_shop/ref/{stem}.jpg").resolve().relative_to(Path.cwd())),
                           ref_kind="icc", experiments=["A", "B"]))
    sel = json.load(open("experiments/s3_blueneg/selection.json"))
    fetched = {f["path"]: f["sha256"] for f in json.load(open("experiments/s3_blueneg/fetch.json"))["files"]}
    for f in sel["frames"]:
        root = "data/raw/public/blueneg/"
        frames.append(dict(id=f"blueneg:{f['filename']}", domain="blueneg", stem=f["filename"], roll=f["roll_id"],
                           film=f["film_type"], raw=root + f["negative_path"], ref=root + f["printed_path"],
                           ref_kind="srgb_no_icc", raw_sha256=fetched[f["negative_path"]],
                           ref_sha256=fetched[f["printed_path"]], experiments=["A"]))
    inv = {r["id"]: r for r in json.load(open("experiments/s3_data_audit/inventory.json")) if r.get("id")}
    pairing = json.load(open("experiments/s3_data_audit/pairing.json"))
    by_dir = {v: k for k, v in EXT_DEV.items()}
    for p in pairing["pairs"]:
        neg, ref = inv[p["best"]], inv[p["ref"]]
        if ref["orig_dir"] not in by_dir or not p["mutual"] or neg["orig_dir"] != ref["orig_dir"] or p["best"] in EXT_BLANK:
            continue
        frames.append(dict(id=f"ext:{p['best']}", domain="ext_own", stem=p["best"], roll=by_dir[ref["orig_dir"]],
                           film=neg["film_or_source"], raw=str(ext_root / neg["rel"]), ref=str(ext_root / ref["rel"]),
                           # 6 prints of dsc6161 were exported without ICC or EXIF (8256x5504): read as sRGB, flagged
                           ref_kind="icc" if ref["ProfileDescription"] else "srgb_assumed",
                           raw_sha256=neg["sha256"], ref_sha256=ref["sha256"],
                           orig_name=neg["orig_name"], ref_id=p["ref"], experiments=["B"]))
    for f in frames:  # local files are hashed now; downloaded and external ones carry verified hashes
        f.setdefault("raw_sha256", sha256_file(f["raw"]))
        f.setdefault("ref_sha256", sha256_file(f["ref"]))
    check_hashes([h for f in frames for h in (f["raw_sha256"], f["ref_sha256"])])
    counts = {d: sum(f["domain"] == d for f in frames) for d in ("nlp", "shop", "blueneg", "ext_own")}
    rolls = {d: len({f["roll"] for f in frames if f["domain"] == d}) for d in counts}
    json.dump(dict(design="docs/S3_ML_DESIGN.md v2.1", counts=counts, rolls=rolls,
                   excluded=dict(nlp=[NLP_BW], shop=["DSC_3822", "DSC_3858"], ext_own=sorted(EXT_BLANK)),
                   frozen_check="no file of this list is in data/manifests/frozen_test_*.json (checked when built)",
                   frames=frames), open(out_json, "w"), ensure_ascii=False, indent=1)
    print(counts, rolls)


if __name__ == "__main__":
    main(*sys.argv[1:3])
