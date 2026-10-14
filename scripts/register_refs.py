"""Register reference positives (e.g. shop JPEGs) paired by content (pair_shop_refs.py) against a RAW manifest.

Writes data/manifests/<ref_id>_reference.json (checksum, size, ICC description, software, pairing score per
reference) and fills ``reference_files`` in the RAW manifest. Rejects a pairing whose best match is not
clearly ahead of the runner-up.

  register_refs.py RAW_SOURCE_ID REF_ID PAIRING_JSON RIGHTS_TEXT
"""
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageCms

MIN_MARGIN = 0.05  # best NCC minus runner-up NCC

root = Path(__file__).resolve().parents[1]


def main(raw_source_id, ref_id, pairing_json, rights):
    ref_dir = root / "data" / "reference" / ref_id
    man_csv = root / "data" / "manifests" / f"{raw_source_id}.csv"
    pairing = json.loads(Path(pairing_json).read_text())
    rows = list(csv.DictReader(man_csv.open(newline="")))
    by_name = {Path(r["file"]).name: r for r in rows}
    refs, used = {}, {}
    for ref_name, p in sorted(pairing.items(), key=lambda kv: int(Path(kv[0]).stem.split("(")[1].rstrip(")"))):
        margin = p["ncc"] - p["ncc_second"]
        if margin < MIN_MARGIN:
            raise SystemExit(f"{ref_name}: ambiguous pairing (margin {margin:.3f} < {MIN_MARGIN})")
        if p["best"] in used:
            raise SystemExit(f"{ref_name}: {p['best']} already paired with {used[p['best']]}")
        used[p["best"]] = ref_name
        path = ref_dir / ref_name
        im = Image.open(path)
        icc = im.info.get("icc_profile")
        ex = im.getexif()
        refs[ref_name] = dict(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
            size=list(im.size), mode=im.mode,
            icc=ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip() if icc else None,
            software=ex.get(0x0131), source_camera=f"{ex.get(0x010f)} {ex.get(0x0110)}",
            raw=p["best"], ncc=round(p["ncc"], 4), ncc_margin=round(margin, 4),
            agrees_with_file_order=p["agrees"], rot90=p["rot90"], scale=p["scale"],
        )
    for name, r in by_name.items():
        r["reference_files"] = f"reference/{ref_id}/{used[name]}" if name in used else ""
    with man_csv.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    out = dict(reference_id=ref_id, raw_source_id=raw_source_id, rights=rights,
               pairing=dict(method="content alignment of negative density vs positive L* (pair_shop_refs.py)",
                            min_margin_required=MIN_MARGIN),
               n_references=len(refs), n_raws=len(rows), unpaired_raws=sorted(set(by_name) - set(used)),
               references=refs)
    out_json = root / "data" / "manifests" / f"{ref_id}_reference.json"
    out_json.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", out_json, "and reference_files in", man_csv, f"({len(refs)} paired)")


if __name__ == "__main__":
    main(*sys.argv[1:5])
