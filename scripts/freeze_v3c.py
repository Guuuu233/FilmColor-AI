"""Write data/manifests/frozen_test_v3.json from the confirmed v3c hold-out table (2026-10-23).

  freeze_v3c.py MATERIAL_ROOT V3C_CANDIDATES_JSON NEW_FILES_JSON OUT_JSON

Seals the 3 units selected by the v3c stratified rule (Canon EOS R5 body-e60dbcf684, 26 frames; Sony ILCE-7M4
body-d19097ecea, 5 frames; Nikon Z 6II body-e8199a65dd, 5 frames; 36 files in total). Recomputes the SHA256 of
every file on disk and checks it matches new_files.json; reads no pixels. Registers both SHA256 hashes (files)
and stem / unit blocks (blocked.names, blocked.roll_ids) for filmcolor.frozen and scripts/import_check.py.
"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from filmcolor.frozen import sha256_file, stem_of  # noqa: E402


def main(mat_root, v3c_json, new_files_json, out_json):
    mat = Path(mat_root)
    vc_raw = Path(v3c_json).read_bytes()
    nf_raw = Path(new_files_json).read_bytes()
    vc = json.loads(vc_raw)
    nf = {f["rel"]: f for f in json.loads(nf_raw)}
    by_unit = {u["unit"]: u for u in vc["units"]}
    sealed_cands = [c for c in vc["candidates"] if c["proposal"] == "seal"]

    units_out, files_out, blocked_names, blocked_rolls = [], [], set(), []
    for c in sealed_cands:
        u = by_unit[c["unit"]]
        blocked_rolls.append(c["unit"])
        units_out.append(dict(
            unit=c["unit"], run_label=c["run_label"], camera=c["camera"],
            stratum=c["stratum"], position=c["position"], sort_key=c["sort_key"],
            sessions=c["sessions"], frames=len(u["frames"]),
            source_confirmed="group member provided, confirmed usable for research (2026-10-23)"))
        for rel in sorted(u["frames"]):
            f = nf[rel]
            h = sha256_file(mat / rel)
            if h != f["sha256"]:
                raise SystemExit(f"{rel}: sha256 mismatch ({h} != {f['sha256']})")
            blocked_names.add(stem_of(rel))
            if f.get("orig_name"):
                blocked_names.add(stem_of(f["orig_name"]))
            files_out.append(dict(
                id=f"material:{rel}", file_id=f["id"], rel=rel, orig_name=f["orig_name"],
                unit=c["unit"], camera=f["camera"], sha256=h))

    files_out.sort(key=lambda x: x["rel"])
    res = dict(
        id="frozen_test_v3",
        created="2026-10-23",
        decision="2026-10-23 T1b/T3 review and user confirmation: seal 3 camera-body units (36 frames) under rule v3c for held-out validation of base estimation and inversion across unseen camera bodies; never decode during development",
        rule=vc["rule"],
        v3c_candidates_sha256=hashlib.sha256(vc_raw).hexdigest(),
        new_files_sha256=hashlib.sha256(nf_raw).hexdigest(),
        units=units_out,
        blocked=dict(names=sorted(blocked_names), roll_ids=sorted(blocked_rolls)),
        files=files_out,
    )
    Path(out_json).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {out_json}: {len(units_out)} units, {len(files_out)} files, {len(blocked_names)} blocked names")


if __name__ == "__main__":
    main(*sys.argv[1:5])
