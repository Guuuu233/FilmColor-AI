"""Reconciliation of the dataset report's counts by unit (review of 2026-10-18): physical files, unique contents,
frames, negatives after the by-eye exclusions, known rolls, negative -> reference relations, and frames with at
least one reference per strength of evidence. Reads only the outputs of audit_dataset_report.py /
audit_dataset_pairing.py, the visual review, the dev manifest and the S3 user-label alignments; nothing under the
material folder.

  audit_dataset_units.py CATALOG_JSON PAIRING_JSON VISUAL_REVIEW_JSON OUT_JSON
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STRONG = ("dev", "frozen", "official")      # registered / checked by S3, sealed manifests, release metadata
MEDIUM = ("photos", "content")             # content match (luminance thumbnails), not registered
WEAK = ("index",)                          # material index only


def main(catalog, pairing, visual, out_json):
    c = json.loads(Path(catalog).read_text())
    files = {f["rel"]: f for f in c["files"]}
    v = json.loads(Path(visual).read_text())
    nn = v["not_a_negative"]
    excluded = set(nn["positive_certain"]) | set(nn["positive_likely"]) | set(nn["not_film"])
    pairs = json.loads(Path(pairing).read_text())["pairs"]
    candidates = defaultdict(list)
    for p in pairs:
        if p["level"] == "candidate":
            candidates[p["best"]].append(p["ref"])
    dev = {f["raw_sha256"]: f for f in json.loads((REPO / "data/manifests/s3_dev_frames.json").read_text())["frames"]}
    user_stems = {p.stem for p in (REPO / "experiments/s3_ml/user_labels").glob("*.json")}

    out = dict(files=dict(on_disk_material=len(files), unique_content=sum(f["dup_of"] is None for f in files.values())))
    for kind in ("colour_neg", "bw_neg"):
        frs = [fr for fr in c["frames"] if fr["kind"] == kind]
        ids = lambda fr: {files[r]["id"] for r in fr["files"]} - {None}
        kept = [fr for fr in frs if not ids(fr) & excluded]
        rel = Counter()
        best = Counter()
        two_graders = 0
        for fr in kept:
            mem = [files[r] for r in fr["files"]]
            n = set()
            for m in mem:
                n |= {("index", p) for p in m["partners"]}
                n |= {("content", r) for r in m.get("content_refs", [])}
                n |= {("photo", k) for k in m.get("photo_refs", {})}
                d = dev.get(m["sha256"])
                if d:
                    n.add(("dev_ref", d["ref_sha256"]))
                    if d["stem"] in user_stems:
                        n.add(("user_edit", d["stem"]))
            if fr["partition"] and "official" in fr["pair_levels"]:
                n.add(("blueneg_print", fr["frame"]))
            two_graders += any(k == "user_edit" for k, _ in n)
            for k, _ in n:
                rel[k] += 1
            lv = set(fr["pair_levels"])
            best["strong" if lv & set(STRONG) else "medium" if lv & set(MEDIUM) else "weak" if lv & set(WEAK)
                 else "candidate_only" if any(r in candidates for r in fr["files"]) else "none"] += 1
        rolls = {fr["roll"] for fr in kept if fr["roll"]}
        out[kind] = dict(
            frames=len(frs), excluded_not_negative=len(frs) - len(kept), negatives=len(kept),
            known_rolls=len(rolls), frames_with_known_roll=sum(fr["roll"] is not None for fr in kept),
            frozen_negatives=sum(fr["frozen"] for fr in kept),
            frames_by_strongest_reference=dict(best),
            frames_with_reference=sum(n for k, n in best.items() if k in ("strong", "medium", "weak")),
            frames_with_reference_dev=sum(1 for fr in kept if fr["pair"] and not fr["frozen"]),
            frames_with_reference_frozen=sum(1 for fr in kept if fr["pair"] and fr["frozen"]),
            frames_with_user_edit_and_shop=two_graders,
            # relations per evidence source; the sources overlap (an index partner can be the same file as a
            # content match or a dev reference: they use different identifiers), so they are not summed
            reference_relations_by_source=dict(rel),
        )
    Path(out_json).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
