"""T5a §3 deliverables from rolls_{baseline,candidate}.json: default-routing before / after mosaics of the foma
pan100 and Eastman 5294 (no suffix) rolls, and the list of frames moved from roll_nat to auto. No original reads.
Usage: python -I scripts/t5a_roll_mosaics.py
Mosaics (local JPG, not committed): roll_<name>_before_after.jpg; list: roll_routing_changes.json.
"""
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t5a"
THUMBS = OUT / "roll_thumbs"
OPENED = ("foma_pan100", "eastman_5294")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    runs = {label: json.loads((OUT / f"rolls_{label}.json").read_text()) for label in ("baseline", "candidate")}
    freeze = json.loads((OUT / "source_freeze.json").read_text())
    base_src = json.loads((OUT / "baseline_source.json").read_text())["src_content_sha256"]
    assert runs["candidate"]["src_content_sha256"] == freeze["src_content_sha256"]
    assert runs["baseline"]["src_content_sha256"] == base_src
    assert runs["baseline"]["script_sha256"] == runs["candidate"]["script_sha256"] == sha(ROOT / "scripts/t5a_roll_routing.py")
    out = dict(source_sha256=freeze["src_content_sha256"], baseline_source_sha256=base_src,
               rolls_baseline_sha256=sha(OUT / "rolls_baseline.json"),
               rolls_candidate_sha256=sha(OUT / "rolls_candidate.json"), rolls={}, mosaics={})
    for name in runs["candidate"]["rolls"]:
        rb, rc = runs["baseline"]["rolls"][name], runs["candidate"]["rolls"][name]
        assert rb["complete"] and rc["complete"] and set(rb["frames"]) == set(rc["frames"])
        frames = sorted(rc["frames"])
        summary = dict(folder=rc["folder"], n=rc["n"], roll_warnings=rc["roll_warnings"], n_base=rc["n_base"],
                       roll_gt_limit=[f for f in frames if rc["frames"][f]["roll_gt_limit"]],
                       routed_before=[f for f in frames if rb["frames"][f]["route_roll_nat"]],
                       routed_after=[f for f in frames if rc["frames"][f]["route_roll_nat"]])
        for f in frames:
            b, c = rb["frames"][f], rc["frames"][f]
            assert b["raw_sha256"] == c["raw_sha256"]
            for k in ("own_rgb", "roll_rgb", "level_diff", "chroma_part", "own_error", "exposure"):
                assert b[k] == c[k], (name, f, k)  # roll analysis unchanged; only the routing decision may differ
            assert c["roll_implausible"] == c["roll_gt_limit"] and b["roll_implausible"] is None
        summary["moved_to_auto"] = sorted(set(summary["routed_before"]) - set(summary["routed_after"]))
        assert set(summary["routed_after"]) <= set(summary["routed_before"])
        assert set(summary["moved_to_auto"]) <= set(summary["roll_gt_limit"])
        if rc["opened"]:
            per = {}
            for f in frames:
                b, c = rb["frames"][f], rc["frames"][f]
                assert "error" not in b and "error" not in c, (name, f)
                for label, r in (("baseline", b), ("candidate", c)):
                    assert sha(THUMBS / r["thumb"]) == r["thumb_sha256"], (label, f)
                assert b["base_method"] == ("roll_nat" if f in summary["routed_before"] else "auto")
                assert c["base_method"] == ("roll_nat" if f in summary["routed_after"] else "auto")
                if b["base_method"] == c["base_method"]:  # same base: same estimate (the display may differ only
                    assert b["params"] == c["params"] and b["inversion"] == c["inversion"], (name, f)  # by holder paint)
                per[f] = dict(base_before=b["base_method"], base_after=c["base_method"],
                              status_before=b["status"], status_after=c["status"],
                              flags_before=b["flags"], flags_after=c["flags"],
                              d_hi_before=b["inversion"]["d_hi"], d_hi_after=c["inversion"]["d_hi"],
                              ev_before=b["params"].get("ev"), ev_after=c["params"].get("ev"),
                              roll_max=c["roll_max"], level_diff=c["level_diff"],
                              input_warnings_after=[w["code"] for w in c["input_warnings"] or []],
                              display_changed=b["display_sha256"] != c["display_sha256"])
            summary["frames"] = per
            summary["status_counts"] = {label: {s: sum(1 for p in per.values() if p["status_" + key] == s)
                                                for s in sorted({p["status_" + key] for p in per.values()})}
                                        for label, key in (("before", "before"), ("after", "after"))}
            cols, cw, ch, head = 8, 250, 290, 50
            rows = (len(frames) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * cw * 2 + 20, head + rows * ch), (20, 20, 20))
            d = ImageDraw.Draw(sheet)
            d.text((10, 10), f"T5a {name}: default routing, left block T4z ({base_src[:12]}), right block T5a "
                              f"({freeze['src_content_sha256'][:12]}); film=bw; red caption = route changed", fill=(240, 240, 240))
            for i, f in enumerate(frames):
                r, cidx = divmod(i, cols)
                for j, (label, rec) in enumerate((("baseline", rb["frames"][f]), ("candidate", rc["frames"][f]))):
                    x = j * (cols * cw + 20) + cidx * cw
                    y = head + r * ch
                    with Image.open(THUMBS / rec["thumb"]) as im:
                        img = im.convert("RGB")
                    sheet.paste(img, (x + (cw - img.width) // 2, y + (240 - img.height) // 2))
                    moved = f in summary["moved_to_auto"]
                    d.text((x + 6, y + 244), f"{Path(f).stem} {rec['base_method']} {rec['status'][:6]}",
                           fill=(255, 90, 90) if moved else (210, 210, 210))
                    d.text((x + 6, y + 262), f"roll {rec['roll_max']:.2f} ld {rec['level_diff']:.2f}",
                           fill=(255, 90, 90) if moved else (160, 160, 160))
            p = OUT / f"roll_{name}_before_after.jpg"
            sheet.save(p, quality=90)
            out["mosaics"][name] = dict(file=p.name, sha256=sha(p))
        out["rolls"][name] = summary
        print(name, "routed", len(summary["routed_before"]), "->", len(summary["routed_after"]),
              "moved", len(summary["moved_to_auto"]), "roll>1.25", len(summary["roll_gt_limit"]))
    (OUT / "roll_routing_changes.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
