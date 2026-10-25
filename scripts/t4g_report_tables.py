"""Recompute T4g tables from evaluation / roll evidence (no image reads).
Usage: python -I scripts/t4g_report_tables.py [OUT_MD]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "experiments/t4_auto_ui"


def load(version):
    frames = {}
    for part in ("a", "b"):
        p = ROOT / version / f"eval_{version}_{part}.json"
        if p.exists():
            frames.update(json.loads(p.read_text())["frames"])
    return frames


def stats(rows):
    de = np.array([v["metrics"]["de2000_median"] for v in rows])
    return [float(np.median(de)), float(de.mean()), float(np.percentile(de, 90)),
            float(np.median([v["metrics"]["mean_dL"] for v in rows]))]


def tables():
    old, new = load("t4f"), load("t4g")
    common = sorted(k for k in old.keys() & new.keys() if all(
        "error" not in d[k] and d[k].get("ncc", 0) >= 0.7 for d in (old, new)))
    lines = [f"Common valid frames: {len(common)}; evaluated T4g records: {len(new)}.", "",
             "ΔE p90 is across frames' median ΔE2000; dL is the median of frame mean_dL.", ""]
    variants = ("standard", "natural", "portrait", "vivid", "print_classic", "wb_t4f")
    for domain in ("nlp", "shop", "blueneg", "ext_own", "all"):
        ids = [k for k in common if domain == "all" or new[k]["domain"] == domain]
        if not ids:
            continue
        lines += [f"### {domain} ({len(ids)})", "", "| Version / preset | ΔE median | mean | p90 | dL median | Better / worse vs T4f standard |", "|---|---:|---:|---:|---:|---:|"]
        b = np.array([old[k]["variants"]["standard"]["metrics"]["de2000_median"] for k in ids])
        for label, data, variant in [("T4f standard", old, "standard")] + [("T4g " + v, new, v) for v in variants]:
            rows = [data[k]["variants"][variant] for k in ids]
            vals = stats(rows)
            delta = np.array([v["metrics"]["de2000_median"] for v in rows]) - b
            bw = "—" if data is old else f"{int((delta < -0.25).sum())} / {int((delta > 0.25).sum())}"
            lines.append("| " + label + " | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {bw} |")
        lines.append("")
    lines += ["### Key frames", "", "| Frame | Version / preset | Base | EV | Temp | Tint | Contrast | ΔE | mean dL | mean da | mean db |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    key = json.loads((ROOT / "t4g/key_eval.json").read_text())["frames"]
    for fid, frame in key.items():
        for label, data in [("T4f standard", old[fid])]+[("T4g " + v, frame) for v in variants]:
            vname = "standard" if label == "T4f standard" else label[4:]
            v = data["variants"][vname]
            p, m = v["params"], v["metrics"]
            vals = [p[k] for k in ("ev", "temp", "tint", "contrast")] + [m["de2000_median"], m["mean_dL"], m.get("mean_da", float("nan")), m.get("mean_db", float("nan"))]
            lines.append(f"| {fid} | {label} | {data['base_method']} | " + " | ".join(f"{x:.2f}" for x in vals) + " |")
    lines += ["", "### WB ablation (T4g standard vs fixed-radius T4f WB on the same T4g positive)", ""]
    if common:
        delta = np.array([new[k]["variants"]["standard"]["metrics"]["de2000_median"] - new[k]["variants"]["wb_t4f"]["metrics"]["de2000_median"] for k in common])
        lines += [f"Better / worse (>0.25 ΔE): {int((delta < -0.25).sum())} / {int((delta > 0.25).sum())}; median ΔE change {np.median(delta):+.3f}.", ""]
        for variant in variants:
            ev = [new[k]["variants"][variant]["params"]["ev"] for k in common]
            lines.append(f"{variant}: EV minimum {min(ev):+.4f}; below −0.30: {sum(v < -0.30 for v in ev)}.")
        lines += ["", "### Base-method changes", ""]
        for domain in ("nlp", "shop", "blueneg", "ext_own"):
            ids = [k for k in common if new[k]["domain"] == domain]
            count = sum(old[k]["base_method"] != new[k]["base_method"] for k in ids)
            lines.append(f"{domain}: {count}/{len(ids)} changed automatic base mode.")
        lines += ["", "### Alignment changes", ""]
        changes = sorted((abs(new[k]["ncc"] - old[k]["ncc"]), k, old[k]["ncc"], new[k]["ncc"]) for k in common)
        lines.append(f"Minimum T4g NCC {min(new[k]['ncc'] for k in common):.4f}; largest NCC change {changes[-1][0]:.4f} ({changes[-1][1]}).")
        lines += ["", "### Original T4e low-EV subset", ""]
        early = load("t4e")
        low = [k for k in common if early[k]["variants"]["standard"]["params"]["ev"] < -0.30]
        for name, data in (("T4f", old), ("T4g", new)):
            if low:
                vals = stats([data[k]["variants"]["standard"] for k in low])
                lines.append(f"{name}, {len(low)} frames: " + ", ".join(f"{v:.2f}" for v in vals) + " (median / mean / p90 / dL).")
        lines += ["", "### Largest standard regressions vs T4f", "", "| Frame | T4f ΔE | T4g ΔE | Change |", "|---|---:|---:|---:|"]
        changes = sorted((new[k]["variants"]["standard"]["metrics"]["de2000_median"] - old[k]["variants"]["standard"]["metrics"]["de2000_median"], k) for k in common)
        for delta, k in reversed(changes[-10:]):
            a = old[k]["variants"]["standard"]["metrics"]["de2000_median"]
            b = new[k]["variants"]["standard"]["metrics"]["de2000_median"]
            lines.append(f"| {k} | {a:.2f} | {b:.2f} | {delta:+.2f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    text = tables()
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text)
    else:
        print(text)
