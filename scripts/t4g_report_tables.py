"""Recompute T4g / T4h tables from evaluation / roll evidence (no image reads).
Usage: python -I scripts/t4g_report_tables.py [OUT_MD [TAG]]   (TAG t4g, default, or t4h; t4h adds T4g standard rows)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "experiments/t4_auto_ui"
TAG = sys.argv[2] if len(sys.argv) > 2 else "t4g"
NAME = "T4" + TAG[2:]


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
    old, new = load("t4f"), load(TAG)
    mid = load("t4g") if TAG != "t4g" else None
    common = sorted(k for k in old.keys() & new.keys() if all(
        "error" not in d[k] and d[k].get("ncc", 0) >= 0.7 for d in (old, new)))
    if mid is not None:
        common = [k for k in common if k in mid and "error" not in mid[k]]
    lines = [f"Common valid frames: {len(common)}; evaluated {NAME} records: {len(new)}.", "",
             "ΔE p90 is across frames' median ΔE2000; dL is the median of frame mean_dL.", ""]
    variants = ("standard", "natural", "portrait", "vivid", "print_classic", "wb_t4f") + (("wb_sky_red1",) if mid is not None else ())
    earlier = [("T4f standard", old, "standard")] + ([("T4g standard", mid, "standard")] if mid is not None else [])
    for domain in ("nlp", "shop", "blueneg", "ext_own", "all"):
        ids = [k for k in common if domain == "all" or new[k]["domain"] == domain]
        if not ids:
            continue
        lines += [f"### {domain} ({len(ids)})", "", "| Version / preset | ΔE median | mean | p90 | dL median | Better / worse vs T4f standard |", "|---|---:|---:|---:|---:|---:|"]
        b = np.array([old[k]["variants"]["standard"]["metrics"]["de2000_median"] for k in ids])
        for label, data, variant in earlier + [(f"{NAME} " + v, new, v) for v in variants]:
            rows = [data[k]["variants"][variant] for k in ids]
            vals = stats(rows)
            delta = np.array([v["metrics"]["de2000_median"] for v in rows]) - b
            bw = "—" if data is old else f"{int((delta < -0.25).sum())} / {int((delta > 0.25).sum())}"
            lines.append("| " + label + " | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {bw} |")
        lines.append("")
    lines += ["### Key frames", "", "| Frame | Version / preset | Base | EV | Temp | Tint | Contrast | ΔE | mean dL | mean da | mean db |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    key = json.loads((ROOT / TAG / "key_eval.json").read_text())["frames"]
    for fid, frame in key.items():
        rows = [(label, data[fid], vname) for label, data, vname in earlier] + [(f"{NAME} " + v, frame, v) for v in variants]
        for label, data, vname in rows:
            v = data["variants"][vname]
            p, m = v["params"], v["metrics"]
            vals = [p[k] for k in ("ev", "temp", "tint", "contrast")] + [m["de2000_median"], m["mean_dL"], m.get("mean_da", float("nan")), m.get("mean_db", float("nan"))]
            lines.append(f"| {fid} | {label} | {data['base_method']} | " + " | ".join(f"{x:.2f}" for x in vals) + " |")
    lines += ["", f"### WB ablation ({NAME} standard vs fixed-radius T4f WB on the same {NAME} positive)", ""]
    if common:
        delta = np.array([new[k]["variants"]["standard"]["metrics"]["de2000_median"] - new[k]["variants"]["wb_t4f"]["metrics"]["de2000_median"] for k in common])
        lines += [f"Better / worse (>0.25 ΔE): {int((delta < -0.25).sum())} / {int((delta > 0.25).sum())}; median ΔE change {np.median(delta):+.3f}.", ""]
        for variant in variants:
            ev = [new[k]["variants"][variant]["params"]["ev"] for k in common]
            if variant == "wb_sky_red1":
                d1 = np.array([new[k]["variants"]["standard"]["metrics"]["de2000_median"] - new[k]["variants"]["wb_sky_red1"]["metrics"]["de2000_median"] for k in common])
                lines.append(f"(standard red factor {0.75} vs wb_sky_red1 1.0: better / worse {int((d1 < -0.25).sum())} / {int((d1 > 0.25).sum())}; median ΔE change {np.median(d1):+.3f}.)")
            lines.append(f"{variant}: EV minimum {min(ev):+.4f}; below −0.30: {sum(v < -0.30 for v in ev)}.")
        lines += ["", "### Base-method changes", ""]
        for domain in ("nlp", "shop", "blueneg", "ext_own"):
            ids = [k for k in common if new[k]["domain"] == domain]
            count = sum(old[k]["base_method"] != new[k]["base_method"] for k in ids)
            extra = f"; vs T4g {sum(mid[k]['base_method'] != new[k]['base_method'] for k in ids)}/{len(ids)}" if mid is not None else ""
            lines.append(f"{domain}: {count}/{len(ids)} changed automatic base mode vs T4f{extra}.")
        lines += ["", "### Alignment changes", ""]
        changes = sorted((abs(new[k]["ncc"] - old[k]["ncc"]), k, old[k]["ncc"], new[k]["ncc"]) for k in common)
        lines.append(f"Minimum {NAME} NCC {min(new[k]['ncc'] for k in common):.4f}; largest NCC change {changes[-1][0]:.4f} ({changes[-1][1]}).")
        lines += ["", "### Original T4e low-EV subset", ""]
        early = load("t4e")
        low = [k for k in common if early[k]["variants"]["standard"]["params"]["ev"] < -0.30]
        for name, data in [("T4f", old)] + ([("T4g", mid)] if mid is not None else []) + [(NAME, new)]:
            if low:
                vals = stats([data[k]["variants"]["standard"] for k in low])
                lines.append(f"{name}, {len(low)} frames: " + ", ".join(f"{v:.2f}" for v in vals) + " (median / mean / p90 / dL).")
        lines += ["", "### Largest standard regressions vs T4f", "", f"| Frame | T4f ΔE | {NAME} ΔE | Change |", "|---|---:|---:|---:|"]
        changes = sorted((new[k]["variants"]["standard"]["metrics"]["de2000_median"] - old[k]["variants"]["standard"]["metrics"]["de2000_median"], k) for k in common)
        for delta, k in reversed(changes[-10:]):
            a = old[k]["variants"]["standard"]["metrics"]["de2000_median"]
            b = new[k]["variants"]["standard"]["metrics"]["de2000_median"]
            lines.append(f"| {k} | {a:.2f} | {b:.2f} | {delta:+.2f} |")
    probes = sorted((ROOT / TAG / "probe").glob("*_uncapped.json"))
    if TAG != "t4g" and probes:
        # Frames whose roll_nat black point is held by max_black_pull_d (from the roll diagnostics of TAG/after),
        # and t4g_base_probe.py uncapped (same roll, cap lifted) vs the evaluated standard preset.
        lines += ["", "### Black-pull cap (0.04 D) ablation", ""]
        checks = json.loads((ROOT / TAG / "after/checks.json").read_text())
        held = []
        for roll in checks["rolls"].values():
            for rel, fe in roll["info"]["frames"].items():
                ni = fe.get("nat_info") or {}
                if not ni or ni["d_lo_min"] < ni["d_lo_b1"]:
                    continue
                w = ni["b1_weight"]
                pull = w * ni["d_lo_b1"] + (1 - w) * (ni["d_lo_min"] - ni["paper_black_d"]) - ni["d_lo_b1"]
                if pull > ni["max_black_pull_d"] + 1e-9:
                    held.append((rel.split("_")[0], pull))
        lines.append(f"Frames held by the cap: {len(held)} ({', '.join(f'{n} {p:.3f} D' for n, p in held)}).")
        lines += ["", f"| Frame | Automatic base | T4f ΔE | {NAME} uncapped ΔE | {NAME} ΔE (cap) | Cap effect |", "|---|---|---:|---:|---:|---:|"]
        effect = []
        for p in probes:
            fid = "ext:" + p.name.split("_")[0]
            u = json.loads(p.read_text())["metrics"]["de2000_median"]
            c = new[fid]["variants"]["standard"]["metrics"]["de2000_median"]
            effect.append(c - u)
            lines.append(f"| {fid} | {new[fid]['base_method']} | {old[fid]['variants']['standard']['metrics']['de2000_median']:.2f} | {u:.2f} | {c:.2f} | {c - u:+.2f} |")
        effect = np.array(effect)
        lines += ["", f"Cap better / worse (>0.25 ΔE): {int((effect < -0.25).sum())} / {int((effect > 0.25).sum())}; median change {np.median(effect):+.3f}."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    text = tables()
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text)
    else:
        print(text)
