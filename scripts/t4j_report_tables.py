"""Recompute the T4j tables from evaluation evidence (no image reads).
Usage: python -I scripts/t4j_report_tables.py [OUT_MD]

T4f, T4h and T4i standard and the T4j variants per source on the common valid frames; wb_t4h (the cool-cast check
off) against T4h standard as a reproduction check; the frames where the cool-cast rule fired; key frames from
t4j/key_eval.json; largest changes vs T4h.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "experiments/t4_auto_ui"
VARIANTS = ("standard", "natural", "portrait", "vivid", "print_classic", "wb_t4h")
DOMAINS = ("nlp", "shop", "blueneg", "ext_own", "all")


def load(version):
    frames = {}
    for part in ("a", "b"):
        p = ROOT / version / f"eval_{version}_{part}.json"
        if p.exists():
            frames.update(json.loads(p.read_text())["frames"])
    return frames


def de(d, k, v):
    return d[k]["variants"][v]["metrics"]["de2000_median"]


def stats(rows):
    x = np.array([v["metrics"]["de2000_median"] for v in rows])
    return [float(np.median(x)), float(x.mean()), float(np.percentile(x, 90)),
            float(np.median([v["metrics"]["mean_dL"] for v in rows]))]


def bw(delta):
    return f"{int((delta < -0.25).sum())} / {int((delta > 0.25).sum())}"


def tables():
    f, h, i, j = load("t4f"), load("t4h"), load("t4i"), load("t4j")
    common = sorted(k for k in f.keys() & h.keys() & i.keys() & j.keys() if all(
        "error" not in d[k] and d[k].get("ncc", 0) >= 0.7 for d in (f, h, i, j)))
    lines = [f"Common valid frames: {len(common)}; evaluated T4j records: {len(j)}.", "",
             "ΔE p90 is across frames' median ΔE2000; dL is the median of frame mean_dL; better / worse: change > 0.25.",
             ""]
    for domain in DOMAINS:
        ids = [k for k in common if domain == "all" or j[k]["domain"] == domain]
        lines += [f"### {domain} ({len(ids)})", "",
                  "| Version / variant | ΔE median | mean | p90 | dL median | vs T4h standard | vs T4i standard |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        bh = np.array([de(h, k, "standard") for k in ids])
        bi = np.array([de(i, k, "standard") for k in ids])
        rows = [("T4f standard", f, "standard"), ("T4h standard", h, "standard"), ("T4i standard", i, "standard")] + \
               [("T4j " + v, j, v) for v in VARIANTS]
        for label, d, v in rows:
            r = [d[k]["variants"][v] for k in ids]
            x = np.array([y["metrics"]["de2000_median"] for y in r])
            lines.append("| " + label + " | " + " | ".join(f"{s:.2f}" for s in stats(r)) +
                         f" | {'—' if d is h else bw(x - bh)} | {'—' if d is i else bw(x - bi)} |")
        lines.append("")

    lines += ["### Checks (standard preset, all frames)", ""]
    pairs = [("wb_t4h vs T4h standard (cool-cast check off = T4h)", lambda k: de(j, k, "wb_t4h") - de(h, k, "standard")),
             ("standard vs wb_t4h (the cool-cast rule alone)", lambda k: de(j, k, "standard") - de(j, k, "wb_t4h")),
             ("standard vs T4h standard", lambda k: de(j, k, "standard") - de(h, k, "standard"))]
    for label, fn in pairs:
        d = np.array([fn(k) for k in common])
        lines.append(f"- {label}: better / worse {bw(d)}; changed (|change| > 1e-6) on {int((abs(d) > 1e-6).sum())}; "
                     f"median change {np.median(d):+.3f}; max |change| {abs(d).max():.3f}.")
    pj = {k: j[k]["variants"]["standard"]["params"] for k in common}
    ph = {k: h[k]["variants"]["standard"]["params"] for k in common}
    same = [k for k in common if pj[k] == ph[k]]
    lines.append(f"- Standard parameters identical to T4h on {len(same)}/{len(common)} frames.")
    lines.append("")

    cast = [k for k in common if j[k]["variants"]["standard"].get("wb_cool_cast")]
    lines += [f"### Cool-cast frames (standard preset): {len(cast)}", "",
              "| Frame | T4h ΔE | T4i ΔE | T4j ΔE | wb_t4h ΔE | T4h Temp | T4j Temp | T4h Tint | T4j Tint | T4j db |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in cast:
        x = j[k]["variants"]["standard"]
        lines.append(f"| {k} | {de(h, k, 'standard'):.2f} | {de(i, k, 'standard'):.2f} | {de(j, k, 'standard'):.2f} | "
                     f"{de(j, k, 'wb_t4h'):.2f} | {ph[k]['temp'] * 100:+.1f} | {x['params']['temp'] * 100:+.1f} | "
                     f"{ph[k]['tint'] * 100:+.1f} | {x['params']['tint'] * 100:+.1f} | {x['metrics']['mean_db']:+.2f} |")
    for domain in DOMAINS[:-1]:
        lines.append(f"- {domain}: {sum(j[k]['domain'] == domain for k in cast)} frames")
    lines.append("")

    lines += ["### Key frames", "", "| Frame | Version / variant | Base | EV | Temp | Tint | Contrast | Blacks | ΔE | dL | da | db |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    key = json.loads((ROOT / "t4j/key_eval.json").read_text())["frames"]
    for fid, frame in key.items():
        rows = [("T4h standard", h[fid], "standard"), ("T4i standard", i[fid], "standard")] + \
               [("T4j " + v, frame, v) for v in VARIANTS]
        for label, d, v in rows:
            x = d["variants"][v]
            p, m = x["params"], x["metrics"]
            vals = [p["ev"], p["temp"] * 100, p["tint"] * 100, p["contrast"], p.get("blacks", 0.0) * 100,
                    m["de2000_median"], m["mean_dL"], m["mean_da"], m["mean_db"]]
            lines.append(f"| {fid} | {label} | {d['base_method']} | " + " | ".join(f"{y:.2f}" for y in vals) + " |")
    lines += ["", "Temp, tint and blacks in UI units (x 100).", ""]

    lines += ["### Largest standard changes vs T4h", "", "| Frame | T4h ΔE | T4i ΔE | T4j ΔE | Change vs T4h | cool cast |",
              "|---|---:|---:|---:|---:|---|"]
    ch = sorted((de(j, k, "standard") - de(h, k, "standard"), k) for k in common)
    for item in list(reversed(ch[-10:])) + [None] + ch[:10]:
        if item is None:
            lines.append("| … | | | | | |")
            continue
        d, k = item
        lines.append(f"| {k} | {de(h, k, 'standard'):.2f} | {de(i, k, 'standard'):.2f} | {de(j, k, 'standard'):.2f} | "
                     f"{d:+.2f} | {'yes' if k in cast else ''} |")
    lines += ["", "### Alignment and base", ""]
    lines.append(f"Minimum T4j NCC {min(j[k]['ncc'] for k in common):.4f}; largest NCC change vs T4h "
                 f"{max(abs(j[k]['ncc'] - h[k]['ncc']) for k in common):.4f}; automatic base mode changed vs T4h on "
                 f"{sum(j[k]['base_method'] != h[k]['base_method'] for k in common)} frames.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    text = tables()
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text)
    else:
        print(text)
