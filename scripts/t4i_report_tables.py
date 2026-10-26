"""Recompute the T4i tables from evaluation evidence (no image reads).
Usage: python -I scripts/t4i_report_tables.py [OUT_MD]

T4f standard, T4h standard and the T4i variants per source on the common valid frames; the T4i ablations against
t4h_equiv (both T4i changes off, the T4h rule); key frames from t4i/key_eval.json; largest changes vs T4h.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "experiments/t4_auto_ui"
VARIANTS = ("standard", "natural", "portrait", "vivid", "print_classic", "wb_t4h", "no_auto_levels", "no_auto_slope",
            "t4h_equiv", "levels_display")


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
    f, h, i = load("t4f"), load("t4h"), load("t4i")
    common = sorted(k for k in f.keys() & h.keys() & i.keys() if all(
        "error" not in d[k] and d[k].get("ncc", 0) >= 0.7 for d in (f, h, i)))
    lines = [f"Common valid frames: {len(common)}; evaluated T4i records: {len(i)}.", "",
             "ΔE p90 is across frames' median ΔE2000; dL is the median of frame mean_dL; better / worse: change > 0.25.",
             ""]
    for domain in ("nlp", "shop", "blueneg", "ext_own", "all"):
        ids = [k for k in common if domain == "all" or i[k]["domain"] == domain]
        lines += [f"### {domain} ({len(ids)})", "",
                  "| Version / variant | ΔE median | mean | p90 | dL median | vs T4f standard | vs T4h standard |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        bf = np.array([de(f, k, "standard") for k in ids])
        bh = np.array([de(h, k, "standard") for k in ids])
        rows = [("T4f standard", f, "standard"), ("T4h standard", h, "standard")] + [("T4i " + v, i, v) for v in VARIANTS]
        for label, d, v in rows:
            r = [d[k]["variants"][v] for k in ids]
            x = np.array([y["metrics"]["de2000_median"] for y in r])
            lines.append("| " + label + " | " + " | ".join(f"{s:.2f}" for s in stats(r)) +
                         f" | {'—' if d is f else bw(x - bf)} | {'—' if d is h else bw(x - bh)} |")
        lines.append("")

    lines += ["### Ablations (standard preset, all frames)", ""]
    pairs = [("t4h_equiv vs T4h standard (reproduction of T4h)", lambda k: de(i, k, "t4h_equiv") - de(h, k, "standard")),
             ("standard vs t4h_equiv (both T4i changes)", lambda k: de(i, k, "standard") - de(i, k, "t4h_equiv")),
             ("standard vs wb_t4h (T4i white balance alone, auto levels on)", lambda k: de(i, k, "standard") - de(i, k, "wb_t4h")),
             ("standard vs no_auto_levels (auto levels alone, T4i white balance)", lambda k: de(i, k, "standard") - de(i, k, "no_auto_levels")),
             ("standard vs no_auto_slope (auto slopes alone)", lambda k: de(i, k, "standard") - de(i, k, "no_auto_slope")),
             ("levels_display vs t4h_equiv (research side's per-channel display levels on T4h)",
              lambda k: de(i, k, "levels_display") - de(i, k, "t4h_equiv")),
             ("standard vs levels_display", lambda k: de(i, k, "standard") - de(i, k, "levels_display"))]
    for label, fn in pairs:
        lines.append(f"- {label}: better / worse {bw(np.array([fn(k) for k in common]))}; median change "
                     f"{np.median([fn(k) for k in common]):+.3f}; max |change| {max(abs(fn(k)) for k in common):.3f}.")
        for domain in ("nlp", "shop", "blueneg", "ext_own"):
            ids = [k for k in common if i[k]["domain"] == domain]
            dd = np.array([fn(k) for k in ids])
            lines.append(f"  - {domain}: {bw(dd)}, median {np.median(dd):+.3f}")
    lines.append("")

    lines += ["### Automatic parameters (standard)", ""]
    for domain in ("nlp", "shop", "blueneg", "ext_own", "all"):
        ids = [k for k in common if domain == "all" or i[k]["domain"] == domain]
        p = [i[k]["variants"]["standard"]["params"] for k in ids]
        bl = np.array([x.get("blacks", 0.0) for x in p])
        c = np.array([x["contrast"] for x in p])
        ch = np.array([h[k]["variants"]["standard"]["params"]["contrast"] for k in ids])
        sr = np.array([x.get("slope_r", 0.0) for x in p])
        sb = np.array([x.get("slope_b", 0.0) for x in p])
        t = np.array([x["temp"] for x in p]) - np.array([h[k]["variants"]["standard"]["params"]["temp"] for k in ids])
        lines.append(f"- {domain}: blacks < 0 on {int((bl < 0).sum())}/{len(ids)} (median {np.median(bl):+.3f}, at the "
                     f"-0.28 cap {int((bl <= -0.28 + 1e-6).sum())}); contrast raised on {int((c > ch + 1e-3).sum())} "
                     f"(median {np.median(c):.2f}); slope_r set on {int((sr != 0).sum())}, slope_b on {int((sb != 0).sum())}; "
                     f"temp change vs T4h median {np.median(t) * 100:+.1f}, |change| > 0.05 on {int((abs(t) > 0.05).sum())}.")
    lines.append("")

    lines += ["### Key frames", "", "| Frame | Version / variant | Base | EV | Temp | Tint | Contrast | Blacks | ΔE | dL | da | db |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    key = json.loads((ROOT / "t4i/key_eval.json").read_text())["frames"]
    for fid, frame in key.items():
        rows = [("T4f standard", f[fid], "standard"), ("T4h standard", h[fid], "standard")] + \
               [("T4i " + v, frame, v) for v in VARIANTS]
        for label, d, v in rows:
            x = d["variants"][v]
            p, m = x["params"], x["metrics"]
            vals = [p["ev"], p["temp"] * 100, p["tint"] * 100, p["contrast"], p.get("blacks", 0.0) * 100,
                    m["de2000_median"], m["mean_dL"], m["mean_da"], m["mean_db"]]
            lines.append(f"| {fid} | {label} | {d['base_method']} | " + " | ".join(f"{y:.2f}" for y in vals) + " |")
    lines += ["", "Temp, tint and blacks in UI units (x 100).", ""]

    lines += ["### Largest standard changes vs T4h", "", "| Frame | T4h ΔE | T4i ΔE | Change | t4h_equiv | levels_display |",
              "|---|---:|---:|---:|---:|---:|"]
    ch = sorted((de(i, k, "standard") - de(h, k, "standard"), k) for k in common)
    for item in list(reversed(ch[-10:])) + [None] + ch[:10]:
        if item is None:
            lines.append("| … | | | | | |")
            continue
        d, k = item
        lines.append(f"| {k} | {de(h, k, 'standard'):.2f} | {de(i, k, 'standard'):.2f} | {d:+.2f} | "
                     f"{de(i, k, 't4h_equiv'):.2f} | {de(i, k, 'levels_display'):.2f} |")
    lines += ["", "### Alignment and base", ""]
    lines.append(f"Minimum T4i NCC {min(i[k]['ncc'] for k in common):.4f}; largest NCC change vs T4h "
                 f"{max(abs(i[k]['ncc'] - h[k]['ncc']) for k in common):.4f}; automatic base mode changed vs T4h on "
                 f"{sum(i[k]['base_method'] != h[k]['base_method'] for k in common)} frames.")
    hs = {k: i[k].get("holder_share") for k in common}
    lines.append("Film holder found (preview share > 0.005) on " +
                 ", ".join(f"{k} {v:.3f}" for k, v in hs.items() if v and v > 0.005) +
                 f"; share in (0, 0.005] on {sum(1 for v in hs.values() if v and v <= 0.005)} frames (edge rows of the scan).")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    text = tables()
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text)
    else:
        print(text)
