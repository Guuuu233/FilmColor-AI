"""Recompute the T4k tables from evaluation evidence (no image reads).
Usage: python -I scripts/t4k_report_tables.py DIR [OUT_MD]   (DIR: t4k_tasklist or t4k, below experiments/t4_auto_ui)

T4h, T4j and the T4k run in DIR (eval_t4k_{a,b}.json, variants as recorded there) per source on the common valid
frames; each estimator change alone (standard vs the variant with that change off, or on); the automatic base routing
and the blue white point guard, from the records and the subset runs with them off (DIR/routing_off.json,
DIR/guard_off.json); the frames each mechanism touched; key frames (DIR/key_eval.json) next to the research side's
expected values; the NLP frames opened from nlp_official in three base modes (DIR/roll_modes); largest changes vs T4j.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "experiments/t4_auto_ui"
DIR = sys.argv[1] if len(sys.argv) > 1 else "t4k"
# variant -> what it changes against standard ("off": the change is on in standard; "on": off in standard)
ABLATIONS = (("no_border", "off", "black rebate / gap filter (cells and tone pixels)"),
             ("no_sky_check", "off", "backlit bright-share condition"),
             ("no_thin", "off", "thin-negative ev cap and blacks"),
             ("wb_t4j", "off", "white balance changes (cooling damp, tint boost, base tint, grey-world undamping, cast cap)"),
             ("t4j_noref", "off", "all estimator changes"),
             ("border_on", "on", "black rebate / gap filter (cells and tone pixels)"),
             ("sky_check_on", "on", "backlit bright-share condition"),
             ("wb_t4k_on", "on", "white balance changes (cooling damp, tint boost, base tint, grey-world undamping, cast cap)"),
             ("tasklist_est", "on", "all T4k task-list estimator changes"))
# expected standard dE2000 from the T4k task list and its notes (research side; not measured here)
CLAIMED = {"nlp:_DSF8983": 7.42, "nlp:_DSF8951": 7.98, "nlp:_DSF9239": 7.78, "nlp:_DSF9238": 5.89,
           "ext:A224": 6.53, "blueneg:19960812D-09-las-vegas": 7.39, "shop:DSC_3827": 8.77}
CLAIMED_ROLL = {"nlp:_DSF8983": 18.03, "nlp:_DSF8951": 14.79, "nlp:_DSF9239": 22.93, "nlp:_DSF9238": 13.67}
DOMAINS = ("nlp", "shop", "blueneg", "ext_own", "all")
PRESETS = ("standard", "natural", "portrait", "vivid", "print_classic")


def ref_estimator(variants):
    """The variant that runs the T4j estimator on this run's positive (t4j_noref in the task-list run, no_thin in
    the adopted one)."""
    return "t4j_noref" if "t4j_noref" in variants else "no_thin"


def load(version, name=None):
    frames = {}
    for part in ("a", "b"):
        p = ROOT / version / f"eval_{name or version}_{part}.json"
        if p.exists():
            frames.update(json.loads(p.read_text())["frames"])
    return frames


def load_one(path):
    return json.loads(path.read_text())["frames"] if path.exists() else {}


def de(d, k, v="standard"):
    return d[k]["variants"][v]["metrics"]["de2000_median"]


def stats(rows):
    x = np.array([v["metrics"]["de2000_median"] for v in rows])
    return [float(np.median(x)), float(x.mean()), float(np.percentile(x, 90)),
            float(np.median([v["metrics"]["mean_dL"] for v in rows]))]


def bw(delta):
    return f"{int((delta < -0.25).sum())} / {int((delta > 0.25).sum())}"


def tables():
    h, j, k = load("t4h"), load("t4j"), load(DIR, "t4k")
    variants = json.loads((ROOT / DIR / "eval_t4k_a.json").read_text())["variants"]
    extra = [v for v in variants if v not in PRESETS]  # the ablation variants of this run
    est_ref = ref_estimator(variants)
    common = sorted(x for x in h.keys() & j.keys() & k.keys() if all(
        "error" not in d[x] and d[x].get("ncc", 0) >= 0.7 for d in (h, j, k)))
    lines = [f"Common valid frames: {len(common)}; evaluated T4k records: {len(k)}.", "",
             "ΔE p90 is across frames' median ΔE2000; dL is the median of frame mean_dL; better / worse: change > 0.25.",
             ""]
    for domain in DOMAINS:
        ids = [x for x in common if domain == "all" or k[x]["domain"] == domain]
        lines += [f"### {domain} ({len(ids)})", "",
                  "| Version / variant | ΔE median | mean | p90 | dL median | vs T4j standard |",
                  "|---|---:|---:|---:|---:|---:|"]
        bj = np.array([de(j, x) for x in ids])
        rows = [("T4h standard", h, "standard"), ("T4j standard", j, "standard")] + [("T4k " + v, k, v) for v in variants]
        for label, d, v in rows:
            r = [d[x]["variants"][v] for x in ids]
            xs = np.array([y["metrics"]["de2000_median"] for y in r])
            lines.append("| " + label + " | " + " | ".join(f"{s:.2f}" for s in stats(r)) +
                         f" | {'—' if d is j else bw(xs - bj)} |")
        lines.append("")

    lines += ["### Each estimator change alone (change on minus change off, all frames; negative = the change helps)", ""]
    for v, sense, label in ABLATIONS:
        if v not in variants:
            continue
        sign = 1.0 if sense == "off" else -1.0

        def eff(x):
            return sign * (de(k, x) - de(k, x, v))
        d = np.array([eff(x) for x in common])
        lines.append(f"- {label} (`{v}`): better / worse {bw(d)}; changed on {int((abs(d) > 1e-6).sum())}; median "
                     f"{np.median(d):+.3f}; largest gain {d.min():+.2f} ({common[int(d.argmin())]}), largest loss "
                     f"{d.max():+.2f} ({common[int(d.argmax())]}).")
        for domain in DOMAINS[:-1]:
            ids = [x for x in common if k[x]["domain"] == domain]
            dd = np.array([eff(x) for x in ids])
            lines.append(f"  - {domain}: {bw(dd)}, median {np.median(dd):+.3f}")
    ref_v = est_ref
    if ref_v in variants:
        d = np.array([de(k, x, ref_v) - de(j, x) for x in common])
        lines.append(f"- Check: T4k `{ref_v}` vs T4j standard (the T4j estimator; differs only where the base routing or "
                     f"the blue guard changed the positive): changed on {int((abs(d) > 1e-6).sum())}, better / worse {bw(d)}, "
                     f"max |change| {abs(d).max():.3f}.")
    lines.append("")

    routed = [x for x in common if k[x]["base_method"] != j[x]["base_method"]]
    guarded = [x for x in common if ((k[x].get("white_point") or {}).get("blue_guard") or {}).get("applied")]
    lines += [f"### Automatic base routing changed vs T4j: {len(routed)} frames", "",
              f"| Frame | T4j → T4k base | dC | level diff | T4j ΔE | T4k ΔE | T4k {est_ref} | routing off (T4k) |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    r_off = load_one(ROOT / DIR / "routing_off.json")
    for x in routed:
        b = j[x]["base"]
        off = f"{de(r_off, x):.2f}" if x in r_off and "error" not in r_off[x] else "—"
        lines.append(f"| {x} | {j[x]['base_method']} → {k[x]['base_method']} | {b.get('chroma_part') or 0:.3f} | "
                     f"{b.get('level_diff') or 0:.3f} | {de(j, x):.2f} | {de(k, x):.2f} | {de(k, x, est_ref):.2f} | {off} |")
    if routed:
        dd = np.array([de(k, x) - de(j, x) for x in routed])
        lines.append(f"\nRouted frames, T4k standard vs T4j: better / worse {bw(dd)}, median {np.median(dd):+.2f}.")
        if r_off:
            ok = [x for x in routed if x in r_off and "error" not in r_off[x]]
            dr = np.array([de(k, x) - de(r_off, x) for x in ok])
            lines.append(f"Routing alone (T4k standard vs the same with ROLL_NAT_LEVEL_DIFF None, {len(ok)} frames): "
                         f"better / worse {bw(dr)}, median {np.median(dr):+.2f}.")
    lines += ["", f"### Blue white point guard applied: {len(guarded)} frames", "",
              f"| Frame | d_hi B/G own | used | T4j ΔE | T4k ΔE | T4k {est_ref} | guard off (T4k) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    g_off = load_one(ROOT / DIR / "guard_off.json")
    for x in guarded:
        g = k[x]["white_point"]["blue_guard"]
        off = f"{de(g_off, x):.2f}" if x in g_off and "error" not in g_off[x] else "—"
        lines.append(f"| {x} | {g['ratio_bg']:.3f} | {g['ratio_bg_used']:.3f} | {de(j, x):.2f} | {de(k, x):.2f} | "
                     f"{de(k, x, est_ref):.2f} | {off} |")
    if guarded and g_off:
        ok = [x for x in guarded if x in g_off and "error" not in g_off[x]]
        dg = np.array([de(k, x) - de(g_off, x) for x in ok])
        lines.append(f"\nGuard alone (T4k standard vs the same with WHITE_B_GUARD None, {len(ok)} frames): better / worse "
                     f"{bw(dg)}, median {np.median(dg):+.2f}.")
    # the frame's own white point: before the blue guard where it ran, else the recorded d_hi
    ratios = np.array([((k[x].get("white_point") or {}).get("blue_guard") or {}).get("ratio_bg")
                       or k[x]["d_hi"][2] / k[x]["d_hi"][1] for x in common])
    lines.append(f"\nd_hi B/G over the frames (own white point): median {np.nanmedian(ratios):.3f}, below 1.12 on "
                 f"{int((ratios < 1.12).sum())} frames.")

    thin = [x for x in common if ((k[x]["variants"]["standard"].get("t4k") or {}).get("thin_negative") or {}).get("applied")]
    lines += ["", f"### Thin-negative check applied (standard): {len(thin)} frames", "",
              "| Frame | span_d_g | bright share | ev cap | T4j ev | T4k ev | T4j ΔE | T4k ΔE | no_thin | T4k dL |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in thin:
        s = k[x]["variants"]["standard"]
        t = s["t4k"]["thin_negative"]
        lines.append(f"| {x} | {t['span_d_g']:.3f} | {t['bright_share']:.3f} | {t['ev_cap']:.2f} | "
                     f"{j[x]['variants']['standard']['params']['ev']:+.2f} | {s['params']['ev']:+.2f} | {de(j, x):.2f} | "
                     f"{de(k, x):.2f} | {de(k, x, 'no_thin'):.2f} | {s['metrics']['mean_dL']:+.1f} |")
    spans = [((k[x]["variants"]["standard"].get("t4k") or {}).get("thin_negative") or {}).get("span_d_g") for x in common]
    spans = np.array([s for s in spans if s is not None])
    lines.append(f"\nspan_d_g over the frames: median {np.median(spans):.3f}; below 0.62 on {int((spans < 0.62).sum())}, "
                 f"below 0.52 on {int((spans < 0.52).sum())}.")
    bl_j = np.array([j[x]["variants"]["standard"]["backlit"]["weight"] for x in common])
    bl_k = np.array([k[x]["variants"]["standard"]["backlit"]["weight"] for x in common])
    lines.append(f"Backlit weight > 0: T4j {int((bl_j > 0).sum())} frames, T4k {int((bl_k > 0).sum())}; lowered on "
                 f"{int((bl_k < bl_j - 1e-6).sum())}, raised on {int((bl_k > bl_j + 1e-6).sum())}.")
    cast = [x for x in common if k[x]["variants"]["standard"].get("wb_cool_cast")]
    capped = [x for x in cast if abs(k[x]["variants"]["standard"]["params"]["temp"] - 0.38) < 1e-3]
    lines.append(f"Cool-cast frames: {len(cast)}; temp at the 0.38 cap on {len(capped)}.")
    tj = np.array([j[x]["variants"]["standard"]["metrics"]["mean_da"] for x in common])
    tk = np.array([k[x]["variants"]["standard"]["metrics"]["mean_da"] for x in common])
    lines.append(f"mean_da median: T4j {np.median(tj):+.2f} (green < 0 on {int((tj < 0).sum())}), T4k {np.median(tk):+.2f} "
                 f"(green < 0 on {int((tk < 0).sum())}).")

    lines += ["", "### Key frames", "",
              "| Frame | T4j base | T4k base | T4j ΔE | T4k ΔE | claimed | " + " | ".join(extra) + " | T4k dL | T4k EV | T4k temp |",
              "|---|---|---|---:|---:|---:|" + "---:|" * (len(extra) + 3)]
    key = load_one(ROOT / DIR / "key_eval.json")
    for x, fr in key.items():
        s = fr["variants"]["standard"]
        cl = f"{CLAIMED[x]:.2f}" if x in CLAIMED else "—"
        lines.append(f"| {x} | {j[x]['base_method']} | {fr['base_method']} | {de(j, x):.2f} | {de(key, x):.2f} | {cl} | " +
                     " | ".join(f"{de(key, x, v):.2f}" for v in extra)
                     + f" | {s['metrics']['mean_dL']:+.1f} | {s['params']['ev']:+.2f} | {s['params']['temp'] * 100:+.1f} |")

    lines += ["", "### NLP frames opened from nlp_official (a collection) in three base modes (standard preset)", "",
              "| Frame | T4j ① auto | T4j ③ roll_nat | T4j ⑤ roll_b1 | T4k ① auto | T4k ③ roll_nat | T4k ⑤ roll_b1 | "
              "claimed ③/⑤ before | T4j ③ da | T4k ③ da |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    rm = {(t, m): load_one(ROOT / DIR / f"roll_modes/{t}_{m}.json") for t in ("t4j", "t4k") for m in ("auto", "roll_nat", "roll_b1")}
    for x in sorted(rm[("t4k", "auto")]):
        def cell(t, m):
            d = rm[(t, m)]
            return f"{de(d, x):.2f}" if x in d and "error" not in d[x] else "—"

        def da(t):
            d = rm[(t, "roll_nat")]
            return f"{d[x]['variants']['standard']['metrics']['mean_da']:+.1f}" if x in d and "error" not in d[x] else "—"
        lines.append(f"| {x} | " + " | ".join(cell(t, m) for t in ("t4j", "t4k") for m in ("auto", "roll_nat", "roll_b1")) +
                     f" | {CLAIMED_ROLL.get(x, '—')} | {da('t4j')} | {da('t4k')} |")

    lines += ["", "### Largest standard changes vs T4j", "",
              "| Frame | T4j ΔE | T4k ΔE | Change | T4k base | " + " | ".join(extra) + " |",
              "|---|---:|---:|---:|---|" + "---:|" * len(extra)]
    ch = sorted((de(k, x) - de(j, x), x) for x in common)
    for item in list(reversed(ch[-12:])) + [None] + ch[:12]:
        if item is None:
            lines.append("| … |" + " |" * (4 + len(extra)))
            continue
        dlt, x = item
        lines.append(f"| {x} | {de(j, x):.2f} | {de(k, x):.2f} | {dlt:+.2f} | {k[x]['base_method']} | " +
                     " | ".join(f"{de(k, x, v):.2f}" for v in extra) + " |")
    lines += ["", "### Alignment", ""]
    lines.append(f"Minimum T4k NCC {min(k[x]['ncc'] for x in common):.4f}; largest NCC change vs T4j "
                 f"{max(abs(k[x]['ncc'] - j[x]['ncc']) for x in common):.4f}.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    text = tables()
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text)
    else:
        print(text)
