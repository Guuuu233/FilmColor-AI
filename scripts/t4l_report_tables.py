"""Recompute the T4l tables from evaluation evidence (no image reads).
Usage: python -I scripts/t4l_report_tables.py [OUT_MD]

185 dev frames: T4j, T4k and the T4l variants per source; each estimator change alone (T4l standard vs the variant
with it off); the inversion changes alone (t4k_est, the T4k estimator on the T4l positive, vs T4k standard), split into
the self-ratio frames and the rest (spike clamp only); the frames each mechanism touched; protected and key frames.
BlueNeg check set
(t4l/cross_t4l.json, t4l/cross_inv_off.json: 124 roll-base frames outside the dev manifest). Multi-device samples
(t4l/devices_t4k.json, t4l/devices_t4l.json). Largest changes vs T4k.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "experiments/t4_auto_ui"
DOMAINS = ("nlp", "shop", "blueneg", "ext_own", "all")
PRESETS = ("standard", "natural", "portrait", "vivid", "print_classic")
EST_OFF = (("no_meter", "metering frame (holder box and black strips)"), ("no_holder_box", "holder box alone"),
           ("no_strips", "black strips alone"), ("no_ev_down", "adaptive ev lowering"),
           ("t4k_est", "all estimator changes (metering, lowering, low-information check)"))
PROTECTED = ("ext:A008", "ext:A012", "ext:A013", "shop:DSC_3833", "nlp:_DSF8948")
NAMED = ("shop:DSC_3827", "ext:A006", "ext:A196", "ext:A222", "nlp:_DSF9239", "nlp:_DSF8983", "ext:A224",
         "blueneg:19960812D-09-las-vegas")


def load(version, name=None):
    frames = {}
    for part in ("a", "b"):
        p = ROOT / version / f"eval_{name or version}_{part}.json"
        if p.exists():
            frames.update(json.loads(p.read_text())["frames"])
    return frames


def load_one(name):
    p = ROOT / "t4l" / name
    return json.loads(p.read_text())["frames"] if p.exists() else {}


def de(d, k, v="standard"):
    return d[k]["variants"][v]["metrics"]["de2000_median"]


def stats(x):
    return f"{np.median(x):.2f} | {np.mean(x):.2f} | {np.percentile(x, 90):.2f}"


def bw(delta):
    return f"{int((delta < -0.25).sum())} / {int((delta > 0.25).sum())}"


def valid(d):
    return {k for k, v in d.items() if "error" not in v and v.get("ncc", 0) >= 0.7}


def tables():
    j, k, l = load("t4j"), load("t4k"), load("t4l")
    s6 = load_one("self_ratio_off_6.json")
    variants = json.loads((ROOT / "t4l/eval_t4l_a.json").read_text())["variants"]
    common = sorted(valid(j) & valid(k) & valid(l))

    def de_final(x):
        return de(s6, x, "standard") if x in s6 and "error" not in s6[x] else de(l, x, "no_ev_down")

    lines = [f"Common valid frames: {len(common)}; evaluated T4l records: {len(l)}.", "",
             "ΔE: per-frame median ΔE2000, then median | mean | p90 over frames; better / worse: change > 0.25.", ""]
    for domain in DOMAINS:
        ids = [x for x in common if domain == "all" or l[x]["domain"] == domain]
        bk = np.array([de(k, x) for x in ids])
        lines += [f"### {domain} ({len(ids)})", "", "| Version / variant | ΔE median | mean | p90 | vs T4k standard |",
                  "|---|---:|---:|---:|---:|"]
        for label, d, v in [("T4j standard", j, "standard"), ("T4k standard", k, "standard")]:
            x = np.array([de(d, y, v) for y in ids])
            lines.append(f"| {label} | {stats(x)} | {'—' if d is k else bw(x - bk)} |")
        if s6:
            xf = np.array([de_final(y) for y in ids])
            lines.append(f"| **T4l finalized (4 mechanisms)** | {stats(xf)} | {bw(xf - bk)} |")
        for v in variants:
            x = np.array([de(l, y, v) for y in ids])
            lines.append(f"| T4l (caa0ba9) {v} | {stats(x)} | {bw(x - bk)} |")
        lines.append("")

    lines += ["### Each change alone (change on minus change off; negative = the change helps)", ""]
    for v, label in EST_OFF:
        d = np.array([de(l, x) - de(l, x, v) for x in common])
        changed = [x for x in common if l[x]["variants"]["standard"]["params"] != l[x]["variants"][v]["params"]]
        lines.append(f"- {label} (`{v}`): parameters changed on {len(changed)}; better / worse {bw(d)}; median "
                     f"{np.median(d):+.3f}; largest gain {d.min():+.2f} ({common[int(d.argmin())]}), largest loss "
                     f"{d.max():+.2f} ({common[int(d.argmax())]}).")
        for domain in DOMAINS[:-1]:
            dd = np.array([de(l, x) - de(l, x, v) for x in common if l[x]["domain"] == domain])
            lines.append(f"  - {domain}: {bw(dd)}, median {np.median(dd):+.3f}")
    srs = {x for x in common if ((l[x].get("white_point") or {}).get("self_ratio") or {}).get("applied")}
    for label, ids in (("inversion changes, all frames", common),
                       ("self-ratio white point frames (with the spike clamp)", sorted(srs)),
                       ("other frames (spike clamp only)", [x for x in common if x not in srs])):
        d = np.array([de(l, x, "t4k_est") - de(k, x) for x in ids])
        lines.append(f"- {label} ({len(ids)}; T4l t4k_est vs T4k standard): better / worse {bw(d)}; median "
                     f"{np.median(d):+.3f}; largest gain {d.min():+.2f} ({ids[int(d.argmin())]}), largest loss "
                     f"{d.max():+.2f} ({ids[int(d.argmax())]}).")
        if label.startswith("inversion"):
            for domain in DOMAINS[:-1]:
                dd = np.array([de(l, x, "t4k_est") - de(k, x) for x in ids if l[x]["domain"] == domain])
                lines.append(f"  - {domain}: {bw(dd)}, median {np.median(dd):+.3f}")
    lines.append("")

    lines += ["### Frames each mechanism touched (standard preset)", ""]
    clamp = {x: (l[x].get("spike_clamp") or {}) for x in common}
    cl = sorted((v.get("share", 0.0), x) for x, v in clamp.items() if v.get("pixels", 0) > 0)
    lines.append(f"- spike clamp changed pixels on {len(cl)} frames; largest share " +
                 ", ".join(f"{x} {s * 100:.3f} %" for s, x in reversed(cl[-8:])) + ".")
    sr = [(x, ((l[x].get("white_point") or {}).get("self_ratio") or {})) for x in common]
    srx = [(x, v) for x, v in sr if v.get("applied")]
    lines.append(f"- self-ratio white point applied on {len(srx)} frames: " + ", ".join(
        f"{x} {v['channels']} ({de(k, x):.2f} → {de(l, x):.2f})" for x, v in srx) + ".")
    for x in ("shop:DSC_3827", "ext:A006", "ext:A196"):  # the task list's targets
        v = dict(sr).get(x) or {}
        if v.get("ratio_mid"):
            lines.append(f"  - {x}: highlight R/G, B/G {v['ratio_hi'][0]:.3f}, {v['ratio_hi'][1]:.3f}; midtone "
                         f"{v['ratio_mid'][0]:.3f}, {v['ratio_mid'][1]:.3f}; applied {bool(v.get('applied'))}")
    for name, cond in (("holder box used", lambda m: "film_box" in m and m.get("used")),
                       ("black strips trimmed", lambda m: sum((m.get("strips") or {}).values()) > 0 and m.get("used"))):
        hit = [x for x in common if cond((l[x]["variants"]["standard"].get("t4l") or {}).get("metering") or {})]
        lines.append(f"- {name} on {len(hit)} frames" + (": " + ", ".join(
            f"{x} ({de(k, x):.2f} → {de(l, x):.2f})" for x in hit[:30]) if hit else "") + ".")
    dn = [x for x in common if (((l[x]["variants"]["standard"].get("t4l") or {}).get("ev_refine") or {})
                                .get("down_max", 0.2) > 0.2 + 1e-6)]
    ev_d = [l[x]["variants"]["standard"]["params"]["ev"] - l[x]["variants"]["no_ev_down"]["params"]["ev"] for x in dn]
    lines.append(f"- lowering limit widened on {len(dn)} frames; ev lowered further on "
                 f"{sum(1 for e in ev_d if e < -1e-3)} (median {np.median(ev_d) if ev_d else 0:+.2f} EV).")
    man = [x for x in common if l[x]["variants"]["standard"]["status"] != "ok"]
    lines.append(f"- manual review (any reason) on {len(man)} frames" + (": " + ", ".join(man) if man else "") + ".")
    lines.append("")

    lines += ["### Protected and named frames (standard)", "",
              "| Frame | T4j | T4k | **T4l finalized** | T4l (caa0ba9) | no_meter | no_strips | no_ev_down | t4k_est |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for x in PROTECTED + NAMED:
        if x not in common:
            continue
        lines.append(f"| {x} | {de(j, x):.2f} | {de(k, x):.2f} | **{de_final(x):.2f}** | {de(l, x):.2f} | " +
                     " | ".join(f"{de(l, x, v):.2f}" for v in ("no_meter", "no_strips", "no_ev_down", "t4k_est")) + " |")
    lines.append("")

    c_l, c_o = load_one("cross_t4l.json"), load_one("cross_inv_off.json")
    cc = sorted(valid(c_l) & valid(c_o))
    if cc:
        t4k = np.array([de(c_o, x, "t4k_est") for x in cc])
        lines += [f"### BlueNeg check set ({len(cc)} of {len(c_l)} records valid; same 14 rolls, frames outside the dev "
                  "manifest)", "", "| Version / variant | ΔE median | mean | p90 | vs T4k |", "|---|---:|---:|---:|---:|",
                  f"| T4k (both inversion changes off, t4k_est) | {stats(t4k)} | — |"]
        for label, d, v in (("T4l standard", c_l, "standard"), ("T4l no_meter", c_l, "no_meter"),
                            ("T4l no_ev_down", c_l, "no_ev_down"), ("T4l t4k_est (inversion changes alone)", c_l, "t4k_est"),
                            ("inversion off, T4l estimator", c_o, "standard")):
            x = np.array([de(d, y, v) for y in cc])
            lines.append(f"| {label} | {stats(x)} | {bw(x - t4k)} |")
        lines.append("")

    if (ROOT / "t4l/devices_t4l.json").exists():  # scripts/t4l_probe_devices.py on the T4k and T4l sources
        dk, dl = (json.loads((ROOT / f"t4l/devices_{v}.json").read_text())["samples"] for v in ("t4k", "t4l"))
        lines += ["### Multi-device samples (no reference; status and automatic parameters)", "",
                  "| Sample | Camera / scanner | T4k status | T4l status | T4l flags | EV T4k → T4l | Temp T4k → T4l | "
                  "spike pixels T4k → T4l | span_d_g | holder share |", "|---|---|---|---|---|---|---|---|---:|---:|"]
        for name in dl:
            a, b = dk.get(name, {}), dl[name]

            def st(r):
                return r.get("status") or ("error: " + r["error"][:40] if "error" in r else "—")

            def p(r, key, scale=1.0):
                v = (r.get("params") or {}).get(key)
                return "—" if v is None else f"{v * scale:+.2f}"
            lines.append(f"| {name} | {(b.get('camera') or {}).get('model') or '—'} | {st(a)} | {st(b)} | "
                         f"{'; '.join(b.get('flags') or []) or '—'} | {p(a, 'ev')} → {p(b, 'ev')} | "
                         f"{p(a, 'temp', 100)} → {p(b, 'temp', 100)} | {a.get('spike_px', '—')} → {b.get('spike_px', '—')} | "
                         f"{b.get('span_d_g', float('nan')):.3f} | {b.get('holder_share') or 0:.3f} |")
        lines.append("")

    lines += ["### Largest standard changes vs T4k", "", "| Frame | T4k | T4l | Change | no_meter | no_ev_down | t4k_est |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    ch = sorted((de(l, x) - de(k, x), x) for x in common)
    for item in list(reversed(ch[-10:])) + [None] + ch[:10]:
        if item is None:
            lines.append("| … | | | | | | |")
            continue
        dlt, x = item
        lines.append(f"| {x} | {de(k, x):.2f} | {de(l, x):.2f} | {dlt:+.2f} | " +
                     " | ".join(f"{de(l, x, v):.2f}" for v in ("no_meter", "no_ev_down", "t4k_est")) + " |")
    lines += ["", f"Minimum T4l NCC {min(l[x]['ncc'] for x in common):.4f}; largest NCC change vs T4k "
              f"{max(abs(l[x]['ncc'] - k[x]['ncc']) for x in common):.4f}.", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    text = tables()
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(text)
    else:
        print(text)
