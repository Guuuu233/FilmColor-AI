"""Roll-level scan exposure audit and per-frame film-base candidates (analysis path only, no demosaic).

For every RAW of a roll: capture settings (EXIF), raw signal level, and the automatic base estimate of the
frame interior (brightest uniform, chromatically consistent blocks), normalised to a common exposure.
Settings that are identical across the roll (ISO, aperture, lens, focal length) cancel; only exposure time
is normalised, both with the EXIF nominal value and with the exact 1/3-stop value 2^(-k/3) the nominal
label usually stands for. The two differ by up to 6.7 %, an unknown that is neutral (same for all channels).

  roll_exposure.py MANIFEST_CSV OUT_JSON
"""
import csv
import json
import sys
from fractions import Fraction
from pathlib import Path

import exifread
import numpy as np

from filmcolor.base import estimate_base_auto
from filmcolor.raw import linearize, saturation_mask, superpixel
from filmcolor.rawpy_io import load_raw

root = Path(__file__).resolve().parents[1]
BLOCK = 24


def exif(path):
    with open(path, "rb") as fh:
        t = exifread.process_file(fh, details=False)
    r = t["EXIF ExposureTime"].values[0]
    nominal = Fraction(r.num, r.den)
    stops = round(-3 * np.log2(float(nominal))) / 3  # nearest 1/3 stop
    return dict(exposure_nominal=float(nominal), exposure_third_stop=float(2.0 ** -stops),
                fnumber=str(t.get("EXIF FNumber")), iso=str(t.get("EXIF ISOSpeedRatings")),
                focal=str(t.get("EXIF FocalLength")), lens=str(t.get("EXIF LensModel")),
                program=str(t.get("EXIF ExposureProgram")), bias=str(t.get("EXIF ExposureBiasValue")),
                metering=str(t.get("EXIF MeteringMode")), wb=str(t.get("MakerNote Whitebalance", "")),
                datetime=str(t.get("EXIF DateTimeOriginal")), subsec=str(t.get("EXIF SubSecTimeOriginal", "")))


def frame_report(path):
    f = load_raw(path)
    lin, col, sat = linearize(f), f.colors, saturation_mask(f)
    sp, sp_sat = superpixel(lin, col, sat, f.pattern.shape[0], BLOCK)
    est = estimate_base_auto(lin, col, sat, f.pattern.shape[0], crop=f.crop)
    Y = sp.mean(-1)
    hb, wb = Y.shape
    kr0, kc0, kr1, kc1 = est.diagnostics["kept_bbox"]
    centre = ((kr0 + kr1) / 2 / (hb * BLOCK), (kc0 + kc1) / 2 / (wb * BLOCK))  # relative position
    return dict(
        raw_max_frac=float(lin.max()), raw_min=float(lin.min()), below_black_frac=float((lin < 0).mean()),
        sat_frac=float(sat.mean()),
        block_pct={n: [float(x) for x in np.percentile(sp[..., i], [1, 50, 99, 99.9])] for i, n in enumerate("RGB")},
        base=dict(rgb=list(est.rgb), n_kept=est.diagnostics["n_kept"], kept_bbox=est.diagnostics["kept_bbox"],
                  kept_centre_rel=centre, chroma_log2_RG_BG=est.diagnostics["chroma_centre_log2_RG_BG"],
                  rel_spread={k: v["rel_spread"] for k, v in est.stats.items()}, warnings=est.warnings),
        blocks=sp, crop=f.crop,
    )


def main(manifest, out_json):
    rows = list(csv.DictReader(open(manifest, newline="")))
    frames, blocks = {}, {}
    for r in rows:
        p = root / "data" / "raw" / r["file"]
        e = exif(p)
        fr = frame_report(p)
        blocks[p.name] = (fr.pop("blocks"), e)
        frames[p.name] = dict(exif=e, **fr)
        b = fr["base"]["rgb"]
        print(p.name, f"t={e['exposure_nominal']:.4f}", f"max {fr['raw_max_frac']:.3f}",
              "base/t " + " ".join(f"{x / e['exposure_nominal']:.3f}" for x in b), flush=True)
    t_ref = 1 / 30
    for norm in ("exposure_nominal", "exposure_third_stop"):
        for name, fr in frames.items():
            k = t_ref / fr["exif"][norm]
            fr.setdefault("base_norm", {})[norm] = [x * k for x in fr["base"]["rgb"]]
    # roll candidate = frame whose normalised base is the most transparent (highest G); check every other
    # frame for blocks more transparent than it (a true base cannot be exceeded except by noise / light leaks)
    summary = {}
    for norm in ("exposure_nominal", "exposure_third_stop"):
        top = max(frames, key=lambda n: frames[n]["base_norm"][norm][1])
        cand = np.array(frames[top]["base_norm"][norm])
        exceed = {}
        for name, (sp, e) in blocks.items():
            spn = sp * (t_ref / e[norm])
            exceed[name] = float(np.mean(np.all(spn > cand * 1.0, -1)))
        summary[norm] = dict(candidate_frame=top, candidate_rgb=cand.tolist(),
                             candidate_density_offsets_vs_frames={
                                 n: [float(np.log10(c / x)) for c, x in zip(cand, fr["base_norm"][norm])]
                                 for n, fr in frames.items()},
                             blocks_exceeding_candidate_frac=exceed)
    settings = {k: sorted({fr["exif"][k] for fr in frames.values()}) for k in
                ("fnumber", "iso", "focal", "lens", "program", "bias", "metering", "wb")}
    out = dict(manifest=str(manifest), block=BLOCK, t_ref=t_ref, settings_across_roll=settings,
               frames=frames, summary=summary)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(out, indent=1))
    print("wrote", out_json)


if __name__ == "__main__":
    main(*sys.argv[1:3])
