"""T3a step 2 (task list 2026-10-23): which visible-rebate frames give a usable physical film-base reference, and
how far the automatic base estimates are from it.

  rebate_verify.py MATERIAL_ROOT CANDIDATES_JSON ROIS_JSON OUT_JSON SHEET_DIR

Inputs: candidates.json of t3a_inspect.py (1/8 caches T8 / S8 under $FILMCOLOR_DATA_ROOT/t3_inversion/T8) and the
boxes of ROIS_JSON, placed by eye on the grid previews before this script was first run (x0, x1, y0, y1 in % of
width / height, sensor orientation). A7R5 boxes are those of scripts/a7_measure.py (REBATE / IMAGE).
Not measured: copies of the same capture (same camera model and capture time, different bytes; the first id in sort
order is kept) and frames ROIS_JSON excludes (e.g. cross-processed slide film); both listed in summary.skipped.

Criteria, all four needed for a usable measured-base frame (fixed before measuring):
  C1 film rebate   at least one rebate box on unexposed film (gap between frames, edge, between sprocket holes),
                   not on a sprocket hole or a holder: the by-eye placement; no box -> fails with the reason noted
  C2 signal        no saturated sample in any rebate box (S8 == 0) and every channel above the noise floor
                   (every block > 0 and the box median >= FLOOR of the white level)
  C3 uniform       every rebate box: per-channel CV = std / mean of the 1/8 blocks <= MAX_CV; with two or more
                   boxes the largest per-channel density difference between box medians <= MAX_DD
  C4 interior box  IMAGE_BOX given: strictly inside the picture, no rebate, sprocket area or holder
Measured base b_meas = per-channel median of the 1/8 blocks of all rebate boxes together. Backlight (optional box on
film-free light, e.g. inside a sprocket hole): used for the absolute base density only when no block of it is
saturated; otherwise the frame has relative numbers only (dD below, the backlight cancels in the ratio).
Estimates, same decode as the engine's camera path:
  b_own  pipeline.positive's automatic base (whole frame, decoder crop)
  b_int  the same estimator with the block search limited to IMAGE_BOX (a scan cropped inside the picture)
  b_roll_int  estimate_base_roll over the b_int of a session group (same body, day, ISO, aperture, lens, film in
         the index; exposure time normalised; >= 3 frames passing C1-C4), only for groups formed that way
dD_c = -log10(b_est,c / b_meas,c) (+ = estimate denser than the measured base); level part = mean over channels,
chroma part = per-channel deviation from it (|.| max reported as in sep_summary.py), R-G and B-G as dD_R - dD_G,
dD_B - dD_G. Full-resolution percentiles of L = -log10 T inside IMAGE_BOX (unsaturated pixels) are stored for T3b.
Writes OUT_JSON and one ROI sheet per frame in SHEET_DIR (git-ignored): rebate boxes cyan, backlight box white,
IMAGE_BOX yellow, saturated blocks magenta.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from a7_measure import IMAGE as A7_IMAGE, REBATE as A7_REBATE  # noqa: E402

from filmcolor.base import estimate_base_auto, estimate_base_roll  # noqa: E402
from filmcolor.frozen import check_source, load_cache  # noqa: E402
from filmcolor.invert import EPS, base_from_regions  # noqa: E402
from filmcolor.paths import data_dir  # noqa: E402
from filmcolor.raw import linearize, saturation_mask  # noqa: E402
from filmcolor.rawpy_io import demosaic_linear, load_raw  # noqa: E402

FLOOR, MAX_CV, MAX_DD, SPLIT = 0.01, 0.05, 0.05, 0.1
PCTS = (0.01, 0.1, 0.5, 1, 50, 99, 99.9)


def rect(a, r):
    h, w = a.shape[:2]
    x0, x1, y0, y1 = r
    return a[int(h * y0 / 100):int(h * y1 / 100), int(w * x0 / 100):int(w * x1 / 100)]


def crop_of(shape, r):
    """(top, left, height, width) of a % box in an array of ``shape``."""
    H, W = shape[:2]
    x0, x1, y0, y1 = r
    return int(H * y0 / 100), int(W * x0 / 100), int(H * (y1 - y0) / 100), int(W * (x1 - x0) / 100)


def dD(est, ref):
    return (-np.log10(np.asarray(est, np.float64) / np.asarray(ref, np.float64))).round(4)


def parts(d):
    lvl = float(d.mean())
    return dict(dD=d.tolist(), level=round(lvl, 4), chroma=(d - lvl).round(4).tolist(),
                chroma_part=round(float(np.abs(d - lvl).max()), 4), level_part=round(abs(lvl), 4),
                R_minus_G=round(float(d[0] - d[1]), 4), B_minus_G=round(float(d[2] - d[1]), 4))


def boxes(stem, rois):
    if stem.startswith("A7R5_RAW_"):
        s = stem.removeprefix("A7R5_RAW_")
        return dict(rebate=A7_REBATE[s], image_box=A7_IMAGE[s], backlight=None,
                    note="boxes of scripts/a7_measure.py (placed 2026-10-18)")
    return rois[stem]


def estimates(path, box, cache_file, sha):
    """b_own and b_int (render-path medians over the estimator's blocks), warnings, full-res L percentiles."""
    box = list(box)
    if cache_file.exists():
        d = json.loads(cache_file.read_text())
        if d.get("sha256") == sha and d.get("image_box") == box:
            return ((None if d["b_own"] is None else np.array(d["b_own"])), d["own_warnings"],
                    (None if d["b_int"] is None else np.array(d["b_int"])), d["int_warnings"],
                    d["L_percentiles_box"], d["box_pixels"])
    f = load_raw(path)
    lin, col, sat = linearize(f), f.colors, saturation_mask(f)
    est = {}
    for name, crop in (("own", f.crop), ("int", crop_of(lin.shape, box))):
        try:
            est[name] = estimate_base_auto(lin, col, sat, f.pattern.shape[0], crop=crop)
        except ValueError as e:  # e.g. no uniform unsaturated block inside the box
            est[name] = str(e)
    del lin, col, f
    rgb = demosaic_linear(path)
    if rgb.shape[:2] != sat.shape:
        raise ValueError(f"{path}: render shape {rgb.shape[:2]} != CFA shape {sat.shape}")
    out = {}
    for name, e in est.items():
        out[name] = (np.array(base_from_regions(rgb, e.regions)["rgb"]), e.warnings) if not isinstance(e, str) \
            else (None, ["estimator failed: " + e])
    t, l, h, w = crop_of(rgb.shape, box)
    px = rgb[t:t + h, l:l + w][~sat[t:t + h, l:l + w]]
    L = -np.log10(np.maximum(px.astype(np.float64), EPS))
    pct = {str(q): np.percentile(L, q, 0).round(6).tolist() for q in PCTS}
    pinfo = dict(n=int(len(px)), sat_in_box=float(sat[t:t + h, l:l + w].mean()))
    cache_file.write_text(json.dumps(dict(
        sha256=sha, image_box=box,
        b_own=None if out["own"][0] is None else out["own"][0].tolist(), own_warnings=out["own"][1],
        b_int=None if out["int"][0] is None else out["int"][0].tolist(), int_warnings=out["int"][1],
        L_percentiles_box=pct, box_pixels=pinfo)))
    return (*out["own"], *out["int"], pct, pinfo)


def sheet(T8, S8, b, path, title):
    v = (np.clip(T8 / np.percentile(T8, 99.9), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)
    v[S8 > 0] = (255, 0, 255)
    img = Image.fromarray(v)
    d = ImageDraw.Draw(img)
    h, w = T8.shape[:2]
    xy = lambda r: (w * r[0] / 100, h * r[2] / 100, w * r[1] / 100, h * r[3] / 100)
    for r in b["rebate"].values():
        d.rectangle(xy(r), outline=(0, 255, 255), width=3)
    if b.get("backlight"):
        d.rectangle(xy(b["backlight"]), outline=(255, 255, 255), width=3)
    if b.get("image_box"):
        d.rectangle(xy(b["image_box"]), outline=(255, 255, 0), width=3)
    d.text((6, 6), title, fill=(255, 255, 255))
    img.save(path, quality=88)


def main(mat, candidates, rois_json, out_json, sheet_dir):
    mat, sd = Path(mat), Path(sheet_dir)
    sd.mkdir(parents=True, exist_ok=True)
    cands = [r for r in json.loads(Path(candidates).read_text())["frames"] if "excluded" not in r]
    rois = json.loads(Path(rois_json).read_text())["frames"]
    cache = data_dir("t3_inversion", "T8")
    # same capture under two names (same camera model and capture time, different bytes): counted once, the first
    # id in sort order is kept
    first, skipped = {}, {}
    for c in sorted(cands, key=lambda c: c["id"]):
        key = (c["exif"].get("Model"), c["exif"].get("DateTimeOriginal"))
        if key[1] and not str(key[1]).startswith("0000"):
            if key in first:
                skipped[c["id"]] = f"same capture as {first[key]} (model and capture time)"
            else:
                first[key] = c["id"]
    rows = {}
    for c in cands:
        if c["id"] in skipped:
            continue
        stem, b = c["stem"], boxes(c["stem"], rois)
        if b.get("exclude"):
            skipped[c["id"]] = "excluded: " + b["exclude"]
            continue
        if check_source(mat / c["rel"]) != c["sha256"]:
            raise SystemExit(f"{c['rel']}: file changed since t3a_inspect.py")
        T8 = load_cache(cache / f"{stem}_T8.npy").astype(np.float64)
        S8 = load_cache(cache / f"{stem}_S8.npy")
        r = dict(id=c["id"], rel=c["rel"], source=c["source"], camera=c["exif"].get("Model"),
                 film_index=c["film_index"], rebate_boxes=b["rebate"], backlight_box=b.get("backlight"),
                 image_box=b.get("image_box"), placement_note=b.get("note"), fails=[])
        if not b["rebate"]:
            r["fails"].append("C1: " + (b.get("no_rebate_reason") or "no rebate box placed"))
        else:
            per = {}
            for side, box in b["rebate"].items():
                v, s = rect(T8, box).reshape(-1, 3), rect(S8, box)
                per[side] = dict(n=len(v), median=np.median(v, 0).tolist(), cv=(v.std(0) / v.mean(0)).round(4).tolist(),
                                 min=v.min(0).tolist(), sat_blocks=int((s > 0).sum()))
            r["rebate"] = per
            if any(p["sat_blocks"] for p in per.values()):
                r["fails"].append("C2: saturated samples in a rebate box")
            if any(min(p["min"]) <= 0 or min(p["median"]) < FLOOR for p in per.values()):
                r["fails"].append(f"C2: rebate signal at the noise floor (block <= 0 or median < {FLOOR})")
            if any(max(p["cv"]) > MAX_CV for p in per.values()):
                r["fails"].append(f"C3: rebate box not uniform (CV > {MAX_CV})")
            meds = [np.array(p["median"]) for p in per.values()]
            if len(meds) > 1:
                dd = max(float(np.abs(dD(a, m)).max()) for a in meds for m in meds)
                r["between_boxes_max_dD"] = round(dd, 4)
                if dd > MAX_DD:
                    r["fails"].append(f"C3: rebate boxes disagree by {dd:.3f} D (> {MAX_DD})")
            pooled = np.concatenate([rect(T8, box).reshape(-1, 3) for box in b["rebate"].values()])
            r["b_meas"] = np.median(pooled, 0).tolist()
        if not b.get("image_box"):
            r["fails"].append("C4: " + (b.get("no_image_box_reason") or "no interior box"))
        if b.get("backlight") and "b_meas" in r:
            v, s = rect(T8, b["backlight"]).reshape(-1, 3), rect(S8, b["backlight"])
            r["backlight"] = dict(median=np.median(v, 0).tolist(), sat_blocks=int((s > 0).sum()))
            if r["backlight"]["sat_blocks"] == 0:
                r["D_meas_abs"] = dD(r["b_meas"], r["backlight"]["median"]).tolist()
        r["reference_kind"] = ("absolute (unsaturated backlight)" if "D_meas_abs" in r else "relative only")
        r["usable"] = not r["fails"]
        if b.get("image_box") and "b_meas" in r:
            b_own, w_own, b_int, w_int, pct, pinfo = estimates(
                mat / c["rel"], b["image_box"], cache / f"{stem}_est.json", c["sha256"])
            r.update(b_own=None if b_own is None else b_own.tolist(), b_int=None if b_int is None else b_int.tolist(),
                     own_warnings=w_own, int_warnings=w_int, L_percentiles_box=pct, box_pixels=pinfo)
            if "b_meas" in r:
                for key, est in (("own_vs_meas", b_own), ("int_vs_meas", b_int)):
                    if est is not None:
                        r[key] = parts(dD(est, r["b_meas"]))
        r["session"] = "|".join(str(c["exif"].get(k)) for k in ("body", "Model", "ISO", "FNumber", "LensModel")) \
            + "|" + str(c["exif"].get("DateTimeOriginal") or "")[:10] + "|" + str(c["film_index"])
        r["exposure_time"] = c["exif"].get("ExposureTime")
        rows[stem] = r
        sheet(T8, S8, b, sd / f"{stem}_roi.jpg",
              f"{c['id']}  {'usable' if r['usable'] else '; '.join(r['fails'])}")
        print(stem, "usable" if r["usable"] else r["fails"], r.get("int_vs_meas", {}).get("dD"), flush=True)

    groups = defaultdict(list)
    for stem, r in rows.items():
        if r["usable"] and r["exposure_time"] and r.get("b_int"):
            groups[r["session"]].append(stem)
    roll = {}
    for key, stems in sorted(groups.items()):
        if len(stems) < 3:
            continue
        gid = f"group{len(roll) + 1}"
        try:
            rb = estimate_base_roll({s: rows[s]["b_int"] for s in stems},
                                    {s: float(rows[s]["exposure_time"]) for s in stems})
        except ValueError as e:
            roll[gid] = dict(frames=sorted(stems), error=str(e))
            for s in stems:
                rows[s]["roll_group"] = gid
            continue
        roll[gid] = dict(frames=sorted(stems), used=rb.used, warnings=rb.warnings, t_ref=rb.t_ref)
        for s in stems:
            r = rows[s]
            b_ri = np.array(rb.for_exposure(float(r["exposure_time"])))
            r["roll_group"], r["b_roll_int"] = gid, b_ri.tolist()
            r["roll_int_vs_meas"] = parts(dD(b_ri, r["b_meas"]))

    use = [r for r in rows.values() if r["usable"]]

    def dist(key, part):
        x = np.array([r[key][part] for r in use if key in r])
        if not len(x):
            return None
        return dict(n=len(x), median=round(float(np.median(x)), 4),
                    iqr=[round(float(v), 4) for v in np.percentile(x, [25, 75])],
                    share_ge_0_1=round(float(np.mean(np.abs(x) >= SPLIT)), 4))

    by_source = defaultdict(lambda: defaultdict(int))
    for r in rows.values():
        by_source[r["source"]]["measured"] += 1
        by_source[r["source"]]["usable"] += r["usable"]
    by_camera = defaultdict(lambda: defaultdict(int))
    for r in rows.values():
        by_camera[r["camera"]]["measured"] += 1
        by_camera[r["camera"]]["usable"] += r["usable"]
    fails = defaultdict(int)
    for r in rows.values():
        for f in r["fails"]:
            fails[f.split(":")[0]] += 1
    summary = dict(
        candidates_linear=len(cands), skipped=dict(sorted(skipped.items())),
        measured=len(rows), usable=len(use),
        usable_absolute=sum(1 for r in use if "D_meas_abs" in r),
        fails_by_criterion=dict(sorted(fails.items())),
        by_source={k: dict(v) for k, v in sorted(by_source.items())},
        by_camera={k: dict(v) for k, v in sorted(by_camera.items(), key=lambda kv: str(kv[0]))},
        roll_groups=roll,
        deviation={key: {part: dist(key, part) for part in ("level", "level_part", "chroma_part", "R_minus_G",
                                                             "B_minus_G")}
                   for key in ("int_vs_meas", "own_vs_meas", "roll_int_vs_meas")},
        int_ge_0_1_either_part=round(float(np.mean([max(r["int_vs_meas"]["level_part"], r["int_vs_meas"]["chroma_part"])
                                                    >= SPLIT for r in use if "int_vs_meas" in r])), 4)
        if any("int_vs_meas" in r for r in use) else None,
        thresholds=dict(FLOOR=FLOOR, MAX_CV=MAX_CV, MAX_DD=MAX_DD, SPLIT=SPLIT))
    Path(out_json).write_text(json.dumps(dict(summary=summary, frames=rows), ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:6])
