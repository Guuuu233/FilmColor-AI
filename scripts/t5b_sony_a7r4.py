"""T5b §6: A1298 / A1299 / A1300 (Sony ILCE-7RM4) opened as the app opens them, T5a vs T5b.
Usage: FILMCOLOR_DATA_ROOT=/tmp/fcroot_t5b_sony_LABEL python -I scripts/t5b_sony_a7r4.py SRC baseline|candidate
       python -I scripts/t5b_sony_a7r4.py --compare
Each frame: check_source(allow_frozen=False), a temporary folder holding only its symlink, State.open(frame, "color")
at max_side 1600. Records base, scan diagnostics, d_hi, status / flags / params, core sRGB q50 and a preview JPG.
--compare writes sony_a7r4.json and the local before_after.jpg (T5a top row, T5b bottom row) from the two records,
plus, for A1298, the count of blue pixels (B > R + 60 and B > 120) in the left film-edge band of the 900 px previews.
"""
import gc
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t5b"
MATERIAL = Path("/Volumes/waizhiyingpan/胶片负片素材/01_彩色负片_未反相")
FRAMES = ("A1298.ARW", "A1299.ARW", "A1300.ARW")
EDGE_BAND = (128, 260, 152, 472)  # A1298 left film edge in the 900 x 600 preview: x0, y0, x1, y1 (film_box left at x 125)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_sha(src):
    h = hashlib.sha256()
    for p in sorted(p for p in (src / "filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(src)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def probe(src, label):
    assert os.environ.get("FILMCOLOR_DATA_ROOT") == f"/tmp/fcroot_t5b_sony_{label}"  # T5b §0.6
    sys.path.insert(0, str(src))
    import numpy as np
    from PIL import Image
    import filmcolor.app as fc_app
    from filmcolor.app import State, to8
    from filmcolor.frozen import check_source
    from filmcolor.holder import film_box
    assert Path(fc_app.__file__).resolve().is_relative_to(src)
    fingerprint = source_sha(src)
    result = dict(label=label, src=str(src), src_content_sha256=fingerprint, script_sha256=sha(__file__),
                  max_side=1600, frames={})
    for name in FRAMES:
        p = MATERIAL / name
        digest = check_source(p, allow_frozen=False)
        with tempfile.TemporaryDirectory(prefix=f"fc_t5b_sony_{label}_") as td:
            (Path(td) / name).symlink_to(p)
            st = State(Path(td), Path(td) / "out", False, max_side=1600)
            st.roll_info()
            meta = st.open(name, "color")
            pv = st.preview
            disp = pv.render(pv.auto["params"])
            fb = film_box(pv.holder)
            core = disp if fb is None else disp[fb[0]:fb[0] + fb[2], fb[1]:fb[1] + fb[3]]
            scan = st.frame_scans.get(name) or {}
            diag = (scan.get("estimate") or {}).get("diagnostics") or {}
            img = Image.fromarray(to8(disp))
            img.thumbnail((900, 900))
            jpg = OUT / f"sony_{Path(name).stem}_{label}.jpg"
            img.save(jpg, quality=92)
            result["frames"][name] = dict(
                raw_sha256=digest, status=pv.auto["status"], flags=pv.auto["flags"], params=pv.auto["params"],
                base_method=meta["base"]["method"], base_rgb=meta["base"]["rgb"], base_warnings=meta["base"]["warnings"],
                own_rgb=scan.get("own_rgb"),
                base_diagnostics={k: diag.get(k) for k in ("near_saturated_blocks_excluded", "bare_backlight_excluded",
                                                           "clipped_base_excluded", "n_kept") if k in diag},
                d_hi=list(pv.inversion["d_hi"]), input_warnings=meta.get("input_warnings"),
                holder_share=float(pv.holder.mean()), film_box=fb,
                core_srgb_q50=np.median(core, axis=(0, 1)).tolist(),
                display_sha256=hashlib.sha256(to8(disp).tobytes()).hexdigest(),
                preview_image=jpg.name, preview_image_sha256=sha(jpg))
            print(label, name, meta["base"]["method"], [round(x, 4) for x in meta["base"]["rgb"]],
                  [round(x, 3) for x in pv.inversion["d_hi"]], pv.auto["status"], flush=True)
            st.preview = None
            del pv, disp, core
            gc.collect()
    assert source_sha(src) == fingerprint
    (OUT / f"sony_a7r4_{label}.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def compare():
    from PIL import Image, ImageDraw
    recs = {k: json.loads((OUT / f"sony_a7r4_{k}.json").read_text()) for k in ("baseline", "candidate")}
    sheet = Image.new("RGB", (3 * 520, 2 * 420), "#222222")
    draw = ImageDraw.Draw(sheet)
    rows = {}
    for col, name in enumerate(FRAMES):
        b, c = recs["baseline"]["frames"][name], recs["candidate"]["frames"][name]
        assert b["raw_sha256"] == c["raw_sha256"]
        rg = lambda rgb: [round(rgb[0] / rgb[1], 3), round(rgb[2] / rgb[1], 3)]
        rows[name] = dict(
            raw_sha256=c["raw_sha256"],
            t5a=dict(base_rgb=b["base_rgb"], base_RG_BG=rg(b["base_rgb"]), d_hi=b["d_hi"], d_hi_g=b["d_hi"][1],
                     core_srgb_q50=b["core_srgb_q50"], status=b["status"], base_diagnostics=b["base_diagnostics"],
                     ev=b["params"]["ev"]),
            t5b=dict(base_rgb=c["base_rgb"], base_RG_BG=rg(c["base_rgb"]), d_hi=c["d_hi"], d_hi_g=c["d_hi"][1],
                     core_srgb_q50=c["core_srgb_q50"], status=c["status"], base_diagnostics=c["base_diagnostics"],
                     ev=c["params"]["ev"], base_warnings=c["base_warnings"]))
        for row, (label, version) in enumerate((("baseline", "T5a"), ("candidate", "T5b"))):
            r = recs[label]["frames"][name]
            p = OUT / r["preview_image"]
            assert sha(p) == r["preview_image_sha256"]
            with Image.open(p) as opened:
                image = opened.convert("RGB")
            image.thumbnail((510, 340))
            x, y = col * 520, row * 420
            sheet.paste(image, (x + (520 - image.width) // 2, y + 72 + (340 - image.height) // 2))
            base = r["base_rgb"]
            draw.text((x + 10, y + 5), f"{name}  {version}  {r['status']}  base {[round(v, 3) for v in base]}", fill="white")
            draw.text((x + 10, y + 22), f"R/G {base[0] / base[1]:.2f}  B/G {base[2] / base[1]:.2f}  d_hi G {r['d_hi'][1]:.3f}"
                                        f"  EV {r['params']['ev']:.3f}", fill="white")
            draw.text((x + 10, y + 39), f"core q50 {[round(v, 3) for v in r['core_srgb_q50']]}  "
                                        f"clipped_base_excluded {bool(r['base_diagnostics'].get('clipped_base_excluded'))}",
                      fill="white")
    jpg = OUT / "before_after.jpg"
    sheet.save(jpg, quality=94)
    import numpy as np
    edge = {}
    for label, version in (("baseline", "t5a"), ("candidate", "t5b")):
        r = recs[label]["frames"]["A1298.ARW"]
        with Image.open(OUT / r["preview_image"]) as opened:
            im = np.asarray(opened.convert("RGB")).astype(np.int32)
        x0, y0, x1, y1 = EDGE_BAND
        band = im[y0:y1, x0:x1]
        edge[version] = int(((band[..., 2] > band[..., 0] + 60) & (band[..., 2] > 120)).sum())
    rows["A1298.ARW"]["edge_band_blue_px"] = dict(edge, band_xyxy=list(EDGE_BAND), rule="B > R + 60 and B > 120",
                                                  preview_side=900)
    out = dict(baseline_src_content_sha256=recs["baseline"]["src_content_sha256"],
               candidate_src_content_sha256=recs["candidate"]["src_content_sha256"],
               baseline_sha256=sha(OUT / "sony_a7r4_baseline.json"), candidate_sha256=sha(OUT / "sony_a7r4_candidate.json"),
               before_after=dict(file=jpg.name, sha256=sha(jpg)), frames=rows)
    (OUT / "sony_a7r4.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    for name, r in rows.items():
        print(name, "base", [round(x, 3) for x in r["t5a"]["base_rgb"]], "->", [round(x, 3) for x in r["t5b"]["base_rgb"]],
              "d_hi_G", round(r["t5a"]["d_hi_g"], 3), "->", round(r["t5b"]["d_hi_g"], 3),
              "core q50", [round(x, 3) for x in r["t5a"]["core_srgb_q50"]], "->", [round(x, 3) for x in r["t5b"]["core_srgb_q50"]])


if __name__ == "__main__":
    if sys.argv[1:] == ["--compare"]:
        compare()
    else:
        assert len(sys.argv) == 3 and sys.argv[2] in ("baseline", "candidate")
        probe(Path(sys.argv[1]).resolve(), sys.argv[2])
