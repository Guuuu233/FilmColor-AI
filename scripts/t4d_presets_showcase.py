"""T4d style-preset sheet: one row per frame, one column per noref.STYLE_PRESETS entry (the app's automatic starting
point for that preset, nothing moved by hand), plus the frame's reference conversion as the last column.

  t4d_presets_showcase.py OUT_JPG [--tile-h=PX] ID [ID ...]

ID: frame ids of data/manifests/s3_dev_frames.json (e.g. nlp:_DSF9237, ext:A012). Each frame is opened as the app
opens it (State.open, automatic base choice, on a roll folder of symlinks to the manifest's frames of that roll, the
same folders as t4d_source_eval.py), each preset applied with State.set_preset (re-estimate on the kept positive), and
the 1000 px preview rendered with the preset's automatic parameters; tiles are turned by the app's orientation for
the file; camera-frame references (ICC-tagged, in the scan's own orientation) are turned the same way, BlueNeg prints
are shown as stored. Files are checked against the frozen sets before they are read. Writes OUT_JPG and OUT_JPG's .json
(parameters per frame and preset, software commit); images are not committed (experiments/**/*.jpg is ignored).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import filmcolor.app as fc_app
from filmcolor.frozen import check_source
from filmcolor.noref import STYLE_PRESETS

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data/manifests/s3_dev_frames.json"
ROLLS = Path("/tmp/fc_t4d/rolls")
LABELS_ASCII = {"standard": "STANDARD", "natural": "NATURAL", "portrait": "PORTRAIT", "vivid": "VIVID",
                "print_classic": "PRINT CLASSIC"}  # the default bitmap font has no CJK glyphs


def path_of(p: str) -> Path:
    return Path(p) if os.path.isabs(p) else REPO / p


def roll_folder(f: dict, frames: list) -> Path:
    d = ROLLS / f["domain"] / f["roll"]
    d.mkdir(parents=True, exist_ok=True)
    for g in frames:
        if (g["domain"], g["roll"]) == (f["domain"], f["roll"]):
            link = d / Path(g["raw"]).name
            if not link.is_symlink():
                link.symlink_to(path_of(g["raw"]).resolve())
    return d


def reference_rgb(f: dict) -> Image.Image:
    ref = path_of(f["ref"])
    check_source(ref)
    if ref.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        a = tifffile.imread(ref)[..., :3]
        a = (a / 257).round().astype(np.uint8) if a.dtype == np.uint16 else a.astype(np.uint8)
        return Image.fromarray(a)  # TIFF values shown as sRGB (NLP positives are sRGB-tagged; BlueNeg assumed sRGB)
    return Image.open(ref).convert("RGB")


def tile(im: Image.Image, h: int, title: str, sub: str = "") -> Image.Image:
    w = int(round(im.width * h / im.height))
    out = Image.new("RGB", (w, h + 34), (22, 22, 25))
    out.paste(im.resize((w, h), Image.LANCZOS), (0, 34))
    d = ImageDraw.Draw(out)
    d.text((6, 3), title, fill=(235, 235, 240))
    d.text((6, 18), sub, fill=(170, 195, 220))
    return out


def main(out_jpg: str, ids: list[str], tile_h: int):
    frames = json.loads(MANIFEST.read_text())["frames"]
    by_id = {f["id"]: f for f in frames}
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "src"], capture_output=True,
                                text=True).stdout.strip())
    rows, record = [], dict(git_commit=commit, src_dirty=dirty, presets=list(STYLE_PRESETS), frames={})
    states = {}
    for fid in ids:
        f = by_id[fid]
        raw = path_of(f["raw"])
        check_source(raw)
        d = roll_folder(f, frames)
        if d not in states:
            states[d] = fc_app.State(d, Path("/tmp/fc_t4d/out"), allow_frozen=False, max_side=1000)
            states[d].roll_info()
        st = states[d]
        m = st.open(raw.name, "color", preset="standard")
        k = int(m["rot90"])
        cols, rec = [], dict(base_method=m["base"]["method"], rot90=k, presets={})
        for name in STYLE_PRESETS:
            mm = st.set_preset(dict(token=st.token, preset=name))
            p = mm["auto"]["params"]
            disp = st.preview.render(p)
            im = Image.fromarray(np.round(np.clip(np.rot90(disp, k), 0, 1) * 255).astype(np.uint8))
            sub = " ".join(f"{a}={p.get(a, 0.0):+.2f}" for a in ("ev", "contrast", "midtone", "shoulder", "sat"))
            cols.append(tile(im, tile_h, f"{LABELS_ASCII[name]}  {fid}", sub))
            rec["presets"][name] = dict(params=p, flags=mm["auto"]["flags"],
                                        backlit=(mm["auto"]["diagnostics"] or {}).get("backlit_check"))
        ref_im = reference_rgb(f)
        if f["ref_kind"] == "icc" and k:
            ref_im = ref_im.rotate(90 * k, expand=True)  # PIL rotate is counter-clockwise, as numpy.rot90
        cols.append(tile(ref_im, tile_h, f"REFERENCE ({f['domain']})", Path(f["ref"]).name.encode("ascii", "replace").decode()))
        w = sum(c.width for c in cols) + 6 * (len(cols) - 1)
        row = Image.new("RGB", (w, cols[0].height + 6), (15, 15, 18))
        x = 0
        for c in cols:
            row.paste(c, (x, 0))
            x += c.width + 6
        rows.append(row)
        record["frames"][fid] = rec
        st.preview = None
        print(fid, m["base"]["method"], {n: round(r["params"]["ev"], 2) for n, r in rec["presets"].items()}, flush=True)
    W = max(r.width for r in rows)
    sheet = Image.new("RGB", (W, sum(r.height for r in rows)), (15, 15, 18))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    Path(out_jpg).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_jpg, quality=88)
    Path(out_jpg).with_suffix(".json").write_text(json.dumps(record, indent=1, default=float))
    print("saved", out_jpg, sheet.size)


if __name__ == "__main__":
    args = sys.argv[1:]
    th = 300
    if any(a.startswith("--tile-h=") for a in args):
        th = int(next(a for a in args if a.startswith("--tile-h=")).split("=", 1)[1])
        args = [a for a in args if not a.startswith("--tile-h=")]
    main(args[0], args[1:], th)
