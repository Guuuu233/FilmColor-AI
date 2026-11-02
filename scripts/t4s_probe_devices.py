"""T4s guarded device evidence. Originals read-only; no reference colour claims.
Usage: python -I scripts/t4s_probe_devices.py SRC baseline|candidate
"""
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]).resolve()
LABEL = sys.argv[2]
assert LABEL in ("baseline", "candidate")
sys.path.insert(0, str(SRC))
import filmcolor.app as app
from filmcolor.frozen import check_source
from filmcolor.scanner import load_scanner

OUT = ROOT / "experiments/t4_auto_ui/t4s"
MATERIAL = Path("/Volumes/waizhiyingpan/胶片负片素材")
TIFFS = ("A1310.TIF", "A304.tif", "A1301.TIF", "A255.tif", "A1027.TIF", "A1028.TIF", "B032.tif", "B034.tiff")


def source_sha():
    h = hashlib.sha256()
    for p in sorted(p for p in (SRC / "filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(SRC)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def source(name):
    folder = "02_黑白负片_未反相" if name.startswith("B") else "01_彩色负片_未反相"
    p = MATERIAL / folder / name
    return p, check_source(p, allow_frozen=False)


def links(group, paths):
    folder = Path("/tmp/fc_t4s_devices") / LABEL / group
    folder.mkdir(parents=True, exist_ok=True)
    for p in paths:
        link = folder / p.name
        if not link.exists():
            link.symlink_to(p)
        assert link.is_symlink() and link.resolve() == p.resolve()
    assert sorted(p.name for p in folder.iterdir()) == sorted(p.name for p in paths)
    return folder


def record(st, name, film):
    m = st.open(name, film)
    pv = st.preview
    pos = pv.positive
    h, w = pos.shape[:2]
    center = pos[int(.2*h):int(.8*h), int(.2*w):int(.8*w)]
    band = pos[int(.1*h):int(.2*h), int(.2*w):int(.8*w)]
    before = hashlib.sha256(pos.tobytes()).hexdigest()
    display = pv.render(pv.auto["params"])
    out = dict(status=pv.auto["status"], flags=pv.auto.get("flags"), params=pv.auto["params"],
               base=m["base"], span_d_g=float(pv.inversion["d_hi"][1]-pv.inversion["d_lo"][1]),
               preview_shape=list(pos.shape), full_shape=list(pv.full_shape),
               center_p99=np.percentile(center, 99, axis=(0, 1)).tolist(),
               upper_band_p99=np.percentile(band, 99, axis=(0, 1)).tolist(),
               full_positive_finite=bool(np.isfinite(pv.full_positive).all()),
               render_preserves_positive=before == hashlib.sha256(pos.tobytes()).hexdigest(),
               spike_clamp=pv.report.get("spike_clamp"), diagnostics=pv.auto.get("diagnostics"))
    if name == "A1296.CR2":
        from PIL import Image
        img = OUT / f"a1296_{LABEL}.jpg"
        Image.fromarray(app.to8(display)).save(img)
        out["preview_image"] = img.name
    st.preview = None
    del pv, pos, center, band, display
    gc.collect()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"devices_{LABEL}.json"
    result = dict(src_content_sha256=source_sha(), label=LABEL, a1296={}, acros={}, tiff={})
    if target.exists():
        result = json.loads(target.read_text())
        assert result["src_content_sha256"] == source_sha(), "source changed"

    def save():
        assert result["src_content_sha256"] == source_sha(), "source changed during probe"
        target.write_text(json.dumps(result, indent=2) + "\n")

    p, sha = source("A1296.CR2")
    if not result["a1296"]:
        st = app.State(links("a1296", [p]), OUT / "exports", False)
        st.roll_info()
        result["a1296"] = dict(raw_sha256=sha, **record(st, p.name, "color"))
        save()
        print(LABEL, "A1296", result["a1296"]["upper_band_p99"], flush=True)

    # Membership is the actual existing NEFs within the task's explicit B002..B029 range.
    paths = [MATERIAL / "02_黑白负片_未反相" / f"B{i:03d}.NEF" for i in range(2, 30)]
    paths = [p for p in paths if p.is_file()]
    assert len(paths) == 28, [p.name for p in paths]  # all existing members; task's claimed 24 has no supplied subset
    hashes = {p.name: check_source(p, allow_frozen=False) for p in paths}
    st = app.State(links("acros", paths), OUT / "exports", False)
    ri = st.roll_info()
    result["acros_roll"] = dict(members=[p.name for p in paths], sub_rolls=ri["sub_rolls"], mixed_rolls=ri["mixed_rolls"])
    for p in paths:
        if p.name in result["acros"]:
            assert result["acros"][p.name]["raw_sha256"] == hashes[p.name]
            continue
        try:
            r = record(st, p.name, "bw")
            fe = ri["frames"][p.name]
            r.update(level_diff=fe["level_diff"], chroma_part=fe["chroma_part"])
        except Exception as e:
            r = dict(error=f"{type(e).__name__}: {e}")
        result["acros"][p.name] = dict(raw_sha256=hashes[p.name], **r)
        save()
        print(LABEL, p.name, r.get("status", r.get("error")), r.get("span_d_g"), flush=True)

    if LABEL == "candidate":
        for name in ("A1027.TIF", *[n for n in TIFFS if n != "A1027.TIF"]):
            p, sha = source(name)
            if name in result["tiff"]:
                assert result["tiff"][name]["raw_sha256"] == sha
                continue
            try:
                fr = load_scanner(p, allow_frozen=False)
                meta = {k: fr.meta.get(k) for k in ("bits", "model", "infrared_layer", "monochrome_dng")}
                del fr
                st = app.State(links(p.stem, [p]), OUT / "exports", False)
                st.roll_info()
                r = record(st, name, "bw" if name.startswith("B") else "color")
                r["scanner"] = meta
            except Exception as e:
                r = dict(error=f"{type(e).__name__}: {e}")
            result["tiff"][name] = dict(raw_sha256=sha, **r)
            save()
            print(LABEL, name, r.get("status", r.get("error")), flush=True)
    save()


if __name__ == "__main__":
    assert os.environ.get("FILMCOLOR_DATA_ROOT") == "/Volumes/waizhiyingpan/filmcolor_data"
    main()
