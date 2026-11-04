"""Guarded T4y real-image comparison; images are diagnostics, not colour references.
Usage: python -I scripts/t4y_probe_devices.py SRC baseline|candidate [--targets-only]
"""
import gc
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]).resolve()
LABEL = sys.argv[2]
assert LABEL in ("baseline", "candidate")
assert sys.argv[3:] in ([], ["--targets-only"])
sys.path.insert(0, str(SRC))
from filmcolor.app import State, to8
from filmcolor.frozen import check_source
from filmcolor.holder import film_box
from filmcolor.scanner import load_scanner

OUT = ROOT / "experiments/t4_auto_ui/t4y"
MATERIAL = Path("/Volumes/waizhiyingpan/胶片负片素材")
TARGETS = ("A305.tif", "A1040.TIF", "A1041.TIF", "A232.NEF", "A1069.NEF", "A239.ORF", "A281.RAF", "A1006.CR2")
TIFFS = ("A1310.TIF", "A304.tif", "A1301.TIF", "A255.tif", "A1027.TIF", "A1028.TIF", "B032.tif", "B034.tiff")
NEW_TIFFS = ("A1228.tif", "A229.TIF", "A655.tif", "A849.tif", "A850.tif", "A1093.tif", "B036.tif")
NEW_TARGETS = ("A1004.DNG", "A1005.CR2", "A1007.CR2", "A1033.CR2", "A262.arw", "A263.arw", "A1061.CR2", "A237.cr2")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_sha():
    h = hashlib.sha256()
    for p in sorted(p for p in (SRC / "filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(SRC)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def source(name):
    p = MATERIAL / ("02_黑白负片_未反相" if name.startswith("B") else "01_彩色负片_未反相") / name
    return p, check_source(p, allow_frozen=False)


def record(st, name, film, image=False):
    scanner = None
    if Path(name).suffix.lower() in (".tif", ".tiff"):
        source_path, _ = source(name)
        frame = load_scanner(source_path, allow_frozen=False)
        scanner = {k: frame.meta.get(k) for k in ("bits", "tiff_eotf", "prestretched_reconstruction", "single_channel_prestretched")}
        del frame
        gc.collect()
    meta = st.open(name, film)
    pv = st.preview
    pos = pv.positive
    h, w = pos.shape[:2]
    center = pos[int(.2*h):int(.8*h), int(.2*w):int(.8*w)]
    band = pos[int(.1*h):int(.2*h), int(.2*w):int(.8*w)]
    before = hashlib.sha256(pos.tobytes()).hexdigest()
    disp = pv.render(pv.auto["params"])
    fb = film_box(pv.holder)
    core = disp if fb is None else disp[fb[0]:fb[0]+fb[2], fb[1]:fb[1]+fb[3]]
    out = dict(status=pv.auto["status"], flags=pv.auto["flags"], params=pv.auto["params"],
               base=meta["base"], inversion=pv.inversion, report=pv.report, diagnostics=pv.auto["diagnostics"],
               config_rev=pv.auto.get("config_rev"), preview_shape=list(pos.shape), full_shape=list(pv.full_shape),
               center_p99=np.percentile(center, 99, axis=(0, 1)).tolist(),
               upper_band_p99=np.percentile(band, 99, axis=(0, 1)).tolist(),
               upper_band_max=band.max(axis=(0, 1)).tolist(),
               holder_share=float(pv.holder.mean()), film_box=fb,
               srgb_q50=np.median(disp, axis=(0, 1)).tolist(), core_srgb_q50=np.median(core, axis=(0, 1)).tolist(),
               full_positive_finite=bool(np.isfinite(pv.full_positive).all()),
               render_preserves_positive=before == hashlib.sha256(pos.tobytes()).hexdigest())
    if scanner is not None:
        out["scanner"] = scanner
    scan_diag = (st.frame_scans.get(name) or {}).get("estimate", {}).get("diagnostics") or {}
    base_diag = {}
    for k in ("near_saturated_blocks_excluded", "bare_backlight_excluded"):
        if k in scan_diag:
            base_diag[k] = scan_diag[k]
    if base_diag:
        out["base_diagnostics"] = base_diag
    if name == "A1296.CR2":
        full = pv.full_positive
        fh, fw = full.shape[:2]
        full_band = full[int(.1*fh):int(.2*fh), int(.2*fw):int(.8*fw)]
        out["full_upper_band_share_any_channel_gt_10"] = float((full_band > 10).any(-1).mean())
        out["full_upper_band_max"] = full_band.max(axis=(0, 1)).tolist()
    if image:
        img = Image.fromarray(to8(disp))
        img.thumbnail((900, 900))
        target = OUT / f"{Path(name).stem}_{LABEL}.jpg"
        img.save(target, quality=92)
        out["preview_image"] = target.name
        out["preview_image_sha256"] = sha(target)
    st.preview = None
    del pv, pos, center, band, disp, core
    gc.collect()
    return out


def main():
    assert os.environ.get("FILMCOLOR_DATA_ROOT") == "/Volumes/waizhiyingpan/filmcolor_data"
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"devices_{LABEL}.json"
    fingerprint = source_sha()
    script_sha = sha(__file__)
    result = dict(label=LABEL, src=str(SRC), src_content_sha256=fingerprint, script_sha256=script_sha,
                  max_side=1600, targets={}, new_targets={}, a1296={}, acros={}, tiff={}, new_tiff={})
    if target.exists():
        result = json.loads(target.read_text())
        assert result["src_content_sha256"] == fingerprint, "source changed"
        assert result["script_sha256"] == script_sha, "probe script changed"

    def save():
        assert source_sha() == fingerprint, "source changed during probe"
        assert sha(__file__) == script_sha, "script changed during probe"
        target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    with tempfile.TemporaryDirectory(prefix=f"fc_t4y_{LABEL}_") as temp:
        def state(group, paths):
            folder = Path(temp) / group
            folder.mkdir()
            for p in paths:
                (folder / p.name).symlink_to(p)
            st = State(folder, OUT / "exports", False, max_side=1600)
            ri = st.roll_info()
            return st, ri

        for group, names in (("a1296", ("A1296.CR2",)), ("targets", TARGETS),
                             ("new_targets", NEW_TARGETS),
                             ("new_tiff", NEW_TIFFS),
                             ("acros", tuple(f"B{i:03d}.NEF" for i in range(2, 30))), ("tiff", TIFFS)):
            if sys.argv[3:] and group in ("acros", "tiff"):
                continue
            paths, hashes = [], {}
            for name in names:
                p, digest = source(name)
                paths.append(p)
                hashes[name] = digest
            if group == "acros":
                st, ri = state(group, paths)
                result["acros_roll"] = dict(members=list(names), sub_rolls=ri["sub_rolls"], mixed_rolls=ri["mixed_rolls"])
            for p in paths:
                old = result[group].get(p.name) if group != "a1296" else result[group]
                if old:
                    assert old["raw_sha256"] == hashes[p.name]
                    continue
                try:
                    if group != "acros":
                        st, ri = state(group + "_" + p.stem, [p])
                    r = record(st, p.name, "bw" if p.name.startswith("B") else "color", image=group in ("a1296", "targets", "new_targets", "new_tiff"))
                except Exception as e:
                    r = dict(error=f"{type(e).__name__}: {e}")
                r["raw_sha256"] = hashes[p.name]
                if group == "a1296":
                    result[group] = r
                else:
                    result[group][p.name] = r
                save()
                print(LABEL, p.name, r.get("status", r.get("error")), r.get("inversion", {}).get("d_hi"), flush=True)
                st.preview = None
                gc.collect()
    save()


if __name__ == "__main__":
    main()
