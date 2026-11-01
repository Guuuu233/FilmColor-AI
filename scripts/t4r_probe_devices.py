"""T4r device checks using the actual app path, with source and input hashes and frozen guard.
Usage: python -I scripts/t4r_probe_devices.py SRC OUT_JSON PATH [PATH ...]
No reference matching: status, automatic parameters, RGBI metadata, holder display and export invariants only.
"""
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(SRC))
import filmcolor.app as app
from filmcolor.frozen import check_source
from filmcolor.render import RenderParams
from filmcolor.scanner import is_scanner_dng, load_scanner

ROOT = Path(__file__).resolve().parents[1]
ROLLS = Path("/tmp/fc_t4r_devices/rolls")


def source_sha():
    h = hashlib.sha256()
    for p in sorted(p for p in (SRC / "filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(SRC)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def main():
    target = Path(sys.argv[2])
    film = __import__("os").environ.get("FILMCOLOR_PROBE_FILM", "color")
    out = dict(src=str(SRC), src_content_sha256=source_sha(), samples={})
    if target.exists():
        previous = json.loads(target.read_text())
        if previous["src_content_sha256"] != out["src_content_sha256"]:
            raise ValueError(f"cannot resume {target}: source changed")
        out["samples"] = previous["samples"]
    for arg in sys.argv[3:]:
        p = Path(arg).resolve()
        sha = check_source(p, allow_frozen=False)
        if p.name in out["samples"]:
            assert out["samples"][p.name]["raw_sha256"] == sha
            continue
        rec = dict(path_name=p.name, raw_sha256=sha, film=film)
        folder = ROLLS / p.stem
        folder.mkdir(parents=True, exist_ok=True)
        link = folder / p.name
        if not link.is_symlink():
            link.symlink_to(p)
        st = app.State(folder, Path("/tmp/fc_t4r_devices/out"), allow_frozen=False)
        pv = None
        t0 = time.monotonic()
        try:
            st.roll_info()
            m = st.open(p.name, film)
            pv = st.preview
            hold = pv.holder
            auto = pv.auto or {}
            rec.update(status=auto.get("status"), flags=auto.get("flags"), params=auto.get("params"),
                       diagnostics=auto.get("diagnostics"), d_hi=pv.inversion["d_hi"], d_lo=pv.inversion["d_lo"],
                       span_d_g=float(pv.inversion["d_hi"][1] - pv.inversion["d_lo"][1]),
                       base=pv.base_info, scan=st.frame_scans.get(p.name),
                       full_positive_finite=bool(np.isfinite(pv.full_positive).all()),
                       white_point=pv.report.get("white_point"), holder_share=float(hold.mean()),
                       film_box=m.get("film_box"), preview_shape=list(pv.positive.shape[:2]))
            inside = pv.mask & ~hold
            x = pv.positive[inside]
            s = np.sort(x, axis=-1)
            rec["spike_px"] = int(((s[:, 2] > 4) & (s[:, 1] <= 1.15)).sum())
            rec["positive_q"] = np.percentile(x, [50, 99, 99.9, 100], axis=0).tolist() if len(x) else []
            before = hashlib.sha256(pv.positive.tobytes()).hexdigest()
            display = pv.render(RenderParams(**auto["params"]))
            after = hashlib.sha256(pv.positive.tobytes()).hexdigest()
            rec["render_preserves_positive"] = before == after
            rec["display_rgb_q50"] = np.median(display[inside], axis=0).tolist() if inside.any() else []
            image_path = target.parent / (target.stem + "_" + p.stem + ".jpg")
            Image.fromarray(np.round(np.clip(np.rot90(display, m["rot90"]), 0, 1) * 255).astype(np.uint8)).save(image_path)
            rec["preview_image"] = image_path.name
            if hold.mean() >= 0.15 and m.get("film_box"):
                r, c, h, w = m["film_box"]
                exterior = np.ones(hold.shape, bool)
                exterior[r:r+h, c:c+w] = False
                rec["outer_pixels"] = int(exterior.sum())
                rec["outer_display_grey_max_error"] = float(np.abs(display[exterior] - 0.02).max()) if exterior.any() else 0.0
            del x, s, display
            st.preview = None
            pv = None
            gc.collect()
            if is_scanner_dng(p):
                scanner = load_scanner(p, allow_frozen=False)
                rec["scanner"] = {k: scanner.meta[k] for k in ("infrared_layer", "linearization_table_len", "model", "bits", "monochrome_dng")
                                  if k in scanner.meta}
                del scanner
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["seconds"] = round(time.monotonic() - t0, 2)
        out["samples"][p.name] = rec
        print(p.name, rec.get("error") or {k: rec.get(k) for k in ("status", "span_d_g", "holder_share", "film_box")}, flush=True)
        st.preview = None
        pv = None
        del st
        gc.collect()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
