"""T4l multi-device probe: open each sample as the app does (State.open, standard preset) from a folder of one
symlink per sample (the material folder stays read-only), and record what the T4l task list claims about it.

  python -I scripts/t4l_probe_devices.py SRC OUT_JSON PATH [PATH ...]
  (T4l: SRC = the T4k tree, git archive e0d17d3, and src/; samples in 胶片负片素材/01_彩色负片_未反相: A1106.CR2,
  A1141.NEF, A1283.DNG, A1100.ARW, A1105.ARW, A222.NEF)

Per sample: open error (if any), estimate status / flags, automatic parameters, white point d_hi and its info, G density
span of the inversion, holder share and film_box, positive per-channel quantiles inside the image region outside the
holder, and the pixels where one channel is far past white while the middle channel is not (noise spikes).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

import filmcolor.app as fc_app  # noqa: E402
from filmcolor.frozen import check_source  # noqa: E402

ROLLS = Path("/tmp/fc_t4l/rolls")  # one folder of one symlink per sample


def main():
    out_json, paths = Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
    git = subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    res = dict(src=str(SRC), git_commit=git, samples={})
    for path in paths:
        check_source(path)  # frozen sets refused before any read
        d = ROLLS / path.stem
        d.mkdir(parents=True, exist_ok=True)
        link = d / path.name
        if not link.is_symlink():
            link.symlink_to(path)
        rec = dict(path_name=path.name)
        t0 = time.time()
        st = fc_app.State(d, Path("/tmp/fc_t4l/out"), allow_frozen=False)
        try:
            st.roll_info()
            m = st.open(path.name, "color")
        except Exception as e:  # noqa: BLE001 - recorded
            rec.update(error=f"{type(e).__name__}: {e}", seconds=round(time.time() - t0, 1))
            res["samples"][path.name] = rec
            print(path.name, rec["error"], flush=True)
            continue
        pv = st.preview
        auto = pv.auto or {}
        inv = pv.inversion
        P, msk, hold = pv.positive, pv.mask, pv.holder
        inside = msk & ~hold if hold is not None else msk
        x = P[inside].reshape(-1, 3)
        s = np.sort(x, axis=-1)
        spike = (s[:, 2] > 4.0) & (s[:, 1] <= 1.15)  # one channel four past white, middle channel not past it
        rec.update(
            base_method=m["base"]["method"], status=auto.get("status"), flags=auto.get("flags"),
            params={k: auto.get("params", {}).get(k) for k in ("ev", "temp", "tint", "contrast", "midtone", "blacks")},
            d_hi=list(inv["d_hi"]), d_lo=list(inv["d_lo"]), span_d_g=float(inv["d_hi"][1] - inv["d_lo"][1]),
            white_point=pv.report.get("white_point"), stats=auto.get("stats"),
            holder_share=float(hold.mean()) if hold is not None else None, film_box=m.get("film_box"),
            preview_shape=list(P.shape[:2]), mask_share=float(msk.mean()),
            positive_q=np.percentile(x, [50, 99, 99.9, 100], axis=0).round(4).tolist(),
            spike_px=int(spike.sum()), spike_share=float(spike.mean()), spike_channels=np.bincount(
                np.argmax(x[spike], axis=-1), minlength=3).tolist() if spike.any() else [0, 0, 0],
            camera=dict(model=fc_app._exif_info([link]).get(link, {}).get("model")),
            seconds=round(time.time() - t0, 1))
        res["samples"][path.name] = rec
        print(path.name, json.dumps({k: rec[k] for k in ("status", "flags", "params", "span_d_g", "holder_share",
                                                          "spike_px", "d_hi")}), flush=True)
        st.preview = None
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
