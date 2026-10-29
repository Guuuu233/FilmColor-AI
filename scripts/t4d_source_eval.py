"""T4d cross-source evaluation: the app's automatic starting point (NOREF_V3 variants / style presets) against the
reference conversions of the four development sources of data/manifests/s3_dev_frames.json (nlp, shop, blueneg,
ext_own; 191 frames, none in a frozen set - every file is checked again with filmcolor.frozen.check_source).

  t4d_source_eval.py SRC_DIR OUT_JSON [--domains=nlp,shop,...] [--ids=ID,...] [--variants=JSON] [--max-side=1000]

SRC_DIR: the ``src`` folder whose ``filmcolor`` package is used (a ``git worktree`` of an earlier commit, or this
repo's src), so before / after are measured with the same script.

``--variants``: JSON object ``name -> spec`` (default ``{"default": {}}``). A spec may hold ``preset`` (a name of
``noref.STYLE_PRESETS``: start from ``noref.preset_config``), ``cfg`` (NOREF_V3 entries overridden), ``cam_mix``
(camera matrix share; default the config's ``camera_matrix_mix``, else ``app.CAMERA_MATRIX_MIX``) and
``display_levels`` (T4i, measurement only, not in the app: ``{"lo": 1.0, "hi": 99.2, "mix": 0.8}`` stretches each
channel of the rendered display image from its ``lo`` .. ``hi`` percentile over the preview mask to 0 .. 1, the
per-channel points mixed ``mix`` : ``1 - mix`` with the same percentiles of the channel mean).

Per frame: the frame is opened as the app opens it - ``State.open`` with the automatic base choice (per-frame B0, or
``roll_nat`` when B0 failed or deviates from the roll) on a roll folder of symlinks to the manifest's frames of that
roll (``/tmp/fc_t4d/rolls/<domain>/<roll>``; only listed frames, so no unlisted file is opened). Each variant is
estimated on the same positive exactly as ``load_preview`` does (``estimation_input`` of the full-resolution positive,
camera RAW: camera matrix at the variant's share; scanner DNG: identity) and rendered on the 1000 px preview,
converted to CIELAB D50 with the export's sRGB profile. The reference: ICC-tagged TIFF / JPEG via
``compare.load_lab``; BlueNeg prints (no ICC) read as sRGB (``eval_blueneg.print_lab``, an assumption). It is
block-averaged to the preview's size and aligned once, on the L* of the default render (``RenderParams()``, the same
in every run and code version, the positive being the same): camera frames with
``compare.align`` (scales 0.80..1.03, as t4c_ref_eval.py), BlueNeg with ``eval_blueneg.register`` (rotation, mirror,
scale). Metrics: ``compare.metrics`` per variant on the same aligned grid, plus the reference's and our median L*
and C*. Summaries leave out frames whose alignment NCC is below 0.7 (eval_blueneg.NCC_MIN, fixed before T4d).
"""
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(SRC))
sys.path.insert(1, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import filmcolor  # noqa: E402
import filmcolor.app as fc_app  # noqa: E402
from filmcolor import noref  # noqa: E402
from filmcolor.colorimetry import MatrixTRCProfile, xyz_to_lab  # noqa: E402
from filmcolor.compare import align, aligned_pair, block_mean, load_lab, metrics  # noqa: E402
from filmcolor.export import icc_profile  # noqa: E402
from filmcolor.frozen import check_source  # noqa: E402
from filmcolor.render import RenderParams, camera_color_matrix  # noqa: E402
from eval_blueneg import NCC_MIN, pair as pair_blueneg, print_lab, register as register_blueneg  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data/manifests/s3_dev_frames.json"
ROLLS = Path("/tmp/fc_t4d/rolls")
SCALES = np.arange(0.80, 1.0301, 0.01)


def src_content_sha256(src: Path) -> str:
    """SHA256 over (relative path, bytes) of every tracked-type file of the package: equal values = same code, so a
    run on an uncommitted tree can be matched to the commit made afterwards."""
    h = hashlib.sha256()
    for f in sorted(p for p in (src / "filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(f.relative_to(src)).encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def variant_cfg(spec: dict) -> dict:
    cfg = noref.preset_config(spec["preset"]) if spec.get("preset") else dict(noref.NOREF_V3)
    over = spec.get("cfg") or {}
    unknown = set(over) - set(cfg)
    assert not unknown, unknown
    return dict(cfg, **{k: tuple(v) if isinstance(v, list) else v for k, v in over.items()})


def roll_state(f: dict, states: dict, max_side: int, roll_folder: Path | None = None):
    """State on a folder of symlinks to the manifest's frames of ``f``'s roll (built once per roll); with
    ``roll_folder`` (T4k), a State on that folder itself for every frame (e.g. nlp_official, a collection)."""
    if roll_folder is not None:
        if "folder" not in states:
            states["folder"] = fc_app.State(roll_folder, Path("/tmp/fc_t4d/out"), allow_frozen=False, max_side=max_side)
            states["folder"].roll_info()
        return states["folder"]
    key = (f["domain"], f["roll"])
    if key not in states:
        d = ROLLS / f["domain"] / f["roll"]
        d.mkdir(parents=True, exist_ok=True)
        for g in FRAMES:
            if (g["domain"], g["roll"]) == key:
                link = d / Path(g["raw"]).name
                if not link.is_symlink():
                    link.symlink_to((REPO / g["raw"]).resolve() if not os.path.isabs(g["raw"]) else Path(g["raw"]))
        st = fc_app.State(d, Path("/tmp/fc_t4d/out"), allow_frozen=False, max_side=max_side)
        st.roll_info()
        states[key] = st
    return states[key]


def display_levels(disp, mask, lo=1.0, hi=99.2, mix=0.8):
    x = disp[mask].reshape(-1, 3)
    q_lo, q_hi = np.percentile(x, lo, axis=0), np.percentile(x, hi, axis=0)
    y = x.mean(1)
    y_lo, y_hi = np.percentile(y, lo), np.percentile(y, hi)
    a, b = mix * q_lo + (1 - mix) * y_lo, mix * q_hi + (1 - mix) * y_hi
    return np.clip((disp - a) / np.maximum(b - a, 1e-3), 0.0, 1.0).astype(np.float32)


def lab_of(prof, disp):
    return xyz_to_lab(prof.to_xyz(disp.astype(np.float64)))


def med_lc(lab):
    x = lab.reshape(-1, 3)
    return float(np.median(x[:, 0])), float(np.median(np.hypot(x[:, 1], x[:, 2])))


def main(out_json: str, opts: list[str]):
    domains, ids, variants, max_side, app_config = None, None, {"default": {}}, 1000, {}
    roll_folder, base_mode = None, None
    for opt in opts:
        k, v = opt.split("=", 1)
        if k == "--domains":
            domains = set(v.split(","))
        elif k == "--ids":
            ids = set(v.split(","))
        elif k == "--variants":
            variants = json.loads(v)
        elif k == "--max-side":
            max_side = int(v)
        elif k == "--app-config":  # T4k: module constants of filmcolor.app for an ablation run (e.g. routing off)
            app_config = json.loads(v)
        elif k == "--roll-folder":  # T4k: open every frame from this folder (its own roll / sub-roll analysis)
            roll_folder = (REPO / v).resolve()
        elif k == "--base-mode":  # T4k: open with this base mode instead of the app's automatic choice
            base_mode = v
        else:
            raise SystemExit(f"unknown option {k}")
    assert Path(filmcolor.__file__).resolve().is_relative_to(SRC), filmcolor.__file__
    commit = subprocess.run(["git", "-C", str(SRC), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(SRC), "status", "--porcelain", "--", "."], capture_output=True,
                                text=True).stdout.strip())
    for name, value in app_config.items():
        assert hasattr(fc_app, name), name
        setattr(fc_app, name, tuple(value) if isinstance(value, list) else value)
    cfgs = {name: variant_cfg(spec) for name, spec in variants.items()}
    # T4k: the estimator takes the inversion's G density range (load_preview passes it); older sources do not
    span_ok = "span_d_g" in inspect.signature(noref.estimate_params_v2).parameters
    prof = MatrixTRCProfile(icc_profile("srgb"))
    sel = [f for f in FRAMES if (domains is None or f["domain"] in domains) and (ids is None or f["id"] in ids)]
    res = dict(src=str(SRC), git_commit=commit, src_dirty=dirty, src_content_sha256=src_content_sha256(SRC), manifest=str(MANIFEST.relative_to(REPO)),
               environment=dict(python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform()),
               variants=variants, ncc_min=NCC_MIN, max_side=max_side, app_config=app_config,
               roll_folder=str(roll_folder.relative_to(REPO)) if roll_folder else None, base_mode=base_mode, frames={})
    if Path(out_json).exists():
        previous = json.loads(Path(out_json).read_text())
        for key in ("src_content_sha256", "variants", "ncc_min", "max_side", "manifest", "app_config", "roll_folder",
                    "base_mode"):
            if previous.get(key) != res[key]:
                raise ValueError(f"cannot resume {out_json}: {key} changed")
        res["frames"] = previous["frames"]
    states = {}
    for f in sel:
        if f["id"] in res["frames"]:
            continue
        t0 = time.time()
        raw = REPO / f["raw"] if not os.path.isabs(f["raw"]) else Path(f["raw"])
        ref = REPO / f["ref"] if not os.path.isabs(f["ref"]) else Path(f["ref"])
        check_source(raw), check_source(ref)
        st = roll_state(f, states, max_side, roll_folder)
        try:
            m = st.open(raw.name, "color", base_mode=base_mode)
        except Exception as e:  # noqa: BLE001 - recorded, frame left out
            res["frames"][f["id"]] = dict(domain=f["domain"], roll=f["roll"], error=f"{type(e).__name__}: {e}")
            print(f["id"], "open failed:", e, flush=True)
            continue
        pv = st.preview
        P4, mask4 = noref.estimation_input(pv.full_positive, pv.image_region)
        cam = st._raw_meta(pv.path).get("rgb_xyz_matrix") if st.camera_matrix(pv.path) is not None else None
        out = dict(domain=f["domain"], roll=f["roll"], base_method=m["base"]["method"], base=m["base"],
                   camera_matrix=cam is not None,
                   holder_share=float(pv.holder.mean()) if getattr(pv, "holder", None) is not None else None,
                   d_hi=list(pv.inversion["d_hi"]), white_point=pv.report.get("white_point"),
                   raw_sha256=hashlib.file_digest(raw.open("rb"), "sha256").hexdigest(),
                   ref_sha256=hashlib.file_digest(ref.open("rb"), "sha256").hexdigest(), variants={})
        ref_small = al = None
        lab0 = lab_of(prof, pv.render(RenderParams()))  # alignment on the default render: the same in every run
        for name, cfg in cfgs.items():
            mix = variants[name].get("cam_mix", cfg.get("camera_matrix_mix", getattr(fc_app, "CAMERA_MATRIX_MIX", 0.0)))
            cfg_v = dict(cfg, matrix=camera_color_matrix(cam, mix)) if cam is not None else cfg
            t1 = time.time()
            span = dict(span_d_g=float(pv.inversion["d_hi"][1] - pv.inversion["d_lo"][1])) if span_ok else {}
            est = noref.estimate_params_v2(P4, mask4, cfg_v, **span)
            est_s = time.time() - t1
            disp = pv.render(est["params"])
            if variants[name].get("display_levels"):
                disp = display_levels(disp, pv.mask, **variants[name]["display_levels"])
            lab = lab_of(prof, disp)
            if al is None:  # align once, on lab0
                if f["ref_kind"] == "srgb_no_icc":
                    ref_full, lf = print_lab(ref), 1
                else:
                    lf = 1 if ref.suffix.lower() in (".tif", ".tiff") else 2  # JPEG scans / edits read at half size
                    try:
                        ref_full = load_lab(ref, factor=lf)
                    except ValueError as e:  # e.g. a JPEG without its ICC profile: left out, not guessed
                        out["error"] = f"reference not read: {e}"
                        break
                # reference block factor: a full-frame print / scan matches our preview by its long side; a crop at
                # sensor resolution (NLP's positives) by the preview's own factor. Both are tried, best NCC kept.
                fbs = {max(1, int(round(max(ref_full.shape[:2]) / max(lab0.shape[:2]))))}
                if pv.factor % lf == 0:
                    fbs.add(int(pv.factor) // lf)
                for fb in sorted(fbs):
                    small = block_mean(ref_full, fb) if fb > 1 else ref_full
                    try:
                        if f["ref_kind"] == "srgb_no_icc":
                            cand = register_blueneg(lab0[..., 0], small[..., 0])
                            cand_d, ncc = dict(cand, method="eval_blueneg.register"), cand["ncc"]
                        else:
                            cand = align(lab0[..., 0], small[..., 0], 1, scales=SCALES)
                            cand_d, ncc = dict(cand.to_dict(), method="compare.align"), cand.ncc
                    except (ValueError, TypeError):  # reference does not fit at this factor
                        continue
                    if al is None or ncc > out["alignment"]["ncc"]:
                        al, ref_small, out["alignment"] = cand, small, dict(cand_d, ref_block_factor=fb)
                del ref_full
                if al is None:
                    out["error"] = "reference could not be aligned"
                    break
            if f["ref_kind"] == "srgb_no_icc":
                a, r = pair_blueneg(lab, ref_small, al)
            else:
                a, r = aligned_pair(lab, ref_small, al)
            mt = metrics(a, r)
            mt.pop("L_percentiles", None)
            (oL, oC), (rL, rC) = med_lc(a), med_lc(r)
            out["variants"][name] = dict(params=est["params"], flags=est.get("flags"), status=est.get("status"),
                                         backlit=est.get("diagnostics", {}).get("backlit_check"),
                                         wb_cool_cast=est.get("diagnostics", {}).get("wb", {}).get("diagnostics", {})
                                         .get("cool_cast"),
                                         t4k={k: est.get("diagnostics", {}).get(k) for k in ("thin_negative", "tone_pixels")}
                                         | dict(border_cells=est.get("diagnostics", {}).get("textured_cells", {})
                                                .get("border_cells"),
                                                gw_cool_undamped=est.get("diagnostics", {}).get("wb", {})
                                                .get("diagnostics", {}).get("gw_cool_undamped")),
                                         estimate_seconds=round(est_s, 3), cam_mix=mix if cam is not None else None,
                                         metrics=mt, ours_L_median=oL, ours_C_median=oC, ref_L_median=rL,
                                         ref_C_median=rC)
        if "error" in out:
            res["frames"][f["id"]] = out
            print(f["id"], out["error"], flush=True)
            st.preview = None
            continue
        out["ncc"] = float(out["alignment"]["ncc"])
        out["seconds"] = round(time.time() - t0, 1)
        res["frames"][f["id"]] = out
        print(f["id"], out["base_method"], "ncc %.3f" % out["ncc"], " | ".join(
            f"{n} dE {v['metrics']['de2000_median']:.2f} dL {v['metrics']['mean_dL']:+.1f}" for n, v in out["variants"].items()),
            flush=True)
        st.preview = None
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(out_json).write_text(json.dumps(res, indent=1, default=float))  # partial results survive an abort
    Path(out_json).write_text(json.dumps(res, indent=1, default=float))


FRAMES = json.loads(MANIFEST.read_text())["frames"]

if __name__ == "__main__":
    main(sys.argv[2], sys.argv[3:])
