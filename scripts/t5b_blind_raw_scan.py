"""T5b §6: camera-RAW film base, T5a vs T5b estimate_base_blocks on the same blocks of every blind-tested file.
Usage: python -I scripts/t5b_blind_raw_scan.py BASELINE_SRC BATCH19_21_JSON BATCH22_JSON

Each file is checked with check_source(allow_frozen=False) first (frozen files are listed, not decoded) and read
once the way State._scan_one reads it (scanner DNG: load_scanner + block_means; camera RAW: load_raw + superpixel,
24 px blocks, decoder crop); both estimators get the same sp / sp_sat with exclude_near_saturated = not is_scan.
The T5a estimator is BASELINE_SRC/filmcolor/base.py (git 1c4c9aa), loaded beside this tree's filmcolor; the other
modules it imports (raw) are checked to be byte-identical. Required set: the camera RAWs of Batch 19-22 plus
A1298 / A1300; extended set: every camera-RAW file of the two negative folders and A7R5_RAW. No images are written.
"""
import gc
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import filmcolor.base as cand_base  # noqa: E402
from filmcolor.frozen import FrozenDataError, check_source  # noqa: E402
from filmcolor.raw import linearize, saturation_mask, superpixel  # noqa: E402
from filmcolor.rawpy_io import load_raw  # noqa: E402
from filmcolor.scanner import block_means, is_scanner_dng, load_scanner  # noqa: E402

OUT = ROOT / "experiments/t4_auto_ui/t5b"
MATERIAL = Path("/Volumes/waizhiyingpan/胶片负片素材")
FOLDERS = ("01_彩色负片_未反相", "02_黑白负片_未反相", "A7R5_RAW")
RAW_SUFFIXES = {".nef", ".nrw", ".raf", ".cr2", ".cr3", ".arw", ".rw2", ".orf", ".pef", ".srw", ".dng"}
EXPECTED_CHANGED = ("01_彩色负片_未反相/A1298.ARW", "01_彩色负片_未反相/A1299.ARW", "01_彩色负片_未反相/A1300.ARW")
DIAG = ("near_saturated_blocks_excluded", "bare_backlight_excluded", "clipped_base_excluded", "mode_seed",
        "relaxed_uniformity", "n_kept")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_sha(src):
    h = hashlib.sha256()
    for p in sorted(p for p in (src / "filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(src)).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def baseline_module(src):
    """T5a base.py as ``filmcolor.base_t5a``: its relative imports resolve to this tree's (identical) modules."""
    committed = subprocess.check_output(["git", "show", "1c4c9aa:src/filmcolor/base.py"], cwd=ROOT)
    assert (src / "filmcolor/base.py").read_bytes() == committed, "baseline base.py is not T5a's"
    assert (src / "filmcolor/raw.py").read_bytes() == (ROOT / "src/filmcolor/raw.py").read_bytes()
    spec = importlib.util.spec_from_file_location("filmcolor.base_t5a", src / "filmcolor/base.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["filmcolor.base_t5a"] = mod
    spec.loader.exec_module(mod)
    assert not hasattr(mod, "CLIPPED_BASE_MAX") and cand_base.CLIPPED_BASE_MAX == 0.99
    return mod


def summary(est):
    d = est.diagnostics
    return dict(rgb=[float(x) for x in est.rgb], max_channel=float(max(est.rgb)),
                diag={k: d.get(k) for k in DIAG if k in d}, warnings=list(est.warnings))


def main():
    base_src = Path(sys.argv[1]).resolve()
    batch_a, batch_b = Path(sys.argv[2]), Path(sys.argv[3])
    assert len(sys.argv) == 4
    t5a = baseline_module(base_src)
    rows = {}
    for name, rs in json.loads(batch_a.read_text())["batches"].items():
        for r in rs:
            rows.setdefault(r["rel"], set()).add(name.split("_")[0])
    for r in json.loads(batch_b.read_text())["frames"]:
        rows.setdefault(r["rel"], set()).add("batch22")
    required = sorted({rel for rel in rows if Path(rel).suffix.lower() in RAW_SUFFIXES and rel.startswith("01_")}
                      | {"01_彩色负片_未反相/A1298.ARW", "01_彩色负片_未反相/A1300.ARW"})
    extended = sorted(str(p.relative_to(MATERIAL)) for d in FOLDERS for p in (MATERIAL / d).rglob("*")
                      if p.is_file() and p.suffix.lower() in RAW_SUFFIXES and not p.name.startswith("."))
    assert set(required) <= set(extended)
    target = OUT / "blind_raw_scan.json"
    fingerprint, base_fp, script_sha = source_sha(ROOT / "src"), source_sha(base_src), sha(__file__)
    inputs = dict(batch19_21=dict(path=str(batch_a), sha256=sha(batch_a)), batch22=dict(path=str(batch_b), sha256=sha(batch_b)))
    result = dict(src_content_sha256=fingerprint, baseline_src_content_sha256=base_fp, script_sha256=script_sha,
                  inputs=inputs, required=required, n_extended=len(extended), files={})
    if target.exists():
        old = json.loads(target.read_text())
        for k in ("src_content_sha256", "baseline_src_content_sha256", "script_sha256", "inputs", "required"):
            assert old[k] == result[k], f"cannot resume: {k} changed"
        result = old

    def save():
        assert source_sha(ROOT / "src") == fingerprint and sha(__file__) == script_sha
        target.write_text(json.dumps(result, indent=1, ensure_ascii=False, allow_nan=False) + "\n")

    for i, rel in enumerate(required + [r for r in extended if r not in set(required)]):  # required set first
        if rel in result["files"]:
            continue
        p = MATERIAL / rel
        rec = dict(groups=sorted(rows.get(rel, ())), required=rel in required)
        try:
            rec["raw_sha256"] = check_source(p, allow_frozen=False)
        except FrozenDataError as e:
            rec["frozen"] = str(e)
            result["files"][rel] = rec
            save()
            print(i, rel, "frozen, not decoded", flush=True)
            continue
        try:
            is_scan = is_scanner_dng(p, allow_frozen=False)
            if is_scan:
                fr = load_scanner(p, allow_frozen=False)
                sp, sp_sat = block_means(fr.rgb, fr.sat, 24)
                crop = None
                del fr
            else:
                f = load_raw(p)
                sp, sp_sat = superpixel(linearize(f), f.colors, saturation_mask(f), f.pattern.shape[0], 24)
                crop = f.crop
                del f
            rec["is_scan"] = is_scan
            out = {}
            for label, mod in (("t5a", t5a), ("t5b", cand_base)):
                try:
                    est = mod.estimate_base_blocks(sp, sp_sat, block=24, crop=crop, exclude_near_saturated=not is_scan)
                    out[label] = (summary(est), json.dumps(est.to_dict(), sort_keys=True))
                except ValueError as e:
                    out[label] = (dict(error=str(e)), str(e))
            rec["t5a"], rec["t5b"] = out["t5a"][0], out["t5b"][0]
            rec["identical"] = out["t5a"][1] == out["t5b"][1]
            del sp, sp_sat
        except Exception as e:  # noqa: BLE001  (decoder limits are the same for both versions)
            rec["error"] = f"{type(e).__name__}: {e}"
        result["files"][rel] = rec
        save()
        print(i, rel, rec.get("identical", rec.get("error")), rec.get("t5b", {}).get("rgb"), flush=True)
        gc.collect()
    files = result["files"]
    changed = sorted(k for k, v in files.items() if v.get("identical") is False)
    result["summary"] = dict(
        files=len(files), compared=sum("identical" in v for v in files.values()),
        frozen_not_decoded=sum("frozen" in v for v in files.values()),
        decode_errors=sorted(k for k, v in files.items() if "error" in v),
        changed=changed, unexpected_changes=sorted(set(changed) - set(EXPECTED_CHANGED)),
        missing_expected=sorted(set(EXPECTED_CHANGED) - set(changed)),
        required_compared=sum("identical" in files[k] for k in required),
        required_changed=sorted(k for k in required if files[k].get("identical") is False),
        clipped_hits=sorted(k for k, v in files.items() if (v.get("t5b") or {}).get("diag", {}).get("clipped_base_excluded")),
        t5a_base_max_ge_099=sorted(k for k, v in files.items() if (v.get("t5a") or {}).get("max_channel", 0) >= 0.99))
    save()
    print(json.dumps(result["summary"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
