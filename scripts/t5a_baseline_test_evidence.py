"""Reproduce the new T5a synthetic tests failing against isolated committed T4z source (red before green).
Usage: python -I scripts/t5a_baseline_test_evidence.py
The T4z tree lacks holder.scene_highlight_components, preview.input_warnings / INPUT_* and app.ROLL_BASE_MAX; the
tests reach them through ``from ... import`` or attributes, so every test must fail (ImportError / AttributeError /
AssertionError are all behaviour that T4z does not have). The THUMB_REV assertion of the T4y test fails too.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t5a"


def main():
    baseline = json.loads((OUT / "baseline_source.json").read_text())
    src = Path(baseline["source_dir"])
    h = hashlib.sha256()
    for p in sorted(p for p in (src / "filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(src)).encode() + b"\0" + p.read_bytes() + b"\0")
    assert h.hexdigest() == baseline["src_content_sha256"]
    tests = [ROOT / f"tests/test_{name}.py" for name in ("preview", "app")]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tests}
    code = ("import sys; sys.path.insert(0,sys.argv[1]); import filmcolor.preview as pv, filmcolor.app as a; "
            "print('baseline_preview=' + pv.__file__, 'baseline_app=' + a.__file__, flush=True); import pytest; "
            "raise SystemExit(pytest.main(sys.argv[2:]+['-k','t5a or t4y_app_scan','-q','-rf']))")
    cmd = [str(ROOT / ".venv/bin/python"), "-I", "-c", code, str(src), *map(str, tests)]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    text = result.stdout + result.stderr
    assert result.returncode == 1 and str(src / "filmcolor/preview.py") in text and str(src / "filmcolor/app.py") in text
    failed = re.findall(r"^FAILED (\S+)", text, re.M)
    log = OUT / "baseline_test_failures.log"
    log.write_text(text)

    def count(word):
        match = re.search(r"(\d+) " + word, text)
        return int(match[1]) if match else 0
    assert count("failed") == 7 and count("passed") == 0, text
    for p, digest in hashes.items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
    # Behaviour of the T4z code itself on the same synthetic inputs (not only missing names): the window component is
    # painted HOLDER_GREY, and the frame whose roll base is past white is routed to roll_nat.
    probe = (
        "import sys, json, numpy as np; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]);\n"
        "from pathlib import Path\n"
        "from filmcolor.preview import Preview, HOLDER_GREY\n"
        "h = np.zeros((1000, 400), bool); h[:300, 200:] = True; h[600:, 200:] = True; h[300:360, 70:130] = True\n"
        "pv = Preview(path=Path('d'), sha256='x', factor=1, positive=np.full((1000, 400, 3), .5, np.float32),"
        " mask=np.ones((1000, 400), bool), image_region=(0, 0, 1000, 400), region_source='t', film='color',"
        " base_info={}, inversion={}, white_pct=99.9, report={}, base_inside_image=None, full_shape=(1000, 400), holder=h)\n"
        "e = pv._paint_holder(np.full((1000, 400, 3), .5, np.float32))\n"
        "import filmcolor.app as a\n"
        "st = a.State(Path(sys.argv[3]), Path(sys.argv[3]) / 'o', False)\n"
        "fe = dict(sub_roll='', roll_rgb=[1.30, 0.58, 0.87], own_rgb=[0.45, 0.2, 0.3], chroma_part=0.0, level_diff=0.46,"
        " own_error=None, estimate=dict(diagnostics={}))\n"
        "st.roll_cache['r'] = dict(frames={'x.dng': fe}, sub_rolls={'': dict(n_base=4, roll=dict(warnings=[]))})\n"
        "print(json.dumps(dict(window_painted_grey=bool(np.all(e[300:360, 70:130] == np.float32(HOLDER_GREY))),"
        " routed_bw=bool(st._route_roll_nat('r', 'x.dng', film='bw')), has_roll_base_max=hasattr(a, 'ROLL_BASE_MAX'))))\n")
    import tempfile
    with tempfile.TemporaryDirectory(prefix="fc_t5a_red_") as td:
        r2 = subprocess.run([str(ROOT / ".venv/bin/python"), "-I", "-c", probe, str(src), str(ROOT / "tests"), td],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    behaviour = json.loads(r2.stdout.strip().splitlines()[-1])
    assert behaviour == dict(window_painted_grey=True, routed_bw=True, has_roll_base_max=False), behaviour
    evidence = dict(command=cmd, baseline_source_sha256=h.hexdigest(), failed=count("failed"), passed=count("passed"),
                    deselected=count("deselected"), failed_tests=failed, t4z_behaviour=behaviour,
                    log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(), test_sha256=hashes)
    (OUT / "baseline_failure_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print({key: evidence[key] for key in ("failed", "passed", "deselected")})
    print("\n".join(failed))


if __name__ == "__main__":
    main()
