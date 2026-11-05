"""Reproduce the new T5b synthetic tests failing against isolated committed T5a source (red before green).
Usage: python -I scripts/t5b_baseline_test_evidence.py
The T5a tree lacks base.CLIPPED_BASE_MAX and the clipped-base rule, and its THUMB_REV is "t5a": the new
test_t5b_clipped_base_excluded and the THUMB_REV assertion of the T4y test must fail there. The other two new tests
assert behaviour T5b keeps from T5a (no near-saturated blocks; scanner path), so they must pass on T5a. Also records
the T5a estimate on the same synthetic blocks (a clipped base, max channel >= 0.99): behaviour, not only a name.
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t5b"


def main():
    baseline = json.loads((OUT / "baseline_source.json").read_text())
    src = Path(baseline["source_dir"])
    h = hashlib.sha256()
    for p in sorted(p for p in (src / "filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(src)).encode() + b"\0" + p.read_bytes() + b"\0")
    assert h.hexdigest() == baseline["src_content_sha256"]
    tests = [ROOT / f"tests/test_{name}.py" for name in ("base", "app")]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tests}
    code = ("import sys; sys.path.insert(0,sys.argv[1]); import filmcolor.base as b, filmcolor.app as a; "
            "print('baseline_base=' + b.__file__, 'baseline_app=' + a.__file__, flush=True); import pytest; "
            "raise SystemExit(pytest.main(sys.argv[2:]+['-k','t5b or t4y_app_scan','-q','-rf']))")
    cmd = [str(ROOT / ".venv/bin/python"), "-I", "-c", code, str(src), *map(str, tests)]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    text = result.stdout + result.stderr
    assert result.returncode == 1 and str(src / "filmcolor/base.py") in text and str(src / "filmcolor/app.py") in text
    failed = re.findall(r"^FAILED (\S+)", text, re.M)
    log = OUT / "baseline_test_failures.log"
    log.write_text(text)

    def count(word):
        match = re.search(r"(\d+) " + word, text)
        return int(match[1]) if match else 0
    assert count("failed") == 2 and count("passed") == 2, text
    assert set(failed) == {"tests/test_base.py::test_t5b_clipped_base_excluded",
                           "tests/test_app.py::test_t4y_app_scan_isolates_near_sat_between_camera_and_scanner"}, failed
    for p, digest in hashes.items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
    probe = (
        "import sys, json; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2])\n"
        "import filmcolor.base as b\n"
        "from test_base import _t5b_clipped_light_blocks\n"
        "sp, sat, film = _t5b_clipped_light_blocks()\n"
        "e = b.estimate_base_blocks(sp, sat, block=24)\n"
        "print(json.dumps(dict(rgb=[float(x) for x in e.rgb], near_sat=e.diagnostics.get('near_saturated_blocks_excluded'),"
        " clipped=e.diagnostics.get('clipped_base_excluded'), has_clipped_base_max=hasattr(b, 'CLIPPED_BASE_MAX'))))\n")
    r2 = subprocess.run([str(ROOT / ".venv/bin/python"), "-I", "-c", probe, str(src), str(ROOT / "tests")],
                        cwd=ROOT, capture_output=True, text=True, check=True)
    behaviour = json.loads(r2.stdout.strip().splitlines()[-1])
    assert max(behaviour["rgb"]) >= 0.99 and behaviour["near_sat"] == 400 and behaviour["clipped"] is None
    assert behaviour["has_clipped_base_max"] is False
    evidence = dict(command=cmd, baseline_source_sha256=h.hexdigest(), failed=count("failed"), passed=count("passed"),
                    deselected=count("deselected"), failed_tests=failed, t5a_behaviour=behaviour,
                    log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(), test_sha256=hashes)
    (OUT / "baseline_failure_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print({key: evidence[key] for key in ("failed", "passed", "deselected")}, behaviour)
    print("\n".join(failed))


if __name__ == "__main__":
    main()
