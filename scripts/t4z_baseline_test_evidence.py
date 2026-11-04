"""Reproduce new T4z synthetic failures against isolated committed T4y source."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4z"


def main():
    baseline = json.loads((OUT / "baseline_source.json").read_text())
    src = Path(baseline["source_dir"])
    h = hashlib.sha256()
    for p in sorted(p for p in (src / "filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(src)).encode() + b"\0" + p.read_bytes() + b"\0")
    assert h.hexdigest() == baseline["src_content_sha256"]
    tests = [ROOT / f"tests/test_{name}.py" for name in ("invert", "noref", "app")]
    tests = [p for p in tests if p.exists()]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tests}
    code = ("import sys; sys.path.insert(0,sys.argv[1]); import filmcolor.noref as n; "
            "n.NOREF_V3.update(overlift_night_bl_check=True, overlift_night_bl_min_weight=0.65, "
            "overlift_night_bl_max_bright_share=0.090, overlift_night_bl_ev_min=1.50, "
            "overlift_night_bl_q25_min=0.175); "
            "print('baseline_noref=' + n.__file__,flush=True); import pytest; "
            "raise SystemExit(pytest.main(sys.argv[2:]+['-k','t4z','-q']))")
    cmd = [str(ROOT / ".venv/bin/python"), "-I", "-c", code, str(src), *map(str, tests)]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    text = result.stdout + result.stderr
    assert result.returncode == 1 and str(src / "filmcolor/noref.py") in text
    failed = re.search(r"(\d+) failed", text)
    assert failed and int(failed[1]) >= 3, text
    log = OUT / "baseline_test_failures.log"
    log.write_text(text)
    def count(word):
        match = re.search(r"(\d+) " + word, text)
        return int(match[1]) if match else 0
    for p, digest in hashes.items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
    evidence = dict(command=cmd, baseline_source_sha256=h.hexdigest(),
                    config_overlay="New night gate keys supplied in memory so failures test missing behavior rather than KeyError; old source and thin thresholds unchanged",
                    failed=count("failed"),
                    passed=count("passed"), deselected=count("deselected"),
                    log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(), test_sha256=hashes)
    (OUT / "baseline_failure_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print({key: evidence[key] for key in ("failed", "passed", "deselected")})


if __name__ == "__main__":
    main()
