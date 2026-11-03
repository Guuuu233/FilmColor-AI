"""Run T4v synthetic tests against isolated T4u source with Python -I."""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4v"


def main():
    baseline = json.loads((OUT / "baseline_source.json").read_text())
    src = Path(baseline["source_dir"])
    digest = hashlib.sha256()
    for p in sorted(p for p in (src / "filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        digest.update(str(p.relative_to(src)).encode() + b"\0" + p.read_bytes() + b"\0")
    assert digest.hexdigest() == baseline["src_content_sha256"]
    test = ROOT / "tests/test_scanner.py"
    test_sha = hashlib.sha256(test.read_bytes()).hexdigest()
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import filmcolor.scanner as scanner; "
            "print('baseline_scanner=' + scanner.__file__, flush=True); "
            "import pytest; raise SystemExit(pytest.main([sys.argv[2], '-k', 't4v', '-q']))")
    cmd = [str(ROOT / ".venv/bin/python"), "-I", "-c", code, str(src), str(test)]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    text = result.stdout + result.stderr
    assert result.returncode == 1 and str(src / "filmcolor/scanner.py") in text
    failed = re.search(r"(\d+) failed", text)
    passed = re.search(r"(\d+) passed", text)
    deselected = re.search(r"(\d+) deselected", text)
    assert failed and int(failed[1]) >= 3 and passed and deselected
    assert hashlib.sha256(test.read_bytes()).hexdigest() == test_sha
    log = OUT / "baseline_test_failures.log"
    log.write_text(text)
    evidence = dict(command=cmd, baseline_source_sha256=digest.hexdigest(),
                    failed=int(failed[1]), passed=int(passed[1]), deselected=int(deselected[1]),
                    log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                    test_sha256={str(test.relative_to(ROOT)): test_sha})
    (OUT / "baseline_failure_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Baseline: {evidence['failed']} failed, {evidence['passed']} passed; source and test hashes verified")


if __name__ == "__main__":
    main()
