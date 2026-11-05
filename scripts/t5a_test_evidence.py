"""Extract successful full pytest evidence from its local log without committing logs.
Usage: python -I scripts/t5a_test_evidence.py
First run: /usr/bin/time -l .venv/bin/python -I -m pytest -q > experiments/t4_auto_ui/t5a/tests.log 2>&1
"""
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t5a"


def main():
    log = OUT / "tests.log"
    text = log.read_text()
    match = re.search(r"(\d+) passed in ([\d.]+)s", text)
    assert match and "FAILED " not in text and "ERROR " not in text, "pytest did not complete successfully"
    peak = re.findall(r"(\d+)\s+peak memory footprint", text)
    assert peak
    h = hashlib.sha256()
    for p in sorted(p for p in (ROOT / "src/filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(ROOT / "src")).encode() + b"\0" + p.read_bytes() + b"\0")
    evidence = dict(command="/usr/bin/time -l .venv/bin/python -I -m pytest -q",
                    passed=int(match[1]), seconds=float(match[2]), peak_memory_footprint_bytes=max(map(int, peak)),
                    log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(), source_sha256=h.hexdigest(),
                    test_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT / "tests").rglob("*.py"))})
    result = subprocess.run([str(ROOT / ".venv/bin/python"), "-I", "-m", "pytest", "--collect-only", "-q"],
                            cwd=ROOT, check=True, capture_output=True, text=True)
    count = re.search(r"(\d+) tests collected", result.stdout)
    assert count and int(count[1]) == evidence["passed"], result.stdout
    (OUT / "test_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f'{evidence["passed"]} passed in {evidence["seconds"]:.2f}s; collection matches; '
          f'peak {evidence["peak_memory_footprint_bytes"] / 2**30:.2f} GiB')


if __name__ == "__main__":
    main()
