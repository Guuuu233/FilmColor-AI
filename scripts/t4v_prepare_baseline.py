"""Extract committed T4u sources, preserving the recorded local paths.py override.
Usage: python -I scripts/t4v_prepare_baseline.py NEW_DIRECTORY
No images are copied or decoded. Does not modify the checkout.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "98ded3c"


def main():
    target = Path(sys.argv[1]).resolve()
    assert not target.exists(), "choose a new directory"
    target.mkdir(parents=True)
    files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", COMMIT, "src/filmcolor", "data/manifests"],
                                    cwd=ROOT, text=True).splitlines()
    for name in files:
        p = target / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(subprocess.check_output(["git", "show", f"{COMMIT}:{name}"], cwd=ROOT))
    paths = target / "src/filmcolor/paths.py"
    original_sha = hashlib.sha256(paths.read_bytes()).hexdigest()
    local = (ROOT / "src/filmcolor/paths.py").read_bytes()
    paths.write_bytes(local)
    h = hashlib.sha256()
    for p in sorted(p for p in (target / "src/filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(target / "src")).encode() + b"\0" + p.read_bytes() + b"\0")
    result = dict(commit=subprocess.check_output(["git", "rev-parse", COMMIT], cwd=ROOT, text=True).strip(),
                  source_dir=str(target / "src"), src_content_sha256=h.hexdigest(),
                  paths_runtime_sha256=hashlib.sha256(local).hexdigest(), paths_committed_sha256=original_sha,
                  n_files=len(files))
    (target / "baseline_source.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
