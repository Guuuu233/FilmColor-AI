"""Verify T4y provenance and the exact commit manifest without reading original images."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/t4_auto_ui/t4y"
COMMITS = {
    "t4o": "4cd76b7",
    "t4r": "fbd48a0",
    "t4s": "7c5107c",
    "t4t": "567aaf9",
    "t4u": "98ded3c",
    "t4v": "0d1d730",
    "t4w": "b3a7510",
    "t4x": "f6340aa",
}
PATHS_SHA = "fced78f38a0787e473d882024d289878b99a224c604242f6561c7fb05e8be913"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_sha():
    h = hashlib.sha256()
    for p in sorted(p for p in (ROOT / "src/filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        h.update(str(p.relative_to(ROOT / "src")).encode() + b"\0" + p.read_bytes() + b"\0")
    return h.hexdigest()


def main():
    assert git("branch", "--show-current").decode().strip() == "t4y"
    assert git("rev-parse", "main").decode().strip() == git("rev-parse", "f6340aa").decode().strip()
    assert sha(ROOT / "src/filmcolor/paths.py") == PATHS_SHA
    protocol = json.loads((OUT / "input_protocol_sha256.json").read_text())
    for rel, digest in protocol["historical_json_sha256"].items():
        version = Path(rel).parts[2]
        data = (ROOT / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        assert data == git("show", f"{COMMITS[version]}:{rel}"), rel
    for rel, digest in protocol["frozen_manifest_sha256"].items():
        assert sha(ROOT / rel) == digest, rel
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    acceptance = json.loads((OUT / "acceptance.json").read_text())
    assert acceptance["all_passed"] and acceptance["devices"]["failures"] == []

    # Verify source SHA matches acceptance recording
    fingerprint = source_sha()
    assert acceptance["source_sha256"] == fingerprint, "acceptance source_sha256 mismatch with current working tree"

    # Verify script_sha256 matches disk
    scripts_record = json.loads((OUT / "script_sha256.json").read_text())
    for rel, digest in scripts_record.items():
        assert sha(ROOT / rel) == digest, f"script {rel} modified after report calculation"

    files = subprocess.check_output(
        [str(ROOT / ".venv/bin/python"), "-I", str(ROOT / "scripts/t4y_commit_manifest.py")],
        cwd=ROOT,
        text=True,
    ).splitlines()
    assert "src/filmcolor/paths.py" not in files
    assert all(Path(f).suffix in (".py", ".json", ".md", ".txt") for f in files)
    changed = set(git("diff", "--name-only", "f6340aa").decode().splitlines())
    assert changed <= set(files) | {"src/filmcolor/paths.py"}, changed - set(files)
    staged = set(git("diff", "--cached", "--name-only").decode().splitlines())
    assert not staged or staged == set(files), staged ^ set(files)
    for f in files:
        if Path(f).suffix == ".json":
            text = (ROOT / f).read_text().lower()
            assert "@" not in text, f
            assert not any(word in text for word in ('"serial_number"', '"bodyserialnumber"', '"cameraserialnumber"')), f
    print(
        json.dumps(
            dict(
                all_passed=True,
                manifest_files=len(files),
                historical_json_verified=len(protocol["historical_json_sha256"]),
                frozen_manifests_verified=len(protocol["frozen_manifest_sha256"]),
                paths_sha256=PATHS_SHA,
                source_sha256=fingerprint,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
