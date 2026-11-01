"""List the exact T4p/T4q deliverables to include in the local T4q commit.
Usage: python -I scripts/t4q_commit_manifest.py
Read-only: never stages files, raw material, images, logs or the user's paths.py change.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = ["src/filmcolor/app.py", "src/filmcolor/noref.py", "src/filmcolor/scanner.py",
             "tests/test_noref.py", "tests/test_scanner.py", "scripts/t4d_source_eval.py",
             "experiments/t4_auto_ui/T4P_REPORT.md", "experiments/t4_auto_ui/T4Q_REPORT.md"]
    for version in ("t4p", "t4q"):
        names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "scripts").glob(f"{version}_*.py"))]
        folder = ROOT / f"experiments/t4_auto_ui/{version}"
        names += [str(p.relative_to(ROOT)) for p in sorted(folder.iterdir())
                  if p.is_file() and p.suffix in (".json", ".md", ".txt")]
    assert len(names) == len(set(names))
    for name in names:
        assert (ROOT / name).is_file(), name
        assert name != "src/filmcolor/paths.py"
        print(name)


if __name__ == "__main__":
    main()
