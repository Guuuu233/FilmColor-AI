"""List exact T4s deliverables; exclude originals, images, logs, paths.py and interrupted checkpoints."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = ["src/filmcolor/scanner.py", "src/filmcolor/invert.py", "src/filmcolor/app.py",
             "src/filmcolor/noref.py", "src/filmcolor/pipeline.py", "tests/test_scanner.py",
             "tests/test_invert.py", "tests/test_app.py", "experiments/t4_auto_ui/T4S_REPORT.md"]
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "scripts").glob("t4s_*.py"))]
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "configs").glob("noref_v3*.json"))]
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "experiments/t4_auto_ui/t4s").iterdir())
              if p.is_file() and p.suffix in (".json", ".md", ".txt")]
    assert len(names) == len(set(names))
    for name in names:
        assert (ROOT / name).is_file(), name
        assert name != "src/filmcolor/paths.py"
        print(name)


if __name__ == "__main__":
    main()
