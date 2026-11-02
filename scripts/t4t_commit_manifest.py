"""List T4t deliverables only; exclude user paths.py, originals, images and logs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = ["src/filmcolor/invert.py", "src/filmcolor/noref.py", "src/filmcolor/app.py",
             "tests/test_invert.py", "tests/test_noref.py", "experiments/t4_auto_ui/T4T_REPORT.md"]
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "scripts").glob("t4t_*.py"))]
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "configs").glob("noref_v3*.json"))]
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "experiments/t4_auto_ui/t4t").iterdir())
              if p.is_file() and p.suffix in (".json", ".md", ".txt")]
    assert len(names) == len(set(names))
    for name in names:
        assert (ROOT / name).is_file(), name
        assert name != "src/filmcolor/paths.py"
        print(name)


if __name__ == "__main__":
    main()
