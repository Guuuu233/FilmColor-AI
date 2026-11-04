"""List T4y deliverables only after all formal acceptance gates pass."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "experiments/t4_auto_ui/t4y"
    assert json.loads((out / "acceptance.json").read_text())["all_passed"] is True
    names = [
        "src/filmcolor/render.py",
        "src/filmcolor/base.py",
        "src/filmcolor/scanner.py",
        "src/filmcolor/app.py",
        "src/filmcolor/noref.py",
        "tests/test_render.py",
        "tests/test_scanner.py",
        "tests/test_noref.py",
        "tests/test_app.py",
        "experiments/t4_auto_ui/T4Y_REPORT.md",
    ]
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "scripts").glob("t4y_*.py"))]
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "configs").glob("noref_v3*.json"))]
    names += [str(p.relative_to(ROOT)) for p in sorted(out.iterdir()) if p.is_file() and p.suffix in (".json", ".md", ".txt")]
    assert len(names) == len(set(names))
    for name in names:
        assert (ROOT / name).is_file() and name != "src/filmcolor/paths.py", name
        print(name)


if __name__ == "__main__":
    main()
