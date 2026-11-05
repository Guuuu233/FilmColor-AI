"""List T5a deliverables only after all formal acceptance gates pass."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "experiments/t4_auto_ui/t5a"
    assert json.loads((out / "acceptance.json").read_text())["all_passed"] is True
    names = [
        "src/filmcolor/app.py",
        "src/filmcolor/app_static/index.html",
        "src/filmcolor/holder.py",
        "src/filmcolor/pipeline.py",
        "src/filmcolor/preview.py",
        "tests/test_app.py",
        "tests/test_preview.py",
        "scripts/t4d_source_eval.py",
        "experiments/t4_auto_ui/t5a/T5A_REPORT.md",
    ]
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "scripts").glob("t5a_*.py"))]
    names += [str(p.relative_to(ROOT)) for p in sorted(out.iterdir()) if p.is_file() and p.suffix in (".json", ".md", ".txt")]
    names = sorted(set(names))
    for name in names:
        assert (ROOT / name).is_file() and name != "src/filmcolor/paths.py", name
        print(name)


if __name__ == "__main__":
    main()
