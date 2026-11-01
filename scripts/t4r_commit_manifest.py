"""List exact T4r local commit deliverables, never originals, images, logs or paths.py.
Usage: python -I scripts/t4r_commit_manifest.py
Read-only; staging remains an explicit separate action.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = ['src/filmcolor/base.py', 'src/filmcolor/invert.py', 'src/filmcolor/noref.py', 'src/filmcolor/app.py',
             'tests/test_base.py', 'tests/test_invert.py', 'tests/test_noref.py', 'tests/test_app.py',
             'experiments/t4_auto_ui/T4R_REPORT.md']
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'scripts').glob('t4r_*.py'))]
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT / 'experiments/t4_auto_ui/t4r').iterdir())
              if p.is_file() and p.suffix in ('.json', '.md', '.txt')]
    assert len(names) == len(set(names))
    for name in names:
        assert (ROOT / name).is_file(), name
        assert name != 'src/filmcolor/paths.py'
        print(name)


if __name__ == '__main__':
    main()
