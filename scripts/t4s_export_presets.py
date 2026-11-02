"""Export the five runtime NOREF_V3 configurations as reviewable JSON snapshots.
Usage: python -I scripts/t4s_export_presets.py [--check]
The application continues to use preset_config(); snapshots do not add a second runtime source.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filmcolor.noref import NOREF_V3, STYLE_PRESETS, preset_config


def main():
    target = ROOT / "configs"
    check = sys.argv[1:] == ["--check"]
    assert not sys.argv[1:] or check, "usage: t4s_export_presets.py [--check]"
    assert NOREF_V3["rev"] == "t4s-2026-11-02"
    for name in STYLE_PRESETS:
        cfg = preset_config(name)
        data = dict(source_config_rev=NOREF_V3["rev"], preset=name, config=cfg)
        p = target / ("noref_v3.json" if name == "standard" else f"noref_v3_{name}.json")
        text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if check:
            assert p.read_text() == text, f"stale preset snapshot: {p}"
        else:
            target.mkdir(exist_ok=True)
            assert not p.exists(), f"refuse overwriting existing snapshot: {p}"
            p.write_text(text)
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
