"""Synchronise the five reviewable runtime preset snapshots for T4u.
Usage: python -I scripts/t4u_export_presets.py [--check]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filmcolor.noref import NOREF_V3, STYLE_PRESETS, preset_config


def main():
    check = sys.argv[1:] == ["--check"]
    assert not sys.argv[1:] or check, "usage: t4u_export_presets.py [--check]"
    assert NOREF_V3["rev"] == "t4u-2026-11-03"
    for name in STYLE_PRESETS:
        p = ROOT / "configs" / ("noref_v3.json" if name == "standard" else f"noref_v3_{name}.json")
        old = json.loads(p.read_text())
        assert old["preset"] == name
        assert old["source_config_rev"] in ("t4t-2026-11-02", NOREF_V3["rev"])
        text = json.dumps(dict(source_config_rev=NOREF_V3["rev"], preset=name, config=preset_config(name)),
                          indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if check:
            assert p.read_text() == text, f"stale preset snapshot: {p}"
        else:
            p.write_text(text)
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
