"""Output pins for the conversion paths not pinned elsewhere, and scripts/convert.py behaviour (M0 of
docs/SOFTWARE_PLAN_v1.md).

The pins are the outputs of the code as of 2026-10-18 (commit e102154), not physical ground truth: they make any
change to the default results visible. The default colour path on _DSF8948.RAF is pinned in test_real.py.
"""
import json
import sys
from pathlib import Path

import pytest

from filmcolor.noref import NOREF_V2
from filmcolor.pipeline import write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import convert as cli  # noqa: E402

from test_bw import bw_scan, write_dng as write_bw_dng  # noqa: E402
from test_scanner import scan, write_dng  # noqa: E402

RAF = ROOT / "data/raw/public/nlp_official/_DSF8948.RAF"
needs_raf = pytest.mark.skipif(not RAF.exists(), reason="NLP sample RAF not present")
SCAN_REGION = (20, 120, 200, 220)
PARAMS = dict(ev=0.3, temp=-0.2, tint=0.1, sat=1.2, contrast=1.3, midtone=0.3, shoulder=0.2)


def test_pin_scanner_colour_default(tmp_path):
    rgb, irl = scan()
    write_dng(tmp_path / "s.dng", rgb, irl)
    sc = write(tmp_path / "s.dng", tmp_path / "o", image_region=SCAN_REGION)
    assert sc["output"]["sha256"] == "9148a756cf980a476a8c2294608e4b3886cdb9ab1df38665d41d2ff86966c83f"


def test_pin_scanner_bw(tmp_path):
    write_bw_dng(tmp_path / "bw.dng", bw_scan())
    sc = write(tmp_path / "bw.dng", tmp_path / "o", film="bw", image_region=SCAN_REGION)
    assert sc["output"]["sha256"] == "1e219285d1824c07ca34be08049b854db71f0d82a7429a836a570d6b007e00d0"


@needs_raf
@pytest.mark.parametrize("kind,kwargs,digest", [
    ("auto", dict(auto=NOREF_V2), "f9b26d3c02b08c33d031d96944247ca83f131cafd986d432bf2918433966a60d"),
    ("bw", dict(film="bw"), "eef64c4cba27fbbb991fbac1d59a3572ba5927a0413877f2150f384fb3241037"),
    ("params", dict(render_params=PARAMS), "972ce161c3acabc232b3a9a1182804d86d3fbd750b8f48844cec690d3d470958"),
])
def test_pin_camera_paths(tmp_path, kind, kwargs, digest):
    assert write(RAF, tmp_path / kind, **kwargs)["output"]["sha256"] == digest


def test_cli_replay_refuses_another_input(tmp_path, capsys):
    raw = tmp_path / "x.RAF"
    raw.write_bytes(b"not the recorded file")
    side = tmp_path / "x.json"
    side.write_text(json.dumps(dict(input=dict(sha256="0" * 64))))
    with pytest.raises(SystemExit) as e:
        cli.main([str(tmp_path / "out"), str(raw), "--replay", str(side)])
    assert e.value.code == 2 and "SHA256 differs" in capsys.readouterr().err


def test_cli_batch_continues_after_a_failed_frame(tmp_path, capsys):
    bad = [tmp_path / "a.RAF", tmp_path / "b.RAF"]
    for p in bad:
        p.write_bytes(b"plain text, not a raw file")
    assert cli.main([str(tmp_path / "out")] + [str(p) for p in bad]) == 1
    out = capsys.readouterr().out
    assert out.count("FAILED") == 2 and "2 of 2 frame(s) failed" in out


def test_cli_converts_and_reports_qc(tmp_path, capsys):
    rgb, irl = scan()
    p = tmp_path / "s.dng"
    write_dng(p, rgb, irl)
    assert cli.main([str(tmp_path / "out"), str(p), "--image-region", ",".join(map(str, SCAN_REGION))]) == 0
    out = capsys.readouterr().out
    side = json.loads((tmp_path / "out" / "s.json").read_text())
    assert f"qc {side['qc']['status']}" in out
    assert side["output"]["sha256"] == "9148a756cf980a476a8c2294608e4b3886cdb9ab1df38665d41d2ff86966c83f"
    for c in side["qc"]["checks"]:
        assert (f"qc {c['name']} {c['status']}" in out) == (c["status"] != "pass")


def test_cli_roll_base_missing_frame_is_a_failed_frame(tmp_path, capsys):
    rgb, irl = scan()
    p = tmp_path / "s.dng"
    write_dng(p, rgb, irl)
    roll = tmp_path / "roll.json"
    roll.write_text(json.dumps(dict(frames={})))
    assert cli.main([str(tmp_path / "out"), str(p), "--roll-base", str(roll)]) == 1
    assert "KeyError" in capsys.readouterr().out
