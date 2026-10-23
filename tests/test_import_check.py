"""scripts/import_check.py: frozen data cannot enter the material folder under a new name, and every file needs a
real source and licence (T1b review 2026-10-22: seven renamed copies of sealed BlueNeg frames had been imported)."""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from import_check import TEMPLATE, check, main  # noqa: E402

from filmcolor.frozen import frozen_blocks  # noqa: E402


def row(file, **kw):
    return dict(file=file, original_name=kw.get("original_name", ""), roll=kw.get("roll", ""),
                source=kw.get("source", "group member Lee"), licence=kw.get("licence", TEMPLATE))


def staged(tmp_path, *names):
    for i, n in enumerate(names):
        (tmp_path / n).write_bytes(f"file {i}".encode())
    return tmp_path


def by_file(res):
    return {r["file"]: r for r in res}


def test_frozen_hash_is_refused_whatever_the_name(tmp_path):
    staged(tmp_path, "A2001.tif")
    sha = hashlib.sha256(b"file 0").hexdigest()
    res = by_file(check(tmp_path, [row("A2001.tif")], fz={sha: "frozen_test_x:1"}, blocks=dict(names={}, rolls={})))
    assert res["A2001.tif"]["refused"] == ["frozen by hash: frozen_test_x:1"]


def test_sealed_blueneg_name_or_roll_is_refused_under_a_new_file_name(tmp_path):
    staged(tmp_path, "A2002.tif", "A2003.tif", "A2004.tif")
    rows = [row("A2002.tif", original_name="19960815F-23-san-francisco.dng"),  # a sealed frame (frozen_test_v2_ds)
            row("A2003.tif", original_name="19960815F-99-not-in-release.dng"),  # unknown frame, sealed roll
            row("A2004.tif", original_name="19941218B-32-tokyo.dng")]  # development roll
    res = by_file(check(tmp_path, rows, fz={}, blocks=frozen_blocks()))
    assert any(w.startswith("frozen by name") for w in res["A2002.tif"]["refused"])
    assert any(w.startswith("frozen roll") for w in res["A2003.tif"]["refused"])
    assert res["A2004.tif"]["refused"] == []


def test_source_and_licence_must_be_real(tmp_path):
    staged(tmp_path, "a.NEF", "b.NEF", "c.NEF", "d.NEF")
    rows = [row("a.NEF", source="drive.google.com/file/d/xyz"),  # web link with the group template
            row("b.NEF", source=""), row("c.NEF", licence=""),
            row("d.NEF", source="https://discuss.pixls.us/t/1/2", licence="CC BY-SA 4.0 (post footer)")]
    res = by_file(check(tmp_path, rows + [row("missing.NEF")], fz={}, blocks=dict(names={}, rolls={})))
    assert "template licence" in res["a.NEF"]["refused"][0]
    assert res["b.NEF"]["refused"] == ["no source"] and res["c.NEF"]["refused"] == ["no licence"]
    assert res["d.NEF"]["refused"] == []
    assert res["missing.NEF"]["refused"] == ["row without a file"]


def test_unlisted_file_is_refused_and_copies_are_listed_not_refused(tmp_path):
    staged(tmp_path, "x.CR3", "y.CR3")
    sha_x = hashlib.sha256(b"file 0").hexdigest()
    res = by_file(check(tmp_path, [row("x.CR3")], fz={}, blocks=dict(names={}, rolls={}),
                        material={sha_x: "01_彩色负片_未反相/A0001.CR3"}))
    assert res["x.CR3"]["refused"] == [] and res["x.CR3"]["already_in_material"] == "01_彩色负片_未反相/A0001.CR3"
    assert res["y.CR3"]["refused"] == ["no row in the sources list"]


def test_command_exit_status(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    staged(d, "ok.NEF")
    csv_ok = tmp_path / "ok.csv"
    csv_ok.write_text("file,original_name,roll,source,licence\nok.NEF,DSC_1.NEF,,group member Lee,"
                      f"{TEMPLATE}\n", encoding="utf-8")
    assert main([str(d), str(csv_ok)]) == 0
    csv_bad = tmp_path / "bad.csv"
    csv_bad.write_text("file,original_name,roll,source,licence\nok.NEF,19960815F-23-san-francisco.dng,,x,y\n",
                       encoding="utf-8")
    assert main([str(d), str(csv_bad)]) == 1
