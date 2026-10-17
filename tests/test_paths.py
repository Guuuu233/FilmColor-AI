import pytest

from filmcolor.paths import ENV, DataRootError, data_dir, data_root


def test_data_root_must_be_set_and_exist(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV, raising=False)
    with pytest.raises(DataRootError, match="not set"):
        data_root()
    missing = tmp_path / "unmounted"
    monkeypatch.setenv(ENV, str(missing))
    with pytest.raises(DataRootError, match="does not exist"):
        data_dir("s3_ml", "cache")
    assert not missing.exists()  # nothing created when the root is missing
    monkeypatch.setenv(ENV, str(tmp_path))
    d = data_dir("s3_ml", "cache")
    assert d == tmp_path / "s3_ml" / "cache" and d.is_dir() and data_root() == tmp_path
