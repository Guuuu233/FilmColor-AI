import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from filmcolor.frozen import (FrozenDataError, check_hashes, check_name, check_roll, check_source, frozen_blocks,
                              frozen_hashes, load_cache, save_cache, sha256_file)
from filmcolor.scanner import load_scanner

ROOT = Path(__file__).resolve().parents[1]
FROZEN_DIRS = ("Eastman 5219(ecn-2次抛冲洗 尼康数字化)-", "Eastman 5207(ecn-2次抛冲洗 尼康数字化)-2")
DEV_DIRS = ("Kodak G200（尼康数字化）-", "Eastman 5207(ecn-2次抛冲洗 尼康数字化)-",
            "Eastman 5207(ecn-2次抛冲洗 尼康数字化)-1", "Eastman 5207(ecn-2次抛冲洗 尼康数字化)-3")


@pytest.fixture
def frozen_file(tmp_path):
    f = tmp_path / "held_out.NEF"
    f.write_bytes(b"held-out negative")
    m = tmp_path / "frozen_test_x.json"
    m.write_text(json.dumps(dict(id="frozen_test_x", files=[dict(id="A1", sha256=sha256_file(f))])))
    return f, frozen_hashes([m])


def test_frozen_file_is_refused_also_as_copy_rename_or_symlink(frozen_file, tmp_path):
    f, frozen = frozen_file
    copy = tmp_path / "sub" / "renamed.dat"
    copy.parent.mkdir()
    shutil.copy(f, copy)
    link = tmp_path / "link.NEF"
    os.symlink(f, link)
    for p in (f, copy, link):
        with pytest.raises(FrozenDataError, match="frozen_test_x:A1"):
            check_source(p, frozen=frozen)
    assert check_source(f, frozen=frozen, allow_frozen=True) == sha256_file(f)
    other = tmp_path / "dev.NEF"
    other.write_bytes(b"development negative")
    assert check_source(other, frozen=frozen) == sha256_file(other)


def test_caches_carry_provenance_and_are_refused_when_derived_from_frozen_data(frozen_file, tmp_path):
    f, frozen = frozen_file
    sha = sha256_file(f)
    save_cache(tmp_path / "x_P8.npy", np.ones(3), [sha])
    with pytest.raises(FrozenDataError):
        load_cache(tmp_path / "x_P8.npy", frozen=frozen)
    assert load_cache(tmp_path / "x_P8.npy", frozen=frozen, allow_frozen=True).sum() == 3
    save_cache(tmp_path / "y_P8.npy", np.zeros(2), ["0" * 64])
    assert load_cache(tmp_path / "y_P8.npy", frozen=frozen).shape == (2,)
    np.save(tmp_path / "bare.npy", np.zeros(1))  # written without provenance
    with pytest.raises(FrozenDataError, match="without source record"):
        load_cache(tmp_path / "bare.npy", frozen=frozen)
    with pytest.raises(ValueError, match=".npy"):
        save_cache(tmp_path / "z_P8", np.zeros(1), [])


def test_real_manifest_covers_the_frozen_rolls_and_misses_every_development_roll():
    frozen = frozen_hashes([ROOT / "data/manifests/frozen_test_v1.json"])
    inv = json.loads((ROOT / "experiments/s3_data_audit/inventory.json").read_text())
    in_frozen_dirs = [r for r in inv if r.get("orig_dir") in FROZEN_DIRS]
    in_dev_dirs = [r for r in inv if r.get("orig_dir") in DEV_DIRS]
    assert len(frozen) == 144 and len(in_frozen_dirs) == 144
    everything = frozen_hashes()  # default: every frozen manifest (photo-folder addendum, 2026-10-18 dataset holdout)
    # + 289 group-pack files of the sealed BlueNeg rolls + 475 further official files (LFS sha256, not downloaded)
    # + 7 copies of sealed BlueNeg frames added to the material folder under new names (addendum, 2026-10-22, T1b)
    assert len(everything) == 144 + 71 + 289 + 475 + 7 and set(frozen) <= set(everything)
    add = json.loads((ROOT / "data/manifests/frozen_test_v2_ds_addendum_20261022.json").read_text())["files"]
    with pytest.raises(FrozenDataError, match="frozen_test_v2_ds_addendum_20261022"):
        check_hashes([add[0]["sha256"]], frozen=everything)
    dev = json.loads((ROOT / "data/manifests/s3_dev_frames.json").read_text())["frames"]
    check_hashes([h for f in dev for h in (f["raw_sha256"], f["ref_sha256"])], frozen=everything)  # does not raise
    assert all(r["sha256"] in frozen for r in in_frozen_dirs)
    assert len(in_dev_dirs) == 154 + 134 + 7  # negatives, prints, Camera Raw sidecars of the 4 rolls
    check_hashes([r["sha256"] for r in in_dev_dirs], frozen=frozen)  # does not raise
    with pytest.raises(FrozenDataError, match="frozen_test_v1"):
        check_hashes([r["sha256"] for r in in_dev_dirs + in_frozen_dirs[:1]], frozen=frozen)
    m = json.loads((ROOT / "data/manifests/frozen_test_v1.json").read_text())
    assert sum(r["scored_pairs"] for r in m["rolls"]) == 69


def test_sealed_blueneg_rolls_are_refused_by_name_and_roll_before_any_download(tmp_path):
    blocks = frozen_blocks()
    assert len(blocks["names"]) == 164
    sealed = "19960815F-23-san-francisco"  # roll 199608F, sealed in frozen_test_v2_ds
    for p in (f"negative-16bit/blue-corrupted/{sealed}.dng", f"printed-16bit/blue-corrupted/{sealed}.tif",
              f"https://huggingface.co/x/resolve/main/negative-preview-8bit/{sealed}.preview.png"):
        with pytest.raises(FrozenDataError, match="by name"):
            check_name(p)
    check_name(p, allow_frozen=True)  # final-test entry point only
    for r in ("ds-199608F", "19960815F", "19960814F"):
        with pytest.raises(FrozenDataError, match="held-out test roll"):
            check_roll(r)
    check_roll("19960808B")  # dev roll 199608B: allowed
    f = tmp_path / f"{sealed}.dng"  # not a real scan: the name alone must stop the reader and the hasher
    f.write_bytes(b"x")
    with pytest.raises(FrozenDataError):
        load_scanner(f)
    with pytest.raises(FrozenDataError):
        check_source(f)
    # nothing on the development side is blocked: dev manifest frames, the BlueNeg download selection
    dev = json.loads((ROOT / "data/manifests/s3_dev_frames.json").read_text())["frames"]
    for fr in dev:
        check_name(fr["raw"])
        check_roll(fr["roll"])
    sel = json.loads((ROOT / "experiments/s3_blueneg/selection_intact_dev.json").read_text())["frames"]
    for fr in sel:
        check_name(fr["negative_path"])
        check_name(fr["printed_path"])
