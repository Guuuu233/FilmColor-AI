"""Tests on the user's own roll ultramax400_r01 (Nikon Z 7_2 NEF, Bayer, no film border). Skipped when
the files are absent.

Expected values are regression baselines observed on 2026-10-14 (experiments/own_roll01), not physical
ground truth.
"""
from pathlib import Path

import exifread
import numpy as np
import pytest

from filmcolor.base import estimate_base_auto, estimate_base_roll
from filmcolor.raw import linearize, saturation_mask
from filmcolor.rawpy_io import load_raw

SRC = Path(__file__).resolve().parents[1] / "data/raw/own/ultramax400_r01"
FILES = sorted(SRC.glob("*.NEF")) if SRC.is_dir() else []
pytestmark = pytest.mark.skipif(len(FILES) != 37, reason="ultramax400_r01 NEFs not present")


def _exposure(path):
    with open(path, "rb") as fh:
        r = exifread.process_file(fh, details=False)["EXIF ExposureTime"].values[0]
    return r.num / r.den


def _auto(path):
    f = load_raw(path)
    lin = linearize(f)
    return estimate_base_auto(lin, f.colors, saturation_mask(f), f.pattern.shape[0], crop=f.crop)


def test_roll_base_rejects_dense_first_frame():
    # 3847 / 3823: most transparent frames of the roll; 3822: uniformly dense first frame (fogged leader,
    # hypothesis), once taken for a base candidate
    names = ["DSC_3847", "DSC_3823", "DSC_3853", "DSC_3822"]
    paths = {n: SRC / f"{n}.NEF" for n in names}
    cand = {n: _auto(p).rgb for n, p in paths.items()}
    t = {n: _exposure(p) for n, p in paths.items()}
    r = estimate_base_roll(cand, t, t_ref=1 / 30)
    assert "DSC_3822" not in r.members
    assert sorted(r.members) == ["DSC_3823", "DSC_3847", "DSC_3853"]
    assert r.diagnostics["frames"]["DSC_3822"]["density_above_floor"][1] > 0.4
    np.testing.assert_allclose(r.diagnostics["cluster_chroma_log2_RG_BG"], [0.0, -1.31], atol=0.03)
    assert not r.warnings
