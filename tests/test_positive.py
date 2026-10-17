"""pipeline.positive: the scene-linear positive that convert renders (S3 model input definition)."""
import numpy as np

from filmcolor.pipeline import convert, positive
from filmcolor.render import RenderParams, render
from test_bw import bw_scan, write_dng


def test_convert_renders_exactly_the_positive(tmp_path):
    path = tmp_path / "scan.dng"
    write_dng(path, bw_scan())
    img, sc = convert(path)
    p = positive(path)
    disp, _ = render(p["positive"], RenderParams(), p["mask"])
    assert np.array_equal(disp, img)
    assert p["inversion"].params.to_dict() == sc["inversion"]["params"]
    assert p["inversion"].params.exposure == 1.0 and p["positive"].dtype == np.float32
    assert list(p["image_region"]) == sc["image_region"]["region"]
