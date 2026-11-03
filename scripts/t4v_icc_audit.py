"""Inspect named non-frozen TIFF ICC headers and curve policy; no source metadata is copied."""
import hashlib
import json
import os
import sys
from pathlib import Path

import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filmcolor.frozen import check_source
from filmcolor.scanner import _main_page, _parse_tiff_eotf, _tag

NAMES = ("A1040.TIF", "A1041.TIF", "A1228.tif", "A229.TIF", "A655.tif", "A849.tif", "A850.tif", "A1093.tif", "B036.tif",
         "A1310.TIF", "A304.tif", "A1301.TIF", "A255.tif", "A1027.TIF", "A1028.TIF", "B032.tif", "B034.tiff", "A305.tif")


def main():
    assert os.environ.get("FILMCOLOR_DATA_ROOT") == "/Volumes/waizhiyingpan/filmcolor_data"
    result = {}
    for name in NAMES:
        p = Path("/Volumes/waizhiyingpan/胶片负片素材") / ("02_黑白负片_未反相" if name.startswith("B") else "01_彩色负片_未反相") / name
        digest = check_source(p, allow_frozen=False)
        with tifffile.TiffFile(p) as tf:
            page = _main_page(tf, allow_tiff=True)
            assert page is not None, name
            bits = int(__import__("numpy").atleast_1d(_tag(page, "BitsPerSample"))[0])
            icc = _tag(page, "InterColorProfile")
            if icc is None:
                icc = _tag(tf.pages[0], "InterColorProfile")
            mode, gamma, curve = _parse_tiff_eotf(icc, bits)
            raw = bytes(icc) if icc is not None else b""
            result[name] = dict(raw_sha256=digest, bits=bits, compression=int(page.compression),
                                device_class=raw[12:16].decode("ascii", "replace") if len(raw) >= 16 else None,
                                icc_sha256=hashlib.sha256(raw).hexdigest() if raw else None,
                                mode=mode, gamma=gamma, lut_entries=len(curve) if curve is not None else None)
    (ROOT / "experiments/t4_auto_ui/t4v/icc_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
