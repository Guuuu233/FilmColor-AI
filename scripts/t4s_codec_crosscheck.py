"""Cross-check native LZW against Pillow's independent libtiff encoder; no original images read."""
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, features

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filmcolor.scanner import _read_page_array


def main():
    assert features.check("libtiff"), "Pillow libtiff encoder unavailable"
    rng = np.random.default_rng(20261102)
    cases = {
        "random_full_dictionary": rng.integers(0, 65536, (512, 1024), dtype=np.uint16),
        "gradient_wrap": (np.arange(512 * 1024, dtype=np.uint32) * 73 % 65536).astype(np.uint16).reshape(512, 1024),
    }
    result = {}
    with tempfile.TemporaryDirectory(prefix="fc_t4s_codec_") as folder:
        for name, original in cases.items():
            p = Path(folder) / (name + ".tif")
            Image.fromarray(original).save(p, compression="tiff_lzw")
            started = time.monotonic()
            with tifffile.TiffFile(p) as tf:
                page = tf.pages[0]
                assert int(page.compression) == 5
                decoded = _read_page_array(page)
                equal = np.array_equal(original, decoded)
                assert equal, name
                result[name] = dict(shape=list(original.shape), pixels_equal=equal,
                                    expected_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
                                    actual_sha256=hashlib.sha256(decoded.tobytes()).hexdigest(),
                                    seconds=time.monotonic()-started, strips=len(page.dataoffsets))
    out = dict(encoder="Pillow libtiff", pillow_libtiff=features.version("libtiff"), cases=result)
    (ROOT / "experiments/t4_auto_ui/t4s/codec_crosscheck.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
