"""Diagnostic tool: compare pure-Python TIFF LZW + Predictor2 decoding with system libtiff on A1027.TIF strips."""
from __future__ import annotations

import ctypes
import hashlib
import json
import sys
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import tifffile
from filmcolor.frozen import check_source
from filmcolor.scanner import _decode_tiff_lzw


def find_libtiff_path() -> str:
    """Locate an existing libtiff dylib on the host (for validation only, not a runtime dependency)."""
    candidates = [
        REPO_ROOT / ".venv/lib/python3.13/site-packages/PIL/.dylibs/libtiff.6.dylib",
        REPO_ROOT / ".venv/lib/python3.12/site-packages/PIL/.dylibs/libtiff.6.dylib",
        Path("/opt/homebrew/lib/libtiff.dylib"),
        Path("/usr/local/lib/libtiff.dylib"),
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    raise FileNotFoundError("libtiff library not found for diagnostic comparison")


def main():
    target_path = Path("/Volumes/waizhiyingpan/胶片负片素材/01_彩色负片_未反相/A1027.TIF")
    if not target_path.is_file():
        print(f"Target file {target_path} not found on host, skipping diagnostic.")
        return

    # Security guard: verify non-frozen
    input_sha = check_source(target_path, allow_frozen=False)

    libtiff_dylib = find_libtiff_path()
    libtiff = ctypes.CDLL(libtiff_dylib)
    libtiff.TIFFOpen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    libtiff.TIFFOpen.restype = ctypes.c_void_p
    libtiff.TIFFReadEncodedStrip.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_ssize_t]
    libtiff.TIFFReadEncodedStrip.restype = ctypes.c_ssize_t
    libtiff.TIFFClose.argtypes = [ctypes.c_void_p]

    tif_handle = libtiff.TIFFOpen(str(target_path).encode("utf-8"), b"r")
    if not tif_handle:
        raise RuntimeError("libtiff failed to open TIFF file")

    test_strips = [0, 100, 405, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 7503]
    records = []

    try:
        with tifffile.TiffFile(target_path) as tf:
            p = tf.pages[0]
            fh = tf.filehandle
            offsets = p.dataoffsets
            bytecounts = p.databytecounts
            w = p.imagewidth
            spp = p.samplesperpixel
            expected_bytes = w * spp * 2
            bo = p.parent.byteorder
            dt = np.dtype(f"{bo}u2")

            c_buf = ctypes.create_string_buffer(expected_bytes)

            for s_idx in test_strips:
                fh.seek(offsets[s_idx])
                raw = fh.read(bytecounts[s_idx])
                raw_sha = hashlib.sha256(raw).hexdigest()

                # Pure Python LZW decode + Predictor 2
                dec = _decode_tiff_lzw(raw, expected_len=expected_bytes)
                sarr = np.frombuffer(dec, dtype=dt).astype(p.dtype, copy=False).reshape(1, w, spp)
                sarr = np.cumsum(sarr, axis=1, dtype=np.uint16)
                py_pixels_sha = hashlib.sha256(sarr.tobytes()).hexdigest()

                # libtiff decode (TIFFReadEncodedStrip applies Predictor 2)
                nread = libtiff.TIFFReadEncodedStrip(tif_handle, s_idx, c_buf, expected_bytes)
                assert nread == expected_bytes, f"libtiff read {nread} != {expected_bytes}"
                libtiff_arr = np.frombuffer(bytes(c_buf), dtype=dt).astype(p.dtype, copy=False).reshape(1, w, spp)
                libtiff_pixels_sha = hashlib.sha256(libtiff_arr.tobytes()).hexdigest()

                match = bool(py_pixels_sha == libtiff_pixels_sha and np.array_equal(sarr, libtiff_arr))
                records.append({
                    "strip_index": s_idx,
                    "raw_bytecount": bytecounts[s_idx],
                    "raw_sha256": raw_sha,
                    "expected_bytes": expected_bytes,
                    "py_sha256": py_pixels_sha,
                    "libtiff_sha256": libtiff_pixels_sha,
                    "match": match,
                })
    finally:
        libtiff.TIFFClose(tif_handle)

    all_matched = all(r["match"] for r in records)
    out_dict = {
        "target": str(target_path),
        "input_sha256": input_sha,
        "libtiff_path": libtiff_dylib,
        "strips_tested_count": len(records),
        "all_strips_matched": all_matched,
        "strips": records,
    }

    out_json = REPO_ROOT / "experiments/t4_auto_ui/t4s" / "lzw_strip_check.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out_dict, indent=2))
    print(f"Diagnostic completed: {len(records)} strips tested, all_matched={all_matched}.")
    print(f"Results written to {out_json}")


if __name__ == "__main__":
    main()
