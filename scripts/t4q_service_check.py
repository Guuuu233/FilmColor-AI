"""Verify the running local T4q HTTP service; save only selected non-sensitive evidence.
Usage: python -I scripts/t4q_service_check.py
The service must already be running via the FilmColor preview configuration.
"""
import hashlib
import io
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from filmcolor.app import THUMB_REV
from filmcolor.noref import NOREF_V3

URL = "http://127.0.0.1:62872"


def get_json(route):
    with urllib.request.urlopen(URL + route, timeout=180) as response:
        return json.load(response)


def main():
    files = get_json("/api/files")
    roll = get_json("/api/roll")
    sample = "_DSF9239.RAF"
    assert sample in files["files"]
    req = urllib.request.Request(URL + "/api/open", method="POST",
                                 data=json.dumps(dict(path=sample, film="color", preset="standard")).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as response:
        opened = json.load(response)
    auto = opened["auto"]
    assert auto["status"] == "ok", auto
    query = urllib.parse.urlencode(dict(token=opened["token"], **auto["params"]))
    with urllib.request.urlopen(URL + "/api/render?" + query, timeout=180) as response:
        png = response.read()
        assert response.headers.get_content_type() == "image/png"
    with Image.open(io.BytesIO(png)) as image:
        image.verify()
    out = dict(url=URL, sub_rolls=len(roll["sub_rolls"]),
               flagged=sum(bool(f["flag"]) for f in roll["frames"].values()),
               sample=sample, status=auto["status"], ev=auto["params"]["ev"],
               highlight_hold=auto["diagnostics"].get("highlight_hold"),
               render_png_bytes=len(png), render_sha256=hashlib.sha256(png).hexdigest(),
               source_thumb_rev=THUMB_REV, source_config_rev=NOREF_V3["rev"])
    assert THUMB_REV == "t4q" and NOREF_V3["rev"] == "t4q-2026-11-01"
    target = ROOT / "experiments/t4_auto_ui/t4q/service_check.json"
    target.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
