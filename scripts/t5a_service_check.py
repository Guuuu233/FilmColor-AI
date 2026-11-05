"""Check the already-running FilmColor service (T5a source) using its real POST open API."""
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
    assert opened["input_warnings"] == [], opened["input_warnings"]  # T5a field served, no hint on this frame
    assert auto["config_rev"] == "t4z-2026-11-05", auto.get("config_rev")
    query = urllib.parse.urlencode(dict(token=opened["token"], **auto["params"]))
    with urllib.request.urlopen(URL + "/api/render?" + query, timeout=180) as response:
        png = response.read()
        assert response.headers.get_content_type() == "image/png"
    with Image.open(io.BytesIO(png)) as image:
        image.verify()
    source = hashlib.sha256()
    for p in sorted(p for p in (ROOT / "src/filmcolor").rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".html")):
        source.update(str(p.relative_to(ROOT / "src")).encode() + b"\0" + p.read_bytes() + b"\0")
    out = dict(url=URL, source_sha256=source.hexdigest(), open_method="POST", sub_rolls=len(roll["sub_rolls"]),
               flagged=sum(bool(f["flag"]) for f in roll["frames"].values()),
               sample=sample, status=auto["status"], ev=auto["params"]["ev"],
               http_config_rev=auto["config_rev"], render_png_bytes=len(png),
               render_sha256=hashlib.sha256(png).hexdigest(),
               source_thumb_rev=THUMB_REV, source_config_rev=NOREF_V3["rev"], input_warnings=opened["input_warnings"])
    assert THUMB_REV == "t5a" and NOREF_V3["rev"] == "t4z-2026-11-05"
    (ROOT / "experiments/t4_auto_ui/t5a/service_check.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
