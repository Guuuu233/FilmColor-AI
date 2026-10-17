"""Fetch the frozen DINOv2-Small weights for M3 (docs/S3_ML_DESIGN.md 4.1) into
$FILMCOLOR_DATA_ROOT/models/dinov2-small/<revision>/.

Only model.safetensors (sha256 checked against the value registered in the design), config.json and
preprocessor_config.json, at the pinned revision; no torch.hub, no pickle weights. Each file is written to
<name>.part and renamed only after its hash is known (and, for the weights, matches). Prints the hashes.

  s3_fetch_dino.py
"""
import hashlib
import json
import subprocess
import sys

from filmcolor.paths import data_dir

REPO, REVISION = "facebook/dinov2-small", "ed25f3a31f01632728cabb09d1542f84ab7b0056"
EXPECTED = {"model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"}
FILES = ("config.json", "preprocessor_config.json", "model.safetensors")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def main():
    d = data_dir("models", "dinov2-small", REVISION)
    hashes = {}
    for name in FILES:
        dst, part = d / name, d / (name + ".part")
        if not dst.exists():
            url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
            subprocess.run(["curl", "-fsSL", "--retry", "3", "-o", str(part), url], check=True)
            h = sha256(part)
            if name in EXPECTED and h != EXPECTED[name]:
                part.unlink()
                sys.exit(f"{name}: sha256 {h} != registered {EXPECTED[name]}")
            part.rename(dst)
        hashes[name] = sha256(dst)
        if name in EXPECTED and hashes[name] != EXPECTED[name]:
            sys.exit(f"{name}: sha256 {hashes[name]} != registered {EXPECTED[name]}")
    (d / "FETCH.json").write_text(json.dumps(dict(repo=REPO, revision=REVISION, sha256=hashes), indent=1))
    print(d, json.dumps(hashes, indent=1))


if __name__ == "__main__":
    main()
