"""F-dino for M3 (docs/S3_ML_DESIGN.md 3, 4.1): frozen DINOv2-Small features (CLS + mean of the patch tokens,
768 values) of every labelled development frame, from the cached full-frame X8 only.

Weights from $FILMCOLOR_DATA_ROOT/models/dinov2-small/<revision>/ (scripts/s3_fetch_dino.py), loaded offline
from model.safetensors (hash checked again here); no fine-tuning, eval mode, CPU, no gradients. Writes
$FILMCOLOR_DATA_ROOT/s3_ml/dino/<domain>/<stem>_dino.npy with the source hashes of the X8 cache.

  s3_dino_features.py FRAMES_JSON LABEL_DIR OUT_JSON
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import Dinov2Model  # noqa: E402

from filmcolor.frozen import load_cache, save_cache  # noqa: E402
from filmcolor.learn import dino_input  # noqa: E402
from filmcolor.paths import data_dir  # noqa: E402
from filmcolor.render import RenderParams, render  # noqa: E402

REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
WEIGHTS_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"


def main(frames_json, label_dir, out_json):
    mdir = data_dir("models", "dinov2-small", REVISION)
    h = hashlib.sha256((mdir / "model.safetensors").read_bytes()).hexdigest()
    if h != WEIGHTS_SHA256:
        sys.exit(f"model.safetensors sha256 {h} != registered {WEIGHTS_SHA256}")
    pre = json.loads((mdir / "preprocessor_config.json").read_text())
    torch.manual_seed(0)
    model = Dinov2Model.from_pretrained(str(mdir), use_safetensors=True, local_files_only=True).eval()
    defaults = RenderParams()
    rows = []
    for f in json.load(open(frames_json))["frames"]:
        if not (Path(label_dir) / f["domain"] / f"{f['stem']}.json").exists():
            continue  # pipeline failure: no label, no features
        cache = data_dir("s3_ml", "cache", f["domain"])
        X8 = load_cache(cache / f"{f['stem']}_X8.npy")
        x = dino_input(X8, pre["image_mean"], pre["image_std"], lambda a: render(a, defaults)[0])
        with torch.no_grad():
            hs = model(pixel_values=torch.from_numpy(x[None])).last_hidden_state[0].numpy()
        feat = np.r_[hs[0], hs[1:].mean(0)].astype(np.float32)
        sources = json.loads((cache / f"{f['stem']}_X8.npy.sources.json").read_text())["sources"]
        save_cache(data_dir("s3_ml", "dino", f["domain"]) / f"{f['stem']}_dino.npy", feat, sources)
        rows.append(dict(id=f["id"], norm=float(np.linalg.norm(feat))))
        print(f["id"], feat.shape, "%.2f" % rows[-1]["norm"], flush=True)
    Path(out_json).write_text(json.dumps(dict(
        model=dict(repo="facebook/dinov2-small", revision=REVISION, weights_sha256=h,
                   config_sha256=hashlib.sha256((mdir / "config.json").read_bytes()).hexdigest(),
                   preprocessor_sha256=hashlib.sha256((mdir / "preprocessor_config.json").read_bytes()).hexdigest()),
        software=dict(torch=torch.__version__, transformers=transformers.__version__),
        input="X8 area-averaged to long side 224, rendered with engine defaults (sRGB), short side padded with grey "
              "0.5 to 224 x 224, normalised with the preprocessor mean / std; CPU, eval, no gradients",
        feature="last_hidden_state: CLS (384) + mean of the patch tokens (384)", frames=rows), indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
