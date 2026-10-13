"""Run automatic film base estimation on a source; write JSON results and an overlay contact sheet."""
import json, sys
from pathlib import Path
import numpy as np, tifffile
from filmcolor.raw import linearize, saturation_mask, superpixel
from filmcolor.rawpy_io import load_raw
from filmcolor.base import estimate_base_auto

src = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
tiles, results = [], {}
for p in sorted(src.glob("*.RAF")):
    f = load_raw(p); lin, col, sat = linearize(f), f.colors, saturation_mask(f)
    n = f.pattern.shape[0]
    est = estimate_base_auto(lin, col, sat, n, crop=f.crop)
    results[p.name] = est.to_dict() | {"regions": f"{len(est.regions)} blocks (bbox {est.diagnostics['kept_bbox']})"}
    print(f"{p.name}: base RGB {tuple(round(x,4) for x in est.rgb)} kept {est.diagnostics['n_kept']} of {est.diagnostics['n_candidates']}"
          f" chroma {[round(x,3) for x in est.diagnostics['chroma_centre_log2_RG_BG']]} warn {est.warnings}")
    b = est.params["block"]
    sp, ss = superpixel(lin, col, sat, n, b)
    g = np.log2(np.clip(sp.mean(-1), 1e-4, None)); g = (g - g.min()) / (g.max() - g.min())
    vis = np.repeat(g[..., None] * 0.8, 3, -1)
    vis[ss] = (0, 0, 1)
    for t, l, _, _ in est.regions:
        vis[t // b, l // b] = (0, 1, 0)
    step = max(1, max(vis.shape[:2]) // 300)
    tiles.append((vis[::step, ::step] * 255).astype(np.uint8))
(out / "base_auto.json").write_text(json.dumps(results, indent=1))
th = max(t.shape[0] for t in tiles); tw = max(t.shape[1] for t in tiles); cols = 4
sheet = np.full((-(-len(tiles)//cols) * (th + 6), cols * (tw + 6), 3), 255, np.uint8)
for i, t in enumerate(tiles):
    y, x = (i // cols) * (th + 6), (i % cols) * (tw + 6); sheet[y:y+t.shape[0], x:x+t.shape[1]] = t
tifffile.imwrite(out / "base_auto_overlay.tif", sheet)
