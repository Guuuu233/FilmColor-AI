"""Colour-only check of the fixed look (S3_ML_DESIGN section 2.2): refit global_look_v1's recipe without the
black-and-white frame _DSF9228 and apply the pre-registered rule to choose between global_look_v1 and the
candidate global_look_c1.

Same recipe as noref_global.py (pooled P8 / R8, 3000 pixels per frame, seed 0, mode 7, lam 0, fit_config_v1
highlight constraints). Two fits: the v1 recipe as before (12 NLP frames incl. 9228, to check that the current
code reproduces global_look_v1) and c1 (11 colour NLP frames). Both evaluated only on the 32 colour frames.
Rule: keep v1 if contrast, sat, shoulder and ev of c1 differ from v1 by at most 0.25 of their fit.SCALES and the
median dE2000 changes by at most 0.2 on both domains; otherwise adopt c1. BlueNeg and ext_own are not read.

  global_look_c1.py OUT_JSON
"""
import json
import statistics as st
import sys
from pathlib import Path

from filmcolor.fit import FIT_CONFIG_V1, SCALES, fit_reference

sys.path.insert(0, str(Path(__file__).parent))
from noref_global import evaluate, load, pooled  # noqa: E402

V1 = json.load(open("src/filmcolor/configs/global_look_v1.json"))
BW = "_DSF9228"
RULE_KEYS = ("contrast", "sat", "shoulder", "ev")
MAX_SCALED_DIFF, MAX_DE_CHANGE = 0.25, 0.2


def fit(frames):
    c = FIT_CONFIG_V1
    P, R = pooled(frames)
    return fit_reference(P, R, "7", lam=0.0, n_sample=len(P), highlight_weight=c["highlight_weight"],
                         area_weight=c["area_weight"])["params"]


def main(out):
    nlp, shop = load("nlp"), load("shop")
    colour = {"nlp": {s: v for s, v in nlp.items() if s != BW}, "shop": shop}
    params = {"v1_frozen": V1["params"],
              "v1_refit": fit({**nlp, **shop}),
              "c1": fit({**colour["nlp"], **shop})}
    ev = {k: {d: evaluate(colour[d], p) for d in colour} for k, p in params.items()}
    med = {k: {d: st.median(v.values()) for d, v in e.items()} for k, e in ev.items()}
    for k, p in params.items():
        print(f"{k:10s}", " ".join(f"{n}={p[n]:.4f}" for n in SCALES), "| median dE",
              " ".join(f"{d} {m:.3f}" for d, m in med[k].items()), flush=True)
    reproduce = {n: params["v1_refit"][n] - V1["params"][n] for n in SCALES}
    scaled = {n: (params["c1"][n] - V1["params"][n]) / SCALES[n] for n in RULE_KEYS}
    de_change = {d: med["c1"][d] - med["v1_frozen"][d] for d in colour}
    keep_v1 = all(abs(v) <= MAX_SCALED_DIFF for v in scaled.values()) and \
        all(abs(v) <= MAX_DE_CHANGE for v in de_change.values())
    decision = "global_look_v1" if keep_v1 else "global_look_c1"
    print("v1 refit - frozen v1:", {n: round(v, 6) for n, v in reproduce.items()})
    print("c1 - v1 in scale units:", {n: round(v, 3) for n, v in scaled.items()})
    print("median dE change c1 - v1:", {d: round(v, 3) for d, v in de_change.items()})
    print("decision:", decision, flush=True)
    json.dump(dict(rule=dict(keys=RULE_KEYS, max_scaled_diff=MAX_SCALED_DIFF, max_median_de_change=MAX_DE_CHANGE),
                   train=dict(v1="nlp 12 (incl. _DSF9228) + shop 21", c1="nlp 11 colour + shop 21"),
                   eval_frames={d: sorted(v) for d, v in colour.items()}, params=params, per_frame_de=ev,
                   median_de=med, v1_refit_minus_frozen=reproduce, c1_minus_v1_scaled=scaled,
                   median_de_change=de_change, decision=decision),
              open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
