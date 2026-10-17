"""S3 learning baselines (docs/S3_ML_DESIGN.md sections 3-6): image statistics, ridge regression with in-fold
selection, and the roll-level bootstrap. numpy only; not used by the engine.

Every fitted quantity (feature standardisation, ridge coefficients, the regulariser) comes from the training
frames passed in; nothing here sees a test frame.
"""
from __future__ import annotations

import numpy as np

MARGIN = 0.03
EPS = 2.0 ** -14
PCTS = (1, 5, 25, 50, 75, 95, 99)
NEUTRAL_PCT = 20
ALPHAS = np.logspace(-3, 3, 13)
GLOB_NAMES = ([f"l_p{p}" for p in PCTS] + ["cr_mean", "cb_mean", "cr_median", "cb_median", "cr_neutral", "cb_neutral"]
              + ["logR_p99.9", "logG_p99.9", "logB_p99.9", "s_median"])
GRID_NAMES = [f"{q}_{i}{j}" for i in range(3) for j in range(3) for q in ("l", "cr", "cb")]


def _pixels(X8):
    h, w = X8.shape[:2]
    my, mx = int(h * MARGIN), int(w * MARGIN)
    a = np.maximum(np.asarray(X8, np.float64)[my : h - my, mx : w - mx], EPS)
    y = a @ np.array([0.2126, 0.7152, 0.0722])
    lum = np.log2(np.maximum(y, EPS))
    cr, cb = np.log2(a[..., 0] / a[..., 1]), np.log2(a[..., 2] / a[..., 1])
    mx_, mn_ = a.max(-1), a.min(-1)
    s = (mx_ - mn_) / np.maximum(mx_, EPS)
    return a, lum, cr, cb, s


def features_glob(X8) -> np.ndarray:
    """F-glob, 17 values (GLOB_NAMES) from the full-frame positive X8 (margin-cropped)."""
    a, lum, cr, cb, s = _pixels(X8)
    neutral = s <= np.percentile(s, NEUTRAL_PCT)
    return np.array([*np.percentile(lum, PCTS), cr.mean(), cb.mean(), np.median(cr), np.median(cb),
                     np.median(cr[neutral]), np.median(cb[neutral]),
                     *np.log2(np.percentile(a.reshape(-1, 3), 99.9, axis=0)), np.median(s)])


def features_grid(X8) -> np.ndarray:
    """F-grid, 27 values (GRID_NAMES): mean log luminance and log chroma ratios in a 3 x 3 grid."""
    _, lum, cr, cb, _ = _pixels(X8)
    h, w = lum.shape
    ys, xs = np.linspace(0, h, 4).astype(int), np.linspace(0, w, 4).astype(int)
    out = []
    for i in range(3):
        for j in range(3):
            sl = (slice(ys[i], ys[i + 1]), slice(xs[j], xs[j + 1]))
            out += [lum[sl].mean(), cr[sl].mean(), cb[sl].mean()]
    return np.array(out)


class Ridge:
    """Ridge regression on standardised features (statistics from the training frames), unpenalised intercept,
    multi-output, closed form."""

    def __init__(self, alpha: float):
        self.alpha = float(alpha)

    def fit(self, X, Y):
        X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
        self.mu, sd = X.mean(0), X.std(0)
        self.sd = np.where(sd > 0, sd, 1.0)
        Z = (X - self.mu) / self.sd
        self.y0 = Y.mean(0)
        self.coef = np.linalg.solve(Z.T @ Z + self.alpha * np.eye(Z.shape[1]), Z.T @ (Y - self.y0))
        return self

    def predict(self, X):
        return ((np.asarray(X, np.float64) - self.mu) / self.sd) @ self.coef + self.y0


def select_alpha(X, Y, groups, alphas=ALPHAS, validate=None) -> tuple[float, dict]:
    """Regulariser by leave-one-group-out inside the given (training) frames: mean squared error summed over the
    outputs on the held-out frames; ties go to the larger value. ``validate`` limits which groups are held out
    (the others always stay in training)."""
    X, Y, groups = np.asarray(X, np.float64), np.asarray(Y, np.float64), np.asarray(groups)
    held = [g for g in dict.fromkeys(groups) if validate is None or g in validate]
    if len(held) < 2:
        raise ValueError("need at least two groups to hold out")
    err = {}
    for a in alphas:
        se, n = 0.0, 0
        for g in held:
            tr, te = groups != g, groups == g
            r = Ridge(a).fit(X[tr], Y[tr])
            se += float(((r.predict(X[te]) - Y[te]) ** 2).sum())
            n += int(te.sum())
        err[float(a)] = se / n
    best = min(err.values())
    return max(a for a, e in err.items() if e <= best * (1 + 1e-12)), err


def roll_bootstrap(diff, rolls, n: int = 10000, seed: int = 0, level: float = 0.95) -> dict:
    """Per-frame paired differences ``diff`` (method minus baseline) -> one value per roll (mean over its frames),
    then rolls resampled with replacement; statistic = mean of the roll values; percentile interval."""
    diff, rolls = np.asarray(diff, np.float64), np.asarray(rolls)
    names = list(dict.fromkeys(rolls))
    D = np.array([diff[rolls == r].mean() for r in names])
    rng = np.random.default_rng(seed)
    boot = D[rng.integers(0, len(D), (n, len(D)))].mean(1)
    lo, hi = np.percentile(boot, [100 * (1 - level) / 2, 100 * (1 + level) / 2])
    return dict(mean=float(D.mean()), ci=[float(lo), float(hi)], level=level, n_rolls=len(D),
                improved_rolls=float(np.mean(D < 0)), roll_values={r: float(v) for r, v in zip(names, D)})
