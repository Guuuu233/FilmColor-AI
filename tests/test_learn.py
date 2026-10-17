import numpy as np
import pytest

from filmcolor.learn import (ALPHAS, GLOB_NAMES, GRID_NAMES, Ridge, features_glob, features_grid, roll_bootstrap,
                             select_alpha)


def frame(seed=0, h=120, w=180):
    rng = np.random.default_rng(seed)
    return (rng.uniform(0.02, 0.8, (h, w, 3)) * np.array([1.0, 0.9, 0.7])).astype(np.float32)


def test_feature_definitions_follow_exposure_and_colour():
    X = frame()
    g, g2 = features_glob(X), features_glob(X * 2)
    assert len(g) == len(GLOB_NAMES) == 17 and len(features_grid(X)) == len(GRID_NAMES) == 27
    lum = [i for i, n in enumerate(GLOB_NAMES) if n.startswith(("l_p", "log"))]
    col = [i for i, n in enumerate(GLOB_NAMES) if n.startswith(("cr", "cb", "s_"))]
    np.testing.assert_allclose(g2[lum] - g[lum], 1.0, atol=1e-9)  # twice the light = one stop
    np.testing.assert_allclose(g2[col], g[col], atol=1e-9)        # colour ratios unchanged
    Y = X.copy()
    Y[:, : X.shape[1] // 3] *= 4  # left third two stops brighter
    d = features_grid(Y) - features_grid(X)
    for i in range(3):
        for j in range(3):
            assert d[GRID_NAMES.index(f"l_{i}{j}")] == pytest.approx(2.0 if j == 0 else 0.0, abs=0.05)


def test_ridge_matches_least_squares_without_penalty_and_keeps_training_statistics():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 5)) * [1, 10, 0.1, 3, 1]
    Y = X @ rng.normal(size=(5, 2)) + [0.5, -2]
    r = Ridge(1e-9).fit(X, Y)
    A = np.c_[X, np.ones(len(X))]
    np.testing.assert_allclose(r.predict(X), A @ np.linalg.lstsq(A, Y, rcond=None)[0], atol=1e-6)
    mu = r.mu.copy()
    r.predict(X * 100)  # prediction does not refit the standardisation
    np.testing.assert_array_equal(r.mu, mu)
    assert np.allclose(Ridge(1e6).fit(X, Y).predict(X), Y.mean(0), atol=1e-2)  # heavy penalty -> intercept


def test_alpha_selection_is_leave_one_group_out_inside_training_data():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 6))
    groups = np.repeat(list("abcdefgh"), 10)
    lin = X @ rng.normal(size=(6, 1))
    a_lin, _ = select_alpha(X, lin, groups)
    a_noise, _ = select_alpha(X, rng.normal(size=(80, 1)), groups)
    assert a_lin == ALPHAS.min() and a_noise > 1.0
    _, err = select_alpha(X, lin, groups, validate={"a", "b"})  # only these groups are held out
    assert set(err) == set(map(float, ALPHAS))
    with pytest.raises(ValueError):
        select_alpha(X, lin, groups, validate={"a"})


def test_roll_bootstrap_aggregates_frames_per_roll_first():
    r = roll_bootstrap([-1.0, -1.0, 1.0], ["a", "a", "b"], n=2000)
    assert r["mean"] == 0.0 and r["n_rolls"] == 2 and r["improved_rolls"] == 0.5
    assert r["roll_values"] == {"a": -1.0, "b": 1.0}
    r2 = roll_bootstrap([-1.0] * 10, list("abcdefghij"))
    assert r2["ci"] == [-1.0, -1.0] and r2["improved_rolls"] == 1.0
