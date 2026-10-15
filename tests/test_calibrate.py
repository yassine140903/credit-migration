"""src.calibrate: the primitives and the apply function, on synthetic data (no panel, no model)."""

import numpy as np

from src import calibrate as C


def _corr(a=1.1, b=-0.2):
    corr = {"origin_states": C.TRANSIENT, "origins": {}, "_iso": {}}
    for o in C.TRANSIENT:
        corr["_iso"][o] = (np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.4, 1.0]))
        corr["origins"][o] = {"platt": {s: {"a": a, "b": b} for s in C.STATES if s != o}}
    return corr


def test_platt_recovers_parameters():
    rng = np.random.default_rng(0)
    z = rng.normal(-3, 2, 400_000)
    t = rng.random(len(z)) < C.expit(0.7 * z - 0.4)
    r = C.fit_platt(z, t)
    assert r["converged"] and abs(r["a"] - 0.7) < 0.03 and abs(r["b"] + 0.4) < 0.05


def test_apply_rows_sum_to_one_and_mixed_origins_match_single():
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(6) * 0.3, size=20_000)
    origin = rng.integers(0, 4, len(P))
    out = C.apply_correction(P, origin, _corr())
    assert np.abs(out.sum(axis=1) - 1.0).max() < C.SUM_TOL and not np.isnan(out).any()
    one = C.apply_correction(P[origin == 2], "DQ60", _corr())
    assert np.allclose(one, out[origin == 2])


def test_identity_correction_leaves_matrix_unchanged():
    rng = np.random.default_rng(2)
    P = rng.dirichlet(np.ones(6), size=5_000)
    corr = _corr(a=1.0, b=0.0)
    for o in C.TRANSIENT:
        corr["_iso"][o] = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert np.allclose(C.apply_correction(P, "Current", corr), P, atol=1e-9)


def test_fold_assignment_is_by_loan_and_order_free():
    ids = np.array(["F%07d" % i for i in range(50_000)], dtype=object)
    f = C.fold_of(ids)
    assert (C.fold_of(ids[::-1])[::-1] == f).all() and set(np.unique(f)) == set(range(C.N_FOLDS))
