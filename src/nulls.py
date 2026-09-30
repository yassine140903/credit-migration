"""Null reference models for the horizon harness.

Both produce the same thing the baseline does - a {horizon: (n_loans, 6)}
array of predicted distributions - so src.scoring scores them unchanged.

MODEL A  PERSISTENCE
    Every loan keeps its state at the cut with probability 1.0 at every horizon.
    Implemented as the IDENTITY transition matrix pushed through
    src.horizon.chain / homogeneous_step - the same path as the baseline - not
    as a special case.

MODEL B  MARGINAL
    One distribution for every loan, whatever its state at the cut, and the
    same at every horizon; it does not chain. It is the empirical distribution
    of next_state across ALL fit rows of the panel (role == "fit"), read
    directly from the panel.

MODEL C  ORACLE MARGINAL - ORACLE (not achievable)
    A diagnostic yardstick, NOT a model: it uses test-period outcomes and could
    not be built at forecast time. For each horizon AND each cohort separately,
    the realised state distribution across that cohort, assigned identically
    to every loan in it. It does not chain and ignores the state at the cut.
    Unlike the marginal null (one fit-window distribution for all horizons and
    cohorts), it differs per horizon and per cohort.
"""

import numpy as np

from src import horizon as hz
from src.panel_io import load_panel

FIT_ROLE = "fit"
TOL = 1e-12


def persistence_matrix(n=hz.N_STATES):
    return np.eye(n, dtype=np.float64)


def persistence_predictions(start_states, horizons=hz.HORIZONS):
    """The identity matrix, chained exactly as the baseline is."""
    return hz.chain(start_states, hz.homogeneous_step(persistence_matrix()),
                    horizons)


def fit_next_state_counts():
    """Counts of next_state over every fit row, in the harness state order."""
    df = load_panel(roles=[FIT_ROLE], columns=["next_state"])
    vc = df["next_state"].value_counts(dropna=False)
    unknown = [s for s in vc.index if s not in hz.STATES]
    if unknown:
        raise hz.HorizonError("next_state carries unknown or null value(s) %s"
                              % unknown)
    return np.array([int(vc.get(s, 0)) for s in hz.STATES], dtype=np.int64), \
        len(df)


def marginal_distribution(counts):
    counts = np.asarray(counts, dtype=np.int64)
    if counts.sum() == 0:
        raise hz.HorizonError("no fit rows to estimate the marginal from")
    dist = counts / counts.sum()
    if abs(dist.sum() - 1.0) > TOL:
        raise hz.HorizonError("marginal sums to %.17g, not 1" % dist.sum())
    return dist


def marginal_predictions(dist, n_loans, horizons=hz.HORIZONS):
    """The same distribution for every loan and every horizon."""
    dist = np.asarray(dist, dtype=np.float64)
    return {h: np.tile(dist, (n_loans, 1)) for h in horizons}


# ==========================================================================
# ORACLE (not achievable) - a floor, not a model
# ==========================================================================
ORACLE = "oracle"
ORACLE_LABEL = "ORACLE (not achievable)"
CENSORED = hz.CENSORED


def oracle_distribution(realised, member, n_states=hz.N_STATES):
    """(distribution, counts) of realised states across one cohort."""
    r = np.asarray(realised, dtype=np.int64)[np.asarray(member, dtype=bool)]
    if len(r) == 0:
        raise hz.HorizonError("%s: empty cohort" % ORACLE_LABEL)
    if np.any(r == CENSORED):
        raise hz.HorizonError("%s: censored loans in the cohort" % ORACLE_LABEL)
    counts = np.bincount(r, minlength=n_states)
    dist = counts / counts.sum()
    if abs(dist.sum() - 1.0) > TOL:
        raise hz.HorizonError("%s distribution sums to %.17g, not 1"
                              % (ORACLE_LABEL, dist.sum()))
    return dist, counts


def oracle_predictions(real, masks):
    """{(h, cohort): (n_loans, 6)} - one realised distribution per cohort,
    tiled over every loan (the scorer reads only the cohort's rows)."""
    out, dists = {}, {}
    for (h, c), mask in masks.items():
        dist, counts = oracle_distribution(real[h], mask)
        dists[h, c] = (dist, counts)
        out[h, c] = np.tile(dist, (len(mask), 1))
    return out, dists
