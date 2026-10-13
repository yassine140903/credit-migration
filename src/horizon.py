"""Horizon evaluation: eligibility at the cut, realised outcomes, cohorts, chaining.

Model-agnostic. Nothing here knows how a transition matrix was estimated: the
chainer takes a step function, and everything else is read from the panel.

TIMELINE
    cut         2008-12 (src.panel's CUT_YEAR / CUT_MONTH - the fit window's end)
    horizon h   h months after the cut; target month T_h = cut + h.
                h in {6, 12, 24, 60}: 2009-06, 2009-12, 2010-12, 2013-12.

OBSERVATIONS
    A panel row (loan, t, from_state, to_state) observes the loan in TWO
    months: t (from_state) and t + 1 (to_state). The observation table is
    the union of both, one state per (loan, month); two rows disagreeing about
    a month is an error. Only test-split rows (condition, evaluate) are read.

    Consequence at the boundary: the transition 2008-12 -> 2009-01 is a
    CONDITION row (its from month is <= the cut), yet its to_state is the
    first post-cut observation. An absorption landing in 2009-01 exists ONLY on
    that row, because the build emits nothing from an absorbing state (R1).
    It is therefore read as an outcome; the report counts how many outcomes
    depend on it.

STEP 1  ELIGIBILITY (depends only on the state at the cut)
    universe    every loan with a condition or evaluate row (test split). Test
                loans with no panel row at all are outside the panel and so
                outside the universe.
    absorbed    the loan's first absorbing observation is at or before the cut
    eligible    observed at the cut month in a transient state. That month is
                observed only through condition rows (from_period 2008-12, or the
                to_state of from_period 2008-11).
    unobserved  not absorbed, no observation at the cut, split by reason:
                    enters after the cut        first observation > cut
                    follow-up ends before cut   last observation < cut
                    reporting hole at the cut   observed on both sides only

STEP 2  REALISED STATE AT T_h
    absorbed    first absorbing observation at a month <= T_h: that absorbing
                state, whether or not the loan is observed at T_h
    observed    otherwise, the state observed at T_h
    censored    otherwise (no observation at T_h, not absorbed by T_h)

STEP 3  COHORTS
    varying     eligible and not censored at h
    fixed       eligible and not censored at every horizon
"""

import time

import numpy as np
import pandas as pd

from src import panel
from src.panel_io import load_panel, read_manifest

STATES = list(panel.STATES)
N_STATES = len(STATES)
TRANSIENT = STATES[:4]
ABSORBING = STATES[4:]
I_ABSORBING = np.array([STATES.index(s) for s in ABSORBING])
CUT = panel.CUT                                   # month index of 2008-12
HORIZONS = (6, 12, 24, 60)
TEST_ROLES = ["condition", "evaluate"]
COLUMNS = ["loan_id", "from_period", "from_state", "to_state", "role"]
CENSORED = -1                                     # realised-state code

EXCLUDED = ["absorbed at or before the cut",
            "enters the panel after the cut",
            "follow-up ends before the cut",
            "reporting hole at the cut"]
TOL = 1e-12


class HorizonError(RuntimeError):
    pass


def month_index(yyyymm):
    yyyymm = np.asarray(yyyymm, dtype=np.int64)
    return (yyyymm // 100) * 12 + yyyymm % 100 - 1


def ym(idx):
    idx = int(idx)
    return "%04d-%02d" % (idx // 12, idx % 12 + 1)


def target_months(horizons=HORIZONS, cut=CUT):
    return {h: cut + h for h in horizons}


# ==========================================================================
# reading the test split
# ==========================================================================
def _codes(series):
    cats = list(series.cat.categories)
    unknown = [c for c in cats if c not in STATES]
    if unknown:
        raise HorizonError("panel carries unknown state label(s) %s" % unknown)
    remap = np.array([STATES.index(c) for c in cats] + [-1], dtype=np.int64)
    out = remap[series.cat.codes.to_numpy()]
    if (out < 0).any():
        raise HorizonError("%d null state(s) in the panel" % int((out < 0).sum()))
    return out


def _per_loan(df, horizons, cut, cut_role="condition"):
    """Per-loan summary for one partition's rows. The cut month must be observed through rows of role `cut_role`
    ("condition" for the production cut at 2008-12; "fit" when the cut is inside the fit window)."""
    key, ids = pd.factorize(df["loan_id"].to_numpy())
    t = month_index(df["from_period"].to_numpy())
    o, d = _codes(df["from_state"]), _codes(df["to_state"])
    is_cond = (df["role"] == cut_role).to_numpy()

    # every month a row observes: its from month and the month after it
    obs = pd.DataFrame({"key": np.concatenate([key, key]),
                        "month": np.concatenate([t, t + 1]),
                        "state": np.concatenate([o, d]),
                        "cond": np.concatenate([is_cond, is_cond])})
    if not obs.loc[obs["month"] == cut, "cond"].all():
        raise HorizonError("the cut month is observed through a row whose role is not %r" % cut_role)
    obs = obs.drop_duplicates(["key", "month", "state"])
    clash = obs.duplicated(["key", "month"])
    if clash.any():
        raise HorizonError("%d (loan, month) pairs carry two different states"
                           % int(clash.sum()))

    n = len(ids)
    month, state, k = (obs["month"].to_numpy(), obs["state"].to_numpy(),
                       obs["key"].to_numpy())
    first = np.full(n, np.iinfo(np.int64).max)
    last = np.full(n, -1, dtype=np.int64)
    np.minimum.at(first, k, month)
    np.maximum.at(last, k, month)

    is_ab = np.isin(state, I_ABSORBING)
    absorb_month = np.full(n, np.iinfo(np.int64).max)
    np.minimum.at(absorb_month, k[is_ab], month[is_ab])
    has_ab = absorb_month < np.iinfo(np.int64).max
    absorb_state = np.full(n, -1, dtype=np.int64)
    hit = is_ab & (month == absorb_month[k])
    absorb_state[k[hit]] = state[hit]
    if np.any(has_ab[k] & (month > np.where(has_ab, absorb_month, 0)[k])):
        raise HorizonError("observations after absorption")

    L = pd.DataFrame({"first": first, "last": last,
                      "absorb_month": np.where(has_ab, absorb_month, -1),
                      "absorb_state": absorb_state},
                     index=pd.Index(ids, name="loan_id"))
    for name, m in [("cut", cut)] + [("h%d" % h, cut + h) for h in horizons]:
        col = np.full(n, -1, dtype=np.int64)
        sel = month == m
        col[k[sel]] = state[sel]
        L["state_" + name] = col
    return L


def read_test_loans(horizons=HORIZONS, cut=CUT, progress=print, roles=None, cut_role="condition", period_min=None):
    """One pass over the panel: a per-loan frame for every test-split loan.

    roles       roles read (default TEST_ROLES = condition, evaluate)
    cut_role    the role through which the cut month must be observed (default "condition")
    period_min  YYYYMM; rows with an earlier from_period are dropped before the per-loan summary (default: none dropped).
                Safe for loans that are transient at the cut (an absorbing state is never left, so nothing before cut - 1
                can change their outcome); the eligibility reasons of loans that left earlier are then not meaningful.
    """
    manifest = read_manifest()
    parts = []
    for q in [p["quarter"] for p in manifest["partitions"]]:
        t0 = time.time()
        df = load_panel(roles=TEST_ROLES if roles is None else roles, columns=COLUMNS, quarters=[q])
        if period_min is not None:
            df = df[df["from_period"].to_numpy(dtype=np.int64, na_value=0) >= period_min].reset_index(drop=True)
        parts.append(_per_loan(df, horizons, cut, cut_role))
        progress("    %s  %s test rows  (%.1fs)" % (q, "{:,}".format(len(df)),
                                                   time.time() - t0))
        del df
    L = pd.concat(parts)
    if L.index.duplicated().any():
        raise HorizonError("a loan appears in more than one partition")
    return L, manifest["build_stamp"]


# ==========================================================================
# step 1 - eligibility; step 2 - outcomes; step 3 - cohorts
# ==========================================================================
def eligibility(L, cut=CUT):
    """Series of reason codes: -1 eligible, else an index into EXCLUDED."""
    reason = np.full(len(L), -1, dtype=np.int64)
    absorbed = (L["absorb_month"].to_numpy() >= 0) & \
               (L["absorb_month"].to_numpy() <= cut)
    at_cut = L["state_cut"].to_numpy()
    if np.any(~absorbed & np.isin(at_cut, I_ABSORBING)):
        raise HorizonError("absorbing state at the cut without absorption")
    unobserved = ~absorbed & (at_cut < 0)
    reason[absorbed] = 0
    reason[unobserved & (L["first"].to_numpy() > cut)] = 1
    reason[unobserved & (L["last"].to_numpy() < cut)] = 2
    reason[unobserved & (L["first"].to_numpy() < cut)
           & (L["last"].to_numpy() > cut)] = 3
    left = (reason < 0) & (at_cut < 0)
    if left.any():
        raise HorizonError("%d loans neither eligible nor excluded" % left.sum())
    return pd.Series(reason, index=L.index)


def realised(L, horizons=HORIZONS, cut=CUT):
    """{h: realised state code per loan, CENSORED where unobservable}."""
    out = {}
    am, ast = L["absorb_month"].to_numpy(), L["absorb_state"].to_numpy()
    for h in horizons:
        T = cut + h
        r = L["state_h%d" % h].to_numpy().copy()
        done = (am >= 0) & (am <= T)
        r[done] = ast[done]
        r[~done & (r < 0)] = CENSORED
        if np.any(~done & np.isin(r, I_ABSORBING)):
            raise HorizonError("absorbing state observed at T_%d without "
                               "absorption" % h)
        out[h] = r
    return out


def cohorts(real, horizons=HORIZONS):
    """({h: varying mask}, fixed mask) over the eligible loans."""
    varying = {h: real[h] != CENSORED for h in horizons}
    fixed = np.logical_and.reduce([varying[h] for h in horizons])
    return varying, fixed


# ==========================================================================
# chaining
# ==========================================================================
def one_hot(states, n=N_STATES):
    states = np.asarray(states, dtype=np.int64)
    out = np.zeros((len(states), n), dtype=np.float64)
    out[np.arange(len(states)), states] = 1.0
    return out


def chain(start_states, step, horizons=HORIZONS):
    """Push per-loan distributions forward one month at a time.

    start_states  int state code per loan at the cut
    step          step(dist, k) -> dist one month later; dist is (n_loans, 6)
                  and k = 1..max(h) is the month being reached
    Returns {h: (n_loans, 6)} distributions at each horizon, each row summing
    to 1 within TOL (asserted).
    """
    dist = one_hot(start_states)
    out = {}
    for k in range(1, max(horizons) + 1):
        dist = step(dist, k)
        if k in horizons:
            sums = dist.sum(axis=1)
            if not np.all(np.abs(sums - 1.0) <= TOL):
                raise HorizonError("h=%d: %d predicted rows do not sum to 1 "
                                   "(worst |sum-1| %.3e)"
                                   % (k, int((np.abs(sums - 1) > TOL).sum()),
                                      float(np.abs(sums - 1).max())))
            out[k] = dist.copy()
    return out


def homogeneous_step(P):
    """The step function of a time-homogeneous chain with matrix P."""
    P = np.asarray(P, dtype=np.float64)

    def step(dist, k):
        return dist @ P
    return step


def assert_same_by_start(pred, start_states):
    """Every loan with the same start state has an identical distribution."""
    for h, arr in pred.items():
        for s in np.unique(start_states):
            block = arr[start_states == s]
            if not np.array_equal(block, np.broadcast_to(block[0], block.shape)):
                raise HorizonError("h=%d, start %s: predictions differ across "
                                   "loans" % (h, STATES[s]))
    return True
